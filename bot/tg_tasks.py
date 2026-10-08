"""Running one queued task to completion.

Split out of tg_bot.py. Owns _execute_task and _run_task_inner -- the stage
plumbing, per-chat context scoping, cancellation checks, depth/ETA reporting
and the delivery of whatever the graph produced (text, image, video, voice,
report file) -- plus the voice transcription helper and the forwarded-voice
shortcut.

Module-level helpers and constants are read as tg_bot.<name> rather than
imported by value: _MEMORY_DIR is rebound by redirect_data_dir(), and the rest
must keep a single definition so patching tg_bot.X steers this module too.
"""
from __future__ import annotations

import copy as _copy_mod
import html as _html_mod
import os
import subprocess
import tempfile
import threading
import turn_trace
import time
from typing import Optional

import stages as _stages
import graph as _graph
import re as _re


_WORD_RE = _re.compile(r"[^\W\d_]+", _re.UNICODE)
LANG_PIN_TTL = 6 * 3600     # «answer in English from now on» lasts until 6 h of quiet


def _lang_pin_after(sess, mode: str, named_en: bool, script: str, now: float = None) -> None:
    """The «from now on in English» pin after one typed message: set when English
    is asked for by name, dropped on «по-русски», and over after hours of quiet
    once the user writes in another alphabet -- a pin from yesterday answered a
    Russian chat in English for good (10-03)."""
    now = time.time() if now is None else now
    pin = getattr(sess, "lang_pin", "")
    if named_en:
        sess.lang_pin = "en"
    elif mode == "ru":
        sess.lang_pin = ""
    elif pin and script and script != pin and now - getattr(sess, "lang_pin_ts", 0) > LANG_PIN_TTL:
        sess.lang_pin = ""
    if sess.lang_pin:
        sess.lang_pin_ts = now


def _is_button_payload(text: str) -> bool:
    """True for text a button generated rather than the user typed: a
    bracket-tagged internal command, or one of tg_bot's own English
    _DIRECT_KB/_PROMPT_KB/_CB_CMDS phrases (e.g. "describe and analyze this
    image in detail", "create a presentation about: <topic>"). None of these
    carry a real language signal -- treating them as one flipped a Russian
    session's turn_lang to English on a bare 📷 Analyze Photo press (live
    2026-09-18). change_clothes/regenerate/outpaint (_CB_CMDS) are plain prose
    with no bracket tag, the same gap that let this leak through again for a
    change_clothes press in a Russian chat, in English, without asking what
    outfit (live, 2026-09-19)."""
    t = (text or "")
    if t.startswith("["):
        return True
    if t in tg_bot._DIRECT_KB.values():
        return True
    if t in tg_bot._CB_CMDS.values():
        return True
    return any(t.startswith(p) for p in tg_bot._PROMPT_KB.values())


def _message_script(text: str) -> str:
    """'en' for a message written wholly in Latin letters, 'ru' for one wholly
    in Cyrillic, '' when it is neither (mixed text, a button payload, an emoji,
    a bare link, a one-word command). At least three words are needed: 'ok' or
    'upscale' says nothing about what language the user wants back."""
    t = (text or "").strip()
    if not t or t.startswith("[") or _re.search(r"https?://", t):
        return ""
    # "и то же самое по-английски" is Cyrillic and wants an English reply --
    # the alphabet rule alone pinned it to Russian (live 2026-09-13, mega
    # journey re-run). An explicit ask for English wins; it is a one-turn
    # request, so the caller's turn_lang is set for this turn only as well.
    from graph_language import asks_for_english as _ask_en
    if _ask_en(t):
        return "en"
    words = _WORD_RE.findall(t)
    if len(words) < 3:
        return ""
    # «Qual è la capitale…»: accented Latin is still Latin (was '' -> Russian reply)
    latin = sum(1 for w in words if all(ch < "\u0250" for ch in w))
    cyr = sum(1 for w in words if all("\u0400" <= ch <= "\u04ff" for ch in w))
    if latin == len(words):
        return "en"
    if cyr == len(words):
        return "ru"
    return ""


def _plainly_not_a_doc_question(text: str) -> bool:
    """True when the message is not for document retrieval -- and for a
    message that carries its own material: a pasted press release under
    'сократи до 3 пунктов' was wrapped as a QUESTION to the rental contract,
    and the follow-up 'выдели 5 ключевых слов' then found no text in the chat
    (live 2026-09-13, mega journey).

    Which messages the documents could answer is the model's read
    (agent/intent.py doc_question); a word list sent «нарисуй витрину
    пекарни» and «какая погода в Казани» into the RAG prompt (live
    2026-09-13) and was patched word by word since."""
    t = (text or "").strip()
    # Files in the working folder carry their own material and are read whole:
    # three logs were answered from old indexed fragments, none opened (owner
    # 10-03: «он прочитал только 1 из 3»)
    if "[The file '" in t or "[The last thing the user sent was the file" in t:
        return True
    from prompt_guard import has_quote
    if has_quote(t) or "\n\n" in t or len(t.split()) > 60:
        return True
    # machine payloads and bare links/reactions carry no question
    if t.startswith("[") or "://" in t or not any(ch.isalnum() for ch in t):
        return True
    import intent
    return not intent.read(None, t, "", "the user's own documents are searchable")["doc_question"]


def _about_the_last_video(last_image_path: str, user_input: str) -> bool:
    """True when `user_input` is plainly a follow-up about the video sheet
    still on screen -- mirrors graph.needs_relook's own video-sheet branch,
    checked here (before ctx.last_image_path is populated) so a question
    about the video does not fall through to document retrieval.

    Live 2026-09-18 20:14 (chat 100000001): "Что ответить?" right after a
    forwarded video's summary has no image_id/target_image (a video is not a
    photo), so the RAG gate below let it through -- it got wrapped around 25
    passages of an UNRELATED indexed .md file, and the resulting bloated,
    irrelevant context sent the model into a non-terminating generation
    (task still "running" 9+ minutes later)."""
    if not last_image_path or not _graph.is_video_sheet(last_image_path):
        return False
    import intent
    r = intent.read(None, user_input or "", "", "a video")
    return bool(r["is_question"] or r["about_picture"])


# Stage label -> icon. Module level because both the queued-task path and the
# character render read it through _make_stage_callback.
_STAGE_ICON = {
        "search":   "🔎", "generat": "🎨", "draw": "🎨",
        "edit":     "✏️", "inpaint": "✏️", "insert": "➕",
        "transcri": "🎙", "analyz":  "👁",  "research": "🔬",
        # graph.py and tools.py say "Looking at the image/picture", which
        # matched no keyword — the vision stages showed the generic gear.
        "looking":  "👁",
        "compact":  "📦", "speak":   "🔊", "upscal": "🔍",
        "translat": "🌐", "remov":   "🗑",  "restor": "🪄",
}


def _turn_messages(new_msgs: list, snapshot_len: int, inputs) -> list:
    """The messages THIS turn added to `new_msgs`.

    Torture run #5 2026-09-14: the graph sanitizes (drops tool turns) and
    trims the history to 8 non-system messages before appending the turn, so
    slicing at the persisted length skipped the turn's own user message
    whenever the old history held tool results -- the archive requests were
    never in the transcript and a later «заново» redid the cat edit from
    six turns earlier. The turn starts at its user message: the last user
    entry whose text is this turn's input; the old slice is the fallback.
    """
    wants = {(w or "").strip() for w in (inputs if isinstance(inputs, tuple) else (inputs,))}
    wants.discard("")
    for i in range(len(new_msgs) - 1, -1, -1):
        m = new_msgs[i]
        if (m.get("role") == "user" and isinstance(m.get("content"), str)
                and m["content"].strip() in wants):
            return new_msgs[i:]
    return new_msgs[snapshot_len:]


# Several forwarded pieces are held as «Name [🎤 ГС 1]: …» lines (_fwd_batch).
_CONVO_RULE = ("If the transcript is a conversation between several people (lines "
               "«Name: …» or «Name [label]: …»), retell it AS a conversation, in order: "
               "who said what, who asked whom for what, what was agreed. ")


def retell_system(lang: str, is_video: bool) -> str:
    """System prompt of the one-shot retelling of a forwarded voice/video."""
    _lang_line = {"ru": "Write in Russian.", "en": "Write in English."}.get(lang, "Write in Russian.")
    if is_video:
        return ("You retell forwarded VIDEO messages. You get the TRANSCRIPT of "
                "what is said and a STORYBOARD of what is shown (one line per "
                "timestamp). Retell the video as one whole: what happens on screen "
                "and what the speaker says about it, tying the words to what is "
                "visible, plus anything asked of the viewer. Call it a video, never "
                "a voice message. A video whose transcript is «(без слов)» / «(no speech)» is "
                "retold from the storyboard alone: what is shown, in order. Never "
                "ask for a transcript or say one is missing -- there is nothing more "
                "to send. " + _CONVO_RULE + _lang_line + "\n"
                + _tg_video._structure_rules(lang, timecoded=False))
    return ("You summarize forwarded voice messages from their transcript: the "
            "key points and anything asked of the listener. " + _CONVO_RULE + _lang_line + "\n"
            + _tg_video._structure_rules(lang, timecoded=False))


class TaskRunnerMixin:
    def _do_fwd_voice(self, chat_id: int, what: str, said: str) -> None:
        """Deliver a forwarded voice note as transcript / summary / both.

        One implementation for all three ways the choice can arrive: an inline
        button, a caption sent WITH the note ("перескажи"), or the next thing the
        user types. Asking a question the user has already answered is the thing
        that makes the bot feel like it is not listening.
        """
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        # Validate BEFORE discarding the transcript. Every branch below is keyed
        # on `what`, so an unrecognised value (a keyboard from an older build, a
        # payload typo) fell through all of them — having already wiped the one
        # copy of the transcript. The user pressed a button, got total silence,
        # and the note was gone: the next press answered "I no longer have that
        # voice message". Keep it and ask again instead.
        if what == "board":
            _speech, board = tg_bot._split_board(said)
            if not board:
                self._send_text(chat_id, tg_bot._t("fwd_board_none", lang))
                return
            for chunk in tg_bot._split_html(tg_bot._t("fwd_board_head", lang) + "\n"
                                            + tg_bot._board_html(board)):
                self._send_text(chat_id, chunk, parse_mode="HTML")
            return
        if what == "own":
            sess.fwd_own = said[:20000]
            self._store.put(sess)
            self._send_text(chat_id, tg_bot._t("fwd_own_ask", lang))
            return
        if what not in ("text", "sum", "both"):
            tg_bot.logger.warning("forwarded-voice choice %r is not one of "
                           "text/sum/both — keeping the transcript", what)
            self._send_text(chat_id, tg_bot._t("fwd_voice_ask", lang, mins=""),
                            parse_mode="HTML",
                            keyboard=tg_bot._fwd_voice_kb(lang, sess.fwd_transcript_id,
                                                          board=bool(tg_bot._split_board(said)[1])))
            return
        # The transcript is KEPT, not wiped. It used to be cleared here, which
        # made the three-button keyboard single-use: the keyboard is INLINE, so
        # it stays on screen and clickable forever, but pressing 📋 after 📝 hit
        # the empty slot and answered "that voice note is no longer in my hands
        # -- forward it again" for a note the user could still see. Reproduced
        # live: 📝 Расшифровка delivered the transcript, and the very next press
        # on the same keyboard claimed the voice was gone.
        #
        # Retaining it is safe because the id guard in the fwdv: handler already
        # rejects a button belonging to an EARLIER forward, so a stale keyboard
        # cannot deliver the wrong voice. What does get retired is the
        # type-your-answer shortcut, so a "перескажи" typed much later about
        # something else cannot reach back into a note already dealt with.
        sess.fwd_transcript_done = True
        self._store.put(sess)
        if what in ("text", "both"):
            # The transcript is DATA, not a prompt: it goes straight out, split for
            # Telegram's limit, without a round trip through the model that could
            # paraphrase what someone actually said.
            # Words only: the storyboard has its own button.
            _speech = tg_bot._split_board(said)[0] or tg_bot._t("fwd_no_speech", lang)
            for chunk in tg_bot._split_html(tg_bot._t("fwd_voice_head", lang) + "\n"
                                     + _html_mod.escape(_speech)):
                self._send_text(chat_id, chunk, parse_mode="HTML")
        if what in ("sum", "both"):
            # A retelling is DATA in, text out: it runs as a one-shot model
            # call with no chat history. Through the agent it inherited the
            # chat's last picture (live 23:02: a collage -> vision path ->
            # «контекст нашего диалога»), and a second press of the same
            # button was answered «вы прислали тот же текст, хотите иначе?»
            # -- wording could not talk the model out of noticing the repeat,
            # so the repeat is simply not visible to it.
            turn_trace.spawn(self._retell_transcript, chat_id, said, lang, "👁 " in said,
                             name=f"fwd-retell-{chat_id}")
        elif what == "text":
            self._send_text(chat_id, tg_bot._t("main_menu", lang),
                            keyboard=self._main_menu_kb(sess, lang))
            # No retelling to remember -- the transcript itself is the turn.
            self._commit_media_turn(chat_id, said, "(transcript delivered verbatim)", "👁 " in said)
        self._activity.log(chat_id, "system", f"[fwd voice] {what}")

    def _fwd_batch(self, chat_id: int, items: list, ask: bool = True) -> str:
        """Several forwarded pieces at once -- voices, кружки, texts, from one
        person or several, in any order -- become ONE conversation: each line
        under its author, each note labelled («🎤 ГС 2», «🎥 видео 1») so the
        storyboard can say which clip a frame came from. Held like a single
        forwarded note (same buttons, same typed shortcuts); with `ask` the
        "what do I do with it?" goes out once. Returns the held text ('' when
        nothing could be read)."""
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        ctx = self._get_ctx()
        if ctx is None:
            self._send_text(chat_id, tg_bot._t("not_ready", lang),
                            keyboard=self._main_menu_kb(sess, lang))
            return ""
        _video = any(it.get("media") == "video" for it in items)
        self._send_text(chat_id, tg_bot._t("fwd_video_work" if _video else "fwd_voice_work", lang))
        self._api_post("sendChatAction", {"chat_id": chat_id, "action": "typing"})
        lines, boards, names, sheets = [], [], [], []
        n = {"voice": 0, "video": 0, "photo": 0}
        seen_caps: list = []
        media_list: list = []              # what the user can pick from afterwards
        for it in items:
            who = (it.get("author") or "").strip()
            if who and who not in names:
                names.append(who)
            _cap = (it.get("caption") or "").strip()
            if _cap and _cap not in seen_caps and it.get("type") != "text":
                seen_caps.append(_cap)           # the post's own words ride on its first media
                lines.append(f"{who}: {_cap}" if who else _cap)
            if it.get("type") == "text":
                body = (it.get("text") or "").strip()
                if not it.get("own"):
                    body = tg_bot._unclaim(body)
                if body:
                    lines.append(f"{who}: {body}" if who else body)
                continue
            if it.get("type") in ("photo", "album"):
                import llm as _llm
                for fid in (it.get("file_ids") or [it.get("file_id")]):
                    n["photo"] += 1
                    label = tg_bot._t("fwd_lbl_photo", lang, n=n["photo"])
                    media_list.append({"kind": "photo", "file_id": fid, "label": label})
                    try:
                        data = self._dl_bytes(fid)
                        seen_pic = " ".join((_llm.analyze_image_with_llm(
                            ctx, image_bytes=data, temperature=0.1,
                            user_text="Describe this picture in one or two sentences, in " + ("Russian" if (sess.lang or "ru") == "ru" else "English") + ".") or "").split()) if data else ""
                    except Exception:
                        tg_bot.logger.warning("forwarded picture read failed", exc_info=True)
                        seen_pic = ""
                    if seen_pic:
                        lines.append(f"{who} [{label}]: {seen_pic}" if who else f"[{label}]: {seen_pic}")
                continue
            media = "video" if it.get("media") == "video" else "voice"
            n[media] += 1
            label = tg_bot._t("fwd_lbl_" + media, lang, n=n[media])
            data = self._dl_bytes(it["file_id"]) if media == "video" else None
            if media == "video":
                media_list.append({"kind": "video", "file_id": it["file_id"], "label": label})
            # A note with a known author is that person speaking; diarization
            # only helps an unattributed recording of several people.
            heard = (self._transcribe(ctx, it["file_id"], media, lang_hint=(sess.lang or "ru"),
                                      data=data, speakers=not who) or "").strip()
            if media == "video":
                seen = self._look_video(ctx, it["file_id"], data=data,
                                        lang=(sess.lang or "ru"), transcript=heard)
                if (seen.get("description") or "").strip():
                    boards.append(label + (f" · {who}" if who else "") + "\n"
                                  + seen["description"].strip())
                if seen.get("sheet") and os.path.exists(seen["sheet"]):
                    # The emoji is dropped: the bar is drawn with a plain font.
                    sheets.append((label.split(" ", 1)[-1] + (f" · {who}" if who else ""),
                                   seen["sheet"]))
            body = heard or tg_bot._t("fwd_no_speech", lang)
            lines.append(f"{who} [{label}]: {body}" if who else f"[{label}]: {body}")
        said = "\n".join(lines).strip()
        if boards:
            said += "\n\n" + tg_bot._t("fwd_video_seen", lang, seen="\n\n".join(boards))
        if not said:
            self._send_text(chat_id, tg_bot._t("fwd_voice_none", lang),
                            keyboard=self._main_menu_kb(sess, lang))
            return ""
        said = said[:20000]
        sheet = sheets[0][1] if len(sheets) == 1 else ""
        if len(sheets) > 1:
            # Every clip on the chat's picture, so "look again" sees them all.
            try:
                import video_look
                sheet = str(video_look.combine_sheets(
                    sheets, Path(tg_bot._IMAGE_DIR) / f"video_{uuid.uuid4().hex[:8]}" / "sheet.jpg"))
            except Exception:
                tg_bot.logger.exception("combining the batch's video sheets failed")
                sheet = sheets[-1][1]
        sess.fwd_transcript = said
        sess.fwd_transcript_id = uuid.uuid4().hex[:8]
        sess.remember_fwd(sess.fwd_transcript_id, said)
        sess.fwd_transcript_done = False
        # One video: its frames are the picture in play, as before. Several pieces: nothing is attached at random,
        # the user picks which picture or video to work with (the 🎞 button).
        if sheet and len(media_list) <= 1:
            tg_bot._put_in_play(sess, sheet, "video frames")
        sess.__dict__.setdefault("fwd_media", {})[sess.fwd_transcript_id] = media_list
        while len(sess.fwd_media) > 8:
            sess.fwd_media.pop(next(iter(sess.fwd_media)))
        self._store.put(sess)
        self._activity.log(chat_id, "system",
                           f"[fwd batch] {len(items)} pieces, {len(names)} senders, "
                           f"{len(boards)} storyboards", str(chat_id))
        if ask:
            self._commit_media_turn(chat_id, said, "(read and heard; asked what to do with it)",
                                    bool(boards))
            parts = ", ".join(x for x in (
                tg_bot._t("fwd_lbl_voice", lang, n=n["voice"]) if n["voice"] else "",
                tg_bot._t("fwd_lbl_video", lang, n=n["video"]) if n["video"] else "",
                tg_bot._t("fwd_lbl_photo", lang, n=n["photo"]) if n["photo"] else "",
                tg_bot._t("fwd_lbl_text", lang, n=sum(1 for i in items if i.get("type") == "text"))
                if any(i.get("type") == "text" for i in items) else "") if x)
            who_line = tg_bot._t("fwd_batch_who", lang,
                                 names=_html_mod.escape(", ".join(names))) if names else ""
            self._send_text(chat_id, tg_bot._t("fwd_batch_ask", lang, parts=parts, who=who_line),
                            parse_mode="HTML",
                            keyboard=tg_bot._fwd_voice_kb(lang, sess.fwd_transcript_id,
                                                          board=bool(boards),
                                                          cont=len(media_list) == 1 and media_list[0]["kind"] == "video",
                                                          pick=len(media_list) > 1))
        return said

    def _commit_media_turn(self, chat_id: int, said: str, reply: str, is_video: bool) -> None:
        """Put a forwarded voice/video into the chat history the model sees.

        The retelling is deliberately a one-shot call outside the agent, and
        the transcript goes out as data -- so neither ever reached
        sess.history. Live 15:39: right after «Что требуется: дать экспертное
        мнение…» the user typed «Дай экспертное мнение» and the model,
        with no trace of the video in its history, offered topics from
        conversations days old. The turn is committed the way a deep-research
        report is: the media as the user's message, the retelling (or the
        transcript itself) as the assistant's, so a follow-up lands on it.
        """
        kind = "video message" if is_video else "voice message"
        # Framed as something the bot WATCHED and heard: "не видя самого
        # видео… пришлите видео" (live 16:07) came from a bare label.
        user_msg = ((f"[forwarded {kind} -- you have WATCHED it (its keyframes are the chat's "
                     "current picture) and heard it; the transcript and storyboard follow. Answer "
                     "as someone who saw it; never say you cannot see the video.]"
                     if is_video else
                     f"[forwarded {kind} -- you have heard it; the transcript follows.]")
                    + chr(10) + (said or "")[:6000])
        try:
            with self._history_lock(chat_id):
                sess = self._get_session(chat_id)
                hist = sess.get_history()
                # Committed once on arrival (with the storyboard shown), then
                # each button adds only ITS answer: the transcript is not
                # repeated as a new user turn for every press.
                already = any(m.get("role") == "user" and m.get("content") == user_msg
                              for m in hist[-6:])
                if not already:
                    hist.append({"role": "user", "content": user_msg})
                hist.append({"role": "assistant", "content": (reply or "")[:6000]})
                sess.set_history(hist)
                self._store.put(sess)
        except Exception:
            tg_bot.logger.exception("fwd media history save failed chat=%s", chat_id)

    def _retell_transcript(self, chat_id: int, said: str, lang: str, is_video: bool) -> None:
        """One-shot retelling of a forwarded voice/video transcript (see
        _do_fwd_voice). A video transcript carries the 👁 storyboard: speech
        and picture are retold as one whole, tied together."""
        system = retell_system(lang, is_video)
        self._api_post("sendChatAction", {"chat_id": chat_id, "action": "typing"})
        out = ""
        try:
            import llm as _llm
            out = (_llm.call_llm_simple(self._get_ctx(), system, said, temperature=0.3,
                                        max_tokens=900, prefill="<think></think>") or "").strip()
        except Exception:
            tg_bot.logger.exception("fwd retell failed chat=%s", chat_id)
        import turn_trace
        turn_trace.answer("final", out)
        sess = self._get_session(chat_id)
        if not out:
            self._send_text(chat_id, tg_bot._t("fwd_voice_none", lang),
                            keyboard=self._main_menu_kb(sess, lang)); return
        title = tg_bot._t("retell_title_video" if is_video else "retell_title_voice", lang)
        chunks = tg_bot._split_html(_tg_video.render_retelling(out, title))
        for n, chunk in enumerate(chunks):
            self._send_text(chat_id, chunk, parse_mode="HTML",
                            keyboard=self._main_menu_kb(sess, lang) if n == len(chunks) - 1 else None)
        self._activity.log(chat_id, "assistant", out[:500])
        self._commit_media_turn(chat_id, said, out, is_video)

    def _deliver_salvaged_render(self, ctx, chat_id, sess, task, lang, uname) -> bool:
        """Send a picture a timed-out turn had already finished. True if sent.

        The watchdog fires on the whole turn, not on the render, so a picture
        that was drawn, scored and accepted can still be thrown away by tool
        calls that came after it -- which is exactly what happened live: the
        render finished at 09:06:57 with 9/10 and the user got "try again".

        Reads ctx.last_render_path, never ctx.last_image_path: the latter holds
        the user's own upload just as readily, and posting that back at them
        would be worse than the timeout notice.
        """
        path = (getattr(ctx, "last_render_path", None) or "") if ctx else ""
        status = (getattr(ctx, "last_render_status", "") or "") if ctx else ""
        if not path or status in ("fail", "error"):
            return False
        if not os.path.exists(path):
            self._activity.log(chat_id, "error",
                f"[salvage] render path is gone: {path}", uname)
            return False
        try:
            import image as _img_mod
            if _img_mod.is_intermediate_artifact(path):
                self._activity.log(chat_id, "system",
                    "[salvage] skipped intermediate artifact", uname)
                return False
        except Exception:
            pass
        try:
            self._api_post("sendChatAction",
                           {"chat_id": chat_id, "action": "upload_photo"})
            img_id = tg_bot._log_image(sess, path,
                                       label=(task.user_text or "")[:80], src="bot",
                                       parent=(getattr(ctx, "source_image_id", "") or "") if ctx else "")
            self._last_photo_msg_id = 0
            if not self._send_photo(chat_id, path,
                                    keyboard=tg_bot._image_kb(lang, img_id)):
                return False
            tg_bot._log_image(sess, path,
                              msg_id=getattr(self, "_last_photo_msg_id", 0), src="bot")
            try: self._store.put(sess)
            except Exception: pass
            self._activity.log(chat_id, "system", f"[salvage] sent {path}", uname)
            return True
        except Exception:
            tg_bot.logger.exception("salvaged render delivery failed chat=%s", chat_id)
            return False

    def _execute_task(self, task: tg_bot._Task):  # noqa: C901
        chat_id = task.chat_id
        sess    = self._get_session(chat_id)
        user    = self._user_store.get(chat_id)
        uname   = user.name if user else str(chat_id)
        lang    = self._lang(sess)
        _shared = self._get_ctx()
        graph   = self._get_graph()

        if _shared is None or graph is None:
            self._send_text(chat_id, tg_bot._t("not_ready", lang),
                            keyboard=self._main_menu_kb(sess, lang))
            return

        # THIS task's own view of the context: its own working image, memory,
        # facts, output size and cancel event. Without it, running two tasks at
        # once would let one chat see another chat's picture and let one chat's
        # Stop cancel another chat's request.
        _cancel = threading.Event()
        ctx = tg_bot._scoped_ctx(_shared, cancel_event=_cancel)
        # The reply language follows the USER's session language: an English
        # session used to get every answer in Russian (the house default).
        ctx.reply_lang = lang
        # A page read for this turn (tg_links) is named in the reply's shape.
        try:
            import tg_reply_shape as _shape
            ctx.turn_sources.extend(_shape.links_read(task.user_text))
        except Exception:
            pass
        # ...and a message written wholly in the OTHER alphabet flips it for
        # this turn and for the machine payloads (buttons, emoji, links) that
        # follow, until the user types in their own alphabet again.
        #
        # A button-generated payload ("[style_preset] call redraw_image...",
        # "describe and analyze this image in detail") is machine text, not
        # typed by the user -- its English wording carries no language
        # signal, and it must not consult (or update) turn_lang. Before this
        # a stale/newly-set turn_lang="en" leaked English into the interim
        # note and the final reply for a plain button press in a Russian
        # session ("Got it, redrawing"; 📷 Analyze Photo's own answer).
        is_internal = _is_button_payload(task.user_text)
        _tl = "" if is_internal else _message_script(task.user_text)
        from graph_language import asks_for_english as _ask_en
        if _tl and not _ask_en(task.user_text):
            sess.turn_lang = _tl          # an explicit "по-английски" is one turn, not a mode
        # Live: «can you answer in English from now on?» -- "Of course!",
        # then the next Russian line got a Russian answer. The model's read.
        if not is_internal and (task.user_text or "").strip():
            import intent
            from graph_language import names_english as _names_en
            _mode = intent.read(None, task.user_text)["language_mode"]
            _lang_pin_after(sess, _mode, _mode == "en" and _names_en(task.user_text), _tl)
        if is_internal:
            _turn_lang = lang
        else:
            _turn_lang = ("en" if _ask_en(task.user_text)
                          else getattr(sess, "lang_pin", "") or sess.turn_lang)
        if _turn_lang and _turn_lang != lang:
            ctx.reply_lang = _turn_lang
            self._activity.log(chat_id, "system", f"[lang] reply in {_turn_lang} for this turn", uname)

        # This chat's own memory directory. remember_fact persists immediately
        # via ctx.save_memory(ctx.active_memory_dir), and the scoped copy
        # inherits the DESKTOP app's directory — so a Telegram user's fact was
        # written straight into the owner's profile, and (now that the scoped
        # facts list holds only this chat's facts) it OVERWROTE facts.json with
        # them. Same reasoning as the per-chat document index.
        try:
            _mem_dir = tg_bot._MEMORY_DIR / f"chat_{chat_id}"
            _mem_dir.mkdir(parents=True, exist_ok=True)
            ctx.active_memory_dir = _mem_dir
            # _scoped_ctx hands every task a FRESH, empty pinned_facts list --
            # without this, a fact saved by remember_fact in an earlier task
            # (persisted to facts.json) was never read back, so it vanished
            # the moment that task ended (live, journey 9, 2026-09-18: "как
            # меня зовут" forgot a name given three turns earlier).
            ctx.load_memory(_mem_dir)
        except Exception:
            tg_bot.logger.exception("could not set a per-chat memory dir for %s", chat_id)

        # This chat's own SANDBOX, attached only if this user has been granted
        # one. Same reasoning as the per-chat memory directory above, with a
        # sharper edge: a sandbox on the shared context would be one Telegram
        # user editing another's files, or the desktop owner's. copy.copy in
        # _scoped_ctx means setting it here cannot reach the shared context.
        #
        # Attached by GRANT, not by mode: the tools are dropped from the payload
        # when ctx.sandbox is None, so a user without the grant never even sees
        # that a filesystem exists.
        ctx.sandbox = None
        ctx.sandbox_user = user
        # Every chat has its own Ozon session and shopping list, grant or not.
        ctx.ozon_owner = str(chat_id)
        ctx.reminder_owner = chat_id
        # Reasoning is the user's own switch (Settings -> 🧠), off by default.
        ctx.no_think = not bool(getattr(sess, "think", False))
        try:
            import sandbox_access as _sa
            if _sa.may_use_files(user):
                import code_sandbox as _cs
                ctx.sandbox = _cs.sandbox_for(chat_id)
        except Exception:
            tg_bot.logger.exception("could not attach a sandbox for %s", chat_id)

        # …and a graph BOUND TO IT. build_graph closes over the context it is
        # handed, so the app's prebuilt graph runs every node and every tool
        # against the SHARED context no matter what we scope here. That made the
        # isolation above decorative: remember_fact wrote its fact into the
        # desktop app's context instead of this task's (so 🧠 My facts stayed
        # empty right after the bot said "я запомнил" — found live), and the same
        # hole applies to the working image and the cancel event, across users.
        # Rebuilding costs ~17 ms against a multi-second turn.
        try:
            from graph import build_graph as _build_graph
            graph = _build_graph(ctx)
        except Exception:
            tg_bot.logger.exception("could not build a task-scoped graph — falling back "
                             "to the shared one (facts/images may cross chats)")

        # Register as in-flight: powers ⛔ Stop (only cancels YOUR task), the
        # per-request Cancel button, the watchdog's runaway check, and the
        # after-a-crash notification.
        with self._task_lock:
            self._running_task.setdefault(chat_id, []).append(task)
            self._task_started[task.task_id] = time.monotonic()
            self._task_cancels[task.task_id] = _cancel
            # No longer merely "pushed" — it now has its own _running_task /
            # inflight coverage, so the pending-journal copy would otherwise
            # double-report it (once as "pending", once as "running") to any
            # crash that lands moments later.
            self._pending_journal.pop(task.task_id, None)
        self._write_inflight()
        import chatlog as _chatlog
        tg_bot.logger.info("Task %s START chat=%s text=%r", task.task_id[:8], chat_id,
                           (task.user_text or "")[:120])
        _outcome = "crashed"
        _t0 = time.monotonic()      # the watchdog pops _task_started on cancel
        import turn_trace
        _trace = turn_trace.start(chat=chat_id, task=task.task_id[:8], user=uname,
                                  text=(task.user_text or "")[:1500])
        try:
            with _chatlog.bind(chat_id), tg_transport.replying_to(chat_id, task.reply_to):
                self._run_task_inner(task, sess, user, uname, lang, ctx, graph)
            _outcome = "cancelled" if task.task_id in self._cancelled else "done"
        finally:
            # Every task leaves an END line with its time: a start with no end
            # is exactly where a silent death happened.
            tg_bot.logger.info("Task %s END %s in %.1fs chat=%s", task.task_id[:8], _outcome,
                               time.monotonic() - _t0,
                               chat_id)
            turn_trace.finish(_trace, outcome=_outcome)
            self._requeue_unread_steer(task)
            with self._task_lock:
                lst = self._running_task.get(chat_id)
                if lst is not None:
                    try: lst.remove(task)
                    except ValueError: pass
                    if not lst:
                        self._running_task.pop(chat_id, None)
                self._task_started.pop(task.task_id, None)
                self._task_cancels.pop(task.task_id, None)
                self._cancelled.discard(task.task_id)
            self._write_inflight()

    def _try_steer(self, chat_id: int, task) -> bool:
        """A text that arrives while this chat's task is still running goes
        into that task's inbox instead of the queue (steer.py). True when
        it was taken."""
        import steer as _steer
        with self._task_lock:
            running = [t for t in (self._running_task.get(chat_id) or [])
                       if not getattr(t, "delivered", False)
                       and t.task_id in self._steer_inboxes]
            interruptible = self._chat_interruptible.get(chat_id, False)
            if not running or not _steer.eligible(
                    task.user_text, bool(task.image_path) or bool(task.image_id), interruptible):
                return False
            target = running[-1]
            self._steer_inboxes[target.task_id].put(task.user_text)
        sess = self._get_session(chat_id); lang = self._lang(sess)
        user = self._user_store.get(chat_id); uname = user.name if user else str(chat_id)
        self._send_text(chat_id, _steer.ack(task.user_text, lang))
        self._activity.log(chat_id, "system", "[steer] " + task.user_text[:200], uname)
        tg_bot.logger.info("steer: chat=%s -> task %s: %r", chat_id, target.task_id[:8],
                           task.user_text[:80])
        return True

    def _requeue_unread_steer(self, task) -> None:
        """A note the task never read (it ended first) becomes an ordinary
        request, so nothing the user typed is lost."""
        with self._task_lock:
            inbox = self._steer_inboxes.pop(task.task_id, None)
        unread = inbox.unread() if inbox is not None else []
        for text in unread:
            try:
                self._enqueue_item(task.chat_id, {"type": "text", "text": text})
                tg_bot.logger.info("steer: unread note re-queued for chat=%s: %r",
                                   task.chat_id, text[:80])
            except Exception:
                tg_bot.logger.exception("steer: could not re-queue %r", text[:80])

    def _make_stage_callback(self, chat_id: int, lang: str, uname: str,
                             update_status):
        """The one place a stage becomes a status line.

        Extracted from _run_task_inner so a render that does NOT go through the
        task queue -- the Персонажи flow -- reports progress through the same
        pipeline rather than growing a second, slightly different one. Icon
        choice, translation, the desktop feed, the activity log and the
        interruptible gate all live here, and all of them were previously
        reachable only from inside one closure.
        """
        def on_stage(stage: str) -> None:
            # A blank stage means "no stage", not "a stage with no name". The
            # post-delivery compaction ends with ctx.set_stage(""), which used to
            # reach here after _clear_status had already run — so it opened a
            # BRAND NEW "⚙️ …" bubble carrying a live ⛔ Cancel button, under
            # every single answer, and nothing ever cleared it. That is the stack
            # of dead Cancel buttons users kept seeing.
            if not (stage or "").strip():
                with self._stages_lock:
                    self._active_stages.pop(chat_id, None)
                return
            # The icon is chosen from the ENGLISH label, before translating: the
            # stage vocabulary belongs to the shared agent core (graph/tools/
            # image/llm), which reasons in English, and keying the icons off a
            # translated string would silently drop every icon in Russian.
            icon = next((v for k, v in _STAGE_ICON.items()
                         if k in stage.lower()), "⚙️")
            shown = _stages.translate(stage, lang)
            update_status(f"{icon} <b>{_html_mod.escape(shown)}…</b>")
            # The admin panel, the desktop feed and the log are operator-facing
            # and stay English so they remain greppable and comparable.
            with self._stages_lock:
                self._active_stages[chat_id] = f"{icon} {stage}"
            self._on_stage(chat_id, f"{icon} {stage}", False)
            self._activity.log(chat_id, "stage", stage, uname)
            # These stages are the genuinely slow, backgroundable ones (a GPU
            # render, an ffmpeg pass, or a multi-source web crawl) — a plain
            # LLM reply or a single vision look never reaches here, so gating
            # on this keyword set does not open the second slot for short turns.
            if any(k in stage.lower() for k in tg_bot._INTERRUPTIBLE_STAGE_KEYWORDS):
                self._mark_interruptible(chat_id)
        return on_stage

    def _run_task_inner(self, task: tg_bot._Task, sess, user, uname: str, lang: str,
                        ctx, graph):  # noqa: C901
        chat_id = task.chat_id
        import nice_names                      # files sent by this task are named after it
        nice_names.TITLES[chat_id] = task.user_text or ""
        # What the user says while ANY task runs lands here (steer.py): the
        # agent loop folds it in each round and skips a tool call it would
        # change; songs and research read it at their own checkpoints.
        import steer as _steer
        ctx.steer_inbox = _steer.Inbox()
        with self._task_lock:
            self._steer_inboxes[task.task_id] = ctx.steer_inbox
        # «видео моим голосом»: this chat's cloned voice, for generate_video(use_my_voice).
        ctx.voice_ref = next((p for p in (getattr(sess, "clone_ref", ""), getattr(sess, "assistant_ref", ""))
                              if p and os.path.exists(p)), "")
        # 🎙 samples ride only the animate request they were collected for.
        ctx.anim_voices = []
        # 🗣 own assistant voice: the graph's TTS speaks with it directly (ctx is this task's copy).
        # Always set: the copy inherits the DESKTOP's custom voice (settings / Дурдом tab
        # write it on the shared ctx), which every chat's replies used to speak with.
        _aref = getattr(sess, "assistant_ref", "")
        _own = bool(_aref and os.path.exists(_aref))
        ctx.custom_ref_wav = _aref if _own else None
        ctx.custom_ref_text = getattr(sess, "assistant_ref_text", "") if _own else ""
        # 🎙 samples (and «стандартные») ride the ONE clip request they were given
        # for: an animate preset, or the request rerun after the voice question.
        # "" = not asked yet: generate_video asks before rendering people talking.
        ctx.voice_choice = getattr(sess, "voice_choice", "") or ""
        if ctx.voice_choice or (getattr(sess, "anim_voices", None) and
                                task.user_text.startswith(("[animate]", "animate this photo"))):
            ctx.anim_voices = [p for p in sess.anim_voices if os.path.exists(p)]
            ctx.voice_choice = ctx.voice_choice or "own"
            sess.anim_voices, sess.voice_choice, sess.voice_pending = [], "", ""
            self._store.put(sess)
        elif task.user_text.startswith(("[animate]", "animate this photo")):
            ctx.voice_choice = "default"     # the presets carry «🎙 Добавить свои голоса»
        # ▶️ Continue video: the clip's tail (its motion + sound) rides this ONE request.
        ctx.continue_tail = ctx.continue_src = ""
        ctx.continue_people = []
        _ct = getattr(sess, "continue_tail", "")
        if _ct and os.path.exists(_ct) and task.user_text.startswith("animate this photo"):
            ctx.continue_tail, ctx.continue_src = _ct, getattr(sess, "continue_src", "")
            ctx.voice_choice = "default"     # no «свои голоса» question: the tail brings the voices
            ctx.continue_people = [p for p in (getattr(sess, "continue_people", None) or []) if os.path.exists(p)]
            sess.continue_tail = sess.continue_src = ""
            sess.continue_people = []
            self._store.put(sess)

        # No model, no turn. The desktop app can be started with the LLM
        # deliberately not loaded; the graph still exists, so nothing here would
        # fail loudly -- the request would run, every LM Studio call would come
        # back empty, and the user would get "(no response)" after a long wait.
        # Say what is actually going on instead, and do it before the status
        # message with its cancel button is even opened.
        # An ABSENT attribute means "this context does not report a model" --
        # only an explicitly EMPTY one is the deliberate "started without a
        # model" state. The difference matters: defaulting a missing attribute
        # to "" makes this gate refuse every turn for any caller holding a
        # context that simply never declared it.
        _mn = getattr(ctx, "model_name", None)
        if _mn is not None and not str(_mn).strip():
            tg_bot.logger.info("chat %s: turn refused, no model loaded", chat_id)
            self._send_text(chat_id, tg_bot._t("no_model_loaded", lang))
            return

        img_bytes: Optional[bytes] = None
        if task.image_path and os.path.exists(task.image_path):
            try:
                with open(task.image_path, "rb") as fh:
                    img_bytes = fh.read()
            except Exception as exc:
                tg_bot.logger.warning("img tmp read failed %s: %s", task.image_path, exc)

        status_id: Optional[int] = None

        # The live status message carries its own Cancel button, so a user can
        # abandon THIS request without touching anything else they have queued —
        # and without having to find the Stop key first.
        cancel_kb = {"inline_keyboard": [[
            {"text": tg_bot._t("cancel_btn", lang),
             "callback_data": f"cancel:{task.task_id}"}]]}

        # Set once the status line has been retired for good. After that a late
        # stage may still EDIT nothing — it must never open a NEW status message.
        status_done = [False]

        def _update_status(text: str) -> None:
            nonlocal status_id
            try:
                if status_id:
                    self._edit_text(chat_id, status_id, text, parse_mode="HTML",
                                    keyboard=cancel_kb)
                elif not status_done[0]:
                    status_id = self._send_get_id(chat_id, text, parse_mode="HTML",
                                                  keyboard=cancel_kb)
            except Exception: pass

        def _clear_status() -> None:
            nonlocal status_id
            status_done[0] = True
            if status_id:
                try:
                    # Edit to invisible marker first — if delete fails the message
                    # won't show a stale "🔊 Speaking…" stuck in chat. The EMPTY
                    # keyboard matters as much as the text: leaving it attached
                    # left live "⛔ Отменить запрос" buttons stacked in the chat,
                    # pointing at requests that had already finished.
                    self._edit_text(chat_id, status_id, "✅", parse_mode=None,
                                    keyboard={"inline_keyboard": []})
                except Exception:
                    pass
                try:
                    self._delete(chat_id, status_id)
                except Exception:
                    pass
                status_id = None

        _update_status(f"⚙️ <b>{_stages.translate('Starting', lang)}…</b>")

        # ── Deep research bypass ───────────────────────────────────────────────
        # "do a deep research on:" is a button-driven intent that must call
        # dr.run_deep_research() directly — the agent loop always drops
        # deep_research from the tool list (to prevent auto-pick for ordinary
        # questions), so routing it through the loop produces a shallow search.
        _DR_PREFIX = "do a deep research on:"
        if task.user_text.lower().startswith(_DR_PREFIX):
            topic = task.user_text[len(_DR_PREFIX):].strip()
            if not topic:
                _clear_status()
                self._send_text(chat_id, tg_bot._t("dr_no_topic", lang),
                                keyboard=self._main_menu_kb(sess, lang))
                return
            depth = tg_bot._resolve_depth(sess)
            self._on_stage(chat_id, "🔬 Deep Research", False)
            self._activity.log(chat_id, "stage", f"Deep Research ({depth})", uname)
            # Quote the wait BEFORE the wait. A run this long with no number
            # attached is indistinguishable from a hung bot. The moment the user
            # commits to the wait is exactly the moment a 0-sample SEED must not
            # be presented as a measured fact — `_depth_eta_text` alone discards
            # that provenance, so append the same disclosure the depth picker
            # shows (`_depth_eta_measured`/`_depth_eta_guess`).
            _dr_secs, _dr_n = tg_bot._depth_eta(depth)
            _dr_provenance = (tg_bot._t("depth_eta_measured", lang, n=_dr_n) if _dr_n
                              else tg_bot._t("depth_eta_guess", lang))
            self._send_text(chat_id,
                            tg_bot._t("dr_started", lang, name=tg_bot._depth_name(depth, lang),
                               eta=tg_bot._depth_eta_text(depth, lang))
                            + "\n" + _html_mod.escape(_dr_provenance),
                            parse_mode="HTML")
            old_cb = getattr(ctx, "stage_callback", None)

            def _dr_stage(s: str) -> None:
                _update_status(
                    f"🔬 <b>{_html_mod.escape(_stages.translate(s, lang))}…</b>")
                with self._stages_lock:
                    self._active_stages[chat_id] = f"🔬 {s}"
                self._on_stage(chat_id, f"🔬 {s}", False)

            # NOTE: this used to `ctx.cancel_event.clear()` here, on the theory
            # that a previous task might have left it set. That is no longer
            # true (or possible): `_cancel` is a brand-new threading.Event per
            # task (see _execute_task), so nothing could have set this event
            # except the user pressing Stop/Cancel between the status message
            # (with its live ⛔ button, sent a few lines above) and here — and
            # THAT press must not be erased.
            with self._stages_lock:
                self._active_stages[chat_id] = "🔬 Deep Research"
            ctx.stage_callback = _dr_stage
            # Deep research is ALWAYS a multi-minute run by the time it reaches
            # here (planning is already done above) — mark this chat
            # interruptible right away so a message the user sends while it
            # crawls gets answered without waiting for the whole report.
            self._mark_interruptible(chat_id)
            keepalive = tg_bot._Typing(self._api_post, chat_id).start()

            result: dict = {}
            try:
                import deep_research as _dr
                # The pipeline searches and briefs in English (that is where the
                # sources are) but the DOCUMENT must come back in the language the
                # TOPIC was actually written in — not the session's RU/EN UI
                # toggle, which a user may have set once and never touched again.
                # A Russian-UI user typing "do a deep research on: quantum
                # computing" wants an English paper, and an English-UI user
                # pasting a Russian topic wants a Russian one.
                # `lang_of_text` only confidently detects RUSSIAN (a Cyrillic
                # ratio check) — anything else, including genuine English, comes
                # back as whatever `default` is. Passing `default=lang` here
                # would silently defeat the whole fix (an English topic under a
                # Russian-UI session would still read back as "ru", since a
                # Latin-script topic looks identical to "no signal" to this
                # function). Use its own default ("en") instead, matching the
                # in-graph tool-calling path (tools.py's deep_research handler),
                # which never passes a UI-lang fallback either.
                _out_lang = _dr.lang_of_text(topic)
                result = _dr.run_deep_research(ctx, topic, out_lang=_out_lang, depth=depth,
                    progress=lambda phase, stats, msg: _dr_stage(phase))
            except Exception as exc:
                tg_bot.logger.exception("Telegram deep research failed chat=%s", chat_id)
                _clear_status()
                err = str(exc)[:300]
                # Recorded HERE, at delivery of THIS failure, not back when the
                # task was merely pushed to the queue. A chat may have a second
                # task queued right behind this one (or admitted to run
                # concurrently — see _mark_interruptible) that hasn't started
                # yet; setting this at push time let that still-pending push
                # overwrite last_task_id before this task even got a chance to
                # fail, so THIS failure's own, genuinely-fresh Retry button was
                # refused as "stale" the instant it was shown — reproduced live
                # via the adversarial harness. Setting it at the moment each
                # failure is actually delivered keeps the existing "newest
                # delivered outcome wins" contract (retry for an outcome the
                # user has since moved past is still correctly refused) without
                # that premature-overwrite race.
                sess.last_task_text = task.user_text
                sess.last_task_id = task.task_id
                try: self._store.put(sess)
                except Exception: pass
                self._send_text(chat_id,
                    tg_bot._t("dr_failed", lang, err=_html_mod.escape(err)),
                    parse_mode="HTML",
                    keyboard={"inline_keyboard": [[
                        {"text": tg_bot._t("retry_btn", lang),
                         "callback_data": f"retry:{task.task_id}"}]]})
                self._on_stage(chat_id, f"❌ Research failed: {err[:80]}", True)
                self._activity.log(chat_id, "error", f"deep_research: {err}", uname)
                return
            finally:
                ctx.stage_callback = old_cb
                keepalive.stop()
                with self._stages_lock:
                    self._active_stages.pop(chat_id, None)

            _clear_status()
            # A Stop/Cancel that landed before the run even started (button was
            # live the whole time above) must not still ship a "partial" report —
            # only `result["cancelled"]` was checked before, which is set by the
            # research loop itself and says nothing about a Stop that pre-empted
            # it entirely.
            if self._is_cancelled(task.task_id) or self._stop_requested_after(task):
                self._send_text(chat_id, tg_bot._t("cancel_done", lang),
                                keyboard=self._main_menu_kb(sess, lang))
                self._on_stage(chat_id, "⛔ Cancelled", True)
                self._activity.log(chat_id, "system",
                                   "[deep research] cancelled before delivery", uname)
                return
            report  = (result.get("report") or "").strip()
            stats   = result.get("stats", {}) or {}
            elapsed = result.get("elapsed_sec", 0) or 0
            main_kb = self._state_kb(sess, lang)
            # Deliberate asymmetry with the normal turn: a cancelled ordinary turn
            # is discarded (cheap to redo), but a cancelled research still ships
            # whatever it gathered, clearly labelled as incomplete — throwing away
            # twenty minutes of crawling helps nobody.
            if report:
                note = tg_bot._t("dr_partial", lang) if result.get("cancelled") else ""
                header = tg_bot._t("dr_header", lang, note=note,
                            sources=stats.get("sources", 0),
                            pages=stats.get("pages", 0),
                            findings=stats.get("findings", 0),
                            elapsed=float(elapsed))
                # A 40-minute report sliced into a dozen chat bubbles is unreadable
                # and unsaveable. Past a threshold it also goes out as a .md file,
                # which the user can keep, forward and open outside Telegram.
                # ALWAYS as a .md file. The old 6000-char threshold meant a short
                # report was pasted into the chat instead — and pasted through
                # _md_to_html, so the user got a wall of chat bubbles with the
                # Markdown converted AWAY. The result of a research run is a
                # document; deliver a document. Chat text is the fallback for a
                # failed upload only.
                sent_as_file = False
                if len(report) > tg_bot._cfg_int("TG_REPORT_FILE_CHARS", 0):
                    sent_as_file = self._send_report_file(
                        chat_id, topic, report, header, lang)
                if sent_as_file:
                    # The .md attachment IS the report. Pasting the same 38k
                    # characters underneath it as a dozen bubbles only buries the
                    # chat. The user gets the stats card plus a file they can
                    # keep, forward and open outside Telegram.
                    self._send_text(chat_id, header, parse_mode="HTML",
                                    keyboard=main_kb)
                else:
                    # No file — either the report is short, or the upload failed.
                    # Then the text is the only copy there is, so it still goes
                    # to the chat rather than being silently lost.
                    reply = header + tg_bot._md_to_html(report)
                    chunks = tg_bot._split_html(reply)
                    for i, chunk in enumerate(chunks):
                        self._send_text(chat_id, chunk, parse_mode="HTML",
                                        keyboard=main_kb if i == len(chunks) - 1 else None)
            else:
                reply = tg_bot._t("dr_nothing", lang)
                self._send_text(chat_id, reply, keyboard=main_kb)

            # Mirror to the GUI and persist the turn, exactly like the normal path —
            # otherwise the research is invisible in the activity feed and a follow-up
            # question ("а сколько стоит?") has no idea what was just researched.
            self._on_message(chat_id, task.user_text, report or reply)
            self._activity.log(chat_id, "bot_reply",
                               f"[deep research] {stats.get('findings', 0)} findings", uname)
            self._on_stage(chat_id, "✅ Done", True)
            try:
                with self._history_lock(chat_id):
                    hist = sess.get_history()
                    hist.append({"role": "user", "content": task.user_text})
                    hist.append({"role": "assistant",
                                 "content": f"[deep research report on '{topic}']\n\n"
                                            + (report or reply)[:6000]})
                    sess.set_history(hist)
                    self._store.put(sess)
            except Exception:
                tg_bot.logger.exception("deep research history save failed chat=%s", chat_id)
            return
        # ── end deep research bypass ───────────────────────────────────────────

        # ── a song ─────────────────────────────────────────────────────────────
        # Same shape as research: a long render that is not the agent's to
        # plan. Runs under this task's status line and cancel event.
        _song_topic, _song_secs = _tg_songs.parse_song_payload(task.user_text)
        if _song_topic:
            self._on_stage(chat_id, "🎵 Writing a song", False)
            self._activity.log(chat_id, "stage", "song", uname)
            # No separate "song_generating" line: the status message with its
            # ⛔ button already says «🎵 Пишу текст песни…» (it doubled, live 2026-09-25).

            def _song_stage(s: str) -> None:
                _update_status(
                    f"🎵 <b>{_html_mod.escape(_stages.translate(s, lang))}…</b>")
                with self._stages_lock:
                    self._active_stages[chat_id] = f"🎵 {s}"
                self._on_stage(chat_id, f"🎵 {s}", False)

            with self._stages_lock:
                self._active_stages[chat_id] = "🎵 Writing a song"
            _old_cb = getattr(ctx, "stage_callback", None)
            ctx.stage_callback = _song_stage
            self._mark_interruptible(chat_id)
            keepalive = tg_bot._Typing(self._api_post, chat_id).start()
            try:
                sent = self._generate_song(chat_id, _song_topic, lang, _song_secs, ctx=ctx)
            finally:
                ctx.stage_callback = _old_cb
                keepalive.stop()
                with self._stages_lock:
                    self._active_stages.pop(chat_id, None)
            _clear_status()
            if self._is_cancelled(task.task_id) or self._stop_requested_after(task) \
                    or (not sent and ctx.is_cancelled()):
                self._send_text(chat_id, tg_bot._t("cancel_done", lang),
                                keyboard=self._main_menu_kb(sess, lang))
                self._on_stage(chat_id, "⛔ Cancelled", True)
                self._activity.log(chat_id, "system", "[song] cancelled", uname)
                return
            self._on_message(chat_id, task.user_text, "[song]" if sent else "[song failed]")
            self._activity.log(chat_id, "bot_reply", "[song] delivered" if sent else "[song] failed", uname)
            self._on_stage(chat_id, "✅ Done" if sent else "❌ Song failed", True)
            return
        # ── end song ───────────────────────────────────────────────────────────

        # ── 🎨 restyle a video (tg_restyle): same shape as the song ─────────────
        import tg_restyle as _tg_restyle
        if task.user_text.startswith(_tg_restyle.PAYLOAD):
            _look = task.user_text[len(_tg_restyle.PAYLOAD):].strip()

            def _rs_stage(s: str) -> None:
                _update_status(f"🎨 <b>{_html_mod.escape(_stages.translate(s, lang))}…</b>")
                with self._stages_lock:
                    self._active_stages[chat_id] = f"🎨 {s}"
                self._on_stage(chat_id, f"🎨 {s}", False)

            _old_cb = getattr(ctx, "stage_callback", None)
            ctx.stage_callback = _rs_stage
            ctx.set_stage("Restyling the video")
            self._mark_interruptible(chat_id)
            try:
                sent = self._run_restyle(chat_id, lang, _look, ctx)
            finally:
                ctx.stage_callback = _old_cb
                with self._stages_lock:
                    self._active_stages.pop(chat_id, None)
            _clear_status()
            if self._is_cancelled(task.task_id) or self._stop_requested_after(task) \
                    or (not sent and ctx.is_cancelled()):
                self._send_text(chat_id, tg_bot._t("cancel_done", lang),
                                keyboard=self._main_menu_kb(sess, lang))
                self._on_stage(chat_id, "⛔ Cancelled", True)
                return
            self._activity.log(chat_id, "bot_reply", "[restyle] delivered" if sent else "[restyle] failed", uname)
            self._on_stage(chat_id, "✅ Done" if sent else "❌ Restyle failed", True)
            return

        on_stage = self._make_stage_callback(chat_id, lang, uname, _update_status)

        with self._stages_lock:
            self._active_stages[chat_id] = "⚙️ Starting"

        # Scoped ctx: this callback and this cancel event belong to this task
        # alone, so there is nothing to save and nothing to clear — the event is
        # freshly created above, and clearing a SHARED one used to wipe a Stop
        # that arrived while the task was still starting up.
        ctx.stage_callback = on_stage
        # The pre-tool note goes out as an ordinary message, above the status
        # line, so the user hears «рисую, это около минуты» right away.
        ctx.interim_callback = lambda text: (
            self._send_text(chat_id, text),
            self._activity.log(chat_id, "bot_reply", "[interim] " + text[:200], uname))
        keepalive = tg_bot._Typing(self._api_post, chat_id).start()

        base = self._get_base_state()
        history = sess.get_history()
        # Snapshot length: if an interject task is admitted for this chat (see
        # _mark_interruptible), the OTHER task may commit its own turn to
        # sess.history before this one finishes. Overwriting wholesale with
        # `new_msgs` (built from THIS snapshot) at the end of the turn would
        # silently erase that other turn. Recorded here so the end-of-turn
        # commit can rebase: keep only the messages THIS turn added and append
        # them to whatever is in sess.history by then, instead of replacing it.
        _hist_snapshot_len = len(history)
        if history:
            base["messages"] = history
        # ── document-backed answering (RAG) ───────────────────────────────────
        # When the user has documents indexed and search is on, the question is
        # answered from retrieved passages instead of the model's own memory.
        user_input = task.user_text
        # A message that POINTS AT a picture (a reply to one, a button under
        # one) is about that picture, not about the documents: «Что
        # нарисовано?» over a forwarded sketch was wrapped as a question to
        # the library and answered with an image edit (live 2026-09-18 00:44).
        _about_a_picture = bool(img_bytes or getattr(task, "image_id", "")
                                or getattr(sess, "target_image", "")
                                or _about_the_last_video(getattr(sess, "last_image_path", ""),
                                                          user_input))
        if sess.use_docs and not _about_a_picture and not _plainly_not_a_doc_question(user_input):
            lib = None
            try:
                if self._library_path(chat_id).exists():
                    import knowledge_client
                    lib = self._open_library(chat_id)
                    rag_prompt, info = knowledge_client.build_rag_prompt(lib, user_input)
                    if rag_prompt != user_input:
                        user_input = rag_prompt
                        on_stage("Reading your documents")
                    if info:
                        self._activity.log(chat_id, "system", f"[library] {info}", uname)
            except Exception:
                tg_bot.logger.exception("library retrieval failed chat=%s", chat_id)
            finally:
                if lib is not None:
                    try: lib.close()
                    except Exception: pass

        base["user_input"]          = user_input
        base["image_data"]          = img_bytes

        # Isolate Telegram session memory from the GUI's ctx.session_memory.
        # We swap in this session's own memory before invoke so the graph sees only
        # this user's past turns, not the GUI's conversation, and we restore the GUI's
        # memory after — Telegram turns never bleed into the main-app context.
        # `ctx` is this task's own scoped copy, so filling it is enough — there is
        # nothing global left to clobber and nothing to restore.
        ctx.session_memory.extend(sess.get_tg_memory())
        base["session_memory_text"] = ctx.memory_text()

        # Same treatment for pinned facts — and they matter more. Session memory
        # rolls out of the window after a few turns; facts written by remember_fact
        # are injected into EVERY prompt, so a globally shared list meant one user's
        # "remember my address is …" was handed to the next user who typed anything.
        # prompt_guard drops injected "facts" saved before the write guard
        # existed; the cleaned list is what gets written back after the turn.
        from prompt_guard import fact_rejection
        # load_memory above already read facts.json, the same facts again:
        # without the text check every turn doubled them (live: 4x «Казань»).
        _seen = {str((f or {}).get("text", "")) for f in ctx.pinned_facts if isinstance(f, dict)}
        for f in sess.get_tg_facts():
            t = str((f or {}).get("text", ""))
            if t not in _seen and not fact_rejection(t):
                _seen.add(t)
                ctx.pinned_facts.append(f)
        # «я живу в Новосибирске»: clock lines and "at 9:00" reminders follow
        # the user's zone, not the server's (live: 23:48 told at 03:48).
        try:
            from tg_weather import city_tz as _city_tz
            _home = self._fact_city(chat_id)
            ctx.user_tz = _city_tz(_home) if _home else ""
        except Exception:
            tg_bot.logger.exception("user tz lookup failed chat=%s", chat_id)
            ctx.user_tz = ""

        # Mirror THIS session's voice setting onto ctx for the duration of the call.
        # graph.tts_node only consults the global ctx.tts_disabled (the GUI mute), so a
        # Telegram user who turned voice OFF still had their reply synthesized: wasted
        # GPU time, and ctx.set_stage("Speaking") surfaced a "recording voice" indicator
        # for a voice note that was never going to be sent. Restored in the finally
        # below so the GUI's own mute setting is never clobbered.
        ctx.tts_disabled = not (sess.voice_on and self._tts_enabled_fn())

        # Same swap for the working image. ctx.last_image_prompt is the base prompt
        # redraw/edit build on, so sharing it globally let one chat's prompt (or a
        # pre-clear prompt) steer the next generation.
        # An image the user POINTED AT (replied to it, or pressed a button under
        # it) outranks "the most recent one". This is the whole fix for a request
        # about one picture being answered about another; without it the pipeline
        # only ever sees a single "current image" slot.
        # `task.image_id` — captured at PRESS time and carried on the task
        # itself — wins over the lazy `sess.target_image` slot: two button
        # presses on two different pictures, queued one behind the other, both
        # used to resolve to whichever picture the SECOND press had already
        # overwritten that slot with by the time either task actually ran.
        _target_id = getattr(task, "image_id", "") or getattr(sess, "target_image", "")
        _target = tg_bot._image_by_id(sess, _target_id)
        if (task.user_text or "").startswith(tg_bot._TEXT_ONLY_MARK):
            # An internal instruction over a transcript (the 📋 button on a
            # forwarded voice/video). Live 23:02: it inherited the chat's last
            # picture (a collage), went down the vision path and answered
            # about «контекст нашего диалога» instead of the video.
            _target = None
            ctx.last_image_path = None
        elif _target and os.path.exists(_target.get("path", "")):
            ctx.last_image_path = _target["path"]
            # graph.needs_relook: the user's OWN words about a picture they
            # pointed at (a reply to it, a message after ❓) are answered by
            # LOOKING at it (live 2026-09-18 00:44 «Что нарисовано?» as a reply
            # to a sketch became an enhance pass). A button's own task carries
            # the picture in task.image_id and is an action, not words about it.
            ctx.image_pointed_at = not getattr(task, "image_id", "")
            self._activity.log(chat_id, "system",
                               f"[target] using image {_target['id']} "
                               f"({os.path.basename(_target['path'])})", uname)
        else:
            # The picture in play: the one the PREVIOUS turn was about (sent,
            # pointed at, delivered, a video's frames). Not "the last picture
            # ever": live 2026-10-01 «Что думаешь?» about a forwarded video
            # re-read a quote photo from an hour before. A picture further
            # back is one reply, button or «на первой картинке» away.
            _in_play = tg_bot._image_by_id(sess, getattr(sess, "turn_image", "") or "")
            ctx.last_image_path = (_in_play["path"] if _in_play and os.path.exists(_in_play.get("path", ""))
                                   else None)
        # «верни как было» in a later message: this chat's own pictures, oldest first.
        ctx.image_undo = [e["path"] for e in (getattr(sess, "image_log", None) or [])
                          if e.get("path") and e["path"] != ctx.last_image_path][-10:]
        # 🎭 Style: the only path in Telegram that hands the agent TWO images.
        # transfer_image (tool_image_handlers._handle_transfer_image) reads
        # ctx.reference_images and nothing else -- the desktop GUI's Transfer
        # tab populates it directly; a TG task never did before this, so
        # transfer_image was unreachable from Telegram entirely.
        _style_ref = getattr(task, "style_ref_path", "")
        if _style_ref and os.path.exists(_style_ref) and ctx.last_image_path:
            ctx.reference_images = [ctx.last_image_path, _style_ref]
        # What THIS turn is about, for the next one's "the picture": a photo
        # sent with the message, or the one pointed at; a delivered picture
        # overwrites it below, and a turn with none leaves it empty.
        sess.turn_image = (_target or {}).get("id", "") or (
            (tg_bot._image_by_path(sess, task.image_path or "") or {}).get("id", "")
            if getattr(task, "image_path", "") else "") or (
            (tg_bot._image_by_path(sess, ctx.last_image_path or "") or {}).get("id", ""))
        # Remember which register entry this task starts from, so the picture
        # it delivers can be recorded as a VERSION of it (see _one_lineage).
        _derives = bool(_target) or (task.user_text or "") in tg_bot._MACHINE_PAYLOADS or (task.user_text or "").startswith(tg_bot._PROMPT_KB.get("edit_image", "edit the image"))
        ctx.source_image_id = ((tg_bot._image_by_path(sess, ctx.last_image_path or "") or {}).get("id", "")
                               if _derives else "")
        # One turn only: a target is a gesture the user made about THIS message,
        # not a mode they switched on.
        sess.target_image = ""
        ctx.last_image_prompt = sess.last_image_prompt or ""
        ctx.last_deck = sess.last_deck or None

        # Every picture still on disk in this chat's register, oldest first. Video
        # generation is the first feature that legitimately wants MORE than one
        # ("make a video from these three"); the single working-image slot cannot
        # express that, and without this the reference modes would silently degrade
        # to animating whichever picture happened to be last.
        #
        # BUT a task that points at ONE SPECIFIC picture (a button pressed under
        # it, e.g. animate_preset) must stay scoped to exactly that picture. Live,
        # 2026-09-19: pressing "Animate" under a freshly style-transferred photo
        # animated the ORIGINAL instead — task.image_path/ctx.last_image_path
        # correctly named the styled result, but this block unconditionally
        # dumped the chat's WHOLE picture history (9 images) into
        # ctx.recent_image_paths too, and tool_image_handlers._current_image_paths
        # concatenates both lists -- generate_video saw 9 candidates, picked
        # "reference" mode instead of a clean single-image animate, and the model
        # ended up drawing on the wrong one. A single-target task offers only its
        # own picture; the multi-image case stays intact for a general request
        # with no specific target ("animate these three photos").
        #
        # And a general request offers the picture IN PLAY, never the history:
        # live 2026-10-01 «анимируй: два деда дерутся» with no target sent all 9
        # pictures of the chat (a lettering, an old portrait…) as references, and
        # the clip lost the user's voices. Several pictures of THIS turn still come
        # through state["image_paths"] (an album sent with the request).
        _one = _target["path"] if _target else ctx.last_image_path
        ctx.recent_image_paths = [_one] if (_one and os.path.exists(_one)) else []

        # The last clip delivered here, so "use that video" resolves.
        _lv = self._chat_videos.get(chat_id) or ""
        ctx.last_video_path = _lv if (_lv and os.path.exists(_lv)) else None

        # Output size (Draw ▸ 📐 Size) — same swap, same reason: image.py reads it
        # off ctx, so leaving it global would give one chat another chat's size.
        ctx.image_aspect  = sess.image_aspect
        ctx.image_quality = sess.image_quality

        final: dict = {}
        invoke_ok = False
        # graph.invoke() is a plain blocking call, and graph.py/tools.py never
        # look at ctx.cancel_event on the ordinary-turn path (only
        # deep_research's own progress loop does — see the bypass above) — so
        # a hung LLM/tool call inside invoke() cannot be unstuck cooperatively
        # the way the watchdog unsticks everything else. Left as a direct call,
        # a genuinely hung invoke() would block THIS consumer thread forever:
        # _execute_task's cleanup (which frees the chat's admission slot and
        # removes the task from _running_task) lives in a `finally` that never
        # runs, so that chat is wedged permanently and the pool loses a worker.
        # Running it on its own thread and joining with a hard timeout lets the
        # consumer thread — and this chat — recover even when the call itself
        # never returns. A Python thread cannot be killed, so a truly hung call
        # still leaks a background thread (same tradeoff already accepted for
        # the wedged-poll-thread case in the watchdog above); its eventual
        # result, if any, is simply never read past this point.
        _invoke_out: dict = {}
        _invoke_err: list = []

        # Cleared per turn so that whatever is found here after a timeout was
        # rendered by THIS task and not left over from the last one.
        try:
            ctx.last_render_path = None
            ctx.last_render_status = ""
        except Exception:
            pass

        def _run_invoke() -> None:
            import chatlog as _chatlog      # its own thread: bind again for the transcript
            try:
                with _chatlog.bind(chat_id):
                    _invoke_out["v"] = graph.invoke(base)
            except Exception as exc:
                _invoke_err.append(exc)

        _invoke_thread = turn_trace.spawn(_run_invoke, name=f"tg-invoke-{task.task_id[:8]}")
        # A flat deadline on the whole turn cannot tell "this task is stuck"
        # apart from "this task has not been given the GPU yet". Measured live:
        # a Music3 track holds the GPU for ~70 minutes, and a picture queued
        # behind one burned its entire 300s WAITING and was abandoned before it
        # ran a single step -- the user saw "took too long, try again" for a
        # request that had not started. Retrying just re-joined the same queue.
        #
        # So: wait out the base deadline, then keep waiting for as long as the
        # render server is genuinely working, up to the hard per-task ceiling.
        # An unreachable ComfyUI reports NOT busy, so a real outage still fires
        # the deadline instead of hanging until the ceiling.
        _invoke_timeout = max(5, tg_bot._cfg_int("TG_INVOKE_TIMEOUT_S", 1800))
        _hard_ceiling = max(_invoke_timeout,
                            tg_bot._cfg_int("TG_TASK_MAX_S", 3600))
        # Coarse in production (a 1800s base needs no finer grain); scales down
        # for small deadlines so tests do not have to wait a quarter minute.
        _STEP = min(15.0, max(0.05, _invoke_timeout / 4.0))
        _waited = 0.0
        _extended = False
        while True:
            _invoke_thread.join(timeout=min(_STEP, max(1.0, _hard_ceiling - _waited)))
            if not _invoke_thread.is_alive():
                break
            _waited += _STEP
            if _waited >= _hard_ceiling:
                break
            if _waited >= _invoke_timeout:
                try:
                    import comfy_client as _comfy_mod
                    if not (_comfy_mod.card_in_use() or _comfy_mod.server_busy()):
                        break
                except Exception:
                    break
                if not _extended:
                    _extended = True
                    tg_bot.logger.info(
                        "graph.invoke past %ss for chat=%s task=%s but the render "
                        "server is busy — extending to %ss",
                        _invoke_timeout, chat_id, task.task_id[:8], _hard_ceiling)
        _invoke_timeout = int(_waited) or _invoke_timeout

        try:
            # Recorded at delivery of each failure below, not at push time —
            # a second task queued (or concurrently admitted) right behind
            # this one must not be able to mark THIS task's own Retry button
            # stale before this task even got a chance to fail. See the
            # matching comment on the deep-research failure handler above.
            if _invoke_thread.is_alive():
                tg_bot.logger.error(
                    "graph.invoke exceeded %ss chat=%s task=%s — abandoning "
                    "(consumer thread freed; call may still run in the "
                    "background)", _invoke_timeout, chat_id, task.task_id[:8])
                _clear_status()
                sess.last_task_text = task.user_text
                sess.last_task_id = task.task_id
                try: self._store.put(sess)
                except Exception: pass
                # The deadline is not proof that nothing was made. A render that
                # already finished is the answer the user asked for, and the
                # inspection calls that ran the clock out do not unmake it --
                # so hand it over before admitting the timeout. Measured live:
                # the picture existed, scored 9/10, and was discarded because
                # two more tool calls came after it.
                # The task's SCOPED ctx, not self._ctx: the shared one belongs
                # to the desktop app and every other chat.
                salvaged = self._deliver_salvaged_render(
                    ctx, chat_id, sess, task, lang, uname)
                # Stop the abandoned graph: it keeps running in its thread and,
                # left alone, goes on forcing renders nobody will receive while
                # the user is already offered Retry -- which would run it all a
                # second time. Measured 2026-09-19: generate_video re-forced
                # after the 3030 s abandon. The round loop and the tool loop
                # both check is_cancelled(); this token is THIS task's own.
                try:
                    ctx.cancel_event.set()
                except Exception:
                    pass
                if not salvaged:
                    self._send_text(chat_id, tg_bot._t("err_timeout", lang),
                        parse_mode="HTML",
                        keyboard={"inline_keyboard": [[
                            {"text": tg_bot._t("retry_btn", lang),
                             "callback_data": f"retry:{task.task_id}"}]]})
                self._on_stage(chat_id, "❌ Timed out", True)
                self._activity.log(chat_id, "error",
                    f"graph.invoke: exceeded {_invoke_timeout}s"
                    + (" (finished render delivered)" if salvaged else ""), uname)
            elif _invoke_err:
                exc = _invoke_err[0]
                tg_bot.logger.error("graph.invoke failed chat=%s: %r", chat_id, exc)
                _clear_status()
                err_text = str(exc)[:300]
                sess.last_task_text = task.user_text
                sess.last_task_id = task.task_id
                try: self._store.put(sess)
                except Exception: pass
                self._send_text(chat_id,
                    tg_bot._t("err_generic", lang, err=_html_mod.escape(err_text)),
                    parse_mode="HTML",
                    keyboard={"inline_keyboard": [[
                        {"text": tg_bot._t("retry_btn", lang),
                         "callback_data": f"retry:{task.task_id}"}]]})
                self._on_stage(chat_id, f"❌ Error: {err_text[:80]}", True)
                self._activity.log(chat_id, "error", f"graph.invoke: {err_text}", uname)
            else:
                _raw = _invoke_out.get("v")
                if isinstance(_raw, dict):
                    final = _raw
                    invoke_ok = True
                else:
                    # A malformed/missing result shape (None, a bare string, a
                    # list, ...) must be treated as a failure here, not handed
                    # to the delivery code below — every step of it assumes a
                    # dict (`final.get(...)`) and would raise uncaught, which
                    # the consumer loop only logs; the user would see nothing.
                    tg_bot.logger.error(
                        "graph.invoke returned %r (expected dict) chat=%s",
                        type(_raw).__name__, chat_id)
                    _clear_status()
                    sess.last_task_text = task.user_text
                    sess.last_task_id = task.task_id
                    try: self._store.put(sess)
                    except Exception: pass
                    self._send_text(chat_id,
                        tg_bot._t("err_generic", lang, err="malformed result"),
                        parse_mode="HTML",
                        keyboard={"inline_keyboard": [[
                            {"text": tg_bot._t("retry_btn", lang),
                             "callback_data": f"retry:{task.task_id}"}]]})
                    self._on_stage(chat_id, "❌ Error: malformed result", True)
                    self._activity.log(chat_id, "error",
                        f"graph.invoke: returned {type(_raw).__name__}, not a dict",
                        uname)
        finally:
            # Save this session's memory and restore the GUI's memory.
            # Only PERSIST what this turn produced. The scoped ctx is discarded
            # with the task, so there is no global state to hand back.
            sess.set_tg_memory(list(ctx.session_memory))
            sess.set_tg_facts(list(ctx.pinned_facts))
            sess.last_image_path   = getattr(ctx, "last_image_path", "") or ""
            sess.last_image_prompt = getattr(ctx, "last_image_prompt", "") or ""
            sess.last_deck = getattr(ctx, "last_deck", None) or {}
            with self._task_lock:
                self._task_cancels.pop(task.task_id, None)
            keepalive.stop()
            with self._stages_lock:
                self._active_stages.pop(chat_id, None)

        if not invoke_ok: return

        _clear_status()

        # Cancelled mid-flight: the graph returns whatever it had, but the user
        # asked for it to stop — delivering the half-finished answer (and a voice
        # note of it) is exactly what they pressed the button to avoid.
        #
        # BOTH cancel paths have to be checked here. This used to test only the
        # inline ⛔ button's set, so the ⛔ Stop KEY was honoured for queued work
        # and ignored for the work in flight: cancellation is cooperative, and a
        # ComfyUI render that finished before the loop reached its next check
        # sailed straight through to delivery. Observed live — Stop answered
        # "Останавливаю…", and the picture arrived anyway two minutes later.
        if self._is_cancelled(task.task_id) or self._stop_requested_after(task):
            self._send_text(chat_id, tg_bot._t("cancel_done", lang),
                            keyboard=self._main_menu_kb(sess, lang))
            self._on_stage(chat_id, "⛔ Cancelled", True)
            return

        new_msgs = final.get("messages", base["messages"])
        # "(no response)" was a debug placeholder that went out verbatim, in
        # English, in a Russian chat. Kept as an internal sentinel (voice must
        # not be synthesised for it) but never shown.
        reply = (final.get("final_answer") or "").strip()
        empty_reply = not reply
        from graph_finalize import is_failure_placeholder
        reply_is_failure = is_failure_placeholder(reply)
        if empty_reply:
            # An image-, video-, document- or report-only turn legitimately has
            # no text: the artefact IS the answer, and it is delivered further
            # down. Announcing "модель ничего не ответила" and then handing over
            # the picture is worse than the placeholder it replaced.
            produced = any(final.get(k) for k in (
                "image_path", "image_status", "video_path", "document_path",
                "research_report", "tts_path"))
            reply = "" if produced else tg_bot._t("empty_reply", lang)
        if final.get("ask_voices"):
            # The clip waits for «свои голоса / стандартные»: the offer below IS the
            # reply. The model's own line said «не удалось из-за технической ошибки»
            # about a clip that was only paused (live 10-02).
            reply = ""

        # ── deliver voice note (before text so keyboard attaches to voice) ────
        # When voice is on, the voice note IS the primary reply — no text duplicate.
        # Text is sent only if voice fails (fallback) or voice is off.
        voice_sent = False
        main_kb = self._state_kb(sess, lang)
        # This turn was ABOUT a specific picture (an upload, or one the user
        # pointed at) and produced no artifact of its own (a tool that DOES
        # deliver a picture/video/document attaches its own _image_kb further
        # down, keyed to what it just made — this must not compete with that).
        # Live, 2026-09-19: sending a bare photo got a description back and
        # then nothing else — every editing action (style, animate, upscale…)
        # required leaving the chat for the 🎨 Creativity menu.
        # NOT sess.turn_image directly: that slot is also filled by a video
        # note's frame SHEET (tg_resolve._look_video's img_bytes, registered
        # the same way a real photo is) -- a first pass here keyed off it and
        # a следующий кружок immediately inherited the picture-editing
        # keyboard. task.image_is_photo is only ever True for an actual
        # photo/album upload (tg_resolve.py's "photo"/"album" branches), so a
        # derived sheet never qualifies no matter what sess.turn_image holds.
        _turn_img_id = (_target or {}).get("id", "") if _target else (
            sess.turn_image if getattr(task, "image_is_photo", False) else "")
        if _turn_img_id and not any(final.get(k) for k in
                                    ("image_status", "video_status", "document_status")):
            _turn_img_entry = tg_bot._image_by_id(sess, _turn_img_id)
            if _turn_img_entry and os.path.exists(_turn_img_entry.get("path", "")):
                main_kb = tg_bot._image_kb(lang, _turn_img_id)
                # Edit buttons under a bare comment read as a mistake (live 2026-09-29): ask first.
                if reply and not reply_is_failure:
                    reply = reply.rstrip() + "\n\n" + tg_bot._t("img_offer", lang)
        try:
            if sess.voice_on and self._tts_enabled_fn() \
                    and reply and not empty_reply:
                tts_wav = final.get("tts_path", "")
                if tts_wav and os.path.exists(tts_wav):
                    voice_sent = self._send_voice_from_wav(
                        chat_id, tts_wav, keyboard=main_kb if sess.reply_mode == "voice" else None)
                    if voice_sent:
                        self._activity.log(chat_id, "system",
                            "[voice] sent from graph tts_path", uname)
                    else:
                        self._activity.log(chat_id, "error",
                            "[voice] sendVoice failed — falling back to text", uname)
                else:
                    self._activity.log(chat_id, "system",
                        f"[voice] tts_path={tts_wav!r} — re-synthesising", uname)
                    # Must honour the return value: when synthesis or ffmpeg
                    # fails this used to set voice_sent=True anyway, so the text
                    # fallback was skipped and the user received NOTHING AT ALL.
                    voice_sent = self._send_voice(ctx, chat_id, reply)
        except Exception:
            tg_bot.logger.exception("voice delivery failed chat=%s", chat_id)

        # ── deliver text (skip when voice already sent the reply) ─────────────
        try:
            # Code cannot be listened to: a reply that carries a fenced block
            # is sent as text as well, even when the voice note went out.
            # And a reply that is not Russian is only APPROXIMATED by the
            # voice: the house TTS speaks English through the Russian voice by
            # transliteration ("а драматик лайтхаус стэндс…" -- live,
            # 2026-09-12, journey 6), which is a hint, not the words. The text
            # goes out beside it so the reader has the actual sentence.
            # Every reply goes out as text too: users read, copy and skim; a
            # voice note alone was the top complaint of the persona run
            # 2026-09-27 («генератор голосовух», «ТЕКСТОМ!» five times).
            # Unless the user picked voice only: then code and non-Russian still go as text.
            if reply and not (voice_sent and sess.reply_mode == "voice" and "```" not in reply
                              and lang == "ru" and len(_re.findall(r"[A-Za-z]", reply)) < len(reply) // 4):
                _shown = reply
                if not reply_is_failure and getattr(ctx, "turn_products", None):
                    import tg_product_cards as _cards
                    _shown = _cards.condense(reply, list(ctx.turn_products))
                reply_html = tg_bot._md_to_html(_shown)
                # A searched answer gets its shape: what was searched, [n]
                # marks instead of "(source: domain)", the sources underneath.
                if not reply_is_failure and (getattr(ctx, "turn_queries", None)
                                             or getattr(ctx, "turn_sources", None)):
                    import tg_reply_shape as _shape
                    reply_html = _shape.shape_search_reply(
                        reply_html, list(ctx.turn_queries or []), list(ctx.turn_sources or []), lang)
                else:
                    # Nothing was searched: a "(source: un.org)" here is the
                    # model imitating the citation format, not a source (live
                    # 2026-09-28). It must not pass for a checked fact.
                    import tg_reply_shape as _shape
                    reply_html = _shape._CITE_RE.sub("", reply_html).replace(" .", ".")
                reply_html = self._sandbox_linkify(chat_id, reply_html)
                chunks = tg_bot._split_html(reply_html)
                for i, chunk in enumerate(chunks):
                    kb = main_kb if i == len(chunks) - 1 else None
                    self._send_text(chat_id, chunk, parse_mode="HTML", keyboard=kb)
            self._on_message(chat_id, task.user_text, reply)
            self._activity.log(chat_id, "bot_reply", reply, uname)
        except Exception:
            tg_bot.logger.exception("text delivery failed chat=%s", chat_id)

        # ── Ozon product cards: one message per product the answer links to ──
        try:
            if reply and not reply_is_failure and getattr(ctx, "turn_products", None):
                import tg_product_cards as _cards
                _cards.send_cards(self, chat_id, reply, list(ctx.turn_products), lang)
        except Exception:
            tg_bot.logger.exception("product cards failed chat=%s", chat_id)

        # ── deliver image ─────────────────────────────────────────────────────
        # Every branch below that does NOT send a picture used to just write a
        # line to the activity log. The text reply has already gone out by then,
        # so when the model narrated "I expanded the borders of your image" and
        # delivery was skipped, the user was told the work was done and received
        # nothing — with no way to tell a silent failure from a slow one.
        # `delivered` is what the correction at the end keys off.
        # ── deliver a generated document (a .pptx deck) ───────────────────────
        # Same contract as an image: the tool says it built a file, so the file has
        # to actually arrive. A tool that reports success while the user receives
        # nothing is the failure mode this whole delivery block exists to prevent.
        doc_path = final.get("document_path") or ""
        doc_delivered = False
        if doc_path and final.get("document_status") == "success":
            if not os.path.exists(doc_path):
                self._activity.log(chat_id, "error",
                    f"[doc delivery] path does not exist: {doc_path}", uname)
                self._send_text(chat_id, tg_bot._t("doc_send_failed", lang))
            else:
                self._api_post("sendChatAction",
                               {"chat_id": chat_id, "action": "upload_document"})
                if self._send_document(chat_id, doc_path,
                                       caption=os.path.basename(doc_path)):
                    doc_delivered = True
                    self._activity.log(chat_id, "system",
                                       f"[doc delivery] sent {doc_path}", uname)
                else:
                    self._activity.log(chat_id, "error",
                        f"[doc delivery] sendDocument failed: {doc_path}", uname)
                    self._send_text(chat_id, tg_bot._t("doc_send_failed", lang))

        # A presentation the user asked for and did not receive must be admitted,
        # exactly like a missing picture. The 📊 button's prefix is machine-made,
        # so "this turn owed a file" is a fact, not a guess. Live, the deck really
        # was built and the state key carrying its path was silently dropped by
        # the graph — the user was told "презентация готова" and got nothing, with
        # no signal anywhere that delivery had been skipped.
        try:
            if (task.user_text or "").strip().lower().startswith(
                    "create a presentation about:") and not doc_delivered:
                self._send_text(chat_id, tg_bot._t("doc_missing", lang),
                                parse_mode="HTML")
                self._activity.log(chat_id, "error",
                    f"[doc delivery] nothing delivered for a presentation request "
                    f"(status={final.get('document_status')!r} "
                    f"path={doc_path!r}) — corrected the reply", uname)
        except Exception:
            tg_bot.logger.exception("doc-delivery correction failed chat=%s", chat_id)

        # ── the voice question generate_video asked ───────────────────────────
        if final.get("ask_voices"):
            try:
                self._offer_voices(chat_id, sess, lang, task.user_text, int(final["ask_voices"]))
            except Exception:
                tg_bot.logger.exception("voice offer failed chat=%s", chat_id)

        # ── video delivery ────────────────────────────────────────────────────
        # Before the picture block, because a turn that made a CLIP should hand over
        # the clip, and the source stills it was built from are still sitting in
        # state["image_path"] — delivering those instead would look like the video
        # silently failed.
        video_delivered = False
        try:
            out_vid = final.get("video_path") or ""
            vid_ok = (final.get("video_status", "") or "") not in ("fail", "error", "")
            if out_vid and vid_ok:
                if not os.path.exists(out_vid):
                    self._activity.log(chat_id, "error",
                        f"[video delivery] path does not exist: {out_vid}", uname)
                else:
                    self._api_post("sendChatAction",
                                   {"chat_id": chat_id, "action": "upload_video"})
                    if self._send_video(chat_id, out_vid, ctx=ctx):
                        video_delivered = True
                        self._chat_videos[chat_id] = out_vid
                        self._activity.log(chat_id, "system",
                            f"[video delivery] sent {out_vid}", uname)
                    else:
                        self._activity.log(chat_id, "error",
                            f"[video delivery] sendVideo+sendDocument both failed: {out_vid}",
                            uname)
                        self._send_text(chat_id, tg_bot._t("video_send_failed", lang),
                                        parse_mode="HTML")
        except Exception:
            tg_bot.logger.exception("video delivery failed chat=%s", chat_id)

        delivered = video_delivered
        out_img = img_status = ""
        try:
            # A turn that already delivered a clip must not also post the stills it
            # was made from as if they were the answer.
            out_img    = "" if video_delivered else (final.get("image_path") or "")
            img_status = final.get("image_status", "")
            status_ok  = img_status and img_status not in ("fail", "error", "")
            if out_img and status_ok:
                if not os.path.exists(out_img):
                    self._activity.log(chat_id, "error",
                        f"[image delivery] path does not exist: {out_img}", uname)
                else:
                    try:
                        import image as _img_mod
                        if _img_mod.is_intermediate_artifact(out_img):
                            self._activity.log(chat_id, "system",
                                "[image delivery] skipped intermediate artifact", uname)
                            out_img = ""
                    except Exception: pass
                    if out_img:
                        self._api_post("sendChatAction",
                                       {"chat_id": chat_id, "action": "upload_photo"})
                        # Register BEFORE sending so the buttons under the picture
                        # can carry its id; the message id is filled in right after,
                        # which is what makes a reply to it resolvable.
                        # The parent is the picture the user pointed at, or —
                        # for a free-text edit — the one the editing tool
                        # started from (state["image_derived_from"]).
                        _parent = getattr(ctx, "source_image_id", "") or ""
                        if not _parent and final.get("image_derived_from"):
                            _src = final.get("image_derived_from")
                            _parent = ((tg_bot._image_by_path(sess, _src)
                                        or tg_bot._image_by_content(sess, _src)
                                        or {}).get("id", ""))
                        img_id = tg_bot._log_image(sess, out_img,
                                            label=(task.user_text or "")[:80], src="bot",
                                            parent=_parent)
                        sess.turn_image = img_id
                        self._last_photo_msg_id = 0
                        ok = self._send_photo(chat_id, out_img,
                                              keyboard=tg_bot._image_kb(lang, img_id))
                        if ok:
                            tg_bot._log_image(sess, out_img,
                                       msg_id=getattr(self, "_last_photo_msg_id", 0),
                                       src="bot")
                            try: self._store.put(sess)
                            except Exception: pass
                            delivered = True
                            # the toggle, or "нарисуй ..., пришли файлом" in this very
                            # request (was honoured only for an already-sent picture)
                            import tg_dispatch as _tgd
                            if getattr(sess, "photo_file", False) or _tgd._asks_as_file(task.user_text or ""):
                                self._send_document(chat_id, out_img)   # full quality, no JPEG re-encode
                            self._activity.log(chat_id, "system",
                                f"[image delivery] sent {out_img}", uname)
                        else:
                            self._activity.log(chat_id, "error",
                                f"[image delivery] sendPhoto+sendDocument both failed: {out_img}",
                                uname)
            elif out_img and not status_ok:
                # The user's OWN uploaded photo is not a withheld result. state
                # ["image_path"] carries the working input too, so this branch fired
                # on every "here is a picture, what is it?" turn.
                try:
                    import image as _img_mod
                    _is_input = _img_mod.is_uploaded_input(out_img)
                except Exception:
                    _is_input = False
                if not _is_input:
                    self._activity.log(chat_id, "system",
                        f"[image delivery] status={img_status!r} path={out_img} — skipped", uname)
        except Exception:
            tg_bot.logger.exception("photo delivery failed chat=%s", chat_id)

        # ── correct a claim the delivery could not back up ────────────────────
        # Two independent signals that this turn owed the user a picture:
        #   · the request itself was one that can only be answered with an image
        #     (the ✨/🔍/🖼 buttons send a fixed prefix, so this is exact); and
        #   · the pipeline reported an image operation at all.
        # The first is what catches the worst case, where an image tool returned
        # [TOOL ERROR] — leaving image_path and image_status empty — and the model
        # narrated success anyway despite the tool telling it not to.
        try:
            # The uploaded photo must not count as evidence that a picture was
            # owed: describing what the user sent produces no image and owes none.
            try:
                import image as _img_mod
                _out_is_ours = bool(out_img) and not _img_mod.is_uploaded_input(out_img)
            except Exception:
                _out_is_ours = bool(out_img)
            owed_image = (bool(tg_bot._IMAGE_PRODUCING_RE.match(task.user_text or ""))
                          or _out_is_ours or bool(img_status))
            # 10-08 live: «Не удалось нарисовать… Попробовать ещё раз?» was followed by
            # «На самом деле не получилось… не обращай внимания на сообщение выше» --
            # a retraction of a reply that had already admitted the failure.
            if owed_image and not delivered and reply:
                import intent as _intent
                if _intent.ask_yes("A bot answered a user: {text} -- does that answer already "
                                   "tell the user that the picture could not be made?", reply,
                                   default=False):
                    owed_image = False
            if owed_image and not delivered:
                self._send_text(chat_id, tg_bot._t("img_missing", lang),
                                parse_mode="HTML", keyboard=main_kb)
                self._activity.log(chat_id, "error",
                    f"[image delivery] nothing delivered for an image request "
                    f"(status={img_status!r} path={out_img!r}) — corrected the reply",
                    uname)
        except Exception:
            tg_bot.logger.exception("image-delivery correction failed chat=%s", chat_id)

        self._on_stage(chat_id, "✅ Done", True)
        task.delivered = True

        # ── compact history AFTER delivering the reply ────────────────────────
        # Compaction requires an extra LLM call (summariser). Running it before
        # delivery blocked the voice note — the user saw "Speaking…" vanish and
        # got silence. Now the reply lands first; compaction happens quietly after.
        _compacted = False
        try:
            from graph import compact_history_if_needed
            ctx.total_user_turns = sess.total_user_turns
            _pre_compact = _copy_mod.deepcopy(new_msgs)
            new_msgs = compact_history_if_needed(ctx, _pre_compact)
            # compact_history_if_needed returns its INPUT object unchanged (by
            # reference) on every early-out path, and only builds a new list
            # when it actually folds turns into a summary — so identity tells
            # us whether the tail below can still be rebased by simple slicing.
            _compacted = new_msgs is not _pre_compact
            sess.total_user_turns = ctx.total_user_turns
        except Exception:
            pass
        finally:
            ctx.set_stage("")   # clear any "Compacting…" stage left by compaction
        # Rebase onto the CURRENT history rather than overwrite with `new_msgs`
        # wholesale: a concurrently-admitted interject task for this same chat
        # (see _mark_interruptible) may have committed its own turn while this
        # one was still running, and a blind overwrite would erase it. Only the
        # messages THIS turn actually added (after `_hist_snapshot_len`) are
        # appended onto whatever is there now.
        # A turn that ended in the "I produced no answer" placeholder (and left
        # no artefact) is not committed at all: the user was told to ask again,
        # and an unanswered question left in the transcript is what made the
        # model answer the OLD question instead of the next one (journey 5).
        _failed_turn = (not _compacted and reply_is_failure and not any(
            final.get(k) for k in ("image_path", "video_path", "document_path",
                                   "research_report")))
        if _failed_turn:
            tg_bot.logger.info("chat=%s turn produced no answer — not committed to history", chat_id)
        with self._history_lock(chat_id):
            if _failed_turn:
                latest = sess.get_history()
            elif _compacted:
                # Compaction rewrote/shortened the whole list (old turns folded
                # into one summary) — the slice-by-index rebase below no longer
                # lines up, so there is no safe partial merge; use the compacted
                # result as-is. This narrow double-edge case (an interject
                # landing in the exact turn where compaction also fires) can
                # lose that interject's own turn — logged, not silently eaten.
                latest = new_msgs
                tg_bot.logger.info("history compacted during a possibly-concurrent "
                           "chat=%s turn — using compacted result wholesale",
                           chat_id)
            else:
                _added = _turn_messages(new_msgs, _hist_snapshot_len,
                                       (final.get("user_input") or "", base.get("user_input") or ""))
                latest = sess.get_history()
                latest.extend(_added)
            # The passages were for THIS turn: kept in history, 40k chars per
            # question, the next question was answered with the previous
            # answer (live 2026-09-28, «склад номер 100» -> the safe code).
            from library import RAG_HEAD, RAG_QUESTION_SEAM
            latest = [dict(m, content="[asked about the user's documents] "
                           + m["content"].rpartition(RAG_QUESTION_SEAM)[2])
                      if m.get("role") == "user" and isinstance(m.get("content"), str)
                      and m["content"].startswith(RAG_HEAD) else m for m in latest]
            sess.set_history(latest)
            try: self._store.put(sess)
            except Exception: pass

    def _look_video(self, ctx, file_id: str, data: bytes | None = None, lang: str = "ru",
                    transcript: str = "") -> dict:
        """See the picture of a video message: frames every 5 s on one sheet,
        one vision call. Live 2026-09-14 21:23: a forwarded round video was
        only listened to. Returns video_look's dict; description "" on any
        failure -- the audio path is never blocked by the eyes."""
        data = data if data is not None else self._dl_bytes(file_id)
        if not data:
            return {"description": "", "sheet": "", "duration": 0, "times": []}
        src_file = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as fh:
                fh.write(data); src_file = fh.name
            import video_look
            work = Path(tg_bot._IMAGE_DIR) / f"video_{uuid.uuid4().hex[:8]}"
            return video_look.look_at_video(ctx, src_file, work_dir=work, lang=lang,
                                            transcript=transcript)
        except Exception:
            tg_bot.logger.exception("video look failed file_id=%s", file_id)
            return {"description": "", "sheet": "", "duration": 0, "times": []}
        finally:
            try:
                if src_file and os.path.exists(src_file): os.unlink(src_file)
            except Exception: pass

    def _transcribe(self, ctx, file_id: str, media: str = "voice",
                    lang_hint: str = "", data: bytes | None = None,
                    speakers: bool = False) -> str:
        """Download `file_id` and transcribe its audio track.

        `media="video"` is a Telegram video_note (a round video message): the
        file is an mp4 container, not an ogg one, but ffmpeg reads either by
        content regardless of the suffix we give the temp file — passing the
        right suffix just keeps the intermediate file honestly named. Either
        way the audio track is extracted the same way and handed to the same
        ASR path, so a video message is transcribed exactly like a voice note.
        """
        data = data if data is not None else self._dl_bytes(file_id)
        if not data: return ""
        suffix = ".mp4" if media == "video" else ".ogg"
        src_file = wav = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
                fh.write(data); src_file = fh.name
            wav = src_file.replace(suffix, ".wav")
            try:
                subprocess.run(["ffmpeg", "-y", "-i", src_file,
                                "-ar", "16000", "-ac", "1", wav],
                               capture_output=True, check=True, timeout=30)
                src = wav
            except Exception:
                src = src_file
            label_lang = lang_hint or "ru"
            if media == "video":
                # The clip's own language, not the chat's (an English video in a
                # Russian chat was forced to Russian on a <85% guess).
                from audio import detect_media_language
                lang_hint = detect_media_language(ctx, src) or lang_hint
            heard = ""
            if speakers:
                # Several people on one forwarded clip: label who said what.
                # None = one speaker / too short / no diarizer -> plain transcript.
                import diarize
                heard = diarize.speaker_transcript(ctx, src, lang=label_lang,
                                                   asr_lang=lang_hint or "ru") or ""
            if not heard:
                from audio import transcribe_audio_file, fix_voice_imperative
                # The session language is the best guess for a short, mumbled
                # clip; the house default is Russian.
                heard = fix_voice_imperative(
                    transcribe_audio_file(ctx, src, lang_hint=lang_hint or "ru") or "")
            # A video with a stray word or two is a silent video: ASR hears «Oh»
            # in 7 s of room noise (live 16:11), and the retelling then asked
            # for a transcript. Decided here, once, for every reader -- they
            # get "" and say «(без слов)». A voice note's «Да» is real speech.
            if media == "video" and len(_re.findall(r"\w+", _re.sub(r"\[[^\]]*\]:?", " ", heard))) <= 2:
                return ""
            return heard
        except Exception:
            tg_bot.logger.exception("transcribe failed file_id=%s media=%s", file_id, media)
            return ""
        finally:
            for p in (src_file, wav):
                try:
                    if p and os.path.exists(p): os.unlink(p)
                except Exception: pass


import uuid
from pathlib import Path
import tg_bot  # noqa: E402  (cycle by design; attrs read at call time)
import tg_songs as _tg_songs  # noqa: E402
import tg_transport  # noqa: E402
import tg_video as _tg_video  # noqa: E402

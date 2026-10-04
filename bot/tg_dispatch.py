"""Update polling, routing and enqueue.

Split out of tg_bot.py. Owns the long-poll loop, _dispatch (the router that
turns a raw Telegram update into an action: callback buttons, commands, menu
labels, photos, albums, documents, voice), _handle_command, the reply-target
and incoming-photo helpers, and _resolve_and_push, which decides what kind of
task a message is and puts it on the queue.

The many module-level tables and helpers this router consults are read as
tg_bot.<name> rather than imported by value, so each keeps a single
definition and a single patch point. _IMAGE_DIR is rebound by
redirect_data_dir(), which a by-value import would not have seen.
"""
from __future__ import annotations

import os
import re
import threading
import time
import uuid
from typing import Optional

import requests


def _largest_photo(photos) -> Optional[dict]:
    """Pick Telegram's largest PhotoSize entry, tolerating a hostile shape.

    `msg["photo"]` is documented as a list of PhotoSize dicts, but nothing
    stops a malformed/adversarial update (a corrupted webhook payload, a
    buggy relay) from sending it as a string, a dict, or a list containing
    non-dict junk. `max(photos, key=lambda p: p.get(...))` on a string
    silently iterates its CHARACTERS and crashes with
    "'str' object has no attribute 'get'" the moment it calls .get() on the
    first one — an unhandled exception straight out of _dispatch. The outer
    poll loop's per-update try/except keeps this from taking the process
    down, but it still drops that update's whole turn with a stack trace
    instead of the ordinary "couldn't make sense of that" silence every
    other malformed-field path in this router already gives. Filter to only
    genuine PhotoSize-shaped entries and return None if nothing usable
    survives, exactly like `if photos:` on an absent/empty field already do.
    """
    if not isinstance(photos, list):
        return None
    usable = [p for p in photos if isinstance(p, dict) and p.get("file_id")]
    if not usable:
        return None
    return max(usable, key=lambda p: (p.get("file_size") or 0,
                                      (p.get("width") or 0) * (p.get("height") or 0)))


_TYPED_STOP = {"stop", "стоп", "хватит", "отмена", "отмени", "cancel", "остановись"}
import re as _re


def _asks_as_file(text: str) -> bool:
    import intent
    return intent.ask_yes("A user wrote to a bot that sends pictures: {text}. Is this a "
                          "request for the BOT to send its picture as a FILE (a document, "
                          "uncompressed) instead of a photo? The user sharing or talking "
                          "about their own document is No.", text)


def _fwd_author_of(msg: dict) -> str:
    """The author label of a forwarded piece: the bot's own words get the one
    label the frames trust; anyone else's name is rewritten if it claims it."""
    return tg_bot._OWN_AUTHOR if msg.get("_fwd_own") else tg_bot._fwd_author(msg)


def _is_button(text: str) -> bool:
    """Any keyboard label in any language. Built from _BTN, so a new button is covered."""
    import tg_strings
    t = text.strip()
    return any(t == v for forms in tg_strings._BTN.values() for v in forms.values())


class DispatchMixin:
    def _poll_loop(self):
        first_poll = True
        backoff = 3.0
        while self._running:
            try:
                updates = self._fetch_updates()
                self._poll_beat = time.monotonic()
                backoff = 3.0
            except Exception:
                # Network blips, DNS failures and Telegram outages are expected.
                # Back off progressively instead of hammering, but never give up:
                # the loop exiting here is the difference between "slow" and "dead".
                tg_bot.logger.exception("getUpdates failed (retry in %.0fs)", backoff)
                time.sleep(backoff)
                backoff = min(60.0, backoff * 1.7)
                continue
            # stop() landed during the long poll. The GUI builds a NEW bot on
            # Start, so this stopped instance must neither dispatch the batch
            # (its consumers are gone: the messages would sit in a dead queue
            # and never be answered) nor move the offset past it -- leaving it
            # unconfirmed is what lets the next instance receive it.
            if not self._running:
                break
            self._commit_offset(updates)

            if first_poll and updates:
                # Any updates present on the first poll were sent while the bot was offline
                catch_up = [u for u in updates
                            if (u.get("message") or u.get("channel_post") or {})
                               .get("date", self._start_time) < self._start_time]
                if catch_up:
                    summary = f"[CATCH-UP] {len(catch_up)} message(s) received while offline"
                    tg_bot.logger.info(summary)
                    self._activity.log(0, "system", summary)
                    for u in catch_up:
                        msg = u.get("message") or u.get("channel_post") or {}
                        cid  = (msg.get("chat") or {}).get("id", 0)
                        uname = (msg.get("from") or {}).get("username", "")
                        text  = msg.get("text") or msg.get("caption") or "[media]"
                        self._activity.log(cid, "offline_msg",
                                           f"[OFFLINE] {text[:300]}", uname)
            first_poll = False

            for upd in updates:
                try: self._dispatch_logged(upd)
                except Exception: tg_bot.logger.exception("dispatch error")

    def _get_updates(self) -> list:
        result = self._fetch_updates()
        self._commit_offset(result)
        return result

    def _commit_offset(self, result: list) -> None:
        if result:
            self._offset = result[-1]["update_id"] + 1
            self._save_offset(self._offset)

    def _fetch_updates(self) -> list:
        r = requests.get(f"{self._api}/getUpdates",
                         params={"offset": self._offset, "timeout": tg_bot._POLL_TIMEOUT},
                         timeout=tg_bot._POLL_TIMEOUT + 5)
        body = r.json()
        if not body.get("ok", False):
            # 409 (a second instance polling) / 401 (revoked token) come back at
            # once: returning [] here spun a tight request loop and still
            # counted as a healthy poll for the watchdog. Raise -> backoff.
            raise RuntimeError(f"getUpdates not ok: {body.get('error_code')} "
                               f"{str(body.get('description'))[:120]}")
        return body.get("result", [])

    def _dispatch_logged(self, upd: dict):
        """_dispatch inside this chat's full transcript (chatlog.py)."""
        import chatlog
        cb = upd.get("callback_query") or {}
        msg = upd.get("message") or upd.get("edited_message") or cb.get("message") or {}
        chat_id = (msg.get("chat") or {}).get("id")
        _rs = str(getattr(self._get_session(chat_id), "reg_state", "") or "") if chat_id else ""
        # Never on a login/password/name answer: it would change the secret AND echo it back.
        # Nor on a keyboard button: «🎵 Songs» has one vowel in five and became «🎵 Ыщтпы».
        if isinstance(msg.get("text"), str) and not cb and not any(
                k in _rs for k in ("password", "login", "name")) and not _is_button(msg["text"]):
            import keyboard_layout            # «ghbdtn» -> «привет», before anything reads it
            fixed = keyboard_layout.fix(msg["text"])
            if fixed != msg["text"]:
                tg_bot.logger.info("layout fix chat=%s: %r -> %r", chat_id, msg["text"][:80], fixed[:80])
                import html as _h
                self._send_text(chat_id, tg_bot._t("layout_fixed", self._lang(self._get_session(chat_id)),
                                                   text=_h.escape(fixed)), parse_mode="HTML")
                msg["text"] = fixed
        if chat_id and msg.get("date") and not cb:
            self.__dict__.setdefault("_msg_date", {})[chat_id] = msg["date"]
            self.__dict__.setdefault("_msg_id", {})[chat_id] = msg.get("message_id", 0)
            # Menu hopping piled «⚙️ Настройки» six times over: a button press wipes the
            # previous press and the menu reply it got; a typed message closes the tail.
            tails = self.__dict__.setdefault("_menu_tail", {})
            if isinstance(msg.get("text"), str) and _is_button(msg["text"]) \
                    and not tg_bot._is_forwarded(msg, self_is_own=False):
                for _mid in (tails.get(chat_id) or {}).get("ids", []):
                    self._delete(chat_id, _mid)
                tails[chat_id] = {"ids": [msg.get("message_id")], "open": True}
            elif chat_id in tails:
                tails[chat_id]["open"] = False
        # Every update is a turn: whatever it sets off -- here or in a thread
        # spawned for it -- lands in one trace. A message that only joins the
        # queue leaves nothing here (the task traces itself when it runs).
        import turn_trace
        _trace = turn_trace.start(chat=chat_id, task="button" if cb else "message",
                                  text=(cb.get("data") if cb else
                                        ("[secret]" if msg.get("_secret") else msg.get("text") or msg.get("caption")
                                         or next((k for k in ("voice", "video_note", "video", "photo", "document",
                                                              "audio") if k in msg), "")) or "")[:1500])
        with chatlog.bind(chat_id):
            try:
                self._dispatch(upd)
            finally:
                # _secret is set inside _dispatch, after start() took the text:
                # a password turn that logged a warning was written in the clear.
                turn_trace.finish(_trace, keep_idle=False,
                                  **({"text": "[secret]"} if msg.get("_secret") else {}))
                # After, so a password _scrub_secret flagged is never written.
                if cb:
                    chatlog.write(chat_id, "in", {"button": cb.get("data"),
                                                  "from": (cb.get("from") or {}).get("id")})
                elif msg.get("_secret"):
                    chatlog.write(chat_id, "in", {"text": "[password, not logged]"})
                else:
                    chatlog.write(chat_id, "in", {k: v for k, v in msg.items()
                                                  if k not in ("chat", "from")})

    def _dispatch(self, upd: dict):
        # An inline-keyboard press is a different update SHAPE, not a
        # variation on a message: no text, no sender gate, its own
        # answerCallbackQuery handshake and its own 20-family routing table.
        # It lives in tg_callbacks.CallbackMixin; this is the whole seam.
        cb = upd.get("callback_query")
        if cb:
            self._dispatch_callback(cb)
            return

        msg = upd.get("message") or upd.get("edited_message") or {}
        chat_id = (msg.get("chat") or {}).get("id")
        if not chat_id: return

        # ── private chats only ────────────────────────────────────────────────
        # Nothing checked the chat type: added to a group the bot would run the
        # registration flow in it, ask for a password in public, and map the whole
        # group to a single account.
        chat_type = (msg.get("chat") or {}).get("type", "private")
        if chat_type not in ("private", ""):
            if not (msg.get("new_chat_members") or msg.get("left_chat_member")):
                sess = self._get_session(chat_id)
                self._send_text(chat_id, tg_bot._t("group_chat", self._lang(sess)))
            return

        # Adopt the client's language on first contact, so a Russian-speaking user
        # is not greeted by an English keyboard they never asked for.
        sess0 = self._get_session(chat_id)
        if not sess0.lang:
            sess0.lang = tg_bot._norm_lang((msg.get("from") or {}).get("language_code", ""))
            try: self._store.put(sess0)
            except Exception: pass
        # The client's language is a guess: a user whose Telegram is English but
        # who WRITES Russian got every menu and status in English (live
        # 2026-09-27). Until they pick a language themselves,
        # the words they type decide.
        _t0 = msg.get("text") or msg.get("caption") or ""
        if (not sess0.lang_chosen and sess0.lang != "ru" and not tg_bot._is_forwarded(msg)
                and len(re.findall(r"[А-Яа-яЁё]{2,}", _t0)) >= 2):
            sess0.lang = "ru"
            try: self._store.put(sess0)
            except Exception: pass

        text = (msg.get("text") or "").strip()
        # A forwarded button label is not something to act on. Live 2026-09-27:
        # a stray multi-forward replayed «🔍 Найти товар» as a button press and
        # fed the bot's own replies into a running song as wishes. BEFORE the
        # gate: the gate answers pending prompts (song topic, city…) and took a
        # forwarded storyboard as the topic.
        # The bot's own REPLIES are material (live 10-03: users could not forward
        # its reasoning back): they go on, labelled as its own words only when
        # Telegram names this bot as the sender (_fwd_is_own); a forwarded batch
        # never steers a running task (tg_resolve). Another bot's message is
        # material like a person's: a song forwarded from a music bot was
        # dropped here while 🎤 Cover waited for it (live 10-03).
        _fwd = tg_bot._is_forwarded(msg, self_is_own=False)
        if _fwd and tg_bot._fwd_is_own(msg, self.token.split(":", 1)[0]):
            msg["_fwd_own"] = True
        if _fwd and text and _is_button(text):
            notes = self.__dict__.setdefault("_fwd_skip_note", {})   # one note per burst
            if time.time() - notes.get(chat_id, 0) > 30:
                notes[chat_id] = time.time()
                sess = self._get_session(chat_id)
                self._send_text(chat_id, tg_bot._t("fwd_own_skipped", self._lang(sess)))
            return
        # ── user registration gate ────────────────────────────────────────────
        if not self._user_gate(chat_id, msg):
            return

        # Stop is handled HERE, before the debounce queue — routing it through the
        # normal path would make it queue up behind the very work it is meant to
        # cancel, so the user would tap Stop and watch nothing happen.
        # A bare typed «Stop» too: live 2026-09-27 it went into the running redraw
        # as a steer note («👌 Noted: Stop») and the render carried on.
        if tg_bot._LABEL2KEY.get(text) == "stop" or text.lower().strip(" .!") in _TYPED_STOP:
            sess = self._get_session(chat_id)
            lang = self._lang(sess)
            self._stop_and_report(chat_id, sess, lang)
            return
        # «пришли файлом» / "as a file": the model has no way to send one and
        # answered with right-click advice five times (persona run 2026-09-27).
        # a keyboard label is never that request: «📚 Документы» was read as one and
        # sent the last picture as a file instead of opening the menu (live 10-03)
        if len(text) <= 60 and not _is_button(text) and _asks_as_file(text):
            sess = self._get_session(chat_id)
            rep = (msg.get("reply_to_message") or {}).get("message_id")
            e = tg_bot._image_by_msg(sess, rep) if rep else None
            if not e:                  # no reply: only a picture from the last hour
                last = (getattr(sess, "image_log", None) or [None])[-1]
                e = last if last and time.time() - float(last.get("ts") or 0) < 3600 else None
            if e and os.path.exists(e.get("path", "")) and self._send_document(chat_id, e["path"]):
                return
            # A reply to the user's OWN photo: it is not in image_log under that message id.
            _ph = (msg.get("reply_to_message") or {}).get("photo") or []
            _data = self._dl_bytes(_ph[-1]["file_id"]) if _ph else None
            if _data:
                tg_bot._IMAGE_DIR.mkdir(parents=True, exist_ok=True)
                _p = tg_bot._IMAGE_DIR / f"tg_{int(time.time() * 1000)}.jpg"
                _p.write_bytes(_data)
                if self._send_document(chat_id, str(_p)):
                    return
        # ── "this one" ────────────────────────────────────────────────────────
        # A Telegram REPLY is the user pointing at something. Nothing read it
        # before, so "describe this picture" sent as a reply to a specific image
        # was resolved against the single "current image" slot and could answer
        # about a completely different one. Resolve it here, before the message
        # joins the debounce queue, because the reply link lives on the raw
        # update and is gone by then.
        # The resolved id travels WITH the item (into task.image_id), not just
        # onto sess.target_image — see _resolve_reply_target's docstring for
        # why a lazy re-read at task-execution time is racy.
        _reply_image_id = self._resolve_reply_target(chat_id, msg)
        if text:
            text = (msg.get("text") or "").strip()       # a forwarded reply gains «in reply to»

        # Only a real command token: "/etc/nginx/nginx.conf почему не
        # работает?" is a question, and was answered "unknown command".
        if _re.match(r"/[A-Za-z0-9_]{1,32}(?:@\w+)?(?:\s|$)", text):
            self._handle_command(chat_id, text)
            return
        if text:
            # A FORWARDED text is material the user wants something done with,
            # not the user talking to the bot. Fed in bare, a forwarded
            # channel post ("Внимание! В субботу перекрывают движение...") was
            # answered as if the user had asked it -- with document search on,
            # "the documents say nothing about a road closure" -- and the
            # follow-up "о чём это?" then had nothing to refer to (live
            # 2026-09-13, mega journey). The flag travels with the item; the
            # resolver frames the text as a quotation.
            self._enqueue_item(chat_id,
                               {"type": "text", "text": text, "image_id": _reply_image_id,
                                "quote": msg.get("_quote", ""),
                                "forwarded": tg_bot._is_forwarded(msg),
                                "own": bool(msg.get("_fwd_own")),
                                "author": _fwd_author_of(msg)})
            return

        # ── a mashup is collecting its two tracks ─────────────────────────────
        # Checked BEFORE the normal audio routing, because every branch below
        # sends audio somewhere else: a voice note becomes speech to the bot, a
        # forwarded one raises the transcript prompt, and an mp3 lands in the
        # document reader. While 🎚 Mashup is waiting, an audio message is a
        # track and nothing else. Non-audio messages fall through untouched so
        # the menu buttons still work and the user is never trapped here.
        _msess = self._get_session(chat_id)
        # 🎙 Animate is collecting voice samples: a clip with sound is one more voice.
        if getattr(_msess, "anim_voice_state", "") and self._anim_voice_take_media(
                chat_id, _msess, self._lang(_msess), msg):
            return
        # 📚 Audiobook armed: the voice sample and then the book are ours (never the library / sandbox / agent).
        if getattr(_msess, "book_state", "") and self._book_take_media(
                chat_id, _msess, self._lang(_msess), msg):
            return
        # ▶️ Continue armed: a video is the clip to go on from.
        if getattr(_msess, "continue_state", "") and self._continue_take_media(
                chat_id, _msess, self._lang(_msess), msg):
            return
        # 🎨 Restyle armed: a video is the clip to redraw (before cover/clone: both eat videos).
        if getattr(_msess, "restyle_state", "") and self._restyle_take_media(
                chat_id, _msess, self._lang(_msess), msg):
            return
        # 🎤 Cover armed: any clip with sound is the song to re-sing.
        if getattr(_msess, "cover_state", "") and self._cover_take_media(
                chat_id, _msess, self._lang(_msess), msg):
            return
        # 🎙 Clone voice armed: any clip with sound is the voice sample.
        if getattr(_msess, "clone_state", "") and self._clone_take_media(
                chat_id, _msess, self._lang(_msess), msg):
            return
        if getattr(_msess, "mashup_state", ""):
            _mlang = self._lang(_msess)
            _aud = msg.get("audio") or msg.get("voice")
            if _aud:
                self._mashup_take_audio(
                    chat_id, _msess, _mlang, _aud["file_id"],
                    is_voice=bool(msg.get("voice")),
                    suffix=".ogg" if msg.get("voice") else ".mp3")
                return
            _doc = msg.get("document")
            if _doc and str(_doc.get("mime_type", "")).startswith("audio"):
                _name = _doc.get("file_name", "track")
                _ext = os.path.splitext(_name)[1] or ".mp3"
                self._mashup_take_audio(chat_id, _msess, _mlang,
                                        _doc["file_id"], is_voice=False,
                                        suffix=_ext)
                return

        voice = msg.get("voice") or msg.get("audio")
        if voice:
            # A FORWARDED voice note is not the user talking to the bot — it is
            # material they want something done to. Treating it as speech-to-the-bot
            # fed someone else's monologue in as an instruction. Transcribe it, then
            # ask what to do with it.
            if tg_bot._is_forwarded(msg, self_is_own=False):
                self._enqueue_item(chat_id, {
                    "type": "fwd_voice",
                    "file_id": voice["file_id"],
                    "author": _fwd_author_of(msg), "own": bool(msg.get("_fwd_own")),
                    "seconds": int(voice.get("duration") or 0),
                    "caption": (msg.get("caption") or "").strip()})
                return
            self._enqueue_item(chat_id, {"type": "voice",
                "file_id": voice["file_id"],
                "caption": (msg.get("caption") or "").strip(),
                "image_id": _reply_image_id})
            return

        # A round "video message" (Telegram's video_note) carries an audio
        # track exactly like a voice note — same two scenarios apply: sent
        # directly, it is the user talking TO the bot (transcribe, treat as
        # spoken text); forwarded, it is material someone wants something DONE
        # WITH (transcribe, then ask transcript/summary/both), never fed in as
        # an instruction. `media="video"` only changes which ffmpeg suffix
        # _transcribe uses to extract the audio track — everything downstream
        # (the ask keyboard, fwd_transcript, _fwdv_intent) is unchanged.
        # An ordinary video (not the round note) rides the same path: it is
        # looked at (frames every 5 s) and listened to. Before 2026-09-14 it
        # was not handled at all.
        video_note = msg.get("video_note") or msg.get("video")
        if video_note:
            if tg_bot._is_forwarded(msg, self_is_own=False):
                self._enqueue_item(chat_id, {
                    "type": "fwd_voice", "media": "video",
                    "file_id": video_note["file_id"],
                    "author": _fwd_author_of(msg), "own": bool(msg.get("_fwd_own")),
                    "seconds": int(video_note.get("duration") or 0),
                    "caption": (msg.get("caption") or "").strip()})
                return
            self._enqueue_item(chat_id, {"type": "voice", "media": "video",
                "file_id": video_note["file_id"],
                "seconds": int(video_note.get("duration") or 0),
                "caption": (msg.get("caption") or "").strip()})
            return

        _doc = msg.get("document") or {}
        if not video_note and str(_doc.get("mime_type") or "").startswith("video/"):
            # A video sent as a file (the way a long recording usually arrives).
            # Duration is unknown on the wire: treat it as long -- the offer
            # shows what will happen and the user decides.
            self._enqueue_item(chat_id, {"type": "fwd_voice" if tg_bot._is_forwarded(msg, self_is_own=False) else "voice",
                "media": "video", "file_id": _doc["file_id"],
                # a GIF also arrives as a video/mp4 document, with its real length
                # on `animation` (live 10-03: a 3 s GIF was offered as «16667 мин»)
                "seconds": int((msg.get("animation") or {}).get("duration") or 10 ** 6),
                "caption": (msg.get("caption") or "").strip()})
            return

        photos = msg.get("photo")
        if photos:
            # file_size is OPTIONAL in Telegram's PhotoSize. When it is missing the
            # old key was 0 for every entry and max() kept the FIRST — the smallest
            # thumbnail — so the vision model silently got a postage stamp. Fall
            # back to pixel area, which is always present.
            largest = _largest_photo(photos)
            if largest is None:
                return
            caption = (msg.get("caption") or "").strip()
            group   = msg.get("media_group_id")
            if group:
                self._buffer_album(chat_id, group, largest["file_id"], caption)
            else:
                self._enqueue_item(chat_id, {"type": "photo",
                    "file_id": largest["file_id"], "caption": caption})
            return

        doc = msg.get("document")
        if doc:
            self._enqueue_item(chat_id, {"type": "document",
                "file_id": doc["file_id"],
                "filename": doc.get("file_name", "file"),
                "file_size": int(doc.get("file_size") or 0),
                "caption": (msg.get("caption") or "").strip()})
            return

        sticker = msg.get("sticker")
        if sticker and sticker.get("file_id"):
            self._enqueue_item(chat_id, {"type": "photo",
                "file_id": sticker["file_id"], "caption": ""})

    def _resolve_reply_target(self, chat_id: int, msg: dict) -> str:
        """Turn a reply into an explicit target for the turn that follows.

        Three cases, in order of certainty:
          1. the replied message IS a photo the bot sent  -> that exact file;
          2. the replied message is a photo the USER sent -> download it again
             (Telegram keeps the file, we may not have that upload on disk);
          3. the replied message is text                  -> quote it, so "what did
             you mean by that" refers to the quoted line and not to the last one.

        Silent on anything it cannot resolve: a reply that resolves to nothing must
        leave the turn exactly as it would have been, never half-targeted.

        Returns the resolved image id ("" if this reply named no picture). The
        caller threads this into the enqueued item's "image_id", the same
        capture-at-commit-time carry the image-verb buttons already use
        (see the "captured here, at PRESS time" note in _dispatch above) —
        `sess.target_image` alone is a SINGLE shared slot, and a task built
        from this reply may sit queued for a while (behind an already-running
        task in this chat) before it actually executes. A second reply to a
        DIFFERENT picture arriving in that window overwrites the slot before
        the first task ever reads it, so relying on a live re-read of
        sess.target_image at execution time silently answered about whichever
        picture was replied to LAST, not the one THIS turn was actually about
        — reproduced live: reply to picture A, then (before A's task runs)
        reply to picture B, and A's task answered about B.
        """
        reply = msg.get("reply_to_message") or {}
        if not reply:
            return ""
        sess = self._get_session(chat_id)
        try:
            entry = tg_bot._image_by_msg(sess, reply.get("message_id"))
            if entry and os.path.exists(entry.get("path", "")):
                sess.target_image = entry["id"]
                self._store.put(sess)
                self._activity.log(chat_id, "system",
                                   f"[reply] targets image {entry['id']}")
                return entry["id"]

            photos = reply.get("photo")
            if photos:
                largest = _largest_photo(photos)
                if largest is None:
                    return ""
                path = self._save_incoming_photo(largest["file_id"])
                if path:
                    img_id = tg_bot._log_image(sess, path,
                                        msg_id=int(reply.get("message_id") or 0),
                                        label=(reply.get("caption") or "")[:80],
                                        src="user")
                    sess.target_image = img_id
                    self._store.put(sess)
                    self._activity.log(chat_id, "system",
                                       f"[reply] targets re-fetched image {img_id}")
                    return img_id
                return ""

            quoted = (reply.get("text") or reply.get("caption") or "").strip()
            if quoted and msg.get("text") and tg_bot._is_forwarded(msg, self_is_own=False):
                # A reply INSIDE forwarded material is its author's, not the user's,
                # and it sits inside the forward's own quote frame: a nested frame
                # was labelled «the user is replying» and its close ended the outer
                # frame early (live 10-01: Stepan's «подтверди» quoting «можно любой
                # положить» was read as the user's words -> an Ozon cart answer).
                one_line = " ".join(quoted.split())[:300]
                msg["text"] = f"(in reply to: «{one_line}») " + msg["text"]
            elif quoted and msg.get("text"):
                # Kept BESIDE the user's words, never glued into them: every
                # check of what the user asked (song? weather? a button? a
                # name?) reads the words alone, and the quote joins the request
                # in ONE place (tg_resolve._as_request). Glued in, a reply to
                # «классная песня» was an order for a song.
                msg["_quote"] = quoted[:1200]
            elif quoted and not tg_bot._is_forwarded(msg, self_is_own=False):
                sess.quoted_text = quoted[:1200]
                self._store.put(sess)
            return ""
        except Exception:
            tg_bot.logger.exception("resolving the reply target failed chat=%s", chat_id)
            return ""

    def _save_incoming_photo(self, file_id: str) -> str:
        """Download a photo to a real file so it can be referred to again later.
        Returns "" on any failure — the caller then behaves as if there was no
        reply, which is the pre-existing behaviour, not a new failure mode."""
        try:
            data = self._dl_bytes(file_id)
            if not data:
                return ""
            tg_bot._IMAGE_DIR.mkdir(parents=True, exist_ok=True)
            path = tg_bot._IMAGE_DIR / f"tg_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}.jpg"
            path.write_bytes(data)
            return str(path)
        except Exception:
            tg_bot.logger.exception("could not save an incoming photo")
            return ""

    def _buffer_album(self, chat_id, group_id, file_id, caption):
        # A QUIET period, not a fixed window: a forwarded album of 10 pages arrives over several seconds, and a
        # timer started by the first page cut it into pieces answered one by one (live 2026-10-05).
        with self._album_lock:
            e = self._albums.get(group_id)
            if e is None:
                e = self._albums[group_id] = {"chat_id": chat_id, "photos": [], "timer": None, "t0": time.monotonic()}
            elif e["timer"] is not None:
                e["timer"].cancel()
            e["photos"].append({"file_id": file_id, "caption": caption})
            wait = max(0.05, min(tg_bot._ALBUM_COLLECT_S, e["t0"] + tg_bot._ALBUM_MAX_S - time.monotonic()))
            t = threading.Timer(wait, self._flush_album, (group_id,))
            t.daemon = True; t.start(); e["timer"] = t

    def _flush_album(self, group_id):
        with self._album_lock: entry = self._albums.pop(group_id, None)
        if not entry: return
        cap = next((p["caption"] for p in entry["photos"] if p["caption"]), "")
        self._enqueue_item(entry["chat_id"], {
            "type": "album",
            "file_ids": [p["file_id"] for p in entry["photos"]],
            "caption": cap})


import tg_bot  # noqa: E402  (cycle by design; attrs read at call time)

# Re-exports: _handle_command and _resolve_and_push moved to tg_commands /
# tg_resolve. Anything that reached them as tg_dispatch.<Mixin> keeps resolving.
import tg_commands as _tg_commands  # noqa: E402
import tg_resolve as _tg_resolve  # noqa: E402

CommandsMixin = _tg_commands.CommandsMixin
ResolveMixin = _tg_resolve.ResolveMixin

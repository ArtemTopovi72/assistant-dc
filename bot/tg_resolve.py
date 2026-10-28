"""Batch resolution: deciding what kind of task an incoming batch is.

_resolve_and_push is the single decision point
between "the debounce timer fired with these messages" and "a task of some kind
went on the queue" -- menu labels, pending prefixes, photos, documents, voice,
forwarded voice and the plain-text fallback all resolve here.

Same rule as the rest of the family: tg_bot attributes are read through the
module at call time, never imported by value. _IMAGE_DIR in particular is
rebound by redirect_data_dir(), which a by-value import would never see."""
from __future__ import annotations

from seams import vision_model, module
import html as _html_mod
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional


# Telegram's cloud Bot API serves files up to 20 MB; a self-hosted
# telegram-bot-api (config.TG_API_BASE) goes up to 2 GB.
def _bot_file_limit(local=None) -> int:
    if local is None:
        import config as _config
        local = getattr(_config, "TG_API_LOCAL", False)
    return (2000 if local else 20) * 1024 * 1024


def _save_image_bytes(data: bytes) -> str:
    """Write downloaded image bytes to a fresh file in tg_bot._IMAGE_DIR; its path."""
    tg_bot._IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    path = str(tg_bot._IMAGE_DIR / f"tg_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}.jpg")
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def _batch_reply_id(batch) -> int:
    """Reply to the user's request, not the last forward or a later navigation press."""
    content = [it for it in batch if it.get('type') != 'text'
               or (it.get('text', '').strip() and it.get('text', '').strip() not in tg_bot._LABEL2KEY)]
    own = [it for it in content if not it.get('forwarded')
           and (it.get('text', '').strip() or it.get('caption', '').strip())]
    return next((int(it.get('_message_id') or 0) for it in reversed(own or content)
                 if '_message_id' in it), 0)


class ResolveMixin:
    @staticmethod
    def _last_bot_text(sess) -> str:
        """The bot's most recent reply in this chat, or ""."""
        for m in reversed(getattr(sess, "history", None) or []):
            if isinstance(m, dict) and m.get("role") == "assistant" and isinstance(m.get("content"), str):
                return m["content"]
        return ""

    @staticmethod
    def _as_request(item: dict, text: str) -> str:
        """The ONE place a replied-to message joins the user's words. Everything
        that decides what the user wants reads `raw` (the words alone); only
        what goes on as the request is built here."""
        q = (item or {}).get("quote") or ""
        if not q:
            return text
        from prompt_guard import wrap_quoted
        return wrap_quoted("the user is replying to this message", q) + "\n" + text

    def _latest_sandbox_file(self, chat_id, sess, within: float = 3600) -> str:
        """The newest file in this chat's working folder when it came after the
        last picture and within the hour; "" otherwise."""
        try:
            import sandbox_access as _sa
            if not _sa.may_use_files(self._user_store.get(chat_id)):
                return ""
            import code_sandbox as _cs
            box = _cs.sandbox_for(chat_id)
            files = [p for p in Path(box.root).iterdir() if p.is_file()]
            if not files:
                return ""
            f = max(files, key=lambda p: p.stat().st_mtime)
            t = f.stat().st_mtime
            img = sess.last_image_path
            if time.time() - t > within or (img and os.path.exists(img)
                                            and os.path.getmtime(img) > t):
                return ""
            return box.relative(f)
        except Exception:
            return ""

    def _read_links(self, ctx, sess, chat_id, user, text: str, scaffold: list,
                    announce: bool = True) -> bool:
        """Attach what a link in `text` holds: a video is downloaded and watched
        (ears + eyes), any other page is read (tg_links). True: a long video was
        offered as the portioned job, which answers the message."""
        generation = sess.context_generation
        _vu = _tg_links.video_url(text)
        if announce and _vu and ctx is not None and _vu not in _tg_links.WATCHED:
            # download + ears + eyes run before the task and its status line: a forwarded
            # link sat 49 s with no word (live 2026-10-17)
            self._send_text(chat_id, tg_bot._t("fwd_video_work", self._lang(sess)))
        _vid = self.links.fetch_video(_vu) if _vu and ctx is not None else {}
        if generation != sess.context_generation:
            return True
        if _vid.get("data"):
            # A video link is watched like a sent video: ears + eyes.
            self._api_post("sendChatAction", {"chat_id": chat_id, "action": "typing"})
            _heard = self._transcribe(ctx, _vu, "video", lang_hint=(sess.lang or "ru"),
                                      data=_vid["data"], speakers=True)   # who said what
            _seen = self._look_video(ctx, _vu, data=_vid["data"], lang=(sess.lang or "ru"),
                                     transcript=_heard or "")
            if generation != sess.context_generation:
                return True
            scaffold.append(f"[The user's message links a {_vid['seconds']}-second video "
                            f"«{_vid['title']}» ({_vu}). It was downloaded and watched "
                            f"for you; answer from it, never say you cannot watch links. "
                            f"Speech in it: {_heard or '(none)'}"
                            + (" «Спикер N:» / «Speaker N:» marks DIFFERENT people talking; "
                               "keep them apart." if "Спикер 2" in (_heard or "") or "Speaker 2" in (_heard or "") else "")
                            + (f" The uploader's description under the video: "
                               f"{_vid['description']}" if _vid.get("description") else "")
                            + "]")
            if _seen.get("description"):
                scaffold.append(tg_bot._video_seen_note(_seen))
            _tg_links.WATCHED[_vu] = "\n".join(scaffold[-2:] if _seen.get("description")
                                               else scaffold[-1:])
            sess.last_link = _vu
            self._activity.log(chat_id, "system", f"[video link] watched {_vu[:80]}",
                               (user.name if user else str(chat_id)))
        elif _vid.get("too_long"):
            # the same portioned job as a long video FILE, not a "send me the file" brush-off
            sess.last_link = _vu
            self._long_video_offer(chat_id, sess, self._lang(sess),
                                   {"url": _vu, "seconds": _vid["seconds"], "caption": text})
            return True
        elif _vu and ctx is not None:
            # YouTube's «confirm you're not a bot» left the page shell («- YouTube») as the
            # whole material, and the agent searched the web for «summary of video <id>» and
            # retold junk (2026-10-17).
            scaffold.append(f"[The video at {_vu} could not be downloaded (the site refused it "
                            "or it is private): it was NOT watched or heard. Tell the user plainly "
                            "and offer to try again later or to send the file; do not look up "
                            "what it says elsewhere.]")
            sess.last_link = _vu
        else:
            try:
                _lc = _tg_links.link_context(text)
                if generation != sess.context_generation:
                    return True
                sess.last_link = _tg_links.urls_in(text)[0] if _lc else None
            except Exception:
                tg_bot.logger.exception("link reading failed")
                _lc = ""
            if _lc:
                scaffold.append(_lc)
                self._activity.log(chat_id, "system",
                                   f"[link] read {sess.last_link[:80]}",
                                   (user.name if user else str(chat_id)))

    _FWD_JOIN_S = 120.0

    def _join_waiting(self, chat_id: int, task) -> bool:
        """Fold this chat's newest still-waiting task into `task` (its text first).

        Only a task nobody has started (drop_task succeeds), queued in the last
        two minutes, and never two pictures into one turn. True when folded."""
        try:
            waiting = [t for t in self._backend.service_order() if t.chat_id == chat_id]
        except Exception:
            return False
        if not waiting:
            return False
        prev = waiting[-1]
        if time.time() - (prev.enqueue_ts or 0) > self._FWD_JOIN_S or prev.style_ref_path:
            return False
        if prev.image_path and task.image_path:
            return False
        if not self._backend.drop_task(prev.task_id):
            return False            # a worker took it meanwhile
        task.replaces_task_id = prev.task_id
        task.incoming_ids.extend(prev.incoming_ids)
        task.user_text = prev.user_text + "\n\n" + task.user_text
        fresh_attachment = bool(task.image_path)
        for f in ("image_path", "image_id", "image_is_photo"):
            if f == 'image_id' and fresh_attachment:
                continue
            if not getattr(task, f):
                setattr(task, f, getattr(prev, f))
        task.enqueue_ts = prev.enqueue_ts
        task.reply_to = prev.reply_to or task.reply_to
        tg_bot.logger.info("Forward joined waiting task %s -> %s chat=%s",
                           prev.task_id[:8], task.task_id[:8], chat_id)
        return True

    def _bind_turn(self, sess, raw: str, batch: list) -> None:
        """Bind the message's words in intent on arrival, with the context the graph would
        give them, so the reads before the graph and the graph's own are one model call."""
        try:
            import types
            import intent
            from graph_personality import _turn_inputs
            state = {"messages": sess.get_history(),
                     "image_data": any(it.get("type") in ("photo", "album") for it in batch)}
            own_ctx = types.SimpleNamespace(last_image_path=getattr(sess, "last_image_path", None))
            words, previous, attached = _turn_inputs(own_ctx, state, raw)
            intent.bind_turn([words or raw, raw], previous, attached, entry=True)
        except Exception:
            tg_bot.logger.debug("binding the turn's reads failed", exc_info=True)

    def _resolve_and_push(self, chat_id: int, batch: list):
        from tg_incoming import BATCH_IDS
        ids = tuple(it["_inbox_id"] for it in batch if it.get("_inbox_id"))
        token = BATCH_IDS.set(ids)
        try:
            result = self._resolve_items(chat_id, batch)
            if not getattr(self, "_stopping", False):
                with self._task_lock:
                    held = {key for keys in self._steer_incoming.values() for key in keys}
                self._incoming.ack(key for key in ids if key not in held)
            return result
        finally:
            BATCH_IDS.reset(token)
            with self._mgmt_lock:
                self._incoming_enqueued.difference_update(ids)

    def _resolve_items(self, chat_id: int, batch: list):
        # Earliest arrival time across the batch, stamped at _enqueue_item —
        # BEFORE the debounce delay — so a Stop that landed while this batch
        # was still debouncing is not blind to it (see _enqueue_item comment).
        # Only what arrived BEFORE a Stop is stale. «привет» sent right after
        # Stop joined the cancelled essay's batch and was dropped with it --
        # the chat would go silent.
        with self._stop_lock:
            _stop_ts0 = self._stop_requests.get(chat_id, 0.0)
        if _stop_ts0:
            _fresh = [it for it in batch if it.get("_arrival_ts", time.time()) > _stop_ts0]
            if _fresh and len(_fresh) < len(batch):
                tg_bot.logger.info("Dropping %d pre-Stop item(s) of a batch chat=%s",
                                   len(batch) - len(_fresh), chat_id)
                batch = _fresh
        _arrival_ts = min((it.get("_arrival_ts", time.time()) for it in batch),
                          default=time.time())
        sess = self._get_session(chat_id)
        batch = [it for it in batch if it.get('_context_generation', sess.context_generation)
                 == sess.context_generation]
        if not batch:
            return
        _generation = sess.context_generation
        _reply_to = _batch_reply_id(batch)
        if sess.expire_photo_waits():
            self._store.put(sess)
        user = self._user_store.get(chat_id)
        ctx  = self._get_ctx()
        lang = self._lang(sess)

        # 🎭 Style (tg_callbacks._cb_style_image) armed pending_style_target
        # earlier: the very next photo/album in this chat IS the style
        # reference, not an ordinary upload. Short-circuits straight to a
        # transfer_image task and skips the rest of this function -- the
        # normal caption/vision flow has no use for "here is my reference
        # photo". A batch with no image in it (the user typed something else
        # instead) falls through and is handled normally, same as every other
        # "ask instead of guessing" flow in this file.
        if sess.pending_style_target:
            _ref_item = next((it for it in batch if it.get("type") in ("photo", "album")), None)
            if _ref_item is not None:
                self._push_style_transfer_task(chat_id, sess, lang, _ref_item, _arrival_ts)
                return

        # 👗 armed pending_outfit_target: a photo now is the CLOTHING reference
        # for the picture 👗 was pressed under, its caption (if any) the
        # wishes. Text alone goes on through pending_prefix as before.
        if sess.pending_outfit_target:
            _ref_item = next((it for it in batch if it.get("type") in ("photo", "album")), None)
            if _ref_item is not None:
                _wish = " ".join((it.get("caption") or "").strip() for it in batch).strip()
                self._push_style_transfer_task(chat_id, sess, lang, _ref_item, _arrival_ts,
                                               outfit_wish=_wish)
                return

        # 🔁 armed and a person picked: the photo is WHO goes in their place.
        if sess.pending_swap_target and sess.swap_where:
            _ref_item = next((it for it in batch if it.get("type") in ("photo", "album")), None)
            if _ref_item is not None:
                _wish = " ".join((it.get("caption") or "").strip() for it in batch).strip()
                self._run_swap(chat_id, sess, lang, _ref_item, _wish)
                return

        # 🎞 Animate photo (__animate_photo__ direct action) armed
        # awaiting_animate_photo: unlike style, this button lives in the
        # Creativity menu with no picture already in view, so it asks for
        # the photo FIRST and only then offers presets/custom motion --
        # registering the photo here and showing that keyboard, rather than
        # pushing a task immediately the way the style reference-photo flow
        # does. A batch with no image falls through unchanged.
        if sess.cast_state.startswith("pic:"):
            _cast_item = next((it for it in batch if it.get("type") in ("photo", "album")), None)
            if _cast_item is not None:
                self._cast_take_photos(chat_id, sess, lang, _cast_item)
                return

        if sess.awaiting_animate_photo:
            _anim_item = next((it for it in batch if it.get("type") in ("photo", "album")), None)
            if _anim_item is not None:
                self._register_animate_photo(chat_id, sess, lang, _anim_item)
                return

        # 🎭 Change style (__style_photo__, item 7) armed awaiting_style_photo
        # the same way: no picture in view yet, so ask for the photo first,
        # then reuse the EXISTING style preset/custom keyboard and callbacks
        # (_cb_style_preset/_cb_style_custom) by arming pending_style_target
        # exactly as _cb_style_image does once a picture is already current.
        if sess.awaiting_style_photo:
            _style_item = next((it for it in batch if it.get("type") in ("photo", "album")), None)
            if _style_item is not None:
                self._register_style_photo(chat_id, sess, lang, _style_item)
                return

        # A held "which picture?" instruction is a ONE-TURN thing — armed only
        # to be resurrected by pressing pick: on the exact prompt it came from
        # (tg_dispatch.py's pick: handler already clears it before re-pushing,
        # so this is a no-op on that path). Any OTHER batch reaching here means
        # the user did something else instead of answering. Telegram buttons
        # never expire, so without this a tap on that old "which picture?"
        # prompt would silently resurrect and run the forgotten instruction.
        # Cleared unconditionally here; the "ask instead of guessing" block
        # below re-arms it fresh if THIS batch also turns out to need asking.
        if sess.pending_instruction:
            sess.pending_instruction = ""
            try: self._store.put(sess)
            except Exception:
                tg_bot.logger.warning("session persist failed", exc_info=True)

        # A Stop pressed while this batch was still debouncing pre-dates it —
        # drop the whole batch rather than let it become a task that
        # _stop_requested_after can no longer see (its enqueue_ts would be
        # stamped here, AFTER the Stop).
        with self._stop_lock:
            _stop_ts = self._stop_requests.get(chat_id, 0.0)
        if _stop_ts and _arrival_ts <= _stop_ts:
            tg_bot.logger.info("Dropping stale debounced batch chat=%s "
                       "(arrived %.3f <= stop %.3f)", chat_id, _arrival_ts, _stop_ts)
            self._activity.log(chat_id, "system",
                               "[stop] dropped batch that arrived before Stop")
            return

        # A batch can legitimately contain button presses aimed at DIFFERENT
        # pictures (tap [upscale] under an old picture, then under a newer one,
        # both inside one debounce window). A merged _Task carries a single
        # image_id (see pressed_image_id below), and identical command text
        # would be collapsed by the double-tap dedup further down, so the
        # earlier press would vanish silently. Split on a genuine conflict
        # instead of merging across targets; a batch that agrees on its target
        # (plain conversation, or several taps on the SAME picture) is
        # unaffected.
        _distinct_targets = {it.get("image_id") for it in batch if it.get("image_id")}
        # Same for two SEPARATE photo uploads (or a photo and an album) in one
        # window: the loop below keeps a single per-batch `img_bytes` slot, so a
        # second image-bearing item would silently overwrite the first's bytes.
        # One image-bearing item with text around it is the normal merge.
        _image_bearing = sum(1 for it in batch if it.get("type") in ("photo", "album"))
        # A forwarded voice/video note is a hard `return` inside this loop on
        # every path (it starts its own ask/queue flow), so any other item in
        # its debounce window would be thrown away with the batch. A batch with
        # a fwd_voice/fwd video note alongside anything else is therefore not a
        # merge candidate: it is split into independent single-item pushes.
        _has_fwd_voice = any(it.get("type") == "fwd_voice" for it in batch)
        # ...except a forwarded note WITH the user's own words in the same
        # batch («Что думаешь?» typed next to a forwarded кружок): the words
        # are the instruction for the note, so they must not be answered apart
        # as small talk.
        if _has_fwd_voice and len(batch) > 1:
            _notes = [it for it in batch if it.get("type") == "fwd_voice"]
            _words = [(it.get("text") or "").strip() for it in batch
                      if it.get("type") == "text" and not it.get("forwarded")
                      and (it.get("text") or "").strip()
                      and not (it.get("text") or "").startswith("/")
                      and (it.get("text") or "").strip() not in tg_bot._LABEL2KEY]
            # Several forwarded pieces (voices, кружки, texts, several people,
            # any order): one conversation, each line under its author, asked
            # about ONCE -- not a "what do I do with it?" per piece.
            # A forwarded POST is its videos and its pictures together: the pictures are read into the same
            # conversation, never answered on their own (no «Что ты хочешь сделать с этим фото?»
            # next to the post's «что с ней сделать?»).
            _fwd_items = [it for it in batch if it.get("type") == "fwd_voice"
                          or (it.get("type") in ("text", "photo", "album") and it.get("forwarded"))]
            if len(_fwd_items) > 1:
                # Anything else in the window -- a menu button pressed while the
                # forward was landing («↩ Назад» among ten кружки),
                # a photo -- goes its own way. It must not veto the merge, or the
                # batch falls apart into a "what do I do with it?" per note.
                _wordset = set(_words)
                for _item in batch:
                    if _item in _fwd_items or (_item.get("type") == "text"
                                               and (_item.get("text") or "").strip() in _wordset):
                        continue
                    self._resolve_and_push(chat_id, [_item])
                _said = self._fwd_batch(chat_id, _fwd_items, ask=not _words)
                if _generation != sess.context_generation:
                    return
                if not _said or not _words:
                    return
                _asked = (getattr(self, "fwdv_intent", None) or tg_bot._fwdv_intent)(" ".join(_words))
                if _asked:
                    self._do_fwd_voice(chat_id, _asked, _said)
                    return
                batch = [{"type": "text", "text": " ".join(_words), "fwd_said": _said}]
            elif len(_notes) == 1 and len(_words) == len(batch) - 1 and _words:
                _notes[0]["companion"] = " ".join(_words)
                batch = _notes
        _has_fwd_voice = any(it.get("type") == "fwd_voice" for it in batch)
        # Several forwarded TEXTS with none of the user's own words are someone else's
        # conversation too: the same «what do I do with it?» (retell, ask...), not a comment
        # as if the user had said it (a forwarded dialogue, live 2026-10-17). One forwarded
        # text stays the quoted material of a turn (forwarded lyrics, a post for a waiting task).
        _fwd_texts = [it for it in batch if it.get("type") == "text" and it.get("forwarded")]
        _own_words = [it for it in batch if it.get("type") == "text" and not it.get("forwarded")
                      and (it.get("text") or "").strip()]
        if len(_fwd_texts) > 1 and not _has_fwd_voice and not _own_words:
            _fwd_items = [it for it in batch if it.get("forwarded")
                          and it.get("type") in ("text", "photo", "album")]
            for _item in batch:
                if _item not in _fwd_items:
                    self._resolve_and_push(chat_id, [_item])
            self._fwd_batch(chat_id, _fwd_items, ask=True)
            return
        if len(_distinct_targets) > 1 or _image_bearing > 1 \
                or (_has_fwd_voice and len(batch) > 1):
            for _item in batch:
                self._resolve_and_push(chat_id, [_item])
            return

        texts: list[str]   = []
        # Text WE wrote for the model, not words the user said. Kept apart so a
        # pending prompt prefix cannot attach itself to our own scaffolding --
        # see where the prefix is applied, below.
        scaffold: list[str] = []
        img_bytes: Optional[bytes] = None
        # True only when img_bytes is a picture the user actually SENT (photo/
        # album), never an internally-derived one (a video's frame sheet, a
        # sandbox render). A video note's sheet rides img_bytes exactly like a
        # real photo and is registered the same way, so downstream code that
        # offers picture-editing actions for "the photo this turn was about"
        # needs this flag to tell them apart -- see its use in tg_tasks.py.
        _photo_upload = False
        file_in_folder = False
        # Every OTHER successfully-downloaded photo from an album, beyond the
        # one that drives this task (a task can only carry one image_path/
        # img_bytes -- see the single-slot design below). They are logged (not
        # run) alongside the driving photo so "upscale the second one" has
        # something to resolve against.
        _album_extra_bytes: list[bytes] = []
        # The picture id captured at PRESS time on an image-verb button
        # (threaded in via the enqueued item's "image_id" key), NOT read out of
        # sess.target_image lazily — that single slot can already have been
        # overwritten by a second button press queued behind this one before
        # this batch is even processed.
        pressed_image_id = ""

        # A long lyric is cut by Telegram into several messages that land in one
        # batch: while 🎤 Cover or ✨/✍️ lyrics wait for words, the pieces are one
        # text.
        if getattr(sess, "cover_state", "") == "want_text" or getattr(sess, "lyrics_state", ""):
            _pieces = [it for it in batch if it.get("type") == "text" and not it.get("fwd_said")
                       and not (it.get("text") or "").startswith("/")
                       and not tg_bot._LABEL2KEY.get((it.get("text") or "").strip())]
            if len(_pieces) > 1:
                _joined = dict(_pieces[0], text="\n".join(it["text"] for it in _pieces))
                batch = [_joined] + [it for it in batch if it not in _pieces]

        fwd, fwd_at = [], None
        for item in batch:
            if _generation != sess.context_generation:
                return
            t = item["type"]
            if item.get("image_id"):
                pressed_image_id = item["image_id"]

            # Collection mode is a file-drop utility, not a chat turn: it
            # never reaches the agent. Handled before everything else so a
            # photo dropped mid-collection can't also get merged into texts/
            # img_bytes below and pushed as an ordinary picture message.
            if getattr(sess, "reg_state", "") == "lora_collect" and t in ("photo", "album"):
                cap = (item.get("caption") or "").strip()
                if t == "photo":
                    self._lora_collect_photo(chat_id, sess, lang, item["file_id"], cap)
                else:
                    self._lora_collect_album(chat_id, sess, lang,
                                             item.get("file_ids", []), cap)
                continue

            if t == "text":
                raw = item["text"]
                if (sess.menu == "randomizer_range" and not raw.startswith("/")
                        and not tg_bot._LABEL2KEY.get(raw.strip())):
                    from tg_randomizer import range_text
                    range_text(self, chat_id, sess, lang, raw)
                    continue
                if not item.get("forwarded") and not item.get("fwd_said") and not raw.startswith("/"):
                    self._bind_turn(sess, raw, batch)
                # 🎨 Restyle / 🎤 Cover waiting for words: this text is them, ahead
                # of everything that reads text as something else -- a forwarded
                # lyric went to the chat, an old forwarded voice note took it as
                # its answer, a song topic left armed sang it.
                if (not item.get("fwd_said") and not raw.startswith("/")
                        and not tg_bot._LABEL2KEY.get(raw.strip())):
                    if (getattr(sess, "lyrics_state", "")
                            and self._lyrics_take_text(chat_id, sess, lang, raw)):
                        continue
                    if (getattr(sess, "cover_state", "")
                            and self._cover_take_link(chat_id, sess, lang, raw)):
                        continue
                    if (getattr(sess, "continue_state", "") == "want_text"
                            and self._continue_take_text(chat_id, sess, lang, raw)):
                        continue
                    if (getattr(sess, "restyle_state", "") == "want_text"
                            and self._restyle_take_text(chat_id, sess, lang, raw)):
                        continue
                    if (getattr(sess, "cover_state", "") == "want_text"
                            and self._cover_take_text(chat_id, sess, lang, raw)):
                        continue
                if item.get("fwd_said"):
                    # The user's words that came with a forwarded batch: the
                    # task, with the conversation as quoted material.
                    texts.append(tg_bot._FWD_TEXT_FRAME.format(text=item["fwd_said"]))
                    texts.append(self._as_request(item, raw))
                    continue
                if item.get("forwarded") and item.get("own"):
                    # The bot's own earlier reply, forwarded back: its words,
                    # still a quotation (tg_bot._FWD_OWN_FRAME).
                    texts.append(tg_bot._FWD_OWN_FRAME.format(text=raw))
                    continue
                if item.get("forwarded"):
                    raw = tg_bot._unclaim(raw)
                    # Someone else's words: frame them so the model reads a
                    # forwarded post as a quotation and "о чём это?" / "ответь
                    # автору" afterwards refer to it. Not a button, not a
                    # command, not a question to the documents. A forwarded
                    # batch is one conversation, each line under its author.
                    fwd.append(f"{item['author']}: {raw}" if item.get("author") else raw)
                    if fwd_at is None:
                        fwd_at = len(texts)
                        texts.append("")
                    texts[fwd_at] = tg_bot._FWD_TEXT_FRAME.format(text="\n".join(fwd))
                    # a forwarded link is material too (a forwarded Short must be watched, not refused)
                    if ("://" in raw and not sess.reg_state and not sess.pending_prefix
                            and not os.getenv("F5_TEST_RUN")):
                        if self._read_links(ctx, sess, chat_id, user, raw, scaffold):
                            return   # a long video: the portioned-job offer is the answer
                    continue
                # A forwarded voice note is waiting on a choice and the user typed
                # the answer instead of pressing the button ("перескажи"). Honour
                # it here — otherwise the text went to the agent as a bare
                # instruction with no transcript attached and the note was lost.
                # `fwd_transcript_done` gates this, not the transcript itself:
                # the transcript now survives delivery so the inline keyboard
                # keeps working (see _do_fwd_voice), but a "перескажи" typed
                # long afterwards, about something else entirely, must NOT be
                # captured as an answer for a note already dealt with.
                # ✍️ armed (fwd_own) means the text IS the free-form question:
                # «что бы сказали твои друзья?» after ✍ must not come back as a
                # transcript.
                if (getattr(sess, "fwd_transcript", "")
                        and not getattr(sess, "fwd_own", "")
                        and not getattr(sess, "fwd_transcript_done", False)):
                    _asked = (getattr(self, "fwdv_intent", None) or tg_bot._fwdv_intent)(raw)
                    if _asked:
                        self._do_fwd_voice(chat_id, _asked, sess.fwd_transcript)
                        return

                # Buttons are resolved by KEY, so a label from any language (or
                # from a keyboard rendered before the user switched language)
                # still triggers its action instead of reaching the agent.
                key = tg_bot._LABEL2KEY.get(raw, "")

                # ✍️ Свой вариант was pressed: this text is the request about
                # the forwarded material, which rides along as a quotation.
                if (not key and not raw.startswith("/") and getattr(sess, "fwd_own", "")):
                    texts.append(tg_bot._FWD_TEXT_FRAME.format(text=sess.fwd_own))
                    texts.append(self._as_request(item, raw))
                    sess.fwd_own = ""
                    self._store.put(sess)
                    continue

                # 📚 a voice is waiting for its name: plain text is the name
                if (not key and not raw.startswith("/") and getattr(sess, "voice_naming", "")
                        and self._vl_take_name(chat_id, sess, lang, raw)):
                    continue

                # 👥 a character's action is the next plain text
                if (not key and not raw.startswith("/") and sess.cast_state.startswith("act:")
                        and self._cast_take_text(chat_id, sess, lang, raw)):
                    continue

                # 🎬 Animate collects voices: a video link's sound is the next sample.
                if (not key and getattr(sess, "anim_voice_state", "") == "collect"
                        and self._anim_voice_take_link(chat_id, sess, lang, raw)):
                    continue

                # 📚 Audiobook waits for the narrator: a video link is the sample, not a video to retell.
                if (not key and getattr(sess, "book_state", "") == "want_voice"
                        and self._book_take_link(chat_id, sess, lang, raw)):
                    continue

                # 📚 Audiobook waits for the book: a pasted text is the book, a short one only a hint.
                if (not key and not raw.startswith("/") and getattr(sess, "book_state", "") == "want_book"
                        and self._book_take_text(chat_id, sess, lang, raw)):
                    continue

                # 🎙 Clone voice armed + a video link: the video's voice is the
                # sample.
                if (not key and getattr(sess, "clone_state", "")
                        and self._clone_take_link(chat_id, sess, lang, raw)):
                    continue

                # 🎙 Clone voice has a voice: plain text (not a button, not a
                # /command) is the next thing to say in it.
                if (not key and not raw.startswith("/")
                        and getattr(sess, "clone_state", "") == "want_text"
                        and self._clone_take_text(chat_id, sess, lang, raw)):
                    continue

                # Voice toggle (fast path)
                if key in ("reply_fmt", "voice_on", "voice_off", "reply_text", "reply_both", "reply_voice"):
                    if key in ("voice_on", "voice_off"):   # old keyboards: still a direct switch
                        sess.voice_on = (key == "voice_on")
                        self._store.put(sess)
                    # Inline, so the reply keyboard stays open underneath.
                    self._send_text(chat_id, tg_bot._t("reply_pick", lang),
                                    keyboard=tg_bot._reply_mode_kb(sess.reply_mode, lang))
                    continue

                if key == "my_voice":
                    self._goto_menu(sess, "settings")
                    # Always arms for a NEW sample; going back to the default is the ↩️ button.
                    sess.clone_state = "want_assistant"
                    self._store.put(sess)
                    self._vl_screen(chat_id, sess, lang)
                    continue

                if key == "photo_file":
                    sess.photo_file = not getattr(sess, "photo_file", False); self._store.put(sess)
                    self._goto_menu(sess, "settings")
                    state = tg_bot._t("on", lang) if sess.photo_file else tg_bot._t("off", lang)
                    self._send_text(chat_id, tg_bot._t("photo_file_state", lang, state=state),
                                    parse_mode="HTML",
                                    keyboard=tg_bot._settings_kb(sess.reply_mode, sess.is_admin, lang))
                    continue

                if key == "think":
                    sess.think = not getattr(sess, "think", False); self._store.put(sess)
                    self._goto_menu(sess, "settings")
                    state = tg_bot._t("on", lang) if sess.think else tg_bot._t("off", lang)
                    self._send_text(chat_id, tg_bot._t("think_state", lang, state=state),
                                    parse_mode="HTML",
                                    keyboard=tg_bot._settings_kb(sess.reply_mode, sess.is_admin, lang))
                    continue

                direct = tg_bot._DIRECT_KB.get(key, "")
                if not direct and _tg_music.is_reset_phrase(raw, ctx=self.reads):
                    direct = "__song_reset__"
                if direct and direct != "__clone_voice__":
                    self._clone_disarm(sess)
                if direct and direct != "__cover__":
                    self._cover_disarm(sess)
                if direct and direct not in ("__lyrics_improve__", "__lyrics_write__"):
                    self._lyrics_disarm(sess)
                if direct and direct != "__restyle__":
                    self._restyle_disarm(sess)
                if direct and direct != "__continue_video__":
                    self._continue_disarm(sess)
                if direct and direct != "__audiobook__":
                    self._book_disarm(sess)
                if direct and direct != "__joint_video__" and sess.cast_state:
                    sess.cast_state = ""            # a menu press leaves the 👥 card
                    self._store.put(sess)
                if direct and (getattr(sess, "anim_voice_state", "") or getattr(sess, "voice_naming", "")):
                    sess.anim_voice_state = ""      # a menu press ends 🎙 sample collection
                    sess.voice_naming = ""          # ... and an unanswered «как подписать»
                    self._store.put(sess)
                if direct.startswith("__randomizer_"):
                    from tg_randomizer import action
                    action(self, chat_id, sess, lang, key)
                    continue
                if direct == "__audiobook__":
                    self._goto_menu(sess, "cr_music")
                    self._start_book_flow(chat_id, sess, lang); continue
                if direct == "__continue_video__":
                    self._goto_menu(sess, "cr_video")
                    self._start_continue_flow(chat_id, sess, lang); continue
                if direct == "__restyle__":
                    self._goto_menu(sess, "cr_video")
                    self._start_restyle_flow(chat_id, sess, lang); continue
                if direct == "__cover__":
                    self._goto_menu(sess, "cr_music")
                    self._start_cover_flow(chat_id, sess, lang); continue
                if direct in ("__lyrics_improve__", "__lyrics_write__"):
                    self._goto_menu(sess, "cr_music")
                    self._start_lyrics_flow(chat_id, sess, lang,
                                            "improve" if direct == "__lyrics_improve__" else "write")
                    continue
                if direct == "__clone_voice__":
                    self._goto_menu(sess, "cr_music")
                    self._start_clone_flow(chat_id, sess, lang); continue
                if direct == "__clear__":
                    self._request_stop(chat_id)
                    sess.clear_context(); self._store.put(sess)
                    tg_bot._delete_chat_facts_file(chat_id)
                    self._send_text(chat_id, tg_bot._t("cleared", lang),
                                    keyboard=self._main_menu_kb(sess, lang)); continue
                if direct == "__help__":
                    self._handle_command(chat_id, "/help"); continue
                # These four live in Settings and answer with the Settings keyboard,
                # so the session has to agree that that is where we are — otherwise
                # ⬅ Back is decided by whatever menu we happened to be in before.
                if direct == "__status__":
                    self._goto_menu(sess, "settings")
                    self._send_status(chat_id, sess); continue
                if direct == "__facts__":
                    self._goto_menu(sess, "settings")
                    self._send_facts(chat_id, sess); continue
                if direct == "__lang__":
                    self._goto_menu(sess, "settings")
                    self._send_lang_menu(chat_id, lang); continue
                if direct == "__wear__":
                    self._goto_menu(sess, "weather")
                    self._start_weather_flow(chat_id, sess, lang); continue
                # One tap per window. Each uses the remembered city rather than
                # asking again; only 🏙 City arms the text capture.
                if direct in ("__wtw_now__", "__wtw_24__", "__wtw_48__"):
                    self._goto_menu(sess, "weather")
                    self._weather_quick(chat_id, sess, lang, {
                        "__wtw_now__": 0, "__wtw_24__": 24, "__wtw_48__": 48,
                    }[direct]); continue
                if direct == "__wtw_pickdate__":
                    self._goto_menu(sess, "weather")
                    self._weather_ask_date(chat_id, sess, lang); continue
                if direct == "__wtw_setcity__":
                    self._goto_menu(sess, "weather")
                    self._start_weather_flow(chat_id, sess, lang); continue
                if direct == "__menu_weather__":
                    self._goto_menu(sess, "weather")
                    self._send_text(chat_id, tg_bot._t("menu_weather_title", lang),
                                    parse_mode="HTML", keyboard=tg_bot._weather_kb(lang)); continue
                if direct in ("__menu_cr_images__", "__menu_cr_music__", "__menu_cr_video__"):
                    tier = direct.strip("_")[5:]
                    self._goto_menu(sess, tier)
                    self._send_text(chat_id, tg_bot._t("menu_%s_title" % tier, lang),
                                    parse_mode="HTML", keyboard=self._state_kb(sess, lang)); continue
                if direct == "__menu_creativity__":
                    self._goto_menu(sess, "creativity")
                    self._send_text(chat_id, tg_bot._t("menu_creativity_title", lang),
                                    parse_mode="HTML", keyboard=tg_bot._creativity_kb(lang)); continue
                if direct == "__joint_video__":
                    self._goto_menu(sess, "cr_video")
                    self._cast_start(chat_id, sess, lang); continue
                if direct == "__animate_photo__":
                    self._goto_menu(sess, "cr_video")
                    sess.awaiting_animate_photo = True
                    self._store.put(sess)
                    self._send_text(chat_id, tg_bot._t("animate_ask_photo", lang)); continue
                if direct == "__style_photo__":
                    # 🎭 Change style from the Creativity menu, not tied to a
                    # picture already in the chat -- asks for the photo first,
                    # same shape as __animate_photo__.
                    self._goto_menu(sess, "cr_images")
                    sess.awaiting_style_photo = True
                    self._store.put(sess)
                    self._send_text(chat_id, tg_bot._t("style_menu_ask_photo", lang)); continue
                if direct == "__menu_characters__":
                    # A sub-screen of Creativity, like __song_settings__: the
                    # reply keyboard stays the creativity one, so ⬅ Back still
                    # means "out of Creativity" and the menu/pending_prefix
                    # invariant holds. The character list itself is INLINE.
                    self._goto_menu(sess, "cr_images")
                    self._send_characters_menu(chat_id, lang); continue
                if direct == "__songs__":
                    self._goto_menu(sess, "cr_music")
                    self._start_song_flow(chat_id, sess, lang); continue
                if direct == "__sandbox__":
                    # Routed through the same handler as /files: one grant
                    # check, one refusal wording, nothing here to keep in sync
                    # with it. The listing carries its own INLINE buttons, so
                    # the Creativity reply keyboard stays on screen and the
                    # session stays in that menu, like songs/mashup above. The
                    # refusal/empty answers show the MAIN keyboard, and
                    # _main_menu_kb clears sess.menu along with it.
                    self._goto_menu(sess, "creativity")
                    self._handle_sandbox_command(chat_id, sess, user, lang,
                                                 "/files")
                    continue
                if direct == "__song_settings__":
                    # Stays in the creativity menu: this is a sub-screen of it,
                    # so ⬅ Back on the reply keyboard still means "out of
                    # Creativity" and the menu/pending_prefix invariant holds.
                    self._goto_menu(sess, "cr_music")
                    self._send_text(chat_id, tg_bot._music_menu_text(sess, lang),
                                    parse_mode="HTML",
                                    keyboard=tg_bot._music_menu_kb(sess, lang)); continue
                if direct == "__song_reset__":
                    # The typed twin of the ♻️ button: same reset, same words,
                    # the menu redrawn so the user SEES every line back on Auto.
                    _tg_music.reset_all(sess); self._store.put(sess)
                    self._goto_menu(sess, "creativity")
                    self._send_text(chat_id, tg_bot._t("ms_was_reset", lang))
                    self._send_text(chat_id, tg_bot._music_menu_text(sess, lang),
                                    parse_mode="HTML",
                                    keyboard=tg_bot._music_menu_kb(sess, lang))
                    self._activity.log(chat_id, "system", "[song settings] reset (typed)"); continue
                if direct == "__video_settings__":
                    import tg_video as _tv
                    self._goto_menu(sess, "cr_video")
                    self._send_text(chat_id, _tv._video_menu_text(sess, lang),
                                    parse_mode="HTML",
                                    keyboard=_tv._video_menu_kb(sess, lang)); continue
                if direct == "__notifications__":
                    self._goto_menu(sess, "settings")
                    if user:
                        sub_on = "startup" in user.subscriptions
                        state = tg_bot._t("on", lang) if sub_on else tg_bot._t("off", lang)
                        # The toggle gates both the startup and the going-offline notice.
                        body = tg_bot._t("notif_state", lang, state=state)
                    else:
                        # No user record behind this session (deleted or rejected
                        # while the keyboard was still on screen): answer anyway,
                        # a silent button looks broken.
                        body = tg_bot._t("no_account", lang)
                    self._send_text(chat_id, tg_bot._t("notif_title", lang) + body,
                        parse_mode="HTML",
                        keyboard=tg_bot._settings_kb(sess.reply_mode, sess.is_admin, lang))
                    continue
                if direct == "__admin_panel__" and sess.is_admin:
                    sess.pending_prefix = ""; self._store.put(sess)
                    self._send_admin_panel(chat_id); continue
                if direct == "__admin_panel__":
                    # a stale keyboard (rights revoked) or a typed label: it fell through and
                    # went to the model as «search the web for: 🔐 Админ-панель»
                    self._send_text(chat_id, tg_bot._t("admin_only", lang)); continue
                if direct == "__account__":
                    sess.pending_prefix = ""; self._store.put(sess)
                    self._send_account_menu(chat_id, user, sess); continue
                if direct == "__feedback__":
                    sess.pending_prefix = ""
                    sess.reg_state = "feedback"; self._store.put(sess)
                    self._send_text(chat_id, tg_bot._t("feedback_ask", lang),
                        parse_mode="HTML",
                        keyboard=self._main_menu_kb(sess, lang)); continue
                # ── documents ─────────────────────────────────────────────────
                if direct == "__menu_library__":
                    self._goto_menu(sess, "library")
                    self._send_text(chat_id, tg_bot._t("lib_menu", lang), parse_mode="HTML",
                                    keyboard=tg_bot._library_kb(sess.use_docs, lang)); continue
                if direct == "__lib_list__":
                    self._send_library_list(chat_id, sess); continue
                if direct == "__lib_toggle__":
                    sess.use_docs = not sess.use_docs; self._store.put(sess)
                    state = tg_bot._t("on", lang) if sess.use_docs else tg_bot._t("off", lang)
                    self._send_text(chat_id, tg_bot._t("lib_toggled", lang, state=state),
                                    parse_mode="HTML",
                                    keyboard=tg_bot._library_kb(sess.use_docs, lang)); continue
                if direct == "__lib_clear__":
                    n = self._clear_library(chat_id)
                    sess.use_docs = False; self._store.put(sess)
                    self._send_text(chat_id, tg_bot._t("lib_cleared", lang, n=n),
                                    keyboard=tg_bot._library_kb(False, lang)); continue
                if direct == "__image_size__":
                    self._send_text(chat_id, tg_bot._size_menu_text(sess, lang),
                                    parse_mode="HTML",
                                    keyboard=tg_bot._size_menu_kb(sess, lang)); continue
                if direct == "__research_depth__":
                    self._send_text(chat_id, tg_bot._depth_menu_text(sess, lang),
                                    parse_mode="HTML",
                                    keyboard=tg_bot._depth_menu_kb(sess, lang)); continue
                # sub-menu navigation
                if direct == "__menu_draw__":
                    # Free-text typed while in this submenu defaults to image generation.
                    # Tapping a submenu button (Generate/Edit) overrides the prefix.
                    self._goto_menu(sess, "draw", "generate an image of: ")
                    self._send_text(chat_id, tg_bot._t("menu_draw_title", lang),
                                    parse_mode="HTML", keyboard=tg_bot._draw_kb(lang)); continue
                if direct in ("__menu_tasks__", "__sch_list__", "__sch_weather__"):
                    self._goto_menu(sess, "tasks")
                    if direct == "__menu_tasks__":
                        self._send_text(chat_id, tg_bot._t("menu_tasks_title", lang),
                                        parse_mode="HTML", keyboard=tg_bot._tasks_kb(lang))
                    if direct == "__sch_weather__":
                        self._send_weather_times(chat_id, lang)
                    else:
                        self._send_tasks_list(chat_id, lang)
                    continue
                if direct == "__menu_ozon__":
                    # Free text typed in this menu is an Ozon product search.
                    self._goto_menu(sess, "ozon", "find on ozon: ")
                    self._send_text(chat_id, tg_bot._t("menu_ozon_title", lang),
                                    parse_mode="HTML", keyboard=tg_bot._ozon_kb(lang)); continue
                if direct == "__menu_search__":
                    # Free-text typed while in this submenu defaults to web search.
                    self._goto_menu(sess, "search", "search the web for: ")
                    self._send_text(chat_id, tg_bot._t("menu_search_title", lang),
                                    parse_mode="HTML", keyboard=tg_bot._search_kb(lang)); continue
                if direct == "__menu_settings__":
                    self._goto_menu(sess, "settings")
                    self._send_text(chat_id, self._settings_header(sess, lang),
                        parse_mode="HTML",
                        keyboard=tg_bot._settings_kb(sess.reply_mode, sess.is_admin, lang)); continue
                if direct == "__menu_back__":
                    # The creativity tiers (and Draw, opened from 🖼 Images) go up one level.
                    up = {"cr_images": "creativity", "cr_music": "creativity",
                          "cr_video": "creativity", "draw": "cr_images",
                          "randomizer_range": "randomizer",
                          "randomizer_prediction": "randomizer", "ozon_prompt": "ozon"}.get(sess.menu)
                    if up:
                        self._goto_menu(sess, up, tg_bot._PROMPT_KB['ozon_find'] if up == 'ozon' else '')
                        self._send_text(chat_id, tg_bot._t(
                            "menu_%s_title" % up, lang), parse_mode="HTML",
                            keyboard=self._state_kb(sess, lang))
                        continue
                    # Documents is a sub-submenu opened from Settings, so its Back
                    # goes UP ONE level, not all the way out.
                    if sess.menu == "library":
                        self._goto_menu(sess, "settings")
                        self._send_text(chat_id, self._settings_header(sess, lang),
                            parse_mode="HTML",
                            keyboard=tg_bot._settings_kb(sess.reply_mode, sess.is_admin, lang))
                        continue
                    self._goto_menu(sess, "")
                    self._send_text(chat_id, tg_bot._t("main_menu", lang),
                                    keyboard=self._main_menu_kb(sess, lang)); continue
                if direct and not direct.startswith("__"):
                    # Direct actions (Regenerate, Analyze Photo, etc.) consume
                    # the pending prefix — clear it so the next free-text message
                    # isn't accidentally prefixed with "generate an image of: ".
                    if sess.pending_prefix or sess.pending_photo:
                        # Same reasoning for the armed photo instruction: running a
                        # direct action satisfies it, and leaving it set means some
                        # later caption-less photo re-runs it out of nowhere.
                        sess.pending_prefix = ""; sess.pending_photo = ""
                        self._store.put(sess)
                    # 📷 Analyze Photo asks about an image that has to already exist.
                    # Pressed FIRST ("tell it what I want, then hand it the photo")
                    # it would fire a full turn against nothing, so arm it instead
                    # and let the next photo trigger it.
                    # `last_image_path` alone is the wrong question: it is a
                    # single slot refreshed at the end of a turn, while the
                    # per-chat image REGISTER is what the buttons and "which
                    # picture?" resolve against. And it outlives the picture it
                    # names (deleted off disk, tmp-dir cleaned), so hold it to the
                    # same existence check `_live_images` applies to the register.
                    _path_alive = bool(sess.last_image_path
                                        and os.path.exists(sess.last_image_path))
                    _have_image = (_path_alive or bool(tg_bot._live_images(sess)))
                    if key == "analyze" and not _have_image and img_bytes is None:
                        sess.pending_photo = direct
                        self._store.put(sess)
                        self._send_text(chat_id, tg_bot._t("await_photo", lang),
                                        keyboard=self._state_kb(sess, lang)); continue
                    # A picture is already in the chat, so analyze runs on it
                    # immediately -- but arm pending_photo alongside, so a fresh
                    # photo sent right after still pairs with the analyze
                    # instruction (the "genuine text turn disarms it" rule below
                    # only fires for plain text, never for the photo itself).
                    if key == "analyze" and _have_image and img_bytes is None:
                        sess.pending_photo = direct
                        if not pressed_image_id:
                            live = tg_bot._live_images(sess)
                            target = (tg_bot._image_by_id(sess, sess.target_image or sess.turn_image)
                                      or tg_bot._image_by_path(sess, sess.last_image_path))
                            if target and not os.path.exists(target.get("path", "")):
                                target = None
                            if not target and len(live) == 1:
                                target = live[0]
                            if target:
                                pressed_image_id = target["id"]
                            elif live:
                                sess.pending_instruction = direct
                                self._store.put(sess)
                                self._send_text(chat_id, tg_bot._t("img_which", lang, n=len(live)),
                                    parse_mode="HTML", keyboard=tg_bot._image_choice_kb(sess, lang))
                                return
                        self._store.put(sess)
                    # 🔄 Regenerate with nothing to regenerate would burn a full
                    # agent turn (and an image-quota unit): give the same honest
                    # refusal 📷 Analyze gets instead of enqueueing a task.
                    if key == "regenerate" and not sess.last_image_prompt \
                            and not _have_image and img_bytes is None:
                        self._send_text(chat_id, tg_bot._t("no_regenerate", lang),
                                        keyboard=self._state_kb(sess, lang)); continue
                    texts.append(direct); continue

                if key in tg_bot._PROMPT_KB:
                    if key.startswith('ozon_'):
                        self._goto_menu(sess, 'ozon_prompt', tg_bot._PROMPT_KB[key])
                    else:
                        sess.pending_prefix = tg_bot._PROMPT_KB[key]; self._store.put(sess)
                    hint = tg_bot._prompt_label(tg_bot._PROMPT_KB[key], lang)
                    # Stay in the submenu the button lives in (the MAIN keyboard
                    # would close the menu the user just opened). 📌 Remember
                    # lives in Settings.
                    self._send_text(chat_id,
                        tg_bot._t("prompt_hint", lang, hint=_html_mod.escape(hint)),
                        parse_mode="HTML",
                        keyboard=(tg_bot._settings_kb(sess.reply_mode, sess.is_admin, lang)
                                  if key == "remember"
                                  else self._state_kb(sess, lang))); continue

                # An armed 📷 Analyze is a promise about the VERY NEXT photo, not
                # a standing mode. Nothing else clears it except another direct
                # action or clear_context, so an unrelated question asked in
                # between would leave it armed and force-analyze some later unrelated
                # caption-less photo the user only meant to hand over quietly.
                # A genuine text turn — one that reaches here, i.e. is neither a
                # recognised button nor a forwarded-voice answer — means the user
                # has moved on; disarm it.
                if sess.pending_photo:
                    sess.pending_photo = ""
                    self._store.put(sess)

                # "Сочини песню про кота" typed in plain chat IS the song flow (the
                # chat model would write lyrics and read them aloud). Only outside
                # the armed capture states -- those already own the text.
                # Same for a forecast asked in plain words: the weather API
                # answers it; web search does not.
                _wx = _tg_weather.weather_request(raw, ctx=self.reads) \
                    if not sess.reg_state and not sess.pending_prefix else {}
                if (not _wx and sess.menu == "weather" and not sess.reg_state
                        and not sess.pending_prefix):
                    _wx = _tg_weather.weather_followup(raw, ctx=self.reads)   # "а в Сочи?"
                if _wx:
                    # "а в Москве?" after "погода завтра в Париже" means
                    # tomorrow in Moscow, not the next 24 hours.
                    _when = self.__dict__.setdefault("_wx_when", {})
                    if _wx["date"] is None and _wx["hours"] is None:
                        _wx = dict(_wx, **dict(zip(("date", "hours"),
                                                   _when.get(chat_id, (None, None)))))
                    import datetime as _dtm
                    if _wx["date"] is not None and _wx["date"] < _dtm.date.today():
                        _wx = dict(_wx, date=None)   # a stale "tomorrow" from days ago
                    _when[chat_id] = (_wx["date"], _wx["hours"])
                    self._goto_menu(sess, "weather")
                    # An English question gets an English forecast, whatever the
                    # session language.
                    from tg_tasks import _message_script
                    if _message_script(raw) == "en":
                        lang = "en"
                    _city = _wx["city"] or self._remembered_city(chat_id)
                    self._activity.log(chat_id, "system",
                                       f"[weather] typed request -> lookup: {raw[:80]}",
                                       (user.name if user else str(chat_id)))
                    self._start_weather_lookup(
                        chat_id, _city, lang,
                        remember_for=(chat_id if _wx["city"] else 0),
                        hours=_wx["hours"], date=_wx["date"])
                    continue

                _is_song, _secs = _tg_songs.song_request(raw, ctx=self.reads)
                _last_song = _tg_songs.LAST_SONG.pop(chat_id, "")
                if not _is_song and _last_song and _tg_songs.song_followup(raw, ctx=self.reads):
                    # «а теперь в стиле рок» after a song reached the agent, which has no song tool
                    _is_song, _secs = True, _tg_songs.request_seconds(raw, after_song=True, ctx=self.reads)
                else:
                    _last_song = ""
                if _is_song and not sess.reg_state and not sess.pending_prefix:
                    self._goto_menu(sess, "creativity")
                    self._activity.log(chat_id, "system",
                                       f"[songs] typed request -> song flow: {raw[:80]}",
                                       (user.name if user else str(chat_id)))
                    _topic = self._as_request(item, raw)    # «спой про это» keeps «это»
                    if _tg_songs.looks_like_lyrics(raw.partition("\n")[2]):
                        # the words typed under the request itself -- first: right after a
                        # song they were taken for «rewrite the last one» and the bot sang
                        # its own lyric, three verses short (2026-10-22)
                        _head, _, _body = raw.partition("\n")
                        _topic = _head + "\n" + _tg_songs.LYRICS_MARK + _body
                    elif _last_song:
                        _topic = _tg_songs.followup_topic(raw, _last_song, ctx=self.reads)
                    elif _tg_songs.refers_to_previous_text(raw, ctx=self.reads):
                        _prev = self._last_bot_text(sess)
                        if _tg_songs.looks_like_lyrics(_prev):
                            _topic = _tg_songs.LYRICS_MARK + _prev
                    texts.append(_tg_songs.song_payload(
                        _topic, _tg_songs.snap_duration(_secs)))
                    continue

                # A double-tap on an inline action button lands as two identical
                # items inside one debounce window, and the batch is joined with
                # newlines — the model would answer "outpaint the current image…"
                # twice in one message. Collapsing a repeat cannot lose meaning.
                _req = self._as_request(item, raw)
                if not texts or texts[-1] != _req:
                    texts.append(_req)
                # A link is a page to read (tg_links). Off under tests: no
                # suite may reach the network. Not inside an armed capture
                # state (a password, a song topic) and not for machine text.
                # (a link is MATERIAL: one in the replied-to message is read too)
                if ("://" in _req and not sess.reg_state and not sess.pending_prefix
                        and raw not in tg_bot._MACHINE_PAYLOADS
                        and not os.getenv("F5_TEST_RUN")):
                    if self._read_links(ctx, sess, chat_id, user, _req, scaffold):
                        return   # a long video: the portioned-job offer is the answer
                elif (getattr(sess, "last_link", None) and "?" in raw and not sess.pending_prefix
                        and not os.getenv("F5_TEST_RUN")):
                    # «а когда основан?» after a link was answered from the cut
                    # fitted to the first question: refit the page to this one.
                    # a watched video is answered from what was seen and heard,
                    # not from its YouTube page
                    _lc = (_tg_links.WATCHED.get(sess.last_link)
                           or _tg_links.link_context(raw, url=sess.last_link))
                    if _lc:
                        scaffold.append(_lc)

            elif t == "voice":
                if ctx is None:
                    self._send_text(chat_id, tg_bot._t("not_ready", lang),
                                    keyboard=self._main_menu_kb(sess, lang)); return
                if self._cover_from_caption(chat_id, sess, lang, item):
                    return
                self._api_post("sendChatAction", {"chat_id": chat_id, "action": "typing"})
                _media = item.get("media", "voice")
                if _media == "video":
                    import tg_video as _tv
                    if _tv.is_long(sess, item.get("seconds") or 0):
                        # A long video is not one turn: offer the portioned
                        # job (tg_video) and let the user press ▶️.
                        self._long_video_offer(chat_id, sess, lang, item); return
                _data = self._dl_bytes(item["file_id"]) if _media == "video" else None
                transcribed = self._transcribe(ctx, item["file_id"], _media,
                                               lang_hint=(sess.lang or "ru"), data=_data,
                                               speakers=(_media == "video"))
                _seen = (self._look_video(ctx, item["file_id"], data=_data, lang=(sess.lang or "ru"),
                                          transcript=transcribed or "")
                         if _media == "video" else {})
                if _generation != sess.context_generation:
                    return
                if not transcribed and not _seen.get("description"):
                    self._send_text(chat_id, tg_bot._t("no_transcribe", lang),
                                    keyboard=self._main_menu_kb(sess, lang)); return
                texts.append(((item.get("caption") or "") + " " + (transcribed or "")).strip())
                if _seen.get("description"):
                    # The picture of the video, for the model: what the
                    # frames show, so «что это у меня?» is answered from the
                    # eyes and the ears both.
                    scaffold.append(tg_bot._video_seen_note(_seen))
                    if _seen.get("sheet"):
                        try:
                            img_bytes = Path(_seen["sheet"]).read_bytes()
                        except Exception:
                            pass
                    # In play like a forwarded clip's sheet: «продолжи» under a GIF sent
                    # straight to the bot continues that GIF (its clip sits beside the
                    # sheet) instead of finding no video (owner, 2026-10-12).
                    if _seen.get("sheet") and os.path.exists(_seen["sheet"]):
                        tg_bot._put_in_play(sess, _seen["sheet"], tg_bot._t("lbl_video_frames", lang))
                        self._store.put(sess)

            elif t == "fwd_voice":
                if ctx is None:
                    self._send_text(chat_id, tg_bot._t("not_ready", lang),
                                    keyboard=self._main_menu_kb(sess, lang)); return
                _media = item.get("media", "voice")
                if _media == "video":
                    import tg_video as _tv
                    if _tv.is_long(sess, item.get("seconds") or 0):
                        self._long_video_offer(chat_id, sess, lang, item); return
                self._send_text(chat_id, tg_bot._t("fwd_video_work" if _media == "video"
                                                   else "fwd_voice_work", lang))
                self._api_post("sendChatAction", {"chat_id": chat_id, "action": "typing"})
                _data = self._dl_bytes(item["file_id"]) if _media == "video" else None
                said = self._transcribe(ctx, item["file_id"], _media,
                                        lang_hint=(sess.lang or "ru"), data=_data,
                                        speakers=True)
                _seen = (self._look_video(ctx, item["file_id"], data=_data, lang=(sess.lang or "ru"),
                                          transcript=said or "", caption=item.get("caption") or "")
                         if _media == "video" else {})
                if _generation != sess.context_generation:
                    return
                if _seen.get("description"):
                    # A forwarded VIDEO is listened to AND looked at: the
                    # transcript the buttons act on carries what was seen.
                    # In the user's language: the 📝 button shows this text as is.
                    # The speech is labelled like the caption and the storyboard: bare, the
                    # retelling took Putin's own words for the sender's caption (2026-10-15).
                    heard = tg_bot._t("fwd_video_heard", lang, heard=said.strip()) if (said or "").strip() else ""
                    said = (heard + "\n\n" + tg_bot._t("fwd_video_seen", lang,
                                                           seen=_seen["description"].strip())).strip()
                _companion = (item.get("companion") or "").strip()
                _asked = (getattr(self, "fwdv_intent", None) or tg_bot._fwdv_intent)(item.get("caption") or _companion)
                _cap = (item.get("caption") or "").strip()
                if _cap and not _asked and _media == "video" and said:
                    # A caption that is not an order is how the sender framed the clip
                    # (a meme's setup): the retelling reads picture and words against it.
                    said = tg_bot._t("fwd_video_caption", lang, caption=_cap) + "\n\n" + said
                if not said:
                    self._send_text(chat_id, tg_bot._t("fwd_voice_none", lang),
                                    keyboard=self._main_menu_kb(sess, lang)); return
                # Hold the transcript and ASK: a two-minute rant wants a summary, a
                # delivery address wants every word, and the bot must not guess.
                # A fresh id per forward — see _Session.fwd_transcript_id.
                sess.fwd_transcript = said[:20000]
                sess.fwd_transcript_id = uuid.uuid4().hex[:8]
                sess.remember_fwd(sess.fwd_transcript_id, sess.fwd_transcript)
                # A new forward re-arms the typed-answer shortcut that the
                # previous one retired once it had been acted on.
                sess.fwd_transcript_done = False
                self._store.put(sess)
                # If they already said what they want ("перескажи", "расшифруй"),
                # do it instead of asking again.
                if _asked:
                    self._do_fwd_voice(chat_id, _asked, said[:20000])
                    return
                if _companion:
                    # The user's words came WITH the note: they are the task.
                    # The note goes in as quoted material, the words as the
                    # request, and the turn runs -- no "what do I do with it?".
                    if _seen.get("sheet") and os.path.exists(_seen["sheet"]):
                        tg_bot._put_in_play(sess, _seen["sheet"], tg_bot._t("lbl_video_frames", lang))
                        self._store.put(sess)
                    texts.append(tg_bot._FWD_TEXT_FRAME.format(text=said[:20000]))
                    texts.append(_companion)
                    self._activity.log(chat_id, "system",
                                       f"[fwd {_media}] {len(said)} chars + the user's words: "
                                       f"{_companion[:60]!r}",
                                       (user.name if user else str(chat_id)))
                    continue
                mins = tg_bot._fmt_secs(item.get("seconds") or 0, lang)
                # Into the chat history NOW, storyboard included: a question
                # typed instead of a button press ("а что за лампа?") must
                # land on this video, not on whatever was discussed last week.
                self._commit_media_turn(chat_id, said, _seen.get("description") or "(listened; asked what to do with it)",
                                        bool(_seen.get("description")), _generation)
                # The keyframe sheet becomes the chat's current picture, so a
                # follow-up ("дай экспертное мнение") looks at the FRAMES
                # again (graph.needs_relook) instead of reasoning from the
                # storyboard's one-liners.
                if _seen.get("sheet") and os.path.exists(_seen["sheet"]):
                    tg_bot._put_in_play(sess, _seen["sheet"], tg_bot._t("lbl_video_frames", lang))
                    self._store.put(sess)
                if _media == "video":
                    sess.__dict__.setdefault("fwd_media", {})[sess.fwd_transcript_id] = [
                        {"kind": "video", "file_id": item["file_id"], "label": tg_bot._t("fwd_lbl_video", lang, n=1)}]
                if _seen.get("description"):
                    self._send_text(chat_id, tg_bot._t("fwd_video_ask", lang, mins=mins),
                                    parse_mode="HTML",
                                    keyboard=tg_bot._fwd_voice_kb(lang, sess.fwd_transcript_id,
                                                                  board=True, cont=True))
                    self._activity.log(chat_id, "system",
                                       f"[fwd video] {len(said)} chars, {len(_seen.get('times') or [])} frames seen",
                                       (user.name if user else str(chat_id)))
                    return
                self._send_text(chat_id, tg_bot._t("fwd_voice_ask", lang, mins=mins),
                                parse_mode="HTML",
                                keyboard=tg_bot._fwd_voice_kb(lang, sess.fwd_transcript_id))
                # `uname` is bound further down, after the quota block: use the
                # user record here.
                self._activity.log(chat_id, "system",
                                   f"[fwd voice] {len(said)} chars transcribed",
                                   (user.name if user else str(chat_id)))
                return

            elif t == "photo":
                data = self._dl_bytes(item["file_id"])
                # Several single photos in one batch (forwarded posts, photos sent one by
                # one) are an album too: each used to overwrite the last, and only one
                # reached the model.
                if data and img_bytes is not None:
                    _album_extra_bytes.append(data)
                elif data: img_bytes = data; _photo_upload = True
                cap = (item.get("caption") or "").strip()
                if cap: texts.append(cap)
                # no auto-analyze: vision_agent_node will describe the image;
                # the model acknowledges receipt and waits for user instructions.
                # Unless 📷 Analyze Photo was pressed first — that arms an
                # instruction for exactly this photo. A caption wins over it.
                elif sess.pending_photo:
                    texts.append(sess.pending_photo)
                    sess.pending_photo = ""
                    self._store.put(sess)

            elif t == "album":
                for fid in item.get("file_ids", []):
                    data = self._dl_bytes(fid)
                    if not data:
                        continue
                    if img_bytes is None:
                        img_bytes = data; _photo_upload = True
                    else:
                        _album_extra_bytes.append(data)
                cap = (item.get("caption") or "").strip()
                if cap: texts.append(cap)
                elif sess.pending_photo:
                    texts.append(sess.pending_photo)
                    sess.pending_photo = ""
                    self._store.put(sess)

            elif t == "document":
                fname = item.get("filename", "file")
                cap = (item.get("caption") or "").strip()
                # Bot API getFile stops at 20 MB: name the limit before trying
                # instead of answering "Could not read X".
                _size = int(item.get("file_size") or 0)
                if _size > _bot_file_limit(getattr(self, "tg_api_local", None)):
                    self._send_text(chat_id,
                        tg_bot._t("doc_too_big", lang, name=_html_mod.escape(fname),
                                  mb=f"{_size / (1024 * 1024):.0f}",
                                  limit=f"{_bot_file_limit(getattr(self, "tg_api_local", None)) // (1024 * 1024)}"),
                        parse_mode="HTML",
                        keyboard=self._main_menu_kb(sess, lang))
                    self._activity.log(chat_id, "system",
                                       f"[document] {fname} {_size} bytes rejected: over the 20 MB Bot API limit",
                                       (user.name if user else str(chat_id)))
                    return
                data = self._dl_bytes(item["file_id"])
                # A download failure or an empty file must always be reported —
                # even with a caption attached; the `elif cap:` arm below would
                # queue the caption alone and let the model answer as if it had
                # read a file that never arrived.
                if not data:
                    self._send_text(chat_id,
                        tg_bot._t("doc_unreadable", lang, name=_html_mod.escape(fname)),
                        parse_mode="HTML",
                        keyboard=self._main_menu_kb(sess, lang)); return

                # If this user has a sandbox, the raw bytes go into it as well.
                # ADDITIONALLY, not instead: a PDF still gets indexed and stays
                # answerable. But a .jar or a .zip is not extractable text at
                # all, so without this the assistant could only ask the user to
                # paste its source code.
                _saved_as = ""
                file_in_folder = True   # the instruction is about THIS file
                try:
                    import sandbox_access as _sa
                    if _sa.may_use_files(self._user_store.get(chat_id)):
                        import code_sandbox as _cs
                        _box = _cs.sandbox_for(chat_id)
                        # The FILENAME is attacker-controlled too: Telegram will
                        # happily carry "../../evil.txt". resolve() is the same
                        # check every other path goes through.
                        _safe = Path(fname).name or "upload.bin"
                        _target = _box.resolve(_safe)
                        _target.write_bytes(data)
                        _saved_as = _box.relative(_target)
                        # say where it went
                        self._send_text(chat_id, tg_bot._t(
                            "doc_in_sandbox", lang, name=_html_mod.escape(_saved_as)),
                            parse_mode="HTML")
                except Exception:
                    tg_bot.logger.exception("could not store %s in the sandbox", fname)
                # A picture sent "as a file" (uncompressed) is still a picture
                # (receipt.jpg must not get «текст не удалось извлечь»). Without a
                # working folder it goes the photo way.
                if (not _saved_as and img_bytes is None
                        and Path(fname).suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                    img_bytes = data; _photo_upload = True
                    if cap: texts.append(cap)
                    continue
                if _saved_as:
                    # Named in ENGLISH and addressed to the model, not the user:
                    # this is scaffolding on the turn, and the reasoning side of
                    # the pipeline works in English.
                    #
                    # The tool named has to MATCH the file: read_file refuses an
                    # image, so a .png needs open_image instead.
                    import tool_code_handlers as _tch
                    _hint = ("open_image, then inspect_image"
                             if Path(_saved_as).suffix.lower()
                             in _tch.IMAGE_SUFFIXES
                             else "list_files, unpack_archive, read_file")
                    scaffold.append(f"[The file '{_saved_as}' is now in your "
                                    f"working folder. Open it yourself with: "
                                    f"{_hint}. Do not ask the user to send its "
                                    f"contents.]")
                doc_text = ""
                tmp = ""
                if data:
                    with tempfile.NamedTemporaryFile(
                            suffix=Path(fname).suffix, delete=False) as fh:
                        fh.write(data); tmp = fh.name
                    try:
                        doc_text = (self.extract_doc or tg_bot._extract_doc)(tmp, fname)
                        # A book pasted into the prompt would be truncated and the
                        # rest silently thrown away, so anything substantial goes
                        # into the user's own hybrid index and questions are
                        # answered by retrieval. Small files are STILL inlined when
                        # the user sent an instruction with them, so asking about a
                        # short file in the same message keeps working.
                        inline_cap = tg_bot._cfg_int("TG_DOC_INLINE_CHARS", 6000)
                        _will_inline = bool(cap) and len(doc_text) <= inline_cap \
                                       and Path(fname).suffix.lower() != ".epub"
                        # Indexing is by the button only: inside 📚 Documents, or
                        # when there is no working folder for the file to live in.
                        # Otherwise it goes to the sandbox and the agent reads it
                        # there (indexing a crash log would stall on embeddings).
                        _index = sess.menu == "library" or not _saved_as
                        if _index:
                            self._index_document(chat_id, sess, tmp, fname,
                                                 sandboxed=bool(_saved_as),
                                                 quiet=_will_inline)
                        if not _index and len(doc_text) > inline_cap:
                            # too big to inline and not indexed: the file is read
                            # from the sandbox (the note above already says so)
                            if cap:
                                texts.append(cap)
                            doc_text = ""
                        elif len(doc_text) > inline_cap \
                                or Path(fname).suffix.lower() == ".epub" or not cap:
                            if cap:
                                texts.append(cap)
                                # The caption is a question about THIS file:
                                # retrieval on «Что думаешь?» alone would miss it,
                                # so say which document the words are about.
                                scaffold.append(
                                    f"[The user attached the document '{fname}' "
                                    f"({len(doc_text)} characters) to this message; "
                                    f"the caption is about that document. Its "
                                    f"passages are retrieved below when relevant"
                                    + (f"; the whole file is also in your working "
                                       f"folder as '{_saved_as}' -- read_file it "
                                       f"when the passages are not enough." if _saved_as else ".")
                                    + "]")
                            else:
                                continue
                            doc_text = ""
                    finally:
                        try:
                            if tmp and os.path.exists(tmp): os.unlink(tmp)
                        except Exception: pass
                if doc_text:
                    from prompt_guard import wrap_document
                    import injection_scan
                    doc_text = injection_scan.scrub(doc_text, fname)
                    texts.append(f"{cap}\n\n{wrap_document(fname, doc_text)}" if cap
                                 else wrap_document(fname, doc_text))
                elif cap:
                    if cap not in texts:
                        texts.append(cap)
                    # «открой» under junk.bin must not be answered «какой файл?» --
                    # the model has to learn a file came.
                    if not any(fname in s for s in scaffold):
                        scaffold.append(
                            f"[The user attached '{fname}', but its text could not be "
                            f"extracted ({'the file is empty' if not data else 'this format is not supported: text, PDF, DOCX, EPUB, ZIP and code files are'}). "
                            f"Say so plainly; do not ask which file.]")
                elif not texts:
                    self._send_text(chat_id,
                        tg_bot._t("doc_unreadable", lang, name=_html_mod.escape(fname)),
                        parse_mode="HTML",
                        keyboard=self._main_menu_kb(sess, lang)); return

        merged = "\n".join(texts).strip()
        if not merged and not scaffold and img_bytes is None:
            return
        # A batch of files is ONE material: three notes would make the agent
        # read the first log and answer about it alone.
        _fnotes = [s for s in scaffold if s.startswith("[The file '")]
        if len(_fnotes) > 1:
            _names = [re.match(r"\[The file '([^']+)'", s).group(1) for s in _fnotes]
            scaffold = [s for s in scaffold if s not in _fnotes]
            scaffold.insert(0, "[The file '" + "', '".join(_names) + "' are now in your working "
                            f"folder ({len(_names)} files sent together). Read ALL of them "
                            "(read_file each) and answer once, about them as a whole -- "
                            "not file by file. Do not ask the user to send their contents.]")
        # Files with no words: «📁 Положил X в песочницу» is the answer; a turn on
        # our own note alone would claim «я всё изучил» without opening anything.
        if (not merged and img_bytes is None and scaffold
                and all(s.startswith("[The file '") for s in scaffold)):
            return
        if merged and img_bytes is None and not any(s.startswith("[The file '") for s in scaffold):
            _latest = self._latest_sandbox_file(chat_id, sess)
            if _latest:
                # «В ЧЕМ ТАМ ОШИБКА?» after forwarded logs must not critique the last
                # picture: the last thing sent was the file
                scaffold.append(f"[The last thing the user sent was the file '{_latest}' "
                                f"(in your working folder, after any picture). A question "
                                f"about «it», «there» or «the error» means that file: "
                                f"read_file it before answering.]")
        # Which picture the user means is read from the USER's words only.
        from prompt_guard import user_words, wrap_quoted
        own_words = user_words(merged)

        # Apply the pending prompt prefix here — AFTER every modality has been merged.
        # Applying it only in the text branch would drop it when a prompt button
        # ("🔬 Deep Research" -> "type your query") is answered with a VOICE
        # message or a photo caption, and the stale prefix would leak onto the
        # next text message. Nav/direct buttons return or clear the prefix before
        # reaching this point, so they still never consume it.
        # ...but only onto something the USER said: a .jar uploaded with no
        # caption yields a turn whose entire text is our own "[The file ... is now
        # in your working folder]" note, and a draw prefix glued to it would
        # answer a file upload by generating a picture. A prompt prefix is an
        # instruction about the user's next words; with none to attach to it must
        # WAIT rather than seize our scaffolding.
        if merged and sess.pending_prefix:
            # «привет» typed in the Озон menu searched shampoo (2026-10-27): an
            # armed prefix may only seize a real request for its kind of work.
            # Plain conversation -- the intent read says needs_tool False --
            # answers as chat, and the prefix stays armed for the next real
            # query («носки» right after that «привет» still searches Ozon).
            from seams import intent_read
            try:
                chatty = intent_read(self.reads)(None, merged, "", "")["needs_tool"] is False
            except Exception:
                chatty = False
            if not chatty:
                merged = sess.pending_prefix + merged
                sess.pending_prefix = ""
                sess.pending_outfit_target = ""   # 👗 answered in words, not a photo
                try: self._store.put(sess)
                except Exception:
                    tg_bot.logger.warning("session persist failed", exc_info=True)

        _age = time.time() - (getattr(self, "_msg_date", {}).get(chat_id) or time.time())
        if _age > 600:
            # A backlog after downtime reads as "now": «ты спишь?» from yesterday must not get «я не сплю».
            scaffold.append("[This message was written %s ago; you were switched off and are only "
                            "answering it now. Do not claim you were awake or online then; say briefly "
                            "that you were offline.]" % (
                                "%d h" % (_age // 3600) if _age >= 3600 else "%d min" % (_age // 60)))
        # Appended after the prefix decision, so it can never be prefixed.
        if scaffold:
            joined = "\n".join(scaffold)
            merged = (merged + "\n" + joined).strip() if merged else joined

        # A replied-to TEXT message becomes explicit context for this turn, in the
        # user's own words, rather than being lost. Consumed once — a quote is a
        # gesture about this message, not a mode.
        _quote = getattr(sess, "quoted_text", "")
        if _quote and merged:
            merged = wrap_quoted("the user is replying to this message", _quote) + "\n" + merged
        if _quote:
            sess.quoted_text = ""
            try: self._store.put(sess)
            except Exception:
                tg_bot.logger.warning("session persist failed", exc_info=True)

        # Photo sent without caption — just acknowledge; vision_agent_node handles it.
        # Deliberately after the prefix block: a caption-less photo must NOT consume
        # a pending "edit the image: " prefix, which is waiting for the next message.
        if not merged and img_bytes is not None:
            merged = "image"

        # ── extremism filter ──────────────────────────────────────────────────
        if tg_bot._check_extremism(merged):
            uname = user.name if user else str(chat_id)
            tg_bot.logger.warning("Extremism filter triggered for chat=%s text=%s",
                           chat_id, merged[:80])
            self._activity.log(chat_id, "error",
                               f"[BLOCKED extremism] {merged[:200]}", uname)
            self._send_text(chat_id, tg_bot._t("extremism", lang), parse_mode="HTML",
                            keyboard=self._main_menu_kb(sess, lang))
            return

        # ── the same job, twice ───────────────────────────────────────────────
        # A repeat tap OUTSIDE the debounce window becomes a second task, and an
        # image job costs a minute of the one GPU. The in-batch collapse above
        # cannot see this one, because by now the first task is already running.
        # Only machine-generated payloads are compared: those are byte-identical
        # by construction, so a match really is the same button pressed twice —
        # whereas two people typing the same sentence a minute apart may well
        # mean it.
        if merged in tg_bot._MACHINE_PAYLOADS:
            with self._task_lock:
                running = list(self._running_task.get(chat_id) or [])
            # The verb string alone is not enough: all six image buttons say
            # "on the current image" regardless of WHICH picture was tapped, so
            # the target image id is compared too. Non-image machine payloads
            # carry no id on either side and degrade to a text-only compare. A
            # chat may have 2 tasks in flight (a slow one plus an admitted
            # interject); a dupe of EITHER counts.
            if any(r.user_text == merged and r.image_id == pressed_image_id
                   for r in running):
                self._send_text(chat_id, tg_bot._t("already_running", lang),
                                parse_mode="HTML",
                                keyboard=self._state_kb(sess, lang))
                self._activity.log(chat_id, "system",
                                   "[dedupe] identical action already running",
                                   user.name if user else str(chat_id))
                return

        # ── fairness: cap how much one user may have waiting ──────────────────
        # Round-robin already stops a burst from starving others, but nothing
        # stopped one user queueing ten deep-research runs and occupying the GPU
        # for the rest of the day.
        is_admin = bool(user and user.is_admin)
        max_queued = max(1, tg_bot._cfg_int("TG_MAX_QUEUED_PER_USER", 3))
        if not is_admin:
            try:
                mine = self._backend.chat_depth(chat_id)
            except Exception:
                mine = 0
            if mine >= max_queued:
                self._send_text(chat_id, tg_bot._t("too_many", lang, n=mine),
                                keyboard=self._main_menu_kb(sess, lang))
                return

        # ── daily quota ───────────────────────────────────────────────────────
        kind = tg_bot._classify_task(merged)
        if not is_admin:
            used = self._user_store.usage_today(chat_id)
            for check in ({kind} | {tg_bot.KIND_TASK}):
                limit = tg_bot._quota_limit(check, self.settings)
                if limit > 0 and used.get(check, 0) >= limit:
                    self._send_text(chat_id,
                        tg_bot._t("quota_hit", lang, kind=tg_bot._kind_label(check, lang),
                           used=used.get(check, 0), limit=limit),
                        parse_mode="HTML",
                        keyboard=self._main_menu_kb(sess, lang))
                    self._activity.log(chat_id, "system",
                                       f"[quota] {check} limit {limit} reached",
                                       user.name if user else str(chat_id))
                    return

        # ── log the user message ──────────────────────────────────────────────
        uname = user.name if user else str(chat_id)
        self._activity.log(chat_id, "user_msg", merged, uname)

        # ── ask instead of guessing ───────────────────────────────────────────
        # An instruction that plainly needs a picture, with no picture attached,
        # nothing pointed at, and several to choose from. Guessing "the newest"
        # would answer a question about one image about another. Ask once; the
        # answer arms the target and re-pushes this same instruction, so nothing
        # is lost.
        # A BUTTON press is exempt: it was pressed on a specific picture, so either
        # it resolved to one above, or the picture is simply the current one.
        # A DOCUMENT in the same message is the thing the instruction is about
        # («убери дубликаты» + DCIM.zip must not get «Какую картинку?»).
        if not any('_target_captured' in it for it in batch):
            pressed_image_id = pressed_image_id or sess.target_image
        if (not img_bytes and not file_in_folder
                and not pressed_image_id
                and merged not in tg_bot._MACHINE_PAYLOADS):
            _named = tg_bot._ordinal_target(own_words, tg_bot._live_images(sess), ctx=self.reads)
            if _named:
                pressed_image_id = _named
                try: self._store.put(sess)
                except Exception:
                    tg_bot.logger.warning("session persist failed", exc_info=True)
                self._activity.log(chat_id, "system", f"[target] named picture {_named}", uname)
        # Quoted material (✍️ after a forward, a reply to a message) is what the words are
        # about: «Что думаешь?» under it got «Какую картинку?» (2026-10-22).
        from prompt_guard import has_quote
        if (not img_bytes and not file_in_folder
                and not pressed_image_id
                and merged not in tg_bot._MACHINE_PAYLOADS
                and not has_quote(merged)
                and len(tg_bot._live_images(sess)) > 1
                and tg_bot._needs_image_choice(own_words, chatlog.recent(chat_id), ctx=self.reads,
                                               in_play=getattr(sess, "turn_image", ""))):
            live = tg_bot._live_images(sess)
            if tg_bot._one_lineage(live):
                # One picture in several states -- the latest is "the picture".
                pressed_image_id = live[-1]["id"]
                try: self._store.put(sess)
                except Exception:
                    tg_bot.logger.warning("session persist failed", exc_info=True)
                self._activity.log(chat_id, "system",
                                   f"[target] one lineage of {len(live)} -> latest {live[-1]['id']}", uname)
            else:
                sess.pending_instruction = merged
                try: self._store.put(sess)
                except Exception:
                    tg_bot.logger.warning("session persist failed", exc_info=True)
                self._send_text(chat_id, tg_bot._t("img_which", lang, n=len(live)),
                                parse_mode="HTML",
                                keyboard=tg_bot._image_choice_kb(sess, lang))
                self._activity.log(chat_id, "system",
                                   f"[target] asked which of {len(live)} images", uname)
                return

        # Save image bytes for cross-thread handoff. This path is ALSO handed
        # to _log_image below and lives on in sess.image_log as the register
        # entry a later "the one I sent earlier" resolves against — so it has
        # to be a PERSISTENT file (tg_bot._IMAGE_DIR, same directory
        # _save_incoming_photo uses for a reply-fetched photo), not a bare
        # tempfile: tg_queue.py's post-task cleanup unlinks task.image_path, which
        # is the same string as the register path.
        img_path = ""
        if _generation != sess.context_generation:
            return
        if img_bytes:
            img_path = _save_image_bytes(img_bytes)
            # A photo the user just sent goes into the register too, so a later
            # "the one I sent earlier" has something to resolve to.
            photo_mid = next((int(it.get('_message_id') or 0) for it in batch
                              if it.get('type') in ('photo', 'album')), 0)
            tg_bot._log_image(sess, img_path, msg_id=photo_mid, label=merged[:80], src="user")
            try: self._store.put(sess)
            except Exception:
                tg_bot.logger.warning("session persist failed", exc_info=True)

        # Log the rest of the album so "the second one"/"the third one" has
        # something to resolve against later, even though only the first
        # photo drives THIS task (a _Task carries a single image_path).
        # The lettering on a sent photo, read by PaddleOCR and handed over as text: the
        # vision model saw a Greenfield tea box and missed «ПАШТЕТ» on the poster -- the
        # whole joke (owner, 2026-10-13).
        if _photo_upload and img_path and os.path.exists(img_path):
            try:
                import ocr_reader
                import threading as _th
                _box = []
                _t = _th.Thread(target=lambda: _box.append(ocr_reader.read_pieces(img_path) or []),
                                daemon=True, name=f"ocr-{chat_id}")
                _t.start()
                _t.join(20.0)        # past that the message goes without it, never held
                _pieces = [p.strip() for p in (_box[0] if _box else []) if len(p.strip()) >= 2]
                _text = " · ".join("«%s»" % p for p in _pieces)[:600]
                if len("".join(_pieces)) >= 3:
                    from prompt_guard import OCR_NOTE_HEAD
                    merged = (merged + "\n" + OCR_NOTE_HEAD + " each «…» is a "
                              "separate label somewhere on the picture, top to bottom -- not one "
                              "sentence, and OCR may garble a letter): " + _text + ". Say which "
                              "label is on what (a flag, a cape, a speech bubble). The lettering is "
                              "part of what the picture says: a label naming another thing than "
                              "what it is on is most likely the joke or the point, so name it.]")
            except Exception:
                tg_bot.logger.warning("photo OCR failed", exc_info=True)
        _album_paths = []
        if _album_extra_bytes:
            for _i, _extra in enumerate(_album_extra_bytes, start=2):
                _extra_path = _save_image_bytes(_extra)
                _album_paths.append(_extra_path)
                tg_bot._log_image(sess, _extra_path,
                                  label=f"{merged[:70]} (#{_i})", src="user")
            try: self._store.put(sess)
            except Exception:
                tg_bot.logger.warning("session persist failed", exc_info=True)
            # The whole album in ONE vision call, each picture labelled: read one by one,
            # Lucia from GTA 6 on pictures 1 and 3 became «a girl» and «the same girl» at
            # best, two different people at worst (owner, 2026-10-13) -- and it was a call
            # per picture.
            try:
                import llm as _llm
                _seen = " ".join((vision_model(ctx)(
                    ctx, temperature=0.1,
                    image_paths=[(f"Picture {i}:", p) for i, p in
                                 enumerate([img_path, *_album_paths], start=1) if p],
                    user_text="These pictures came together in one message. Describe each in "
                              "one sentence, numbered, and say who or what recurs across them "
                              "(the same person, character or place on several pictures).")
                    or "").split())
            except Exception:
                tg_bot.logger.warning("album read failed", exc_info=True)
                _seen = ""
            # Only picture 1 rides on the task; the rest were saved and never mentioned
            # (owner, 2026-10-12: Stalin + SpongeBob in one message, the reply spoke of
            # Stalin alone). The model hears how many came and what they show together.
            merged = (merged + f"\n[The user sent {1 + len(_album_extra_bytes)} pictures in one "
                      "message; picture 1 is the one attached. All of them, seen together: "
                      + (_seen or "(could not be read)") + " Acknowledge every picture, not just "
                      "the first.]")

        # A turn with no letters and no digits anywhere (a bare emoji, a wall of
        # punctuation) carries nothing for any tool or the model to act on -- yet
        # a lone «🎨» reached generate_image and burned a 147 s render on
        # «волшебный лес» (2026-10-28). Ask back, exactly as the owner asked:
        # nonsense never reaches generation, search or any other tool. A photo
        # alone is exempt -- img_path/file_in_folder carry it.
        if (merged and not img_path and not file_in_folder
                and not re.search(r"[^\W_]", merged, re.UNICODE)):
            self._activity.log(chat_id, "system", "[bounce] contentless input: " + merged[:40],
                               uname)
            self._send_text(chat_id, tg_bot._t("contentless_input", lang))
            return

        selection = 'upload' if img_path else 'explicit' if pressed_image_id else 'context'
        if not img_path:
            pressed_image_id = pressed_image_id or sess.turn_image
        if (not any('_target_captured' in it for it in batch)
                and sess.target_image and (img_path or sess.target_image == pressed_image_id)):
            sess.target_image = ''
        task = tg_bot._Task(
            task_id=str(uuid.uuid4()),
            chat_id=chat_id,
            user_text=merged,
            image_path=img_path,
            image_id='' if img_path else pressed_image_id,
            image_is_photo=_photo_upload,
            album_paths=_album_paths,
            enqueue_ts=_arrival_ts,
            reply_to=_reply_to,
            context_generation=_generation,
            image_selection=selection,
        )
        # last_task_id/last_task_text are set at DELIVERY of a failure (in
        # tg_tasks.py _remember_failed_task), not here at push time.
        # Forwarded material alone is not a note to the running task: the bot's
        # own replies forwarded mid-song would go in as wishes.
        _only_fwd = bool(batch) and all(it.get("forwarded") for it in batch)
        if not _only_fwd and self._try_steer(chat_id, task):
            return          # a note to the running task, not a new task: not charged
        # A forward that lands while this chat's previous turn still WAITS (a late
        # part of the same paste, a getUpdates gap) joins it: one answer, not two.
        _joined = any(it.get("forwarded") for it in batch) and self._join_waiting(chat_id, task)
        if not _joined:
            # Charged only now: a steer note or an «which picture?» question is not a task.
            self._user_store.bump_usage(chat_id, tg_bot.KIND_TASK)
            if kind != tg_bot.KIND_TASK:
                self._user_store.bump_usage(chat_id, kind)
        if not self._persist_task(task):
            return
        tg_bot.logger.info("Enqueued task %s chat=%s kind=%s depth=%d",
                    task.task_id[:8], chat_id, kind, self._backend.depth())

        self._queue_note(chat_id, task, lang)

    def _card_busy(self) -> str:
        """What render holds the whole card, "" when it is shared."""
        try:
            import comfy_client
            return comfy_client.card_is_exclusive()
        except Exception:
            return ""

    def _queue_note(self, chat_id: int, task, lang: str) -> None:
        """⏳ for a task that will not start at once: its place among the tasks
        AHEAD OF THIS USER under round-robin (not the global depth), or, with
        nobody ahead, a render holding the whole card -- a photo sent during
        another chat's song sat 55 s with no word."""
        try:
            ahead = self._backend.tasks_ahead(task.task_id)
        except Exception:
            ahead = []
        self._busy_note(chat_id, lang, ahead)

    def _busy_note(self, chat_id: int, lang: str, ahead=()) -> None:
        """The one place that tells a user their request waits. Every entry that starts work
        (a typed task, the plan's ▶ button, a character render) calls it the moment it is
        taken, not after the debounce and the intent read: those wait on the same busy model,
        and a ▶ that answers nothing for minutes reads as a dead button. The chat keeps ONE
        such note: a later call edits it in place and the task's START deletes it -- a second
        «Ты #2 в очереди» under the first, both left hanging after the wait, read as noise
        (owner, 2026-10-22)."""
        with self._task_lock:
            rendering = sum(1 for cid, ts in self._running_task.items()
                            if cid != chat_id and self._chat_interruptible.get(cid, False)
                            and any(not getattr(t, "delivered", False) for t in ts))
            if not (ahead or rendering or self._card_busy()):
                return
        if ahead:
            text = tg_bot._t("queue_pos", lang, pos=len(ahead) + rendering + 1, eta=tg_bot._fmt_eta(ahead, lang))
        elif rendering:
            text = tg_bot._t("queue_run", lang, pos=rendering + 1, n=rendering)
        else:
            text = tg_bot._t("card_busy", lang)
        notes = self.__dict__.setdefault("_busy_notes", {})
        with self._task_lock:
            msg_id, shown = notes.get(chat_id, (None, ""))
        if msg_id and shown == text:
            return
        if msg_id:
            self._edit_text(chat_id, msg_id, text, parse_mode="HTML")
        else:
            msg_id = self._send_get_id(chat_id, text, parse_mode="HTML")
        with self._task_lock:
            notes[chat_id] = (msg_id, text)

    def _drop_busy_note(self, chat_id: int) -> None:
        """The wait is over: its note leaves the chat."""
        with self._task_lock:
            msg_id, _ = self.__dict__.setdefault("_busy_notes", {}).pop(chat_id, (None, ""))
        if msg_id:
            try:
                self._delete(chat_id, msg_id)
            except Exception:
                tg_bot.logger.debug("queue note delete failed", exc_info=True)

    def _register_animate_photo(self, chat_id: int, sess, lang: str, item: dict) -> None:
        """Consume the photo awaiting_animate_photo was waiting for, register
        it as a normal image (so generate_video's use_current_images sees
        it), and offer the motion presets + custom option. Mirrors
        _push_style_transfer_task's download/registration, but this photo
        IS the target -- there is no separate reference image -- so it ends
        in an inline keyboard, not an immediate task push."""
        sess.awaiting_animate_photo = False
        file_id = (item.get("file_id") if item.get("type") == "photo"
                   else next(iter(item.get("file_ids") or []), None))
        data = self._dl_bytes(file_id) if file_id else None
        if not data:
            self._store.put(sess)
            self._send_text(chat_id, tg_bot._t("img_gone", lang))
            return
        img_path = _save_image_bytes(data)
        image_id = tg_bot._log_image(sess, img_path,
                                     label=tg_bot._t("to_animate_label", lang), src="user")
        sess.pending_animate_target = image_id
        # Also the chat's "current image": the custom-motion prefix falls
        # through to the ordinary text path, which attaches whatever
        # sess.target_image names -- without this, typing a custom motion
        # would animate whatever picture was current BEFORE this upload.
        sess.target_image = image_id
        self._store.put(sess)
        self._animate_ask_voices(chat_id, sess, lang)   # voice samples first, then presets

    def _register_style_photo(self, chat_id: int, sess, lang: str, item: dict) -> None:
        """Consume the photo awaiting_style_photo was waiting for (🎭 Change
        style from the Creativity menu), register it, and arm
        pending_style_target -- from here on this is EXACTLY the existing
        under-a-picture style flow (_cb_style_image's keyboard, then
        _cb_style_preset/_cb_style_custom), just entered from a different
        door."""
        import style_presets
        sess.awaiting_style_photo = False
        file_id = (item.get("file_id") if item.get("type") == "photo"
                   else next(iter(item.get("file_ids") or []), None))
        data = self._dl_bytes(file_id) if file_id else None
        if not data:
            self._store.put(sess)
            self._send_text(chat_id, tg_bot._t("img_gone", lang))
            return
        img_path = _save_image_bytes(data)
        image_id = tg_bot._log_image(sess, img_path,
                                     label=tg_bot._t("to_restyle_label", lang), src="user")
        sess.pending_style_target = image_id
        sess.target_image = image_id
        self._store.put(sess)
        rows = []
        for i in range(0, len(style_presets.STYLE_PRESET_ORDER), 2):
            row = []
            for key in style_presets.STYLE_PRESET_ORDER[i:i + 2]:
                row.append({"text": style_presets.preset_label(key, lang),
                            "callback_data": tg_bot._with_img(f"style_preset:{key}", image_id)})
            rows.append(row)
        rows.append([{"text": tg_bot._t("style_custom_btn", lang),
                      "callback_data": tg_bot._with_img("style_custom", image_id)}])
        self._send_text(chat_id, tg_bot._t("style_ask_ref", lang),
                        keyboard={"inline_keyboard": rows})

    def _run_swap(self, chat_id: int, sess, lang: str, item: dict, wish: str) -> None:
        """🔁 the photo of who goes in: the picked person's region is redrawn and pasted back
        (person_swap.swap), on its own thread under the card lock the edit takes itself.
        Not through the agent: its transfer_image re-rendered and coloured the whole photo,
        kept the man and answered «done» (live 2026-10-12)."""
        import threading
        person_swap = module(self, "person_swap")
        target = tg_bot._image_by_id(sess, sess.pending_swap_target)
        box = tuple(sess.swap_boxes[0]) if sess.swap_boxes else None
        sess.pending_swap_target, sess.swap_boxes, sess.swap_where = "", [], ""
        sess.pending_prefix = ""
        self._store.put(sess)
        file_id = (item.get("file_id") if item.get("type") == "photo"
                   else next(iter(item.get("file_ids") or []), None))
        data = self._dl_bytes(file_id) if file_id else None
        if not (target and box and os.path.exists(target.get("path", "")) and data):
            self._send_text(chat_id, tg_bot._t("img_gone", lang))
            return
        ref = _save_image_bytes(data)
        self._send_text(chat_id, tg_bot._t("swap_working", lang))
        ctx = self._get_ctx()

        def _go():
            try:
                out = person_swap.swap(ctx, target["path"], box, ref, wish)
            except Exception:
                tg_bot.logger.exception("person swap failed chat=%s", chat_id)
                out = ""
            if not out:
                self._send_text(chat_id, tg_bot._t("swap_failed", lang))
                return
            s2 = self._get_session(chat_id)
            img_id = tg_bot._log_image(s2, out, label="swap", src="bot")
            s2.target_image = img_id
            self._store.put(s2)
            self._send_photo(chat_id, out, keyboard=tg_bot._image_kb(lang, img_id))
        threading.Thread(target=_go, name=f"swap-{chat_id}", daemon=True).start()

    def _push_style_transfer_task(self, chat_id: int, sess, lang: str,
                                   item: dict, arrival_ts: float, outfit_wish=None) -> None:
        """Consume the reference photo pending_style_target was waiting for
        and push a dedicated transfer_image task with BOTH images attached.

        Bypasses the normal img_bytes/texts merge entirely: this photo is
        never the new "current picture", it is a second, secondary image
        riding alongside the target in _Task.style_ref_path -- tg_tasks.py's
        _execute_task reads it and populates ctx.reference_images, which is
        the only thing transfer_image actually looks at (see
        tool_image_handlers._handle_transfer_image). Telegram had no path to
        that field at all before this: reference_images was populated only by
        the desktop GUI's Transfer tab.
        """
        # This path skips _resolve_and_push's quota gate: a restyle by
        # reference is an image job and is counted like one.
        user = self._user_store.get(chat_id)
        if not (user and user.is_admin):
            used = self._user_store.usage_today(chat_id)
            for check in (tg_bot.KIND_IMAGE, tg_bot.KIND_TASK):
                limit = tg_bot._quota_limit(check, self.settings)
                if limit > 0 and used.get(check, 0) >= limit:
                    self._send_text(chat_id, tg_bot._t("quota_hit", lang, kind=tg_bot._kind_label(check, lang),
                                    used=used.get(check, 0), limit=limit),
                                    parse_mode="HTML", keyboard=self._main_menu_kb(sess, lang))
                    return
        # outfit_wish (str, may be "") = the 👗 flow: same two-image transfer, the
        # reference is the clothes and the caption says what else to do.
        outfit = outfit_wish is not None
        target_id = sess.pending_outfit_target if outfit else sess.pending_style_target
        if outfit:
            sess.pending_outfit_target = ""
            sess.pending_prefix = ""
        else:
            sess.pending_style_target = ""
        target = tg_bot._image_by_id(sess, target_id)
        if not target or not os.path.exists(target.get("path", "")):
            self._store.put(sess)
            self._send_text(chat_id, tg_bot._t("img_gone", lang))
            return
        file_id = (item.get("file_id") if item.get("type") == "photo"
                   else next(iter(item.get("file_ids") or []), None))
        data = self._dl_bytes(file_id) if file_id else None
        if not data:
            self._store.put(sess)
            self._send_text(chat_id, tg_bot._t("img_gone", lang))
            return
        ref_path = _save_image_bytes(data)
        tg_bot._log_image(sess, ref_path, label="outfit reference" if outfit else "style reference",
                          src="user")
        self._store.put(sess)

        task = tg_bot._Task(
            task_id=str(uuid.uuid4()),
            chat_id=chat_id,
            user_text=(_outfit_text(outfit_wish) if outfit else
                "[style] call transfer_image with instructions=\"Change the visual "
                "style of the target image to match the reference image: lighting, "
                "colour palette, contrast, texture, atmosphere and photographic "
                "character. Preserve the exact subject identity, pose, composition "
                "and framing of the target -- do not replace the subject with "
                "anything from the reference.\" using the two loaded images (the "
                "target to edit is the earlier one, the style reference is the one "
                "just sent)."),
            image_path=target["path"],
            image_id=target_id,
            style_ref_path=ref_path,
            enqueue_ts=arrival_ts,
            reply_to=int(item.get('_message_id') or 0),
            context_generation=item.get('_context_generation', sess.context_generation),
            image_selection='explicit',
        )
        if self._try_steer(chat_id, task):
            return
        self._user_store.bump_usage(chat_id, tg_bot.KIND_TASK)
        self._user_store.bump_usage(chat_id, tg_bot.KIND_IMAGE)
        if not self._persist_task(task):
            return
        tg_bot.logger.info("Enqueued task %s chat=%s kind=%s depth=%d",
                    task.task_id[:8], chat_id, "outfit" if outfit else "style", self._backend.depth())
        self._queue_note(chat_id, task, lang)


def _outfit_text(wish: str) -> str:
    """👗 + a photo of clothes: dress the earlier picture's person in them.
    The wish is the user's caption, quoted as theirs."""
    from prompt_guard import wrap_quoted
    extra = ("\n" + wrap_quoted("the user's wishes for the outfit", wish)) if wish else ""
    return ("[outfit] call transfer_image with instructions=\"Dress the person in the target "
            "image in the clothing shown in the reference image (the garment itself: cut, "
            "colour, fabric, print). Remove their current outfit. Keep the person's face, "
            "identity, body, pose, background and framing exactly.\" using the two loaded "
            "images (the target to dress is the earlier one, the clothing reference is the "
            "one just sent)." + (" Follow the user's wishes below in the instructions too."
                                 if wish else "") + extra)


import tg_bot  # noqa: E402  (cycle by design; attrs read at call time)
import chatlog  # noqa: E402
import tg_songs as _tg_songs  # noqa: E402
import tg_weather as _tg_weather  # noqa: E402
import tg_links as _tg_links  # noqa: E402
import tg_music as _tg_music  # noqa: E402

"""Inline-keyboard callbacks: what happens when a user taps a button.

Split out of tg_dispatch.py, where this was the 680-line ``if cb:`` arm of
_dispatch and dwarfed the message path it shared a function with. The two
halves of that router answer different update shapes (``callback_query`` vs
``message``) and share nothing but the chat id, so they are now two functions
in two files; _dispatch still owns routing and hands a callback straight here.

Three things run BEFORE any button is routed, and all three are load-bearing:

  1. answerCallbackQuery, so Telegram stops showing the spinner.
  2. An armed text-capture mode (feedback / rename / new password / badge /
     weather city / song topic) is abandoned. A button press is not an answer
     to "type your …", and leaving the mode armed silently filed the user's
     NEXT ordinary message as one.
  3. The same account-status gate the message path gets from _user_gate.
     A callback has no _user_gate anywhere on its route, and Telegram buttons
     never expire, so without this a banned or logged-out account's old
     keyboard still worked.

Handlers are ordinary methods returning None; each is reached from the prefix
table in _dispatch_callback and is responsible for its own reply. Admin and
"forget all" buttons authorize the PRESSER (``callback_query.from``), never
the chat the button sits in — see the notes at those handlers.

Same import convention as the rest of the tg_* family: tg_bot is imported at
the BOTTOM and read as ``tg_bot.<name>`` at call time, never by value, so
redirect_data_dir() and the test seams that rebind those attributes are seen.
"""
from __future__ import annotations

import html as _html_mod
import os
import time
import uuid

import config as _config
import style_presets


class CallbackMixin:
    def _dispatch_callback(self, cb: dict) -> None:
        cb_id   = cb.get("id", "")
        msg     = cb.get("message") or {}
        chat_id = (msg.get("chat") or {}).get("id")
        data    = cb.get("data", "")
        if data.startswith("adm:") and chat_id:
            # Answered inside, with a toast («⛔ Останавливаю»); presser-gated there.
            return self._cb_admin_panel(chat_id, cb, data)
        self._api_post("answerCallbackQuery", {"callback_query_id": cb_id})
        if not chat_id: return

        # Both of these are unconditional preamble, not routing -- see the
        # module docstring for why each exists.
        self._cb_abandon_capture(chat_id)
        if not self._cb_status_allows(chat_id, data):
            return

        # A verb-with-id button ("upscale:ab12cd") names the picture it sits
        # under. Resolving it can also FAIL (the file is gone), which ends the
        # press -- hence the third element rather than an exception.
        data, _pressed_image_id, _ok = self._cb_retarget_image(chat_id, msg, data)
        if not _ok:
            return

        # One line per button family, in the order they are tested. Each
        # handler owns its own reply; none of them fall through.
        if data == "wait:cancel": return self._cb_wait_cancel(chat_id)
        if data.startswith("busy:cancel:"): return self._cb_busy_cancel(chat_id, data)
        if data == "nav:close": return self._close_menu(chat_id, msg)
        if data.startswith("fwdv:"): return self._cb_forwarded_voice(chat_id, data)
        if data.startswith("pick:"): return self._cb_pick_image(chat_id, data)
        if data.startswith("sbx:"): return self._cb_sandbox(chat_id, msg, data)
        if data.startswith("cancel:"): return self._cb_cancel_request(chat_id, data)
        if data == "retry" or data.startswith("retry:"):
            return self._cb_retry(chat_id, data)
        if data.startswith("lang:"): return self._cb_set_language(chat_id, data)
        if data == "nav:back": return self._cb_nav_back(chat_id)
        if data.startswith("size:"): return self._cb_image_size(chat_id, msg, data)
        if data.startswith("reply:"): return self._cb_reply_mode(chat_id, msg, data)
        if data.startswith("music:"): return self._cb_song_settings(chat_id, msg, data)
        if data == "cover:keep": return self._cb_cover_keep(chat_id, data)
        if data.startswith("video:"): return self._cb_video_settings(chat_id, msg, data)
        if data.startswith("lv:"): return self._cb_long_video(chat_id, data)
        if data.startswith("depth:"): return self._cb_research_depth(chat_id, msg, data)
        if data.startswith("char:"): return self._cb_pick_character(chat_id, data)
        if data == "lora_collect:start": return self._cb_start_lora_collect(chat_id)
        if data in ("facts_clear", "facts_clear_confirm", "facts_clear_cancel"):
            return self._cb_forget_facts(chat_id, cb, data)
        if data == "wtw_default": return self._cb_weather_default(chat_id)
        if data == "wtw_range:48": return self._cb_weather_48h(chat_id)
        if data.startswith("wtwd:"): return self._cb_weather_day(chat_id, data)
        if data == "wtw_pickdate": return self._cb_weather_pickdate(chat_id)
        if (data.startswith("admin_approve:") or data.startswith("admin_reject:")
                or data.startswith("bcast:")):
            return self._cb_admin_action(chat_id, cb, data)
        if data.startswith("acct_"): return self._cb_account(chat_id, data)
        if data == "edit": return self._cb_edit_image(chat_id)
        if data == "ask": return self._cb_ask_image(chat_id)
        if data == "change_clothes": return self._cb_change_clothes(chat_id)
        if data == "remove_object": return self._cb_remove_object(chat_id)
        if data == "style": return self._cb_style_image(chat_id)
        if data.startswith("style_preset:"): return self._cb_style_preset(chat_id, data)
        if data == "style_custom": return self._cb_style_custom(chat_id)
        if data == "animate": return self._cb_animate_image(chat_id)
        if data.startswith("animate_preset:"): return self._cb_animate_preset(chat_id, data)
        if data == "animate_custom": return self._cb_animate_custom(chat_id)
        if data.startswith("anv:"): return self._cb_anim_voices(chat_id, data)
        if data.startswith("bk:"): return self._cb_book(chat_id, data)
        if data.startswith("song_lyr:"): return self._cb_song_lyrics(chat_id, data)
        if data.startswith("lyr:"): return self._cb_lyrics(chat_id, data)
        if data.startswith("myv:"): return self._cb_my_voice(chat_id, data)
        if data.startswith("vl:"): return self._cb_voice_library(chat_id, data)
        cmd = tg_bot._CB_CMDS.get(data)
        if cmd:
            # image_id travels WITH the item (and from there into the
            # _Task) instead of being re-read from sess.target_image later —
            # see the "captured here, at PRESS time" note above.
            self._enqueue_item(chat_id,
                               {"type": "text", "text": cmd,
                                "image_id": _pressed_image_id})
        return

    def _cb_wait_cancel(self, chat_id: int) -> None:
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        sess.drop_waiting()
        self._store.put(sess)
        self._send_text(chat_id, tg_bot._t("cancelled", lang),
                        keyboard=self._main_menu_kb(sess, lang))
        self._activity.log(chat_id, "system", "[wait] cancelled by button")

    def _cb_abandon_capture(self, chat_id: int) -> None:
        # 📝 Feedback / ✏️ Change name / 🔑 Change password all arm
        # sess.reg_state to steal the user's VERY NEXT bit of input as their
        # answer. The message-text path already abandons that mode on a
        # recognized button LABEL, a slash command, or empty text (see
        # tests/test_feedback_trap.py) — but an INLINE keyboard press (a
        # different update shape entirely: callback_query, not message) was
        # never covered by that guard. Press 📝 Feedback, change your mind
        # and tap an unrelated inline button (⚙️ a /size choice, 🌐 a
        # language pick, anything) instead of typing, and reg_state stayed
        # "feedback" forever — silently filing the user's NEXT ordinary
        # message as a feedback submission instead of routing it normally.
        # acct_setname/acct_setpwd/acct_newprofile_yes deliberately RE-ARM
        # (possibly a different) capture mode right after this, so clearing
        # here first is harmless — it is immediately overwritten by them.
        _cb_sess0 = self._get_session(chat_id)
        if _cb_sess0.reg_state in ("feedback", "change_name", "change_password",
                                   "set_badge", "wtw_city", "wtw_date", "song_topic",
                                   "char_prompt") or str(_cb_sess0.reg_state).startswith(("admin_say:", "music_custom:")):
            _cb_sess0.reg_state = ""
            self._store.put(_cb_sess0)

    def _cb_status_allows(self, chat_id: int, data: str) -> bool:
        # A callback has no message-path _user_gate call anywhere on its route,
        # so a banned/pending/logged-out account's OLD keyboard (upscale,
        # retry, lang, facts_clear, acct_setpwd, fwdv:*, ...) silently still
        # worked. `acct_setpwd` in particular could re-authenticate a
        # logged-out chat as a side effect of "changing" its password with no
        # credentials at all. Admin actions authorize the presser themselves
        # (below) and are exempt here; a stopped-by-someone-else cancel is
        # harmless and also exempt.
        if not any(data.startswith(p) for p in self._CB_SELF_AUTHORIZING):
            _cb_user = self._user_store.get(chat_id)
            _cb_sess = self._get_session(chat_id)
            _cb_lang = self._lang(_cb_sess)
            if not _cb_user or _cb_user.status == "pending":
                self._send_text(chat_id, tg_bot._t("pending", _cb_lang), parse_mode="HTML")
                return False
            if _cb_user.status in ("rejected", "banned"):
                self._send_text(chat_id, tg_bot._t("revoked", _cb_lang))
                return False
            if _cb_user.status != "approved":
                return False
            if _cb_sess.reg_state in ("awaiting_login", "awaiting_name",
                                      "awaiting_password"):
                # Logged out (or mid first-time registration) — the account
                # itself is fine, but this CHAT is locked behind a password
                # re-entry. `acct_setpwd` in particular must not be reachable
                # here: it would let a stale button set a brand-new password
                # with zero credentials and silently re-authenticate the chat
                # as a side effect.
                self._send_text(chat_id, tg_bot._t("locked_out", _cb_lang), parse_mode="HTML")
                return False
        return True

    def _cb_retarget_image(self, chat_id: int, msg: dict,
                           data: str) -> tuple:
        """(data, pressed_image_id, ok). ok=False means the press is over."""
        # The image keyboard sends "upscale:ab12cd": the verb, and the id of
        # the picture it sits under. Pressing a button under an OLD image must
        # act on THAT image — previously every button meant "the current one",
        # so scrolling up and pressing ⬆ upscaled whatever was newest instead.
        # A bare verb (a keyboard sent before this change) still works and
        # keeps the old "current image" meaning.
        # Captured here, at PRESS time, and threaded straight into the
        # enqueued item/_Task below — NOT read back out of sess.target_image
        # at execution time. A second press on a different picture while the
        # first is still queued used to overwrite that single slot before the
        # first task ever ran, so both silently resolved to whichever picture
        # was pressed last.
        _pressed_image_id = ""
        if data in tg_bot._IMAGE_VERBS:
            # A keyboard sent before ids existed. The callback still tells us
            # WHICH message it was attached to, and that message is the picture
            # itself — so an old keyboard resolves exactly, and never has to
            # fall through to "which picture did you mean?".
            _sess = self._get_session(chat_id)
            _entry = tg_bot._image_by_msg(_sess, msg.get("message_id"))
            if _entry and os.path.exists(_entry.get("path", "")):
                _sess.target_image = _entry["id"]
                _pressed_image_id = _entry["id"]
                self._store.put(_sess)
                self._activity.log(chat_id, "system",
                                   f"[button] resolved by message to {_entry['id']}")
        elif ":" in data:
            _verb, _, _sfx = data.partition(":")
            if _verb in tg_bot._IMAGE_VERBS and tg_bot._IMAGE_ID_RE.fullmatch(_sfx):
                data = _verb
                _sess = self._get_session(chat_id)
                _entry = tg_bot._image_by_id(_sess, _sfx)
                if _entry and os.path.exists(_entry.get("path", "")):
                    _sess.target_image = _sfx
                    _pressed_image_id = _sfx
                    self._store.put(_sess)
                    self._activity.log(chat_id, "system",
                                       f"[button] targets image {_sfx}")
                else:
                    # The file is gone (cleaned up, or the context was
                    # cleared). Say so instead of silently acting on a
                    # different picture, which is the bug being fixed.
                    self._send_text(chat_id,
                                    tg_bot._t("img_gone", self._lang(_sess)))
                    return data, "", False
        return data, _pressed_image_id, True

    def _cb_forwarded_voice(self, chat_id: int, data: str) -> None:
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        said = getattr(sess, "fwd_transcript", "")
        _parts = data.split(":", 2)
        _choice = _parts[1] if len(_parts) > 1 else ""
        _btn_id = _parts[2] if len(_parts) > 2 else ""
        if _choice in ("cont", "pick") or (_choice[:1] == "m" and _choice[1:].isdigit()):
            return self._cb_fwd_media(chat_id, sess, lang, _choice, _btn_id)
        # A bare "fwdv:<choice>" (no id) is a keyboard sent before ids
        # existed — honoured unconditionally, same backward-compat rule
        # the image-verb buttons use for a pre-id keyboard. One WITH an
        # id must match the currently pending forward: forwarding a
        # SECOND voice before answering the first's prompt silently
        # overwrote the single fwd_transcript slot, so the FIRST (still
        # on-screen) prompt would otherwise deliver the SECOND voice's
        # content — reproduced live. Report it the same way a picture
        # that is gone is reported, instead of silently acting on
        # whatever happens to be pending now. Now each id resolves to its own
        # voice via fwd_recent; "gone" only once it has aged out.
        if _btn_id and _btn_id != getattr(sess, "fwd_transcript_id", ""):
            said = sess.fwd_recent.get(_btn_id, "")
        if not said:
            self._send_text(chat_id, tg_bot._t("fwd_voice_gone", lang))
            return
        self._do_fwd_voice(chat_id, _choice, said)
        return

    def _cb_fwd_media(self, chat_id: int, sess, lang: str, choice: str, fwd_id: str) -> None:
        """The pictures and videos of a forwarded message: continue the video, or pick the one to work with."""
        media = (getattr(sess, "fwd_media", None) or {}).get(fwd_id or getattr(sess, "fwd_transcript_id", ""), [])
        if not media:
            self._send_text(chat_id, tg_bot._t("fwd_voice_gone", lang))
            return
        if choice == "pick":
            rows = [[{"text": m["label"], "callback_data": f"fwdv:m{i}:{fwd_id}"}] for i, m in enumerate(media)]
            self._send_text(chat_id, tg_bot._t("fwd_pick_ask", lang), keyboard={"inline_keyboard": rows})
            return
        m = media[0] if choice == "cont" else media[int(choice[1:])] if int(choice[1:]) < len(media) else None
        if not m:
            self._send_text(chat_id, tg_bot._t("fwd_voice_gone", lang))
            return
        if m["kind"] == "video":
            sess.continue_state = "want_video"
            self._continue_take_media(chat_id, sess, lang, {"video": {"file_id": m["file_id"]}})
            return
        path = self._save_incoming_photo(m["file_id"])
        if not path:
            self._send_text(chat_id, tg_bot._t("img_gone", lang))
            return
        sess.target_image = tg_bot._log_image(sess, path, label=m["label"], src="user")
        sess.turn_image = sess.target_image
        self._store.put(sess)
        self._send_text(chat_id, tg_bot._t("fwd_picked", lang, label=m["label"]))

    def _cb_pick_image(self, chat_id: int, data: str) -> None:
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        choice = data.split(":", 1)[1]
        live = tg_bot._live_images(sess)
        entry = live[-1] if choice == "latest" else tg_bot._image_by_id(sess, choice)
        if not entry or not os.path.exists(entry.get("path", "")):
            self._send_text(chat_id, tg_bot._t("img_gone", lang))
            return
        sess.target_image = entry["id"]
        held = getattr(sess, "pending_instruction", "")
        sess.pending_instruction = ""
        self._store.put(sess)
        idx = len(live) - live.index(entry) if entry in live else 1
        if held:
            # Re-push the instruction the user already typed. Making them
            # retype it would be a worse answer than the wrong guess.
            #
            # image_id travels WITH this item, exactly like every other
            # "captured at press time" carry in this router (button
            # presses, replies) — NOT left to a lazy re-read of
            # sess.target_image at task-execution time. Telegram's
            # "which picture?" keyboard never expires, so without this,
            # a re-push here that only set sess.target_image could still
            # be answered about the WRONG picture: this task queues
            # (behind whatever else is running in the chat), and if the
            # user taps that same never-expiring keyboard a second time
            # before it runs — even for an unrelated already-answered
            # prompt, out of curiosity or by accident — target_image is
            # silently overwritten and the queued task executes against
            # a picture the user picked AFTER this one was already
            # decided. Reproduced: pick B for "upscale this" (queues
            # behind a slow task), then tap the same stale keyboard for
            # A before it runs — "upscale this" ran on A, not B.
            self._resolve_and_push(chat_id, [{"type": "text", "text": held,
                                               "image_id": entry["id"]}])
        else:
            self._send_text(chat_id, tg_bot._t("img_picked", lang, i=idx))
        return

    def _cb_cancel_request(self, chat_id: int, data: str) -> None:
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        task_id = data.split(":", 1)[1]
        status = self._cancel_task(chat_id, task_id)
        # "running": the task's own unwind guard in _run_task_inner will
        # send cancel_done once it actually stops — sending it here too
        # is the double "⛔ Request cancelled." the user used to see.
        # Only acknowledge here; only "dropped"/"already" get the final
        # word from this callback, since nothing else will ever report
        # on those two outcomes.
        if status == "running":
            self._send_text(chat_id, tg_bot._t("cancel_pending", lang))
            self._on_stage(chat_id, "⛔ Cancelling…", False)
        elif status == "dropped":
            self._send_text(chat_id, tg_bot._t("cancel_done", lang),
                            keyboard=self._main_menu_kb(sess, lang))
            self._on_stage(chat_id, "⛔ Cancelled", True)
        else:
            self._send_text(chat_id, tg_bot._t("cancel_gone", lang),
                            keyboard=self._main_menu_kb(sess, lang))
            self._on_stage(chat_id, "⛔ Cancelled", True)
        return

    def _cb_retry(self, chat_id: int, data: str) -> None:
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        # The button carries the id of the request it was sent for.
        # Telegram buttons never expire, and last_task_text/last_task_id
        # are overwritten by every later push (success or failure) — so a
        # button still showing under an OLD failed message may no longer
        # match what last_task_text now holds. Only the bare pre-id form
        # (old messages sent before this fix) is honored unconditionally;
        # an id-bearing button that doesn't match the CURRENT id is stale
        # and must say so rather than silently retrying whatever the user
        # asked most recently.
        btn_id = data.split(":", 1)[1] if data.startswith("retry:") else ""
        text = (sess.last_task_text or "").strip()
        if not text or (btn_id and btn_id != getattr(sess, "last_task_id", "")):
            self._send_text(chat_id, tg_bot._t("retry_gone", lang))
            return
        self._enqueue_item(chat_id, {"type": "text", "text": text})
        return

    def _cb_set_language(self, chat_id: int, data: str) -> None:
        sess = self._get_session(chat_id)
        sess.lang = tg_bot._norm_lang(data.split(":", 1)[1])
        sess.lang_chosen = True
        sess.lang_pin = sess.turn_lang = ""          # the language chosen wins over an old pin
        self._store.put(sess)
        self._send_text(chat_id, tg_bot._t("lang_set", sess.lang), parse_mode="HTML",
                        keyboard=self._main_menu_kb(sess, sess.lang))
        return

    def _cb_reply_mode(self, chat_id: int, msg: dict, data: str) -> None:
        sess = self._get_session(chat_id)
        mode = data[6:]
        if mode not in sess.REPLY_MODES:
            return
        sess.reply_mode = mode
        self._store.put(sess)
        lang = self._lang(sess)
        msg_id = msg.get("message_id")
        if msg_id:   # the ✅ moves in place, no new message per tap
            self._edit_text(chat_id, msg_id, tg_bot._t("reply_pick", lang),
                            keyboard=tg_bot._reply_mode_kb(mode, lang))

    def _cb_nav_back(self, chat_id: int) -> None:
        """⬅ under an inline picker: the keyboard of the menu the user is in."""
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        if getattr(sess, "menu", "") == "settings":
            kb = tg_bot._settings_kb(sess.reply_mode, sess.is_admin, lang)
        else:
            kb = self._state_kb(sess, lang)
        self._send_text(chat_id, tg_bot._b("back", lang), keyboard=kb)

    def _cb_image_size(self, chat_id: int, msg: dict, data: str) -> None:
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        try:
            _, kind, value = data.split(":", 2)
        except ValueError:
            return
        if kind == "ar":
            # "auto" clears the pin so the prompt decides orientation again.
            sess.image_aspect = "" if value == "auto" else value.replace("x", ":")
            if sess.image_aspect and sess.image_aspect not in _config.IMAGE_ASPECTS:
                tg_bot.logger.warning("size callback: unknown aspect %r", value)
                sess.image_aspect = ""
        elif kind == "q":
            if value not in _config.IMAGE_QUALITIES:
                tg_bot.logger.warning("size callback: unknown quality %r", value)
                return
            sess.image_quality = value
        else:
            return
        self._store.put(sess)
        # Redraw in place: a new message per tap buries the chat, and the
        # ✅ has to move or the keyboard now lies about the setting.
        msg_id = msg.get("message_id")
        if msg_id:
            self._edit_text(chat_id, msg_id, tg_bot._size_menu_text(sess, lang),
                            parse_mode="HTML",
                            keyboard=tg_bot._size_menu_kb(sess, lang))
        else:
            self._send_text(chat_id, tg_bot._size_summary(sess, lang),
                            parse_mode="HTML")
        self._activity.log(chat_id, "system",
                           f"[size] aspect={sess.image_aspect or 'auto'} "
                           f"quality={sess.image_quality or _config.DEFAULT_IMAGE_QUALITY}")
        return

    def _cb_video_settings(self, chat_id: int, msg: dict, data: str) -> None:
        """Creativity ▸ 🎬 Video: same shape as the song settings."""
        import tg_video as _tv
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        msg_id = msg.get("message_id")
        parts = data.split(":")
        action = parts[1] if len(parts) > 1 else ""

        def _redraw(text, kb):
            if msg_id:
                self._edit_text(chat_id, msg_id, text, parse_mode="HTML", keyboard=kb)
            else:
                self._send_text(chat_id, text, parse_mode="HTML", keyboard=kb)

        if action == "reset":
            for f in _tv.FIELDS:
                setattr(sess, "video_" + f, "")
            self._store.put(sess)
            _redraw(_tv._video_menu_text(sess, lang), _tv._video_menu_kb(sess, lang))
            self._send_text(chat_id, tg_bot._t("vs_was_reset", lang)); return
        if action == "open" and len(parts) > 2 and parts[2] in _tv.FIELDS:
            _redraw(_tv._field_text(sess, parts[2], lang), _tv._field_kb(sess, parts[2], lang)); return
        if action == "set" and len(parts) > 3 and parts[2] in _tv.FIELDS \
                and parts[3] in _tv._TABLE[parts[2]]:
            setattr(sess, "video_" + parts[2], parts[3]); self._store.put(sess)
            self._activity.log(chat_id, "system", f"[video settings] {parts[2]}={parts[3]}")
        _redraw(_tv._video_menu_text(sess, lang), _tv._video_menu_kb(sess, lang))

    def _cb_song_settings(self, chat_id: int, msg: dict, data: str) -> None:
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        msg_id = msg.get("message_id")
        parts = data.split(":")
        action = parts[1] if len(parts) > 1 else ""

        def _redraw(text, kb):
            # Redraw in place for the same reason as the size and depth
            # pickers: a new message per tap buries the chat, and the ✅
            # has to move or the keyboard lies about the setting.
            if msg_id:
                self._edit_text(chat_id, msg_id, text,
                                parse_mode="HTML", keyboard=kb)
            else:
                self._send_text(chat_id, text, parse_mode="HTML", keyboard=kb)

        if action == "menu":
            _redraw(tg_bot._music_menu_text(sess, lang),
                    tg_bot._music_menu_kb(sess, lang))
            return
        if action == "reset":
            _tg_music.reset_all(sess)
            self._store.put(sess)
            _redraw(tg_bot._music_menu_text(sess, lang),
                    tg_bot._music_menu_kb(sess, lang))
            self._send_text(chat_id, tg_bot._t("ms_was_reset", lang))
            self._activity.log(chat_id, "system", "[song settings] reset")
            return
        if action == "open" and len(parts) > 2 and parts[2] in tg_bot._MUSIC_FIELDS:
            field = parts[2]
            _redraw(tg_bot._field_text(sess, field, lang),
                    tg_bot._field_kb(sess, field, lang))
            return
        if action == "set" and len(parts) > 3 and parts[2] == "steps":
            # The step count lives on the Quality screen; redraw that.
            # "auto" clears back to unchosen so the turbo LoRA's own 8-step
            # gate in music.build_workflow fires again — the only way back
            # to it from this screen once a real step count has been tapped.
            if parts[3] == "auto":
                sess.music_steps = ""
            else:
                n = _music_mod.clamp_steps(parts[3])
                sess.music_steps = str(n) if str(n) == parts[3] else ""
            self._store.put(sess)
            _redraw(tg_bot._field_text(sess, "quality", lang),
                    tg_bot._field_kb(sess, "quality", lang))
            self._activity.log(chat_id, "system", f"[song settings] steps={sess.music_steps or 'default'}")
            return
        if action == "set" and len(parts) > 3 and parts[2] in tg_bot._MUSIC_FIELDS:
            field, value = parts[2], parts[3]
            table = {"genre": tg_bot._MUSIC_GENRES,
                     "tempo": tg_bot._MUSIC_TEMPOS,
                     "vocal": tg_bot._MUSIC_VOCALS,
                     "quality": tg_bot._MUSIC_PRESETS}.get(field)
            if field == "duration":
                ok = value == "auto" or value in {str(d) for d in tg_bot._MUSIC_DURATIONS}
            elif field == "tempo":
                ok = value in table or bool(_music_mod.custom_bpm(value))
            else:
                ok = table is not None and value in table
            if not ok:
                # A stale or hand-crafted tap. Redraw against what IS
                # active rather than returning silently, so the ✅ on
                # screen stays honest instead of lying about a value
                # we just refused to store (same fix as depth:).
                tg_bot.logger.warning("music callback: bad %s=%r", field, value)
                _redraw(tg_bot._field_text(sess, field, lang),
                        tg_bot._field_kb(sess, field, lang))
                return
            # For the length "auto" is a real choice (the bot picks it); for
            # the others it means "unchosen", stored as "".
            setattr(sess, "music_" + field,
                    value if (field == "duration" or value != "auto") else "")
            self._store.put(sess)
            _redraw(tg_bot._field_text(sess, field, lang),
                    tg_bot._field_kb(sess, field, lang))
            self._activity.log(chat_id, "system",
                               f"[song settings] {field}={value}")
            return
        if action == "custom" and len(parts) > 2 and parts[2] in ("duration", "tempo", "steps"):
            # Arm the capture; the number arrives as the next message
            # (tg_registration handles the music_custom: state).
            field = parts[2]
            sess.reg_state = _tg_music.CUSTOM_STATE + field
            self._store.put(sess)
            lo, hi = _tg_music.custom_range(field)
            self._send_text(chat_id, tg_bot._t("ms_ask_" + field, lang, lo=lo, hi=hi))
            return
        # Unrecognised music: payload — fall back to the menu rather
        # than leaving the tap with no visible effect at all.
        _redraw(tg_bot._music_menu_text(sess, lang),
                tg_bot._music_menu_kb(sess, lang))
        return

    def _cb_pick_character(self, chat_id: int, data: str) -> None:
        """🧑 <name> — arm the next message as a scene for that character."""
        sess = self._get_session(chat_id)
        self._pick_character(chat_id, sess, self._lang(sess),
                             data.split(":", 1)[1].strip())

    def _cb_research_depth(self, chat_id: int, msg: dict, data: str) -> None:
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        value = data.split(":", 1)[1].strip().lower()
        if value not in tg_bot._DEPTHS:
            tg_bot.logger.warning("depth callback: unknown depth %r", value)
            # A stale/hand-crafted tap used to just `return` here — the
            # keyboard on screen kept its OLD ✅, silently lying about
            # which depth is actually active. Redraw against whatever
            # IS active (unchanged) so the ✅ is honest again, instead of
            # leaving a keyboard whose state nobody can trust.
            msg_id = msg.get("message_id")
            if msg_id:
                self._edit_text(chat_id, msg_id, tg_bot._depth_menu_text(sess, lang),
                                parse_mode="HTML",
                                keyboard=tg_bot._depth_menu_kb(sess, lang))
            return
        sess.dr_depth = value
        self._store.put(sess)
        # Redraw in place for the same reason as the size picker: a new
        # message per tap buries the chat, and the ✅ has to move or the
        # keyboard lies about the current setting.
        msg_id = msg.get("message_id")
        if msg_id:
            self._edit_text(chat_id, msg_id, tg_bot._depth_menu_text(sess, lang),
                            parse_mode="HTML",
                            keyboard=tg_bot._depth_menu_kb(sess, lang))
        else:
            self._send_text(chat_id,
                            tg_bot._t("depth_set", lang,
                               name=tg_bot._depth_name(value, lang),
                               eta=tg_bot._depth_eta_text(value, lang)),
                            parse_mode="HTML")
        self._activity.log(chat_id, "system", f"[depth] {value}")
        return

    def _cb_forget_facts(self, chat_id: int, cb: dict, data: str) -> None:
        # 🗑 Forget all is an irreversible wipe of the durable fact store, so
        # it needs (a) a presser check — the callback gate above checks the
        # CHAT's account status, not who actually tapped the button, and a
        # stranger's `from.id` in a group (or on an old, never-expiring
        # inline keyboard) could otherwise wipe someone else's facts — and
        # (b) a second confirming tap, like any other irreversible action.
        presser = (cb.get("from") or {}).get("id")
        if presser != chat_id:
            tg_bot.logger.warning("rejected facts_clear callback from %s in chat %s",
                           presser, chat_id)
            return
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        if data == "facts_clear":
            n = len(sess.get_tg_facts())
            if n == 0:
                self._send_text(chat_id, tg_bot._t("facts_cleared", lang, n=0),
                                keyboard=tg_bot._settings_kb(sess.reply_mode, sess.is_admin, lang))
                return
            self._send_text(chat_id, tg_bot._t("facts_clear_confirm", lang, n=n),
                            keyboard={"inline_keyboard": [[
                                {"text": tg_bot._t("facts_clear_yes", lang),
                                 "callback_data": "facts_clear_confirm"},
                                {"text": tg_bot._t("facts_clear_no", lang),
                                 "callback_data": "facts_clear_cancel"}]]})
            return
        if data == "facts_clear_cancel":
            self._send_text(chat_id, tg_bot._t("facts_clear_cancelled", lang),
                            keyboard=tg_bot._settings_kb(sess.reply_mode, sess.is_admin, lang))
            return
        # facts_clear_confirm
        n = len(sess.get_tg_facts())
        sess.set_tg_facts([])
        self._store.put(sess)
        tg_bot._delete_chat_facts_file(chat_id)
        self._send_text(chat_id, tg_bot._t("facts_cleared", lang, n=n),
                        keyboard=tg_bot._settings_kb(sess.reply_mode, sess.is_admin, lang))
        return

    def _cb_weather_default(self, chat_id: int) -> None:
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        user = self._user_store.get(chat_id)
        city = ((user.prefs or {}).get("city") if user else "") or _config.WEATHER_DEFAULT_CITY
        # Re-arm (not clear): the abandon-any-button-press guard above
        # already cleared reg_state before this branch runs, but the
        # lookup below is async and the default city could still fail
        # to geocode (a stale remembered city, a misconfigured
        # WEATHER_DEFAULT_CITY, a transient API hiccup) -- staying
        # armed means a retry after that failure is tried as another
        # city instead of falling through to the agent. _send_weather
        # clears it again once a city actually resolves.
        sess.reg_state = "wtw_city"
        self._store.put(sess)
        self._start_weather_lookup(chat_id, city, lang)
        return

    def _cb_weather_48h(self, chat_id: int) -> None:
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        loc = sess.wtw_loc or None
        if not loc:
            # Nothing resolved yet to expand -- fall back to asking
            # for a city instead of silently doing nothing (can
            # happen if the session was reset between the forecast
            # and this button being pressed).
            self._start_weather_flow(chat_id, sess, lang)
            return
        self._start_weather_lookup(chat_id, loc.get("name", ""), lang,
                                   loc=loc, hours=48)
        return

    def _cb_weather_day(self, chat_id: int, data: str) -> None:
        # today/tomorrow shortcuts under the 📅 Date prompt, so the two
        # commonest answers need no typing at all.
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        try:
            offset = int(data.split(":", 1)[1])
        except (ValueError, IndexError):
            tg_bot.logger.warning("weather date callback: bad payload %r", data)
            return
        if offset not in (0, 1):
            return
        import datetime as _d
        target = _d.date.today() + _d.timedelta(days=offset)
        if sess.reg_state == "wtw_date":
            sess.reg_state = ""
            self._store.put(sess)
        city = self._remembered_city(chat_id)
        loc = sess.wtw_loc or None
        self._start_weather_lookup(chat_id, city, lang, date=target, loc=loc)
        return

    def _cb_weather_pickdate(self, chat_id: int) -> None:
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        if not sess.wtw_loc:
            self._start_weather_flow(chat_id, sess, lang)
            return
        sess.reg_state = "wtw_date"
        self._store.put(sess)
        self._send_text(chat_id, tg_bot._t("wtw_date_prompt", lang), parse_mode="HTML")
        return

    def _cb_admin_action(self, chat_id: int, cb: dict, data: str) -> None:
        # Gate on WHO PRESSED (callback_query.from), not on the chat the
        # button sits in. Inline buttons never expire, so without this a
        # demoted admin could keep approving users from an old admin-panel
        # message forever, and in a group any member could press an approval
        # prompt that was meant for the admin who is also in that group.
        # This branch is reached without the earlier `sess = ...` lines
        # having run, so it needs its own.
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        presser = (cb.get("from") or {}).get("id")
        actor = self._user_store.get(presser) if presser else None
        if not (actor and actor.is_admin and actor.status == "approved"):
            tg_bot.logger.warning("rejected admin callback %r from non-admin %s",
                           data, presser)
            self._send_text(chat_id, tg_bot._t("admins_only", lang))
            return
        if data.startswith("bcast:"):
            self._confirm_broadcast(chat_id, data.split(":", 1)[1])
            return
        try:
            target = int(data.split(":", 1)[1])
        except ValueError:
            return
        if data.startswith("admin_approve:"):
            outcome = self.approve_user(target)
            if outcome == "not_found":
                self._send_text(chat_id,
                                tg_bot._t("user_not_found", lang, id=target))
            elif outcome == "not_pending":
                tgt_user = self._user_store.get(target)
                self._send_text(chat_id,
                                tg_bot._t("user_not_pending", lang, id=target,
                                   status=(tgt_user.status if tgt_user else "?")))
            else:
                self._send_text(chat_id,
                                tg_bot._t("user_approved_by", lang, id=target))
        else:
            outcome = self.reject_user(target)
            if outcome == "not_found":
                self._send_text(chat_id,
                                tg_bot._t("user_not_found", lang, id=target))
            elif outcome == "not_pending":
                tgt_user = self._user_store.get(target)
                self._send_text(chat_id,
                                tg_bot._t("user_not_pending", lang, id=target,
                                   status=(tgt_user.status if tgt_user else "?")))
            else:
                self._send_text(chat_id,
                                tg_bot._t("user_rejected_by", lang, id=target))
        return

    def _cb_account(self, chat_id: int, data: str) -> None:
        # Account self-service callbacks
        user = self._user_store.get(chat_id)
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        if not user or user.status != "approved":
            return
        if data == "acct_setname":
            sess.reg_state = "change_name"
            self._store.put(sess)
            self._send_text(chat_id,
                tg_bot._t("ask_new_name", lang,
                   name=_html_mod.escape(user.name)),
                parse_mode="HTML")
        elif data == "acct_setpwd":
            sess.reg_state = "change_password"
            self._store.put(sess)
            self._send_text(chat_id, tg_bot._t("ask_new_pass", lang),
                            parse_mode="HTML")
        elif data == "acct_lang":
            self._send_lang_menu(chat_id, self._lang(sess))
        elif data == "acct_toggle_startup":
            if "startup" in user.subscriptions:
                user.subscriptions = [s for s in user.subscriptions if s != "startup"]
                label = tg_bot._t("unsubscribed", lang)
            else:
                user.subscriptions.append("startup")
                label = tg_bot._t("subscribed", lang)
            self._user_store.put(user)
            self._send_text(chat_id, label,
                            keyboard=self._main_menu_kb(sess))
        elif data == "acct_toggle_feedback":
            if not user.is_admin:
                self._send_text(chat_id, tg_bot._t("feedback_only_admin", lang))
                return
            subs = list(user.subscriptions or [])
            if tg_bot._wants_feedback(user):
                # Was subscribed -> mute. Drop BOTH markers: the opt-in one
                # (undoing this toggle) and the legacy opt-out one (so a
                # pre-flip admin who re-subscribes isn't immediately muted
                # again next toggle by a stale leftover marker).
                subs = [s for s in subs
                        if s not in (tg_bot.FEEDBACK_OPT_IN, tg_bot.FEEDBACK_OPT_OUT)]
                label = tg_bot._t("feedback_off", lang)
            else:
                subs = [s for s in subs if s != tg_bot.FEEDBACK_OPT_OUT]
                if tg_bot.FEEDBACK_OPT_IN not in subs:
                    subs.append(tg_bot.FEEDBACK_OPT_IN)
                label = tg_bot._t("feedback_on", lang)
            user.subscriptions = subs
            self._user_store.put(user)
            self._activity.log(chat_id, "system",
                f"[feedback] admin {user.name} -> "
                f"{'subscribed' if tg_bot.FEEDBACK_OPT_IN in subs else 'muted'}", user.name)
            self._send_text(chat_id, label,
                            keyboard=self._main_menu_kb(sess))
        elif data == "acct_badge_menu":
            if not user.is_admin:
                self._send_text(chat_id, tg_bot._t("badge_only_admin", lang))
                return
            self._send_badge_menu(chat_id, user, sess)
        elif data.startswith("acct_badge_set:"):
            if not user.is_admin:
                self._send_text(chat_id, tg_bot._t("badge_only_admin", lang))
                return
            badge = data.split(":", 1)[1]
            if self.set_admin_badge(chat_id, badge):
                self._activity.log(chat_id, "system",
                    f"[badge] admin {user.name} -> {badge}", user.name)
                self._send_text(chat_id, tg_bot._t("badge_set", lang, badge=badge))
            else:
                self._send_text(chat_id, tg_bot._t("badge_invalid", lang))
        elif data == "acct_badge_clear":
            if not user.is_admin:
                self._send_text(chat_id, tg_bot._t("badge_only_admin", lang))
                return
            self.set_admin_badge(chat_id, "")
            self._activity.log(chat_id, "system",
                f"[badge] admin {user.name} -> cleared", user.name)
            self._send_text(chat_id, tg_bot._t("badge_cleared", lang))
        elif data == "acct_badge_custom":
            if not user.is_admin:
                self._send_text(chat_id, tg_bot._t("badge_only_admin", lang))
                return
            sess.reg_state = "set_badge"
            self._store.put(sess)
            self._send_text(chat_id, tg_bot._t("badge_custom_prompt", lang))
        elif data == "acct_logout":
            self._logout(chat_id, user, sess)
        elif data == "acct_newprofile":
            # Destructive: replacing the profile deletes the account row, so
            # it is confirmed explicitly rather than done on the logout tap.
            self._send_text(chat_id,
                tg_bot._t("profile_replace", lang,
                   name=_html_mod.escape(user.name)),
                parse_mode="HTML",
                keyboard={"inline_keyboard": [
                    [{"text": tg_bot._t("profile_yes_btn", lang),
                      "callback_data": "acct_newprofile_yes"}],
                    [{"text": tg_bot._t("profile_no_btn", lang),
                      "callback_data": "acct_newprofile_no"}]]})
        elif data == "acct_newprofile_no":
            self._send_text(chat_id, tg_bot._t("profile_kept", lang),
                            parse_mode="HTML")
        elif data == "acct_newprofile_yes":
            name = user.name
            self._user_store.delete(chat_id)
            sess.clear_history()
            sess.set_tg_memory([])
            sess.pending_prefix = ""
            sess.reg_state = "awaiting_name"
            sess.reg_name = ""
            sess.is_admin = False
            self._store.put(sess)
            # _logout (a far less destructive action -- it keeps the
            # account row) already wipes the pinned facts and their
            # on-disk shadow; deleting the account outright must not
            # leave MORE behind than logging out does. Without this,
            # the facts remember_fact pinned about the old profile
            # (name, preferences, whatever the assistant inferred)
            # silently carried over to the brand-new profile being
            # registered under this same chat_id right below.
            sess.set_tg_facts([])
            self._store.put(sess)
            tg_bot._delete_chat_facts_file(chat_id)
            self._activity.log(chat_id, "system",
                               f"[auth] profile '{name}' deleted — re-registering", name)
            self._send_text(chat_id, tg_bot._t("profile_gone", lang),
                            parse_mode="HTML")
        return

    def _cb_style_image(self, chat_id: int) -> None:
        """🎭 pressed under a picture: offer the tuned presets (no second photo
        needed -- redraw_image keeps the composition via controlnet), a free-
        text custom style, or the older reference-photo flow. Arms
        pending_style_target either way, since a preset/custom pick still
        needs to know WHICH picture it targets and a reference photo remains
        a valid third option; tg_resolve.py's photo branch is untouched."""
        sess = self._get_session(chat_id)
        target = sess.target_image
        entry = tg_bot._image_by_id(sess, target) if target else None
        if not entry or not os.path.exists(entry.get("path", "")):
            self._send_text(chat_id, tg_bot._t("img_gone", self._lang(sess)))
            return
        sess.pending_style_target = target
        self._store.put(sess)
        lang = self._lang(sess)
        rows = []
        for i in range(0, len(style_presets.STYLE_PRESET_ORDER), 2):
            row = []
            for key in style_presets.STYLE_PRESET_ORDER[i:i + 2]:
                row.append({"text": style_presets.preset_label(key, lang),
                            "callback_data": f"style_preset:{key}"})
            rows.append(row)
        rows.append([{"text": tg_bot._t("style_custom_btn", lang),
                      "callback_data": "style_custom"}])
        self._send_text(chat_id, tg_bot._t("style_ask_ref", lang),
                        keyboard={"inline_keyboard": rows})
        return

    def _cb_style_preset(self, chat_id: int, data: str) -> None:
        """style_preset:<key> pressed: push a redraw_image task straight to
        the queue with the preset's tuned instructions, the same bypass
        _push_style_transfer_task uses for the reference-photo flow -- a
        button press is a command, not conversation, so it skips the
        debounce/text-merge path entirely."""
        key = data.split(":", 1)[1] if ":" in data else ""
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        target_id = sess.pending_style_target or sess.target_image
        sess.pending_style_target = ""
        self._store.put(sess)
        prompt = style_presets.preset_prompt(key)
        target = tg_bot._image_by_id(sess, target_id) if target_id else None
        if not prompt or not target or not os.path.exists(target.get("path", "")):
            self._send_text(chat_id, tg_bot._t("img_gone", lang))
            return
        task = tg_bot._Task(
            task_id=str(uuid.uuid4()),
            chat_id=chat_id,
            user_text=(f"[style_preset] call redraw_image with mode=\"redraw\" "
                       f"instructions=\"{prompt}\" on the current image. Do not "
                       "generate a new image."),
            image_path=target["path"],
            image_id=target_id,
            enqueue_ts=time.time(),
        )
        if self._try_steer(chat_id, task):
            return
        with self._task_lock:
            self._pending_journal[task.task_id] = task
        self._backend.push(task)
        self._write_inflight()
        tg_bot.logger.info("Enqueued task %s chat=%s kind=style_preset:%s depth=%d",
                    task.task_id[:8], chat_id, key, self._backend.depth())
        try:
            ahead = self._backend.tasks_ahead(task.task_id)
        except Exception:
            ahead = []
        if ahead:
            self._send_text(chat_id,
                tg_bot._t("queue_pos", lang, pos=len(ahead) + 1, eta=tg_bot._fmt_eta(ahead, lang)),
                parse_mode="HTML")
        return

    def _cb_style_custom(self, chat_id: int) -> None:
        """✏️ Custom style pressed: arm pending_prefix so the user's next
        free-text message becomes a redraw instruction, same mechanism as
        ✏️ Edit (_cb_edit_image). pending_style_target is left as-is -- it
        already names the right picture (retargeted at press time, same as
        edit) and a reference photo remains a fallback if they send one
        instead of typing."""
        sess = self._get_session(chat_id)
        sess.pending_prefix = "change the image style to: "
        self._store.put(sess)
        self._send_text(chat_id, tg_bot._t("style_ask_custom", self._lang(sess)),
                        keyboard=self._state_kb(sess))
        return

    def _cb_animate_image(self, chat_id: int) -> None:
        """🎬 pressed under a picture: offer the motion presets (no separate
        upload needed -- the picture already sitting under the button IS the
        target) or a free-text custom motion. Mirrors _cb_style_image exactly;
        arms pending_animate_target the same way _register_animate_photo does
        for the standalone "Animate photo" menu entry point."""
        sess = self._get_session(chat_id)
        target = sess.target_image
        entry = tg_bot._image_by_id(sess, target) if target else None
        if not entry or not os.path.exists(entry.get("path", "")):
            self._send_text(chat_id, tg_bot._t("img_gone", self._lang(sess)))
            return
        sess.pending_animate_target = target
        self._store.put(sess)
        lang = self._lang(sess)
        self._animate_ask_voices(chat_id, sess, lang)   # voice samples first, then presets
        return

    def _cb_animate_preset(self, chat_id: int, data: str) -> None:
        """animate_preset:<key> pressed: push a generate_video task straight
        to the queue for the photo tg_resolve._register_animate_photo
        already registered, same bypass _cb_style_preset uses."""
        import animate_presets
        key = data.split(":", 1)[1] if ":" in data else ""
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        target_id = sess.pending_animate_target
        sess.pending_animate_target = ""
        self._store.put(sess)
        prompt = animate_presets.preset_prompt(key)
        target = tg_bot._image_by_id(sess, target_id) if target_id else None
        if not prompt or not target or not os.path.exists(target.get("path", "")):
            self._send_text(chat_id, tg_bot._t("img_gone", lang))
            return
        task = tg_bot._Task(
            task_id=str(uuid.uuid4()),
            chat_id=chat_id,
            user_text=(f"[animate] call generate_video with description=\"{prompt}\" "
                       "using the current image."),
            image_path=target["path"],
            image_id=target_id,
            enqueue_ts=time.time(),
        )
        if self._try_steer(chat_id, task):
            return
        with self._task_lock:
            self._pending_journal[task.task_id] = task
        self._backend.push(task)
        self._write_inflight()
        tg_bot.logger.info("Enqueued task %s chat=%s kind=animate_preset:%s depth=%d",
                    task.task_id[:8], chat_id, key, self._backend.depth())
        try:
            ahead = self._backend.tasks_ahead(task.task_id)
        except Exception:
            ahead = []
        if ahead:
            self._send_text(chat_id,
                tg_bot._t("queue_pos", lang, pos=len(ahead) + 1, eta=tg_bot._fmt_eta(ahead, lang)),
                parse_mode="HTML")
        return

    def _cb_animate_custom(self, chat_id: int) -> None:
        """✏️ Custom motion pressed: arm pending_prefix so the user's next
        free-text message becomes the motion description, same mechanism as
        ✏️ Custom style (_cb_style_custom). pending_animate_target is left
        as-is -- it already names the right photo."""
        sess = self._get_session(chat_id)
        sess.pending_prefix = "animate this photo: "
        self._store.put(sess)
        self._send_text(chat_id, tg_bot._t("animate_ask_custom", self._lang(sess)),
                        keyboard=self._state_kb(sess))
        return

    def _cb_change_clothes(self, chat_id: int) -> None:
        """👗 pressed under a picture: ask what to change the outfit TO, instead
        of silently applying a random "something different" swap.

        Live, 2026-09-19: the user asked for this to work "in the spirit of
        what-to-what, like I write 'the hat to a hat' [sic, meant a specific
        garment]" -- the same free-text capture _cb_edit_image already uses,
        just with its own prefix so the agent gets "change the outfit to: X"
        rather than a bare edit instruction. _CB_CMDS["change_clothes"] is left
        in place as the underlying enqueued text shape for old code paths/tests;
        this handler is reached BEFORE that generic dispatch (see the `if data
        == "change_clothes"` line above it), so every press -- old keyboard or
        new -- now asks first."""
        sess = self._get_session(chat_id)
        sess.pending_prefix = "change the outfit to: "
        self._store.put(sess)
        self._send_text(chat_id, tg_bot._t("describe_clothes", self._lang(sess)),
                        keyboard=self._state_kb(sess))
        return

    def _cb_remove_object(self, chat_id: int) -> None:
        """🧽 under a picture: ask WHAT to remove (same free-text capture as 👗)."""
        sess = self._get_session(chat_id)
        sess.pending_prefix = "remove from the image: "
        self._store.put(sess)
        self._send_text(chat_id, tg_bot._t("describe_remove", self._lang(sess)),
                        keyboard=self._state_kb(sess))
        return

    def _cb_edit_image(self, chat_id: int) -> None:
        sess = self._get_session(chat_id)
        sess.pending_prefix = "edit the image: "
        self._store.put(sess)
        # _state_kb, NOT _main_menu_kb: the latter exists to CLEAR
        # pending_prefix so the keyboard can never contradict the state,
        # so asking it for a keyboard one line after arming the prefix
        # wiped the prefix. Live consequence — the user tapped ✏️, typed
        # "the lettering is crooked", and the agent received a bare
        # complaint with no instruction to edit anything; it floundered
        # for two rounds and answered that it could not call any tools.
        # "edit the image: " is already in _PREFIX_MENU, so the matching
        # keyboard was there the whole time.
        self._send_text(chat_id, tg_bot._t("describe_edit", self._lang(sess)),
                        keyboard=self._state_kb(sess))
        return

    def _cb_ask_image(self, chat_id: int) -> None:
        """❓ under a picture: the press already made that picture the target
        (the image-verb branch of the callback router), and a message about a
        picture the user pointed at is answered by looking at it
        (graph.needs_relook) -- no text prefix to parse back out."""
        sess = self._get_session(chat_id)
        self._send_text(chat_id, tg_bot._t("describe_ask", self._lang(sess)),
                        keyboard=self._state_kb(sess))


import tg_bot  # noqa: E402  (cycle by design; attrs read at call time)
import tg_music as _tg_music  # noqa: E402
import music as _music_mod  # noqa: E402

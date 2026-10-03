"""_user_gate: everything that must happen before a message is an ordinary one.

Lifted out of tg_accounts.py, where it was a 454-line method wrapped around
the operator actions it has nothing to do with. One entry point, called by
_dispatch on every incoming message, returning True only when the chat is an
approved user in no special state and the message should fall through to
normal dispatch.

Two kinds of thing live here:

  * The account states -- first-time registration (name, then password), the
    login re-prompt for a chat that logged out, and the pending/rejected/banned
    replies.

  * The armed TEXT-CAPTURE modes: 📝 feedback, ✏️ change name, 🔑 change
    password, the admin badge, the weather city and date, the song topic. Each
    steals the user's very next message as its answer.

Every capture mode shares one preamble, and each line of it is a bug that
actually happened: /cancel gets out; any OTHER slash command abandons the mode
and RUNS (the whole body of "/broadcast hello everyone" once became someone's
new password); a recognized button label abandons the mode too, because the
keyboard from before the mode was armed is still on screen and a mis-tap
otherwise became the answer. See _capture_preamble, which is the one copy of
all three.
"""
from __future__ import annotations

import html as _html_mod
import time

import weather as _weather_mod


class RegistrationMixin:
    # ── the opening every armed capture mode shares ──────────────────────────
    # 📝 feedback, ✏️ change name, 🔑 change password, 🎖 badge, the weather
    # city and date, the song topic: each arms sess.reg_state to claim the
    # user's VERY NEXT message as its answer. All six then have to answer the
    # same question first -- is this message an answer at all? -- and every
    # line below is a bug that reached a real user before it was written:
    #
    #   /cancel        the way out, and it must say so.
    #   /anything-else control, not content. "/broadcast hello everyone" once
    #                  became an admin's new password: long enough to pass the
    #                  length check, the broadcast never ran, and nothing
    #                  looked wrong until the next login failed.
    #   a button label the keyboard from BEFORE the mode was armed is still on
    #                  screen -- nothing replaces it -- so a mis-tap arrived
    #                  here as free text. A ⛔ Stop press was eaten as feedback
    #                  while the task it meant to stop kept running; a tap on
    #                  🎨 Draw could become someone's password.
    #
    # In all three cases the mode is dropped and the message is handed back to
    # normal dispatch, so the tap does what it says instead of being swallowed.
    def _clear_capture(self, sess) -> None:
        sess.reg_state = ""
        self._store.put(sess)

    def _capture_preamble(self, chat_id: int, sess, lang: str, text: str,
                          *, empty_abandons: bool = False):
        """None when `text` is really the user's answer; otherwise the value
        _user_gate must return -- False if the message was consumed here, True
        if the mode was abandoned and the message should fall through.

        `empty_abandons` covers a message with no text at all (a photo, a
        sticker, a voice note). It is not a universal rule because the modes
        disagree on purpose: the weather and song flows stay armed so the user
        can simply answer next time, while feedback treats it as a change of
        mind rather than filing an empty report.
        """
        if text.startswith("/cancel"):
            self._clear_capture(sess)
            self._send_text(chat_id, tg_bot._t("cancelled", lang),
                            keyboard=self._main_menu_kb(sess, lang))
            return False
        if text.startswith("/"):
            self._clear_capture(sess)
            return True
        if tg_bot._LABEL2KEY.get(text):
            self._clear_capture(sess)
            return True
        if empty_abandons and not text:
            self._clear_capture(sess)
            return True
        return None

    def _user_gate(self, chat_id: int, msg: dict) -> bool:
        """Returns True if message should be processed normally."""
        user = self._user_store.get(chat_id)
        sess = self._get_session(chat_id)
        text = (msg.get("text") or "").strip()
        from_data = msg.get("from") or {}
        tg_uname  = from_data.get("username", "")
        lang      = self._lang(sess)

        # Commands are CONTROL, never registration input. Without this, restarting
        # the bot mid-registration fed "/start" straight into the awaiting_password
        # branch: it is >=4 chars, so it was accepted AS THE PASSWORD and the account
        # was created silently — the user never knowingly set one, and from their side
        # it looked like the bot "remembered" a password they never entered.
        # /start re-prompts the current step, /cancel abandons registration.
        if user is None and sess.reg_state and text.startswith("/"):
            cmd = text.split()[0].lower().lstrip("/").split("@")[0]
            if cmd == "cancel":
                sess.reg_state = ""
                sess.reg_name = ""
                self._store.put(sess)
                self._send_text(chat_id, tg_bot._t("reg_cancelled", lang))
                return False
            if sess.reg_state == "awaiting_name":
                self._send_text(chat_id, tg_bot._t("resume_name", lang), parse_mode="HTML")
            else:
                self._send_text(chat_id,
                    tg_bot._t("resume_pwd", lang,
                       name=_html_mod.escape(sess.reg_name or "there")),
                    parse_mode="HTML")
            return False

        # ── LOGGED OUT — require the account password before anything else ────
        if user is not None and sess.reg_state == "awaiting_login":
            # This branch used to run unconditionally once reg_state was
            # "awaiting_login", with no check of user.status at all. That local
            # session flag only ever means "logged out, needs the password
            # again" -- it says nothing about whether the account is still in
            # good standing. approve_user/reject_user/ban_user only ever flip
            # the account row's status; none of them reach into every logged-
            # out chat's session to clear this flag (nor should they -- a
            # session store write for an account change that hasn't touched
            # this chat in days is its own kind of surprise). Concretely: log
            # out, then get banned (or rejected) -- the correct password still
            # walked you straight back to "Welcome back" and the full menu,
            # silently undoing the ban. A pending status genuinely cannot reach
            # here (accounts don't go pending -> awaiting_login), but the check
            # is written against "not approved" rather than an enumerated bad
            # set so it still fails closed if that ever changes.
            if user.status != "approved":
                if text.startswith("/"):
                    self._send_text(chat_id, tg_bot._t("locked_out", lang), parse_mode="HTML")
                    return False
                if text:
                    # A rejected attempt is still a password attempt.
                    self._scrub_secret(chat_id, msg)
                self._send_text(chat_id, tg_bot._t("revoked", lang))
                return False
            if text.startswith("/"):
                self._send_text(chat_id, tg_bot._t("locked_out", lang), parse_mode="HTML")
                return False
            if not text:
                return False
            # Throttle: the gate accepted unlimited guesses at machine speed.
            state = self._login_fails.get(chat_id) or [0, 0.0]
            if state[1] > time.time():
                self._send_text(chat_id,
                    tg_bot._t("login_locked", lang, sec=int(state[1] - time.time())))
                return False
            self._scrub_secret(chat_id, msg)
            ok, upgraded = tg_bot._verify_password(text.strip(), chat_id, user.password_hash)
            if not ok:
                state[0] += 1
                max_fails = max(1, tg_bot._cfg_int("TG_LOGIN_MAX_FAILS", 5))
                if state[0] >= max_fails:
                    lockout = max(10, tg_bot._cfg_int("TG_LOGIN_LOCKOUT_S", 300))
                    state[1] = time.time() + lockout
                    state[0] = 0
                    self._login_fails[chat_id] = state
                    self._activity.log(chat_id, "error",
                                       "[auth] locked out after repeated failures",
                                       user.name)
                    self._send_text(chat_id,
                        tg_bot._t("login_locked", lang, sec=lockout))
                    return False
                self._login_fails[chat_id] = state
                self._activity.log(chat_id, "system", "[auth] failed login", user.name)
                self._send_text(chat_id,
                                tg_bot._t("wrong_pwd", lang, left=max_fails - state[0]))
                return False
            self._login_fails.pop(chat_id, None)
            if upgraded:
                # Legacy sha256 hash — replace it now that we have the plaintext.
                user.password_hash = upgraded
                self._user_store.put(user)
                self._activity.log(chat_id, "system",
                                   "[auth] password hash upgraded to scrypt", user.name)
            sess.reg_state = ""
            self._store.put(sess)
            self._activity.log(chat_id, "system", "[auth] logged in", user.name)
            self._send_text(chat_id,
                tg_bot._t("welcome_back", lang, name=_html_mod.escape(user.name)),
                parse_mode="HTML",
                keyboard=self._main_menu_kb(sess, lang))
            return False

        # ── NEW USER — start registration ─────────────────────────────────────
        if user is None:
            if not sess.reg_state:
                # Check if they are a pre-configured admin
                auto_admin = chat_id in self._admin_chat_ids
                sess.reg_state = "awaiting_name"
                self._store.put(sess)
                self._activity.log(chat_id, "system",
                                   f"New user @{tg_uname} started registration")
                self._send_text(chat_id, tg_bot._t("ask_name", lang), parse_mode="HTML")
                return False

            if sess.reg_state == "awaiting_name":
                # A recognized keyboard-button label is never a real name. This
                # state is also reached by acct_newprofile_yes (an EXISTING user
                # re-registering), whose old main-menu keyboard is still visible
                # on screen right when the bot asks for a name — the same class
                # of bug already fixed below for change_name/change_password/
                # feedback: a mis-tap on a still-showing button must not get
                # silently accepted as the answer.
                if text and tg_bot._LABEL2KEY.get(text):
                    self._send_text(chat_id, tg_bot._t("not_a_menu_tap", lang),
                                    parse_mode="HTML")
                    return False
                if not text or len(text.strip()) < 2:
                    self._send_text(chat_id, tg_bot._t("name_again", lang), parse_mode="HTML")
                    return False
                sess.reg_name  = text.strip()[:50]
                sess.reg_state = "awaiting_password"
                self._store.put(sess)
                self._send_text(chat_id,
                    tg_bot._t("ask_password", lang,
                       name=_html_mod.escape(sess.reg_name)),
                    parse_mode="HTML")
                return False

            if sess.reg_state == "awaiting_password":
                # Same guard as awaiting_name above: a mis-tapped menu button is
                # not a password attempt (nothing secret to scrub — it is public
                # button text), and setting the account's password to a button's
                # label would leave the user unable to ever log back in.
                if text and tg_bot._LABEL2KEY.get(text):
                    self._send_text(chat_id, tg_bot._t("not_a_menu_tap", lang),
                                    parse_mode="HTML")
                    return False
                if not text or len(text.strip()) < 4:
                    # A rejected attempt is still a password attempt.
                    self._scrub_secret(chat_id, msg)
                    self._send_text(chat_id, tg_bot._t("pwd_short", lang), parse_mode="HTML")
                    return False
                self._scrub_secret(chat_id, msg)
                pwd_hash = tg_bot._hash_password(text.strip(), chat_id)
                auto_admin = chat_id in self._admin_chat_ids
                user = tg_bot._User(
                    chat_id=chat_id,
                    name=sess.reg_name or "User",
                    tg_username=tg_uname,
                    password_hash=pwd_hash,
                    status="approved" if auto_admin else "pending",
                    is_admin=auto_admin,
                )
                self._user_store.put(user)
                sess.reg_state = ""
                sess.reg_name  = ""
                sess.is_admin  = auto_admin
                self._store.put(sess)

                if auto_admin:
                    self._send_text(chat_id,
                        tg_bot._t("admin_welcome", lang) + "\n" + tg_bot._help_text(lang),
                        parse_mode="HTML",
                        keyboard=self._main_menu_kb(sess, lang))
                    self._activity.log(chat_id, "system",
                                       f"Admin {user.name} auto-approved", user.name)
                else:
                    self._send_text(chat_id, tg_bot._t("reg_pending", lang), parse_mode="HTML")
                    self._activity.log(chat_id, "system",
                                       f"User {user.name} registered, pending approval",
                                       user.name)

                # Notify GUI
                self._on_user_change(chat_id, user.name, user.status)

                # Notify admin users via Telegram
                for admin_id in self._admin_chat_ids:
                    au = self._user_store.get(admin_id)
                    if au and au.status == "approved":
                        try:
                            alang = self._lang(self._get_session(admin_id))
                            self._send_text(admin_id,
                                tg_bot._t("new_reg", alang,
                                   name=_html_mod.escape(user.name),
                                   handle=(f" (@{_html_mod.escape(tg_uname)})"
                                           if tg_uname else ""),
                                   id=chat_id),
                                parse_mode="HTML",
                                keyboard={"inline_keyboard": [[
                                    {"text": tg_bot._t("approve_btn", alang, name=user.name),
                                     "callback_data": f"admin_approve:{chat_id}"},
                                    {"text": tg_bot._t("reject_btn", alang),
                                     "callback_data": f"admin_reject:{chat_id}"},
                                ]]})
                        except Exception: pass
                return False
            # A state left over from an account that no longer exists (deleted
            # by an admin mid-"feedback"/"change_name") used to answer every
            # message with silence, forever. Start registration over.
            sess.reg_state = "awaiting_name"
            self._store.put(sess)
            self._send_text(chat_id, tg_bot._t("ask_name", lang), parse_mode="HTML")
            return False

        # ── ACCOUNT EDIT FLOW (existing, approved users) ──────────────────────
        if user and user.status == "approved" and sess.reg_state == "feedback":
            done = self._capture_preamble(chat_id, sess, lang, text, empty_abandons=True)
            if done is not None:
                return done
            self._save_feedback(chat_id, user, text.strip())
            sess.reg_state = ""
            self._store.put(sess)
            self._send_text(chat_id, tg_bot._t("feedback_ok", lang),
                parse_mode="HTML",
                keyboard=self._main_menu_kb(sess, lang))
            return False

        if user and user.status == "approved" and sess.reg_state == "wtw_city":
            done = self._capture_preamble(chat_id, sess, lang, text)
            if done is not None:
                return done
            if not text:
                return False
            # reg_state deliberately stays "wtw_city" here -- geocoding runs on
            # the background thread _start_weather_lookup kicks off, and
            # whether THIS city resolves is not known yet. Clearing it here
            # unconditionally (the old behaviour) meant a failed lookup
            # ("Спб" -> "couldn't find that city") silently dropped out of
            # weather mode: the user's very next message (another retry, e.g.
            # "Питнр") was routed to the full agent pipeline as an ordinary
            # question instead of being tried as a city. _send_weather clears
            # reg_state itself, and only once geocode_city() actually
            # succeeds.
            self._start_weather_lookup(chat_id, text.strip(), lang, remember_for=chat_id)
            return False

        if user and user.status == "approved" and sess.reg_state == "wtw_date":
            done = self._capture_preamble(chat_id, sess, lang, text)
            if done is not None:
                return done
            if not text:
                return False
            loc = sess.wtw_loc or None
            if not loc:
                # Nothing to anchor the date to (e.g. the session was reset
                # between the "pick a date" prompt and the reply) -- abandon
                # rather than silently doing nothing forever.
                sess.reg_state = ""
                self._store.put(sess)
                return True
            parsed = _weather_mod.parse_date_input(text.strip())
            if parsed is None:
                # Same "stay armed for a retry" contract as wtw_city -- an
                # unparseable date shouldn't drop the user out of the flow.
                self._send_text(chat_id, tg_bot._t("wtw_date_invalid", lang))
                return False
            # reg_state deliberately stays "wtw_date" here, same reasoning as
            # wtw_city above -- _send_weather is the one place that clears it,
            # once the lookup has actually run (loc is already known-good
            # here, so in practice this always resolves, but the contract
            # stays consistent with wtw_city's retry-safe design).
            self._start_weather_lookup(chat_id, loc.get("name", ""), lang,
                                       loc=loc, date=parsed)
            return False

        if (user and user.status == "approved"
                and (sess.reg_state or "").startswith(_tg_music.CUSTOM_STATE)):
            field = sess.reg_state[len(_tg_music.CUSTOM_STATE):]
            done = self._capture_preamble(chat_id, sess, lang, text, empty_abandons=True)
            if done is not None:
                return done
            n = _tg_music.parse_custom(field, text)
            lo, hi = _tg_music.custom_range(field)
            if n is None and len(text.split()) > 2 and any(c.isalpha() for c in text):
                # A sentence, not a mistyped number: the user moved on («нарисуй кота»).
                sess.reg_state = ""
                self._store.put(sess)
                return True
            if n is None:
                # Stays armed: a wrong number is answered, not abandoned.
                self._send_text(chat_id, tg_bot._t("ms_bad_number", lang, lo=lo, hi=hi))
                return False
            _tg_music.store_custom(sess, field, n)
            sess.reg_state = ""
            self._store.put(sess)
            shown = _tg_music.STEPS_HOME if field == "steps" else field
            self._send_text(chat_id, tg_bot._t("ms_custom_set", lang,
                                               what=_tg_music._current_label(sess, shown, lang)))
            if field == "steps":
                # Steps live inside the Quality screen; come back to it.
                self._send_text(chat_id, tg_bot._field_text(sess, shown, lang), parse_mode="HTML",
                                keyboard=tg_bot._field_kb(sess, shown, lang))
            else:
                self._send_text(chat_id, tg_bot._music_menu_text(sess, lang), parse_mode="HTML",
                                keyboard=tg_bot._music_menu_kb(sess, lang))
            self._activity.log(chat_id, "system", f"[song settings] {field}={n} (typed)")
            return False

        if user and user.status == "approved" and sess.reg_state == "song_topic":
            done = self._capture_preamble(chat_id, sess, lang, text)
            if done is not None:
                return done
            if not text:
                return False
            # Ready words (forwarded, or several short lines) are not a topic:
            # ask whether to sing them as they are or write new ones. Live
            # 2026-09-27 a forwarded storyboard was sung as a "topic" unasked.
            import tg_songs as _songs
            if tg_bot._is_forwarded(msg, self_is_own=False) or _songs.looks_like_lyrics(text):
                if getattr(sess, "song_draft", ""):
                    return False                # one question at a time: extra forwards wait
                sess.song_draft = text.strip()
                self._store.put(sess)
                self._send_text(chat_id, tg_bot._t("song_lyrics_choice", lang), keyboard={"inline_keyboard": [[
                    {"text": tg_bot._t("song_keep_btn", lang), "callback_data": "song_lyr:keep"},
                    {"text": tg_bot._t("song_new_btn", lang), "callback_data": "song_lyr:new"}]]})
                return False
            # Disarm NOW: while armed, every next message started another song
            # (live 2026-09-27: a stray multi-forward rendered two at once).
            sess.reg_state = ""
            self._store.put(sess)
            self._start_song_generation(chat_id, text.strip(), lang)
            return False

        if user and user.status == "approved" and sess.reg_state == "char_prompt":
            done = self._capture_preamble(chat_id, sess, lang, text)
            if done is not None:
                return done
            if not text and msg.get("photo"):
                # Photos sent in character mode are REFERENCES for the scene
                # (pose, outfit, framing, style), the caption is the scene.
                # They used to hit `if not text: return False` and vanish
                # without a word (live 2026-09-17 16:10, two stills of
                # American Psycho + «Нарисуй Степана так, чтобы он ... в этом
                # стиле» -- no reaction at all).
                self._collect_character_reference(chat_id, sess, lang, msg)
                return False
            if not text:
                return False
            # reg_state stays armed until the render thread finishes, same
            # contract as song_topic: _render_character is the one place that
            # clears it, so a failed render does not leave the user typing
            # scenes into a mode nobody is listening to.
            self._start_character_render(chat_id, sess, lang, text.strip())
            return False

        if user and user.is_admin and (sess.reg_state or "").startswith("admin_say:") and text:
            self._admin_take_say(chat_id, sess, text)
            return False

        if user and user.status == "approved" and sess.reg_state == "set_badge":
            # empty_abandons, because set_admin_badge("") is the CLEAR action.
            # A message with no text -- a photo, a sticker, a voice note -- sent
            # while 🎖 Custom badge was armed fell straight through to
            # set_admin_badge(chat_id, "".strip()), which happily removed the
            # admin's badge, returned True, and reported "🏷 Badge set: ." with
            # nothing after the colon. The message itself was swallowed too.
            # A photo is not an answer to "send me an emoji": drop the mode and
            # let the message be dispatched normally, exactly as feedback does.
            done = self._capture_preamble(chat_id, sess, lang, text,
                                          empty_abandons=True)
            if done is not None:
                return done
            if not user.is_admin or not self.set_admin_badge(chat_id, text.strip()):
                self._send_text(chat_id, tg_bot._t("badge_invalid", lang))
                return False
            sess.reg_state = ""
            self._store.put(sess)
            self._activity.log(chat_id, "system",
                f"[badge] admin {user.name} -> {text.strip()}", user.name)
            self._send_text(chat_id, tg_bot._t("badge_set", lang, badge=text.strip()),
                            keyboard=self._main_menu_kb(sess, lang))
            return False

        if user and user.status == "approved" and sess.reg_state in ("change_name", "change_password"):
            done = self._capture_preamble(chat_id, sess, lang, text)
            if done is not None:
                return done
            if sess.reg_state == "change_name":
                if not text or len(text.strip()) < 2:
                    self._send_text(chat_id, tg_bot._t("name_too_short", lang))
                    return False
                old_name = user.name
                user.name = text.strip()[:50]
                self._user_store.put(user)
                sess.reg_state = ""
                self._store.put(sess)
                self._activity.log(chat_id, "system",
                                   f"Name changed: {old_name} → {user.name}", user.name)
                self._send_text(chat_id,
                    tg_bot._t("name_updated", lang, name=_html_mod.escape(user.name)),
                    parse_mode="HTML",
                    keyboard=self._main_menu_kb(sess, lang))
                return False

            if sess.reg_state == "change_password":
                if not text or len(text.strip()) < 4:
                    # A rejected attempt is still a password attempt.
                    self._scrub_secret(chat_id, msg)
                    self._send_text(chat_id, tg_bot._t("pwd_short", lang), parse_mode="HTML")
                    return False
                self._scrub_secret(chat_id, msg)
                user.password_hash = tg_bot._hash_password(text.strip(), chat_id)
                self._user_store.put(user)
                sess.reg_state = ""
                self._store.put(sess)
                self._activity.log(chat_id, "system", "Password changed", user.name)
                self._send_text(chat_id, tg_bot._t("pass_updated", lang),
                    parse_mode="HTML",
                    keyboard=self._main_menu_kb(sess, lang))
                return False

        # ── EXISTING USER ────────────────────────────────────────────────────
        if user.status == "pending":
            self._send_text(chat_id, tg_bot._t("pending", lang), parse_mode="HTML")
            return False

        if user.status in ("rejected", "banned"):
            self._send_text(chat_id, tg_bot._t("revoked", lang))
            return False

        # Sync admin flag to session in case it was changed via GUI
        if sess.is_admin != user.is_admin:
            sess.is_admin = user.is_admin
            self._store.put(sess)

        return True   # approved user — continue normal dispatch


import tg_bot  # noqa: E402  (cycle by design; attrs read at call time)
import tg_music as _tg_music  # noqa: E402

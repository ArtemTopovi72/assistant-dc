"""Slash commands: everything a user can type starting with "/".

Split out of tg_dispatch.py, which kept growing three unrelated concerns in one
class: the long-poll/route loop, this command table, and batch resolution.

_handle_command reads every table and string it needs as ``tg_bot.<name>`` --
_t, _LANGS, _SLASH, _help_text and friends -- so each keeps a single definition
and a single patch point. Importing them by value here would give this module a
private copy, and a suite patching tg_bot._t would silently move only half the
behaviour."""
from __future__ import annotations

import html as _html_mod


class CommandsMixin:
    def _handle_command(self, chat_id: int, text: str):
        sess = self._get_session(chat_id)
        user = self._user_store.get(chat_id)
        lang = self._lang(sess)
        cmd  = text.split()[0].lower().split("@")[0]

        _arg = text.split(maxsplit=1)[1].strip() if len(text.split()) > 1 else ""
        if cmd == "/start" and _arg.startswith("f_"):
            self._send_sandbox_link(chat_id, sess, lang, user, _arg)
            return

        if cmd in ("/start", "/help"):
            self._send_text(chat_id,
                tg_bot._t("welcome", lang) + "\n\n" + tg_bot._help_text(lang),
                parse_mode="HTML",
                keyboard=self._main_menu_kb(sess, lang))
            return

        if cmd in ("/cancel", "/stop"):   # /stop: what people type (was "unknown command")
            # Same path as the ⛔ Stop button: also discards QUEUED work, which the
            # old cancel_event-only version left running after the current task died.
            self._stop_and_report(chat_id, sess, lang)
            return

        if cmd == "/clear":
            sess.clear_context(); self._store.put(sess)
            tg_bot._delete_chat_facts_file(chat_id)
            self._send_text(chat_id, tg_bot._t("cleared", lang),
                            keyboard=self._main_menu_kb(sess, lang))
            return

        if cmd == "/lang":
            arg = text[len("/lang"):].strip().lower()
            if arg in tg_bot._LANGS:
                sess.lang = arg; sess.lang_chosen = True
                sess.lang_pin = sess.turn_lang = ""; self._store.put(sess)
                self._send_text(chat_id, tg_bot._t("lang_set", arg), parse_mode="HTML",
                                keyboard=self._main_menu_kb(sess, arg))
            else:
                self._send_lang_menu(chat_id, lang)
            return

        if cmd == "/status":
            self._send_status(chat_id, sess)
            return

        if cmd == "/facts":
            self._send_facts(chat_id, sess)
            return

        if cmd == "/docs":
            self._send_library_list(chat_id, sess)
            return

        # --- the coding sandbox ---
        if cmd in ("/files", "/sandbox", "/reset_sandbox"):
            # "/files thief_unpacked/data" -- an unpacked modpack is hundreds of
            # files in a tree, and a listing that only ever shows the top level
            # is a listing of one folder name.
            self._handle_sandbox_command(chat_id, sess, user, lang, cmd,
                                         text[len(cmd):].strip())
            return

        if cmd == "/broadcast":
            if not (user and user.is_admin):
                return
            body = text[len("/broadcast"):].strip()
            if not body:
                self._send_text(chat_id, tg_bot._t("bcast_usage", lang), parse_mode="HTML")
                return
            self._pending_broadcast[chat_id] = body
            # Same "startup" subscription gate as _broadcast_startup/_shutdown —
            # /unsubscribe promises "unsubscribed from all notifications", so the
            # count shown here must match who actually receives it.
            n = sum(1 for u in self._user_store.approved()
                    if "startup" in (u.subscriptions or []))
            self._send_text(chat_id,
                tg_bot._t("bcast_confirm", lang, n=n,
                   body=_html_mod.escape(body[:500])),
                parse_mode="HTML",
                keyboard={"inline_keyboard": [
                    [{"text": tg_bot._t("bcast_go_btn", lang), "callback_data": "bcast:go"}],
                    [{"text": tg_bot._t("bcast_no_btn", lang), "callback_data": "bcast:no"}]]})
            return

        if cmd == "/voice":
            sess.voice_on = not sess.voice_on; self._store.put(sess)
            state = tg_bot._t("on", lang) if sess.voice_on else tg_bot._t("off", lang)
            self._send_text(chat_id, tg_bot._t("voice_state", lang, state=state),
                            parse_mode="HTML",
                            keyboard=self._main_menu_kb(sess, lang))
            return

        if cmd == "/size":
            self._send_text(chat_id, tg_bot._size_menu_text(sess, lang),
                            parse_mode="HTML", keyboard=tg_bot._size_menu_kb(sess, lang))
            return

        if cmd == "/depth":
            # "/depth deep" sets it outright; a bare "/depth" opens the picker.
            arg = text[len("/depth"):].strip().lower()
            if arg in tg_bot._DEPTHS:
                sess.dr_depth = arg; self._store.put(sess)
                self._send_text(chat_id,
                                tg_bot._t("depth_set", lang, name=tg_bot._depth_name(arg, lang),
                                   eta=tg_bot._depth_eta_text(arg, lang)),
                                parse_mode="HTML")
            else:
                self._send_text(chat_id, tg_bot._depth_menu_text(sess, lang),
                                parse_mode="HTML",
                                keyboard=tg_bot._depth_menu_kb(sess, lang))
            return

        if cmd == "/settings":
            vl = tg_bot._t("on", lang) if sess.voice_on else tg_bot._t("off", lang)
            bk = self._backend.name()
            sub = ", ".join(tg_bot._t("set_notif_" + s, lang) if s == "startup" else _html_mod.escape(s)
                            for s in user.subscriptions) if user and user.subscriptions else tg_bot._t("set_notif_none", lang)
            self._send_text(chat_id,
                f"<b>{tg_bot._b('settings', lang)}</b>\n"
                # Every line names its setting — an unlabelled "• 🎙 OFF ❌" is
                # just a dot, a mic and a cross, and reads as a rendering glitch.
                f"• {tg_bot._t('voice_state', lang, state=vl)}\n"
                f"• {tg_bot._b('lang', lang)}: <b>{lang}</b>\n"
                f"• {tg_bot._size_summary(sess, lang)}\n"
                # The backend class name is operator detail, not a user setting.
                + (f"• {tg_bot._t('set_queue', lang)}: <code>{_html_mod.escape(bk)}</code>\n"
                   if sess.is_admin else "")
                + f"• {tg_bot._t('set_notif', lang)}: <b>{sub}</b>",
                parse_mode="HTML",
                keyboard=tg_bot._settings_kb(sess.reply_mode, sess.is_admin, lang))
            return

        if cmd == "/subscribe":
            if user:
                if "startup" not in user.subscriptions:
                    user.subscriptions.append("startup")
                    self._user_store.put(user)
                self._send_text(chat_id, tg_bot._t("subscribed", lang),
                    parse_mode="HTML",
                    keyboard=self._main_menu_kb(sess, lang))
            return

        if cmd == "/unsubscribe":
            if user:
                user.subscriptions = [s for s in user.subscriptions if s != "startup"]
                self._user_store.put(user)
                self._send_text(chat_id, tg_bot._t("unsubscribed", lang),
                    parse_mode="HTML",
                    keyboard=self._main_menu_kb(sess, lang))
            return

        if cmd == "/feedback":
            arg = text[len("/feedback"):].strip()
            if not arg:
                sess.reg_state = "feedback"
                self._store.put(sess)
                self._send_text(chat_id, tg_bot._t("feedback_ask", lang), parse_mode="HTML")
            else:
                self._save_feedback(chat_id, user, arg)
                self._send_text(chat_id, tg_bot._t("feedback_ok", lang),
                    parse_mode="HTML",
                    keyboard=self._main_menu_kb(sess, lang))
            return

        if cmd in ("/account", "/setname"):
            if user:
                sess.reg_state = "change_name"
                self._store.put(sess)
                self._send_text(chat_id,
                    tg_bot._t("ask_new_name", lang, name=_html_mod.escape(user.name)),
                    parse_mode="HTML")
            return

        if cmd == "/setpassword":
            if user:
                sess.reg_state = "change_password"
                self._store.put(sess)
                self._send_text(chat_id, tg_bot._t("ask_new_pass", lang),
                                parse_mode="HTML")
            return

        # The panel exists and has its own button, but an admin's first instinct
        # is to type the word — and typing it did nothing at all.
        if cmd == "/admin":
            if sess.is_admin:
                self._send_admin_panel(chat_id)
            else:
                self._send_text(chat_id, tg_bot._t("admins_only", lang))
            return

        prefix = tg_bot._SLASH.get(cmd)
        if prefix:
            payload = text[len(cmd):].strip()
            if payload:
                self._enqueue_item(chat_id, {"type": "text", "text": prefix + payload})
            else:
                # _main_menu_kb clears the prefix it was just given: a bare /draw
                # then «кот на велосипеде» got the text echoed back, no picture.
                self._goto_menu(sess, self._PREFIX_MENU.get(prefix, ""), prefix)
                self._send_text(chat_id,
                    tg_bot._t("prompt_hint", lang,
                       hint=_html_mod.escape(tg_bot._prompt_label(prefix, lang))),
                    parse_mode="HTML",
                    keyboard=self._state_kb(sess, lang))
            return

        # Anything else beginning with "/" fell off the end of this function and
        # the user got SILENCE — a typo ("/serch") and a dead bot look exactly
        # the same from the chat. Say so, and show what does exist.
        self._send_text(chat_id,
                        tg_bot._t("unknown_cmd", lang,
                           cmd=_html_mod.escape(cmd[:32])) + tg_bot._help_text(lang),
                        parse_mode="HTML",
                        keyboard=self._main_menu_kb(sess, lang))

    # -- coding sandbox -------------------------------------------------------

    def _handle_sandbox_command(self, chat_id, sess, user, lang, cmd, path=""):
        """/files, /sandbox and /reset_sandbox.

        All three answer from the SAME grant check, so a user without it gets
        one consistent "not enabled" rather than learning the shape of the
        feature from which command answers differently.
        """
        import sandbox_access as _sa

        if not _sa.may_use_files(user):
            self._send_text(chat_id, tg_bot._t("sbx_denied", lang),
                            keyboard=self._main_menu_kb(sess, lang))
            return

        import code_sandbox as _cs
        box = _cs.sandbox_for(chat_id)

        # Every answer carries the how-to. The mode has no switch to flip -- it
        # is simply on for a granted account -- and that was the confusing part:
        # people looked for a "code mode" button, found a folder listing, and
        # had no idea what to do next. Sending a file IS the entry point, so
        # say so, every time, wherever they arrive from.
        how = tg_bot._t("sbx_how", lang)

        if cmd == "/reset_sandbox":
            removed = box.reset()
            self._send_text(chat_id,
                            tg_bot._t("sbx_reset", lang, n=removed) + how,
                            keyboard=self._main_menu_kb(sess, lang))
            return

        if cmd == "/sandbox":
            import code_runner as _cr
            files, total = box.usage()
            _backend, _why = _cr.backend_status()
            self._send_text(chat_id,
                            tg_bot._t("sbx_status", lang,
                                      access=_sa.describe(user), files=files,
                                      kb=total // 1024, backend=_why) + how,
                            keyboard=self._main_menu_kb(sess, lang))
            return

        # /files [path]
        self._send_sandbox_listing(chat_id, sess, lang, box, (path or ".").strip())

    # How many entries get a button. Telegram allows ~100 buttons per keyboard;
    # past a screenful they are unusable anyway, and /files <folder> narrows.
    _SBX_BUTTONS = 40

    def _send_sandbox_listing(self, chat_id, sess, lang, box, where, msg_id=None):
        """The folder as one button per entry: a file is sent on a tap, a
        folder opens. The paths live in _sbx_lists, keyed by chat -- a
        callback_data is capped at 64 bytes and a real path is not."""
        how = tg_bot._t("sbx_how", lang)
        try:
            entries = box.list_dir(where)
        except Exception as exc:
            # The sandbox's own wording, which already names what is nearby --
            # a mistyped folder answers with the folders that DO exist rather
            # than a bare "no".
            self._send_text(chat_id, str(exc)[:600] + how,
                            keyboard=self._main_menu_kb(sess, lang))
            return
        if not entries and where in (".", ""):
            self._send_text(chat_id,
                            tg_bot._t("sbx_empty", lang) + how,
                            parse_mode="HTML",
                            keyboard=self._main_menu_kb(sess, lang))
            return
        items = []                      # (relative path, is_dir, label)
        for e in entries:
            if e.startswith("…"):      # list_dir's own truncation marker
                continue
            if e.endswith("/"):
                rel = e[:-1]
                items.append((rel, True, "📁 " + rel.rsplit("/", 1)[-1]))
            else:
                rel, _, size = e.rpartition(" (")
                n = int(size.split(" ", 1)[0]) if size[:1].isdigit() else 0
                items.append((rel, False, f"📄 {rel.rsplit('/', 1)[-1]} · {_human_size(n)}"))
        shown = items[:self._SBX_BUTTONS]
        lists = self.__dict__.setdefault("_sbx_lists", {})
        lists[chat_id] = {"where": where, "items": [(r, d) for r, d, _ in shown]}
        rows = [[{"text": label[:60], "callback_data": f"sbx:{i}"}]
                for i, (_, _, label) in enumerate(shown)]
        if where not in (".", ""):
            rows.append([{"text": tg_bot._t("sbx_up", lang), "callback_data": "sbx:up"}])
        where_label = "" if where in (".", "") else " · " + _html_mod.escape(where)
        text = tg_bot._t("sbx_title", lang) + where_label + tg_bot._t("sbx_tap", lang)
        if len(items) > len(shown):
            text += tg_bot._t("sbx_more", lang, n=len(items) - len(shown))
        text += how
        kb = {"inline_keyboard": rows}
        if msg_id:
            self._edit_text(chat_id, msg_id, text, parse_mode="HTML", keyboard=kb)
        else:
            self._send_text(chat_id, text, parse_mode="HTML", keyboard=kb)

    def _bot_username(self) -> str:
        """@name of this bot for t.me deep links, asked once."""
        if not getattr(self, "_bot_uname", ""):
            self._bot_uname = ((self._api_post("getMe").get("result") or {}).get("username") or "")
        return self._bot_uname

    def _sandbox_linkify(self, chat_id, reply_html: str) -> str:
        """File names from this user's sandbox in a reply -> tap-to-download links."""
        try:
            import sandbox_access as _sa
            if not _sa.may_use_files(self._user_store.get(chat_id)):
                return reply_html
            import code_sandbox as _cs
            import tg_sandbox_links as _sl
            return _sl.linkify(reply_html, chat_id, _cs.sandbox_for(chat_id), self._bot_username())
        except Exception:
            tg_bot.logger.exception("sandbox linkify failed chat=%s", chat_id)
            return reply_html

    def _send_sandbox_link(self, chat_id, sess, lang, user, tok: str) -> None:
        """A tapped file link (/start f_<hash>): send that file from THIS user's sandbox."""
        import sandbox_access as _sa
        import code_sandbox as _cs
        import tg_sandbox_links as _sl
        if not _sa.may_use_files(user):
            self._send_text(chat_id, tg_bot._t("sbx_denied", lang))
            return
        box = _cs.sandbox_for(chat_id)
        rel = _sl.find(box, chat_id, tok)
        full = box.resolve(rel) if rel else None      # the sandbox's own escape gate
        if full is None or not full.is_file():
            self._send_text(chat_id, tg_bot._t("sbx_stale", lang))
            return
        self._api_post("sendChatAction", {"chat_id": chat_id, "action": "upload_document"})
        if not self._send_document(chat_id, str(full)):
            self._send_text(chat_id, tg_bot._t("sbx_send_fail", lang,
                                               name=_html_mod.escape(rel.rsplit("/", 1)[-1])))

    def _cb_sandbox(self, chat_id, msg, data):
        """A tap on a /files button: open the folder in place, or send the file."""
        import sandbox_access as _sa
        import code_sandbox as _cs
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        if not _sa.may_use_files(self._user_store.get(chat_id)):
            self._send_text(chat_id, tg_bot._t("sbx_denied", lang))
            return
        box = _cs.sandbox_for(chat_id)
        listing = self.__dict__.setdefault("_sbx_lists", {}).get(chat_id)
        msg_id = msg.get("message_id")
        arg = data.split(":", 1)[1]
        if listing is None:
            # The bot restarted since the list was drawn: the numbers no longer
            # mean anything, so redraw instead of guessing.
            self._send_text(chat_id, tg_bot._t("sbx_stale", lang))
            self._send_sandbox_listing(chat_id, sess, lang, box, ".")
            return
        if arg == "up":
            parent = listing["where"].rstrip("/").rpartition("/")[0] or "."
            self._send_sandbox_listing(chat_id, sess, lang, box, parent, msg_id=msg_id)
            return
        try:
            rel, is_dir = listing["items"][int(arg)]
        except (ValueError, IndexError):
            self._send_sandbox_listing(chat_id, sess, lang, box, listing["where"], msg_id=msg_id)
            return
        if is_dir:
            self._send_sandbox_listing(chat_id, sess, lang, box, rel, msg_id=msg_id)
            return
        try:
            full = box.resolve(rel)       # the sandbox's own escape gate
        except Exception:
            full = None
        name = _html_mod.escape(rel.rsplit("/", 1)[-1])
        if full is None or not full.is_file():
            self._send_text(chat_id, tg_bot._t("sbx_stale", lang))
            self._send_sandbox_listing(chat_id, sess, lang, box, listing["where"])
            return
        self._api_post("sendChatAction", {"chat_id": chat_id, "action": "upload_document"})
        if not self._send_document(chat_id, str(full)):
            self._send_text(chat_id, tg_bot._t("sbx_send_fail", lang, name=name))


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


import tg_bot  # noqa: E402  (cycle by design; attrs read at call time)

"""Operator actions on accounts, and the admin surface in chat.

One subject -- what an operator can DO to an account, and what an admin
sees. (The registration gate lives in tg_registration.py.)

  * approve / reject / ban / make_admin / revoke_admin, called from the GUI
    and from the inline admin buttons, plus the DB backup listing and restore.
  * the admin badge shown beside an admin's name.
  * the account menu a user sees for themselves, logout, and the broadcast
    confirmation, stats and admin panel.

approve/reject/ban take `allow_from_any_status` because the GUI and the chat
buttons disagree on purpose: a button on a never-expiring inline keyboard may
only act on a still-pending account, while an operator at the GUI may move an
account from any state to any other.

Shared constants, the _User record and the translation helper are read through
``tg_bot`` rather than imported by value, so they keep a single definition and
stay patchable from one place.
"""
from __future__ import annotations

import html as _html_mod
import time
from pathlib import Path
from seams import module


class AccountsMixin:
    def get_users(self) -> list[tg_bot._User]:
        return self._user_store.all()

    def approve_user(self, chat_id: int, *, allow_from_any_status: bool = False) -> str:
        """Move a user to 'approved'. Returns a short outcome code:
        'ok', 'not_found', or 'not_pending' (status guard — see caller for
        why un-banning must be an explicit separate action, not a side
        effect of a stale approval button)."""
        user = self._user_store.get(chat_id)
        if not user: return "not_found"
        if not allow_from_any_status and user.status not in ("pending", "approved"):
            return "not_pending"
        user.status = "approved"
        self._user_store.put(user)
        sess = self._get_session(chat_id)
        sess.is_admin = user.is_admin
        # A self-service edit mode armed before this user was banned or
        # rejected is stale once they are "approved" again: reg_state survives
        # status transitions, and _user_gate's edit-mode branch only checks
        # status == "approved". Without this, arming 🔑 Change password,
        # getting banned mid-flow and re-approved from the GUI would swallow
        # the next ordinary message as the new password. Cleared here, at the
        # one choke point every path back to "approved" goes through, rather
        # than in ban_user/reject_user (the session may not be loaded).
        if getattr(sess, "reg_state", "") in ("change_name", "change_password",
                                              "feedback", "wtw_city", "wtw_date",
                                              "set_badge", "song_topic",
                                              "char_prompt") or str(getattr(sess, "reg_state", "")).startswith("sch_"):
            sess.reg_state = ""
        self._store.put(sess)
        lang = self._lang(sess)
        try:
            self._send_text(chat_id,
                tg_bot._t("acct_approved", lang) + tg_bot._help_text(lang),
                parse_mode="HTML",
                keyboard=self._main_menu_kb(sess, lang))
        except Exception: pass
        self._activity.log(chat_id, "system", "Account approved", user.name)
        self._on_user_change(chat_id, user.name, "approved")
        return "ok"

    def reject_user(self, chat_id: int, *, allow_from_any_status: bool = False) -> str:
        """Move a user to 'rejected'. Returns 'ok', 'not_found', or
        'not_pending' (status guard).

        Inline admin buttons never expire, so a stale admin_reject: press on
        an old panel message must not downgrade an applicant another admin
        has since approved. Reject decides on a pending applicant (ban_user
        revokes an approved one), so the Telegram callback path only allows
        it from 'pending'. The desktop GUI's explicit Reject opts in via
        allow_from_any_status=True, as with approve_user.
        """
        user = self._user_store.get(chat_id)
        if not user: return "not_found"
        if not allow_from_any_status and user.status != "pending":
            return "not_pending"
        user.status = "rejected"
        self._user_store.put(user)
        self._request_stop(chat_id)
        lang = self._lang(self._get_session(chat_id))
        try:
            self._send_text(chat_id, tg_bot._t("acct_rejected", lang), parse_mode="HTML")
        except Exception: pass
        self._activity.log(chat_id, "system", "Account rejected", user.name)
        self._on_user_change(chat_id, user.name, "rejected")
        return "ok"

    def ban_user(self, chat_id: int, *, allow_from_any_status: bool = False) -> str:
        """Move a user to 'banned'. Returns 'ok', 'not_found', or
        'not_banned_already' (status guard).

        Same contract as approve_user/reject_user: a missing user is reported
        as 'not_found', and a user already banned is left alone rather than
        re-banned (no second acct_banned, log entry or _on_user_change); the
        GUI's explicit Ban action opts in via allow_from_any_status=True.
        """
        user = self._user_store.get(chat_id)
        if not user: return "not_found"
        if not allow_from_any_status and user.status == "banned":
            return "not_banned_already"
        user.status = "banned"
        self._user_store.put(user)
        self._request_stop(chat_id)
        lang = self._lang(self._get_session(chat_id))
        try:
            self._send_text(chat_id, tg_bot._t("acct_banned", lang))
        except Exception: pass
        self._activity.log(chat_id, "system", "Account banned", user.name)
        self._on_user_change(chat_id, user.name, "banned")
        return "ok"

    def make_admin(self, chat_id: int):
        user = self._user_store.get(chat_id)
        if not user: return
        user.is_admin = True
        self._user_store.put(user)
        sess = self._get_session(chat_id)
        sess.is_admin = True
        self._store.put(sess)
        try:
            self._send_text(chat_id,
                tg_bot._t("acct_admin_ok", self._lang(sess)),
                parse_mode="HTML", keyboard=self._main_menu_kb(sess))
        except Exception: pass
        self._on_user_change(chat_id, user.name, "admin")

    def revoke_admin(self, chat_id: int):
        """Demote an admin back to a regular approved user. GUI-only, like
        make_admin -- there is no Telegram-side admin_revoke: callback, so an
        admin can never demote themselves or another admin from inside a chat."""
        user = self._user_store.get(chat_id)
        if not user or not user.is_admin: return
        user.is_admin = False
        self._user_store.put(user)
        sess = self._get_session(chat_id)
        sess.is_admin = False
        self._store.put(sess)
        try:
            self._send_text(chat_id,
                tg_bot._t("acct_admin_revoked", self._lang(sess)),
                parse_mode="HTML", keyboard=self._main_menu_kb(sess))
        except Exception: pass
        self._on_user_change(chat_id, user.name, "admin_revoked")

    def _send_badge_menu(self, chat_id: int, user, sess) -> None:
        lang = self._lang(sess)
        buttons = [{"text": e, "callback_data": f"acct_badge_set:{e}"}
                  for e in tg_bot.ADMIN_BADGE_SUGGESTIONS]
        rows = [buttons[i:i + 4] for i in range(0, len(buttons), 4)]
        rows.append([{"text": tg_bot._t("badge_custom_btn", lang),
                      "callback_data": "acct_badge_custom"}])
        if (user.prefs or {}).get("badge"):
            rows.append([{"text": tg_bot._t("badge_clear_btn", lang),
                          "callback_data": "acct_badge_clear"}])
        self._send_text(chat_id, tg_bot._t("badge_menu_title", lang),
                        parse_mode="HTML", keyboard={"inline_keyboard": rows})

    def set_admin_badge(self, chat_id: int, badge: str) -> bool:
        """Set (or clear, with badge="") an admin's badge. Returns False if the
        chat isn't a known admin or the badge fails the emoji check."""
        user = self._user_store.get(chat_id)
        if not user or not user.is_admin:
            return False
        if badge and not tg_bot._looks_like_emoji(badge):
            return False
        prefs = dict(user.prefs or {})
        if badge:
            prefs["badge"] = badge
        else:
            prefs.pop("badge", None)
        user.prefs = prefs
        self._user_store.put(user)
        return True

    def list_db_backups(self) -> list[Path]:
        """Return list of backup snapshot paths, newest first."""
        return self._user_store.list_backups()

    def restore_db_from_backup(self, backup_path) -> bool:
        """Restore the user DB from a snapshot. Returns True on success."""
        return self._user_store.restore_from_backup(backup_path)

    def _send_account_menu(self, chat_id: int, user, sess):
        if not user:
            # The session outlived the account record -- a purge, a restored
            # backup, a deleted row. Returning silently made the 👤 Аккаунт
            # button look broken, with no way to find out that re-registering
            # is the fix.
            self._send_text(chat_id, tg_bot._t("acct_gone", self._lang(sess)))
            return
        sub = "startup" in user.subscriptions
        name_safe = _html_mod.escape(user.name)
        uname_safe = f"@{_html_mod.escape(user.tg_username)}" if user.tg_username else "—"
        status_icon = {"approved": "✅", "pending": "⏳", "rejected": "❌",
                       "banned": "⛔"}.get(user.status, "👤")
        lang = self._lang(sess)
        mark = lambda on: tg_bot._t("on_mark" if on else "off_mark", lang)
        text = (
            tg_bot._t("acct_title", lang) + "\n"
            + tg_bot._t("acct_name", lang, name=name_safe) + "\n"
            + tg_bot._t("acct_handle", lang, handle=uname_safe) + "\n"
            + tg_bot._t("acct_status", lang, icon=status_icon,
                 state=tg_bot._t(f"st_{user.status}", lang)) + "\n"
            + tg_bot._t("acct_notif", lang, state=mark(sub)) + "\n"
        )
        fb_on = tg_bot._wants_feedback(user)
        if user.is_admin:
            text += tg_bot._t("acct_feedback", lang, state=mark(fb_on)) + "\n"
            badge = (user.prefs or {}).get("badge") or ""
            text += tg_bot._t("acct_badge", lang,
                       badge=badge if badge else tg_bot._t("badge_none", lang)) + "\n"
        # Usage is part of the account, not a hidden server-side counter — a user
        # who hits a limit should be able to see it coming.
        used = self._user_store.usage_today(chat_id)
        quota_bits = []
        for k in (tg_bot.KIND_TASK, tg_bot.KIND_IMAGE, tg_bot.KIND_RESEARCH):
            limit = tg_bot._quota_limit(k)
            if limit > 0 and not user.is_admin:
                quota_bits.append(f"{tg_bot._kind_label(k, lang)} {used.get(k, 0)}/{limit}")
        if quota_bits:
            text += tg_bot._t("acct_today", lang, quotas=" · ".join(quota_bits)) + "\n"
        text += tg_bot._t("acct_ask", lang)
        rows = [
            [{"text": tg_bot._t("acct_name_btn", lang), "callback_data": "acct_setname"}],
            [{"text": tg_bot._t("acct_pass_btn", lang), "callback_data": "acct_setpwd"}],
            [{"text": tg_bot._t("acct_lang_btn", lang), "callback_data": "acct_lang"}],
            [{"text": tg_bot._t("acct_notif_btn", lang),
              "callback_data": "acct_toggle_startup"}],
        ]
        if user.is_admin:
            # Only admins receive feedback at all, so the toggle is meaningless for
            # everyone else — don't show a control that cannot change anything.
            rows.append([{"text": tg_bot._t("acct_mute_btn" if fb_on
                                     else "acct_unmute_btn", lang),
                          "callback_data": "acct_toggle_feedback"}])
            rows.append([{"text": tg_bot._t("acct_badge_btn", lang),
                          "callback_data": "acct_badge_menu"}])
        rows.append([{"text": tg_bot._t("acct_logout_btn", lang),
                      "callback_data": "acct_logout"}])
        from tg_strings import _nav_back_row
        rows.append(_nav_back_row(lang))   # every account screen needs a way back
        self._send_text(chat_id, text, parse_mode="HTML",
                        keyboard={"inline_keyboard": rows})

    def _logout(self, chat_id: int, user, sess) -> None:
        """Lock this chat behind the account password again.

        Deliberately NON-destructive: the account row is kept, only the local session
        (history, per-session memory, pending prefix) is wiped and the chat is put into
        `awaiting_login`. A Telegram account maps 1:1 to a chat_id, so "switch profile"
        cannot mean "hold two accounts at once" — it means re-authenticating, or
        explicitly replacing this profile, which is offered as a separate confirmed
        action rather than done silently by a Log-out tap.
        """
        if not user:
            self._send_text(chat_id, tg_bot._t("acct_gone", self._lang(sess)))
            return
        self._request_stop(chat_id)
        sess.clear_history()
        sess.set_tg_memory([])
        sess.pending_prefix = ""
        sess.reg_state = "awaiting_login"
        sess.reg_name = ""
        self._store.put(sess)
        # The facts the assistant pinned about this person are session data too —
        # leaving them behind would let the next login inherit them.
        sess.set_tg_facts([])
        self._store.put(sess)
        # ...and so is the write-only disk shadow remember_fact left in
        # tg_memory/chat_<id>/facts.json — logout must not leave it behind.
        tg_bot._delete_chat_facts_file(chat_id)
        self._activity.log(chat_id, "system", "[auth] logged out", user.name)
        lang = self._lang(sess)
        self._send_text(chat_id,
            tg_bot._t("logged_out", lang, name=_html_mod.escape(user.name)),
            parse_mode="HTML",
            keyboard={"inline_keyboard": [[
                {"text": tg_bot._t("other_profile_btn", lang),
                 "callback_data": "acct_newprofile"}]]})

    def _confirm_broadcast(self, chat_id: int, decision: str) -> None:
        body = self._pending_broadcast.pop(chat_id, "")
        user = self._user_store.get(chat_id)
        if not (user and user.is_admin) or not body:
            return
        lang = self._lang(self._get_session(chat_id))
        if decision != "go":
            self._send_text(chat_id, tg_bot._t("bcast_cancelled", lang))
            return
        # Announcing to every user is outward-facing and irreversible, so it only
        # runs after the explicit confirmation above — never straight off /broadcast.
        sent = failed = 0
        for u in self._user_store.approved():
            # Mirror _broadcast_startup/_broadcast_shutdown: honour the same
            # "startup" subscription opt-out /unsubscribe promises, so a manual
            # /broadcast doesn't reach users who turned notifications off.
            if "startup" not in (u.subscriptions or []):
                continue
            try:
                self._send_text(u.chat_id,
                                tg_bot._t("bcast_announce",
                                   self._lang(self._get_session(u.chat_id)),
                                   body=_html_mod.escape(body)),
                                parse_mode="HTML")
                sent += 1
            except Exception:
                failed += 1
        self._activity.log(chat_id, "system",
                           f"[broadcast] sent={sent} failed={failed}", user.name)
        self._send_text(chat_id, tg_bot._t("bcast_sent", lang, sent=sent, failed=failed))

    def _send_admin_stats(self, chat_id: int) -> None:
        lang = self._lang(self._get_session(chat_id))
        rows = self._user_store.usage_totals(7)
        if not rows:
            self._send_text(chat_id, tg_bot._t("stats_none", lang))
            return
        per_user: dict = {}
        for cid, kind, count in rows:
            per_user.setdefault(cid, {})[kind] = count
        lines = [tg_bot._t("adm_stats_title", lang)]
        ranked = sorted(per_user.items(),
                        key=lambda kv: kv[1].get(tg_bot.KIND_TASK, 0), reverse=True)
        for cid, kinds in ranked[:25]:
            u = self._user_store.get(cid)
            name = u.name if u else str(cid)
            lines.append(
                f"• <b>{_html_mod.escape(name)}</b> — "
                f"{kinds.get(tg_bot.KIND_TASK, 0)} {tg_bot._kind_label(tg_bot.KIND_TASK, lang)} · "
                f"{kinds.get(tg_bot.KIND_IMAGE, 0)} {tg_bot._kind_label(tg_bot.KIND_IMAGE, lang)} · "
                f"{kinds.get(tg_bot.KIND_RESEARCH, 0)} {tg_bot._kind_label(tg_bot.KIND_RESEARCH, lang)}")
        self._send_text(chat_id, "\n".join(lines), parse_mode="HTML")

    def _training_line(self, lang: str) -> str:
        """One line about the LoRA run that owns the card, for the admin panel.

        Reads the same lora_training.progress() the desktop tab does -- disk
        only, no process handle -- so it reports a run started from anywhere,
        including a previous session.
        """
        left = "осталось ~" if lang == "ru" else "left ~"
        try:
            _C = module(self, "characters")
            LT = module(self, "lora_training")
            latest = None
            for rec in _C.list_characters():
                p = LT.progress(rec["slug"])
                kind = LT.run_state(p)
                if kind != LT.RUN_RUNNING:
                    # Remember the most advanced non-live run, so an operator
                    # who checks after the fact learns whether it finished or
                    # died rather than reading "не идёт" for both.
                    # A run with no recorded step count cannot be judged --
                    # run_state has to call it "stopped", and reporting
                    # "ОСТАНОВЛЕНО на шаге 1147/0" reads as a bug. Skip it.
                    # Only a RECENT one: a run stopped weeks ago sat in the
                    # admin header as «Обучение: neurostepan…» and read as a
                    # live run with the GPU at 0%.
                    try:
                        _age = time.time() - LT.log_path(rec["slug"]).stat().st_mtime
                    except OSError:
                        _age = 0.0          # age unknown: do not hide it
                    if kind in (LT.RUN_DONE, LT.RUN_STOPPED) and p.get("total") and _age < 12 * 3600 and (
                            latest is None or p["step"] > latest[1]["step"]):
                        latest = (kind, p)
                    continue
                total = p.get("total") or 0
                pct = int(100.0 * p["step"] / total) if total else 0
                return tg_bot._t(
                    "adm_training", lang,
                    slug=_html_mod.escape(str(p.get("slug") or rec["slug"])),
                    step=p["step"], total=total, pct=pct,
                    eta=(" · %s%s" % (left, LT.human_eta(p["eta"]))) if p.get("eta") else "",
                    rate=(" · %s" % p["rate"]) if p.get("rate") else "")
            if latest is not None:
                kind, p = latest
                return tg_bot._t(
                    "adm_training_done" if kind == LT.RUN_DONE
                    else "adm_training_stopped", lang,
                    slug=_html_mod.escape(str(p.get("slug") or "")),
                    step=p["step"], total=p.get("total") or 0)
        except Exception:
            tg_bot.logger.warning("admin panel: training line failed", exc_info=True)
        return tg_bot._t("adm_training_idle", lang)

    def _send_admin_panel(self, chat_id: int):
        # The live console (tg_admin): one message, refreshed in place.
        self._admin_show(chat_id)


import tg_bot  # noqa: E402  (cycle by design; attrs read at call time)

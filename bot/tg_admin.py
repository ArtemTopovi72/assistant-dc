"""🔐 Admin console: one live picture of the bot and the controls to steer it.

The same snapshot feeds the Telegram panel (one message, edited in place every
few seconds while it is open) and the desktop «Админка» tab, so the two can
never disagree about what is running.

Actions: cancel a running or queued request, serve a queued one next, stop
everything in a chat, write to a user as the administration, ban / unban.
Every action is logged to the activity log with the admin's name.
"""
from __future__ import annotations

import html as _h
import subprocess
import threading
import time
from typing import Optional

_LIVE_S = 600            # a panel keeps refreshing this long after the last press
_TICK_S = 4.0            # Telegram rate limit: edits under 1/s per chat are safe
_TEXT_CUT = 70
_gpu_cache = {"t": 0.0, "v": ""}


def _short(text: str, n: int = _TEXT_CUT) -> str:
    t = " ".join((text or "").split())
    if t.startswith("[song"):
        t = "🎵 " + t.split("]", 1)[-1].strip()
    elif t.lower().startswith("do a deep research on:"):
        t = "🔬 " + t.split(":", 1)[-1].strip()
    return t if len(t) <= n else t[: n - 1] + "…"


def _dur(s: float) -> str:
    s = int(max(0, s))
    return f"{s // 60}:{s % 60:02d}" if s < 3600 else f"{s // 3600}ч{s % 3600 // 60:02d}"


def gpu_line() -> str:
    """«🎮 RTX 3090 · 19.2/24.0 ГБ · 87%», cached for a few seconds."""
    if time.time() - _gpu_cache["t"] < 5:
        return _gpu_cache["v"]
    v = ""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip().splitlines()[0]
        name, used, total, util = [x.strip() for x in out.split(",")]
        v = f"🎮 {name.replace('NVIDIA GeForce ', '')} · {float(used)/1024:.1f}/{float(total)/1024:.1f} ГБ · {util}%"
    except Exception:
        pass
    _gpu_cache.update(t=time.time(), v=v)
    return v


class AdminMixin:
    # ── the one snapshot ─────────────────────────────────────────────────────
    def admin_snapshot(self) -> dict:
        now_m, now = time.monotonic(), time.time()
        with self._task_lock:
            running_pairs = [(cid, t) for cid, lst in self._running_task.items() for t in lst
                             if not getattr(t, "delivered", False)]
            started = dict(self._task_started)
        with self._stages_lock:
            stages = dict(self._active_stages)
        names = {u.chat_id: u for u in self._user_store.all()}
        name = lambda cid: (names[cid].name if cid in names and names[cid].name else str(cid))
        running = [{"task_id": t.task_id, "chat_id": cid, "user": name(cid),
                    "text": _short(t.user_text), "stage": stages.get(cid, "") or stages.get(str(cid), ""),
                    "elapsed": now_m - started.get(t.task_id, now_m)} for cid, t in running_pairs]
        try:
            order = self._backend.service_order()
        except Exception:
            order = []
        queued = [{"task_id": t.task_id, "chat_id": t.chat_id, "user": name(t.chat_id),
                   "text": _short(t.user_text), "waited": now - (t.enqueue_ts or now)} for t in order]
        users = []
        for u in names.values():
            try:
                today = self._user_store.usage_today(u.chat_id)
            except Exception:
                today = {}
            users.append({"chat_id": u.chat_id, "name": u.name, "username": u.tg_username,
                          "status": u.status, "is_admin": u.is_admin,
                          "today": int(sum(v for v in today.values() if isinstance(v, (int, float)))
                                       if today else 0)})
        users.sort(key=lambda r: (r["status"] != "approved", -r["today"], r["name"].lower()))
        try:
            import tg_bot
            training = self._training_line("ru")
            if training == tg_bot._t("adm_training_idle", "ru"):
                training = ""           # nothing to say: keep the header clean
        except Exception:
            training = ""
        return {"running": running, "queued": queued, "users": users,
                "backend": self._backend.name(), "gpu": gpu_line(), "training": training,
                "ts": time.strftime("%H:%M:%S")}

    # ── actions (TG panel and desktop tab share these) ───────────────────────
    def _admin_log(self, admin: str, what: str, chat_id: int = 0) -> None:
        self._activity.log(chat_id, "system", f"[admin {admin}] {what}", admin)
        import tg_bot
        tg_bot.logger.info("admin %s: %s", admin, what)

    def _lang_of(self, chat_id: int) -> str:
        return self._lang(self._get_session(int(chat_id)))

    def admin_cancel(self, task_id: str, admin: str = "admin") -> str:
        import tg_bot
        snap = self.admin_snapshot()
        row = next((r for r in snap["running"] + snap["queued"] if r["task_id"] == task_id), None)
        if row is None:
            return "gone"
        res = self._cancel_task(row["chat_id"], task_id)
        if res == "dropped":
            self._send_text(row["chat_id"], tg_bot._t("ax_user_cancelled", self._lang_of(row["chat_id"]),
                                                      text=row["text"]))
        self._admin_log(admin, f"cancel {task_id[:8]} ({row['user']}: {row['text']}) -> {res or 'gone'}",
                        row["chat_id"])
        return res or "gone"

    def admin_bump(self, task_id: str, admin: str = "admin") -> bool:
        ok = bool(self._backend.bump(task_id))
        self._admin_log(admin, f"bump {task_id[:8]} -> {ok}")
        return ok

    def admin_stop_chat(self, chat_id: int, admin: str = "admin") -> int:
        import tg_bot
        dropped = self._request_stop(int(chat_id))
        self._send_text(int(chat_id), tg_bot._t("ax_user_stopped", self._lang_of(chat_id)))
        self._admin_log(admin, f"stop chat, dropped {dropped}", int(chat_id))
        return dropped

    def admin_say(self, chat_id: int, text: str, admin: str = "admin") -> bool:
        import tg_bot
        text = (text or "").strip()
        if not text:
            return False
        self._send_text(int(chat_id), tg_bot._t("ax_user_say", self._lang_of(chat_id), text=_h.escape(text)),
                        parse_mode="HTML")
        self._admin_log(admin, f"say: {text[:300]}", int(chat_id))
        return True

    def admin_set_status(self, chat_id: int, status: str, admin: str = "admin") -> str:
        if status == "banned":
            res = self.ban_user(int(chat_id), allow_from_any_status=True)
        else:
            res = self.approve_user(int(chat_id), allow_from_any_status=True)
        self._admin_log(admin, f"status -> {status}: {res}", int(chat_id))
        return res

    # ── Telegram: one live message per admin ─────────────────────────────────
    def _admin_panels(self) -> dict:
        if not hasattr(self, "_adm_live"):
            self._adm_live = {}          # admin chat -> {"msg", "view", "until", "last"}
            self._adm_lock = threading.Lock()
            threading.Thread(target=self._admin_refresh_loop, daemon=True,
                             name="tg-admin-live").start()
        return self._adm_live

    def _admin_render(self, view: str, lang: str = "ru") -> tuple:
        import tg_bot
        t = lambda k, **kw: tg_bot._t(k, lang, **kw)
        s = self.admin_snapshot()
        if view.startswith("u:"):
            return self._admin_render_user(s, int(view[2:]), t)
        if view == "users":
            return self._admin_render_users(s, t)
        L = [t("adm_title").strip() + "   <i>🔄 %s</i>" % s["ts"]]
        L += [x for x in (s["gpu"], s["training"]) if x]
        L.append("")
        kb = []
        L.append(t("ax_running", n=len(s["running"])))
        for i, r in enumerate(s["running"], 1):
            L.append("  %d. <b>%s</b> · ⏱ %s · %s\n      <i>%s</i>" % (
                i, _h.escape(r["user"]), _dur(r["elapsed"]), _h.escape(r["stage"] or "…"),
                _h.escape(r["text"])))
            kb.append([{"text": t("ax_btn_cancel", i=i, u=r["user"][:14]),
                        "callback_data": "adm:c:" + r["task_id"]},
                       {"text": "✉️", "callback_data": "adm:say:%d" % r["chat_id"]}])
        if not s["running"]:
            L.append(t("ax_quiet"))
        L.append("")
        L.append(t("ax_queue", n=len(s["queued"])))
        for i, q in enumerate(s["queued"][:10], 1):
            L.append("  %d. <b>%s</b> · %s\n      <i>%s</i>" % (
                i, _h.escape(q["user"]), t("ax_waits", t=_dur(q["waited"])), _h.escape(q["text"])))
            row = [{"text": "✖ #%d" % i, "callback_data": "adm:c:" + q["task_id"]}]
            if i > 1:
                row.insert(0, {"text": t("ax_btn_first", i=i), "callback_data": "adm:b:" + q["task_id"]})
            kb.append(row)
        if len(s["queued"]) > 10:
            L.append(t("ax_more", n=len(s["queued"]) - 10))
        if not s["queued"]:
            L.append(t("ax_empty"))
        pend = [u for u in s["users"] if u["status"] == "pending"]
        L.append("")
        L.append(t("ax_users_line", a=sum(u["status"] == "approved" for u in s["users"]), p=len(pend),
                   b=sum(u["status"] == "banned" for u in s["users"]), be=_h.escape(s["backend"])))
        L.append(t("adm_bcast_tip").strip())
        if len(pend) > 5:                    # the rest wait in 👥 — say so, or they are invisible
            L.append(t("ax_more", n=len(pend) - 5))
        for u in pend[:5]:
            kb.append([{"text": "✅ " + u["name"][:20], "callback_data": "admin_approve:%d" % u["chat_id"]},
                       {"text": "❌", "callback_data": "admin_reject:%d" % u["chat_id"]}])
        kb.append([{"text": t("ax_btn_users"), "callback_data": "adm:v:users"},
                   {"text": t("ax_btn_stats"), "callback_data": "adm:stats"}])
        kb.append([{"text": "🔄", "callback_data": "adm:v:main"},
                   {"text": t("ax_btn_close"), "callback_data": "adm:close"}])
        return "\n".join(L), {"inline_keyboard": kb}

    def _admin_render_users(self, s: dict, t) -> tuple:
        icon = {"approved": "🟢", "pending": "🟡", "banned": "🔴"}
        busy = {r["chat_id"] for r in s["running"]} | {q["chat_id"] for q in s["queued"]}
        L = [t("ax_users_title") + "   <i>🔄 %s</i>" % s["ts"], ""]
        kb, row = [], []
        for u in s["users"][:30]:
            L.append("%s%s <b>%s</b>%s · %s%s" % (
                icon.get(u["status"], "⚪"), "👑" if u["is_admin"] else "", _h.escape(u["name"] or "—"),
                (" @" + _h.escape(u["username"])) if u["username"] else "", t("ax_today", n=u["today"]),
                " · ⚡" if u["chat_id"] in busy else ""))
            row.append({"text": "%s %s" % (icon.get(u["status"], "⚪"), (u["name"] or str(u["chat_id"]))[:16]),
                        "callback_data": "adm:v:u:%d" % u["chat_id"]})
            if len(row) == 2:
                kb.append(row); row = []
        if row:
            kb.append(row)
        kb.append([{"text": t("ax_btn_back"), "callback_data": "adm:v:main"},
                   {"text": t("ax_btn_close"), "callback_data": "adm:close"}])
        return "\n".join(L), {"inline_keyboard": kb}

    def _admin_render_user(self, s: dict, cid: int, t) -> tuple:
        u = next((x for x in s["users"] if x["chat_id"] == cid), None)
        if u is None:
            return t("ax_not_found"), {"inline_keyboard": [[{"text": "⬅️", "callback_data": "adm:v:users"}]]}
        mine_r = [r for r in s["running"] if r["chat_id"] == cid]
        mine_q = [q for q in s["queued"] if q["chat_id"] == cid]
        L = ["👤 <b>%s</b>%s   <i>🔄 %s</i>" % (_h.escape(u["name"] or "—"),
                                             (" @" + _h.escape(u["username"])) if u["username"] else "", s["ts"]),
             t("ax_card_meta", id=cid, st=u["status"], adm=t("ax_admin_mark") if u["is_admin"] else "",
               n=u["today"]), ""]
        for r in mine_r:
            L.append("⚡ ⏱ %s · %s\n   <i>%s</i>" % (_dur(r["elapsed"]), _h.escape(r["stage"] or "…"),
                                                   _h.escape(r["text"])))
        for q in mine_q:
            L.append("⏳ %s · <i>%s</i>" % (t("ax_waits", t=_dur(q["waited"])), _h.escape(q["text"])))
        if not (mine_r or mine_q):
            L.append(t("ax_idle_user"))
        kb = [[{"text": t("ax_btn_say"), "callback_data": "adm:say:%d" % cid}]]
        if mine_r or mine_q:
            kb.append([{"text": t("ax_btn_stopall"), "callback_data": "adm:s:%d" % cid}])
        kb.append([{"text": t("ax_btn_unban"), "callback_data": "adm:ok:%d" % cid} if u["status"] == "banned"
                   else {"text": t("ax_btn_ban"), "callback_data": "adm:ban:%d" % cid}])
        kb.append([{"text": t("ax_btn_users"), "callback_data": "adm:v:users"},
                   {"text": t("ax_btn_home"), "callback_data": "adm:v:main"}])
        return "\n".join(L), {"inline_keyboard": kb}

    def _admin_show(self, chat_id: int, view: str = "main", msg_id: Optional[int] = None) -> None:
        live = self._admin_panels()
        body, kb = self._admin_render(view, self._lang_of(chat_id))
        if msg_id is None:
            msg_id = self._send_get_id(chat_id, body, parse_mode="HTML", keyboard=kb)
        else:
            self._edit_text(chat_id, msg_id, body, parse_mode="HTML", keyboard=kb)
        if msg_id:
            with self._adm_lock:
                live[chat_id] = {"msg": msg_id, "view": view, "until": time.time() + _LIVE_S,
                                 "last": body.split("\n", 1)[-1]}

    def _admin_refresh_loop(self) -> None:
        import tg_bot
        while getattr(self, "_running", True):
            time.sleep(_TICK_S)
            with self._adm_lock:
                panels = list(self._adm_live.items())
            for cid, p in panels:
                try:
                    lang, mid = self._lang_of(cid), p["msg"]
                    body, kb = self._admin_render(p["view"], lang)
                    if time.time() > p["until"]:
                        self._edit_text(cid, mid, body + tg_bot._t("ax_paused", lang),
                                        parse_mode="HTML", keyboard=kb)
                        with self._adm_lock:
                            self._adm_live.pop(cid, None)
                        continue
                    rest = body.split("\n", 1)[-1]          # the clock line alone is no reason to edit
                    if rest != p["last"]:
                        self._edit_text(cid, mid, body, parse_mode="HTML", keyboard=kb)
                        p["last"] = rest
                except Exception:
                    tg_bot.logger.debug("admin panel refresh failed", exc_info=True)

    def _cb_admin_panel(self, chat_id: int, cb: dict, data: str) -> None:
        import tg_bot
        lang = self._lang_of(chat_id)
        t = lambda k, **kw: tg_bot._t(k, lang, **kw)
        _answer = lambda msg="": self._api_post("answerCallbackQuery",
                                                {"callback_query_id": cb.get("id", ""), "text": msg})
        presser = (cb.get("from") or {}).get("id")
        actor = self._user_store.get(presser) if presser else None
        if not (actor and actor.is_admin and actor.status == "approved"):
            tg_bot.logger.warning("rejected admin panel callback %r from %s", data, presser)
            _answer(t("ax_only_admins"))
            return
        admin = actor.name or str(presser)
        msg_id = (cb.get("message") or {}).get("message_id")
        view = (self._admin_panels().get(chat_id) or {}).get("view", "main")
        cmd, _, arg = data[4:].partition(":")
        toast = ""
        if cmd != "say":                     # any other panel press drops a pending ✉️
            _s = self._get_session(chat_id)
            if str(getattr(_s, "reg_state", "")).startswith("admin_say:"):
                _s.reg_state = ""
                self._store.put(_s)
        if cmd == "close":
            _answer()
            with self._adm_lock:
                self._adm_live.pop(chat_id, None)
            self._api_post("deleteMessage", {"chat_id": chat_id, "message_id": msg_id})
            return
        if cmd == "v":
            view = arg
        elif cmd == "stats":
            self._send_admin_stats(chat_id)
        elif cmd == "c":
            toast = {"running": t("ax_t_stopping"), "dropped": t("ax_t_dropped")}.get(
                self.admin_cancel(arg, admin), t("ax_t_gone"))
        elif cmd == "b":
            toast = t("ax_t_next") if self.admin_bump(arg, admin) else t("ax_t_notq")
        elif cmd == "s":
            toast = t("ax_t_stopped", n=self.admin_stop_chat(int(arg), admin))
        elif cmd in ("ban", "ok"):
            self.admin_set_status(int(arg), "banned" if cmd == "ban" else "approved", admin)
            toast = t("ax_t_banned") if cmd == "ban" else t("ax_t_unbanned")
        elif cmd == "say":
            sess = self._get_session(chat_id)
            sess.reg_state = "admin_say:" + arg
            self._store.put(sess)
            u = self._user_store.get(int(arg))
            self._send_text(chat_id, tg_bot._t("ax_say_prompt", lang, name=_h.escape(u.name if u else arg)), parse_mode="HTML")
        _answer(toast)
        self._admin_show(chat_id, view, msg_id)

    def _admin_take_say(self, chat_id: int, sess, text: str) -> bool:
        """The admin's next text after ✉️: deliver it. True when consumed."""
        import tg_bot
        state = getattr(sess, "reg_state", "") or ""
        if not state.startswith("admin_say:"):
            return False
        lang = self._lang(sess)
        user = self._user_store.get(chat_id)
        sess.reg_state = ""
        self._store.put(sess)
        # A menu button or a command is the admin changing their mind, not the message.
        if (text or "").startswith("/") or text in tg_bot._LABEL2KEY or not (user and user.is_admin):
            self._send_text(chat_id, tg_bot._t("ax_say_cancel", lang))
            return True
        self.admin_say(int(state.split(":", 1)[1]), text, user.name or str(chat_id))
        self._send_text(chat_id, tg_bot._t("ax_say_sent", lang))
        return True

"""Regression tests for four confirmed admin-panel bugs (audit_08_admin.md):

1. Approving a chat id that does not exist used to report success anyway
   (`approve_user` returned silently, the caller announced success
   unconditionally). Now `approve_user`/`reject_user` return an outcome code
   and the caller sends an honest "no such user" reply.
2. `admin_approve:<id>` used to silently UN-BAN a banned/rejected account
   with no status guard. Now approval only proceeds from pending/approved;
   anything else is refused with an explicit "not pending" reply, and the
   desktop GUI opts in explicitly via allow_from_any_status=True.
3. `/broadcast` ignored each recipient's subscription opt-out while the
   startup/shutdown broadcasts filtered correctly. `_confirm_broadcast` now
   mirrors that same "startup" subscription filter.
4. Only the first 5 pending users were ever actionable, with no indication
   more existed. The admin panel now appends a count of the hidden rest.

Run: venv/Scripts/python.exe tests/test_admin_fixes.py
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T

_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_adminfix_")
T.redirect_data_dir(_DATA_DIR)

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")


def make_bot():
    bot = T.TelegramBot("123:TEST", lambda: None, lambda: None, lambda: {}, silent_mode=True)
    sent = []
    def rec(method, payload=None, **kw):
        sent.append((method, payload))
        return {"ok": True, "result": {"message_id": len(sent)}}
    bot._api_post = rec
    return bot, sent


def cb_update(chat_id, data, presser_id=None, msg_id=555):
    cbq = {"id": "cbq1", "data": data,
           "message": {"chat": {"id": chat_id}, "message_id": msg_id}}
    if presser_id is not None:
        cbq["from"] = {"id": presser_id}
    return {"callback_query": cbq}


def set_user(bot, chat_id, status, is_admin=False, subscriptions=None):
    u = T._User(chat_id=chat_id, name=f"U{chat_id}", tg_username="u",
                password_hash="x", status=status, is_admin=is_admin,
                subscriptions=list(subscriptions or []))
    bot._user_store.put(u)
    bot._get_session(chat_id)
    return u


print("=" * 70)
print("1. Approving a NONEXISTENT chat id reports failure, not success")
print("=" * 70)

bot, sent = make_bot()
ADMIN = 1001
set_user(bot, ADMIN, "approved", is_admin=True)
TARGET = 999999  # never registered
sent.clear()
bot._dispatch(cb_update(ADMIN, f"admin_approve:{TARGET}", presser_id=ADMIN))
texts = [str(p) for _, p in sent]
check("nonexistent target: no success message sent",
      not any("approved" in t.lower().replace("не", "") for t in texts) or
      any("no longer exists" in t or "больше не существует" in t for t in texts),
      texts)
check("nonexistent target: an honest 'not found' reply was sent",
      any("no longer exists" in t or "больше не существует" in t for t in texts), texts)
check("nonexistent target: still does not exist in the store",
      bot._user_store.get(TARGET) is None)

print()
print("=" * 70)
print("2. A stale admin_approve: on a BANNED user does not silently un-ban them")
print("=" * 70)

bot, sent = make_bot()
ADMIN = 1002
set_user(bot, ADMIN, "approved", is_admin=True)
BANNED = 4001
set_user(bot, BANNED, "banned")
sent.clear()
bot._dispatch(cb_update(ADMIN, f"admin_approve:{BANNED}", presser_id=ADMIN))
fresh = bot._user_store.get(BANNED)
check("banned user's status is still 'banned' after a stale approve press",
      fresh.status == "banned", fresh.status)
texts = [str(p) for _, p in sent]
check("admin was told the approval did not happen",
      any("not_pending".replace("_", " ") in t.lower() or "не изменено" in t or
          "banned" in t.lower() or "banned".replace("banned", "") for t in texts) or
      any("no change" in t.lower() or "не изменено" in t for t in texts), texts)
check("outcome code from approve_user itself is 'not_pending' for a banned user",
      bot.approve_user(BANNED) == "not_pending")
check("desktop GUI path (allow_from_any_status=True) can still explicitly un-ban",
      bot.approve_user(BANNED, allow_from_any_status=True) == "ok")
check("...and after the explicit override, status really did flip",
      bot._user_store.get(BANNED).status == "approved")

print()
print("=" * 70)
print("2b. Rejected user is likewise protected from a stale approve")
print("=" * 70)

bot, sent = make_bot()
ADMIN = 1003
set_user(bot, ADMIN, "approved", is_admin=True)
REJECTED = 4002
set_user(bot, REJECTED, "rejected")
outcome = bot.approve_user(REJECTED)
check("rejected user: approve_user refuses ('not_pending')", outcome == "not_pending", outcome)
check("rejected user: status untouched",
      bot._user_store.get(REJECTED).status == "rejected")

print()
print("=" * 70)
print("2c. Pending -> approved still works normally (no over-refusal)")
print("=" * 70)

bot, sent = make_bot()
ADMIN = 1004
set_user(bot, ADMIN, "approved", is_admin=True)
PENDING = 4003
set_user(bot, PENDING, "pending")
sent.clear()
bot._dispatch(cb_update(ADMIN, f"admin_approve:{PENDING}", presser_id=ADMIN))
check("pending user is approved normally through the callback",
      bot._user_store.get(PENDING).status == "approved")

print()
print("=" * 70)
print("3. /broadcast (manual admin command) honours the subscription opt-out")
print("=" * 70)

bot, sent = make_bot()
ADMIN = 1005
set_user(bot, ADMIN, "approved", is_admin=True)
SUBSCRIBED = 6001
set_user(bot, SUBSCRIBED, "approved", subscriptions=["startup"])
UNSUBSCRIBED = 6002
set_user(bot, UNSUBSCRIBED, "approved", subscriptions=[])
bot._pending_broadcast[ADMIN] = "Announcement text"
sent.clear()
bot._confirm_broadcast(ADMIN, "go")
recipients = [p.get("chat_id") for m, p in sent if m == "sendMessage" and p]
check("subscribed recipient received the broadcast", SUBSCRIBED in recipients, recipients)
check("unsubscribed recipient did NOT receive the broadcast",
      UNSUBSCRIBED not in recipients, recipients)

print()
print("=" * 70)
print("3b. The confirmation prompt's recipient count matches who actually gets it")
print("=" * 70)

bot, sent = make_bot()
ADMIN = 1106
set_user(bot, ADMIN, "approved", is_admin=True)
set_user(bot, 6103, "approved", subscriptions=["startup"])
set_user(bot, 6104, "approved", subscriptions=["startup"])
set_user(bot, 6105, "approved", subscriptions=[])
n_subscribed = sum(1 for u in bot._user_store.approved()
                   if "startup" in (u.subscriptions or []))
n_all_approved = len(bot._user_store.approved())
check("test setup sanity: subscribed count is strictly less than all-approved count",
      n_subscribed < n_all_approved, (n_subscribed, n_all_approved))
sent.clear()
bot._dispatch({"message": {"chat": {"id": ADMIN}, "from": {"id": ADMIN},
                            "text": "/broadcast hello everyone"}})
texts = [str(p) for _, p in sent]
check(f"confirm prompt shows the SUBSCRIBED count ({n_subscribed}), "
      f"not the all-approved count ({n_all_approved})",
      any(f">{n_subscribed}</b>" in t for t in texts) and
      not any(f">{n_all_approved}</b>" in t for t in texts), texts)

print()
print("=" * 70)
print("4. Admin panel discloses hidden pending users beyond the first 5")
print("=" * 70)

bot, sent = make_bot()
ADMIN = 1007
set_user(bot, ADMIN, "approved", is_admin=True)
for i in range(8):
    set_user(bot, 7000 + i, "pending")
sent.clear()
bot._send_admin_panel(ADMIN)
texts = [str(p) for _, p in sent]
check("panel mentions the 3 hidden pending users (8 total - 5 shown)",
      any("3" in t and ("more" in t.lower() or "ещё" in t) for t in texts), texts)

print()
print(f"{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)

"""Admins must NOT receive feedback/service notifications until they opt in.

Before this fix, _wants_feedback() defaulted every admin to subscribed and
required an explicit mute -- a freshly promoted admin got paged with every
piece of user feedback the moment they were promoted. This flips the default
and locks in both directions of the toggle plus the legacy-opt-out migration
path (an admin who muted under the OLD scheme must not come back subscribed
just because the marker's meaning changed).

Run: venv/Scripts/python.exe tests/test_tg_feedback_default.py
"""
import sys, os, types, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import tg_bot as T

_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_fbdef_")
T.redirect_data_dir(_DATA_DIR)

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print(f"PASS  {name}")
    else:
        BAD += 1; print(f"FAIL  {name}   {extra}")


def make_bot():
    bot = T.TelegramBot("123:TEST", lambda: object(), lambda: object(),
                        lambda: {}, silent_mode=True)
    bot.sent = []
    bot._send_text = lambda cid, text, **kw: (bot.sent.append(text), 1)[1]
    bot._activity.log = lambda *a, **k: None
    bot._backend = types.SimpleNamespace(
        push=lambda t: None, depth=lambda: 0, name=lambda: "stub",
        drop_chat=lambda cid: 0, close=lambda: None)
    return bot


def cb_update(chat_id, data, msg_id=555):
    return {"callback_query": {
        "id": "cbq1", "data": data, "from": {"id": chat_id},
        "message": {"chat": {"id": chat_id}, "message_id": msg_id}}}


bot = make_bot()
CID = 999701

print("=" * 66)
print("DEFAULT STATE")
print("=" * 66)

non_admin = T._User(chat_id=1, name="Reg", status="approved", is_admin=False)
fresh_admin = T._User(chat_id=2, name="NewAdmin", status="approved", is_admin=True)
check("a non-admin never wants feedback regardless of subscriptions",
      T._wants_feedback(non_admin) is False)
check("a freshly promoted admin (empty subscriptions) is MUTED by default",
      T._wants_feedback(fresh_admin) is False)

print()
print("=" * 66)
print("LEGACY OPT-OUT MARKER STILL HONOURED")
print("=" * 66)

legacy_muted = T._User(chat_id=3, name="OldAdmin", status="approved",
                        is_admin=True, subscriptions=[T.FEEDBACK_OPT_OUT])
check("an admin muted under the old opt-out scheme stays muted",
      T._wants_feedback(legacy_muted) is False)

print()
print("=" * 66)
print("EXPLICIT OPT-IN")
print("=" * 66)

subscribed_admin = T._User(chat_id=4, name="SubAdmin", status="approved",
                            is_admin=True, subscriptions=[T.FEEDBACK_OPT_IN])
check("an admin who explicitly opted in receives feedback",
      T._wants_feedback(subscribed_admin) is True)

print()
print("=" * 66)
print("TOGGLE ROUND-TRIP VIA THE REAL CALLBACK")
print("=" * 66)

bot._user_store.put(T._User(chat_id=CID, name="Admin", status="approved", is_admin=True))
bot._dispatch(cb_update(CID, "acct_toggle_feedback"))
u = bot._user_store.get(CID)
check("first tap subscribes a fresh admin (was muted by default)",
      T._wants_feedback(u) is True, repr(u.subscriptions))

bot._dispatch(cb_update(CID, "acct_toggle_feedback"))
u = bot._user_store.get(CID)
check("second tap mutes again",
      T._wants_feedback(u) is False, repr(u.subscriptions))
check("muting drops the opt-in marker entirely (no stale leftover)",
      T.FEEDBACK_OPT_IN not in (u.subscriptions or []), repr(u.subscriptions))

# Round-trip from the OTHER starting point: an admin already muted under the
# legacy scheme taps the button -- must end up subscribed, not toggle back
# into a still-muted state via the old marker.
bot._user_store.put(T._User(chat_id=CID, name="Admin", status="approved",
                            is_admin=True, subscriptions=[T.FEEDBACK_OPT_OUT]))
bot._dispatch(cb_update(CID, "acct_toggle_feedback"))
u = bot._user_store.get(CID)
check("tapping from a legacy-muted state subscribes cleanly",
      T._wants_feedback(u) is True, repr(u.subscriptions))
check("the legacy opt-out marker is cleared once subscribed",
      T.FEEDBACK_OPT_OUT not in (u.subscriptions or []), repr(u.subscriptions))

print()
print(f"{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)

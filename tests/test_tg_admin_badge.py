"""Admin badge: a per-admin customizable emoji shown next to their name.

Not a fixed/hardcoded assignment -- ADMIN_BADGE_SUGGESTIONS is just a set of
quick-pick buttons; any admin can send an arbitrary emoji via "Custom..." and
it's stored in the new generic User.prefs dict (so future personalizations
don't need their own schema migration). Covers: storage round-trip through
SQLite (incl. the ALTER TABLE migration path for a pre-existing DB), the
inline-button picker, free-text entry, validation, non-admin gating, and the
clear action.

Run: venv/Scripts/python.exe tests/test_tg_admin_badge.py
"""
import sys, os, types, tempfile, sqlite3
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import tg_bot as T
import tg_userstore as U

_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_badge_")
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


def msg(cid, text, mid=1):
    return {"chat": {"id": cid}, "from": {"id": cid, "username": "u"},
            "message_id": mid, "text": text}


print("=" * 66)
print("STORAGE: SQLite round-trip and legacy-DB migration")
print("=" * 66)

d = Path(tempfile.mkdtemp(prefix="tgtest_badge_store_"))
us = U._UserStore(d / "u.db", backup_dir=d / "b", legacy_json=d / "nope.json")
u = U._User(chat_id=1, name="A", is_admin=True, prefs={"badge": "owl"})
us.put(u)
check("prefs round-trips through SQLite", us.get(1).prefs == {"badge": "owl"})

d2 = Path(tempfile.mkdtemp(prefix="tgtest_badge_legacy_"))
conn = sqlite3.connect(str(d2 / "old.db"))
conn.execute("""CREATE TABLE users (
    chat_id INTEGER PRIMARY KEY, name TEXT NOT NULL DEFAULT '',
    tg_username TEXT NOT NULL DEFAULT '', password_hash TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending', registered_at REAL NOT NULL DEFAULT 0,
    subscriptions TEXT NOT NULL DEFAULT '[]', is_admin INTEGER NOT NULL DEFAULT 0)""")
conn.execute("INSERT INTO users (chat_id, name) VALUES (2, 'Old')")
conn.commit(); conn.close()
us2 = U._UserStore(d2 / "old.db", backup_dir=d2 / "b", legacy_json=d2 / "nope.json")
check("a pre-existing DB without the prefs column loads without error",
      us2.get(2) is not None)
check("its prefs default to an empty dict, not a crash",
      us2.get(2).prefs == {})
us2.put(U._User(chat_id=2, name="Old", is_admin=True, prefs={"badge": "star"}))
check("and can be written to after the migration",
      us2.get(2).prefs == {"badge": "star"})

print()
print("=" * 66)
print("VALIDATION")
print("=" * 66)

check("a curated suggestion validates as a single emoji",
      T._looks_like_emoji("🦉"))
check("plain text does not validate",
      not T._looks_like_emoji("hello"))
check("an empty string does not validate",
      not T._looks_like_emoji(""))
check("a whole sentence does not validate",
      not T._looks_like_emoji("this is my badge"))

print()
print("=" * 66)
print("set_admin_badge()")
print("=" * 66)

bot = make_bot()
bot._user_store.put(T._User(chat_id=100, name="Reg", status="approved", is_admin=False))
bot._user_store.put(T._User(chat_id=101, name="Admin", status="approved", is_admin=True))

check("a non-admin cannot be given a badge",
      bot.set_admin_badge(100, "🦉") is False)
check("an admin can be given any curated suggestion",
      bot.set_admin_badge(101, "🦉") is True)
check("the badge lands in prefs",
      bot._user_store.get(101).prefs.get("badge") == "🦉")
check("an admin can be given a badge NOT in the curated list (free choice)",
      bot.set_admin_badge(101, "🎯") is True)
check("...and it's stored verbatim",
      bot._user_store.get(101).prefs.get("badge") == "🎯")
check("garbage text is rejected",
      bot.set_admin_badge(101, "not an emoji") is False)
check("...and the previous valid badge survives a rejected attempt",
      bot._user_store.get(101).prefs.get("badge") == "🎯")
check("clearing (empty string) removes the badge key entirely",
      bot.set_admin_badge(101, "") is True and
      "badge" not in bot._user_store.get(101).prefs)
check("an unknown chat_id is handled without raising",
      bot.set_admin_badge(999999999, "🦉") is False)

print()
print("=" * 66)
print("PICKER CALLBACKS (real _dispatch)")
print("=" * 66)

CID = 999801
bot._user_store.put(T._User(chat_id=CID, name="Admin2", status="approved", is_admin=True))
bot._dispatch(cb_update(CID, "acct_badge_set:🔥"))
check("tapping a curated suggestion sets the badge",
      bot._user_store.get(CID).prefs.get("badge") == "🔥")

bot._dispatch(cb_update(CID, "acct_badge_clear"))
check("the Clear button removes it",
      "badge" not in bot._user_store.get(CID).prefs)

print()
print("=" * 66)
print("FREE-TEXT ENTRY VIA THE MESSAGE PATH")
print("=" * 66)

bot._dispatch(cb_update(CID, "acct_badge_custom"))
check("Custom... arms set_badge on the session",
      bot._get_session(CID).reg_state == "set_badge")

bot._user_gate(CID, msg(CID, "🧙"))
check("typing a single emoji while armed sets the badge",
      bot._user_store.get(CID).prefs.get("badge") == "🧙")
check("and clears reg_state afterwards",
      bot._get_session(CID).reg_state == "")

bot._dispatch(cb_update(CID, "acct_badge_custom"))
bot._user_gate(CID, msg(CID, "not an emoji at all"))
check("garbage input while armed is rejected, not silently accepted",
      bot._user_store.get(CID).prefs.get("badge") == "🧙")   # unchanged
check("...and stays armed so the admin can retry",
      bot._get_session(CID).reg_state == "set_badge")

sess = bot._get_session(CID); sess.reg_state = "set_badge"; bot._store.put(sess)
bot._user_gate(CID, msg(CID, "/cancel"))
check("/cancel abandons the custom-badge prompt",
      bot._get_session(CID).reg_state == "")

# The "/start became the password" bug shape: only "/cancel" was special-
# cased, so any OTHER command (e.g. /broadcast) while armed for set_badge
# used to be rejected as "not a valid emoji" -- silently eating the command
# instead of abandoning the mode and letting it run.
_badge_before_cmd = bot._user_store.get(CID).prefs.get("badge")
sess = bot._get_session(CID); sess.reg_state = "set_badge"; bot._store.put(sess)
allowed = bot._user_gate(CID, msg(CID, "/broadcast hello everyone"))
check("a non-/cancel command abandons set_badge instead of being rejected as garbage",
      bot._get_session(CID).reg_state == "", bot._get_session(CID).reg_state)
check("...and is let through to run as a real command", allowed is True, repr(allowed))
check("...and the badge is left untouched",
      bot._user_store.get(CID).prefs.get("badge") == _badge_before_cmd)

# Same family, the case nobody had covered: a message with NO TEXT while
# set_badge is armed. set_admin_badge("") is the CLEAR action, so a photo fell
# through to it, removed the admin's badge, returned True, and was reported as
# "🏷 Badge set: ." with nothing after the colon -- and the photo itself was
# swallowed instead of being dispatched.
bot.set_admin_badge(CID, "🧙")
check("precondition: the admin has a badge to lose",
      bot._user_store.get(CID).prefs.get("badge") == "🧙")
sess = bot._get_session(CID); sess.reg_state = "set_badge"; bot._store.put(sess)
_sent_before = len(bot.sent)
photo_only = {"chat": {"id": CID}, "from": {"id": CID, "username": "u"},
              "message_id": 77, "photo": [{"file_id": "f1", "file_size": 100}]}
allowed = bot._user_gate(CID, photo_only)
check("a photo does not wipe the badge it was never an answer to",
      bot._user_store.get(CID).prefs.get("badge") == "🧙",
      bot._user_store.get(CID).prefs)
check("...it abandons the badge prompt", bot._get_session(CID).reg_state == "",
      bot._get_session(CID).reg_state)
check("...and the photo is let through to normal dispatch", allowed is True,
      repr(allowed))
check("...with no 'badge set' claim about an empty badge",
      not any(T._t("badge_set", "ru", badge="").rstrip(". ") in s
              for s in bot.sent[_sent_before:]), bot.sent[_sent_before:])

print()
print("=" * 66)
print("NON-ADMIN GATING")
print("=" * 66)

NID = 999802
bot._user_store.put(T._User(chat_id=NID, name="NotAdmin", status="approved", is_admin=False))
bot._dispatch(cb_update(NID, "acct_badge_menu"))
check("a non-admin pressing the badge menu is refused, not shown a picker",
      bot._user_store.get(NID).prefs == {})
bot._dispatch(cb_update(NID, "acct_badge_set:🦉"))
check("a non-admin cannot set a badge via a forged callback either",
      bot._user_store.get(NID).prefs == {})

print()
print("=" * 66)
print("ABANDON ON UNRELATED BUTTON PRESS")
print("=" * 66)

bot._dispatch(cb_update(CID, "acct_badge_custom"))
check("armed before the unrelated press", bot._get_session(CID).reg_state == "set_badge")
bot._dispatch(cb_update(CID, "acct_lang"))
check("an unrelated inline button abandons the armed custom-badge mode",
      bot._get_session(CID).reg_state == "")

print()
print(f"{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)

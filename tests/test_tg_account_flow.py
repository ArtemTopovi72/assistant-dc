"""Registration, login and the account-edit state machine.

`reg_state` decides how the NEXT free-text message is interpreted — as a name, as a
password, as feedback, or as an ordinary question. Getting it wrong is not cosmetic:
a stuck state silently turns whatever the user types next into their password (that
is what the "/start became the password" bug was), and every password step asks the
user to type a secret into a chat log that Telegram keeps forever.

Run: venv/Scripts/python.exe tests/test_tg_account_flow.py
"""
import sys, types, tempfile, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import tg_bot as T

# Before any bot exists — otherwise fixture users land in the LIVE store.
_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_acct_")
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
    bot.deleted = []
    bot._send_text = lambda cid, text, **kw: (bot.sent.append(text), 1)[1]
    bot._send_get_id = lambda cid, text, **kw: (bot.sent.append(text), 1)[1]
    bot._api_post = lambda method, payload=None, **k: (
        bot.deleted.append((payload or {}).get("message_id"))
        if method == "deleteMessage" else None) or {}
    bot._api_get = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    bot._backend = types.SimpleNamespace(
        push=lambda t: None, depth=lambda: 0, name=lambda: "stub",
        drop_chat=lambda cid: 0, close=lambda: None)
    return bot


def msg(cid, text, mid=1):
    return {"chat": {"id": cid}, "from": {"id": cid, "username": "u"},
            "message_id": mid, "text": text}


bot = make_bot()
CID = 999601

print("=" * 66)
print("REGISTRATION")
print("=" * 66)

s = bot._get_session(CID)
bot._user_gate(CID, msg(CID, "hello"))
check("a brand new chat is asked to register",
      bot._get_session(CID).reg_state == "awaiting_name",
      repr(bot._get_session(CID).reg_state))

bot._user_gate(CID, msg(CID, "Ivan"))
check("the name moves it to the password step",
      bot._get_session(CID).reg_state == "awaiting_password",
      repr(bot._get_session(CID).reg_state))

# The bug this flow was born from: a command is CONTROL, never a password.
for cmd in ("/start", "/help", "/settings"):
    bot._user_gate(CID, msg(CID, cmd))
    check(f"{cmd} is not accepted as the password",
          bot._get_session(CID).reg_state == "awaiting_password"
          and bot._user_store.get(CID) is None,
          f"state={bot._get_session(CID).reg_state!r} user={bot._user_store.get(CID)}")

bot.deleted.clear()
bot._user_gate(CID, msg(CID, "abc", mid=41))          # too short
check("a short password is rejected",
      bot._user_store.get(CID) is None and
      bot._get_session(CID).reg_state == "awaiting_password")
check("even a rejected password attempt is scrubbed from the chat",
      41 in bot.deleted, str(bot.deleted))

bot.deleted.clear()
bot._user_gate(CID, msg(CID, "hunter2!", mid=42))
u = bot._user_store.get(CID)
check("a valid password creates the account", u is not None)
check("the password is not stored in the clear",
      u is not None and "hunter2!" not in (u.password_hash or ""),
      repr(u.password_hash if u else None)[:80])
check("the password message is deleted from the chat", 42 in bot.deleted,
      str(bot.deleted))
check("registration state is cleared", bot._get_session(CID).reg_state == "")

# A recognized keyboard-button label is never a real name/password. Reachable
# even on a genuinely fresh registration via acct_newprofile_yes (an existing
# user re-registering with their OLD main-menu keyboard still visible on
# screen) -- a mis-tap there must not silently become the new account's name
# or password.
CID_BTN = 999603
bot._user_gate(CID_BTN, msg(CID_BTN, "hi"))
_menu_label = T._b("draw", T._DEFAULT_LANG)
bot._user_gate(CID_BTN, msg(CID_BTN, _menu_label))
check("a menu-button label is not accepted as the registration name",
      bot._get_session(CID_BTN).reg_state == "awaiting_name",
      bot._get_session(CID_BTN).reg_state)
bot._user_gate(CID_BTN, msg(CID_BTN, "Real Name"))
check("a real name after the mis-tap still works",
      bot._get_session(CID_BTN).reg_state == "awaiting_password",
      bot._get_session(CID_BTN).reg_state)
bot._user_gate(CID_BTN, msg(CID_BTN, _menu_label))
check("a menu-button label is not accepted as the registration password",
      bot._user_store.get(CID_BTN) is None
      and bot._get_session(CID_BTN).reg_state == "awaiting_password",
      (bot._user_store.get(CID_BTN), bot._get_session(CID_BTN).reg_state))
bot._user_gate(CID_BTN, msg(CID_BTN, "a-real-password"))
check("a real password after the mis-tap still creates the account",
      bot._user_store.get(CID_BTN) is not None, bot._user_store.get(CID_BTN))

# /cancel must abandon registration rather than leave a half-armed state.
CID2 = 999602
bot._user_gate(CID2, msg(CID2, "hi"))
bot._user_gate(CID2, msg(CID2, "/cancel"))
check("/cancel abandons registration",
      bot._get_session(CID2).reg_state == "" and bot._user_store.get(CID2) is None,
      repr(bot._get_session(CID2).reg_state))


print()
print("=" * 66)
print("LOGIN")
print("=" * 66)

u = bot._user_store.get(CID); u.status = "approved"; bot._user_store.put(u)
s = bot._get_session(CID)
s.reg_state = "awaiting_login"
bot._store.put(s)
bot.deleted.clear()
bot._user_gate(CID, msg(CID, "wrong-one", mid=50))
check("a wrong password does not log the user in",
      bot._get_session(CID).reg_state == "awaiting_login")
check("a wrong password is still scrubbed from the chat", 50 in bot.deleted,
      str(bot.deleted))

bot.deleted.clear()
bot._user_gate(CID, msg(CID, "hunter2!", mid=51))
check("the right password logs the user in",
      bot._get_session(CID).reg_state == "",
      repr(bot._get_session(CID).reg_state))
check("the login password is scrubbed too", 51 in bot.deleted, str(bot.deleted))

# Brute force must hit the lockout rather than run at machine speed.
s = bot._get_session(CID); s.reg_state = "awaiting_login"; bot._store.put(s)
bot._login_fails.pop(CID, None)
bot.sent.clear()
for i in range(12):
    bot._user_gate(CID, msg(CID, f"guess{i}", mid=60 + i))
# Anchor on the message the bot actually sends, not on a guessed wording — and in
# the language it actually sends it in. This said "en" while the house language is
# Russian (TG_DEFAULT_LANG), so it compared an English prefix against "⏱ Слишком
# много неудачных попыток…" and failed even though the lockout fired correctly.
_locked = T._t("login_locked", T._DEFAULT_LANG, sec=1).split("{")[0][:20]
check("repeated wrong passwords trigger the lockout",
      any(t.startswith(_locked[:12]) for t in bot.sent),
      str(bot.sent[-2:]))
check("a locked-out chat is still not logged in",
      bot._get_session(CID).reg_state == "awaiting_login")


print()
print("=" * 66)
print("ACCOUNT EDITS")
print("=" * 66)

s = bot._get_session(CID); s.reg_state = ""; bot._store.put(s)
bot._login_fails.pop(CID, None)
u = bot._user_store.get(CID); u.status = "approved"; bot._user_store.put(u)
_old_hash = u.password_hash

s.reg_state = "change_name"; bot._store.put(s)
bot._user_gate(CID, msg(CID, "X"))
check("a one-character name is rejected",
      bot._get_session(CID).reg_state == "change_name"
      and bot._user_store.get(CID).name != "X",
      bot._user_store.get(CID).name)
bot._user_gate(CID, msg(CID, "New Name"))
check("a valid name is applied", bot._user_store.get(CID).name == "New Name",
      bot._user_store.get(CID).name)
check("the name edit ends the state", bot._get_session(CID).reg_state == "")

s = bot._get_session(CID); s.reg_state = "change_password"; bot._store.put(s)
bot.deleted.clear()
bot._user_gate(CID, msg(CID, "ab", mid=69))
check("a too-short new password is refused",
      bot._user_store.get(CID).password_hash == _old_hash
      and bot._get_session(CID).reg_state == "change_password",
      repr(bot._get_session(CID).reg_state))
check("the refused attempt is scrubbed too", 69 in bot.deleted, str(bot.deleted))

bot.deleted.clear()
bot._user_gate(CID, msg(CID, "a-new-password", mid=70))
check("the password is actually changed",
      bot._user_store.get(CID).password_hash != _old_hash)
check("the new password never stays in the chat", 70 in bot.deleted,
      str(bot.deleted))
check("the password edit ends the state", bot._get_session(CID).reg_state == "")

s = bot._get_session(CID); s.reg_state = "change_password"; bot._store.put(s)
bot._user_gate(CID, msg(CID, "/cancel"))
check("/cancel leaves the password alone and ends the state",
      bot._get_session(CID).reg_state == "")

# The bug this whole guard exists for: ✏️ Change name / 🔑 Change password leave
# the OLD main-menu keyboard on screen (nothing replaces it), so a mis-tap on
# any of its still-live buttons used to be silently accepted as the new
# name/password -- e.g. the account's password silently became "🎨 Draw".
_hash_before_tap = bot._user_store.get(CID).password_hash
s = bot._get_session(CID); s.reg_state = "change_password"; bot._store.put(s)
bot._user_gate(CID, msg(CID, T._b("draw", T._DEFAULT_LANG)))
check("a menu-button label does not become the new password",
      bot._user_store.get(CID).password_hash == _hash_before_tap,
      "password was silently overwritten by a button label")
check("the mis-tap abandons change_password instead of leaving it armed",
      bot._get_session(CID).reg_state == "", bot._get_session(CID).reg_state)

_name_before_tap = bot._user_store.get(CID).name
s = bot._get_session(CID); s.reg_state = "change_name"; bot._store.put(s)
bot._user_gate(CID, msg(CID, T._b("draw", T._DEFAULT_LANG)))
check("a menu-button label does not become the new name",
      bot._user_store.get(CID).name == _name_before_tap,
      bot._user_store.get(CID).name)
check("the mis-tap abandons change_name too",
      bot._get_session(CID).reg_state == "", bot._get_session(CID).reg_state)

# A stuck edit state must not swallow ordinary conversation forever.
s = bot._get_session(CID); s.reg_state = ""; bot._store.put(s)
allowed = bot._user_gate(CID, msg(CID, "what is the weather"))
check("with no pending state an ordinary message passes the gate", allowed is True,
      repr(allowed))

# The "/start became the password" bug shape, recurring for the edit modes:
# only "/cancel" was special-cased, so any OTHER slash command (e.g. an admin
# typing /broadcast while armed for change_password) got silently swallowed
# as the answer text instead of running. /broadcast's own body is >=4 chars,
# so it silently became the account's new password with no error at all.
_hash_before_cmd = bot._user_store.get(CID).password_hash
s = bot._get_session(CID); s.reg_state = "change_password"; bot._store.put(s)
allowed = bot._user_gate(CID, msg(CID, "/broadcast hello everyone, this is a test"))
check("a non-/cancel command abandons change_password instead of becoming the password",
      bot._user_store.get(CID).password_hash == _hash_before_cmd,
      "password was silently overwritten by a command's own text!")
check("...and is let through to run as a real command",
      allowed is True, repr(allowed))
check("...and the state is cleared, not left armed",
      bot._get_session(CID).reg_state == "", bot._get_session(CID).reg_state)

_name_before_cmd = bot._user_store.get(CID).name
s = bot._get_session(CID); s.reg_state = "change_name"; bot._store.put(s)
allowed = bot._user_gate(CID, msg(CID, "/settings"))
check("a non-/cancel command does not become the new name either",
      bot._user_store.get(CID).name == _name_before_cmd, bot._user_store.get(CID).name)
check("...and is let through", allowed is True, repr(allowed))


print()
print("=" * 66)
print("ADMIN GRANT / REVOKE")
print("=" * 66)

u = bot._user_store.get(CID); u.status = "approved"; u.is_admin = False
bot._user_store.put(u)
bot.make_admin(CID)
check("make_admin sets is_admin on the user record",
      bot._user_store.get(CID).is_admin is True)
check("make_admin mirrors is_admin onto the session",
      bot._get_session(CID).is_admin is True)

bot.revoke_admin(CID)
check("revoke_admin clears is_admin on the user record",
      bot._user_store.get(CID).is_admin is False)
check("revoke_admin mirrors the clear onto the session",
      bot._get_session(CID).is_admin is False)

# A no-op on a user who was never admin -- must not crash or flip a random field.
before = bot._user_store.get(CID)
bot.revoke_admin(CID)
after = bot._user_store.get(CID)
check("revoke_admin on a non-admin is a no-op",
      before.is_admin == after.is_admin == False)

check("revoke_admin on an unknown chat_id does not raise",
      (bot.revoke_admin(999999999) or True) is True)

print()
print("=" * 66)
print("STATUS GATES")
print("=" * 66)

for status, label in (("pending", "pending"), ("rejected", "rejected"),
                      ("banned", "banned")):
    u = bot._user_store.get(CID); u.status = status; bot._user_store.put(u)
    s = bot._get_session(CID); s.reg_state = ""; bot._store.put(s)
    allowed = bot._user_gate(CID, msg(CID, "let me in"))
    check(f"a {label} account cannot use the assistant", allowed is False,
          repr(allowed))

print()
print("=" * 66)
print("BAN / RE-APPROVE ACROSS A LEFTOVER SESSION STATE")
print("=" * 66)

# Bug 1: awaiting_login never checked user.status. Log out, then get banned
# (or rejected) while logged out -- the correct password still walked the
# account straight back to "logged in", silently undoing the ban.
CID_BAN_LOGIN = 999610
bot._user_store.put(T._User(chat_id=CID_BAN_LOGIN, name="Frank", status="approved",
                             password_hash=T._hash_password("realpass1", CID_BAN_LOGIN)))
s = bot._get_session(CID_BAN_LOGIN); s.reg_state = "awaiting_login"; bot._store.put(s)
u = bot._user_store.get(CID_BAN_LOGIN); u.status = "banned"; bot._user_store.put(u)
bot.sent.clear()
allowed = bot._user_gate(CID_BAN_LOGIN, msg(CID_BAN_LOGIN, "realpass1"))
check("the CORRECT password does not log a banned, logged-out user back in",
      allowed is False and bot._get_session(CID_BAN_LOGIN).reg_state == "awaiting_login",
      (allowed, bot._get_session(CID_BAN_LOGIN).reg_state))
check("they are told access was revoked, not welcomed back",
      any("evoked" in t or "тозван" in t for t in bot.sent), bot.sent)

u = bot._user_store.get(CID_BAN_LOGIN); u.status = "rejected"; bot._user_store.put(u)
bot.sent.clear()
bot._user_gate(CID_BAN_LOGIN, msg(CID_BAN_LOGIN, "realpass1"))
check("same guard covers a rejected-while-logged-out account",
      bot._get_session(CID_BAN_LOGIN).reg_state == "awaiting_login")

u = bot._user_store.get(CID_BAN_LOGIN); u.status = "approved"; bot._user_store.put(u)
bot.sent.clear()
allowed = bot._user_gate(CID_BAN_LOGIN, msg(CID_BAN_LOGIN, "realpass1"))
check("an approved, logged-out user can still log in normally (guard is not over-broad)",
      allowed is False and bot._get_session(CID_BAN_LOGIN).reg_state == "",
      (allowed, bot._get_session(CID_BAN_LOGIN).reg_state))

# Bug 2: a self-service edit mode (change_name / change_password / feedback)
# armed just before a ban survives the ban and is still armed once the same
# chat is later re-approved -- the next ordinary message from a person who has
# no idea any of this happened is silently swallowed as their new password.
CID_BAN_EDIT = 999611
bot._user_store.put(T._User(chat_id=CID_BAN_EDIT, name="Gina", status="approved",
                             password_hash=T._hash_password("origpass1", CID_BAN_EDIT)))
_orig_hash = bot._user_store.get(CID_BAN_EDIT).password_hash
s = bot._get_session(CID_BAN_EDIT); s.reg_state = "change_password"; bot._store.put(s)
u = bot._user_store.get(CID_BAN_EDIT); u.status = "banned"; bot._user_store.put(u)
check("reg_state survives the ban (that part is fine on its own)",
      bot._get_session(CID_BAN_EDIT).reg_state == "change_password")
bot.approve_user(CID_BAN_EDIT, allow_from_any_status=True)
check("re-approving clears the stale change_password arm",
      bot._get_session(CID_BAN_EDIT).reg_state == "",
      bot._get_session(CID_BAN_EDIT).reg_state)
bot.sent.clear()
allowed = bot._user_gate(CID_BAN_EDIT, msg(CID_BAN_EDIT, "hi there, how are you"))
check("an ordinary message after re-approval is NOT swallowed as a password",
      allowed is True, repr(allowed))
check("the account password was not silently overwritten",
      bot._user_store.get(CID_BAN_EDIT).password_hash == _orig_hash,
      "password was overwritten by an unrelated chat message!")

# Same leak, change_name flavour, and via the ordinary pending -> approved path
# (not just the any-status override) to prove the fix lives at the one choke
# point every approval goes through.
CID_BAN_NAME = 999612
bot._user_store.put(T._User(chat_id=CID_BAN_NAME, name="Hank", status="pending",
                             password_hash=T._hash_password("hankpass1", CID_BAN_NAME)))
s = bot._get_session(CID_BAN_NAME); s.reg_state = "change_name"; bot._store.put(s)
bot.approve_user(CID_BAN_NAME)
check("approving a pending user with a leftover change_name arm clears it too",
      bot._get_session(CID_BAN_NAME).reg_state == "",
      bot._get_session(CID_BAN_NAME).reg_state)

# Same leak, but for the two reg_states added later (wtw_city, set_badge) --
# they were never added to approve_user's stale-arm cleanup list, so an
# ordinary message after re-approval would still be swallowed as a city name
# or rejected as "not a valid emoji" instead of reaching the user normally.
CID_BAN_WTW = 999614
bot._user_store.put(T._User(chat_id=CID_BAN_WTW, name="Ida", status="approved"))
s = bot._get_session(CID_BAN_WTW); s.reg_state = "wtw_city"; bot._store.put(s)
u = bot._user_store.get(CID_BAN_WTW); u.status = "banned"; bot._user_store.put(u)
bot.approve_user(CID_BAN_WTW, allow_from_any_status=True)
check("re-approving clears a stale wtw_city arm",
      bot._get_session(CID_BAN_WTW).reg_state == "",
      bot._get_session(CID_BAN_WTW).reg_state)
allowed = bot._user_gate(CID_BAN_WTW, msg(CID_BAN_WTW, "hi there, how are you"))
check("an ordinary message after re-approval is not swallowed as a city",
      allowed is True, repr(allowed))

CID_BAN_BADGE = 999615
bot._user_store.put(T._User(chat_id=CID_BAN_BADGE, name="Jax", status="approved",
                             is_admin=True))
s = bot._get_session(CID_BAN_BADGE); s.reg_state = "set_badge"; bot._store.put(s)
u = bot._user_store.get(CID_BAN_BADGE); u.status = "banned"; bot._user_store.put(u)
bot.approve_user(CID_BAN_BADGE, allow_from_any_status=True)
check("re-approving clears a stale set_badge arm",
      bot._get_session(CID_BAN_BADGE).reg_state == "",
      bot._get_session(CID_BAN_BADGE).reg_state)
allowed = bot._user_gate(CID_BAN_BADGE, msg(CID_BAN_BADGE, "hi there, how are you"))
check("an ordinary message after re-approval is not rejected as an invalid badge",
      allowed is True, repr(allowed))

print()
print("=" * 66)
print("PROFILE DELETION AND FACTS LEAKAGE")
print("=" * 66)

# Deleting your profile (acct_newprofile_yes) is strictly MORE destructive
# than logging out -- it drops the account row entirely, not just the
# session. Logging out already wipes pinned facts and their on-disk shadow;
# a "delete and start over" must not leave the new profile silently
# inheriting facts pinned about the old one.
def cb_update(chat_id, data, msg_id=555):
    return {"callback_query": {
        "id": "cbq1", "data": data, "from": {"id": chat_id},
        "message": {"chat": {"id": chat_id}, "message_id": msg_id}}}

CID_DEL = 999613
bot._user_store.put(T._User(chat_id=CID_DEL, name="Ivy", status="approved",
                             password_hash=T._hash_password("ivypass1", CID_DEL)))
s = bot._get_session(CID_DEL)
s.set_tg_facts([{"fact": "lives in Berlin"}])
bot._store.put(s)
check("facts are pinned before deletion",
      len(bot._get_session(CID_DEL).get_tg_facts()) == 1)

bot._dispatch(cb_update(CID_DEL, "acct_newprofile_yes"))
check("the account row is gone", bot._user_store.get(CID_DEL) is None)
check("profile deletion wipes the pinned facts too, same as logout",
      bot._get_session(CID_DEL).get_tg_facts() == [],
      bot._get_session(CID_DEL).get_tg_facts())

print()
# ── the session outlived the account record ─────────────────────────────────
# A purge, a restored backup, a deleted row: the session still exists and still
# offers 👤 Аккаунт, but there is no user behind it. Both handlers used to
# return in silence, which is indistinguishable from a broken button.
_b = make_bot()
_cid = 909090
_sess = _b._get_session(_cid)
_b.sent.clear()
_b._send_account_menu(_cid, None, _sess)
check("the account menu says the account is gone",
      any("/start" in t for t in _b.sent), _b.sent)
_b.sent.clear()
_b._logout(_cid, None, _sess)
check("and so does the log-out path",
      any("/start" in t for t in _b.sent), _b.sent)

# A leftover state from an account that no longer exists answered with silence forever.
_bz = make_bot(); _cz = 999699
_sz = _bz._get_session(_cz); _sz.reg_state = "feedback"; _bz._store.put(_sz)
_bz._user_gate(_cz, msg(_cz, "hello?"))
check("a stale reg_state without an account restarts registration",
      _bz._get_session(_cz).reg_state == "awaiting_name" and _bz.sent, (_bz._get_session(_cz).reg_state, _bz.sent))

# The account card is a menu like any other: it needs a way back (live 10-03:
# «НЕТ КНОПКИ НАЗАД»)
_bk = make_bot(); _ck = 999700
_kbs = []
_bk._send_text = lambda cid, text, **kw: (_kbs.append(kw.get("keyboard")), 1)[1]
_bk._user_store.put(T._User(chat_id=_ck, name="Back", status="approved"))
_bk._send_account_menu(_ck, _bk._user_store.get(_ck), _bk._get_session(_ck))
check("the account menu has ⬅ Back",
      any(b.get("callback_data") == "nav:back"
          for kb in _kbs if kb for row in kb["inline_keyboard"] for b in row), _kbs)

print(f"{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)

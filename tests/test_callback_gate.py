"""The callback_query path used to skip the user gate ENTIRELY.

A 12-family real-chat button audit (2026-08-03) proved the same hole from six
angles: a banned/pending/logged-out account's OLD inline keyboard still worked —
image verbs enqueued real renders, `retry`/`lang:`/`facts_clear` executed with no
check at all, and worst of all `acct_setpwd` let a LOGGED-OUT chat set a brand new
password with zero credentials and silently re-authenticate itself, bypassing the
login brute-force throttle completely (that throttle only guards the MESSAGE path).

The fix is one gate at the top of `_dispatch`'s callback branch, mirroring the
`_user_gate` checks already used on the message path. Admin actions and `cancel:`
keep their own authorization and are exempt.

Run: venv/Scripts/python.exe tests/test_callback_gate.py
"""
import os, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T

_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_cbgate_")
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


def cb_update(chat_id, data, msg_id=555, presser_id=None):
    # facts_clear/_confirm/_cancel require the presser's `from.id` to match the
    # chat (a stranger's tap on an old keyboard must not wipe someone else's
    # facts) — default the presser to the chat itself, same as a private chat.
    return {"callback_query": {"id": "cbq1", "data": data,
            "from": {"id": presser_id if presser_id is not None else chat_id},
            "message": {"chat": {"id": chat_id}, "message_id": msg_id}}}


def set_user(bot, chat_id, status, is_admin=False, reg_state=""):
    u = T._User(chat_id=chat_id, name="U", tg_username="u", password_hash="x",
                status=status, is_admin=is_admin)
    bot._user_store.put(u)
    sess = bot._get_session(chat_id)
    sess.reg_state = reg_state
    bot._store.put(sess)
    return u, sess


print("=" * 70)
print("BANNED / PENDING / REJECTED — every non-admin callback is refused")
print("=" * 70)

for status in ("pending", "rejected", "banned"):
    bot, sent = make_bot()
    CID = 9000 + hash(status) % 1000
    set_user(bot, CID, status)
    sess = bot._get_session(CID)
    sess.lang = "ru"; sess.tg_facts = ["a fact"]; bot._store.put(sess)

    sent.clear(); bot._dispatch(cb_update(CID, "lang:en"))
    check(f"status={status}: lang: does not change the language",
          bot._get_session(CID).lang == "ru", bot._get_session(CID).lang)

    sent.clear(); bot._dispatch(cb_update(CID, "facts_clear"))
    check(f"status={status}: facts_clear does not clear the facts",
          bot._get_session(CID).tg_facts == ["a fact"], bot._get_session(CID).tg_facts)

    sent.clear(); bot._dispatch(cb_update(CID, "retry"))
    check(f"status={status}: retry does not enqueue anything",
          not any(m in ("editMessageText",) for m, _ in sent) and
          not bot._backend.depth(), sent)

    sent.clear(); bot._dispatch(cb_update(CID, "upscale:deadbeef"))
    check(f"status={status}: upscale: does not enqueue a render",
          not bot._backend.depth(), sent)

    # every one of these must still have answered the user, not gone silent
    sent.clear(); bot._dispatch(cb_update(CID, "depth:deep"))
    check(f"status={status}: a refusal reply was sent, not silence",
          any(m == "sendMessage" for m, _ in sent), sent)

print()
print("=" * 70)
print("LOGGED-OUT (awaiting_login) — acct_setpwd cannot re-authenticate the chat")
print("=" * 70)

bot, sent = make_bot()
CID = 8001
user, sess = set_user(bot, CID, "approved", reg_state="awaiting_login")
old_hash = user.password_hash
sent.clear()
bot._dispatch(cb_update(CID, "acct_setpwd"))
fresh = bot._user_store.get(CID)
fresh_sess = bot._get_session(CID)
check("password hash unchanged by a stale acct_setpwd while logged out",
      fresh.password_hash == old_hash, fresh.password_hash)
check("reg_state still awaiting_login (not silently re-authenticated)",
      fresh_sess.reg_state == "awaiting_login", fresh_sess.reg_state)
check("the user was told to log in, not left silent",
      any(m == "sendMessage" for m, _ in sent), sent)

print()
print("=" * 70)
print("APPROVED users still work normally — the gate must not over-refuse")
print("=" * 70)

bot, sent = make_bot()
CID = 8002
set_user(bot, CID, "approved")
sent.clear()
bot._dispatch(cb_update(CID, "lang:en"))
sess = bot._get_session(CID)
check("an approved user's lang: callback still applies", sess.lang == "en", sess.lang)

bot, sent = make_bot()
CID = 8003
set_user(bot, CID, "approved")
sent.clear()
bot._dispatch(cb_update(CID, "facts_clear"))
check("an approved user's facts_clear still runs (some reply sent)",
      any(m == "sendMessage" or m == "editMessageText" for m, _ in sent), sent)

print()
print("=" * 70)
print("EXEMPT CALLBACKS — admin actions and cancel: are not blocked by this gate")
print("=" * 70)

bot, sent = make_bot()
CID = 8004
set_user(bot, CID, "banned")
sent.clear()
bot._dispatch(cb_update(CID, "cancel:doesnotexist"))
# cancel: is self-authorizing (harmless no-op on a nonexistent id) and must not be
# swallowed by the pending/banned refusal text.
check("cancel: is not rewritten into a 'you are banned' message",
      not any("revoked" in str(p) for _, p in sent), sent)

print()
print(f"{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)

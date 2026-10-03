"""Bot-wide broadcasts: who gets them, and in which language.

Two paired notices tell users the bot went down and came back. They were not
actually a pair:

  · `_broadcast_startup` honoured `"startup" in user.subscriptions`;
  · `_broadcast_shutdown` sent to EVERY approved user, unconditionally.

So /unsubscribe — which answers "🔕 Unsubscribed from all notifications" — silenced
exactly half of what it promised, and the user still got pinged on every restart.

Both were also hardcoded English in an otherwise bilingual bot, so a Russian user
got two English messages per restart cycle.

Run: venv/Scripts/python.exe tests/test_tg_broadcast.py
"""
import sys, types, tempfile, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T

# Before any bot exists — otherwise fixture users land in the LIVE store.
_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_bcast_")
T.redirect_data_dir(_DATA_DIR)

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")


SUB_EN, SUB_RU, UNSUB = 999601, 999602, 999603


def make_bot():
    bot = T.TelegramBot("123:TEST", lambda: None, lambda: None,
                        lambda: {}, silent_mode=True)
    bot.sent = []                     # (chat_id, text)
    bot._send_text = lambda cid, text, **kw: (bot.sent.append((cid, text)), 1)[1]
    bot._activity.log = lambda *a, **k: None
    bot._backend = types.SimpleNamespace(close=lambda: None, depth=lambda: 0,
                                         name=lambda: "stub")
    return bot


def seed(bot):
    for cid, subs, lang in ((SUB_EN, ["startup"], "en"),
                            (SUB_RU, ["startup"], "ru"),
                            (UNSUB,  [],          "en")):
        u = T._User(chat_id=cid, name=f"u{cid}", status="approved",
                    subscriptions=list(subs))
        bot._user_store.put(u)
        s = bot._get_session(cid)
        s.lang = lang
        bot._store.put(s)


bot = make_bot()
seed(bot)

print("=" * 70)
print("SUBSCRIPTION GATES BOTH HALVES OF THE PAIR")
print("=" * 70)

bot.sent.clear(); bot._broadcast_shutdown()
got = {cid for cid, _ in bot.sent}
check("subscribed users get the offline notice", {SUB_EN, SUB_RU} <= got, str(got))
check("an unsubscribed user gets NO offline notice", UNSUB not in got, str(got))

bot.sent.clear(); bot._broadcast_startup()
got = {cid for cid, _ in bot.sent}
check("subscribed users get the online notice", {SUB_EN, SUB_RU} <= got, str(got))
check("an unsubscribed user gets NO online notice", UNSUB not in got, str(got))

# The two must agree exactly — that is what makes them a pair.
bot.sent.clear(); bot._broadcast_shutdown()
down = {cid for cid, _ in bot.sent}
bot.sent.clear(); bot._broadcast_startup()
up = {cid for cid, _ in bot.sent}
check("the same audience receives both notices", down == up, f"{down} vs {up}")


print()
print("=" * 70)
print("EACH USER IS ADDRESSED IN THEIR OWN LANGUAGE")
print("=" * 70)

for name, fn in (("offline", bot._broadcast_shutdown), ("online", bot._broadcast_startup)):
    bot.sent.clear(); fn()
    by_chat = dict(bot.sent)
    key = "going_offline" if name == "offline" else "back_online"
    check(f"the {name} notice is English for an English user",
          by_chat.get(SUB_EN) == T._t(key, "en"), repr(by_chat.get(SUB_EN)))
    check(f"the {name} notice is Russian for a Russian user",
          by_chat.get(SUB_RU) == T._t(key, "ru"), repr(by_chat.get(SUB_RU)))
    check(f"the two languages actually differ ({name})",
          by_chat.get(SUB_EN) != by_chat.get(SUB_RU))


print()
print("=" * 70)
print("ROBUSTNESS")
print("=" * 70)

# A row whose subscriptions column is JSON `null` must not crash the broadcasts.
# Both loops iterate every approved user, so ONE bad row used to abort the notice
# for everyone after it — and _broadcast_startup runs in its own thread, so the
# only trace was a dead thread.
u = T._User(chat_id=999604, name="legacy", status="approved", subscriptions=None)
check("a None subscriptions list is coerced at construction", u.subscriptions == [])
bot._user_store.put(u)

# Write the NULL straight into the DB, bypassing the dataclass, to prove the read
# path survives what an old database actually contains.
with bot._user_store._conn() as conn:
    conn.execute("UPDATE users SET subscriptions = 'null' WHERE chat_id = ?", (999604,))
    conn.commit()
back = bot._user_store.get(999604)
check("a JSON-null column reads back as an empty list", back.subscriptions == [],
      repr(back.subscriptions))

for name, fn in (("shutdown", bot._broadcast_shutdown),
                 ("startup", bot._broadcast_startup)):
    bot.sent.clear()
    try:
        fn(); ok, err = True, ""
    except Exception as exc:
        ok, err = False, repr(exc)
    check(f"a null subscriptions row does not break the {name} broadcast", ok, err)
    check(f"the other users still get the {name} notice",
          {SUB_EN, SUB_RU} <= {cid for cid, _ in bot.sent},
          str({cid for cid, _ in bot.sent}))
    check(f"and the null-row user is treated as unsubscribed ({name})",
          999604 not in {cid for cid, _ in bot.sent})

# A send failure for one user must not silence the rest.
bad = 999605
bot._user_store.put(T._User(chat_id=bad, name="bad", status="approved",
                            subscriptions=["startup"]))
real_send = bot._send_text
def flaky(cid, text, **kw):
    if cid == bad:
        raise RuntimeError("403 blocked by user")
    return real_send(cid, text, **kw)
bot._send_text = flaky
bot.sent.clear()
bot._broadcast_shutdown()
check("one blocked user does not stop the broadcast",
      {SUB_EN, SUB_RU} <= {cid for cid, _ in bot.sent},
      str({cid for cid, _ in bot.sent}))
bot._send_text = real_send

# Pending / rejected users are never in the audience.
bot._user_store.put(T._User(chat_id=999606, name="pend", status="pending",
                            subscriptions=["startup"]))
bot.sent.clear(); bot._broadcast_startup()
check("a pending user is not broadcast to",
      999606 not in {cid for cid, _ in bot.sent})

# silent_mode must suppress the broadcasts entirely (stop() honours the flag).
quiet = make_bot()
quiet._user_store = bot._user_store
quiet.sent.clear()
quiet._running = True
quiet.stop()
check("silent mode sends no shutdown notice", quiet.sent == [], str(quiet.sent))

print()
print(f"{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)

"""Regression tests for the 2026-08-03 audit's 🧠 Facts findings.

Offline only (no LM Studio, no ComfyUI, no GPU). See
scratchpad/audit_11_facts.md for the original proof of each bug.

Covers:
F1. 🗑 Forget all had no presser check and no confirmation before an
    irreversible wipe (a stranger's `from.id` on an old inline keyboard could
    clear someone else's facts with one tap).
F2. remember_fact's on-disk shadow copy (tg_memory/chat_<id>/facts.json) was
    never deleted by Forget-all, Clear Chat, or logout.
F3. 🧠 My facts silently truncated at 4096 chars with no "showing N of M".
F4. 🗑 Clear Chat destroyed pinned facts with no warning in either language.
F5. The 40-fact cap silently dropped the oldest fact while the confirmation
    still said "saved permanently".

Run: venv/Scripts/python.exe tests/test_facts_fixes.py
"""
import os, sys, tempfile, threading, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T
import tools as TOOLS
import config

_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_factsfixes_")
T.redirect_data_dir(_DATA_DIR)

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")


def section(title):
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


CID = 998801


def make_bot():
    bot = T.TelegramBot("123:TEST", lambda: None, lambda: None, lambda: {}, silent_mode=True)
    bot.sent = []
    bot._send_text = lambda cid, text, **kw: bot.sent.append((cid, text, kw)) or 1
    bot._send_get_id = lambda cid, text, **kw: (bot.sent.append((cid, text, kw)), 1)[1]
    bot._api_post = lambda *a, **k: {"ok": True, "result": {"message_id": len(bot.sent) + 1}}
    bot._api_get = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    bot._user_store.put(T._User(chat_id=CID, name="Owner", status="approved"))
    return bot


def cb(chat_id, data, from_id):
    return {"callback_query": {"id": "1", "data": data,
            "message": {"chat": {"id": chat_id}, "message_id": 1},
            "from": {"id": from_id}}}


# ══════════════════════════════════════════════════════════════════════════
section("F1. facts_clear checks the PRESSER, and requires a confirming tap")

bot = make_bot()
sess = bot._get_session(CID)
sess.set_tg_facts([{"ts": time.time(), "text": "lives in Berlin"}])
bot._store.put(sess)

# A stranger (not the chat's own account) presses the button.
bot._dispatch(cb(CID, "facts_clear", 777777))
check("a stranger's press did not wipe the facts",
      len(bot._get_session(CID).get_tg_facts()) == 1,
      bot._get_session(CID).get_tg_facts())
check("a stranger's press got no destructive confirmation card either",
      not any("Забыто" in str(t) or "Forgot" in str(t) for _, t, _ in bot.sent),
      bot.sent)

# The real owner presses it: this must ask for confirmation, not wipe instantly.
bot.sent.clear()
bot._dispatch(cb(CID, "facts_clear", CID))
check("the owner's first press did NOT wipe the facts yet (needs confirmation)",
      len(bot._get_session(CID).get_tg_facts()) == 1,
      bot._get_session(CID).get_tg_facts())
check("the owner's first press showed a confirm/cancel keyboard",
      any(kw.get("keyboard") for _, _, kw in bot.sent), bot.sent)

# A stranger must not be able to confirm someone else's pending wipe either.
bot._dispatch(cb(CID, "facts_clear_confirm", 777777))
check("a stranger cannot confirm the wipe either",
      len(bot._get_session(CID).get_tg_facts()) == 1)

# The owner confirms: NOW it wipes.
bot.sent.clear()
bot._dispatch(cb(CID, "facts_clear_confirm", CID))
check("the owner's confirming tap actually wipes the facts",
      bot._get_session(CID).get_tg_facts() == [])
check("a 'forgot N facts' confirmation was sent",
      any("1" in str(t) for _, t, _ in bot.sent), bot.sent)

# The cancel path must leave the facts untouched.
bot2 = make_bot()
sess2 = bot2._get_session(CID)
sess2.set_tg_facts([{"ts": time.time(), "text": "hates olives"}])
bot2._store.put(sess2)
bot2._dispatch(cb(CID, "facts_clear", CID))
bot2._dispatch(cb(CID, "facts_clear_cancel", CID))
check("tapping Cancel on the confirmation keeps the facts",
      len(bot2._get_session(CID).get_tg_facts()) == 1)


# ══════════════════════════════════════════════════════════════════════════
section("F2. the on-disk facts.json shadow copy is actually deleted")

bot = make_bot()
sess = bot._get_session(CID)
sess.set_tg_facts([{"ts": time.time(), "text": "owns a cat"}])
bot._store.put(sess)

facts_path = T._MEMORY_DIR / f"chat_{CID}" / "facts.json"
facts_path.parent.mkdir(parents=True, exist_ok=True)
facts_path.write_text(json.dumps([{"text": "owns a cat"}]), encoding="utf-8")
check("the shadow file exists before any clear", facts_path.exists())

bot._dispatch(cb(CID, "facts_clear", CID))
bot._dispatch(cb(CID, "facts_clear_confirm", CID))
check("Forget-all deletes the on-disk shadow copy too", not facts_path.exists())

# Clear Chat (/clear) must also delete it.
facts_path.parent.mkdir(parents=True, exist_ok=True)
facts_path.write_text(json.dumps([{"text": "owns a cat"}]), encoding="utf-8")
bot._handle_command(CID, "/clear")
check("🗑 Clear Chat (/clear) deletes the on-disk shadow copy too",
      not facts_path.exists())

# Logout must also delete it.
facts_path.parent.mkdir(parents=True, exist_ok=True)
facts_path.write_text(json.dumps([{"text": "owns a cat"}]), encoding="utf-8")
user = bot._user_store.get(CID)
sess3 = bot._get_session(CID)
bot._logout(CID, user, sess3)
check("logout deletes the on-disk shadow copy too", not facts_path.exists())


# ══════════════════════════════════════════════════════════════════════════
section("F3. 🧠 My facts no longer silently truncates at 4096 chars")

bot = make_bot()
sess = bot._get_session(CID)
many = [{"ts": time.time(), "text": f"fact number {i} " + ("x" * 250)}
        for i in range(40)]
sess.set_tg_facts(many)
bot._store.put(sess)
bot.sent.clear()
bot._send_facts(CID, sess)

total_chars = sum(len(t) for _, t, _ in bot.sent)
check("the facts were split across more than one message instead of one "
      "blind 4096-char slice",
      len(bot.sent) > 1, len(bot.sent))
check("all 40 facts are present SOMEWHERE across the messages (nothing silently lost)",
      all(f"fact number {i}" in "".join(t for _, t, _ in bot.sent) for i in range(40)))
check("no single chunk exceeds Telegram's hard limit",
      all(len(t) <= 4096 for _, t, _ in bot.sent), [len(t) for _, t, _ in bot.sent])
check("the Forget-all button is still attached (to the last chunk)",
      any(kw.get("keyboard") for _, _, kw in bot.sent), bot.sent)


# ══════════════════════════════════════════════════════════════════════════
section("F4. 🗑 Clear Chat's confirmation mentions facts being wiped, in both languages")

en_text = T._t("cleared", "en")
ru_text = T._t("cleared", "ru")
check("the English confirmation mentions facts",
      "fact" in en_text.lower(), en_text)
check("the Russian confirmation mentions facts",
      "факт" in ru_text.lower(), ru_text)

bot = make_bot()
sess = bot._get_session(CID)
sess.set_tg_facts([{"ts": time.time(), "text": "some fact"}])
bot._store.put(sess)
bot.sent.clear()
bot._handle_command(CID, "/clear")
check("clearing chat actually still wipes the facts (unchanged behaviour)",
      bot._get_session(CID).get_tg_facts() == [])
check("...and the sent message says so",
      any("fact" in str(t).lower() or "факт" in str(t).lower()
          for _, t, _ in bot.sent), bot.sent)


# ══════════════════════════════════════════════════════════════════════════
section("F5. the 40-fact cap no longer claims 'permanently' when it evicts")

class FakeCtx:
    def __init__(self):
        self.pinned_facts = []
        self.memory_lock = threading.Lock()
        self.active_memory_dir = None
    def pin_fact(self, text):
        text = (text or "").strip()
        if not text:
            return False
        if any(f["text"].casefold() == text.casefold() for f in self.pinned_facts):
            return False
        self.pinned_facts.append({"ts": time.time(), "text": text})
        del self.pinned_facts[:-config.PINNED_FACTS_LIMIT]
        return True
    def save_memory(self, d): pass

ctx = FakeCtx()
# Fill to exactly the cap.
for i in range(config.PINNED_FACTS_LIMIT):
    TOOLS._handle_remember_fact(ctx, {}, {"fact": f"fact {i}"})
check(f"the store holds exactly the {config.PINNED_FACTS_LIMIT}-fact cap",
      len(ctx.pinned_facts) == config.PINNED_FACTS_LIMIT, len(ctx.pinned_facts))

# One more push evicts the oldest ("fact 0").
result = TOOLS._handle_remember_fact(ctx, {}, {"fact": "one fact too many"})
check("the store size is still capped after the eviction",
      len(ctx.pinned_facts) == config.PINNED_FACTS_LIMIT, len(ctx.pinned_facts))
check("the oldest fact ('fact 0') was actually evicted",
      not any(f["text"] == "fact 0" for f in ctx.pinned_facts))
check("the tool no longer claims 'permanently' when it just evicted a fact",
      "permanently" not in result.lower(), result)
check("the tool result says an older fact was displaced",
      "displac" in result.lower() or "dropped" in result.lower() or "oldest" in result.lower(),
      result)

# A save comfortably under the cap must still say "permanently" (no regression).
ctx2 = FakeCtx()
r2 = TOOLS._handle_remember_fact(ctx2, {}, {"fact": "just one fact"})
check("an ordinary save well under the cap still says 'permanently'",
      "permanently" in r2.lower(), r2)


print(f"\n{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)

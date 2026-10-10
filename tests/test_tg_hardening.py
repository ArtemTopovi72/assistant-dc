"""Scenario tests for the multi-user hardening pass on tg_bot.

Covers: round-robin fairness, per-user caps, daily quotas, scrypt passwords +
login throttling, per-request cancel, localization, the document library hook,
crash recovery, group-chat rejection and pinned-fact isolation.

Drives the REAL TelegramBot with only the network and the agent stubbed.
"""
import hashlib
import os
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import sys
import threading
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tg_bot as T

# MUST come before any TelegramBot is constructed: the bot's data paths are
# module-level, so an unredirected run registers its fixture chat ids in the
# LIVE tg_users.db (this happened — 999101/999102 ended up approved there).
import tempfile
_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_hardening_")
T.redirect_data_dir(_DATA_DIR)

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1
        print(f"PASS  {name}")
    else:
        BAD += 1
        print(f"FAIL  {name}   {extra}")


def section(title):
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


def make_bot():
    bot = T.TelegramBot("123:TEST", lambda: None, lambda: None,
                        lambda: {}, silent_mode=True)
    bot.sent = []
    bot.docs = []
    bot._send_text = lambda cid, text, **kw: bot.sent.append((cid, text, kw)) or 1
    bot._send_get_id = lambda cid, text, **kw: (bot.sent.append((cid, text, kw)), 1)[1]
    bot._send_document = lambda cid, path, caption="": bot.docs.append((cid, path)) or True
    bot._api_post = lambda *a, **k: {}
    bot._api_get = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    bot._dl_bytes = lambda fid: b"\xff\xd8\xff" + b"0" * 100
    return bot


def texts(bot):
    return " || ".join(t for _, t, _ in bot.sent)


CID = 999101
CID2 = 999102


def fresh_session(bot, cid=CID):
    s = bot._get_session(cid)
    s.clear_context()
    s.reg_state = ""
    s.lang = "en"
    s.use_docs = False
    s.set_tg_facts([])
    bot._store.put(s)
    bot.sent.clear()
    return s


# ══════════════════════════════════════════════════════════════════════════════
section("FAIRNESS — round-robin scheduling")

b = T.InMemoryBackend()
mk = lambda c, n, txt="hi": T._Task(task_id=f"{c}-{n}", chat_id=c, user_text=txt)
for i in range(5):
    b.push(mk(1, i))
b.push(mk(2, 0))
order = [t.task_id for t in b.service_order()]
check("second user is served 2nd, not 6th", order[1] == "2-0", order)

popped = []
while True:
    t = b.pop(timeout=0.05)
    if t is None:
        break
    popped.append(t.task_id)
check("simulated order equals real pop order", popped == order, f"{popped} vs {order}")

b = T.InMemoryBackend()
for i in range(3):
    b.push(mk(1, i))
b.push(mk(2, 0))
check("tasks_ahead counts only what precedes you",
      [t.task_id for t in b.tasks_ahead("2-0")] == ["1-0"],
      [t.task_id for t in b.tasks_ahead("2-0")])
check("chat_depth is per chat", b.chat_depth(1) == 3 and b.chat_depth(2) == 1)
check("drop_chat removes only that chat", b.drop_chat(1) == 3 and b.depth() == 1)
check("drop_task removes a single queued task", b.drop_task("2-0") and b.depth() == 0)
check("drop_task on an unknown id is False", b.drop_task("ghost") is False)

# empty backend must not spin or crash
check("pop on empty returns None", b.pop(timeout=0.05) is None)

# ETA reflects task kind, not just count
eta_tasks = T._fmt_eta([mk(1, 0), mk(1, 1)], "en")
eta_research = T._fmt_eta([mk(1, 0, "do a deep research on: x")], "en")
check("deep research ETA >> ordinary task ETA",
      "min" in eta_research and eta_research != eta_tasks,
      f"{eta_tasks} / {eta_research}")


# ══════════════════════════════════════════════════════════════════════════════
section("PER-USER CAP + DAILY QUOTA")

bot = make_bot()
bot._backend = T.InMemoryBackend()
fresh_session(bot)
# make sure this test user is not an admin (admins are exempt)
u = T._User(chat_id=CID, name="Tester", status="approved", is_admin=False)
bot._user_store.put(u)

for i in range(6):
    bot._resolve_and_push(CID, [{"type": "text", "text": f"question {i}"}])
cap = T._cfg_int("TG_MAX_QUEUED_PER_USER", 3)
check(f"queued tasks capped at {cap}", bot._backend.chat_depth(CID) == cap,
      bot._backend.chat_depth(CID))
check("user is told why", "already waiting" in texts(bot).lower(), texts(bot)[-200:])

# quota: force the limit down to 1 and confirm the 2nd research is refused
bot._backend = T.InMemoryBackend()
fresh_session(bot)
import config as _cfg
_saved = _cfg.TG_QUOTA_DEEP_RESEARCH
_cfg.TG_QUOTA_DEEP_RESEARCH = 1
try:
    # clear today's counters for a deterministic run
    with bot._user_store._conn() as conn:
        conn.execute("DELETE FROM usage WHERE chat_id=?", (CID,))
    bot._resolve_and_push(CID, [{"type": "text", "text": "do a deep research on: a"}])
    first_ok = bot._backend.depth() == 1
    bot._resolve_and_push(CID, [{"type": "text", "text": "do a deep research on: b"}])
    check("first research accepted", first_ok)
    check("second research refused by quota", bot._backend.depth() == 1,
          bot._backend.depth())
    check("refusal explains the limit", "limit" in texts(bot).lower(), texts(bot)[-200:])
    # an ordinary question is still allowed
    bot._resolve_and_push(CID, [{"type": "text", "text": "hello"}])
    check("other task kinds unaffected by the research quota",
          bot._backend.depth() == 2, bot._backend.depth())
finally:
    _cfg.TG_QUOTA_DEEP_RESEARCH = _saved

# admins bypass both
bot._backend = T.InMemoryBackend()
fresh_session(bot)
bot._user_store.put(T._User(chat_id=CID, name="Boss", status="approved", is_admin=True))
for i in range(6):
    bot._resolve_and_push(CID, [{"type": "text", "text": f"admin q{i}"}])
check("admin is exempt from the queue cap", bot._backend.chat_depth(CID) == 6,
      bot._backend.chat_depth(CID))
bot._user_store.put(T._User(chat_id=CID, name="Tester", status="approved", is_admin=False))


# ══════════════════════════════════════════════════════════════════════════════
section("PASSWORDS — scrypt, upgrade, throttle")

h = T._hash_password("correct horse", CID)
check("stored hash is versioned scrypt with its cost", h.startswith("v3$14$"), h[:16])
check("correct password verifies", T._verify_password("correct horse", CID, h)[0])
check("wrong password rejected", not T._verify_password("wrong", CID, h)[0])
check("hash is salted (two hashes differ)",
      T._hash_password("x", 1) != T._hash_password("x", 1))

legacy = hashlib.sha256(("old pass" + str(CID)).encode()).hexdigest()
ok, upgraded = T._verify_password("old pass", CID, legacy)
check("legacy sha256 password still logs in", ok)
check("legacy login returns an upgraded hash", upgraded.startswith("v3$"))
check("upgraded hash verifies the same password",
      T._verify_password("old pass", CID, upgraded)[0])
check("legacy wrong password still rejected",
      not T._verify_password("nope", CID, legacy)[0])
check("garbage stored hash fails closed",
      not T._verify_password("x", CID, "not-a-hash")[0])
check("empty stored hash fails closed", not T._verify_password("x", CID, "")[0])

# Raising TG_SCRYPT_N_LOG2 used to lock every user out: v2 hashes carry no
# cost and were checked at the NEW one.
import hashlib as _hl, config as _cfg
_salt = b"0123456789abcdef"
_v2 = "v2$" + _salt.hex() + "$" + _hl.scrypt(b"pw", salt=_salt, n=2 ** 14, r=8, p=1,
                                             dklen=32, maxmem=256 * 1024 * 1024).hex()
_saved_cost = getattr(_cfg, "TG_SCRYPT_N_LOG2", None)
_cfg.TG_SCRYPT_N_LOG2 = 15
try:
    ok, up = T._verify_password("pw", CID, _v2)
    check("a v2 hash still verifies after the cost is raised", ok)
    check("...and is re-hashed at the new cost", up.startswith("v3$15$"), up[:10])
    check("a wrong password on a v2 hash still fails", not T._verify_password("no", CID, _v2)[0])
    h15 = T._hash_password("pw", CID)
    _cfg.TG_SCRYPT_N_LOG2 = 14
    ok, up = T._verify_password("pw", CID, h15)
    check("a v3 hash verifies at its own recorded cost", ok and up.startswith("v3$14$"), up[:10])
    check("a v3 hash at the current cost needs no upgrade",
          T._verify_password("pw", CID, up) == (True, ""))
    check("a v3 hash with an absurd cost fails closed",
          not T._verify_password("pw", CID, "v3$40$00$00")[0])
finally:
    if _saved_cost is None:
        del _cfg.TG_SCRYPT_N_LOG2
    else:
        _cfg.TG_SCRYPT_N_LOG2 = _saved_cost

# throttle: repeated wrong passwords lock the chat, correct one is then refused
bot = make_bot()
user = T._User(chat_id=CID, name="Tester", status="approved",
               password_hash=T._hash_password("rightpass", CID))
bot._user_store.put(user)
s = fresh_session(bot)
s.reg_state = "awaiting_login"
bot._store.put(s)
max_fails = T._cfg_int("TG_LOGIN_MAX_FAILS", 5)
for i in range(max_fails):
    bot._user_gate(CID, {"text": "guess", "from": {}})
check("lockout message after repeated failures",
      "too many" in texts(bot).lower() or "много" in texts(bot).lower(),
      texts(bot)[-200:])
bot.sent.clear()
allowed = bot._user_gate(CID, {"text": "rightpass", "from": {}})
check("correct password refused while locked out",
      bot._get_session(CID).reg_state == "awaiting_login" and not allowed)
# expire the lock and try again
bot._login_fails[CID] = [0, 0.0]
bot.sent.clear()
bot._user_gate(CID, {"text": "rightpass", "from": {}})
check("login works once the lock expires",
      bot._get_session(CID).reg_state == "" and "Welcome back" in texts(bot),
      texts(bot)[:120])

# legacy hash is rewritten in the DB on a successful login
bot._user_store.put(T._User(chat_id=CID, name="Tester", status="approved",
                            password_hash=hashlib.sha256(
                                ("legacy1" + str(CID)).encode()).hexdigest()))
s = bot._get_session(CID)
s.reg_state = "awaiting_login"
bot._store.put(s)
bot._login_fails.pop(CID, None)
bot._user_gate(CID, {"text": "legacy1", "from": {}})
check("DB hash upgraded in place after legacy login",
      bot._user_store.get(CID).password_hash.startswith("v3$"),
      bot._user_store.get(CID).password_hash[:12])


# ══════════════════════════════════════════════════════════════════════════════
section("CANCEL — per-request interrupt")

bot = make_bot()
bot._backend = T.InMemoryBackend()
fresh_session(bot)
queued = T._Task(task_id="q1", chat_id=CID, user_text="slow thing")
bot._backend.push(queued)
check("queued task is cancelled by dropping it",
      bot._cancel_task(CID, "q1") and bot._backend.depth() == 0)

# running task: cancel_event must be set for THIS chat only
events = {}


class FakeCtx:
    def __init__(self):
        self.cancel_event = threading.Event()


ctx = FakeCtx()
bot._get_ctx = lambda: ctx
# Each in-flight task owns its cancel Event (see _scoped_ctx). A single shared
# event was safe only while exactly one task ran at a time; with several workers
# it let one chat's Stop cancel whatever else happened to be in flight.
running = T._Task(task_id="r1", chat_id=CID, user_text="running")
bot._running_task[CID] = [running]
ev1 = threading.Event()
bot._task_cancels["r1"] = ev1
check("cancelling the running task signals ITS OWN cancel event",
      bot._cancel_task(CID, "r1") and ev1.is_set())

ev1.clear()
check("cancelling someone else's task does not signal ours",
      not bot._cancel_task(CID2, "r1") and not ev1.is_set())

# Two chats running AT THE SAME TIME — the case a single shared event broke.
other = T._Task(task_id="r2", chat_id=CID2, user_text="other chat's work")
ev2 = threading.Event()
bot._running_task[CID2] = [other]
bot._task_cancels["r2"] = ev2

ev1.clear(); ev2.clear()
bot._request_stop(CID2)
check("Stop in one chat leaves another chat's task running", not ev1.is_set())
check("Stop cancels the OTHER chat's own task", ev2.is_set())

ev1.clear(); ev2.clear()
bot._request_stop(CID)
check("Stop cancels your own running task", ev1.is_set())
check("and does not touch the concurrent task", not ev2.is_set())

bot._running_task.pop(CID2, None)
bot._task_cancels.pop("r2", None)

# a cancelled task is dropped rather than executed
bot._cancelled.add("z9")
check("_is_cancelled reports queued cancellations", bot._is_cancelled("z9"))

# in-flight bookkeeping registers during the task and is cleaned up after,
# even when the task raises
bot._running_task.clear()
bot._task_started.clear()
bot._get_graph = lambda: object()
seen = {}


def _spy(task, *a, **kw):
    seen["running"] = dict(bot._running_task)
    seen["started"] = dict(bot._task_started)
    raise RuntimeError("boom")


bot._run_task_inner = _spy
try:
    bot._execute_task(T._Task(task_id="t9", chat_id=CID, user_text="x"))
except RuntimeError:
    pass
check("task is registered as in-flight while it runs",
      CID in seen.get("running", {}) and "t9" in seen.get("started", {}), seen)
check("in-flight bookkeeping is cleaned up even when the task raises",
      CID not in bot._running_task and "t9" not in bot._task_started,
      (bot._running_task, bot._task_started))

# a task that starts already-cancelled must not leave the flag behind
bot._cancelled.add("t10")
bot._run_task_inner = lambda *a, **kw: None
bot._execute_task(T._Task(task_id="t10", chat_id=CID, user_text="x"))
check("cancel flag is cleared after the task settles", not bot._is_cancelled("t10"))


# ══════════════════════════════════════════════════════════════════════════════
section("LOCALIZATION")

check("every button has all languages",
      all(set(f) >= set(T._LANGS) for f in T._BTN.values()),
      [k for k, f in T._BTN.items() if set(f) < set(T._LANGS)])
check("every message has all languages",
      all(set(f) >= set(T._LANGS) for f in T._MSG.values()),
      [k for k, f in T._MSG.items() if set(f) < set(T._LANGS)])
dupes = [lbl for lbl in T._LABEL2KEY
         if sum(1 for k, f in T._BTN.items() if lbl in f.values()) > 1]
check("no button label is ambiguous across languages", not dupes, dupes)
check("Russian label maps to the same key as English",
      T._LABEL2KEY["⛔ Стоп"] == T._LABEL2KEY["⛔ Stop"] == "stop")
check("language code normalisation", T._norm_lang("ru-RU") == "ru"
      and T._norm_lang("en-GB") == "en"
      and T._norm_lang("de") == T._DEFAULT_LANG
      and T._norm_lang("") == T._DEFAULT_LANG,
      f"default={T._DEFAULT_LANG} de={T._norm_lang('de')}")
check("format args are applied", "5" in T._t("stop_dropped", "en", n=5))
check("unknown key degrades to the key itself", T._t("nope", "en") == "nope")
check("missing translation falls back to the house default",
      T._t("cleared", "zz") == T._t("cleared", T._DEFAULT_LANG))
check("a half-translated form still renders, never the raw key",
      T._pick_form({"en": "only english"}, "ru") == "only english"
      and T._pick_form({"ru": "только русский"}, "en") == "только русский")

# a Russian keyboard label still triggers the action, and switching language
# does not strand a keyboard rendered in the old one
bot = make_bot()
bot._backend = T.InMemoryBackend()
s = fresh_session(bot)
s.lang = "ru"
bot._store.put(s)
bot._resolve_and_push(CID, [{"type": "text", "text": "🗑 Очистить чат"}])
check("Russian Clear button clears instead of reaching the agent",
      bot._backend.depth() == 0 and "очищ" in texts(bot).lower(), texts(bot))
bot.sent.clear()
bot._resolve_and_push(CID, [{"type": "text", "text": "🗑 Clear Chat"}])
check("stale English keyboard still works after switching to Russian",
      bot._backend.depth() == 0 and "очищ" in texts(bot).lower(), texts(bot))
bot.sent.clear()
bot._resolve_and_push(CID, [{"type": "text", "text": "🔬 Глубокое исследование"}])
check("Russian Deep Research sets the English prompt prefix",
      bot._get_session(CID).pending_prefix == "do a deep research on: ",
      repr(bot._get_session(CID).pending_prefix))


# ══════════════════════════════════════════════════════════════════════════════
section("GROUP CHATS")

bot = make_bot()
bot._backend = T.InMemoryBackend()
bot.sent.clear()
bot._dispatch({"message": {"chat": {"id": -100123, "type": "supergroup"},
                           "from": {"id": 5}, "text": "hello"}})
check("group message is refused, not queued",
      bot._backend.depth() == 0
      and any(T._t("group_chat", lg) in texts(bot) for lg in T._LANGS),
      texts(bot))
bot.sent.clear()
bot._dispatch({"message": {"chat": {"id": -100123, "type": "group"},
                           "new_chat_members": [{"id": 1}]}})
check("being added to a group does not spam the group", not bot.sent, texts(bot))


# ══════════════════════════════════════════════════════════════════════════════
section("PINNED FACTS — per-session isolation")

bot = make_bot()
s1 = fresh_session(bot, CID)
s2 = fresh_session(bot, CID2)
s1.set_tg_facts([{"ts": 1, "text": "user one lives in Berlin"}])
s2.set_tg_facts([{"ts": 1, "text": "user two lives in Tokyo"}])
check("sessions hold separate fact lists",
      s1.get_tg_facts()[0]["text"] != s2.get_tg_facts()[0]["text"])
check("facts survive a round-trip through the store",
      T._Session(CID, s1.to_dict()).get_tg_facts() == s1.get_tg_facts())
bot.sent.clear()
bot._send_facts(CID, s1)
check("facts are listed to the user", "Berlin" in texts(bot), texts(bot)[:200])
# Facts are injected into EVERY turn, so leaving them behind meant a user who
# cleared the chat was still told their own earlier steps back. "Clear" has to
# mean the model remembers nothing about this conversation.
check("clear_context drops pinned facts too",
      (s1.clear_context() or True) and s1.get_tg_facts() == [],
      s1.get_tg_facts())
check("and it does not touch the OTHER session's facts",
      len(s2.get_tg_facts()) == 1, s2.get_tg_facts())
s1.set_tg_facts([{"ts": 1, "text": "user one lives in Berlin"}])
s1.set_tg_facts([])
bot.sent.clear()
bot._send_facts(CID, s1)
check("empty fact list says so", "haven't saved" in texts(bot), texts(bot)[:200])


# ══════════════════════════════════════════════════════════════════════════════
section("CRASH RECOVERY")

bot = make_bot()
fresh_session(bot)
bot._running_task[CID] = [T._Task(task_id="dead1", chat_id=CID,
                                  user_text="find me a bus tour")]
bot._write_inflight()
check("in-flight tasks are persisted", T._INFLIGHT_FILE.exists())
bot._running_task.clear()
bot.sent.clear()
bot._recover_inflight()
check("interrupted user is notified", "restarted" in texts(bot).lower(), texts(bot)[:200])
check("retry text is stored for the retry button",
      bot._get_session(CID).last_task_text == "find me a bus tour",
      bot._get_session(CID).last_task_text)
check("inflight file is consumed", not T._INFLIGHT_FILE.exists())
bot.sent.clear()
bot._recover_inflight()
check("recovery is idempotent when nothing was interrupted", not bot.sent)

# The first message after a restart enqueues a task, which rewrites the
# journal. start() now takes the old journal before any thread runs; recovery
# used to read the file afterwards and found only the new task.
bot._running_task[CID] = [T._Task(task_id="dead2", chat_id=CID, user_text="old request")]
bot._write_inflight()
bot._running_task.clear()
taken = bot._take_inflight()
bot._pending_journal["new1"] = T._Task(task_id="new1", chat_id=CID + 1, user_text="hi again")
bot._write_inflight()                       # the new message, before recovery ran
bot.sent.clear()
bot._recover_inflight(taken)
check("a message right after restart does not erase the interrupted-task notice",
      "restarted" in texts(bot).lower() and bot._get_session(CID).last_task_text == "old request",
      texts(bot)[:200])
check("and the new task's journal entry is left alone",
      T._INFLIGHT_FILE.exists() and "new1" in T._INFLIGHT_FILE.read_text(encoding="utf-8"))
bot._pending_journal.clear()
T._INFLIGHT_FILE.unlink()
_s = bot._get_session(CID)                  # the retry checks below use the first task
_s.last_task_text, _s.last_task_id = "find me a bus tour", "dead1"
bot._store.put(_s)
import inspect as _inspect
_src = _inspect.getsource(T.TelegramBot.start)
check("start() takes the journal before starting consumer/poll threads",
      _src.index("_take_inflight()") < _src.index("_consumer_loop"))

# retry re-queues the stored request
bot._backend = T.InMemoryBackend()
enq = []
bot._enqueue_item = lambda cid, item: enq.append(item)
bot._dispatch({"callback_query": {"id": "1", "data": "retry",
                                  "message": {"chat": {"id": CID, "type": "private"}}}})
check("retry re-queues the last request",
      enq and enq[-1]["text"] == "find me a bus tour", enq)

# a STALE retry button (id-bearing, from a request that already got superseded
# by later activity) must NOT silently retry whatever the user asked most
# recently -- it must say the button is gone.
check("the recovered request's id was stored alongside its text",
      bot._get_session(CID).last_task_id == "dead1",
      bot._get_session(CID).last_task_id)

bot._get_session(CID).last_task_text = "what is 2+2"
bot._get_session(CID).last_task_id = "live2"
bot.sent.clear()
enq.clear()
bot._dispatch({"callback_query": {"id": "2", "data": "retry:dead1",
                                  "message": {"chat": {"id": CID, "type": "private"}}}})
check("a stale id-bearing retry button does not enqueue anything",
      not enq, enq)
check("a stale id-bearing retry button says it is gone",
      any("Nothing to retry" in t or "Нечего повторять" in t for _, t, _ in bot.sent),
      bot.sent)

bot.sent.clear()
enq.clear()
bot._dispatch({"callback_query": {"id": "3", "data": "retry:live2",
                                  "message": {"chat": {"id": CID, "type": "private"}}}})
check("a CURRENT id-bearing retry button re-queues its own request",
      enq and enq[-1]["text"] == "what is 2+2", enq)


# ══════════════════════════════════════════════════════════════════════════════
section("RETRY BUTTON — pushing a SECOND task must not stale a FIRST task's "
        "own retry before the first even fails")
# A chat may have two tasks in flight for the same chat (a slow one plus an
# admitted interject — see _mark_interruptible), or simply a second message
# queued right behind a slow first one. sess.last_task_id/last_task_text used
# to be written at task-PUSH time (tg_resolve.py): pushing task B immediately
# overwrote the pointer, even though B had not started running yet — so if
# task A then failed and showed its OWN, genuinely-fresh retry:<A's id>
# button, pressing it was refused as "stale" because the pointer already
# named B. Fixed by writing the pointer only at the moment each failure is
# actually DELIVERED (tg_tasks.py's two `except Exception` retry-button
# sites), not when a task is merely pushed to the queue.

bot = make_bot()
fresh_session(bot)
bot._backend = T.InMemoryBackend()
bot._get_session(CID).last_task_text = ""
bot._get_session(CID).last_task_id = ""

# Push A, then push B right behind it — mirroring two messages landing in
# separate debounce windows for the same chat. Neither has RUN yet.
bot._resolve_and_push(CID, [{"type": "text", "text": "task A will fail"}])
check("pushing A alone does not yet set the retry pointer",
      bot._get_session(CID).last_task_id == "",
      bot._get_session(CID).last_task_id)
bot._resolve_and_push(CID, [{"type": "text", "text": "task B queued behind"}])
check("pushing B (queued behind, not yet run) still does not touch the "
      "retry pointer -- this is the exact overwrite that used to stale A's "
      "not-yet-delivered failure",
      bot._get_session(CID).last_task_id == "",
      bot._get_session(CID).last_task_id)

task_a = bot._backend.pop(timeout=1.0)
check("A is the task that was actually popped first", task_a.user_text == "task A will fail",
      task_a.user_text if task_a else None)


class _RaisingGraph:
    def invoke(self, base):
        raise RuntimeError("boom-A")


class _FakeCtx:
    def __init__(self):
        self.session_memory = []
        self.pinned_facts = []
        self.cancel_event = threading.Event()
        self.last_image_path = ""
        self.last_image_prompt = ""
        self.stage_callback = None
        self.total_user_turns = 0
        self.tts_disabled = True
        self.memory_lock = threading.Lock()
    def memory_text(self): return ""
    def set_stage(self, s): pass


import graph as _graph_mod
_orig_build_graph = _graph_mod.build_graph
_graph_mod.build_graph = lambda ctx: _RaisingGraph()
bot._get_ctx = lambda: _FakeCtx()
bot._get_graph = lambda: _RaisingGraph()
try:
    bot._execute_task(task_a)
finally:
    _graph_mod.build_graph = _orig_build_graph

check("A's own failure sets the retry pointer to A's OWN id",
      bot._get_session(CID).last_task_id == task_a.task_id,
      (bot._get_session(CID).last_task_id, task_a.task_id))
check("A's own failure carries a matching retry: button",
      any(f"retry:{task_a.task_id}" in str(kw.get("keyboard"))
          for _, _, kw in bot.sent),
      bot.sent)

# Pressing A's own retry button now, with B still sitting in the backend
# never having run, must retry A -- not be refused as stale.
enq = []
bot._enqueue_item = lambda cid, item: enq.append(item)
bot.sent.clear()
bot._dispatch({"callback_query": {"id": "rt1", "data": f"retry:{task_a.task_id}",
                                  "message": {"chat": {"id": CID, "type": "private"}}}})
check("A's own retry button re-queues A's OWN request, not refused as stale "
      "merely because B was pushed behind it",
      enq and enq[-1]["text"] == "task A will fail", enq)
check("the retry was not refused with the 'nothing to retry' message",
      not any("Nothing to retry" in t or "Нечего повторять" in t
              for _, t, _ in bot.sent),
      bot.sent)


# ══════════════════════════════════════════════════════════════════════════════
section("DOCUMENT LIBRARY")

bot = make_bot()
bot._backend = T.InMemoryBackend()
s = fresh_session(bot)
indexed = []
bot._index_document = lambda cid, sess, path, fname, **kw: indexed.append(fname)

small = "short note"
big = "x " * 20000
bot._extract = None
_orig_extract = T._extract_doc
T._extract_doc = lambda path, fname: (big if "big" in fname else small)
try:
    bot._resolve_and_push(CID, [{"type": "document", "file_id": "d1",
                                 "filename": "big.pdf", "caption": ""}])
    check("a large document is indexed, not truncated into the prompt",
          indexed == ["big.pdf"] and bot._backend.depth() == 0, (indexed, bot._backend.depth()))
    indexed.clear()
    bot._resolve_and_push(CID, [{"type": "document", "file_id": "d2",
                                 "filename": "small.txt", "caption": ""}])
    # CONTRACT CHANGE (measured live): a small file is indexed as well, so it
    # actually appears in 📚 My documents — indexing only the big ones made a
    # short upload look like it had failed. With no caption there is nothing to
    # ask, so it is acknowledged rather than pushed to the agent.
    check("a small document is indexed too",
          indexed == ["small.txt"] and bot._backend.depth() == 0,
          (indexed, bot._backend.depth()))
    indexed.clear()
    bot._resolve_and_push(CID, [{"type": "document", "file_id": "d3",
                                 "filename": "small2.txt",
                                 "caption": "what is this?"}])
    check("a small document sent with a question still reaches the agent",
          bot._backend.depth() == 1, bot._backend.depth())
    check("inline document text is no longer truncated at 8k",
          "short note" in bot._backend.service_order()[0].user_text)
finally:
    T._extract_doc = _orig_extract

# toggling document search
bot.sent.clear()
bot._resolve_and_push(CID, [{"type": "text", "text": T._b("lib_on", "en")}])
check("document search toggles on", bot._get_session(CID).use_docs is True)
bot._resolve_and_push(CID, [{"type": "text", "text": T._b("lib_off", "en")}])
check("document search toggles off", bot._get_session(CID).use_docs is False)
bot.sent.clear()
bot._send_library_list(CID, bot._get_session(CID))
check("empty library explains how to add documents",
      "no documents" in texts(bot).lower(), texts(bot)[:200])
check("library path is per user",
      bot._library_path(CID) != bot._library_path(CID2))


# ══════════════════════════════════════════════════════════════════════════════
section("LONG REPORT DELIVERY")

bot = make_bot()
bot._send_report_file(CID, "bus tours in spb", "R" * 9000, "hdr", "en")
check("a long report is delivered as a file", len(bot.docs) == 1, bot.docs)
check("the attachment is a .md file",
      bot.docs and bot.docs[0][1].endswith(".md"), bot.docs)


# ══════════════════════════════════════════════════════════════════════════════
section("WATCHDOG")

bot = make_bot()
bot._running = True
dead = threading.Thread(target=lambda: None)
dead.start()
dead.join()
bot._consumers = [dead]
bot._poll_thread = dead
import config as _cfg
_sv = _cfg.TG_WATCHDOG_S
_cfg.TG_WATCHDOG_S = 5          # clamped to a 5s floor inside the loop
try:
    bot._poll_loop = lambda: None
    bot._consumer_loop = lambda: None
    wd = threading.Thread(target=bot._watchdog_loop, daemon=True)
    wd.start()
    import time as _time
    _time.sleep(6.0)
    bot._running = False
    check("watchdog restarted the dead poll thread",
          bot._poll_thread is not dead, bot._poll_thread)
    check("watchdog restarted the dead consumer",
          bot._consumers[0] is not dead, bot._consumers)
finally:
    _cfg.TG_WATCHDOG_S = _sv


# ══════════════════════════════════════════════════════════════════════════════
section("ADMIN — broadcast, stats, and the new commands")

bot = make_bot()
bot._backend = T.InMemoryBackend()
admin = T._User(chat_id=CID, name="Boss", status="approved", is_admin=True)
bot._user_store.put(admin)
# A confirmed bug fix (2026-08-03) made manual /broadcast honor each recipient's
# notification opt-out, the same way startup/shutdown broadcasts already did —
# so a member with no explicit subscription must be given one here, or this
# fixture is testing the exact behavior ("reaches everyone regardless of
# opt-out") that was just fixed.
bot._user_store.put(T._User(chat_id=CID2, name="Member", status="approved",
                            subscriptions=["startup"]))
fresh_session(bot)

bot._handle_command(CID, "/broadcast server maintenance at 20:00")
check("broadcast asks for confirmation before sending",
      "Send this to" in texts(bot) and not any(c == CID2 for c, _, _ in bot.sent),
      texts(bot)[:200])
check("draft is held pending confirmation",
      bot._pending_broadcast.get(CID, "").startswith("server maintenance"))

bot.sent.clear()
bot._confirm_broadcast(CID, "no")
check("declining sends nothing to users",
      not any(c == CID2 for c, _, _ in bot.sent), texts(bot)[:200])

bot._handle_command(CID, "/broadcast hello everyone")
bot.sent.clear()
bot._confirm_broadcast(CID, "go")
check("confirmed broadcast reaches other users",
      any(c == CID2 and "hello everyone" in t for c, t, _ in bot.sent), texts(bot)[:300])
check("draft is consumed after sending", CID not in bot._pending_broadcast)

bot.sent.clear()
bot._confirm_broadcast(CID, "go")
check("a consumed draft cannot be re-sent", not bot.sent)

# non-admins cannot broadcast
bot._user_store.put(T._User(chat_id=CID2, name="Member", status="approved"))
bot.sent.clear()
bot._handle_command(CID2, "/broadcast i am not an admin")
check("non-admin broadcast is ignored",
      not bot.sent and CID2 not in bot._pending_broadcast, texts(bot)[:200])

bot.sent.clear()
bot._send_admin_stats(CID)
check("usage stats render", "Usage" in texts(bot) or "No usage" in texts(bot),
      texts(bot)[:200])

# /lang switches and the keyboard follows
bot._user_store.put(admin)
fresh_session(bot)
bot._handle_command(CID, "/lang ru")
check("/lang ru switches the session language",
      bot._get_session(CID).lang == "ru", bot._get_session(CID).lang)
kb = bot.sent[-1][2].get("keyboard", {})
check("keyboard is rendered in the new language",
      any("Стоп" in b for row in kb.get("keyboard", []) for b in row), kb)
bot.sent.clear()
bot._handle_command(CID, "/lang")
check("/lang with no argument offers a choice",
      "language" in texts(bot).lower() or "язык" in texts(bot).lower(), texts(bot)[:120])
bot._dispatch({"callback_query": {"id": "1", "data": "lang:en",
                                  "message": {"chat": {"id": CID, "type": "private"}}}})
check("language callback applies", bot._get_session(CID).lang == "en")

# An explicit choice must survive the NEXT message, whatever the client's own
# locale says — otherwise "adopt the client language on first contact" would
# quietly undo every switch the user makes.
bot._dispatch({"message": {"chat": {"id": CID, "type": "private"},
                           "from": {"id": CID, "language_code": "ru-RU"},
                           "text": "/status"}})
check("an explicit English choice is not overwritten by the client locale",
      bot._get_session(CID).lang == "en", bot._get_session(CID).lang)

# A brand-new user whose client speaks a language we do not: Russian, the house
# default — not English, and not a screenful of raw message keys.
NEWCID = CID + 4242
bot._dispatch({"message": {"chat": {"id": NEWCID, "type": "private"},
                           "from": {"id": NEWCID, "language_code": "de-DE"},
                           "text": "/start"}})
check("an unknown client locale starts in Russian",
      bot._get_session(NEWCID).lang == "ru", bot._get_session(NEWCID).lang)
# And an English client still gets English on first contact.
ENCID = CID + 4343
bot._dispatch({"message": {"chat": {"id": ENCID, "type": "private"},
                           "from": {"id": ENCID, "language_code": "en-US"},
                           "text": "/start"}})
check("an English client still starts in English",
      bot._get_session(ENCID).lang == "en", bot._get_session(ENCID).lang)

# account menu shows quota for normal users, and hides it for admins
bot._user_store.put(T._User(chat_id=CID, name="Tester", status="approved"))
bot.sent.clear()
bot._send_account_menu(CID, bot._user_store.get(CID), bot._get_session(CID))
check("account menu shows today's usage", "Today" in texts(bot), texts(bot)[:300])
check("account menu offers a language button",
      "acct_lang" in str(bot.sent[-1][2]), str(bot.sent[-1][2])[:200])
bot._user_store.put(admin)
bot.sent.clear()
bot._send_account_menu(CID, bot._user_store.get(CID), bot._get_session(CID))
check("admins are not shown a quota they do not have",
      "Today" not in texts(bot), texts(bot)[:300])

section("markdown -> Telegram HTML")

# A stray "_" reached a user as a literal character because only *italic* was
# handled. Underscore emphasis is now supported, but the load-bearing half of
# these cases is the NEGATIVE ones: research replies are mostly links, and
# corrupting a URL is far worse than a stray underscore.
_MD_CASES = [
    ("_italic_", "<i>italic</i>"),
    ("__bold__", "<b>bold</b>"),
    ("suits you._", "suits you._"),                      # lone trailing underscore
    ("_Here is a list._", "<i>Here is a list.</i>"),
    ("_multi word italic_ tail", "<i>multi word italic</i> tail"),
    # negatives: identifiers, numbers, code and URLs must survive untouched
    ("snake_case_name stays", "snake_case_name stays"),
    ("5_000 and 6_000", "5_000 and 6_000"),
    ("`a_b_c`", "<code>a_b_c</code>"),
    ("visit https://site.ru/_foo_/bar now", "visit https://site.ru/_foo_/bar now"),
    ("[nevapiter](https://nevapiter.com/?utm_source=x_y)",
     '<a href="https://nevapiter.com/?utm_source=x_y">nevapiter</a>'),
    ("a _b_ and [l](https://x.com/a_b)",
     'a <i>b</i> and <a href="https://x.com/a_b">l</a>'),
    # existing dialects must keep working
    ("**bold** and *it*", "<b>bold</b> and <i>it</i>"),
    ("# Heading", "<b>Heading</b>"),
    ("- item", "• item"),
    # A URL may contain its own balanced parentheses — Wikipedia titles are full
    # of them, and a truncated href is a dead link plus a stray ")" in the text.
    ("See [Mercury](https://en.wikipedia.org/wiki/Mercury_(planet)) now",
     'See <a href="https://en.wikipedia.org/wiki/Mercury_(planet)">Mercury</a> now'),
    # …but a link inside a parenthesised aside must not swallow the closing ")"
    ("(see [x](https://a.example/b))",
     '(see <a href="https://a.example/b">x</a>)'),
    # NUL is the internal stash marker; text carrying one must not raise
    ("a \x000\x00 b _x_ c", "a 0 b <i>x</i> c"),
    # Emphasis passes must NEST, not cross: overlapping dialects used to yield
    # "<i><b>x</i></b>", which Telegram refuses outright.
    ("*a **b** c*", "<i>a <b>b</b> c</i>"),
    ("**bold with [l](https://e.example/a) inside**",
     '<b>bold with <a href="https://e.example/a">l</a> inside</b>'),
    ("## Sub *heading*", "<b>Sub <i>heading</i></b>"),
    ("- **a**: text\n- **b**: more", "• <b>a</b>: text\n• <b>b</b>: more"),
    # a report line as the models actually emit it
    ("1. **Nevapiter** — [site](https://nevapiter.com/tours_(bus))",
     '1. <b>Nevapiter</b> — <a href="https://nevapiter.com/tours_(bus)">site</a>'),
]
for _src, _want in _MD_CASES:
    _got = T._md_to_html(_src)
    check(f"md: {_src[:38]}", _got == _want, f"got {_got!r} want {_want!r}")

section("HTML chunking never emits a half-written tag")

# Telegram rejects the WHOLE message on a malformed tag, so a bad split does not
# degrade the formatting — it deletes the answer. _hard_split used to accept any
# space as a cut point, but tags carry spaces of their own (<a href="a b">).
_over = "x" * 4000 + ' <a href="https://e.example/a b c">tail</a>' + "y" * 500
for _i, _c in enumerate(T._split_html(_over, limit=4096)):
    check(f"hard-split chunk {_i} has no dangling tag",
          _c.count("<") == _c.count(">"), repr(_c[-70:]))
    check(f"hard-split chunk {_i} within limit", len(_c) <= 4096, str(len(_c)))

# Deep nesting + long hrefs reopened across a boundary must still fit.
_deep = "".join(
    f'<b><i><a href="https://example.com/{"p" * 200}">link {_n} body text</a></i></b>\n\n'
    for _n in range(80))
check("balanced chunks stay within the Telegram limit",
      all(len(c) <= 4096 for c in T._split_html(_deep, limit=4096)),
      str(max(len(c) for c in T._split_html(_deep, limit=4096))))

section("a rejected-markup reply is resent as plain text, never dropped")

# The response from sendMessage was previously ignored, so "can't parse entities"
# was indistinguishable from success and the user silently got NOTHING.
class _FakeAPI:
    """Rejects the first HTML attempt the way Telegram does, accepts plain text."""
    def __init__(self): self.calls = []
    def __call__(self, method, payload=None, **kw):
        self.calls.append((method, dict(payload or {})))
        if (payload or {}).get("parse_mode"):
            return {"ok": False, "error_code": 400,
                    "description": "Bad Request: can't parse entities: "
                                   "Unclosed start tag at byte offset 100"}
        return {"ok": True, "result": {"message_id": 7}}

_bot_pm = bot
_saved_api = _bot_pm._api_post
_fake = _FakeAPI()
_bot_pm._api_post = _fake
# make_bot() stubs _send_text for capture, so call the REAL one explicitly —
# the retry lives there and a stub would test nothing.
_real_send_text = T.TelegramBot._send_text
try:
    _real_send_text(_bot_pm, CID,
                    '<b>Answer</b> <a href="https://a.example/x">source</a>',
                    parse_mode="HTML")
finally:
    _bot_pm._api_post = _saved_api

check("markup rejection triggers a second attempt", len(_fake.calls) == 2,
      str(_fake.calls))
_retry = _fake.calls[-1][1] if len(_fake.calls) == 2 else {}
check("retry drops parse_mode", "parse_mode" not in _retry, str(_retry))
check("retry keeps the answer text", "Answer" in _retry.get("text", ""),
      str(_retry.get("text")))
check("retry keeps the link target so sources are not lost",
      "https://a.example/x" in _retry.get("text", ""), str(_retry.get("text")))
check("retry carries no leftover tags", "<" not in _retry.get("text", ""),
      str(_retry.get("text")))

# A non-formatting failure must NOT be retried: resending on "bot was blocked"
# would just double every failed delivery.
class _BlockedAPI:
    def __init__(self): self.calls = []
    def __call__(self, method, payload=None, **kw):
        self.calls.append(method)
        return {"ok": False, "error_code": 403,
                "description": "Forbidden: bot was blocked by the user"}
_blocked = _BlockedAPI()
_bot_pm._api_post = _blocked
try:
    _real_send_text(_bot_pm, CID, "<b>hi</b>", parse_mode="HTML")
finally:
    _bot_pm._api_post = _saved_api
check("a non-markup failure is not retried", len(_blocked.calls) == 1,
      str(_blocked.calls))

check("_html_to_plain keeps link targets",
      T._html_to_plain('<b>a</b> <a href="https://u.example">label</a>')
      == "a label (https://u.example)",
      T._html_to_plain('<b>a</b> <a href="https://u.example">label</a>'))
check("_html_to_plain does not duplicate a bare-URL label",
      T._html_to_plain('<a href="https://u.example">https://u.example</a>')
      == "https://u.example")
check("_html_to_plain unescapes entities",
      T._html_to_plain("a &amp; b &lt;c&gt;") == "a & b <c>")

section("debounce worker never leaves an item unserved")

# The worker deregisters itself on an idle timeout. If an item can be put into
# the queue between "worker timed out" and "worker deregistered", is_alive() is
# still True, no replacement starts, and the message sits in a queue nobody
# serves — the user just gets no reply until they send something else.
# The invariant that closes it: the put and the is_alive check happen under the
# same lock that the worker's exit path takes.
import queue as _pyq
import threading as _th

class _AssertingQueue(_pyq.Queue):
    def __init__(self, probe):
        super().__init__(); self._probe = probe; self.put_under_lock = []
    def put(self, item, *a, **kw):
        self.put_under_lock.append(self._probe())
        return super().put(item, *a, **kw)

class _TrackingLock:
    def __init__(self): self._l = _th.RLock(); self.depth = 0
    def __enter__(self): self._l.acquire(); self.depth += 1; return self
    def __exit__(self, *a): self.depth -= 1; self._l.release()
    def acquire(self, *a, **k): return self._l.acquire(*a, **k)
    def release(self): return self._l.release()

_tl = _TrackingLock()
bot._mgmt_lock = _tl
_aq = _AssertingQueue(lambda: _tl.depth > 0)
bot._queues[CID] = _aq
bot._workers.pop(CID, None)
_saved_debounce = bot._debounce_loop
bot._debounce_loop = lambda cid: None          # don't actually drain in this check
bot._enqueue_item(CID, {"type": "text", "text": "hello"})
check("item is enqueued while holding the worker-registry lock",
      _aq.put_under_lock == [True], str(_aq.put_under_lock))
bot._debounce_loop = _saved_debounce

# …and the exit path must not deregister while work is pending: with a non-empty
# queue the loop keeps serving instead of returning.
bot._queues[CID] = _pyq.Queue()
bot._queues[CID].put({"type": "text", "text": "raced in"})
bot._workers[CID] = _th.current_thread()
_served = []
_saved_resolve = bot._resolve_and_push
bot._resolve_and_push = lambda cid, batch: _served.extend(batch)
_saved_idle = T._WORKER_IDLE_S
T._WORKER_IDLE_S = 0.01
try:
    _t = _th.Thread(target=bot._debounce_loop, args=(CID,), daemon=True)
    _t.start(); _t.join(5.0)
finally:
    T._WORKER_IDLE_S = _saved_idle
    bot._resolve_and_push = _saved_resolve
check("an item already queued at idle-timeout is still served",
      [b.get("text") for b in _served] == ["raced in"], str(_served))

section("photo size selection")

# file_size is OPTIONAL in Telegram's PhotoSize. With it missing, keying on
# file_size alone made every candidate 0 and max() returned the FIRST entry —
# the smallest thumbnail — so the vision model was handed a postage stamp.
_photo_sizes = [
    {"file_id": "thumb", "width": 90, "height": 60},
    {"file_id": "mid", "width": 320, "height": 213},
    {"file_id": "full", "width": 1280, "height": 853},
]
bot.sent.clear()
_captured = {}
bot._enqueue_item = lambda cid, item: _captured.update(item)
bot._dispatch({"message": {"chat": {"id": CID}, "from": {"id": CID},
                           "message_id": 1, "photo": _photo_sizes}})
check("largest photo chosen when file_size is absent",
      _captured.get("file_id") == "full", str(_captured))

# …and file_size still wins when Telegram does provide it
_captured.clear()
bot._dispatch({"message": {"chat": {"id": CID}, "from": {"id": CID}, "message_id": 2,
                           "photo": [
                               {"file_id": "small", "width": 90, "height": 60,
                                "file_size": 1000},
                               {"file_id": "big", "width": 1280, "height": 853,
                                "file_size": 90000}]}})
check("largest photo chosen when file_size is present",
      _captured.get("file_id") == "big", str(_captured))

# A hostile/corrupted update can send `photo` as anything -- a string, a
# dict, a list full of non-dict junk. `max(photos, key=lambda p: p.get(...))`
# on a string silently iterates its CHARACTERS and crashes the instant it
# calls .get() on the first one ('str' object has no attribute 'get'), an
# unhandled exception straight out of _dispatch. Every other malformed-field
# path in this router degrades to silence; this one must too.
for _bad_photos in ("not-a-list", {"file_id": "x"}, ["junk", 123, None],
                    [], None, [{"width": 10}]):   # last: dict but no file_id
    bot.sent.clear()
    _captured.clear()
    _crashed = False
    try:
        bot._dispatch({"message": {"chat": {"id": CID}, "from": {"id": CID},
                                   "message_id": 3, "photo": _bad_photos}})
    except Exception as exc:
        _crashed = True
        _detail = repr(exc)
    check(f"malformed photo field {_bad_photos!r} does not crash _dispatch",
          not _crashed, _detail if _crashed else "")
    check(f"malformed photo field {_bad_photos!r} enqueues nothing "
          "(silent no-op, matching every other malformed-field path)",
          not _captured, _captured)

# a reply that carries the same malformed shape (_resolve_reply_target's own
# call site) must degrade the same way, not crash the reply-target resolver
_crashed = False
try:
    _resolved = bot._resolve_reply_target(CID, {
        "reply_to_message": {"message_id": 999, "photo": "not-a-list"}})
except Exception as exc:
    _crashed = True
    _detail = repr(exc)
check("malformed photo in a reply_to_message does not crash "
      "_resolve_reply_target", not _crashed, _detail if _crashed else "")

section("edited_message -- documenting CURRENT behaviour, not asserting "
        "it is the right one")

# _dispatch reads `upd.get("message") or upd.get("edited_message")` at a
# single point (tg_dispatch.py) with no branch that tells the two apart
# anywhere downstream. An edited_message is therefore processed EXACTLY like
# a brand-new message with the edited text: same registration gate, same
# debounce queue, same task push -- including running the agent a SECOND
# time for a message the user only meant to fix a typo in. There is no
# dedup against the original send (Telegram reuses the same message_id for
# an edit, but nothing here keys off message_id to recognise that). This is
# pinned as the CURRENT, observed contract; whether an edit SHOULD trigger a
# fresh turn is a product decision out of scope here.
_edit_bot = make_bot()
_edit_bot._backend = T.InMemoryBackend()
_edit_bot._activity.log = lambda *a, **k: None
_EID = 999109
_edit_bot._user_store.put(T._User(chat_id=_EID, name="Edit", status="approved"))
_esess = _edit_bot._get_session(_EID); _esess.clear_context(); _esess.lang = "en"
_edit_bot._store.put(_esess)

import time as _time

_edit_bot._dispatch({"message": {"message_id": 42,
                                 "chat": {"id": _EID, "type": "private"},
                                 "from": {"id": _EID}, "text": "orginal typo"}})
_time.sleep(2.2)   # clear the debounce window so it resolves to a pushed task
_task1 = _edit_bot._backend.pop(timeout=0.5)
check("an ordinary message pushes a task with its own text",
      _task1 is not None and _task1.user_text == "orginal typo",
      _task1)

_edit_bot._dispatch({"edited_message": {"message_id": 42,   # SAME message_id
                                        "chat": {"id": _EID, "type": "private"},
                                        "from": {"id": _EID}, "text": "original fixed"}})
_time.sleep(2.2)
_task2 = _edit_bot._backend.pop(timeout=0.5)
check("CURRENT BEHAVIOUR: an edited_message for the SAME message_id is "
      "processed as an entirely independent new turn -- it pushes its own "
      "task with the edited text, with no awareness this message_id was "
      "already handled once",
      _task2 is not None and _task2.user_text == "original fixed", _task2)

# A crashed/malformed edited_message (no "text" key at all) must degrade the
# same way an equally malformed ordinary message would -- not crash, not
# raise past _dispatch.
_crashed = False
try:
    _edit_bot._dispatch({"edited_message": {"message_id": 43,
                                             "chat": {"id": _EID, "type": "private"},
                                             "from": {"id": _EID}}})   # no text
except Exception as exc:
    _crashed = True; _detail = repr(exc)
check("a text-less edited_message does not crash _dispatch",
      not _crashed, _detail if _crashed else "")

section("stop() must not cancel the DESKTOP assistant")
# ctx is shared with the GUI. Pressing "Stop bot" in the Telegram tab used to set
# ctx.cancel_event unconditionally, aborting whatever the desktop turn was doing.
import threading as _threading

_gui_ctx = types.SimpleNamespace(cancel_event=_threading.Event())
bot2 = make_bot()
bot2._get_ctx = lambda: _gui_ctx
bot2._broadcast_shutdown = lambda: None
bot2._backend.close = lambda: None
bot2._on_status = lambda *a, **k: None
bot2._running_task.clear()          # no bot task in flight
# stop() is now a no-op for a bot that never started (it used to log "Bot stopped"
# and broadcast "going offline" for one that had never been online), so these two
# cases have to represent a bot that IS up — which is the only state in which
# pressing Stop means anything.
bot2._running = True
bot2.stop()
check("stop() leaves an idle ctx uncancelled",
      not _gui_ctx.cancel_event.is_set(),
      "Stop bot cancelled the desktop assistant's turn")

# …but it must still cancel when one of OUR tasks is genuinely running —
# via THAT TASK'S OWN scoped cancel event (_task_cancels), never the shared
# desktop ctx. Every task runs on its own _scoped_ctx with its own event
# (_execute_task); setting the shared ctx here used to abort whatever the
# desktop GUI happened to be doing while the actual Telegram task in flight
# was left running unaffected — the same bug fixed on the watchdog path.
_gui_ctx.cancel_event.clear()
bot3 = make_bot()
bot3._get_ctx = lambda: _gui_ctx
bot3._broadcast_shutdown = lambda: None
bot3._backend.close = lambda: None
bot3._on_status = lambda *a, **k: None
own_ev = _threading.Event()
bot3._running_task[CID] = object()
bot3._task_cancels["own-task"] = own_ev
bot3._running = True
bot3.stop()
check("stop() cancels its own running task's OWN event",
      own_ev.is_set())
check("stop() still does NOT touch the shared desktop ctx",
      not _gui_ctx.cancel_event.is_set())

print()
print(f"{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)

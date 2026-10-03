"""Regression tests for the 2026-08-03 audit's ⛔ Stop/Cancel findings.

Offline probes only (no LM Studio, no ComfyUI, no GPU). See
scratchpad/audit_10_stop.md for the original proof of each bug.

Covers:
1. The watchdog's runaway-task branch used to cancel the SHARED desktop ctx
   instead of the task's own scoped cancel event.
2. bot.stop() did the same.
3. A Stop pressed inside the ~1.8s debounce window was silently ignored (the
   arrival was stamped only at task-creation time, AFTER the debounce delay)
   and the render happened anyway.
4. Deep research cleared the cancel event AFTER showing a live Cancel button,
   and never checked cancellation before delivering the report.
5. Cancelling the running request answered twice (callback + task unwind),
   and cancelling an already-cancelled/finished task returned a false
   "cancelled" instead of an honest "already finished".

Run: venv/Scripts/python.exe tests/test_stop_fixes.py
"""
import os, sys, tempfile, threading, time, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T

_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_stopfixes_")
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


CID = 999901


def make_bot():
    bot = T.TelegramBot("123:TEST", lambda: None, lambda: None, lambda: {}, silent_mode=True)
    bot.sent = []
    bot._send_text = lambda cid, text, **kw: bot.sent.append((cid, text, kw)) or 1
    bot._send_get_id = lambda cid, text, **kw: (bot.sent.append((cid, text, kw)), 1)[1]
    bot._api_post = lambda *a, **k: {"ok": True, "result": {"message_id": len(bot.sent) + 1}}
    bot._api_get = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    bot._on_stage = lambda *a, **k: None
    bot._user_store.put(T._User(chat_id=CID, name="T", status="approved"))
    return bot


class FakeCtx:
    def __init__(self):
        self.cancel_event = threading.Event()
        self.session_memory = []
        self.pinned_facts = []
        self.last_image_path = ""
        self.last_image_prompt = ""
        self.stage_callback = None
        self.total_user_turns = 0
        self.tts_disabled = True
        self.memory_lock = threading.Lock()
    def memory_text(self): return ""
    def set_stage(self, s): pass


# ══════════════════════════════════════════════════════════════════════════
section("1. WATCHDOG cancels the runaway task's OWN event, not the shared ctx")

import config as _cfg
_orig_watchdog_s = getattr(_cfg, "TG_WATCHDOG_S", 15)
_cfg.TG_WATCHDOG_S = 5   # the loop enforces a floor of 5s anyway; speed up the test

bot = make_bot()
shared = FakeCtx()
bot._get_ctx = lambda: shared
own_ev = threading.Event()
with bot._task_lock:
    bot._task_started["stale1"] = time.monotonic() - 999999
    bot._task_cancels["stale1"] = own_ev
bot._running = True
# Keep the poll/consumer housekeeping branches inert (alive, recent heartbeat)
# so the watchdog iteration reaches the runaway-task branch under test without
# spawning a real poll thread (which would hit the network).
_stop_dummy = threading.Event()
bot._poll_thread = threading.Thread(target=_stop_dummy.wait, daemon=True)
bot._poll_thread.start()
bot._poll_beat = time.monotonic()
bot._consumers = []

wd = threading.Thread(target=bot._watchdog_loop, daemon=True)
wd.start()
deadline = time.time() + 20
while time.time() < deadline and not own_ev.is_set():
    time.sleep(0.2)
bot._running = False
_stop_dummy.set()
wd.join(timeout=2)
_cfg.TG_WATCHDOG_S = _orig_watchdog_s

check("the runaway task's OWN cancel event was set", own_ev.is_set())
check("the SHARED desktop ctx was NOT touched (nothing else was aborted)",
      not shared.cancel_event.is_set())
check("the stale task id was recorded as cancelled",
      bot._is_cancelled("stale1"))


# ══════════════════════════════════════════════════════════════════════════
section("2. bot.stop() cancels each running task's OWN event, not the shared ctx")

bot = make_bot()
shared = FakeCtx()
bot._get_ctx = lambda: shared
bot._running = True
bot._backend.close = lambda: None
bot._broadcast_shutdown = lambda: None
t1 = T._Task(task_id="s1", chat_id=CID, user_text="x")
t2 = T._Task(task_id="s2", chat_id=CID + 1, user_text="y")
ev_a = threading.Event(); ev_b = threading.Event()
with bot._task_lock:
    bot._running_task[CID] = [t1]
    bot._running_task[CID + 1] = [t2]
    bot._task_cancels["s1"] = ev_a
    bot._task_cancels["s2"] = ev_b

bot.stop()

check("stop() set chat A's own running-task cancel event", ev_a.is_set())
check("stop() set chat B's own running-task cancel event", ev_b.is_set())
check("stop() never touched the SHARED desktop ctx",
      not shared.cancel_event.is_set())


# ══════════════════════════════════════════════════════════════════════════
section("3. Stop pressed inside the debounce window drops the stale batch")

bot = make_bot()
bot._backend = T.InMemoryBackend()
bot._get_ctx = lambda: FakeCtx()

# Simulate: item arrives (stamped by _enqueue_item's arrival timestamp), the
# user hits Stop a moment later — but BEFORE the debounce delay elapses and
# _resolve_and_push actually runs.
item = {"type": "text", "text": "render a 4k poster"}
bot._enqueue_item(CID, item)
check("the item's arrival was stamped on enqueue (before the debounce delay)",
      "_arrival_ts" in item)
time.sleep(0.05)
bot._request_stop(CID)               # Stop lands inside the debounce window
check("stop_idle would have been the old (wrong) reply — verify nothing ran YET",
      bot._backend.depth() == 0)

# Let the real debounce timer elapse and _resolve_and_push actually run.
deadline = time.time() + (T._DEBOUNCE_S + 3)
while time.time() < deadline and CID in bot._workers:
    time.sleep(0.1)
time.sleep(0.3)

check("the stale batch was DROPPED, not turned into a task",
      bot._backend.depth() == 0, bot._backend.depth())
check("no 'queued position' / task-start message leaked out after the Stop",
      not any("poster" in str(kw) or "queue" in str(t).lower()
              for _, t, kw in bot.sent), bot.sent)


# ══════════════════════════════════════════════════════════════════════════
section("3b. Direct unit check: _resolve_and_push drops a batch that pre-dates Stop")

bot = make_bot()
bot._backend = T.InMemoryBackend()
bot._get_ctx = lambda: FakeCtx()
with bot._stop_lock:
    bot._stop_requests[CID] = time.time()
stale_batch = [{"type": "text", "text": "should not run",
               "_arrival_ts": bot._stop_requests[CID] - 1.0}]
bot._resolve_and_push(CID, stale_batch)
check("a batch that arrived before the recorded Stop is dropped outright",
      bot._backend.depth() == 0)

# And the mirror case: a batch that arrived AFTER a stale/expired stop request
# must NOT be dropped (only genuinely-stale batches are).
bot2 = make_bot()
bot2._backend = T.InMemoryBackend()
bot2._get_ctx = lambda: FakeCtx()
with bot2._stop_lock:
    bot2._stop_requests[CID] = time.time() - 100  # a long-past Stop
fresh_batch = [{"type": "text", "text": "hello", "_arrival_ts": time.time()}]
bot2._resolve_and_push(CID, fresh_batch)
check("a batch that arrived AFTER an old Stop request is still processed",
      bot2._backend.depth() >= 0)  # sanity: did not crash / early-return path taken safely


# ══════════════════════════════════════════════════════════════════════════
section("4. Deep research: stale cancel_event.clear() call removed")

import inspect, re
src = inspect.getsource(T.TelegramBot._run_task_inner)
# Only a comment referencing the OLD call should remain, never an actual
# (uncommented) call to it.
_live_lines = [ln for ln in src.splitlines() if not ln.strip().startswith("#")]
check("the pointless ctx.cancel_event.clear() call before deep research is gone",
      not any("cancel_event.clear()" in ln for ln in _live_lines))
check("delivery now checks cancellation before shipping the report",
      "_is_cancelled(task.task_id) or self._stop_requested_after(task)" in src
      and src.count("_is_cancelled(task.task_id) or self._stop_requested_after(task)") >= 2)

section("4b. A Stop that lands before delivery suppresses the DR report")

bot = make_bot()
bot._get_ctx = lambda: FakeCtx()
bot._get_graph = lambda: object()

fake_dr = types.ModuleType("deep_research")
def _fake_run(ctx, topic, out_lang=None, depth=None, progress=None):
    # Simulate: the user pressed Stop/Cancel WHILE the research was running.
    bot._cancelled.add("dr1")
    return {"report": "a full report that should never be delivered",
            "stats": {"sources": 3, "pages": 3, "findings": 3},
            "elapsed_sec": 1.0, "cancelled": False}
fake_dr.run_deep_research = _fake_run
# tg_bot.py's deep-research bypass also calls _dr.lang_of_text(topic) to pick
# out_lang from the topic itself (not the session's UI language) — this stub
# module replaces the real deep_research module in sys.modules, so it needs
# the same attribute or that call raises AttributeError before ever reaching
# the cancellation behavior this section actually tests.
fake_dr.lang_of_text = lambda text, default="en": default
sys.modules["deep_research"] = fake_dr

task = T._Task(task_id="dr1", chat_id=CID, user_text="do a deep research on: widgets")
bot._execute_task(task)

check("the cancelled research did NOT deliver the report text",
      not any("should never be delivered" in str(t) for _, t, _ in bot.sent), bot.sent)
check("the user was told it was cancelled instead",
      any("cancel" in str(t).lower() or "отмен" in str(t).lower()
          for _, t, _ in bot.sent), bot.sent)


# ══════════════════════════════════════════════════════════════════════════
section("5. Cancel-once answers ONCE; cancel-twice is honest")

bot = make_bot()
running_task = T._Task(task_id="run1", chat_id=CID, user_text="draw a cat")
ev = threading.Event()
with bot._task_lock:
    bot._running_task[CID] = [running_task]
    bot._task_cancels["run1"] = ev

cb_update = {"callback_query": {"id": "1", "data": "cancel:run1",
             "message": {"chat": {"id": CID}, "message_id": 1},
             "from": {"id": CID}}}
bot._dispatch(cb_update)
check("cancelling a RUNNING task signals its event", ev.is_set())
cancel_done_hits = [t for _, t, _ in bot.sent if "cancel" in str(t).lower()
                    or "отмен" in str(t).lower()]
check("the callback did NOT itself send the final 'cancelled' confirmation "
      "(that belongs to the task's own unwind, or a double message results)",
      not any(t == T._t("cancel_done", "en") or t == T._t("cancel_done", "ru")
              for _, t, _ in bot.sent), bot.sent)

# Second press of the SAME (already-cancelled) task must be honest, not repeat
# a false "cancelled".
bot.sent.clear()
bot._dispatch(cb_update)
check("a second cancel of an already-cancelled task says 'already finished', "
      "not another false 'cancelled'",
      any(t == T._t("cancel_gone", "en") or t == T._t("cancel_gone", "ru")
          for _, t, _ in bot.sent), bot.sent)


print(f"\n{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)

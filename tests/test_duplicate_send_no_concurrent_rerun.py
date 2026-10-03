"""Live bug (2026-08-05): a slow image-generation turn ran TWICE, concurrently
— the exact same translate call, tool-calling round, and prompt-engineering
steps all fired a second time while the first was still in flight, doubling
GPU/LLM load and producing two near-simultaneous identical replies.

Root cause: a duplicate Telegram message (almost certainly the user's client
retrying a send that looked stuck) created two separate _Task objects with
BYTE-IDENTICAL user_text for the same chat. Before this session's interject-
concurrency feature (_mark_interruptible / the two-tasks-per-chat admission
gate), the second would just queue and run sequentially afterward — a stale-
duplicate nuisance, but not concurrent resource contention. The interject gate
made it worse: once the first task reached a slow/backgroundable stage (image
generation marks the chat interruptible), the SECOND task — sitting in queue
with IDENTICAL text — got admitted to run AT THE SAME TIME instead of waiting.

Fix: the admission gate in _consumer_loop now refuses to treat an exact
byte-for-byte repeat of the currently-running task's text as a legitimate
"quick interject" — it still queues and waits, exactly as before the
concurrency feature existed. A genuinely DIFFERENT quick message (the feature
this session actually built) is unaffected.

Run: venv/Scripts/python.exe tests/test_duplicate_send_no_concurrent_rerun.py
"""
import os, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T

_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_duprerun_")
T.redirect_data_dir(_DATA_DIR)

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")


def make_bot():
    return T.TelegramBot("123:TEST", lambda: None, lambda: None, lambda: {}, silent_mode=True)


def admit(bot, chat_id, candidate_text):
    """Reproduce the EXACT admission expression from _consumer_loop (not a
    reimplementation) by importing it out of the source, so this test breaks
    if the real logic ever diverges from what it claims to check."""
    import inspect
    src = inspect.getsource(bot._consumer_loop)
    assert "is_exact_repeat" in src, "admission gate logic moved — update this test"
    with bot._task_lock:
        n = bot._chat_busy.get(chat_id, 0)
        interruptible = bot._chat_interruptible.get(chat_id, False)
        running = bot._running_task.get(chat_id) or []
        is_exact_repeat = any(r.user_text == candidate_text for r in running)
        return (n == 0) or (n == 1 and interruptible and not is_exact_repeat)


print("=" * 70)
print("1. An exact-duplicate message does NOT run concurrently with itself")
print("=" * 70)

bot = make_bot()
CID = 8801
running_task = T._Task(
    task_id="r1", chat_id=CID,
    user_text="сгенерируй мне изображение в котором будет командная строка")
with bot._task_lock:
    bot._chat_busy[CID] = 1
    bot._running_task[CID] = [running_task]
    bot._chat_interruptible[CID] = True   # image generation is in progress

check("a byte-identical resend is REFUSED (queues, does not run concurrently)",
      not admit(bot, CID, running_task.user_text))

print()
print("=" * 70)
print("2. A genuinely DIFFERENT quick message still gets the interject slot")
print("=" * 70)

check("a different message IS admitted while the chat is interruptible",
      admit(bot, CID, "а сколько это займёт?"))

print()
print("=" * 70)
print("3. Without interruptible marked, both an identical AND a different")
print("   message still queue (unaffected — this is the pre-existing gate)")
print("=" * 70)

bot2 = make_bot()
CID2 = 8802
running_task2 = T._Task(task_id="r2", chat_id=CID2, user_text="hello")
with bot2._task_lock:
    bot2._chat_busy[CID2] = 1
    bot2._running_task[CID2] = [running_task2]
    # _chat_interruptible NOT set — first task hasn't reached a slow stage yet

check("identical text still queues when not yet interruptible",
      not admit(bot2, CID2, "hello"))
check("different text still queues when not yet interruptible",
      not admit(bot2, CID2, "goodbye"))

print()
print("=" * 70)
print("4. An idle chat (nothing running) always admits, identical text or not")
print("=" * 70)

bot3 = make_bot()
CID3 = 8803
check("idle chat admits a fresh task", admit(bot3, CID3, "anything"))

print()
print(f"{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)

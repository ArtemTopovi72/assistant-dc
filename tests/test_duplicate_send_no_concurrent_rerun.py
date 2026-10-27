"""A duplicate send must not run concurrently with the task it duplicates.

A duplicate Telegram message (typically the client retrying a send that looked
stuck) creates a second _Task with BYTE-IDENTICAL user_text for the same chat.
Once the first task reaches a slow/backgroundable stage (image generation marks
the chat interruptible), the admission gate in _consumer_loop must not treat the
exact repeat as a legitimate "quick interject": it still queues and waits. A
genuinely DIFFERENT quick message is unaffected.

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
    """Exercise the real queue admission, then reset this probe's busy count."""
    bot._user_store.put(T._User(chat_id=chat_id, name="User", status="approved"))
    bot._running = True
    before = bot._chat_busy.get(chat_id, 0)
    candidate = T._Task(task_id="candidate", chat_id=chat_id, user_text=candidate_text)
    bot._backend.push(candidate)
    claimed = bot._claim_task()
    if claimed is None:
        assert bot._backend.pop(timeout=0) is candidate, "refused work was lost"
    if before:
        bot._chat_busy[chat_id] = before
    else:
        bot._chat_busy.pop(chat_id, None)
    return claimed is candidate


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

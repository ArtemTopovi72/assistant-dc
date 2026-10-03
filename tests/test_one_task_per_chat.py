"""One chat must never be answered twice at the same time.

Seen in production after the bot was given several worker threads: the user got
the SAME reply twice in the same second ("Я задумался и не выдал ответ…" ×2),
plus a stack of orphaned "⛔ Отменить запрос" buttons pointing at requests that
had already finished.

Cause: the round-robin backend hands out one task per chat per rotation, but with
several consumers it will hand the SAME chat's next task to a second worker
immediately. `_running_task` is bookkeeping, not a gate.

This drives the REAL consumer threads against the real backend.

Run: venv/Scripts/python.exe tests/test_one_task_per_chat.py
"""
import os, sys, tempfile, threading, time, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

_DATA = tempfile.mkdtemp(prefix="tg_onechat_")
import tg_bot as T
T.redirect_data_dir(_DATA)

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print(f"PASS  {name}")
    else:
        BAD += 1; print(f"FAIL  {name}   {extra}")


CID_A, CID_B = 999961, 999962


def make_bot(hold=0.35):
    bot = T.TelegramBot("123:TEST", lambda: object(), lambda: object(),
                        lambda: {}, silent_mode=True)
    bot._backend = T.InMemoryBackend()
    bot._api_post = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    bot.live = {}            # chat_id -> tasks running right now
    bot.peak = {}            # chat_id -> most ever running at once
    bot.order = []           # (chat_id, text) in execution order
    bot.lock = threading.Lock()

    def fake_execute(task):
        with bot.lock:
            n = bot.live.get(task.chat_id, 0) + 1
            bot.live[task.chat_id] = n
            bot.peak[task.chat_id] = max(bot.peak.get(task.chat_id, 0), n)
            bot.order.append((task.chat_id, task.user_text))
        time.sleep(hold)
        with bot.lock:
            bot.live[task.chat_id] -= 1

    bot._execute_task = fake_execute
    return bot


def run_consumers(bot, n, seconds):
    bot._running = True
    threads = [threading.Thread(target=bot._consumer_loop, daemon=True,
                               name=f"c{i}") for i in range(n)]
    for t in threads:
        t.start()
    time.sleep(seconds)
    bot._running = False
    for t in threads:
        t.join(timeout=8)


# ══════════════════════════════════════════ 1. the bug: same chat, two workers
print("\n" + "=" * 70)
print("1. TWO TASKS FROM ONE CHAT MUST NOT RUN AT THE SAME TIME")
print("=" * 70)

bot = make_bot()
for i in range(4):
    bot._backend.push(T._Task(task_id=f"a{i}", chat_id=CID_A, user_text=f"msg{i}"))
run_consumers(bot, n=3, seconds=3.0)

check("every queued task ran", len(bot.order) == 4, bot.order)
check("never two at once for the same chat", bot.peak.get(CID_A, 0) == 1,
      f"peak={bot.peak.get(CID_A)} — the chat was answered in parallel")
check("and the user's ORDER was preserved",
      [t for _, t in bot.order] == ["msg0", "msg1", "msg2", "msg3"],
      [t for _, t in bot.order])

# ══════════════════════════════ 2. different chats DO still run concurrently
print("\n" + "=" * 70)
print("2. DIFFERENT CHATS STILL OVERLAP — THAT IS THE POINT OF THE QUEUE")
print("=" * 70)

bot2 = make_bot(hold=0.5)
for cid in (CID_A, CID_B):
    for i in range(2):
        bot2._backend.push(T._Task(task_id=f"{cid}-{i}", chat_id=cid,
                                   user_text=f"m{i}"))
t0 = time.time()
run_consumers(bot2, n=3, seconds=2.6)
elapsed = time.time() - t0

check("all four ran", len(bot2.order) == 4, bot2.order)
check("neither chat doubled up",
      bot2.peak.get(CID_A) == 1 and bot2.peak.get(CID_B) == 1, bot2.peak)
# 4 tasks x 0.5s = 2.0s serial; two chats in parallel should finish near 1.0s.
_first_two = {c for c, _ in bot2.order[:2]}
check("the first two to start were from DIFFERENT chats (real overlap)",
      _first_two == {CID_A, CID_B}, bot2.order[:2])

# ══════════════════════════════════════════════ 3. the gate is released
print("\n" + "=" * 70)
print("3. THE GATE IS RELEASED EVEN WHEN A TASK BLOWS UP")
print("=" * 70)

bot3 = make_bot()
boom = {"n": 0}


def exploding(task):
    boom["n"] += 1
    raise RuntimeError("task exploded")


bot3._execute_task = exploding
for i in range(2):
    bot3._backend.push(T._Task(task_id=f"x{i}", chat_id=CID_A, user_text="x"))
run_consumers(bot3, n=2, seconds=2.0)
check("a crashing task does not wedge the chat forever", boom["n"] == 2, boom["n"])
check("the busy set is empty afterwards", not bot3._chat_busy, bot3._chat_busy)

# ══════════════════════════════════════════════ 4. requeue keeps order
print("\n" + "=" * 70)
print("4. REQUEUE PUTS THE TASK BACK AT THE HEAD, NOT THE TAIL")
print("=" * 70)

be = T.InMemoryBackend()
be.push(T._Task(task_id="1", chat_id=CID_A, user_text="first"))
be.push(T._Task(task_id="2", chat_id=CID_A, user_text="second"))
first = be.pop(timeout=1)
be.requeue(first)
again = be.pop(timeout=1)
check("a requeued task comes back before the newer one",
      again.user_text == "first", again.user_text)
check("the queue depth is right", be.depth() == 1, be.depth())

print(f"\n{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)

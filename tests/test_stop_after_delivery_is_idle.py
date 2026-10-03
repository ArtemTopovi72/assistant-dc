"""⛔ Stop pressed after the reply landed says "nothing to stop".

Live 2026-09-13 (mega journey re-run, step 78): Stop one second after the
lighthouse picture arrived replied "Останавливаю…" — the task was still in
_running_task because post-delivery history compaction was running. To the
user nothing was in flight; the honest reply is stop_idle. A task whose
reply has NOT landed yet still gets stopped as before.
"""
import os, sys, tempfile, threading, inspect
os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
import tg_bot as T
T.redirect_data_dir(tempfile.mkdtemp(prefix="tgtest_stopidle_"))

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

CID = 999911
def make_bot():
    bot = T.TelegramBot("123:TEST", lambda: None, lambda: None, lambda: {}, silent_mode=True)
    bot.sent = []
    bot._send_text = lambda cid, text, **kw: bot.sent.append((cid, text, kw)) or 1
    bot._api_post = lambda *a, **k: {"ok": True, "result": {"message_id": 1}}
    bot._api_get = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    bot._on_stage = lambda *a, **k: None
    bot._user_store.put(T._User(chat_id=CID, name="T", status="approved"))
    return bot

def press_stop(bot, task, delivered):
    task.delivered = delivered
    ev = threading.Event()
    with bot._task_lock:
        bot._running_task[CID] = [task]
        bot._task_cancels[task.task_id] = ev
    sess = bot._get_session(CID)
    bot._stop_and_report(CID, sess, "ru")
    return ev

bot = make_bot()
t = T._Task(task_id="done1", chat_id=CID, user_text="нарисуй маяк")
ev = press_stop(bot, t, delivered=True)
check("Stop after delivery answers 'nothing to stop'",
      any(txt == T._t("stop_idle", "ru") for _, txt, _ in bot.sent), bot.sent)
check("...and does not claim to be stopping",
      not any(txt == T._t("stopping", "ru", extra="") for _, txt, _ in bot.sent), bot.sent)
check("...and leaves the housekeeping alone (cancel event untouched)", not ev.is_set())

bot = make_bot()
t = T._Task(task_id="run1", chat_id=CID, user_text="нарисуй маяк")
ev = press_stop(bot, t, delivered=False)
check("Stop while the reply is still in flight stops it",
      any(txt == T._t("stopping", "ru", extra="") for _, txt, _ in bot.sent), bot.sent)
check("...signalling the task's cancel event", ev.is_set())

check("_Task carries the delivered flag (default False)",
      T._Task(task_id="x", chat_id=1, user_text="y").delivered is False)
check("from_dict tolerates journals written before the flag existed",
      T._Task.from_dict({"task_id": "x", "chat_id": 1, "user_text": "y"}).delivered is False)
import tg_tasks
src = inspect.getsource(tg_tasks)
check("the flag is set right after the ✅ Done stage, before compaction",
      'self._on_stage(chat_id, "✅ Done", True)\n        task.delivered = True' in src)

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)

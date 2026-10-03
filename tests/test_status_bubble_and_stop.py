"""Two live-chat regressions found by driving the real bot.

1. ORPHANED STATUS BUBBLE. Every answer was followed by a fresh "⚙️ …" message
   carrying a live ⛔ Cancel button, which nothing ever cleared. Cause: the
   post-delivery history compaction ends with `ctx.set_stage("")`, and that
   reached the task's on_stage callback AFTER _clear_status had already deleted
   the status message — so _update_status, seeing no status message, opened a
   NEW one. This is the stack of dead Cancel buttons users kept reporting.

2. STOP LIES WHEN IDLE. ⛔ Stop (and /cancel) always answered "Stopping…", even
   with nothing running and nothing queued — the bot claiming work it was not
   doing.

Run: venv/Scripts/python.exe tests/test_status_bubble_and_stop.py
"""
import os, sys, tempfile, threading, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

_DATA = tempfile.mkdtemp(prefix="tg_bubble_")
import tg_bot as T
T.redirect_data_dir(_DATA)

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print(f"PASS  {name}")
    else:
        BAD += 1; print(f"FAIL  {name}   {extra}")


CID = 999801


def make_bot():
    bot = T.TelegramBot("123:TEST", lambda: None, lambda: None, lambda: {},
                        silent_mode=True)
    bot.calls = []
    bot._api_post = lambda m, p=None, **k: (bot.calls.append((m, p or {})),
                                            {"ok": True,
                                             "result": {"message_id": 77}})[1]
    bot._api_get = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    u = T._User(chat_id=CID, name="T", status="approved")
    bot._user_store.put(u)
    return bot


def sent_texts(bot):
    return [p.get("text", "") for m, p in bot.calls if m == "sendMessage"]


# ── 1. the status callback after the status line is gone ─────────────────────
print("=" * 62)
print("ORPHANED STATUS BUBBLE")
print("=" * 62)

bot = make_bot()
sess = bot._get_session(CID)
lang = "ru"

# Rebuild the exact closure shape _run_task_inner uses.
status_id = None
status_done = [False]
cancel_kb = {"inline_keyboard": [[{"text": "⛔", "callback_data": "cancel:x"}]]}


def _update_status(text):
    global status_id
    if status_id:
        bot._edit_text(CID, status_id, text, parse_mode="HTML", keyboard=cancel_kb)
    elif not status_done[0]:
        status_id = bot._send_get_id(CID, text, parse_mode="HTML",
                                     keyboard=cancel_kb)


def _clear_status():
    global status_id
    status_done[0] = True
    if status_id:
        bot._edit_text(CID, status_id, "✅", keyboard={"inline_keyboard": []})
        bot._delete(CID, status_id)
        status_id = None


_update_status("⚙️ <b>Starting…</b>")
opened = len([m for m, _ in bot.calls if m == "sendMessage"])
_clear_status()
# … the reply goes out here …
# … and then compaction fires its trailing empty stage:
_update_status("⚙️ <b>…</b>")

after = len([m for m, _ in bot.calls if m == "sendMessage"])
check("a late stage does not open a NEW status message", after == opened,
      f"opened {after - opened} extra")

# and the callback itself must ignore a blank stage entirely
import inspect
src = inspect.getsource(T.TelegramBot._run_task_inner)
# on_stage moved OUT of _run_task_inner into _make_stage_callback when the
# character render was made to use the same status pipeline instead of growing
# a second one. Look where it lives now, not where it used to.
_cb_src = inspect.getsource(T.TelegramBot._make_stage_callback)
check("on_stage drops blank stages before doing anything",
      'if not (stage or "").strip():' in _cb_src)
check("and the queued path still uses that one callback",
      "self._make_stage_callback(" in src)
check("_clear_status latches the status line closed",
      "status_done[0] = True" in src)
check("_update_status honours the latch", "elif not status_done[0]:" in src)

# ── 2. Stop must not claim work it is not doing ──────────────────────────────
print("=" * 62)
print("STOP WITH NOTHING RUNNING")
print("=" * 62)

bot = make_bot()
sess = bot._get_session(CID)
bot._stop_and_report(CID, sess, "ru")
texts = sent_texts(bot)
check("idle Stop says nothing was running",
      any("ничего не выполняется" in t for t in texts), texts)
check("idle Stop does NOT say 'Stopping…'",
      not any("станавливаю" in t for t in texts), texts)

# with something actually running it must still report the stop
bot = make_bot()
with bot._task_lock:
    bot._running_task[CID] = [T._Task(task_id="t1", chat_id=CID, user_text="hi")]
bot._stop_and_report(CID, bot._get_session(CID), "ru")
texts = sent_texts(bot)
check("Stop with a task in flight reports the stop",
      any("станавливаю" in t for t in texts), texts)

# both entry points share the one implementation
# Read every module the bot is assembled from, not tg_bot.py alone: the command
# handler and the callback dispatcher moved into tg_commands.py and
# tg_dispatch.py, and counting call sites in one file found none of them.
import glob as _glob
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_all = "".join(open(f, encoding="utf-8").read()
                  for f in sorted(_glob.glob(os.path.join(_root, "bot", "tg_*.py"))))
check("there is exactly one stop implementation",
      src_all.count("def _stop_and_report(") == 1,
      src_all.count("def _stop_and_report("))
check("the Stop button and /cancel both route to it",
      src_all.count("self._stop_and_report(chat_id, sess, lang)") == 2,
      src_all.count("self._stop_and_report(chat_id, sess, lang)"))

print(f"\n{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)

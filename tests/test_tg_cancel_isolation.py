"""The Telegram bot must not share a cancel token with the desktop chat.

`ctx.cancel_event` is a single global flag. Tabs that run long jobs were given a
private view of the context (`gui._ScopedCtx`) precisely because sharing it makes
Stop buttons reach across features — but the bot was left on the raw ctx:

    get_ctx = lambda: getattr(self.host, "ctx", None)

Three concrete collisions follow from that, and all three are silent:

  1. A Telegram user presses ⛔ Stop -> tg_bot._request_stop sets ctx.cancel_event
     -> the DESKTOP user's in-flight chat turn dies. tg_bot's own comment says
     "ctx is shared, so an unconditional cancel_event would kill whoever happens
     to be running" — it guards inside the bot, not across bot/GUI.
  2. The desktop user presses Stop -> AssistantWindow._cancel_current sets the
     global flag (and reaches transfer_tab/storyboard_tab explicitly, but never
     the bot) -> a remote Telegram user's task aborts mid-answer.
  3. Worst direction: tg_bot._run_task_inner CLEARS ctx.cancel_event when a task
     starts, so a Telegram message arriving one second after the desktop user
     pressed Stop *un-cancels* the desktop turn.

Run: QT_QPA_PLATFORM=offscreen venv/Scripts/python.exe tests/test_tg_cancel_isolation.py
"""
import os, sys, threading, types, tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("USE_GUI", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

from PyQt5.QtWidgets import QApplication
import gui
import tg_bot as _T

# BEFORE any tab exists: TelegramTab._start() PERSISTS the token it is holding,
# so driving a real tab with a fixture token overwrites the user's live
# BotFather token in QSettings. Redirect the store first, always.
_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_cancel_")
gui.redirect_settings(_DATA_DIR)
_T.redirect_data_dir(_DATA_DIR)

_app = QApplication.instance() or QApplication(sys.argv)

OK = BAD = 0
def check(name, cond, detail=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {detail}")


class _Ctx:
    """Just enough of the assistant context: a cancel flag and a writable attr."""
    def __init__(self):
        self.cancel_event = threading.Event()
        self.throttle = 0


class _Host:
    def __init__(self):
        self.ctx = _Ctx()
        self.graph = types.SimpleNamespace()
        self.base_state = {}


def _tab():
    host = _Host()
    tab = gui.TelegramTab(host)
    return host, tab


print("=" * 70)
print("THE BOT GETS ITS OWN CANCEL TOKEN")
print("=" * 70)

host, tab = _tab()
bot_ctx = tab._bot_ctx()

check("the bot's ctx is not the raw host ctx", bot_ctx is not host.ctx)
check("the bot's cancel_event is a different object",
      bot_ctx.cancel_event is not host.ctx.cancel_event)
check("the view is stable across calls (identity comparisons stay valid)",
      tab._bot_ctx() is bot_ctx)
check("the same cancel token is handed out every time",
      tab._bot_ctx().cancel_event is bot_ctx.cancel_event)

# Everything except cancellation still delegates to the real context.
host.ctx.graph_marker = "real"
check("reads delegate to the real ctx", getattr(bot_ctx, "graph_marker") == "real")
bot_ctx.throttle = 7
check("writes land on the real ctx", host.ctx.throttle == 7)


print()
print("=" * 70)
print("THE THREE COLLISIONS")
print("=" * 70)

# 1. Telegram Stop must not kill the desktop turn.
host, tab = _tab()
bot_ctx = tab._bot_ctx()
host.ctx.cancel_event.clear()
bot_ctx.cancel_event.set()                      # what tg_bot._request_stop does
check("a Telegram Stop leaves the desktop turn running",
      not host.ctx.cancel_event.is_set())

# 2. A desktop Stop must not abort the bot's task.
host, tab = _tab()
bot_ctx = tab._bot_ctx()
bot_ctx.cancel_event.clear()
host.ctx.cancel_event.set()                     # what _cancel_current does
check("a desktop Stop leaves the bot's task running",
      not bot_ctx.cancel_event.is_set())

# 3. A bot task starting must not un-cancel a stopped desktop turn.
host, tab = _tab()
bot_ctx = tab._bot_ctx()
host.ctx.cancel_event.set()                     # desktop user pressed Stop
bot_ctx.cancel_event.clear()                    # what _run_task_inner does at start
check("a starting bot task does not un-cancel the desktop turn",
      host.ctx.cancel_event.is_set())


print()
print("=" * 70)
print("THE BOT IS ACTUALLY WIRED TO IT")
print("=" * 70)

# The constructor must hand the bot the scoped view, not the raw ctx. Capture the
# get_ctx callable the tab passes to TelegramBot rather than trusting the source.
host, tab = _tab()
captured = {}

class _FakeBot:
    def __init__(self, **kw):
        captured.update(kw)
    def start(self): captured["started"] = True
    def get_users(self): return []
    def get_queue_stats(self): return {}

import tg_bot as TB
_real = TB.TelegramBot
TB.TelegramBot = _FakeBot
try:
    tab.token_in.setText("123:ABC")
    tab._start()
finally:
    TB.TelegramBot = _real

check("the bot was constructed", "get_ctx" in captured, str(sorted(captured)))
if "get_ctx" in captured:
    handed = captured["get_ctx"]()
    check("the bot is handed the scoped view, not the raw ctx",
          handed is not host.ctx and handed is tab._bot_ctx())
    check("the bot's Stop cannot reach the desktop cancel flag",
          handed.cancel_event is not host.ctx.cancel_event)

# No ctx at all (bot started before the assistant finished loading) must not raise.
host2, tab2 = _tab()
host2.ctx = None
check("a missing ctx yields None rather than raising", tab2._bot_ctx() is None)

# And once the host ctx appears, the view must bind to it.
host2.ctx = _Ctx()
v = tab2._bot_ctx()
check("the view binds once the ctx appears", v is not None and v is not host2.ctx)

# If the host swaps its ctx (profile switch), the view must follow it.
host3, tab3 = _tab()
first = tab3._bot_ctx()
host3.ctx = _Ctx()
second = tab3._bot_ctx()
check("a swapped host ctx is picked up", second is not first)
check("but the bot keeps its own cancel token across the swap",
      second.cancel_event is first.cancel_event
      and second.cancel_event is not host3.ctx.cancel_event)

print()
print(f"{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)

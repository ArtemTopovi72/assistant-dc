"""A bot that did not start must not look like one that did.

Reconstructed from a real activity log: 26 consecutive "Bot stopped" entries with
no matching "Bot started", and every message the user sent went unanswered.

  · TelegramBot.start() returned None on failure and wrote NOTHING to the
    activity log — the one place an operator looks recorded nothing at all;
  · TelegramTab._start() flipped the panel into the running state regardless, so
    Start went grey and Stop went live for a bot that was never up;
  · stop() then logged "Bot stopped" — and broadcast "🔴 Going offline" to every
    subscriber — for a bot that had never been online;
  · and the only diagnosis offered was "Bad token or no network", which are two
    completely different problems and only one of them is the operator's fault.

Run: QT_QPA_PLATFORM=offscreen venv/Scripts/python.exe tests/test_tg_startup_honesty.py
"""
import sys, os, types, tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("USE_GUI", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T

_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_start_")
T.redirect_data_dir(_DATA_DIR)

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")


def make_bot(api_get_result):
    bot = T.TelegramBot("1:T", lambda: None, lambda: None, lambda: {},
                        silent_mode=True)
    bot.status, bot.logged, bot.sent = [], [], []
    bot._on_status = bot.status.append
    bot._activity.log = lambda cid, ev, text, uname="": bot.logged.append((ev, text))
    bot._api_get = lambda method, params=None: api_get_result
    bot._api_post = lambda *a, **k: {}
    bot._send_text = lambda cid, text, **k: (bot.sent.append(text), 1)[1]
    return bot


UNREACHABLE = {}
BAD_TOKEN = {"ok": False, "error_code": 401,
             "description": "Unauthorized: invalid token specified"}
RATE_LIMITED = {"ok": False, "error_code": 429, "description": "Too Many Requests"}
GOOD = {"ok": True, "result": {"username": "TestBot", "id": 1}}


print("=" * 70)
print("A FAILED START SAYS SO")
print("=" * 70)

for label, payload in (("an unreachable API", UNREACHABLE),
                       ("a rejected token", BAD_TOKEN),
                       ("a refusal from Telegram", RATE_LIMITED)):
    bot = make_bot(payload)
    got = bot.start()
    check(f"start() returns False on {label}", got is False, repr(got))
    check(f"the bot does not consider itself running after {label}",
          not bot._running)
    check(f"the failure reaches the activity log for {label}",
          any(ev == "error" and "[start] failed" in text
              for ev, text in bot.logged), str(bot.logged))
    check(f"and the operator is shown a reason for {label}",
          bot.status and bot.status[-1].startswith("❌"), str(bot.status))
    check(f"no poll thread is left behind after {label}",
          bot._poll_thread is None or not bot._poll_thread.is_alive())

# The two causes need different fixes, so they must not share one message.
b_net = make_bot(UNREACHABLE); b_net.start()
b_tok = make_bot(BAD_TOKEN);   b_tok.start()
check("an unreachable API and a bad token give different messages",
      b_net.status[-1] != b_tok.status[-1],
      f"{b_net.status[-1]!r} == {b_tok.status[-1]!r}")
# "Bad token or no network" mentions both, which is exactly why it was useless —
# so each message must name its own cause and NOT the other one.
check("the network message names the network",
      any(w in b_net.status[-1].lower() for w in ("unreachable", "network", "proxy")),
      repr(b_net.status[-1]))
check("and does not blame the token",
      "token" not in b_net.status[-1].lower(), repr(b_net.status[-1]))
check("the token message names the token",
      "token" in b_tok.status[-1].lower(), repr(b_tok.status[-1]))
check("and does not blame the network",
      "network" not in b_tok.status[-1].lower(), repr(b_tok.status[-1]))
check("the token message says where to get a new one",
      "botfather" in b_tok.status[-1].lower(), repr(b_tok.status[-1]))
b_429 = make_bot(RATE_LIMITED); b_429.start()
check("an unexpected refusal carries Telegram's own reason",
      "429" in b_429.status[-1], repr(b_429.status[-1]))


print()
print("=" * 70)
print("STOPPING SOMETHING THAT NEVER STARTED IS NOT AN EVENT")
print("=" * 70)

bot = make_bot(UNREACHABLE)
bot.start()
bot.logged.clear(); bot.sent.clear()
bot.stop()
check("a never-started bot logs no 'Bot stopped'",
      not any("Bot stopped" in text for _ev, text in bot.logged), str(bot.logged))
check("and it still reports a stopped status to the UI",
      bot.status and "Stopped" in bot.status[-1], str(bot.status[-1:]))

# The shutdown broadcast is the visible half: users were told a bot they never
# saw come online was going offline.
noisy = T.TelegramBot("1:T", lambda: None, lambda: None, lambda: {},
                      silent_mode=False)
noisy.sent = []
noisy._send_text = lambda cid, text, **k: (noisy.sent.append(text), 1)[1]
noisy._activity.log = lambda *a, **k: None
noisy._on_status = lambda *a, **k: None
noisy._api_get = lambda *a, **k: UNREACHABLE
noisy._api_post = lambda *a, **k: {}
noisy._user_store.put(T._User(chat_id=999801, name="U", status="approved",
                              subscriptions=["startup"]))
noisy.start()
noisy.sent.clear()
noisy.stop()
check("a never-started bot does not announce going offline",
      noisy.sent == [], str(noisy.sent))

# Repeated stops must stay quiet — this is what produced 26 log lines.
quiet = make_bot(UNREACHABLE)
quiet.start(); quiet.logged.clear()
for _ in range(10):
    quiet.stop()
check("ten Stop presses on a dead bot log nothing",
      not any("Bot stopped" in t for _e, t in quiet.logged), str(quiet.logged))


print()
print("=" * 70)
print("A SUCCESSFUL START IS STILL FULLY REPORTED")
print("=" * 70)

good = make_bot(GOOD)
good._register_commands = lambda: None
good._recover_inflight = lambda: None
good._broadcast_startup = lambda: None
started = good.start()
try:
    check("start() returns True", started is True, repr(started))
    check("the bot considers itself running", good._running)
    check("the start is logged with the bot's name",
          any("Bot started as @TestBot" in t for _e, t in good.logged),
          str(good.logged))
    check("the operator sees a success status",
          any("@TestBot" in s for s in good.status), str(good.status))
    check("starting twice is a no-op that still reports success",
          good.start() is True)
    good.logged.clear()
    good.stop()
    check("stopping a running bot IS logged",
          any("Bot stopped" in t for _e, t in good.logged), str(good.logged))
finally:
    good._running = False


print()
print("=" * 70)
print("THE PANEL ONLY CLAIMS 'RUNNING' WHEN IT IS")
print("=" * 70)

from PyQt5.QtWidgets import QApplication
import gui
# _start() persists the token; without this the fixture token below lands in the
# user's real QSettings and destroys their live BotFather token.
gui.redirect_settings(_DATA_DIR)
_app = QApplication.instance() or QApplication(sys.argv)


class _Host:
    def __init__(self):
        self.ctx = None
        self.graph = None
        self.base_state = {}


def drive_panel(start_result):
    tab = gui.TelegramTab(_Host())
    tab.token_in.setText("123:ABC")
    # Start from the state a retry actually begins in: the previous attempt left
    # the panel locked. A fresh tab has these enabled already, so asserting on it
    # would pass whether or not the failure path re-enables anything.
    tab.start_btn.setEnabled(False)
    tab.stop_btn.setEnabled(True)
    tab.token_in.setEnabled(False)
    tab.admin_ids_in.setEnabled(False)

    class _FakeBot:
        def __init__(self, **kw): pass
        def start(self): return start_result
        def get_users(self): return []
        def get_queue_stats(self): return {}
        def stop(self): pass

    real = T.TelegramBot
    T.TelegramBot = _FakeBot
    try:
        tab._start()
    finally:
        T.TelegramBot = real
    return tab


tab = drive_panel(False)
check("a failed start leaves Start pressable", tab.start_btn.isEnabled())
check("a failed start leaves Stop disabled", not tab.stop_btn.isEnabled())
check("the token stays editable so it can be corrected",
      tab.token_in.isEnabled())
check("the admin field stays editable", tab.admin_ids_in.isEnabled())
check("no dead bot object is kept around", tab._bot is None, repr(tab._bot))

tab = drive_panel(True)
check("a successful start disables Start", not tab.start_btn.isEnabled())
check("a successful start enables Stop", tab.stop_btn.isEnabled())
check("and locks the token field", not tab.token_in.isEnabled())
check("and keeps the bot object", tab._bot is not None)

print()
print(f"{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)

"""A test run must never touch the user's real Telegram token.

What actually happened: two suites built a real `gui.TelegramTab`, typed the
fixture token "123:ABC" into it and called `_start()`. `_start()` PERSISTS the
token (plus admin ids, TTS, autostart, silent mode) to QSettings before it ever
constructs the bot — and the tab was reading/writing the LIVE scope
("AssistantApp"/"TelegramBot", i.e. HKCU\\Software\\AssistantApp\\TelegramBot).
So running the suite silently replaced the user's real BotFather token with
"123:ABC", and the panel came up empty afterwards.

Same bug class as tg_bot.redirect_data_dir (fixture users landing in the live
tg_users.db) — a test writing into production state.

Two guards here:
  1. redirect_settings() genuinely diverts writes, and the live scope is
     byte-identical before and after a full drive of the panel;
  2. a STATIC scan: any test that constructs a TelegramTab must call
     redirect_settings first. Guard 1 alone cannot stop a NEW suite from
     reintroducing the bug.

Run: QT_QPA_PLATFORM=offscreen venv/Scripts/python.exe tests/test_gui_settings_isolation.py
"""
import os, sys, ast, types, tempfile, threading, hashlib
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("USE_GUI", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QApplication
import gui
import tg_bot as T

_TESTS_DIR = Path(__file__).resolve().parent

OK = BAD = 0
def check(name, cond, detail=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {detail}")


def _live_fingerprint():
    """Hash of the real store's telegram keys. Never print the values."""
    s = QSettings(*gui._TG_SETTINGS_SCOPE)
    blob = "|".join(f"{k}={s.value(k)!r}" for k in sorted(s.allKeys())
                    if k.startswith("telegram/"))
    return hashlib.sha256(blob.encode("utf-8", "replace")).hexdigest()


print("=" * 70)
print("THE REAL STORE SURVIVES A FULL PANEL DRIVE")
print("=" * 70)

before = _live_fingerprint()

_DIR = tempfile.mkdtemp(prefix="tgtest_settings_")
gui.redirect_settings(_DIR)
T.redirect_data_dir(_DIR)
_app = QApplication.instance() or QApplication(sys.argv)


class _Ctx:
    def __init__(self):
        self.cancel_event = threading.Event()

class _Host:
    def __init__(self):
        self.ctx = _Ctx(); self.graph = types.SimpleNamespace(); self.base_state = {}

class _FakeBot:
    def __init__(self, **kw): pass
    def start(self): return True
    def stop(self): pass
    def get_users(self): return []
    def get_queue_stats(self): return {}


_SENTINEL = "999999:TEST-ONLY-NOT-A-REAL-TOKEN"
tab = gui.TelegramTab(_Host())
tab.token_in.setText(_SENTINEL)
tab.admin_ids_in.setText("424242")
_real = T.TelegramBot
T.TelegramBot = _FakeBot
try:
    tab._start()
finally:
    T.TelegramBot = _real

check("the live store is byte-identical after a redirected _start()",
      _live_fingerprint() == before)

# ...and the write really did land somewhere, or guard 1 would pass vacuously
# for a _start() that simply stopped persisting anything.
tab._settings.sync()          # Qt flushes lazily; force it so the file is real
ini = Path(_DIR) / "telegram_test.ini"
check("the redirected store exists on disk", ini.exists(), str(ini))
_written = QSettings(str(ini), QSettings.IniFormat)
check("the token was persisted to the redirected store",
      _written.value(gui._TG_TOKEN_KEY, "") == _SENTINEL,
      repr(_written.value(gui._TG_TOKEN_KEY, "")))
check("the admin ids went there too",
      _written.value(gui._TG_ADMINS_KEY, "") == "424242")

# A fresh tab must READ the redirect as well — a redirect that only covered
# writes would still leak the user's real token into a test process.
tab2 = gui.TelegramTab(_Host())
check("a fresh tab reads back the redirected token",
      tab2.token_in.text() == _SENTINEL, tab2.token_in.text())

# And the redirect must be reversible, or the desktop app would come up blank
# in any process that imported a test module.
# Under a test the live store stays unreachable even after redirect(None):
# a suite that forgot the redirect started the REAL bot (2026-09-28 23:15).
gui.redirect_settings(None)
check("redirect_settings(None) under a test never reaches the live scope",
      gui._tg_settings().fileName() != QSettings(*gui._TG_SETTINGS_SCOPE).fileName())
gui.redirect_settings(_DIR)


# Mutation guard: "the live store is unchanged" would also pass if the
# fingerprint were blind, or if _start() had quietly stopped persisting. Point
# the *scope* at a scratch org, turn the redirect OFF, and confirm the same
# drive DOES move the fingerprint. This is the old, broken behaviour reproduced
# in a sandbox — the real registry is never involved.
_real_scope = gui._TG_SETTINGS_SCOPE
# The scope lives in gui_telegram_tab now, and _tg_settings() reads it THERE.
# Assigning gui._TG_SETTINGS_SCOPE would rebind only this module's copy and
# leave the real ("AssistantApp", "TelegramBot") scope live — i.e. the very
# token-clobbering this suite exists to prevent.
import gui_telegram_tab as _TGT
_TGT._TG_SETTINGS_SCOPE = ("AssistantAppTESTSCOPE", "TelegramBotTEST")
gui._TG_SETTINGS_SCOPE = ("AssistantAppTESTSCOPE", "TelegramBotTEST")
gui.redirect_settings(None)
try:
    _sandbox_before = _live_fingerprint()
    t3 = gui.TelegramTab(_Host())
    t3.token_in.setText("111111:SANDBOX")
    T.TelegramBot = _FakeBot
    try:
        t3._start()
    finally:
        T.TelegramBot = _real
    t3._settings.sync()
    check("without the redirect the same drive DOES rewrite the store "
          "(so the guard above is real)",
          _live_fingerprint() != _sandbox_before)
    QSettings(*gui._TG_SETTINGS_SCOPE).clear()
finally:
    _TGT._TG_SETTINGS_SCOPE = _real_scope
    gui._TG_SETTINGS_SCOPE = _real_scope
    gui.redirect_settings(_DIR)


print()
print("=" * 70)
print("NO SUITE MAY BUILD A TelegramTab WITHOUT REDIRECTING FIRST")
print("=" * 70)

def _audit(path: Path):
    """Return (builds_tab, redirect_line, tab_line) from the source AST."""
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    tab_line = redirect_line = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name == "TelegramTab" and tab_line is None:
            tab_line = node.lineno
        elif name == "redirect_settings" and redirect_line is None:
            redirect_line = node.lineno
    return tab_line, redirect_line

offenders = []
audited = 0
for f in sorted(_TESTS_DIR.glob("test_*.py")):
    tab_line, redirect_line = _audit(f)
    if tab_line is None:
        continue
    audited += 1
    if redirect_line is None or redirect_line > tab_line:
        offenders.append(f"{f.name}: TelegramTab@{tab_line} redirect@{redirect_line}")

check("at least one suite actually builds a TelegramTab (scan is live)",
      audited >= 2, f"audited={audited}")
check("every TelegramTab suite redirects settings BEFORE building the tab",
      not offenders, "; ".join(offenders))

# The tab must go through the redirectable factory. A future edit that inlines
# QSettings("AssistantApp", ...) back into __init__ would defeat every check
# above, and the AST scan would keep passing.
# TelegramTab now lives in gui_telegram_tab.py, so read ITS source. Reading
# gui.py here made the split below raise IndexError rather than fail a check.
src = Path(_TGT.__file__).read_text(encoding="utf-8", errors="replace")
init_src = src.split("class TelegramTab", 1)[1].split("\n    def _build_ui", 1)[0]
check("TelegramTab.__init__ uses the _tg_settings() factory",
      "_tg_settings()" in init_src)
check("TelegramTab.__init__ does not hard-code the live scope",
      "QSettings(\"AssistantApp\"" not in init_src)


print()
print(f"{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)

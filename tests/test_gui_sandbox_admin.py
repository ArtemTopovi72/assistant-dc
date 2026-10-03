"""The sandbox admin controls in the Telegram tab, driven headless.

Builds the REAL panel and clicks the REAL slots. Qt swallows nothing here on
purpose: a slot that raises under offscreen looks like a button that did
nothing, and this project has lost an afternoon to exactly that before, so
every handler is called directly and its effect asserted on the store.

Two protections have to be in place BEFORE the panel exists:
  * redirect_settings — the panel persists the bot token, so a test that builds
    it against the live QSettings destroys the user's real BotFather token;
  * redirect_data_dir — otherwise fixture users land in the live tg_users.db.

Run: QT_QPA_PLATFORM=offscreen venv/Scripts/python.exe tests/test_gui_sandbox_admin.py
"""
import os, sys, tempfile
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("USE_GUI", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import gui_telegram_tab as TT
TT.redirect_settings(tempfile.mkdtemp(prefix="guitest_sbx_settings_"))

import tg_bot as T
T.redirect_data_dir(tempfile.mkdtemp(prefix="guitest_sbx_data_"))

import code_sandbox as CS
from pathlib import Path
CS.SANDBOX_BASE = Path(tempfile.mkdtemp(prefix="guitest_sbx_root_"))

import sandbox_access as A
from PyQt5.QtWidgets import QApplication, QMessageBox

_app = QApplication.instance() or QApplication(sys.argv)

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")


class _Host:
    """The panel's host object — it only ever reads a few attributes."""
    def __init__(self):
        self.ctx = None
    def __getattr__(self, name):
        return None


def build():
    tab = TT.TelegramTab(_Host())
    store = T._UserStore(T._USERS_DB)
    for cid in (11, 22):
        store.put(T._User(chat_id=cid, name=f"u{cid}", status="approved"))
    tab._bot = type("B", (), {"_user_store": store})()
    return tab, store


tab, store = build()
check("the panel builds with the sandbox controls",
      hasattr(tab, "sbx_combo") and hasattr(tab, "sbx_apply_btn"))
check("every level is offered",
      [tab.sbx_combo.itemText(i) for i in range(tab.sbx_combo.count())]
      == ["off", "files", "code", "host"])


# ── nothing selected ─────────────────────────────────────────────────────────
tab._selected_chat_id = lambda: None
tab._apply_sandbox_level()
check("with no row selected it says so, and does not raise",
      "Select a user" in tab.sbx_status_lbl.text(), tab.sbx_status_lbl.text())


# ── grant ────────────────────────────────────────────────────────────────────
tab._selected_chat_id = lambda: 11
tab._refresh_users = lambda: None          # the table needs a live bot
tab.sbx_combo.setCurrentText("files")
tab._apply_sandbox_level()
check("granting writes through to the store",
      A.level_for(store.get(11)) == A.FILES, A.level_for(store.get(11)))
check("the panel reports what it did", "sandbox = files" in tab.sbx_status_lbl.text())

tab.sbx_combo.setCurrentText("code")
tab._apply_sandbox_level()
check("raising the level works", A.may_run_code(store.get(11)))

tab.sbx_combo.setCurrentText("off")
tab._apply_sandbox_level()
check("revoking works", not A.may_use_files(store.get(11)))


# ── host needs a confirmation, and a refusal must not grant ──────────────────
_answers = {"value": QMessageBox.Cancel}
TT.QMessageBox.warning = staticmethod(lambda *a, **k: _answers["value"])

tab.sbx_combo.setCurrentText("host")
tab._apply_sandbox_level()
check("cancelling the host warning grants nothing",
      A.level_for(store.get(11)) == A.OFF, A.level_for(store.get(11)))

_answers["value"] = QMessageBox.Yes
tab._apply_sandbox_level()
check("confirming the host warning grants it",
      A.allow_host_execution(store.get(11)))


# ── one user at a time ───────────────────────────────────────────────────────
check("the other user is untouched",
      A.level_for(store.get(22)) == A.OFF, A.level_for(store.get(22)))


# ── resetting a folder ───────────────────────────────────────────────────────
box = CS.sandbox_for(11)
box.write_text("junk/a.txt", "x")
TT.QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Cancel)
tab._reset_sandbox_selected()
check("cancelling the reset keeps the files", box.list_dir("."))

TT.QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
tab._reset_sandbox_selected()
check("confirming the reset empties the folder", not box.list_dir("."))
check("resetting does not change their access",
      A.allow_host_execution(store.get(11)), "level changed by a reset")


# ── rights are revocable with the bot stopped ────────────────────────────────
tab._bot = None
got = tab._sandbox_user_store()
check("the store is reachable with the bot stopped", got is not None)


print(f"\n{OK}/{OK + BAD} checks passed")
sys.exit(0 if BAD == 0 else 1)

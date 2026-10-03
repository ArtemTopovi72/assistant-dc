"""gui_telegram_tab.py's chat_id-carrying Qt signals must not truncate.

Telegram chat_ids are 64-bit and routinely exceed +-2^31 (e.g. group chat ids,
which are large negative numbers to begin with, and some private chat_ids are
themselves already past 2^31). PyQt's plain `int` signal argument type maps to
a C `int` (32-bit signed), not Python's arbitrary-precision int -- a value
outside that range silently wraps. Live symptom that motivated this test: the
desktop log showed "user -1442377781" for a chat whose real id was
7147556811 -- exactly what 7147556811 wraps to as a signed 32-bit int.

Run: QT_QPA_PLATFORM=offscreen venv/Scripts/python.exe tests/test_gui_telegram_tab.py
"""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

from PyQt5.QtWidgets import QApplication, QWidget, QListWidget, QLabel
_app = QApplication.instance() or QApplication(sys.argv)

import gui_telegram_tab as G

RESULTS = []
import os as _os
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond and _os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))


def _bare_tab() -> "G.TelegramTab":
    """A TelegramTab whose QObject base is properly initialized (required
    before any pyqtSignal can be connected/emitted) but whose __init__ body
    -- which builds the whole widget tree and needs a live bot/host -- never
    runs. Bypassing __new__ alone leaves the QObject base uninitialized and
    signal access raises; QWidget.__init__ fixes that without needing any of
    __init__'s real arguments."""
    tab = G.TelegramTab.__new__(G.TelegramTab)
    QWidget.__init__(tab)
    return tab


# The real chat_id from the live bug report -- a group chat id, large and
# negative in Telegram's own numbering, chosen because it's an actual
# documented failure case, not a synthetic edge value.
_BIG_CHAT_ID = 7147556811
# Its exact 32-bit-signed wraparound, i.e. what the OLD `pyqtSignal(int, ...)`
# declaration produced and what the live screenshot showed.
_WRAPPED = -1442377781


def test_sig_stage_preserves_large_chat_id():
    got = []
    tab = _bare_tab()
    tab._sig_stage.connect(lambda cid, text, done: got.append((cid, text, done)))
    tab._sig_stage.emit(_BIG_CHAT_ID, "Writing a response", False)
    check("sig_stage delivers the exact chat_id, not a 32-bit wraparound",
          got == [(_BIG_CHAT_ID, "Writing a response", False)], detail=repr(got))
    check("...and specifically not the historical wrapped value",
          not got or got[0][0] != _WRAPPED)


def test_sig_message_preserves_large_chat_id():
    got = []
    tab = _bare_tab()
    tab._sig_message.connect(lambda cid, u, b: got.append((cid, u, b)))
    tab._sig_message.emit(_BIG_CHAT_ID, "hi", "hello")
    check("sig_message delivers the exact chat_id",
          got == [(_BIG_CHAT_ID, "hi", "hello")], detail=repr(got))


def test_sig_user_change_preserves_large_chat_id():
    got = []
    tab = _bare_tab()
    tab._sig_user_change.connect(lambda cid, name, status: got.append((cid, name, status)))
    tab._sig_user_change.emit(_BIG_CHAT_ID, "Влад", "approved")
    check("sig_user_change delivers the exact chat_id",
          got == [(_BIG_CHAT_ID, "Влад", "approved")], detail=repr(got))


def test_on_stage_log_line_has_the_real_chat_id():
    # _on_stage is the slot _sig_stage is connected to -- the code path that
    # actually produced the corrupted "user -1442377781" log line in
    # production. Real QListWidget/QLabel here (not stand-ins), since
    # _log_append/_on_stage call real Qt methods (count(), takeItem(),
    # scrollToBottom()) that a minimal fake would have to reimplement anyway.
    tab = _bare_tab()
    tab._active_chats = {}
    tab._LOG_MAX = 500
    tab.status_board = QLabel()
    tab.log = QListWidget()
    tab.host = None
    tab._on_stage(_BIG_CHAT_ID, "Writing a response", False)
    text = tab.log.item(tab.log.count() - 1).text()
    check("the log line names the real chat_id", str(_BIG_CHAT_ID) in text, detail=text)
    check("...and never the wrapped one", str(_WRAPPED) not in text, detail=text)
    check("the status board also shows the real chat_id, not the wrapped one",
          str(_BIG_CHAT_ID) in tab.status_board.text()
          and str(_WRAPPED) not in tab.status_board.text(),
          detail=tab.status_board.text())


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback; traceback.print_exc()
            print("  ERROR in " + fn.__name__ + ": " + type(e).__name__ + ": " + str(e))
    passed = sum(1 for _, c in RESULTS if c)
    print(f"\n{len(fns)-failed}/{len(fns)} functions, {passed}/{len(RESULTS)} checks passed")
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)

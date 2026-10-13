"""POST /shot: the Debug chat tab brings itself to the front of every tab level of the
live window and saves the window, so a debugging session sees the real GUI without a
desktop grant (2026-10-13: the tab had only been checked as a standalone window).

Run: venv/Scripts/python.exe -m pytest -q tests/test_debug_chat_shot.py
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT] + [os.path.join(ROOT, d) for d in ("gui", "bot", "agent", "core")]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QLabel, QTabWidget  # noqa: E402


def test_a_shot_request_shows_the_tab_and_saves_the_window(tmp_path, monkeypatch):
    import debug_user
    import gui_debug_chat_tab as G
    monkeypatch.setattr(debug_user, "SHOT", str(tmp_path / "shot.png"))
    monkeypatch.setattr(G._Poller, "run", lambda self: None)       # no live API in a suite
    app = QApplication.instance() or QApplication([])
    outer, inner = QTabWidget(), QTabWidget()
    tab = G.DebugChatTab()
    inner.addTab(QLabel("telegram"), "Telegram")
    inner.addTab(tab, "Debug chat")
    outer.addTab(QLabel("chat"), "Chat")
    outer.addTab(inner, "Workspace")
    outer.resize(800, 600)
    outer.show()
    open(debug_user.SHOT + ".req", "w").write("1")
    end = time.time() + 5
    while time.time() < end and not os.path.exists(debug_user.SHOT):
        app.processEvents()
        time.sleep(0.05)
    assert outer.currentWidget() is inner and inner.currentWidget() is tab
    assert os.path.getsize(debug_user.SHOT) > 0 and not os.path.exists(debug_user.SHOT + ".req")
    tab._shot_timer.stop()
    outer.close()

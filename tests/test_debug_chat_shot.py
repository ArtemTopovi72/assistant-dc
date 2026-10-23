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

from PyQt5.QtCore import QObject, pyqtSignal  # noqa: E402
from PyQt5.QtWidgets import QApplication, QLabel, QTabWidget  # noqa: E402


class _Quiet(QObject):
    """No live API in a suite: a poller that never polls."""
    got = pyqtSignal(list)
    state = pyqtSignal(bool)

    def start(self):
        pass


def test_a_shot_request_shows_the_tab_and_saves_the_window(tmp_path):
    import gui_debug_chat_tab as G
    SHOT = str(tmp_path / "shot.png")
    app = QApplication.instance() or QApplication([])
    outer, inner = QTabWidget(), QTabWidget()
    tab = G.DebugChatTab(poller=_Quiet(), shot=SHOT)
    inner.addTab(QLabel("telegram"), "Telegram")
    inner.addTab(tab, "Debug chat")
    outer.addTab(QLabel("chat"), "Chat")
    outer.addTab(inner, "Workspace")
    outer.resize(800, 600)
    outer.show()
    open(SHOT + ".req", "w").write("1")
    end = time.time() + 5
    while time.time() < end and not os.path.exists(SHOT):
        app.processEvents()
        time.sleep(0.05)
    assert outer.currentWidget() is inner and inner.currentWidget() is tab
    assert os.path.getsize(SHOT) > 0 and not os.path.exists(SHOT + ".req")
    tab._shot_timer.stop()
    outer.close()


def test_the_keyboard_keeps_the_chat_the_main_part_of_the_tab():
    """A grid gave a lone button a third of the row and six tall rows took half the window, the
    picture above cut to a strip (owner, 2026-10-24). Each row splits its own width now, and the
    keyboard has a ceiling; the chat view keeps the rest."""
    import gui_debug_chat_tab as G
    app = QApplication.instance() or QApplication([])
    tab = G.DebugChatTab(poller=_Quiet())
    tab.resize(1000, 700)
    tab.show()
    rows = [[{"text": f"b{r}{c}", "callback_data": f"x{r}{c}"} for c in range(n)] for r, n in enumerate((3, 2, 1, 3, 2, 1))]
    tab._add(1, False, "picture", [], markup={"inline_keyboard": rows})
    app.processEvents()
    lone = tab.btn_rows.itemAt(2).layout().itemAt(0).widget()
    pair = tab.btn_rows.itemAt(1).layout().itemAt(0).widget()
    assert lone.width() > 1.8 * pair.width(), (lone.width(), pair.width())
    assert tab.btn_scroll.height() < tab.view.height(), (tab.btn_scroll.height(), tab.view.height())
    tab._add(2, False, "no buttons", [])
    tab._msgs[0]["markup"] = None
    tab._render()
    app.processEvents()
    assert not tab.btn_scroll.isVisible()
    tab.hide()

"""The Workspace's grouped side list (gui_nav) mirrors the hidden tab strip.

Twenty tabs in one strip scrolled off the window; the list replaced the strip but the
QTabWidget stayed the model, so closing a tab, re-opening it from the Layout menu and
_focus_tab must all show up in the list, and a click in the list must switch the tab.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("F5_TEST_RUN", "1")
import logging; logging.basicConfig(level=logging.CRITICAL)
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
import gui
import gui_nav

app = QApplication.instance() or QApplication([])
gui.ModelLoader.start = lambda self: None
OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print("PASS ", name)
    else:
        BAD += 1; print("FAIL ", name, " ", str(extra)[:300])


win = gui.AssistantWindow("m", True)
nav, tabs = win.workspace_nav, win.tabs
keys = lambda: [nav.item(i).data(Qt.UserRole) for i in range(nav.count()) if nav.item(i).data(Qt.UserRole)]
heads = [nav.item(i).text() for i in range(nav.count()) if not nav.item(i).data(Qt.UserRole)]

check("the strip is hidden, the list carries every tab",
      tabs.tabBar().isHidden() and sorted(keys()) == sorted(k for k, _, _ in win._tab_registry), keys())
check("grouped: a header per group, every registry key placed in one",
      len(heads) == len(gui_nav.GROUPS)
      and {k for _, ks in gui_nav.GROUPS for k in ks} == {k for k, _, _ in win._tab_registry}, heads)
check("headers cannot be selected", all(not (nav.item(i).flags() & Qt.ItemIsSelectable)
                                        for i in range(nav.count()) if not nav.item(i).data(Qt.UserRole)))

music = next(nav.item(i) for i in range(nav.count()) if nav.item(i).data(Qt.UserRole) == "music")
nav.setCurrentItem(music)
check("a click in the list switches the tab", win._tab_key_for_widget(tabs.currentWidget()) == "music")

win._focus_tab("memory")
check("_focus_tab moves the list's selection too", nav.currentItem().data(Qt.UserRole) == "memory")

win._on_tab_close_requested(win._tab_index_of("weather"))
check("a closed tab leaves the list", "weather" not in keys(), keys())
win._show_tab("weather", True)
check("re-opened from the Layout menu, it comes back in its group", "weather" in keys()
      and keys().index("weather") > keys().index("telegram"), keys())

win._show_all_tabs()
check("show-all keeps the list whole", len(keys()) == len(win._tab_registry))

print(f"\n{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)

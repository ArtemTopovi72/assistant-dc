"""Workspace navigation: a grouped list down the left side of the Workspace page.

Twenty tabs in one strip scrolled off the window (10-09: the strip ran past «Status»
and the rest were behind an arrow nobody saw). The QTabWidget stays the model -- the
layout presets, ✕-to-hide, the Layout menu and _focus_tab all keep working on it --
only its tab bar is hidden and this list mirrors it, grouped by what the tabs are for.
"""
from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QListWidget, QListWidgetItem, QMenu, QTabWidget

from gui_i18n import tr
from ui_scale import px

# (group label, stable tab keys) -- the keys of the dashboard's _tab_registry
GROUPS = (
    ("Creation", ("images", "storyboard", "transfer", "restyle", "music", "voice_clone", "characters")),
    ("Knowledge", ("search", "research", "database", "memory")),
    ("Bot & services", ("telegram", "admin", "weather", "madhouse")),
    ("System", ("model", "status", "log", "code", "stress")),
)
_KEY = Qt.UserRole


class NavTabs(QTabWidget):
    """A QTabWidget that says when a tab comes or goes (Qt has no signal for it)."""
    changed = pyqtSignal()

    def tabInserted(self, index):
        super().tabInserted(index)
        self.changed.emit()

    def tabRemoved(self, index):
        super().tabRemoved(index)
        self.changed.emit()


class WorkspaceNav(QListWidget):
    def __init__(self, tabs: NavTabs, key_of, parent=None):
        super().__init__(parent)
        self.setObjectName("workspaceNav")
        self._tabs, self._key_of = tabs, key_of
        self.setFixedWidth(px(210))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)
        self.currentItemChanged.connect(self._picked)
        tabs.changed.connect(self.rebuild)
        tabs.currentChanged.connect(self._sync)
        tabs.tabBar().tabMoved.connect(lambda *_: self.rebuild())
        self.rebuild()

    def rebuild(self) -> None:
        tabs = self._tabs
        present = {}
        for i in range(tabs.count()):
            w = tabs.widget(i)
            present[self._key_of(w) or f"_{i}"] = w
        groups = [(label, [k for k in keys if k in present]) for label, keys in GROUPS]
        known = {k for _, keys in GROUPS for k in keys}
        rest = [k for k in present if k not in known]
        if rest:
            groups.append(("Other", rest))
        self.blockSignals(True)
        self.clear()
        for label, keys in groups:
            if not keys:
                continue
            head = QListWidgetItem(tr(label).upper())
            head.setFlags(Qt.NoItemFlags)
            head.setData(Qt.AccessibleTextRole, tr(label))
            f = head.font(); f.setBold(True); f.setLetterSpacing(f.PercentageSpacing, 108); head.setFont(f)
            self.addItem(head)
            for k in keys:
                w = present[k]
                it = QListWidgetItem("   " + tabs.tabText(tabs.indexOf(w)))
                it.setData(_KEY, k)
                self.addItem(it)
        self.blockSignals(False)
        self._sync()

    def _item_for(self, widget):
        key = self._key_of(widget)
        for i in range(self.count()):
            if self.item(i).data(_KEY) == key:
                return self.item(i)
        return None

    def _sync(self, *_):
        it = self._item_for(self._tabs.currentWidget())
        if it is not None and it is not self.currentItem():
            self.blockSignals(True)
            self.setCurrentItem(it)
            self.blockSignals(False)

    def _widget_of(self, item):
        key = item.data(_KEY) if item is not None else None
        for i in range(self._tabs.count()):
            if self._key_of(self._tabs.widget(i)) == key:
                return self._tabs.widget(i)
        return None

    def _picked(self, item, _prev=None):
        w = self._widget_of(item)
        if w is not None:
            self._tabs.setCurrentWidget(w)

    def _menu(self, pos):
        item = self.itemAt(pos)
        w = self._widget_of(item)
        if w is None:
            return
        m = QMenu(self)
        close = m.addAction(tr("Close"))
        if m.exec_(self.mapToGlobal(pos)) is close:
            self._tabs.tabCloseRequested.emit(self._tabs.indexOf(w))

    def showEvent(self, e):
        # a language switch re-labels the tabs without a signal; catch up when shown
        self.rebuild()
        super().showEvent(e)

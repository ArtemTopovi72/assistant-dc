"""Window chrome, tab visibility and layout persistence for AssistantWindow.

A mixin, not a module of free functions: every one of these methods is about
THIS window's geometry and THIS window's saved state, so they need `self`.
Splitting them out is still worth it — none of them touch the agent, the
models or any worker, so the half of AssistantWindow that decides what the
window LOOKS like is now readable without scrolling past the half that decides
what it DOES.

The mixin expects three things from the host class: `_settings()`,
`_registry_get()` and `_add_system()`. Nothing else.
"""
import json
import logging

from PyQt5.QtCore import QByteArray, QPoint, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication, QHBoxLayout, QInputDialog, QLabel, QMenu, QPushButton, QWidget,
)

import gui_common
from gui_common import ICON_PATH
from ui_scale import px

logger = logging.getLogger("assistant")


class LayoutMixin:
    """Titlebar, screen clamping, tab show/hide and layout presets."""

    def _build_titlebar(self):
        bar = QWidget()
        bar.setObjectName("titlebar")
        bar.setFixedHeight(px(42))
        h = QHBoxLayout(bar)
        h.setContentsMargins(px(12), 0, px(8), 0)
        h.setSpacing(px(10))
        ic = QLabel()
        pm = QIcon(ICON_PATH).pixmap(px(22), px(22))
        if not pm.isNull():
            ic.setPixmap(pm)
        ttl = QLabel("Assistant DC")
        ttl.setObjectName("titletext")
        self._max_btn = QPushButton("□")
        min_btn = QPushButton("─"); min_btn.setObjectName("winbtn"); min_btn.setFixedSize(px(40), px(28)); min_btn.clicked.connect(self.showMinimized)
        self._max_btn.setObjectName("winbtn"); self._max_btn.setFixedSize(px(40), px(28)); self._max_btn.clicked.connect(self._toggle_max)
        close_btn = QPushButton("×"); close_btn.setObjectName("winclose"); close_btn.setFixedSize(px(40), px(28)); close_btn.clicked.connect(self.close)
        h.addWidget(ic)
        h.addWidget(ttl)
        h.addStretch(1)
        h.addWidget(min_btn)
        h.addWidget(self._max_btn)
        h.addWidget(close_btn)
        # Drag-to-move. The previous event-filter + manual-move approach was
        # unreliable (offset math broke under fractional UI scaling). Robust fix:
        #  1. make the non-interactive children (icon, title) transparent to the
        #     mouse so a press anywhere on them falls THROUGH to the bar itself;
        #  2. give the bar its own mousePressEvent that hands the drag to the OS
        #     window manager via startSystemMove() (DPI-correct, multi-monitor);
        #  3. keep a manual-move fallback for the rare platform that refuses it.
        # The min/max/close buttons stay opaque, so they keep handling their clicks.
        ic.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        ttl.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        bar.mousePressEvent = self._titlebar_press
        bar.mouseMoveEvent = self._titlebar_move
        bar.mouseReleaseEvent = self._titlebar_release
        bar.mouseDoubleClickEvent = lambda e: self._toggle_max()
        self._title_bar = bar
        return bar

    def _titlebar_press(self, e):
        if e.button() != Qt.LeftButton or self.isMaximized():
            return
        if getattr(self, "_faux_max", False):
            # dragging a work-area-filled window un-maximizes it, like a real one
            self._faux_max = False
            self._max_btn.setText("□")
        wh = self.windowHandle()
        if wh is not None and wh.startSystemMove():
            self._drag_offset = None                # OS owns the drag
            return
        # Fallback: track manually relative to the press point.
        self._drag_offset = e.globalPos() - self.frameGeometry().topLeft()

    def _titlebar_move(self, e):
        if self._drag_offset is not None and (e.buttons() & Qt.LeftButton) \
                and not self.isMaximized():
            self.move(e.globalPos() - self._drag_offset)

    def _titlebar_release(self, e):
        self._drag_offset = None

    def _screen_area(self):
        """Work area of the screen this window is actually on (not always the primary)."""
        try:
            wh = self.windowHandle()
            scr = wh.screen() if wh is not None else None
            if scr is None:
                scr = QApplication.screenAt(self.frameGeometry().center())
            return (scr or QApplication.primaryScreen()).availableGeometry()
        except Exception:
            return QApplication.primaryScreen().availableGeometry()

    def _clamp_to_screen(self):
        """Keep the window inside the screen's work area.

        A frameless window can end up larger than the screen (a restored geometry
        from a bigger monitor, or Windows' frameless-maximize overshoot, which
        pushes the frame a few px past every edge). On a desktop that just looks
        untidy; on a TV — where overscan eats the edges too — it clips the
        titlebar buttons and the right-hand tab strip outright.
        """
        try:
            avail = self._screen_area()
            # Qt will not shrink a window below its layout minimum, so if that minimum
            # is bigger than the display the window is cropped by the OS and nothing
            # can be dragged back into view. Release the floor first (the panes scroll,
            # so their content stays reachable), then clamp the geometry.
            hint = self.minimumSizeHint()
            need_w = max(hint.width(), self.minimumWidth())
            need_h = max(hint.height(), self.minimumHeight())
            if need_w > avail.width() or need_h > avail.height():
                logger.warning("window minimum %dx%d exceeds the %dx%d work area — "
                               "capping it so the window fits the screen",
                               need_w, need_h, avail.width(), avail.height())
                self.setMinimumSize(min(need_w, avail.width()),
                                    min(need_h, avail.height()))
            frame, client = self.frameGeometry(), self.geometry()
            # setGeometry() takes the CLIENT rect; the constraint is on the FRAME
            # (title bar + borders), so carry the decoration size across. On the
            # frameless build these deltas are zero and this is a plain clamp.
            dw, dh = frame.width() - client.width(), frame.height() - client.height()
            dx, dy = client.x() - frame.x(), client.y() - frame.y()
            w = min(frame.width(), avail.width())
            h = min(frame.height(), avail.height())
            x = min(max(frame.x(), avail.x()), avail.x() + avail.width() - w)
            y = min(max(frame.y(), avail.y()), avail.y() + avail.height() - h)
            if (w, h) != (frame.width(), frame.height()) or (x, y) != (frame.x(), frame.y()):
                self.setGeometry(x + dx, y + dy, max(px(320), w - dw), max(px(240), h - dh))
        except Exception:
            logger.exception("clamping the window to the screen failed")

    def _toggle_max(self):
        if self.isMaximized() or self._faux_max:
            self._faux_max = False
            self.showNormal(); self._max_btn.setText("□")
            w, h = self._default_window_size()
            self.resize(w, h)
            self._clamp_to_screen()
        elif gui_common.NATIVE_FRAME:
            self.showMaximized(); self._max_btn.setText("❐")
        else:
            # Frameless: fill the WORK AREA by hand. showMaximized() on a frameless
            # window overshoots every edge by the (invisible) resize border, so the
            # titlebar buttons and the tab strip fall off the screen — exactly the
            # "cropped on the TV" full-screen look.
            self._faux_max = True
            self._clamp_to_screen()          # releases an oversized layout minimum first
            self.setGeometry(self._screen_area())
            self._max_btn.setText("❐")
    # ---- layout: dockable-feel tabs, presets, persistence -------------------
    # Three full-window pages (commands · conversation · workspace); the chat and its queue
    # share a drag-resizable splitter, and the window is drag-movable (custom titlebar / OS frame). On top of that: tabs are
    # closable + reorderable, panels can be re-opened from the Layout menu, named presets
    # save/restore the whole arrangement, the last layout is remembered across launches,
    # and Reset returns everything to the hardcoded default.

    def _default_splitter_sizes(self):
        return [px(900), px(280)]   # conversation vs the task queue beside it

    def _default_window_size(self):
        w, h = px(1180), px(760)
        try:
            avail = QApplication.primaryScreen().availableGeometry()
            w = min(w, int(avail.width() * 0.96))
            h = min(h, int(avail.height() * 0.94))
        except Exception:
            pass
        return w, h

    def _focus_tab(self, key):
        """Bring a tab to the front by its stable key.

        Not by widget: pages are wrapped in a scroll area (see _scroll_page), so the
        widget the QTabWidget knows is the wrapper, not e.g. self.research_container.
        """
        entry = self._registry_get(key)
        if entry is None:
            return
        idx = self._tab_index_of(key)
        if idx is None:                       # the tab was closed — re-open it
            self.tabs.addTab(entry[0], entry[1])
            idx = self.tabs.indexOf(entry[0])
        self.tabs.setCurrentIndex(idx)
        self.tabs.tabBar().setCurrentIndex(idx)

    def _tab_key_for_widget(self, widget):
        for k, w, _label in self._tab_registry:
            if w is widget:
                return k
        return None

    def _visible_tab_keys(self):
        keys = []
        for i in range(self.tabs.count()):
            k = self._tab_key_for_widget(self.tabs.widget(i))
            if k:
                keys.append(k)
        return keys

    def _tab_index_of(self, key):
        for i in range(self.tabs.count()):
            if self._tab_key_for_widget(self.tabs.widget(i)) == key:
                return i
        return None

    def _is_tab_visible(self, key):
        return self._tab_index_of(key) is not None

    def _on_tab_close_requested(self, index):
        # Hide (not destroy): the widget stays in the registry so the Layout menu can
        # re-open it later with its full state intact.
        if 0 <= index < self.tabs.count():
            self.tabs.removeTab(index)

    def _show_tab(self, key, show):
        idx = self._tab_index_of(key)
        if show and idx is None:
            item = self._registry_get(key)
            if not item:
                return
            default_order = [k for k, _, _ in self._tab_registry]
            try:
                rank = default_order.index(key)
            except ValueError:
                rank = len(default_order)
            insert_at = sum(1 for vk in self._visible_tab_keys()
                            if default_order.index(vk) < rank)
            self.tabs.insertTab(insert_at, item[0], item[1])
            self.tabs.setCurrentIndex(insert_at)
            # Force the tab bar to scroll so the new tab is visible.
            self.tabs.tabBar().setCurrentIndex(insert_at)
        elif not show and idx is not None:
            self.tabs.removeTab(idx)

    def _show_all_tabs(self):
        self._apply_tab_config([k for k, _, _ in self._tab_registry],
                               self._tab_key_for_widget(self.tabs.currentWidget()),
                               known=[k for k, _, _ in self._tab_registry])

    def _apply_tab_config(self, order_keys, current_key, known=None):
        """Rebuild the tab bar from a saved order. `known` = the set of panels that
        existed when the layout was saved; panels NOT in `known` are brand-new (added in
        a later build) and are appended visible, so an update never hides a new feature —
        while panels the user deliberately closed stay closed."""
        known = set(known if known is not None else order_keys)
        order_keys = [k for k in (order_keys or []) if self._registry_get(k)]
        while self.tabs.count():
            self.tabs.removeTab(0)
        seen = set()
        for k in order_keys:
            if k in seen:
                continue
            w, label = self._registry_get(k)
            self.tabs.addTab(w, label)
            seen.add(k)
        # Append any panel that didn't exist when this layout was saved.
        for k, w, label in self._tab_registry:
            if k not in known and k not in seen:
                self.tabs.addTab(w, label)
                seen.add(k)
        ci = self._tab_index_of(current_key)
        if ci is not None:
            self.tabs.setCurrentIndex(ci)

    def _capture_layout(self):
        return {
            "geometry": bytes(self.saveGeometry().toBase64()).decode("ascii"),
            "splitter": list(self.splitter.sizes()),
            "page": self.pages.currentIndex(),
            "tabs": self._visible_tab_keys(),
            "current": self._tab_key_for_widget(self.tabs.currentWidget()) or "",
            "known": [k for k, _, _ in self._tab_registry],
        }

    def _apply_layout(self, d, *, geometry=True):
        if not d:
            return
        try:
            if geometry and d.get("geometry"):
                self.restoreGeometry(QByteArray.fromBase64(d["geometry"].encode("ascii")))
                # A geometry saved on a larger screen (or at a smaller UI scale) must
                # not hang off the edges of THIS one.
                self._clamp_to_screen()
            # Layouts saved before the pages carried three column widths.
            if d.get("splitter") and len(d["splitter"]) == self.splitter.count():
                self.splitter.setSizes([int(x) for x in d["splitter"]])
            if isinstance(d.get("page"), int) and 0 <= d["page"] < self.pages.count():
                self.pages.setCurrentIndex(d["page"])
            if d.get("tabs") is not None:
                self._apply_tab_config(d.get("tabs"), d.get("current", ""), d.get("known"))
        except Exception:
            logger.warning("layout: failed to apply layout", exc_info=True)

    def _default_layout(self):
        keys = [k for k, _, _ in self._tab_registry]
        return {"geometry": None, "splitter": self._default_splitter_sizes(),
                "page": 1,
                "tabs": keys, "current": keys[0] if keys else "", "known": keys}

    def _reset_layout(self):
        self._apply_layout(self._default_layout(), geometry=False)
        w, h = self._default_window_size()
        self.resize(w, h)
        try:
            self._settings().remove("last")   # so the default sticks until changed again
        except Exception:
            pass
        self._add_system("Layout reset to default.")

    def _persist_layout(self):
        try:
            self._settings().setValue("last", json.dumps(self._capture_layout()))
        except Exception:
            logger.warning("layout: failed to persist last layout", exc_info=True)

    def _restore_last_layout(self):
        try:
            raw = self._settings().value("last", "")
            if raw:
                self._apply_layout(json.loads(raw), geometry=True)
        except Exception:
            logger.warning("layout: failed to restore last layout", exc_info=True)

    def _preset_names(self):
        try:
            raw = self._settings().value("preset_names", "")
            return list(json.loads(raw)) if raw else []
        except Exception:
            return []

    def _save_preset(self):
        name, ok = QInputDialog.getText(self, "Save layout preset", "Preset name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        try:
            s = self._settings()
            s.setValue(f"preset/{name}", json.dumps(self._capture_layout()))
            names = self._preset_names()
            if name not in names:
                names.append(name)
            s.setValue("preset_names", json.dumps(names))
            self._add_system(f"Saved layout preset “{name}”.")
        except Exception:
            logger.warning("layout: save preset failed", exc_info=True)

    def _load_preset(self, name):
        try:
            raw = self._settings().value(f"preset/{name}", "")
            if raw:
                self._apply_layout(json.loads(raw), geometry=True)
                self._add_system(f"Loaded layout preset “{name}”.")
        except Exception:
            logger.warning("layout: load preset failed", exc_info=True)

    def _delete_preset(self, name):
        try:
            s = self._settings()
            s.remove(f"preset/{name}")
            s.setValue("preset_names", json.dumps([n for n in self._preset_names() if n != name]))
            self._add_system(f"Deleted layout preset “{name}”.")
        except Exception:
            logger.warning("layout: delete preset failed", exc_info=True)

    def _open_layout_menu(self):
        menu = QMenu(self)
        panels = menu.addMenu("Panels")
        for key, _w, label in self._tab_registry:
            act = panels.addAction(label)
            act.setCheckable(True)
            act.setChecked(self._is_tab_visible(key))
            act.toggled.connect(lambda chk, k=key: self._show_tab(k, chk))
        panels.addSeparator()
        panels.addAction("Show all panels", self._show_all_tabs)

        menu.addSeparator()
        names = self._preset_names()
        if names:
            load_m = menu.addMenu("Load preset")
            for n in names:
                load_m.addAction(n, lambda _=False, nm=n: self._load_preset(nm))
            del_m = menu.addMenu("Delete preset")
            for n in names:
                del_m.addAction(n, lambda _=False, nm=n: self._delete_preset(nm))
        menu.addAction("Save preset…", self._save_preset)
        menu.addSeparator()
        menu.addAction("Reset to default layout", self._reset_layout)
        menu.exec_(self.layout_btn.mapToGlobal(QPoint(0, self.layout_btn.height())))

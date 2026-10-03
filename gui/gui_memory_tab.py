"""Memory Center tab: the console for everything the assistant remembers.

First tab lifted out of gui.py, chosen because it is the largest (927 lines)
and the most self-contained — AssistantWindow is the only thing that references
it. It became movable only once gui_common.py existed: nine of the eleven
gui.py symbols it needed were shared UI primitives, and without them in a
module of their own this file would have had to import gui.py and rebuild the
import cycle that was just eliminated.

CompactMemoryWorker travels with it (it is the tab's own background compaction
thread) and is re-exported from gui.py, because AssistantWindow drives it too.

The heavy imports — memory_center, llm, prompts — stay FUNCTION-LOCAL exactly
as they were in gui.py: this tab is constructed at startup, and loading the
memory store or the LLM client just to build a widget would slow every launch.
"""
import json
import os
from pathlib import Path

from PyQt5.QtCore import QEvent, QThread, Qt, pyqtSignal
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QFileDialog,
                             QFrame, QGridLayout, QHBoxLayout, QHeaderView,
                             QInputDialog, QLabel, QLineEdit, QListWidget,
                             QListWidgetItem, QMenu, QMessageBox, QPlainTextEdit,
                             QPushButton, QScrollArea, QSlider, QSpinBox,
                             QTabWidget, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)

from ui_scale import px, scale_style as _ss
from gui_common import (BG, MUTED, _FlowWidget, _card, _fit_dialog, _flow,
                        _fmt_bytes, _fmt_ts, _section)

import logging
logger = logging.getLogger("assistant.gui")   # same channel as gui.py


class CompactMemoryWorker(QThread):
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx

    def run(self):
        try:
            from prompts import COMPACT_MEMORY_PROMPT
            from llm import send_to_lm_studio
            from utils import strip_think_tags, strip_reasoning_leak
            memory_text = self.ctx.memory_text(limit=50)
            if not memory_text:
                self.done.emit("")
                return
            messages = [
                {"role": "system", "content": COMPACT_MEMORY_PROMPT},
                {"role": "user", "content": memory_text},
            ]
            resp = send_to_lm_studio(self.ctx, messages, tools=[], tool_choice="none",
                                     temperature=0.3, max_tokens=500)
            text = (resp.get("content") or "").strip() if resp else ""
            # The summary is persisted and re-injected every future turn — leaked
            # <think> reasoning here would be baked into the profile forever.
            text = strip_reasoning_leak(strip_think_tags(text)).strip()
            self.done.emit(text)
        except Exception as exc:
            logger.exception("Memory compact failed")
            self.failed.emit(str(exc))


# The profiles page and the maintenance pages (compaction / timeline /
# diagnostics / backup) live in their own mixins. MemoryCenterTab is one class,
# so these are mixins rather than lifted functions -- they need self.
# gui_memory_maint reaches CompactMemoryWorker back through this module, at
# call time, so the two-way dependency never runs at import.
from gui_memory_profiles import MemoryProfilesMixin as _MemoryProfilesMixin
from gui_memory_maint import MemoryMaintenanceMixin as _MemoryMaintenanceMixin


class MemoryCenterTab(_MemoryProfilesMixin, _MemoryMaintenanceMixin, QWidget):
    """Full memory-management console built on memory_center.MemoryStore.

    Browses ANY profile (independent of the live active profile), and syncs edits
    back into the running Context when they touch the active profile so the
    assistant uses them immediately. Every destructive action is confirmed; deletes
    are soft (recoverable from Trash); compaction is review-before-commit.
    """

    def __init__(self, host):
        super().__init__()
        self.host = host
        from memory_center import MemoryStore, TYPES, SOURCES, SUGGESTED_PROFILES
        self.store = MemoryStore()
        self._TYPES, self._SOURCES, self._SUGGESTED = TYPES, SOURCES, SUGGESTED_PROFILES
        self._rows = []            # parallel list of MemoryEntry for the table
        self._sel = None           # selected MemoryEntry
        self._compact_preview = None
        self._compact_worker = None
        self._build()
        self.refresh_all()

    # ---- shared helpers ----------------------------------------------------
    def _profile(self) -> str:
        return self.profile_combo.currentText() or "default"

    def _active_profile(self) -> str:
        ctx = getattr(self.host, "ctx", None)
        if ctx is not None and getattr(ctx, "active_memory_dir", None):
            return Path(ctx.active_memory_dir).name
        return "default"

    def _confirm(self, title: str, text: str) -> bool:
        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setText(text)
        box.setIcon(QMessageBox.Warning)
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        return box.exec_() == QMessageBox.Yes

    def _toast(self, msg: str):
        if hasattr(self.host, "_set_status"):
            self.host._set_status(msg)

    def _after_mutation(self):
        """Sync the live ctx (if the edited profile is active) and refresh views."""
        ctx = getattr(self.host, "ctx", None)
        try:
            self.store.sync_into_ctx(ctx, self._profile())
        except Exception:
            pass
        self.refresh_all()

    # ---- build -------------------------------------------------------------
    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(px(10), px(10), px(10), px(10))
        root.setSpacing(px(8))

        # profile selector + active badge (shared across sub-tabs).
        # Row 1: PROFILE label + combo (combo gets full width). Row 2: a flow with
        # the active badge + Make Active so they show full text and wrap on narrow
        # panels instead of truncating to "active: de / Make Acti".
        prow = QHBoxLayout(); prow.setSpacing(px(8))
        prow.addWidget(_section("Profile"))
        self.profile_combo = QComboBox()
        self.profile_combo.setMinimumHeight(px(30))
        self.profile_combo.currentTextChanged.connect(lambda _=None: self.refresh_all())
        prow.addWidget(self.profile_combo, 1)
        root.addLayout(prow)

        aflow = _flow(spacing=px(8))
        self.active_badge = QLabel("")
        self.active_badge.setObjectName("chip")
        aflow.addWidget(self.active_badge)
        self.switch_active_btn = QPushButton("Make Active")
        self.switch_active_btn.setObjectName("ghost")
        self.switch_active_btn.setCursor(Qt.PointingHandCursor)
        self.switch_active_btn.setToolTip("Load this profile into the running assistant")
        self.switch_active_btn.clicked.connect(self._make_active)
        aflow.addWidget(self.switch_active_btn)
        root.addWidget(_FlowWidget(aflow))

        self.subtabs = QTabWidget()
        self.subtabs.addTab(self._build_overview(), "Overview")
        self.subtabs.addTab(self._build_explorer(), "Explorer")
        self.subtabs.addTab(self._build_profiles(), "Profiles")
        self.subtabs.addTab(self._build_compaction(), "Compaction")
        self.subtabs.addTab(self._build_timeline(), "Timeline")
        self.subtabs.addTab(self._build_diagnostics(), "Diagnostics")
        self.subtabs.addTab(self._build_backup(), "Backup")
        self.subtabs.currentChanged.connect(lambda _=None: self.refresh_all())
        root.addWidget(self.subtabs, 1)

    # ---- Overview ----------------------------------------------------------
    def _new_vscroll_page(self):
        """Build the scroll+content-layout shell shared by Overview and Explorer:
        a vertically-scrolling area (no horizontal bar) over a dark viewport, holding
        a fresh QWidget/QVBoxLayout pair for the caller to populate. Returns
        (scroll, lay) — keep a reference to `scroll` (the widget to return from the
        tab builder) and add content to `lay`."""
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.viewport().setStyleSheet(_ss(f"background:{BG};"))  # keep dark, not white
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(px(10), px(10), px(10), px(10)); lay.setSpacing(px(12))
        scroll.setWidget(w)
        return scroll, lay

    def _build_overview(self):
        # Scrollable so the stats card + actions never clip on a short panel.
        scroll, lay = self._new_vscroll_page()

        # stats grid inside a card
        stat_card, sc = _card(QVBoxLayout)
        sc.setSpacing(px(8))
        sc.addWidget(_section("At a glance"))
        self.ov_grid = QGridLayout()
        self.ov_grid.setHorizontalSpacing(px(14)); self.ov_grid.setVerticalSpacing(px(8))
        self._ov_labels = {}
        fields = ["Total memories", "Facts", "Session memory", "Summaries",
                  "Active profile", "Disk usage", "Last update", "Last compaction",
                  "Retrieval mode", "Semantic search", "Trash"]
        for i, name in enumerate(fields):
            k = QLabel(name); k.setStyleSheet(_ss(f"color:{MUTED}; font-size:13px;"))
            v = QLabel("—"); v.setStyleSheet(_ss("font-size:14px; font-weight:600;"))
            v.setWordWrap(True)
            self.ov_grid.addWidget(k, i, 0, Qt.AlignTop)
            self.ov_grid.addWidget(v, i, 1)
            self._ov_labels[name] = v
        self.ov_grid.setColumnStretch(1, 1)
        sc.addLayout(self.ov_grid)
        lay.addWidget(stat_card)

        # actions in a card, reflowing so labels are never truncated
        act_card, ac = _card(QVBoxLayout)
        ac.setSpacing(px(8))
        ac.addWidget(_section("Actions"))
        aflow = _flow(spacing=px(8))
        for label, slot in [("↻  Refresh", self.refresh_all),
                            ("Rebuild index", self._rebuild_index),
                            ("Compact…", lambda: self.subtabs.setCurrentIndex(3)),
                            ("Export backup", lambda: self.subtabs.setCurrentIndex(6)),
                            ("Open profile folder", self._open_folder)]:
            b = QPushButton(label); b.setObjectName("ghost")
            b.setCursor(Qt.PointingHandCursor); b.clicked.connect(slot)
            aflow.addWidget(b)
        ac.addWidget(_FlowWidget(aflow))
        lay.addWidget(act_card)
        lay.addStretch(1)
        return scroll

    def _refresh_overview(self):
        ov = self.store.overview(self._profile())
        L = self._ov_labels
        L["Total memories"].setText(str(ov["total"]))
        L["Facts"].setText(str(ov["facts"]))
        L["Session memory"].setText(str(ov["session"]))
        L["Summaries"].setText(str(ov["summary"]))
        L["Active profile"].setText(self._active_profile())
        L["Disk usage"].setText(_fmt_bytes(ov["disk_bytes"]))
        L["Last update"].setText(_fmt_ts(ov["last_update"]))
        L["Last compaction"].setText(_fmt_ts(ov["last_compaction"]))
        L["Retrieval mode"].setText(ov["retrieval_mode"])
        L["Semantic search"].setText("enabled" if ov["semantic_enabled"] else "not yet (keyword only)")
        L["Trash"].setText(str(ov["trash"]))

    # ---- Explorer ----------------------------------------------------------
    # Column model for the memory table. Priority drives responsive hiding:
    # higher-priority columns are dropped first when the viewport gets narrow.
    _COL_TYPE, _COL_SOURCE, _COL_CREATED, _COL_IMP, _COL_TAGS, _COL_PREVIEW = range(6)
    _COL_HIDE_ORDER = [_COL_TAGS, _COL_SOURCE, _COL_CREATED]   # dropped in this order

    def _build_explorer(self):
        # The Explorer can live in a narrow, short side panel, so the whole tab
        # is vertically scrollable: panels stack and the view scrolls rather than
        # collapsing the table or clipping the editor/bulk controls.
        scroll, lay = self._new_vscroll_page()

        # ---- SECTION 1: Search & filter -----------------------------------
        sf_card, sf = _card(QVBoxLayout)
        sf.setSpacing(px(8))
        sf.addWidget(_section("Search & filter"))
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("🔍  Search memories by text…")
        self.search_box.setClearButtonEnabled(True)
        self.search_box.setMinimumHeight(px(34))
        self.search_box.textChanged.connect(self._refresh_table)
        sf.addWidget(self.search_box)

        filters = _flow(spacing=px(8))
        self.type_filter = QComboBox(); self.type_filter.addItems(["all types", *self._TYPES])
        self.type_filter.setToolTip("Filter by memory type")
        self.type_filter.currentTextChanged.connect(self._refresh_table)
        self.source_filter = QComboBox(); self.source_filter.addItems(["all sources", *self._SOURCES])
        self.source_filter.setToolTip("Filter by source")
        self.source_filter.currentTextChanged.connect(self._refresh_table)
        imp_lbl = QLabel("min importance"); imp_lbl.setStyleSheet(_ss(f"color:{MUTED};"))
        self.imp_filter = QSpinBox(); self.imp_filter.setRange(0, 100); self.imp_filter.setValue(0)
        self.imp_filter.setToolTip("Hide memories below this importance")
        self.imp_filter.valueChanged.connect(self._refresh_table)
        for wdg in (self.type_filter, self.source_filter, imp_lbl, self.imp_filter):
            filters.addWidget(wdg)
        sf.addWidget(_FlowWidget(filters))
        lay.addWidget(sf_card)

        # ---- SECTION 2: Memory list ---------------------------------------
        list_card, ll = _card(QVBoxLayout)
        ll.setSpacing(px(8))
        head = QHBoxLayout()
        head.addWidget(_section("Memories"))
        head.addStretch(1)
        self.result_count = QLabel("—")
        self.result_count.setStyleSheet(_ss(f"color:{MUTED}; font-size:12px;"))
        head.addWidget(self.result_count)
        ll.addLayout(head)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Type", "Source", "Created", "Imp", "Tags", "Preview"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.setTextElideMode(Qt.ElideRight)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(px(30))
        hdr = self.table.horizontalHeader()
        hdr.setHighlightSections(False)
        hdr.setStretchLastSection(True)            # Preview absorbs slack
        hdr.setSectionResizeMode(self._COL_TYPE, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(self._COL_SOURCE, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(self._COL_CREATED, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(self._COL_IMP, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(self._COL_TAGS, QHeaderView.Interactive)
        hdr.setSectionResizeMode(self._COL_PREVIEW, QHeaderView.Stretch)
        hdr.setMinimumSectionSize(px(44))
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        self.table.setMinimumHeight(px(240))   # ~8 rows; stays usable when scrolled
        # responsive: hide low-value columns when the viewport is narrow
        self.table.viewport().installEventFilter(self)
        ll.addWidget(self.table, 1)
        lay.addWidget(list_card, 1)            # stretches on tall panels

        # ---- SECTION 3: Editor --------------------------------------------
        ed_card, dl = _card(QVBoxLayout)
        dl.setSpacing(px(8))
        dl.addWidget(_section("Editor"))
        self.detail_meta = QLabel("Select a memory above to view or edit it.")
        self.detail_meta.setWordWrap(True)
        self.detail_meta.setStyleSheet(_ss(f"color:{MUTED}; font-size:12px;"))
        dl.addWidget(self.detail_meta)
        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText("Memory content…")
        self.editor.setMinimumHeight(px(90))
        dl.addWidget(self.editor, 1)

        erow = QHBoxLayout(); erow.setSpacing(px(8))
        il = QLabel("Importance"); il.setStyleSheet(_ss(f"color:{MUTED};"))
        erow.addWidget(il)
        self.imp_slider = QSlider(Qt.Horizontal); self.imp_slider.setRange(0, 100)
        erow.addWidget(self.imp_slider, 1)
        self.imp_val = QLabel("50"); self.imp_val.setMinimumWidth(px(28))
        self.imp_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        erow.addWidget(self.imp_val)
        self.imp_slider.valueChanged.connect(lambda v: self.imp_val.setText(str(v)))
        dl.addLayout(erow)
        self.tags_edit = QLineEdit(); self.tags_edit.setPlaceholderText("tags, comma, separated")
        dl.addWidget(self.tags_edit)

        # editor actions reflow instead of truncating
        eflow = _flow(spacing=px(8))
        for label, slot, obj in [
                ("💾  Save", self._save_entry, None),
                ("Duplicate", self._dup_entry, "ghost"),
                ("Pin / Unpin", self._toggle_pin, "ghost"),
                ("Move ▾", self._move_menu, "ghost"),
                ("Revisions", self._revisions_dialog, "ghost"),
                ("🗑  Delete", self._delete_selected, "ghost")]:
            b = QPushButton(label)
            if obj:
                b.setObjectName(obj)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(slot)
            eflow.addWidget(b)
        dl.addWidget(_FlowWidget(eflow))
        lay.addWidget(ed_card)

        # ---- SECTION 4: Bulk actions --------------------------------------
        bulk_card, bl = _card(QVBoxLayout)
        bl.setSpacing(px(8))
        bhead = QHBoxLayout()
        bhead.addWidget(_section("Bulk actions"))
        bhead.addStretch(1)
        self.sel_count = QLabel("0 selected")
        self.sel_count.setObjectName("chip")
        bhead.addWidget(self.sel_count)
        bl.addLayout(bhead)
        bflow = _flow(spacing=px(8))
        add_btn = QPushButton("➕  Add memory"); add_btn.setCursor(Qt.PointingHandCursor)
        add_btn.clicked.connect(self._add_dialog)
        bflow.addWidget(add_btn)
        for label, slot in [("Delete selected", self._delete_selected),
                            ("Delete all matching", self._delete_matching),
                            ("Trash…", self._trash_dialog)]:
            b = QPushButton(label); b.setObjectName("ghost")
            b.setCursor(Qt.PointingHandCursor); b.clicked.connect(slot)
            bflow.addWidget(b)
        bl.addWidget(_FlowWidget(bflow))
        lay.addWidget(bulk_card)
        return scroll

    def eventFilter(self, obj, event):
        """Responsive table: hide low-value columns (Tags, Source, Created) one by
        one as the viewport narrows so cell text is never clipped and the
        horizontal scrollbar stays away on small windows / high DPI scales."""
        if (getattr(self, "table", None) is not None
                and obj is self.table.viewport() and event.type() == QEvent.Resize):
            self._apply_responsive_columns(event.size().width())
        return super().eventFilter(obj, event)

    def _apply_responsive_columns(self, width: int):
        # thresholds (device-independent → scaled): drop a column below each.
        thresholds = [px(720), px(560), px(440)]   # Tags, Source, Created
        for col, thresh in zip(self._COL_HIDE_ORDER, thresholds):
            self.table.setColumnHidden(col, width < thresh)

    def _current_filters(self):
        types = None if self.type_filter.currentText() == "all types" else (self.type_filter.currentText(),)
        sources = None if self.source_filter.currentText() == "all sources" else (self.source_filter.currentText(),)
        return types, sources, self.imp_filter.value()

    def _refresh_table(self):
        types, sources, min_imp = self._current_filters()
        results = self.store.search(self._profile(), self.search_box.text(),
                                    types=types, sources=sources, min_importance=min_imp)
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        self._rows = []
        for e, score, reason in results:
            r = self.table.rowCount(); self.table.insertRow(r)
            cells = [e.type, e.source, _fmt_ts(e.created), str(e.importance),
                     ", ".join(e.tags), e.preview(120)]
            for c, val in enumerate(cells):
                it = QTableWidgetItem(val)
                if c == 0:
                    it.setData(Qt.UserRole, e.id)
                self.table.setItem(r, c, it)
            self._rows.append(e)
        self.table.setSortingEnabled(True)
        # Header resize modes (set in _build_explorer) size the columns; calling
        # resizeColumnsToContents() here would override Preview's stretch and push
        # the table into horizontal overflow. Just keep responsive hiding current.
        self._apply_responsive_columns(self.table.viewport().width())
        if hasattr(self, "result_count"):
            n = len(self._rows)
            q = self.search_box.text().strip()
            self.result_count.setText(f"{n} shown" + (f" · “{q}”" if q else ""))
        self._update_sel_count()

    def _on_row_selected(self):
        self._update_sel_count()
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if len(rows) != 1:
            return
        rid = self.table.item(rows[0].row(), 0).data(Qt.UserRole)
        e = self.store.get(self._profile(), rid)
        if not e:
            return
        self._sel = e
        self.editor.setPlainText(e.text)
        self.imp_slider.setValue(e.importance)
        self.tags_edit.setText(", ".join(e.tags))
        self.detail_meta.setText(
            f"id {e.id} · {e.type}{('/' + e.kind) if e.kind else ''} · source {e.source} · "
            f"created {_fmt_ts(e.created)} · accessed {e.access_count}× "
            f"(last {_fmt_ts(e.last_accessed)}) · {'PINNED' if e.pinned else 'unpinned'} · "
            f"{len(self.store.revisions(self._profile(), e.id))} revisions")

    def _selected_ids(self):
        ids = []
        for idx in (self.table.selectionModel().selectedRows() if self.table.selectionModel() else []):
            item = self.table.item(idx.row(), 0)
            if item:
                ids.append(item.data(Qt.UserRole))
        return ids

    def _update_sel_count(self):
        self.sel_count.setText(f"{len(self._selected_ids())} selected")

    def _save_entry(self):
        if not self._sel:
            return
        tags = [t.strip() for t in self.tags_edit.text().split(",") if t.strip()]
        self.store.update(self._profile(), self._sel.id, text=self.editor.toPlainText(),
                          importance=self.imp_slider.value(), tags=tags)
        self._toast("Memory saved.")
        self._after_mutation()

    def _dup_entry(self):
        if not self._sel:
            return
        self.store.duplicate(self._profile(), self._sel.id)
        self._after_mutation()

    def _toggle_pin(self):
        """Pin = promote a session note to a durable fact; Unpin = demote a fact."""
        if not self._sel:
            return
        e = self._sel
        new_type = "session" if e.type == "fact" else "fact"
        if e.type == "summary":
            return
        self.store.add(self._profile(), e.text, etype=new_type, importance=e.importance,
                       source=e.source, tags=e.tags, kind=e.kind or "note")
        self.store.delete(self._profile(), [e.id], soft=True)
        self.store.log(self._profile(), "pinned" if new_type == "fact" else "unpinned", {"id": e.id})
        self._after_mutation()

    def _move_menu(self):
        if not self._sel:
            return
        menu = QMenu(self)
        for p in self.store.profiles():
            if p == self._profile():
                continue
            menu.addAction(p, lambda _=False, dst=p: self._do_move(dst))
        menu.exec_(QCursor.pos())

    def _do_move(self, dst):
        self.store.move(self._profile(), self._sel.id, dst)
        self._after_mutation()

    def _revisions_dialog(self):
        if not self._sel:
            return
        revs = self.store.revisions(self._profile(), self._sel.id)
        dlg = QDialog(self); dlg.setWindowTitle("Revision history"); _fit_dialog(dlg, px(560), px(420))
        v = QVBoxLayout(dlg)
        lst = QListWidget()
        for i, r in enumerate(revs):
            lst.addItem(f"{i}: {_fmt_ts(r['ts'])} — {(r['after'] or '')[:60]}")
        v.addWidget(lst, 1)
        diff = QPlainTextEdit(); diff.setReadOnly(True); v.addWidget(diff, 1)

        def show(row):
            if 0 <= row < len(revs):
                r = revs[row]
                diff.setPlainText(f"— BEFORE —\n{r['before']}\n\n— AFTER —\n{r['after']}")
        lst.currentRowChanged.connect(show)
        brow = QHBoxLayout()
        revert = QPushButton("Revert to selected"); revert.clicked.connect(
            lambda: (self.store.revert(self._profile(), self._sel.id, lst.currentRow()),
                     dlg.accept(), self._after_mutation()))
        close = QPushButton("Close"); close.setObjectName("ghost"); close.clicked.connect(dlg.reject)
        brow.addWidget(revert); brow.addStretch(1); brow.addWidget(close)
        v.addLayout(brow)
        dlg.exec_()

    def _add_dialog(self):
        dlg = QDialog(self); dlg.setWindowTitle("Add memory"); _fit_dialog(dlg, px(520), px(380))
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel("Content"))
        content = QPlainTextEdit(); v.addWidget(content, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel("Type"))
        tcombo = QComboBox(); tcombo.addItems(["fact", "session", "summary"]); row.addWidget(tcombo)
        row.addWidget(QLabel("Importance"))
        imp = QSpinBox(); imp.setRange(0, 100); imp.setValue(60); row.addWidget(imp)
        v.addLayout(row)
        tags = QLineEdit(); tags.setPlaceholderText("tags, comma, separated"); v.addWidget(tags)
        prow = QHBoxLayout()
        prow.addWidget(QLabel("Profile"))
        pcombo = QComboBox(); pcombo.addItems(self.store.profiles())
        pcombo.setCurrentText(self._profile()); prow.addWidget(pcombo, 1)
        v.addLayout(prow)
        brow = QHBoxLayout()
        add = QPushButton("Add"); add_pin = QPushButton("Add & pin (fact)")
        cancel = QPushButton("Cancel"); cancel.setObjectName("ghost")
        brow.addWidget(add); brow.addWidget(add_pin); brow.addStretch(1); brow.addWidget(cancel)
        v.addLayout(brow)

        def do(as_fact):
            text = content.toPlainText().strip()
            if not text:
                return
            tg = [t.strip() for t in tags.text().split(",") if t.strip()]
            etype = "fact" if as_fact else tcombo.currentText()
            self.store.add(pcombo.currentText(), text, etype=etype, importance=imp.value(),
                           source="manual", tags=tg)
            dlg.accept()
            self.profile_combo.setCurrentText(pcombo.currentText())
            self._after_mutation()
        add.clicked.connect(lambda: do(False))
        add_pin.clicked.connect(lambda: do(True))
        cancel.clicked.connect(dlg.reject)
        dlg.exec_()

    def _delete_selected(self):
        ids = self._selected_ids() or ([self._sel.id] if self._sel else [])
        if not ids:
            return
        if not self._confirm("Delete memories",
                             f"Move {len(ids)} memitem(s) to Trash? They can be restored "
                             "from Explorer ▸ Trash. Pinned facts included if selected."):
            return
        self.store.delete(self._profile(), ids, soft=True)
        self._sel = None
        self._after_mutation()

    def _delete_matching(self):
        types, sources, min_imp = self._current_filters()
        results = self.store.search(self._profile(), self.search_box.text(),
                                    types=types, sources=sources, min_importance=min_imp)
        ids = [e.id for e, _, _ in results]
        if not ids:
            return
        if not self._confirm("Delete all matching",
                             f"This will move ALL {len(ids)} memories matching the current "
                             f"search/filter to Trash (profile '{self._profile()}'). Continue?"):
            return
        self.store.delete(self._profile(), ids, soft=True)
        self._after_mutation()

    def _trash_dialog(self):
        prof = self._profile()
        trash = self.store.trash(prof)
        dlg = QDialog(self); dlg.setWindowTitle(f"Trash — {prof}"); _fit_dialog(dlg, px(640), px(460))
        v = QVBoxLayout(dlg)
        lst = QListWidget(); lst.setSelectionMode(QAbstractItemView.ExtendedSelection)
        for r in trash:
            it = QListWidgetItem(f"[{r.get('type')}] {(r.get('text') or '')[:80]}  "
                                 f"(deleted {_fmt_ts(r.get('deleted_ts'))})")
            it.setData(Qt.UserRole, r.get("id"))
            lst.addItem(it)
        v.addWidget(lst, 1)
        brow = QHBoxLayout()
        restore = QPushButton("Restore selected")
        purge = QPushButton("Permanently delete selected"); purge.setObjectName("rec")
        empty = QPushButton("Empty trash"); empty.setObjectName("rec")
        close = QPushButton("Close"); close.setObjectName("ghost")
        for b in (restore, purge, empty): brow.addWidget(b)
        brow.addStretch(1); brow.addWidget(close)
        v.addLayout(brow)

        def sel_ids():
            return [i.data(Qt.UserRole) for i in lst.selectedItems()]

        def do_restore():
            self.store.restore(prof, sel_ids()); dlg.accept(); self._after_mutation()

        def do_purge():
            ids = set(sel_ids())
            remaining = [r for r in self.store.trash(prof) if r.get("id") not in ids]
            from memory_center import _atomic_write_json, TRASH_FILE
            _atomic_write_json(self.store.profile_dir(prof) / TRASH_FILE, remaining)
            dlg.accept(); self._after_mutation()

        def do_empty():
            if self._confirm("Empty trash", f"Permanently delete all {len(trash)} trashed items?"):
                self.store.empty_trash(prof); dlg.accept(); self._after_mutation()
        restore.clicked.connect(do_restore); purge.clicked.connect(do_purge)
        empty.clicked.connect(do_empty); close.clicked.connect(dlg.reject)
        dlg.exec_()

    # ---- Profiles ----------------------------------------------------------

    def _make_active(self):
        prof = self._selected_profile_name() or self._profile()
        if hasattr(self.host, "_memory_center_set_profile"):
            self.host._memory_center_set_profile(prof)
        self.refresh_all()

    # ---- Compaction (review before commit) --------------------------------

    # ---- Timeline (audit) --------------------------------------------------

    # ---- Diagnostics -------------------------------------------------------

    # ---- Backup ------------------------------------------------------------

    # ---- misc actions ------------------------------------------------------

    # ---- master refresh ----------------------------------------------------

    def refresh_all(self):
        # The ENTIRE body is fenced: refresh_all runs from several slots
        # (profile combo change, sub-tab change, _on_runtime_ready). An exception
        # escaping any of them aborts the process via PyQt (0xC0000409), so nothing
        # here — including the profile-combo / active-badge setup — may be left
        # outside the guard.
        try:
            # keep the profile combo populated + selection stable
            cur = self.profile_combo.currentText()
            profiles = self.store.profiles() or ["default"]
            active = self._active_profile()
            self.profile_combo.blockSignals(True)
            self.profile_combo.clear(); self.profile_combo.addItems(profiles)
            if cur in profiles:
                self.profile_combo.setCurrentText(cur)
            elif active in profiles:
                self.profile_combo.setCurrentText(active)
            self.profile_combo.blockSignals(False)
            self.active_badge.setText(f"active: {active}")
            self._refresh_overview()
            self._refresh_table()
            self._refresh_profiles()
            self._refresh_compaction()
            self._refresh_timeline()
        except Exception:
            logger.exception("Memory Center refresh error")

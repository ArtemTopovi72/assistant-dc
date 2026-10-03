"""Stress tab: per-word Russian stress overrides for the TTS voice.

Small and completely self-contained — one class, no workers, no shared state.
It edits the override table that stress.py consults, so a word the accentor
gets wrong can be corrected once instead of being re-heard wrong forever.

Its closure is a single symbol and no test patches anything in it, which is
why this extraction needed no seam repair.
"""
from PyQt5.QtWidgets import (QAbstractItemView, QHBoxLayout, QHeaderView,
                             QLabel, QLineEdit, QMessageBox, QPushButton,
                             QTableWidget, QTableWidgetItem, QVBoxLayout,
                             QWidget)

from ui_scale import px
from gui_common import MUTED, _section

import logging
logger = logging.getLogger("assistant.gui")


class StressTab(QWidget):
    """Manual per-word stress overrides for TTS edge cases.

    Each entry is a stressed spelling with a single '+' before the stressed vowel
    (e.g. "звон+ит"). Overrides are applied AFTER the automatic accentor at
    synthesis time, so a hand-set mark always wins. Edits hot-reload — no restart.
    """

    def __init__(self, host):
        super().__init__()
        self.host = host
        from stress_overrides import get_overrides
        self._ov = get_overrides()

        root = QVBoxLayout(self)
        root.setContentsMargins(px(6), px(6), px(6), px(6))
        root.setSpacing(px(8))

        root.addWidget(_section("Manual stress overrides"))
        help_lbl = QLabel(
            "Put <b>+</b> right before the stressed vowel, e.g. <code>звон+ит</code>, "
            "<code>фен+омен</code>, <code>уник+ум</code>. These win over the automatic "
            "accentor. Matching ignores case and is per word.")
        help_lbl.setWordWrap(True); help_lbl.setStyleSheet(f"color:{MUTED};")
        root.addWidget(help_lbl)

        # ---- add / update row ----
        addrow = QHBoxLayout()
        self.add_in = QLineEdit()
        self.add_in.setPlaceholderText("Stressed word, e.g.  звон+ит")
        self.add_in.returnPressed.connect(self._add)
        add_btn = QPushButton("➕  Add / Update"); add_btn.clicked.connect(self._add)
        addrow.addWidget(self.add_in, 1); addrow.addWidget(add_btn)
        root.addLayout(addrow)

        # ---- table ----
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Word", "Stressed (+)"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.itemDoubleClicked.connect(self._edit_selected)
        root.addWidget(self.table, 1)

        btnrow = QHBoxLayout()
        rm_btn = QPushButton("✕  Remove selected"); rm_btn.setObjectName("ghost")
        rm_btn.clicked.connect(self._remove_selected)
        rel_btn = QPushButton("⟳  Reload"); rel_btn.setObjectName("ghost")
        rel_btn.clicked.connect(self._reload)
        self.count_lbl = QLabel(""); self.count_lbl.setObjectName("chip")
        btnrow.addWidget(rm_btn); btnrow.addWidget(rel_btn)
        btnrow.addStretch(1); btnrow.addWidget(self.count_lbl)
        root.addLayout(btnrow)

        # ---- live preview ----
        root.addWidget(_section("Preview"))
        prow = QHBoxLayout()
        self.preview_in = QLineEdit()
        self.preview_in.setPlaceholderText("Type a phrase to see how it gets stressed…")
        self.preview_in.returnPressed.connect(self._preview)
        pv_btn = QPushButton("Preview"); pv_btn.clicked.connect(self._preview)
        prow.addWidget(self.preview_in, 1); prow.addWidget(pv_btn)
        root.addLayout(prow)
        self.preview_out = QLineEdit(); self.preview_out.setReadOnly(True)
        self.preview_out.setPlaceholderText("Result appears here.")
        root.addWidget(self.preview_out)

        self._refresh()

    # ----- actions -----------------------------------------------------------
    def _add(self):
        raw = self.add_in.text().strip()
        if not raw:
            return
        if "+" not in raw:
            QMessageBox.warning(self, "No stress mark",
                                "Add a '+' immediately before the stressed vowel, e.g. звон+ит.")
            return
        if not self._ov.upsert(raw):
            QMessageBox.warning(self, "Invalid entry",
                                "Could not parse that. Use letters with one '+' before a vowel.")
            return
        self.add_in.clear()
        self._refresh()

    def _edit_selected(self, item):
        row = item.row()
        cell = self.table.item(row, 1)
        if cell:
            self.add_in.setText(cell.text())
            self.add_in.setFocus()

    def _remove_selected(self):
        rows = sorted({i.row() for i in self.table.selectedItems()})
        if not rows:
            return
        for r in rows:
            w = self.table.item(r, 0)
            if w:
                self._ov.remove(w.text())
        self._refresh()

    def _reload(self):
        self._ov.reload()
        self._refresh()

    def _refresh(self):
        pairs = self._ov.pairs()
        self.table.setRowCount(0)
        for b, s in pairs:
            r = self.table.rowCount(); self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(b))
            self.table.setItem(r, 1, QTableWidgetItem(s))
        self.count_lbl.setText(f"{len(pairs)} override{'' if len(pairs) == 1 else 's'}")

    def _preview(self):
        txt = self.preview_in.text().strip()
        if not txt:
            self.preview_out.clear(); return
        ctx = getattr(self.host, "ctx", None)
        try:
            if ctx is not None and getattr(ctx.models, "accentor_loaded", False):
                from audio import stress_plus
                res = stress_plus(ctx, txt)          # full pipeline: auto + overrides
            else:
                res = self._ov.apply(txt)            # overrides only (accentor not loaded)
        except Exception as exc:
            res = f"(preview error: {exc})"
        self.preview_out.setText(res)

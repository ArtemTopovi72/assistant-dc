"""Memory Center: compaction, timeline, diagnostics and backup.

Split out of gui_memory_tab.py as a mixin -- MemoryCenterTab is one class, so
these need self and cannot become free functions.

Requires from the host: .store, .host, ._profile(), ._active_profile(),
._confirm(), ._toast(), ._after_mutation(), .refresh_all(), plus the widgets
each _build_* creates. The compaction worker lifecycle (._compact_worker,
._reset_compact_worker) stays with the page that owns it so teardown still has
exactly one owner.
"""
import json
import os
from pathlib import Path
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView, QComboBox, QFileDialog, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMenu, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from ui_scale import px, scale_style as _ss
from gui_common import MUTED, _FlowWidget, _card, _flow, _fmt_bytes, _fmt_ts, _section, open_in_os
import logging

logger = logging.getLogger("assistant.gui")


class MemoryMaintenanceMixin:
    """The compaction / timeline / diagnostics / backup pages."""

    # ---- Compaction (review before commit) --------------------------------
    def _build_compaction(self):
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(px(8), px(8), px(8), px(8)); lay.setSpacing(px(6))
        note = QLabel("Compaction summarises session memory into one summary. Review the "
                      "source memories and the generated summary BEFORE it is applied — "
                      "sources are moved to Trash (recoverable), never silently lost. "
                      "Pinned facts are never touched.")
        note.setWordWrap(True); note.setStyleSheet(_ss(f"color:{MUTED}; font-size:12px;"))
        lay.addWidget(note)
        self.compact_sources = QPlainTextEdit(); self.compact_sources.setReadOnly(True)
        lay.addWidget(QLabel("Source memories that will be compacted:"))
        lay.addWidget(self.compact_sources, 1)
        lay.addWidget(QLabel("Generated summary (editable before approval):"))
        self.compact_summary = QPlainTextEdit()
        lay.addWidget(self.compact_summary, 1)
        row = QHBoxLayout()
        self.gen_btn = QPushButton("Generate preview"); self.gen_btn.clicked.connect(self._gen_compaction)
        self.approve_btn = QPushButton("Approve & apply"); self.approve_btn.clicked.connect(self._approve_compaction)
        self.approve_btn.setEnabled(False)
        self.reject_btn = QPushButton("Reject"); self.reject_btn.setObjectName("ghost")
        self.reject_btn.clicked.connect(self._reject_compaction)
        row.addWidget(self.gen_btn); row.addWidget(self.approve_btn); row.addWidget(self.reject_btn)
        row.addStretch(1)
        lay.addLayout(row)
        return w

    def _refresh_compaction(self):
        srcs = self.store.compaction_sources(self._profile())
        self.compact_sources.setPlainText(
            "\n".join(f"• [{e.kind or 'note'}] {e.text}" for e in srcs) or "(no session memory to compact)")

    def _gen_compaction(self):
        prof = self._profile()
        if prof != self._active_profile():
            QMessageBox.information(self, "Make active first",
                                    "Compaction runs the LLM on the active profile. Click "
                                    "'Make Active' for this profile first.")
            return
        srcs = self.store.compaction_sources(prof)
        if not srcs:
            QMessageBox.information(self, "Nothing to compact", "No session memory in this profile.")
            return
        if self._compact_worker is not None:
            return
        if getattr(self.host, "ctx", None) is None:
            QMessageBox.warning(self, "Not ready", "The model is still loading. Try again in a moment.")
            return
        self.gen_btn.setEnabled(False); self._toast("Generating compaction preview…")
        # Imported at call time, not at module scope: gui_memory_tab imports this
        # mixin, so a top-level import here would be a cycle. Going through the
        # module also guarantees ONE worker class object, never a second copy.
        import gui_memory_tab as _tab
        self._compact_worker = _tab.CompactMemoryWorker(self.host.ctx)
        self._compact_worker.done.connect(self._on_preview_ready)
        self._compact_worker.failed.connect(lambda m: (self._toast(f"Compaction failed: {m}"),
                                                       self._reset_compact_worker()))
        self._compact_worker.finished.connect(self._reset_compact_worker)
        self._compact_worker.start()

    def _reset_compact_worker(self):
        self._compact_worker = None
        self.gen_btn.setEnabled(True)

    def _on_preview_ready(self, summary: str):
        self._compact_preview = summary
        self.compact_summary.setPlainText(summary or "")
        self.approve_btn.setEnabled(bool(summary))
        self._toast("Preview ready — review and approve or reject.")

    def _approve_compaction(self):
        summary = self.compact_summary.toPlainText().strip()
        if not summary:
            return
        if not self._confirm("Apply compaction",
                             "Replace the session memories with this summary? The originals "
                             "move to Trash (recoverable). Pinned facts are untouched."):
            return
        self.store.commit_compaction(self._profile(), summary, remove_sources=True, archive=True)
        self.approve_btn.setEnabled(False); self._compact_preview = None
        self._after_mutation()
        self._toast("Compaction applied.")

    def _reject_compaction(self):
        self.compact_summary.clear(); self.approve_btn.setEnabled(False); self._compact_preview = None

    # ---- Timeline (audit) --------------------------------------------------
    def _build_timeline(self):
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(px(8), px(8), px(8), px(8)); lay.setSpacing(px(6))
        self.audit_table = QTableWidget(0, 3)
        self.audit_table.setHorizontalHeaderLabels(["Time", "Event", "Details"])
        self.audit_table.horizontalHeader().setStretchLastSection(True)
        self.audit_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.audit_table.verticalHeader().setVisible(False)
        # fixed, not ResizeToContents: that re-measures every row on each insert (22 s GUI stall on the user list)
        self.audit_table.verticalHeader().setDefaultSectionSize(px(30))
        lay.addWidget(self.audit_table, 1)
        return w

    def _refresh_timeline(self):
        events = self.store.audit(self._profile())
        self.audit_table.setRowCount(0)
        for ev in events:
            r = self.audit_table.rowCount(); self.audit_table.insertRow(r)
            details = ", ".join(f"{k}={v}" for k, v in ev.items() if k not in ("ts", "event"))
            for c, val in enumerate([_fmt_ts(ev.get("ts")), ev.get("event", ""), details]):
                self.audit_table.setItem(r, c, QTableWidgetItem(str(val)))
        self.audit_table.resizeColumnsToContents()

    # ---- Diagnostics -------------------------------------------------------
    def _build_diagnostics(self):
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(px(8), px(8), px(8), px(8)); lay.setSpacing(px(6))
        note = QLabel("Trace how a message would pull memory: rank → select under the token "
                      "budget → inject. Pinned facts are always injected; session/summary "
                      "compete for the remaining budget.")
        note.setWordWrap(True); note.setStyleSheet(_ss(f"color:{MUTED}; font-size:12px;"))
        lay.addWidget(note)
        row = QHBoxLayout()
        self.diag_query = QLineEdit(); self.diag_query.setPlaceholderText("Simulate a user message…")
        row.addWidget(self.diag_query, 1)
        row.addWidget(QLabel("Token budget"))
        self.diag_budget = QSpinBox(); self.diag_budget.setRange(100, 8000); self.diag_budget.setValue(1500)
        row.addWidget(self.diag_budget)
        run = QPushButton("Run"); run.clicked.connect(self._run_diag); row.addWidget(run)
        lay.addLayout(row)
        self.diag_out = QPlainTextEdit(); self.diag_out.setReadOnly(True)
        lay.addWidget(self.diag_out, 1)
        return w

    def _run_diag(self):
        d = self.store.diagnostics(self._profile(), query=self.diag_query.text(),
                                   token_budget=self.diag_budget.value())
        lines = [
            f"Profile: {self._profile()}    retrieval: {d['retrieval_mode']}"
            f"    semantic: {'on' if d['semantic_enabled'] else 'off (keyword)'}",
            f"Token budget: {d['token_budget']}    used: {d['tokens_used']}",
            f"Pinned facts always injected: {d['facts_injected']} ({d['fact_tokens']} tok)",
            f"Candidates ranked: {d['candidates']}    selected: {len(d['selected'])}    "
            f"dropped (budget): {len(d['dropped_budget'])}",
            "",
            "── SELECTED (injected) ──",
        ]
        for e, score, reason, tok in d["selected"]:
            lines.append(f"  ✓ [{e.type}] imp{e.importance} {tok}tok  {e.preview(60)}"
                         + (f"   ⟵ {reason}" if reason else ""))
        lines.append("")
        lines.append("── DROPPED (over budget) ──")
        for e, score, reason, tok in d["dropped_budget"]:
            lines.append(f"  ✗ [{e.type}] imp{e.importance} {tok}tok  {e.preview(60)}")
        lines.append("")
        lines.append("── Per-profile disk ──")
        for p, b in d["per_profile_bytes"].items():
            lines.append(f"  {p}: {_fmt_bytes(b)}")
        self.diag_out.setPlainText("\n".join(lines))

    # ---- Backup ------------------------------------------------------------
    def _build_backup(self):
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(px(8), px(8), px(8), px(8)); lay.setSpacing(px(8))
        lay.addWidget(_section("Export"))
        erow = QHBoxLayout()
        for label, fmt in [("Export profile (JSON)", "json"),
                           ("Export profile (Markdown)", "md"),
                           ("Export ALL (ZIP)", "zip")]:
            b = QPushButton(label); b.setObjectName("ghost")
            b.clicked.connect(lambda _=False, f=fmt: self._export(f))
            erow.addWidget(b)
        lay.addLayout(erow)
        lay.addWidget(_section("Import"))
        irow = QHBoxLayout()
        imp = QPushButton("Import backup (preview first)…"); imp.clicked.connect(self._import)
        irow.addWidget(imp); irow.addStretch(1)
        lay.addLayout(irow)
        self.backup_log = QPlainTextEdit(); self.backup_log.setReadOnly(True)
        lay.addWidget(self.backup_log, 1)
        return w

    def _export(self, fmt):
        prof = self._profile()
        if fmt == "json":
            path, _ = QFileDialog.getSaveFileName(self, "Export JSON", f"{prof}_memory.json", "JSON (*.json)")
            if path:
                Path(path).write_text(json.dumps(self.store.export_json([prof]), ensure_ascii=False, indent=2), encoding="utf-8")
                self.store.log(prof, "exported", {"fmt": "json"}); self.backup_log.appendPlainText(f"Exported JSON → {path}")
        elif fmt == "md":
            path, _ = QFileDialog.getSaveFileName(self, "Export Markdown", f"{prof}_memory.md", "Markdown (*.md)")
            if path:
                Path(path).write_text(self.store.export_markdown([prof]), encoding="utf-8")
                self.store.log(prof, "exported", {"fmt": "md"}); self.backup_log.appendPlainText(f"Exported Markdown → {path}")
        else:
            path, _ = QFileDialog.getSaveFileName(self, "Export ZIP (all profiles)", "memory_backup.zip", "ZIP (*.zip)")
            if path:
                self.store.export_zip(self.store.profiles(), Path(path))
                self.backup_log.appendPlainText(f"Exported ALL profiles → {path}")

    def _import(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import backup", "", "Backups (*.json *.zip)")
        if not path:
            return
        try:
            counts, data = self.store.preview_import(Path(path))
        except Exception as e:
            QMessageBox.warning(self, "Invalid backup", str(e)); return
        preview = "\n".join(f"  {p}: {n} entries" for p, n in counts.items())
        if not self._confirm("Import preview",
                             f"This backup contains:\n\n{preview}\n\nImport (merge into the "
                             "matching profiles)? Existing memories are kept; imported ones "
                             "are added with source='imported'."):
            return
        n = self.store.commit_import(data, merge=True)
        self.backup_log.appendPlainText(f"Imported {n} entries from {path}")
        self._after_mutation()

    # ---- misc actions ------------------------------------------------------
    def _rebuild_index(self):
        # No persistent index yet (keyword search is live); reload + re-mint ids/meta.
        self.store.load(self._profile())
        self._toast("Index rebuilt (entries re-scanned).")
        self.refresh_all()

    def _open_folder(self):
        d = self.store.profile_dir(self._profile())
        d.mkdir(parents=True, exist_ok=True)
        try:
            open_in_os(d)
        except Exception as e:
            self._toast(f"Could not open folder: {e}")

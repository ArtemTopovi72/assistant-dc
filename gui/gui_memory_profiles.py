"""Memory Center: the profiles page.

Split out of gui_memory_tab.py as a mixin, because MemoryCenterTab is one class
-- these methods cannot be lifted as free functions, they need self.

What it requires from the host: .store, .host, ._active_profile(), ._confirm(),
._toast(), .refresh_all(), and the two widgets it builds (.prof_list,
.profile_combo). Nothing else, which is what makes the seam honest rather than
cosmetic.
"""
import json
import os
from pathlib import Path
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox, QFileDialog, QGridLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMenu, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from ui_scale import px
from gui_common import MUTED, _FlowWidget, _card, _flow, _fmt_bytes, _fmt_ts, _section
import logging

logger = logging.getLogger("assistant.gui")


class MemoryProfilesMixin:
    """Create / rename / duplicate / delete memory profiles."""

    # ---- Profiles ----------------------------------------------------------
    def _build_profiles(self):
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(px(8), px(8), px(8), px(8)); lay.setSpacing(px(6))
        self.prof_list = QListWidget()
        lay.addWidget(self.prof_list, 1)
        grid = QGridLayout()
        actions = [("Create", self._prof_create), ("Rename", self._prof_rename),
                   ("Duplicate", self._prof_duplicate), ("Make active", self._make_active),
                   ("Delete", self._prof_delete), ("Create suggested", self._prof_suggested)]
        for i, (label, slot) in enumerate(actions):
            b = QPushButton(label); b.setObjectName("ghost" if label not in ("Delete",) else "rec")
            b.clicked.connect(slot)
            grid.addWidget(b, i // 3, i % 3)
        lay.addLayout(grid)
        return w

    def _refresh_profiles(self):
        self.prof_list.clear()
        active = self._active_profile()
        for p in self.store.profiles():
            ov = self.store.overview(p)
            mark = "  ● ACTIVE" if p == active else ""
            self.prof_list.addItem(f"{p}{mark}   —   {ov['total']} items, "
                                   f"{ov['facts']} facts, {_fmt_bytes(ov['disk_bytes'])}")

    def _selected_profile_name(self):
        it = self.prof_list.currentItem()
        if not it:
            return None
        return it.text().split("  ●")[0].split("   —")[0].strip()

    def _prof_create(self):
        name, ok = QInputDialog.getText(self, "Create profile", "Profile name:")
        if ok and name.strip():
            try:
                self.store.create_profile(name.strip()); self.refresh_all()
                self.profile_combo.setCurrentText(name.strip())
            except ValueError as e:
                QMessageBox.warning(self, "Invalid name", str(e))

    def _prof_suggested(self):
        created = []
        for p in self._SUGGESTED:
            try:
                if self.store.create_profile(p):
                    created.append(p)
            except Exception:
                pass
        self.refresh_all()
        self._toast(f"Created: {', '.join(created) or 'none (already exist)'}")

    def _prof_rename(self):
        src = self._selected_profile_name()
        if not src:
            return
        name, ok = QInputDialog.getText(self, "Rename profile", f"Rename '{src}' to:")
        if not (ok and name.strip()):
            return
        name = name.strip()
        ctx = getattr(self.host, "ctx", None)
        renaming_active = (src == self._active_profile())
        if renaming_active and ctx is not None:
            # The live session points at the OLD directory. Flush it first, or the
            # in-memory items are lost; without re-pointing afterwards the next
            # save_memory() recreates the old directory and the session silently
            # forks into a resurrected profile under the previous name.
            try:
                ctx.save_memory(ctx.active_memory_dir)
            except Exception:
                logger.exception("could not flush memory before renaming the active profile")
        if not self.store.rename_profile(src, name):
            QMessageBox.warning(self, "Rename failed",
                                f"Could not rename '{src}' to '{name}' — "
                                "the name may already be in use.")
            return
        if renaming_active and ctx is not None:
            ctx.active_memory_dir = self.store.profile_dir(name)
            if hasattr(self.host, "_refresh_mem_combo"):
                try:
                    self.host._refresh_mem_combo()
                except Exception:
                    logger.exception("could not refresh the profile combo after rename")
        self.refresh_all()

    def _prof_duplicate(self):
        src = self._selected_profile_name()
        if not src:
            return
        name, ok = QInputDialog.getText(self, "Duplicate profile", f"Copy '{src}' to:")
        if ok and name.strip():
            self.store.duplicate_profile(src, name.strip()); self.refresh_all()

    def _prof_delete(self):
        src = self._selected_profile_name()
        if not src:
            return
        if src == self._active_profile():
            QMessageBox.warning(self, "Cannot delete", "This profile is currently active. "
                                "Switch to another profile first.")
            return
        if not self._confirm("Delete profile",
                             f"Permanently delete profile '{src}' and ALL its memories? "
                             "This cannot be undone (export it first if unsure)."):
            return
        self.store.delete_profile(src); self.refresh_all()

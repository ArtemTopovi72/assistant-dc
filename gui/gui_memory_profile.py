"""Memory-profile management: list/switch/create profiles, save, and compact.

Split out of gui.py. Reaches MEMORY_DIR and CompactMemoryWorker back through
`import gui as _g` at call time rather than by value: both are patched
directly ON the gui module by test suites (MEMORY_DIR by test_gui_adversarial;
CompactMemoryWorker by test_gui_supplement{5,10,12}, test_gui_window,
and constructed directly as gui.CompactMemoryWorker(ctx) by test_gui_workers).
Binding either by value here would have frozen them at import time and made
every one of those patches a silent no-op — the same dead-seam failure fixed
the same way throughout this refactor (tools.py's render seam, gui.py's
research tab, graph.py's personality_node, gui_image_fix.py's worker seams):
reach a dependency through the module that owns it, never by value across
a split.
"""
from PyQt5.QtWidgets import QInputDialog


class MemoryProfileMixin:
    """List/switch/create memory profiles, and save/compact the active one."""

    # ---- memory profiles ----
    def _list_memory_profiles(self) -> list:
        """Every profile directory under MEMORY_DIR.

        A profile used to count only once it held one of the memory JSON files, so
        a freshly created (still empty) profile was invisible: _new_memory_profile
        made it active but findText() could not locate it, leaving the combo
        showing the OLD name while the session wrote to the new one — and because
        the combo was already on that old name, selecting it again fired no signal,
        so there was no way back.
        """
        import gui as _g
        profiles = {"default"}
        if _g.MEMORY_DIR.exists():
            for d in _g.MEMORY_DIR.iterdir():
                if d.is_dir() and not d.name.startswith("."):
                    profiles.add(d.name)
        return sorted(profiles)

    def _refresh_mem_combo(self):
        self.mem_combo.blockSignals(True)
        self.mem_combo.clear()
        for p in self._list_memory_profiles():
            self.mem_combo.addItem(p)
        active = self.ctx.active_memory_dir.name if self.ctx else "default"
        idx = self.mem_combo.findText(active)
        self.mem_combo.setCurrentIndex(max(0, idx))
        self.mem_combo.blockSignals(False)

    def _switch_memory_profile(self, profile_name: str):
        import gui as _g
        if self.ctx is None or not profile_name:
            return
        current = self.ctx.active_memory_dir.name
        if profile_name == current:
            return
        if self._busy():
            # Switching mid-turn would mix the running turn's memories into the
            # other profile. Refuse and snap the combo back to the active profile.
            self._add_system('Wait for the current operation to finish before switching the memory profile.')
            self.mem_combo.blockSignals(True)
            idx = self.mem_combo.findText(current)
            if idx >= 0:
                self.mem_combo.setCurrentIndex(idx)
            self.mem_combo.blockSignals(False)
            return
        self.ctx.save_memory(self.ctx.active_memory_dir)
        new_dir = _g.MEMORY_DIR / profile_name
        new_dir.mkdir(parents=True, exist_ok=True)
        self.ctx.active_memory_dir = new_dir
        with self.ctx.memory_lock:
            self.ctx.session_memory.clear()
        self.ctx.load_memory(new_dir)
        # Persist immediately (even when empty) so the profile has a session file
        # on disk — otherwise a fresh profile disappears from the list on restart.
        self.ctx.save_memory(new_dir)
        count = len(self.ctx.session_memory)
        # Re-sync the combo to whatever is now actually active. Callers reach this
        # method by several routes (combo change, Memory Center, new profile), and
        # only the combo route leaves the widget already correct.
        self._refresh_mem_combo()
        self._add_system(f"Memory profile switched to '{profile_name}' ({count} items).")
        self.system_info_tab.refresh()
        if hasattr(self, "memory_center_tab"):
            self.memory_center_tab.refresh_all()

    def _memory_center_set_profile(self, profile_name: str):
        """Make `profile_name` the live active profile (called from the Memory Center
        Profiles tab). Routes through the existing combo so all the busy-guards and
        persistence rules apply uniformly."""
        import gui as _g
        if not profile_name:
            return
        if self.ctx is None:
            self._add_system("Load the model first, then switch the active profile.")
            return
        (_g.MEMORY_DIR / profile_name).mkdir(parents=True, exist_ok=True)
        self._refresh_mem_combo()
        idx = self.mem_combo.findText(profile_name)
        if idx >= 0:
            self.mem_combo.setCurrentIndex(idx)  # fires _switch_memory_profile

    def _new_memory_profile(self):
        import gui as _g
        if self.ctx is None:
            return
        import re as _re
        name, ok = QInputDialog.getText(self, "New Memory Profile",
                                        "Profile name (letters, numbers, underscores):")
        if not ok or not name.strip():
            return
        name = _re.sub(r"[^a-zA-Z0-9_\-]", "", name.strip().replace(" ", "_"))
        if not name:
            return
        (_g.MEMORY_DIR / name).mkdir(parents=True, exist_ok=True)
        self._refresh_mem_combo()
        # Switch to the new profile
        self.mem_combo.blockSignals(True)
        idx = self.mem_combo.findText(name)
        if idx >= 0:
            self.mem_combo.setCurrentIndex(idx)
        self.mem_combo.blockSignals(False)
        self._switch_memory_profile(name)

    # ---- memory ----
    def _save_memory(self):
        if self.ctx is None:
            return
        self.ctx.save_memory(self.ctx.active_memory_dir)
        self._add_system(
            f"Memory saved ({len(self.ctx.session_memory)} items) → '{self.ctx.active_memory_dir.name}'."
        )
        self.system_info_tab.refresh()

    def _compact_memory(self):
        import gui as _g
        if self.ctx is None or self.compact_worker is not None:
            return
        if self._busy():
            # A running turn appends to session_memory while compaction would be
            # clearing it — the new items would be lost mid-write.
            self._add_system('Wait for the current operation to finish before compacting memory.')
            return
        self._add_system("Compacting memory…")
        self.compact_btn.setEnabled(False)
        self.compact_worker = _g.CompactMemoryWorker(self.ctx)
        self.compact_worker.done.connect(self._on_compact_done)
        self.compact_worker.failed.connect(self._on_compact_failed)
        self.compact_worker.finished.connect(self._on_compact_worker_finished)
        self.compact_worker.start()
        self._set_busy(True, "Compacting memory…")

    def _on_compact_done(self, summary: str):
        import gui as _g
        self.compact_btn.setEnabled(True)
        if not summary:
            self._add_system("Memory is empty — nothing to compact.")
            return
        import json as _json
        import time as _time
        profile_dir = self.ctx.active_memory_dir if self.ctx else _g.MEMORY_DIR / "default"
        try:
            profile_dir.mkdir(parents=True, exist_ok=True)
            summary_file = profile_dir / "summary.json"
            with open(summary_file, "w", encoding="utf-8") as f:
                _json.dump({"text": summary, "ts": _time.time()}, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            self._add_system(f"Compact done but save failed: {exc}")
            return
        with self.ctx.memory_lock:
            self.ctx.session_memory.clear()
            self.ctx.session_memory.append({"ts": _time.time(), "kind": "summary", "text": summary, "meta": {}})
        self.ctx.save_memory(profile_dir)
        self.system_info_tab.refresh()
        if hasattr(self, "memory_center_tab"):
            self.memory_center_tab.refresh_all()
        self._add_system(f"Memory compacted and saved → '{profile_dir.name}'.")

    def _on_compact_failed(self, msg: str):
        self.compact_btn.setEnabled(True)
        self._add_system(f"Compact failed: {msg}")

    def _on_compact_worker_finished(self):
        self.compact_worker = None
        self._set_busy(False)

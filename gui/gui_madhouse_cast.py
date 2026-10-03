"""The Madhouse cast: the asset pickers, and loading/saving a room of characters.

Split out of gui_madhouse_tab.py. A "character" is two files -- a reference WAV
for F5-TTS voice cloning and a personalities/*.txt -- plus a name, and this is
everything that turns those files into cast entries and cast entries back into
JSON. It is the half of the tab that never touches the transcript, the speech
queue or the auto-dialogue timer.

_rebuild_cast_views is the one method both halves care about: every mutation
here ends by calling it, and it is what repopulates the list, the grid, the
"speak as" combo and the status line in one pass.

MadhouseGrid is resolved through gui_madhouse_tab at CALL time for the reason
given in gui_madhouse_ui.py.
"""
import json
import os
from pathlib import Path

from PyQt5.QtWidgets import QFileDialog, QMessageBox

import logging
logger = logging.getLogger("assistant.gui")   # same channel as gui.py


class MadhouseCastMixin:
    """Cast assets, add/remove, load/save. Mixed into MadhouseTab, never alone."""

    # ----- existing assets ---------------------------------------------------
    def _project_dir(self):
        return Path(__file__).resolve().parents[1]

    def _voice_assets(self):
        """Reference WAVs shipped with the project (the *_short.wav cast clips)."""
        return sorted((p for p in self._project_dir().glob("*.wav") if not p.stem.endswith("_ref12")),
                      key=lambda p: p.name.lower())   # _ref12 = audio.trim_ref_to_cap cache

    def _personality_assets(self):
        return sorted((self._project_dir() / "personalities").glob("*.txt"),
                      key=lambda p: p.name.lower())

    def _refresh_asset_combos(self):
        for combo, items, default in (
                (self.voice_combo, self._voice_assets(), "default voice"),
                (self.pers_combo, self._personality_assets(), "default assistant")):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(default, "")                  # empty payload => fall back
            for p in items:
                combo.addItem(p.stem, str(p))
            combo.addItem("Browse…", "__browse__")
            combo.blockSignals(False)

    def _browse_into(self, combo, kind):
        """The 'Browse…' entry mirrors the Settings dialog's file pickers."""
        if combo.currentData() != "__browse__":
            return
        if kind == "voice":
            path, _ = QFileDialog.getOpenFileName(
                self, "Reference voice", str(self._project_dir()),
                "Audio (*.wav *.mp3 *.ogg *.m4a *.flac *.aac);;All files (*)")
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Personality file", str(self._project_dir() / "personalities"),
                "Text files (*.txt);;All files (*)")
        combo.blockSignals(True)
        if path:
            combo.insertItem(combo.count() - 1, os.path.basename(path), path)
            combo.setCurrentIndex(combo.count() - 2)
        else:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)

    def _read_personality(self, path):
        """Personality file -> system prompt, the same way Settings loads one."""
        if not path:
            return "A participant in a group conversation."
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return fh.read().strip() or "A participant in a group conversation."
        except Exception as exc:
            self._note(f"could not read personality {os.path.basename(path)}: {exc}")
            return "A participant in a group conversation."

    # ----- cast --------------------------------------------------------------
    def _add_character(self):
        name = self.name_in.text().strip()
        if not name:
            self.name_in.setFocus()
            return
        if any(c["name"].lower() == name.lower() for c in self.characters):
            QMessageBox.warning(self, "Duplicate name", f"{name} is already in the cast.")
            return
        voice = self.voice_combo.currentData() or ""
        pers = self.pers_combo.currentData() or ""
        if voice == "__browse__":
            voice = ""
        if pers == "__browse__":
            pers = ""
        self.characters.append({
            "id": f"c{len(self.characters) + 1}_{name.lower()}",
            "name": name,
            "voice": voice,
            "personality": pers,
            "prompt": self._read_personality(pers),
            "_voice_wav": self._resolve_voice(voice),
        })
        self.name_in.clear()
        self._rebuild_cast_views()

    def _add_character_from_files(self):
        """A character is just two files: a personality .txt and a voice clip."""
        pers, _ = QFileDialog.getOpenFileName(
            self, "Personality file (character description)",
            str(self._project_dir() / "personalities"), "Text files (*.txt);;All files (*)")
        if not pers:
            return
        voice, _ = QFileDialog.getOpenFileName(
            self, "Voice file (leave empty for the default voice)", str(self._project_dir()),
            "Audio (*.wav *.mp3 *.ogg *.m4a *.flac *.aac);;All files (*)")
        name = self.name_in.text().strip() or Path(pers).stem.replace("_", " ").title()
        if any(c["name"].lower() == name.lower() for c in self.characters):
            QMessageBox.warning(self, "Duplicate name",
                                f"{name} is already in the cast — type a different name first.")
            return
        self.characters.append({
            "id": f"c{len(self.characters) + 1}_{name.lower().replace(' ', '_')}",
            "name": name,
            "voice": voice or "",
            "personality": pers,
            "prompt": self._read_personality(pers),
            "_voice_wav": self._resolve_voice(voice),
        })
        self.name_in.clear()
        # Surface the two files in the combos too, so they can be reused for the next
        # character without browsing again.
        for combo, path in ((self.pers_combo, pers), (self.voice_combo, voice)):
            if path and combo.findData(path) < 0:
                combo.blockSignals(True)
                combo.insertItem(combo.count() - 1, os.path.basename(path), path)
                combo.blockSignals(False)
        self._rebuild_cast_views()

    def _remove_character(self):
        row = self.cast_list.currentRow()
        if not (0 <= row < len(self.characters)):
            return
        self.characters.pop(row)
        self._rebuild_cast_views()

    def _rebuild_cast_views(self):
        import gui_madhouse_tab as _mh          # MadhouseGrid, at call time
        MadhouseGrid = _mh.MadhouseGrid
        for i, c in enumerate(self.characters):
            c["_cast"] = self.characters
            # Auto-assign avatar if not already set (preserves click-cycled avatars)
            if not c.get("_avatar"):
                c["_avatar"] = MadhouseGrid._AVATARS[i % len(MadhouseGrid._AVATARS)]
        self.cast_list.clear()
        for c in self.characters:
            voice = os.path.basename(c["_voice_wav"]) if c["_voice_wav"] else "default voice"
            pers = Path(c["personality"]).stem if c.get("personality") else "default assistant"
            self.cast_list.addItem(
                f"{c.get('_avatar', '🧑')}  {c['name']}   ·   🔊 {voice}   ·   🎭 {pers}")
        self.grid.set_characters(self.characters)
        cur = self.who.currentData()
        self.who.clear()
        self.who.addItem(f"🙋  {self.user['name']}", self.USER_ID)   # always available
        for c in self.characters:
            self.who.addItem(c["name"], c["id"])
        idx = self.who.findData(cur)
        if idx >= 0:
            self.who.setCurrentIndex(idx)
        n = len(self.characters)
        self.cast_lbl.setText(f"{n} character{'' if n == 1 else 's'}" if n else "no characters")
        self._sync_controls()

    def _save_cast(self):
        if not self.characters:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save cast", str(self._project_dir() / "personalities" / "madhouse_cast.json"),
            "JSON (*.json)")
        if not path:
            return
        payload = [{k: c[k] for k in ("id", "name", "voice", "personality", "prompt")}
                   for c in self.characters]
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            self._note(f"cast saved to {os.path.basename(path)}")
        except Exception as exc:
            QMessageBox.warning(self, "Could not save", f"{exc}")

    def _resolve_voice(self, raw):
        """Character `voice` -> a usable reference WAV path, or None for the default."""
        raw = (raw or "").strip()
        if not raw:
            return None
        names = [raw] if raw.lower().endswith((".wav", ".mp3", ".ogg", ".m4a", ".flac", ".aac")) \
            else [raw, raw + ".wav"]                    # tolerate a bare stem ("DC_short")
        for name in names:
            for cand in (Path(name), Path.cwd() / name, self._project_dir() / name):
                if cand.exists():
                    return str(cand)
        return None

    def _load_characters(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load characters", "", "Characters (*.json *.txt);;All files (*)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            QMessageBox.warning(self, "Could not read file", f"{exc}")
            return
        if isinstance(data, dict):                    # tolerate {"characters": [...]}
            data = data.get("characters", data.get("cast", []))
        if not isinstance(data, list) or not data:
            QMessageBox.warning(self, "Bad format",
                                "Expected a JSON array of {id, name, voice, prompt}.")
            return

        cast, missing, voiceless = [], [], []
        for i, raw in enumerate(data):
            if not isinstance(raw, dict) or not str(raw.get("name", "")).strip():
                continue
            voice = str(raw.get("voice", "") or "")
            wav = self._resolve_voice(voice)
            if voice and not wav:
                missing.append(voice)
            elif not voice:
                voiceless.append(str(raw["name"]).strip())
            pers = str(raw.get("personality", "") or "")
            # An explicit `prompt` wins; otherwise read the personality file, so a
            # saved cast keeps working after the personality text is edited on disk.
            prompt = str(raw.get("prompt", "") or "").strip() or self._read_personality(pers)
            cast.append({
                "id": str(raw.get("id") or f"char{i + 1}"),
                "name": str(raw["name"]).strip(),
                "voice": voice,
                "personality": pers,
                "prompt": prompt,
                "_voice_wav": wav,                    # None => default assistant voice
            })
        if not cast:
            QMessageBox.warning(self, "Nothing loaded", "No entry had a usable `name`.")
            return

        # A new cast resets the room. Old lines are only worth keeping if the user
        # says so — the previous speakers may not exist any more.
        keep = False
        if self.messages:
            keep = QMessageBox.question(
                self, "Keep history?",
                "Loading a new cast. Keep the existing transcript?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) == QMessageBox.Yes
        self._stop_auto()
        self._stop_speech()
        if not keep:
            self.messages.clear()
            self.grid.clear_all()

        self.characters = cast
        self._rebuild_cast_views()
        if missing:
            self._note("voice file not found, using default: " + ", ".join(sorted(set(missing))))
        if voiceless:
            # The default voice is the assistant's own (Stepan_short.wav): a
            # character with no voice of its own sounds exactly like another
            # cast member, which is how «Сан speaks in Степан's voice» looked.
            self._note("no voice set, speaks in the assistant's voice: " + ", ".join(voiceless))

"""Mashup tab: pick two audio files, get the voice of one over the music of
the other.

Modeled on MusicTab -- a self-contained QWidget(host) plus one QThread that
does the slow call and reports back via signals. Two Demucs passes plus a
time-stretch is tens of seconds of GPU work, so none of it can touch the UI
thread.

Every decision that is not layout (which stems make an instrumental, how far a
tempo may be stretched, which direction a key moves) lives in mashup.py, so
this surface and the Telegram flow cannot disagree about what a mashup is.
"""
import os

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QVBoxLayout,
                             QWidget)

from audio import AudioPlayer
from ui_scale import px
from gui_common import MUTED, _section

import logging
logger = logging.getLogger("assistant.gui")   # same channel as gui.py

_AUDIO_FILTER = ("Audio (*.wav *.mp3 *.flac *.ogg *.m4a *.opus *.aac);;"
                 "All files (*)")


class MashupWorker(QThread):
    """Separate, match and mix -- all off the UI thread."""
    done = pyqtSignal(dict)
    failed = pyqtSignal(str)
    stage = pyqtSignal(str)

    def __init__(self, vocal_path: str, instr_path: str, *,
                 vocal_is_speech: bool = False, match_tempo: bool = True,
                 match_key: bool = True):
        super().__init__()
        # Snapshotted at construction, never read off the widgets in run():
        # run() is on the worker thread, and touching Qt widgets from there is
        # the unhandled-slot-exception path that aborts the process with no
        # traceback.
        self.vocal_path = vocal_path
        self.instr_path = instr_path
        self.vocal_is_speech = vocal_is_speech
        self.match_tempo = match_tempo
        self.match_key = match_key

    def run(self):
        try:
            import mashup
            out = mashup.default_out_path("gui")
            rep = mashup.make_mashup(
                self.vocal_path, self.instr_path, out,
                vocal_is_speech=self.vocal_is_speech,
                match_tempo=self.match_tempo, match_key=self.match_key,
                progress=self.stage.emit)
            self.done.emit(rep)
        except Exception as exc:
            # Covers mashup.MashupUnavailable (a RuntimeError carrying a
            # user-facing sentence) as well as anything else the pipeline
            # throws -- one status line, never a traceback in the UI.
            logger.exception("Mashup failed")
            self.failed.emit(str(exc))
        finally:
            # The separator holds GPU memory the next image or song render
            # needs; this machine shares 24GB across everything at once.
            try:
                import mashup
                mashup.release_separator()
            except Exception:
                pass


class MashupTab(QWidget):
    """Two files, a choice of which supplies the voice, and Build."""

    def __init__(self, host):
        super().__init__()
        self.host = host                  # AssistantWindow (unused, kept for parity)
        self.player = AudioPlayer()
        self.worker = None
        self._last_out = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(px(6), px(6), px(6), px(6))
        root.setSpacing(px(8))

        root.addWidget(_section("Mashup"))
        help_lbl = QLabel("Takes the singing from one track and puts it over "
                          "the music of another. The vocal is stretched onto "
                          "the second track's tempo and shifted into its key. "
                          "A spoken recording is used as-is, since stretching "
                          "speech onto a beat sounds seasick.")
        help_lbl.setWordWrap(True)
        help_lbl.setStyleSheet(f"color:{MUTED};")
        root.addWidget(help_lbl)

        self.vocal_in = self._file_row(root, "Voice from:")
        self.instr_in = self._file_row(root, "Music from:")

        opts = QHBoxLayout()
        self.kind = QComboBox()
        # The KEY rides along as item data -- the visible label is display text
        # and must never be what gets resolved.
        self.kind.addItem("Sung (a song)", False)
        self.kind.addItem("Spoken (a voice note)", True)
        self.kind.setToolTip("Spoken input skips tempo and key matching: "
                             "speech has no key, and stretching it onto a beat "
                             "makes it sound seasick.")
        opts.addWidget(QLabel("First track is:"))
        opts.addWidget(self.kind)
        self.tempo_chk = QCheckBox("Match tempo"); self.tempo_chk.setChecked(True)
        self.key_chk = QCheckBox("Match key"); self.key_chk.setChecked(True)
        opts.addWidget(self.tempo_chk)
        opts.addWidget(self.key_chk)
        opts.addStretch(1)
        self.build_btn = QPushButton("🎚  Build mashup")
        self.build_btn.clicked.connect(self._build)
        opts.addWidget(self.build_btn)
        root.addLayout(opts)

        self.status = QLabel("Idle.")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color:{MUTED};")
        root.addWidget(self.status)

        play_row = QHBoxLayout()
        self.play_btn = QPushButton("▶  Play"); self.play_btn.setObjectName("ghost")
        self.play_btn.clicked.connect(self._play_last)
        self.play_btn.setEnabled(False)
        play_row.addWidget(self.play_btn)
        play_row.addStretch(1)
        root.addLayout(play_row)
        root.addStretch(1)

    def _file_row(self, root, label: str) -> QLineEdit:
        row = QHBoxLayout()
        field = QLineEdit()
        field.setPlaceholderText("Pick an audio file…")
        btn = QPushButton("Browse…")
        btn.setObjectName("ghost")
        btn.clicked.connect(lambda _c, f=field: self._browse(f))
        row.addWidget(QLabel(label))
        row.addWidget(field, 1)
        row.addWidget(btn)
        root.addLayout(row)
        return field

    # ----- actions -----------------------------------------------------------
    def _browse(self, field: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(self, "Choose audio", "", _AUDIO_FILTER)
        if path:
            field.setText(path)

    def _busy(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def _build(self):
        vocal = self.vocal_in.text().strip()
        instr = self.instr_in.text().strip()
        # Checked here rather than left to the engine so the message names WHICH
        # of the two is missing -- "that audio file is not there any more" from
        # deep in the pipeline does not tell the user which box to fix.
        for path, what in ((vocal, "voice"), (instr, "music")):
            if not path:
                self.status.setText(f"Pick a file for the {what} track first.")
                return
            if not os.path.exists(path):
                self.status.setText(f"The {what} track is not there: {path}")
                return
        if self._busy():
            # Silent before. This tab has its own status line rather than the
            # chat's system feed, so it says it there.
            self.status.setText('A mashup is already being built — wait for it to finish.')
            return
        try:
            import mashup
            if not mashup.engine_available():
                self.status.setText(
                    "The mashup engine is not installed — "
                    "run: pip install --no-deps demucs")
                return
        except Exception as exc:
            self.status.setText(f"Mashup engine unavailable: {exc}")
            return

        self.play_btn.setEnabled(False)
        self.build_btn.setEnabled(False)
        self.status.setText("Separating stems…")
        # Read the widgets HERE, on the UI thread, and hand the worker plain
        # data -- see MashupWorker.__init__ for why it must not read them.
        self.worker = MashupWorker(
            vocal, instr,
            vocal_is_speech=bool(self.kind.currentData()),
            match_tempo=self.tempo_chk.isChecked(),
            match_key=self.key_chk.isChecked())
        self.worker.stage.connect(self._on_stage)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_stage(self, stage: str):
        self.status.setText(f"{stage.capitalize()}…")

    def _on_done(self, rep: dict):
        self._last_out = rep.get("path", "")
        self.play_btn.setEnabled(bool(self._last_out))
        # The numbers are the report: a mashup that DECLINED to match tempo
        # looks identical to one that could not, and this is the only way for a
        # listener to tell which happened.
        # The warning goes FIRST when there is one: a pair that cannot work is
        # still rendered (the user asked for it), and without the reason on top
        # the result just looks like the feature is broken.
        warn = rep.get("warnings") or []
        head = ("⚠️  These two do not really fit: " + "; ".join(warn)
                + "\n") if warn else ""
        self.status.setText(
            head +
            "Done in {took}s — {secs}s of audio.\n"
            "Voice {kv} at {bv} BPM over music {ki} at {bi} BPM.\n"
            "Stretch ×{stretch}, pitch {semis} semitones.\n"
            "Saved: {path}".format(
                took=rep.get("took"), secs=rep.get("seconds"),
                kv=rep.get("key_vocal") or "?", bv=rep.get("bpm_vocal") or "?",
                ki=rep.get("key_instr") or "?", bi=rep.get("bpm_instr") or "?",
                stretch=rep.get("stretch"), semis=rep.get("semitones"),
                path=rep.get("path")))

    def _on_failed(self, msg: str):
        self.status.setText(f"Mashup failed: {msg}")

    def _on_finished(self):
        self.build_btn.setEnabled(True)
        self.worker = None

    def _play_last(self):
        if not self._last_out:
            return
        try:
            self.player.play(self._last_out)
        except Exception as exc:
            logger.exception("Mashup playback failed")
            self.status.setText(f"Playback failed: {exc}")

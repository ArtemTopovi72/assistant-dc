"""🎙 Voice clone tab: pick any file with speech, then type text to hear it in that voice.

The same engine as the Telegram flow (voice_clone.py): the best 6-11 s of
speech is cut on pauses and transcribed, the text gets its punctuation
polished without a word changed, and synthesis goes through the house TTS
path (acronyms spelled, trailing-dot padding, stress). Only layout lives here.

Two QThreads, one per slow step; neither reads a widget (unhandled slot
exceptions abort the process with no traceback -- see gui_busy_state).
"""
import os

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
                             QPushButton, QVBoxLayout, QWidget)

from audio import AudioPlayer
from ui_scale import px
from gui_common import MUTED, _section

import logging
logger = logging.getLogger("assistant.gui")

_MEDIA_FILTER = ("Audio / video (*.wav *.mp3 *.flac *.ogg *.oga *.m4a *.opus *.aac "
                 "*.mp4 *.mov *.mkv *.webm *.avi);;All files (*)")
_REASONS = {
    "no_audio": 'The file has no sound.',
    "too_little_speech": 'Too little speech — at least ~3 seconds are needed.',
    "no_words": 'I hear sound but no words — music or noise.',
}


def _out_dir() -> str:
    from config import OUTPUT_DIR
    d = os.path.join(str(OUTPUT_DIR), "voice_clone")
    os.makedirs(d, exist_ok=True)
    return d


class ClonePrepareWorker(QThread):
    done = pyqtSignal(str, str)        # ref wav, transcript
    failed = pyqtSignal(str)

    def __init__(self, ctx, src: str):
        super().__init__()
        self.ctx, self.src = ctx, src

    def run(self):
        try:
            import voice_clone
            ref, text = voice_clone.prepare_reference(self.ctx, self.src, _out_dir())
            self.done.emit(ref, text)
        except Exception as exc:
            logger.exception("voice clone: preparing the sample failed")
            self.failed.emit(_REASONS.get(str(exc), f"Failed: {exc}"))


class CloneSpeakWorker(QThread):
    done = pyqtSignal(str, str)        # wav, polished text
    failed = pyqtSignal(str)

    def __init__(self, ctx, ref: str, ref_text: str, text: str):
        super().__init__()
        self.ctx, self.ref, self.ref_text, self.text = ctx, ref, ref_text, text

    def run(self):
        try:
            import voice_clone
            clean = voice_clone.polish(self.ctx, self.text)
            wav = voice_clone.speak(self.ctx, self.ref, self.ref_text, clean, _out_dir())
            if wav:
                self.done.emit(wav, clean)
            else:
                self.failed.emit('Synthesis returned nothing — is the TTS model loaded?')
        except Exception as exc:
            logger.exception("voice clone: synthesis failed")
            self.failed.emit(f"Could not voice it: {exc}")


class VoiceCloneTab(QWidget):
    def __init__(self, host):
        super().__init__()
        self.host = host
        self.player = AudioPlayer()
        self.prep_worker = None
        self.speak_worker = None
        self.ref, self.ref_text, self.last_wav = "", "", ""

        root = QVBoxLayout(self)
        root.setContentsMargins(px(6), px(6), px(6), px(6))
        root.setSpacing(px(8))
        root.addWidget(_section('Voice clone'))
        hint = QLabel('Pick any file with speech: a voice message, audio, video. The best 6–10 seconds of speech are cut out automatically. Then type text — it is spoken in that voice.')
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{MUTED};")
        root.addWidget(hint)

        row = QHBoxLayout()
        self.src_in = QLineEdit(); self.src_in.setPlaceholderText('Voice file…')
        browse = QPushButton('Browse…'); browse.setObjectName("ghost")
        browse.clicked.connect(self._browse)
        self.take_btn = QPushButton('🎙  Take the voice')
        self.take_btn.clicked.connect(self._take)
        row.addWidget(self.src_in, 1); row.addWidget(browse); row.addWidget(self.take_btn)
        root.addLayout(row)

        self.ref_lbl = QLabel('No voice selected.')
        self.ref_lbl.setWordWrap(True)
        self.ref_lbl.setStyleSheet(f"color:{MUTED};")
        root.addWidget(self.ref_lbl)

        self.text_in = QPlainTextEdit()
        self.text_in.setPlaceholderText('Text to speak…')
        self.text_in.setMaximumHeight(px(140))
        root.addWidget(self.text_in)

        act = QHBoxLayout()
        self.speak_btn = QPushButton('🔊  Speak'); self.speak_btn.setEnabled(False)
        self.speak_btn.clicked.connect(self._speak)
        self.play_btn = QPushButton('▶  Again'); self.play_btn.setObjectName("ghost")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self._play)
        act.addWidget(self.speak_btn); act.addWidget(self.play_btn); act.addStretch(1)
        root.addLayout(act)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color:{MUTED};")
        root.addWidget(self.status)
        root.addStretch(1)

    def _ctx(self):
        return getattr(self.host, "ctx", None)

    def _busy(self) -> bool:
        return any(w is not None and w.isRunning() for w in (self.prep_worker, self.speak_worker))

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Voice file', "", _MEDIA_FILTER)
        if path:
            self.src_in.setText(path)

    def _take(self):
        src = self.src_in.text().strip()
        if not src or not os.path.exists(src):
            self.status.setText('Pick a voice file first.')
            return
        if self._busy():
            self.status.setText('Already working — wait for it to finish.')
            return
        self.status.setText('⏳ Listening to the voice…')
        self.take_btn.setEnabled(False)
        self.prep_worker = ClonePrepareWorker(self._ctx(), src)
        self.prep_worker.done.connect(self._on_ref)
        self.prep_worker.failed.connect(self._on_fail)
        self.prep_worker.start()

    def _on_ref(self, ref: str, text: str):
        self.ref, self.ref_text = ref, text
        self.take_btn.setEnabled(True)
        self.speak_btn.setEnabled(True)
        self.ref_lbl.setText(f"✅ Voice ready. Sample: «{text}»")
        self.status.setText('Type text and press «Speak».')

    def _on_fail(self, why: str):
        self.take_btn.setEnabled(True)
        self.speak_btn.setEnabled(bool(self.ref))
        self.status.setText(why)

    def _speak(self):
        text = self.text_in.toPlainText().strip()
        if not self.ref:
            self.status.setText('Take a voice first.')
            return
        if not text:
            self.status.setText('Write what to say.')
            return
        if self._busy():
            self.status.setText('Already working — wait for it to finish.')
            return
        self.status.setText('⏳ Speaking…')
        self.speak_btn.setEnabled(False)
        self.speak_worker = CloneSpeakWorker(self._ctx(), self.ref, self.ref_text, text)
        self.speak_worker.done.connect(self._on_spoken)
        self.speak_worker.failed.connect(self._on_fail)
        self.speak_worker.start()

    def _on_spoken(self, wav: str, clean: str):
        self.last_wav = wav
        self.speak_btn.setEnabled(True)
        self.play_btn.setEnabled(True)
        self.status.setText(f"Done: {wav}\nSpoken: «{clean}»")
        self._play()

    def _play(self):
        if not self.last_wav:
            return
        try:
            self.player.play(self.last_wav)
        except Exception as exc:
            logger.exception("voice clone playback failed")
            self.status.setText(f"Could not play: {exc}")

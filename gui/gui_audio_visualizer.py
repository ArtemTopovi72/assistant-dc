"""Audio visualiser: the waveform bars that move while the assistant speaks.

_wav_envelope travels with it — it reduces a wav to the per-frame amplitudes
the bars are drawn from and has no other caller.

numpy is imported at module level here rather than function-local, matching
what gui.py did: the envelope is computed on every utterance, so deferring the
import would buy nothing and this module is only imported by the main window.
"""
import random

import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor, QPainter
from PyQt5.QtWidgets import QWidget

from ui_scale import px

import logging
logger = logging.getLogger("assistant.gui")


def _wav_envelope(path: str, interval_ms: int = 45):
    """Downsampled amplitude envelope (0..1 per frame) for the audio visualizer."""
    try:
        import soundfile as sf
        data, sr = sf.read(path)
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)
        data = np.abs(np.asarray(data, dtype=np.float32))
        win = max(1, int(sr * interval_ms / 1000))
        n = len(data) // win
        env = [float(min(1.0, data[i * win:(i + 1) * win].mean() * 5.0)) for i in range(n)]
        return env
    except Exception as exc:
        logger.debug("envelope failed: %s", exc)
        return []

# --------------------------------------------------------------------------- #
# Animated audio visualizer
# --------------------------------------------------------------------------- #


class AudioVisualizer(QWidget):
    def __init__(self, bars: int = 26, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(px(96))
        self._n = bars
        self._levels = [0.0] * bars
        self._env = []
        self._idx = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def play_file(self, path: str):
        self._env = _wav_envelope(path)
        if not self._env:
            return
        self._idx = 0
        self._timer.start(45)

    def push_level(self, amp: float):
        """Drive the bars from a single live amplitude (0..1), e.g. the microphone
        RMS while recording. Mic speech RMS is small, so it's boosted before the
        same per-bar jitter the playback path uses, giving a lively waterfall."""
        a = max(0.0, min(1.0, amp * 8.0))
        self._levels = [max(0.04, min(1.0, a * (0.45 + random.random() * 0.85)))
                        for _ in range(self._n)]
        self.update()

    def stop(self):
        self._timer.stop()
        self._levels = [0.0] * self._n
        self.update()

    def set_paused(self, paused: bool):
        if paused:
            self._timer.stop()
        elif self._env and self._idx < len(self._env):
            self._timer.start(45)

    def _tick(self):
        if self._idx >= len(self._env):
            self.stop()
            return
        amp = self._env[self._idx]
        self._idx += 1
        self._levels = [max(0.04, min(1.0, amp * (0.45 + random.random() * 0.85))) for _ in range(self._n)]
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h, n = self.width(), self.height(), self._n
        slot = w / n
        bw = slot * 0.55
        for i, lvl in enumerate(self._levels):
            bh = max(4.0, lvl * h * 0.86)
            x = i * slot + (slot - bw) / 2
            y = (h - bh) / 2
            t = i / max(1, n - 1)
            col = QColor(
                int(0x2b + (0x00 - 0x2b) * t),
                int(0x6c + (0xd3 - 0x6c) * t),
                int(0xf0 + (0xa7 - 0xf0) * t),
            )
            p.setBrush(col)
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(int(x), int(y), int(bw), int(bh), 3, 3)

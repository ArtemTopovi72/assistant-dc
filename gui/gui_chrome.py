"""Main-window chrome: the dark title bar filter and the stage indicator.

Two unrelated-looking pieces that are both "the frame around the app", and
both are consumed by the top level rather than by any tab:

  _DarkTitleBarFilter  re-applies the dark title bar whenever a window is shown.
                       Windows repaints the frame light again on some show/
                       restore transitions, so setting it once at construction
                       is not enough — an event filter is. run_gui installs it.

  StageIndicator       the "Thinking / Searching / Drawing" pill. AssistantWindow
                       drives it from ctx.set_stage, which is called from worker
                       threads, so it only ever changes text through a Qt signal.

Neither is patched by any suite, and their consumers (run_gui, AssistantWindow)
both stay in gui.py, so this move needed no test changes.
"""
from PyQt5.QtCore import QEvent, QObject, QTimer
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QWidget

from ui_scale import scale_style as _ss
from gui_common import ACCENT2, MUTED, TEXT, enable_dark_titlebar

import logging
logger = logging.getLogger("assistant.gui")


class _DarkTitleBarFilter(QObject):
    """App-wide filter: when any top-level window is first shown, darken its native
    title bar. Catches every QDialog / QMessageBox / QInputDialog without touching each
    call site. Each HWND is handled once."""

    def __init__(self):
        super().__init__()
        self._seen = set()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Show and isinstance(obj, QWidget) and obj.isWindow():
            try:
                wid = int(obj.winId())
                if wid not in self._seen:
                    self._seen.add(wid)
                    enable_dark_titlebar(obj)
            except Exception:
                pass
        return False

# --------------------------------------------------------------------------- #
# Animated stage indicator (what the agent is doing right now)
# --------------------------------------------------------------------------- #


class StageIndicator(QWidget):
    _FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self.spinner = QLabel("●")
        self.label = QLabel("Ready")
        lay.addWidget(self.spinner)
        lay.addWidget(self.label)
        self._i = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._spin)
        self.set_stage("Ready")

    def _spin(self):
        self._i = (self._i + 1) % len(self._FRAMES)
        self.spinner.setText(self._FRAMES[self._i])

    def set_stage(self, stage: str):
        s = (stage or "").strip()
        if not s or s.lower() in ("ready", "idle"):
            self._timer.stop()
            self.spinner.setText("●")
            self.spinner.setStyleSheet(_ss(f"font-size:15px; color:{MUTED};"))
            self.label.setText("Ready")
            self.label.setStyleSheet(f"color:{MUTED};")
            return
        self.label.setText(s)
        self.label.setStyleSheet(f"color:{TEXT}; font-weight:600;")
        self.spinner.setStyleSheet(_ss(f"font-size:15px; color:{ACCENT2};"))
        if not self._timer.isActive():
            self._timer.start(80)

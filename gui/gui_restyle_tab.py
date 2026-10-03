"""🎨 Restyle tab: pick a video, name a new look, get the same motion redrawn.

Same engine as the Telegram flow (video_control.restyle on the qi21 ComfyUI,
started for the job). ~15 min for 5 s, so the worker runs on a QThread with its
own cancel token (_ScopedCtx) -- the main chat's Stop must not kill it and its
Stop must not kill the chat. The worker never reads a widget.
"""
import os
import threading
import logging

from PyQt5.QtCore import QThread, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QVBoxLayout, QWidget)

from gui_common import MUTED, _ScopedCtx, _section
from ui_scale import px

logger = logging.getLogger("assistant.gui")
_FILTER = "Video (*.mp4 *.mov *.mkv *.webm *.avi *.gif);;All files (*)"


class RestyleWorker(QThread):
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, ctx, src: str, look: str):
        super().__init__()
        self.ctx, self.src, self.look = ctx, src, look

    def run(self):
        try:
            import graph_language
            import video_control
            try:
                en = graph_language._translate_to_english(self.ctx, self.look) or self.look
            except Exception:
                en = self.look
            out = video_control.restyle(self.ctx, self.src, en)
            if out:
                self.done.emit(out)
            else:
                self.failed.emit('The render returned nothing (details in runtime/qi21.log).')
        except Exception as exc:
            logger.exception("restyle tab: render failed")
            self.failed.emit(f"Failed: {exc}")


class RestyleTab(QWidget):
    def __init__(self, host):
        super().__init__()
        self.host = host
        self.worker = None
        self._cancel = threading.Event()
        self.last = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(px(6), px(6), px(6), px(6))
        root.setSpacing(px(8))
        root.addWidget(_section('🎨 Restyle video'))
        hint = QLabel('Motion and composition come from your clip (the first ~5 s), the look from the description: «anime», «winter, snow», «clay». About 15 minutes per clip; the GPU is fully busy for that time.')
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{MUTED};")
        root.addWidget(hint)

        row = QHBoxLayout()
        self.src_in = QLineEdit(); self.src_in.setPlaceholderText('Video…')
        browse = QPushButton('Browse…'); browse.setObjectName("ghost")
        browse.clicked.connect(self._browse)
        row.addWidget(self.src_in, 1); row.addWidget(browse)
        root.addLayout(row)

        self.look_in = QLineEdit(); self.look_in.setPlaceholderText('What look…')
        self.look_in.returnPressed.connect(self._go)
        root.addWidget(self.look_in)

        act = QHBoxLayout()
        self.go_btn = QPushButton('🎨  Restyle'); self.go_btn.clicked.connect(self._go)
        self.stop_btn = QPushButton('⛔  Stop'); self.stop_btn.setObjectName("ghost")
        self.stop_btn.setEnabled(False); self.stop_btn.clicked.connect(self._stop)
        self.open_btn = QPushButton('▶  Open'); self.open_btn.setObjectName("ghost")
        self.open_btn.setEnabled(False); self.open_btn.clicked.connect(self._open)
        act.addWidget(self.go_btn); act.addWidget(self.stop_btn); act.addWidget(self.open_btn)
        act.addStretch(1)
        root.addLayout(act)

        self.status = QLabel(""); self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color:{MUTED};")
        root.addWidget(self.status)
        root.addStretch(1)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Video', "", _FILTER)
        if path:
            self.src_in.setText(path)

    def _go(self):
        src, look = self.src_in.text().strip(), self.look_in.text().strip()
        if not src or not os.path.exists(src):
            self.status.setText('Pick a video first.')
            return
        if not look:
            self.status.setText('Write what look you want.')
            return
        if self.worker is not None and self.worker.isRunning():
            self.status.setText('Already rendering — wait for it to finish.')
            return
        ctx = getattr(self.host, "ctx", None)
        if ctx is None:
            self.status.setText('The assistant is not ready yet.')
            return
        self._cancel = threading.Event()
        self.status.setText('⏳ Restyling… (~15 min)')
        self.go_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self.worker = RestyleWorker(_ScopedCtx(ctx, self._cancel), src, look)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_fail)
        self.worker.start()

    def _stop(self):
        self._cancel.set()
        self.status.setText('Stopping…')

    def _on_done(self, out: str):
        self.last = out
        self.go_btn.setEnabled(True); self.stop_btn.setEnabled(False); self.open_btn.setEnabled(True)
        self.status.setText(f"✅ Done: {out}")

    def _on_fail(self, why: str):
        self.go_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self.status.setText('Stopped.' if self._cancel.is_set() else why)

    def _open(self):
        if self.last:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.last))

    def shutdown(self):
        self._cancel.set()
        if self.worker is not None:
            self.worker.wait(3000)

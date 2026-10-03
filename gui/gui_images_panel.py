"""Images panel: the thumbnail strip of everything the session has produced.

_ThumbLabel travels with it — it is the clickable thumbnail the panel is made
of and has no other caller. Clicking one opens the shared viewer from
gui_dialogs, which is why that lives in neither this module nor a tab.
"""
import os

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from ui_scale import px, scale_style as _ss
from gui_common import BG, MUTED, _section
from gui_dialogs import show_image_viewer

import logging
logger = logging.getLogger("assistant.gui")


# --------------------------------------------------------------------------- #
# Images gallery
# --------------------------------------------------------------------------- #
class _ThumbLabel(QLabel):
    """A gallery thumbnail that keeps its source pixmap so it can be rescaled to
    the panel width on resize, and opens the full viewer on double-click."""

    def __init__(self, path: str, pix: QPixmap, panel):
        super().__init__()
        self._path = path
        self._pix = pix
        self._panel = panel
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Double-click to view full size — use ← → to browse")
        self.setStyleSheet("border:1px solid #2c2e3a; border-radius:8px; padding:3px;")

    def rescale(self, width: int):
        w = max(80, min(self._pix.width(), width))
        self.setPixmap(self._pix.scaledToWidth(w, Qt.SmoothTransformation))

    def mouseDoubleClickEvent(self, e):
        self._panel.open_viewer(self)


class ImagesPanel(QScrollArea):
    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        # The scroll viewport falls back to the default (white) palette base unless
        # explicitly themed — keep it dark like the rest of the UI.
        self.viewport().setStyleSheet(_ss(f"background:{BG};"))
        host = QWidget()
        self._lay = QVBoxLayout(host)
        self._lay.setAlignment(Qt.AlignTop)
        self._thumbs = []
        self._add_placeholder()
        self.setWidget(host)

    def _add_placeholder(self):
        self._placeholder = QLabel("Generated images will appear here.\n"
                                   "Double-click any image to view it full size.")
        self._placeholder.setStyleSheet(f"color:{MUTED};")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._lay.addWidget(self._placeholder)

    def _avail_width(self) -> int:
        return max(120, self.viewport().width() - 24)

    def add_image(self, path: str):
        pix = QPixmap(path)
        if pix.isNull():
            return
        if self._placeholder is not None:
            self._placeholder.hide()
            self._placeholder.deleteLater()
            self._placeholder = None
        thumb = _ThumbLabel(path, pix, self)
        thumb.rescale(self._avail_width())
        self._thumbs.insert(0, thumb)
        self._lay.insertWidget(0, thumb)

    def open_viewer(self, thumb):
        """Open the full viewer on the whole gallery, starting at `thumb`."""
        paths = [t._path for t in self._thumbs]
        try:
            idx = self._thumbs.index(thumb)
        except ValueError:
            idx = 0
        show_image_viewer(paths, idx, self.window())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        w = self._avail_width()
        for t in self._thumbs:
            t.rescale(w)

    def clear_images(self):
        """Remove all thumbnails and restore the placeholder (full session reset)."""
        while self._lay.count():
            item = self._lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._thumbs = []
        self._add_placeholder()

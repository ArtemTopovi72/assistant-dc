"""The storyboard box canvas: draw, hit-test and drag layout boxes.

Lifted out of gui_storyboard_tab.py. Self-contained QWidget -- it owns the
painting and mouse handling and talks to its parent only through signals, which
is what let it move without dragging tab state along.
"""
import copy
import json
import os
import random
import threading
from PyQt5.QtCore import QRect, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QLabel, QLineEdit, QListWidget, QPushButton, QSpinBox,
    QWidget,
)
from ui_scale import pt, px
from gui_common import BORDER, PANEL2, _FlowWidget, _flow, _section


class _BoxCanvas(QWidget):
    """The storyboard surface: the picture frame with draggable/resizable boxes.

    Boxes are stored as FRACTIONS of the frame (x, y, w, h in 0..1) because that
    is what Ideogram's 0-1000 grid wants — so the layout survives a change of
    output resolution or aspect, and the widget can be any size on screen.
    """

    changed = pyqtSignal()          # a box was moved/resized (live, during drag)
    selected = pyqtSignal(int)      # index of the box under the cursor, -1 for none

    _HANDLE = 16                    # px grab area of the bottom-right resize corner
    _PALETTE = ("#00d3a7", "#2b6cf0", "#e2a33b", "#e2547a", "#8b5cf6", "#3bbfe2")

    def __init__(self):
        super().__init__()
        self.elements = []          # list of dicts (SHARED with the tab's layout)
        self.aspect = 1.0           # width / height of the target image
        self.backdrop = None        # QPixmap of the last render, drawn under the boxes
        self.show_boxes = True
        self.sel = -1
        self._mode = None           # "move" | "resize"
        self._grab = None           # (dx, dy) in fractions, for move
        self.setMinimumHeight(px(220))
        self.setMouseTracking(True)
        self.setCursor(Qt.ArrowCursor)

    # ---- geometry -------------------------------------------------------
    def frame_rect(self) -> QRect:
        """The letterboxed drawing area inside the widget, at the target aspect."""
        m = px(8)
        w, h = max(1, self.width() - 2 * m), max(1, self.height() - 2 * m)
        if w / h > self.aspect:
            w = int(h * self.aspect)
        else:
            h = int(w / self.aspect)
        return QRect(m + (self.width() - 2 * m - w) // 2,
                     m + (self.height() - 2 * m - h) // 2, w, h)

    def _box_rect(self, el) -> QRect:
        f = self.frame_rect()
        return QRect(f.x() + int(el["x"] * f.width()), f.y() + int(el["y"] * f.height()),
                     max(2, int(el["w"] * f.width())), max(2, int(el["h"] * f.height())))

    def _hit(self, pos):
        """Topmost box under `pos` -> (index, "move"|"resize"), else (-1, None).
        Later elements are drawn on top, so they are hit-tested first."""
        for i in range(len(self.elements) - 1, -1, -1):
            r = self._box_rect(self.elements[i])
            corner = QRect(r.right() - self._HANDLE, r.bottom() - self._HANDLE,
                           self._HANDLE * 2, self._HANDLE * 2)
            if corner.contains(pos):
                return i, "resize"
            if r.contains(pos):
                return i, "move"
        return -1, None

    # ---- painting -------------------------------------------------------
    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        f = self.frame_rect()
        if self.backdrop is not None and not self.backdrop.isNull():
            p.drawPixmap(f, self.backdrop)
        else:
            p.fillRect(f, QColor(PANEL2))
        p.setPen(QPen(QColor(BORDER), 1))
        p.drawRect(f.adjusted(0, 0, -1, -1))
        # rule-of-thirds guides: the whole point of the mode is placement
        p.setPen(QPen(QColor(255, 255, 255, 26), 1, Qt.DashLine))
        for k in (1, 2):
            p.drawLine(f.x() + f.width() * k // 3, f.y(),
                       f.x() + f.width() * k // 3, f.bottom())
            p.drawLine(f.x(), f.y() + f.height() * k // 3,
                       f.right(), f.y() + f.height() * k // 3)
        if not self.show_boxes:
            return
        font = p.font(); font.setPointSizeF(pt(8)); p.setFont(font)
        for i, el in enumerate(self.elements):
            r = self._box_rect(el)
            colour = QColor(self._PALETTE[i % len(self._PALETTE)])
            is_sel = (i == self.sel)
            fill = QColor(colour); fill.setAlpha(46 if is_sel else 24)
            p.fillRect(r, fill)
            p.setPen(QPen(colour, px(3) if is_sel else px(1)))
            p.drawRect(r)
            # resize grip
            p.fillRect(QRect(r.right() - px(9), r.bottom() - px(9), px(9), px(9)), colour)
            # label: the text an element renders, else its description
            label = (f'“{el["text"]}”' if el.get("text") else el.get("desc", "")) or "?"
            label = f"{i + 1}. {label}"
            metrics = p.fontMetrics()
            label = metrics.elidedText(label, Qt.ElideRight, max(px(40), r.width() - px(8)))
            tw = metrics.horizontalAdvance(label) + px(8)
            th = metrics.height() + px(4)
            ty = r.y() - th if r.y() - th > f.y() else r.y()
            p.fillRect(QRect(r.x(), ty, tw, th), QColor(20, 21, 27, 210))
            p.setPen(QPen(colour))
            p.drawText(QRect(r.x() + px(4), ty, tw, th), Qt.AlignVCenter | Qt.AlignLeft, label)

    # ---- interaction ----------------------------------------------------
    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        i, mode = self._hit(e.pos())
        self.sel, self._mode = i, mode
        if i >= 0:
            f, el = self.frame_rect(), self.elements[i]
            self._grab = (e.pos().x() / f.width() - el["x"],
                          e.pos().y() / f.height() - el["y"])
        self.selected.emit(i)
        self.update()

    def mouseMoveEvent(self, e):
        f = self.frame_rect()
        if self._mode is None or self.sel < 0 or not (e.buttons() & Qt.LeftButton):
            _i, mode = self._hit(e.pos())
            self.setCursor(Qt.SizeFDiagCursor if mode == "resize"
                           else Qt.SizeAllCursor if mode == "move" else Qt.ArrowCursor)
            return
        el = self.elements[self.sel]
        fx = (e.pos().x() - f.x()) / max(1, f.width())
        fy = (e.pos().y() - f.y()) / max(1, f.height())
        if self._mode == "move":
            el["x"] = min(max(0.0, fx - self._grab[0] + f.x() / max(1, f.width())), 1.0 - el["w"])
            el["y"] = min(max(0.0, fy - self._grab[1] + f.y() / max(1, f.height())), 1.0 - el["h"])
        else:
            el["w"] = min(max(0.02, fx - el["x"]), 1.0 - el["x"])
            el["h"] = min(max(0.02, fy - el["y"]), 1.0 - el["y"])
        self.changed.emit()
        self.update()

    def mouseReleaseEvent(self, _e):
        self._mode = None
        self._grab = None

"""Shared image dialogs: the full-size viewer and the mask editor.

Pulled out while extracting TransferTab, but deliberately NOT put inside it.
The closure said TransferTab needs these; usage says three different places do
— AssistantWindow opens the mask editor, ImagesPanel and AssistantWindow both
open the viewer. Burying them in a tab module would have made two unrelated
callers import "the transfer tab" to draw a mask.

OUTPUT_DIR_GUI_MASK travels with them: it is where the mask editor writes, and
it means nothing without them.
"""
import os
import shutil
import time
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QImage, QKeySequence, QPainter, QPixmap
from PyQt5.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QFileDialog,
                             QHBoxLayout, QLabel, QPushButton, QScrollArea,
                             QShortcut, QSlider, QVBoxLayout, QWidget)

from ui_scale import pt, px
from gui_common import BG, MUTED

import logging
logger = logging.getLogger("assistant.gui")


# --------------------------------------------------------------------------- #
# Full-size image viewer (shared by the gallery and the chat)
# --------------------------------------------------------------------------- #
class ImageViewerDialog(QDialog):
    """Maximised viewer: the image is fit to the window width and scrolls
    vertically for tall pictures. ‹ / › buttons and the Left/Right arrow keys
    step through the supplied gallery; "Save copy" writes the current image out
    and "✕" (or Esc) closes it. Used for both gallery thumbnails and chat images."""

    def __init__(self, paths, index: int = 0, parent=None):
        super().__init__(parent)
        if isinstance(paths, str):
            paths = [paths]
        self._paths = list(paths) or [""]
        self._idx = max(0, min(index, len(self._paths) - 1))
        self._pix = QPixmap()
        # These must exist before any resizeEvent fires. Setting the window state
        # or adding the scroll area can trigger resizeEvent() synchronously on the
        # real platform; if _scroll/_label aren't bound yet, _fit() would raise
        # inside a Qt C++ callback and hard-crash the process (0xC0000409).
        self._scroll = None
        self._label = None
        self.setStyleSheet(f"background:{BG};")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        bar = QHBoxLayout()
        bar.setContentsMargins(px(12), px(8), px(12), px(8))
        self._prev_btn = QPushButton("‹")
        self._prev_btn.setFixedWidth(px(42))
        self._prev_btn.setToolTip("Previous (←)")
        self._prev_btn.clicked.connect(self._prev)
        self._next_btn = QPushButton("›")
        self._next_btn.setFixedWidth(px(42))
        self._next_btn.setToolTip("Next (→)")
        self._next_btn.clicked.connect(self._next)
        self._info = QLabel()
        self._info.setStyleSheet(f"color:{MUTED};")
        save_btn = QPushButton("💾  Save copy")
        save_btn.clicked.connect(self._save_copy)
        close_btn = QPushButton("✕")
        close_btn.setFixedWidth(px(42))
        close_btn.setToolTip("Close (Esc)")
        close_btn.clicked.connect(self.accept)
        bar.addWidget(self._prev_btn)
        bar.addWidget(self._next_btn)
        bar.addSpacing(8)
        bar.addWidget(self._info)
        bar.addStretch(1)
        bar.addWidget(save_btn)
        bar.addWidget(close_btn)
        lay.addLayout(bar)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(False)
        self._scroll.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self._label = QLabel()
        self._label.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self._scroll.setWidget(self._label)
        lay.addWidget(self._scroll, 1)
        self._load_current()
        # Window-level shortcuts so the arrows flip images no matter which child
        # widget has keyboard focus. Relying on keyPressEvent alone is unreliable:
        # when a button or the scroll area has focus it swallows Left/Right (focus
        # navigation / horizontal scroll) before the dialog ever sees them.
        for keys, slot in ((Qt.Key_Left, self._prev), (Qt.Key_Right, self._next),
                           (Qt.Key_Escape, self.accept)):
            sc = QShortcut(QKeySequence(keys), self)
            sc.setContext(Qt.WindowShortcut)
            sc.activated.connect(slot)
        # Maximise only after the UI is fully built so the resulting resizeEvent
        # finds _scroll/_label already bound.
        self.setWindowState(Qt.WindowMaximized)

    # ---- navigation ----
    @property
    def _path(self) -> str:
        return self._paths[self._idx]

    def _load_current(self):
        path = self._path
        self._pix = QPixmap(path)
        nav = f"  ({self._idx + 1}/{len(self._paths)})" if len(self._paths) > 1 else ""
        if self._pix.isNull():
            self._label.setText("(image could not be loaded)")
            self._label.setStyleSheet(f"color:{MUTED}; padding:40px;")
            dims = ""
        else:
            self._label.setStyleSheet("")
            dims = f"  ·  {self._pix.width()}×{self._pix.height()}"
        self._info.setText(f"{os.path.basename(path)}{dims}{nav}")
        self.setWindowTitle(os.path.basename(path) or "Image")
        multi = len(self._paths) > 1
        self._prev_btn.setEnabled(multi)
        self._next_btn.setEnabled(multi)
        self._scroll.verticalScrollBar().setValue(0)
        self._fit()

    def _step(self, delta: int):
        if len(self._paths) > 1:
            self._idx = (self._idx + delta) % len(self._paths)
            self._load_current()

    def _prev(self):
        self._step(-1)

    def _next(self):
        self._step(1)

    def _fit(self):
        if self._scroll is None or self._label is None or self._pix.isNull():
            return
        vw = max(self._scroll.viewport().width() - 4, 100)
        target = min(self._pix.width(), vw)
        scaled = self._pix.scaledToWidth(target, Qt.SmoothTransformation)
        self._label.setPixmap(scaled)
        self._label.resize(scaled.size())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()

    def keyPressEvent(self, e):
        # Left/Right are handled by window-level QShortcuts (above); these are a
        # fallback plus PageUp/PageDown. Space is intentionally NOT bound — it would
        # also activate whichever toolbar button has focus (e.g. close the dialog).
        k = e.key()
        if k == Qt.Key_Escape:
            self.accept()
            return
        if k == Qt.Key_PageUp:
            self._prev()
            return
        if k == Qt.Key_PageDown:
            self._next()
            return
        super().keyPressEvent(e)

    def _save_copy(self):
        if not os.path.exists(self._path) and self._pix.isNull():
            return
        base = os.path.basename(self._path) or "image.png"
        suggested = os.path.join(os.path.expanduser("~"), base)
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save a copy of the image", suggested,
            "PNG image (*.png);;JPEG image (*.jpg *.jpeg);;All files (*)")
        if not dest:
            return
        try:
            if os.path.exists(self._path):
                shutil.copyfile(self._path, dest)
            else:
                self._pix.save(dest)
        except Exception:
            # last resort: re-encode whatever pixels we have
            if not self._pix.isNull():
                self._pix.save(dest)


def show_image_viewer(paths, index: int = 0, parent=None):
    """Open ImageViewerDialog on a single path or a list of paths at `index`."""
    if isinstance(paths, str):
        paths = [paths] if paths and os.path.exists(paths) else []
    paths = [p for p in (paths or []) if p]
    if not paths:
        return
    index = max(0, min(index, len(paths) - 1))
    # Often invoked from a Qt callback (double-click / timer), where an unhandled
    # exception fast-fails the whole process — never let viewer construction or a
    # bad image take the app down.
    try:
        ImageViewerDialog(paths, index, parent).exec_()
    except Exception:
        logger.exception("Image viewer failed to open for %r", paths[index])


class MaskCanvas(QWidget):
    """Paint a mask over an image with the mouse. Left-drag paints, right-drag
    erases. The mask is kept at the displayed resolution and exported (scaled) to
    the source resolution as an 8-bit L PNG (white = edit here).

    Self-contained + headless-testable: stroke_at()/erase_at()/clear()/export_mask()
    work without a window, so the mask logic can be unit-tested offscreen."""

    def __init__(self, image_path, max_side=720, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self._base = QPixmap(image_path)
        self.src_w = self._base.width() or 1
        self.src_h = self._base.height() or 1
        scale = min(1.0, max_side / float(max(self.src_w, self.src_h)))
        self.disp_w = max(1, int(self.src_w * scale))
        self.disp_h = max(1, int(self.src_h * scale))
        self._disp = self._base.scaled(self.disp_w, self.disp_h,
                                       Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # ARGB overlay: translucent red where painted; alpha carries the mask.
        self._overlay = QImage(self.disp_w, self.disp_h, QImage.Format_ARGB32)
        self._overlay.fill(Qt.transparent)
        self.brush = max(6, int(min(self.disp_w, self.disp_h) * 0.06))
        self._last = None
        self.setFixedSize(self.disp_w, self.disp_h)
        self.setCursor(Qt.CrossCursor)

    # -- painting ----------------------------------------------------------- #
    def _paint_dab(self, at, erase=False):
        p = QPainter(self._overlay)
        if erase:
            p.setCompositionMode(QPainter.CompositionMode_Clear)
        else:
            p.setCompositionMode(QPainter.CompositionMode_Source)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 40, 40, 150))
        r = self.brush
        if self._last is not None:
            # interpolate so fast drags leave a continuous stroke
            x0, y0 = self._last.x(), self._last.y()
            x1, y1 = at.x(), at.y()
            steps = max(1, int(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / max(1, r // 2)))
            for i in range(steps + 1):
                xx = x0 + (x1 - x0) * i / steps
                yy = y0 + (y1 - y0) * i / steps
                if erase:
                    p.setBrush(QColor(0, 0, 0, 0))
                    p.setCompositionMode(QPainter.CompositionMode_Clear)
                p.drawEllipse(int(xx - r), int(yy - r), 2 * r, 2 * r)
        else:
            p.drawEllipse(at.x() - r, at.y() - r, 2 * r, 2 * r)
        p.end()
        self._last = at
        self.update()

    def stroke_at(self, x, y):
        from PyQt5.QtCore import QPoint
        self._paint_dab(QPoint(int(x), int(y)), erase=False)

    def erase_at(self, x, y):
        from PyQt5.QtCore import QPoint
        self._paint_dab(QPoint(int(x), int(y)), erase=True)

    def clear(self):
        self._overlay.fill(Qt.transparent)
        self.update()

    def is_empty(self) -> bool:
        # any painted pixel has alpha > 0
        for _ in range(0):  # placeholder; real check below
            pass
        img = self._overlay
        # sample on a grid for speed, then confirm
        step = max(1, min(self.disp_w, self.disp_h) // 64)
        for yy in range(0, self.disp_h, step):
            for xx in range(0, self.disp_w, step):
                if (img.pixel(xx, yy) >> 24) & 0xFF:
                    return False
        return True

    # -- mouse -------------------------------------------------------------- #
    def mousePressEvent(self, e):
        self._last = None
        self._paint_dab(e.pos(), erase=(e.button() == Qt.RightButton))

    def mouseMoveEvent(self, e):
        erase = bool(e.buttons() & Qt.RightButton)
        self._paint_dab(e.pos(), erase=erase)

    def mouseReleaseEvent(self, e):
        self._last = None

    def paintEvent(self, _e):
        p = QPainter(self)
        p.drawPixmap(0, 0, self._disp)
        p.drawImage(0, 0, self._overlay)
        p.end()

    # -- export ------------------------------------------------------------- #
    def export_mask(self, out_path):
        """Write an L-mode mask at SOURCE resolution: white where painted. Returns
        the path, or None if nothing was drawn."""
        from PIL import Image
        import tempfile
        if self.is_empty():
            return None
        tmp = tempfile.NamedTemporaryFile(suffix="_ov.png", delete=False)
        tmp.close()
        try:
            # Inside the try: delete=False means a QImage.save() failure (disk
            # full, a path Qt cannot write) would otherwise strand the scratch
            # PNG, since the cleanup below was never reached.
            self._overlay.save(tmp.name, "PNG")
            ov = Image.open(tmp.name).convert("RGBA")
            alpha = ov.split()[3]                       # mask carried in alpha
            mask = alpha.point(lambda a: 255 if a > 40 else 0).convert("L")
            mask = mask.resize((self.src_w, self.src_h), Image.NEAREST)
            mask.save(out_path)
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
        return out_path


class MaskDrawDialog(QDialog):
    """Modal mask editor for one image. On accept, exposes `mask_path` (an L PNG at
    source resolution) via the saved attribute, or None if the user drew nothing."""

    def __init__(self, image_path, existing_mask=None, parent=None, title="Draw mask"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.image_path = image_path
        self.mask_path = None
        self.protect_face = True
        lay = QVBoxLayout(self)
        info = QLabel("Left-drag to paint the region · right-drag to erase. "
                      "Paint where the edit should happen.")
        info.setWordWrap(True); info.setStyleSheet(f"color:{MUTED};")
        lay.addWidget(info)
        self.canvas = MaskCanvas(image_path)
        holder = QHBoxLayout(); holder.addStretch(1); holder.addWidget(self.canvas); holder.addStretch(1)
        lay.addLayout(holder)
        if existing_mask and os.path.exists(existing_mask):
            self._load_existing(existing_mask)

        ctr = QHBoxLayout()
        ctr.addWidget(QLabel("Brush"))
        self.bslider = QSlider(Qt.Horizontal); self.bslider.setRange(4, 120)
        self.bslider.setValue(self.canvas.brush)
        self.bslider.valueChanged.connect(lambda v: setattr(self.canvas, "brush", v))
        ctr.addWidget(self.bslider, 1)
        clr = QPushButton("Clear"); clr.setObjectName("ghost"); clr.clicked.connect(self.canvas.clear)
        ctr.addWidget(clr)
        lay.addLayout(ctr)

        self.pf_check = QCheckBox("Protect face (uncheck for face tattoos / makeup)")
        self.pf_check.setChecked(True)
        self.pf_check.setToolTip("Keep the face pixels unchanged even if the mask overlaps "
                                 "them — recommended for hats/clothing. Uncheck to edit the face.")
        lay.addWidget(self.pf_check)

        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._save); bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _load_existing(self, mask_file):
        from PIL import Image
        try:
            m = Image.open(mask_file).convert("L").resize(
                (self.canvas.disp_w, self.canvas.disp_h), Image.NEAREST)
            ov = QImage(self.canvas.disp_w, self.canvas.disp_h, QImage.Format_ARGB32)
            ov.fill(Qt.transparent)
            for y in range(self.canvas.disp_h):
                for x in range(self.canvas.disp_w):
                    if m.getpixel((x, y)) > 40:
                        ov.setPixel(x, y, QColor(255, 40, 40, 150).rgba())
            self.canvas._overlay = ov
            self.canvas.update()
        except Exception:
            logger.warning("MaskDrawDialog: could not preload existing mask", exc_info=True)

    def _save(self):
        out = str(OUTPUT_DIR_GUI_MASK() / f"transfer_mask_{int(time.time()*1000)}.png")
        self.mask_path = self.canvas.export_mask(out)
        self.protect_face = self.pf_check.isChecked()
        self.accept()


def OUTPUT_DIR_GUI_MASK():
    try:
        import config as _cfg
        d = Path(getattr(_cfg, "OUTPUT_DIR", "outputs")) / "masks"
    except Exception:
        d = Path("outputs") / "masks"
    d.mkdir(parents=True, exist_ok=True)
    return d

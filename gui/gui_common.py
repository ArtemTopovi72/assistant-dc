"""Shared GUI toolkit: theme, stylesheet, and the widget helpers every tab uses.

gui.py is 10,790 lines. The obvious seam is one module per tab — but every tab
sits on the same handful of primitives (the palette, _card, _section, _flow,
_scroll_page, _fit_dialog, the flow layout, the timestamp/byte formatters), so
those had to come out FIRST. Otherwise each tab extraction would either drag a
copy along or reach back into gui.py and rebuild the import cycle just removed.

Nothing here knows about the assistant, the model, or any tab. It is Qt
plumbing and colour, which is why it can sit underneath all of them.

Two pieces are worth reading before changing:

  _scroll_page  A widget's minimum size is the sum of its children's minimums,
                and every one is px()-scaled — so at 250-300% on a 4K TV a tab's
                minimum grows past the screen, Qt refuses to shrink below it,
                and Windows crops the window. Wrapping the page in a scroll area
                gives Qt a small minimum while the page keeps its natural size.

  _cap_w        The same failure in the width direction.

`logger` is deliberately NOT moved: 33 symbols in gui.py use it, and both
modules simply take the same "assistant.gui" channel.
"""
import logging
import os
import time

import gui_i18n
import ui_scale
from ui_scale import px
from PyQt5.QtCore import QPoint, QRect, QSize, Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPen
from PyQt5.QtWidgets import (QAbstractScrollArea, QApplication, QFrame,
                             QGraphicsDropShadowEffect, QLabel, QLayout,
                             QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

logger = logging.getLogger("assistant.gui")   # same channel as gui.py


def _crash_log_stage(stage: str) -> None:
    """Thin wrapper around crash_diag.log_stage that never raises if it's missing."""
    try:
        import crash_diag
        crash_diag.log_stage(stage)
    except Exception:
        logger.info("LIFECYCLE: %s", stage)


ICON_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "assistant.ico")
# Generated tab-close glyphs land here, not in the tracked assets/ dir.
from config import OUTPUT_DIR
# Custom dark title bar by default; set NATIVE_FRAME=1 to fall back to the OS frame.
NATIVE_FRAME = os.getenv("NATIVE_FRAME", "") not in ("", "0", "false", "False")

# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #
BG = "#14151b"; PANEL = "#1c1e27"; PANEL2 = "#23252f"
ACCENT = "#2b6cf0"; ACCENT2 = "#00d3a7"; REC = "#e23b3b"
TEXT = "#e8e9ee"; MUTED = "#8b8fa3"; BORDER = "#2c2e3a"


def enable_dark_titlebar(widget) -> None:
    """Make a top-level window's NATIVE Windows title bar dark (the main window is
    frameless, but QDialogs — Settings, mask editor, message boxes — use the OS frame,
    whose default white bar clashes with the dark theme). Uses the DWM immersive-dark
    attribute (20 on Win10 2004+, 19 on older builds). No-op off Windows / on failure."""
    try:
        import ctypes
        hwnd = int(widget.winId())
        val = ctypes.c_int(1)
        dwm = ctypes.windll.dwmapi
        for attr in (20, 19):
            if dwm.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(val), ctypes.sizeof(val)) == 0:
                break
    except Exception:
        pass


def _make_tab_close_icons():
    """Draw clean ✕ glyphs for the tab close-buttons so Qt doesn't fall back to its
    default (a bright white/box pixmap that looks like an artifact on the dark theme).
    Returns {normal, hover} forward-slashed paths usable directly in QSS url(...).
    Regenerated on each build so it tracks the current UI scale. Never raises.

    Written under OUTPUT_DIR (gitignored), NOT into assets/. These are build
    artifacts -- redrawn at every startup and at every UI-scale change -- so
    living in a tracked directory meant each GUI suite run left the working
    tree dirty with two re-encoded, pixel-identical PNGs."""
    assets = os.path.join(str(OUTPUT_DIR), "icons")
    s = max(10, ui_scale.px(12))
    out = {}
    try:
        os.makedirs(assets, exist_ok=True)
        for name, col in (("_tabclose.png", MUTED), ("_tabclose_hover.png", "#ffffff")):
            img = QImage(s, s, QImage.Format_ARGB32)
            img.fill(QColor(0, 0, 0, 0))
            pr = QPainter(img)
            pr.setRenderHint(QPainter.Antialiasing, True)
            pen = QPen(QColor(col)); pen.setWidth(max(1, s // 8)); pr.setPen(pen)
            m = s * 0.30
            pr.drawLine(int(m), int(m), int(s - m), int(s - m))
            pr.drawLine(int(s - m), int(m), int(m), int(s - m))
            pr.end()
            path = os.path.join(assets, name)
            img.save(path)
            out[name.split(".")[0]] = path.replace("\\", "/")
    except Exception:
        return {}
    return out


def build_qss() -> str:
    """Build the application stylesheet at the current UI scale.

    Every px literal is multiplied by the effective scale (DPI is handled
    separately by Qt's high-DPI support), so one factor rescales fonts, paddings,
    radii, buttons, tabs, menus, tooltips and scrollbars together. Rebuilt and
    re-applied whenever the UI Scale / TV Mode setting changes.
    """
    p = ui_scale.px
    fp = ui_scale.fpx   # font sizes also honour the separate Font Scale setting
    _ic = _make_tab_close_icons()
    if _ic:
        close_css = (
            f'QTabBar::close-button {{ image: url("{_ic["_tabclose"]}"); '
            f'subcontrol-position: right; margin-left: {p(4)}px; border-radius: {p(3)}px; }}'
            f'QTabBar::close-button:hover {{ image: url("{_ic["_tabclose_hover"]}"); '
            f'background: {REC}; }}'
        )
    else:
        close_css = ""
    return f"""
{close_css}
QSplitter::handle {{ background: {BORDER}; }}
QSplitter::handle:hover {{ background: {ACCENT}; }}
QSplitter::handle:horizontal {{ width: {p(5)}px; margin: {p(3)}px {p(1)}px; border-radius: {p(2)}px; }}
QSplitter::handle:vertical {{ height: {p(5)}px; margin: {p(1)}px {p(3)}px; border-radius: {p(2)}px; }}
QTabBar::scroller {{ width: {p(26)}px; }}
QTabBar QToolButton {{ background: {PANEL}; border: none; border-radius: 0; margin: 0;
                       border-bottom: {p(1)}px solid {BORDER}; }}
QTabBar QToolButton:hover {{ background: {PANEL2}; }}
QTabBar QToolButton:disabled {{ background: {PANEL}; }}
QTabBar QToolButton::left-arrow {{ image: none; width: 0; height: 0;
        border-top: {p(4)}px solid transparent; border-bottom: {p(4)}px solid transparent;
        border-right: {p(5)}px solid {MUTED}; }}
QTabBar QToolButton::right-arrow {{ image: none; width: 0; height: 0;
        border-top: {p(4)}px solid transparent; border-bottom: {p(4)}px solid transparent;
        border-left: {p(5)}px solid {MUTED}; }}
QTabBar QToolButton::left-arrow:disabled, QTabBar QToolButton::right-arrow:disabled {{
        border-right-color: {BORDER}; border-left-color: {BORDER}; }}
QAbstractScrollArea::corner {{ background: {BG}; }}
* {{ font-family: 'Segoe UI', sans-serif; font-size: {fp(14)}px; color: {TEXT}; }}
QMainWindow, QDialog {{ background: {BG}; }}
QWidget#card {{ background: {PANEL}; border: {p(1)}px solid {BORDER}; border-radius: {p(12)}px; }}
QLabel#title {{ font-size: {fp(18)}px; font-weight: 700; }}
QLabel#section {{ color: {MUTED}; font-size: {fp(11)}px; font-weight: 700; letter-spacing: {p(1)}px; }}
QLabel#chip {{ background: {PANEL2}; color: {ACCENT2}; border: {p(1)}px solid {BORDER};
              border-radius: {p(11)}px; padding: {p(3)}px {p(12)}px; font-weight: 600; }}
QTextEdit, QLineEdit, QScrollArea {{ background: {BG}; border: {p(1)}px solid {BORDER};
              border-radius: {p(10)}px; padding: {p(8)}px; }}
QTextEdit, QPlainTextEdit, QLineEdit {{ selection-background-color: #f4b740;
              selection-color: #101218; }}
QCheckBox::indicator {{ width: {p(16)}px; height: {p(16)}px; border: {p(1)}px solid {BORDER}; border-radius: {p(4)}px; background: {BG}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QWidget#radiohost {{ background: {BG}; }}
QRadioButton {{ padding: {p(8)}px {p(6)}px; spacing: {p(10)}px; border-radius: {p(6)}px; color: {TEXT}; background: transparent; }}
QRadioButton:hover {{ background: {PANEL2}; }}
QRadioButton::indicator {{ width: {p(16)}px; height: {p(16)}px; border-radius: {p(9)}px; }}
QRadioButton::indicator:unchecked {{ border: {p(2)}px solid {MUTED}; background: {BG}; }}
QRadioButton::indicator:checked {{ border: {p(2)}px solid {ACCENT}; background: {ACCENT}; }}
QComboBox QAbstractItemView {{ background: {PANEL2}; color: {TEXT}; selection-background-color: {ACCENT}; selection-color: white; }}
QTextEdit#terminal {{ font-family: 'Consolas','Courier New',monospace; font-size: {fp(12)}px; color: #b9c4d6; }}
QPushButton {{ background: {ACCENT}; border: none; border-radius: {p(10)}px; padding: {p(9)}px {p(14)}px; font-weight: 600; }}
QPushButton:hover {{ background: #3b7bff; }}
QPushButton:disabled {{ background: #34363f; color: {MUTED}; }}
QPushButton#ghost {{ background: {PANEL2}; text-align: left; }}
QPushButton#ghost:hover {{ background: #2d2f3a; }}
QPushButton#rec {{ background: {REC}; }}
QPushButton#scrollToBottom {{ background: {ACCENT}; color: #fff; border: none; border-radius: {p(6)}px; padding: 0; font-size: 14px; font-weight: bold; }}
QPushButton#scrollToBottom:hover {{ background: {ACCENT2}; color: #000; }}
QTabWidget::pane {{ border: {p(1)}px solid {BORDER}; border-radius: 0px {p(8)}px {p(8)}px {p(8)}px; top: -{p(1)}px; }}
QTabBar {{ qproperty-drawBase: 0; }}
QTabBar::tab {{ background: {PANEL2}; padding: {p(6)}px {p(10)}px; border-top-left-radius: {p(7)}px;
               border-top-right-radius: {p(7)}px; color: {MUTED}; margin-right: {p(2)}px; }}
QTabBar::tab:selected {{ background: {ACCENT}; color: white; margin-bottom: -{p(1)}px;
                         padding-bottom: {p(7)}px; }}
QTabBar::tab:hover:!selected {{ background: #2d2f3a; color: {TEXT}; }}
QTabWidget#pages > QTabBar {{ font-size: {p(14)}px; font-weight: bold; }}
QTabWidget#pages > QTabBar::tab {{ padding: {p(9)}px {p(22)}px; }}
QComboBox::drop-down {{ border: none; width: {p(22)}px; }}
QComboBox::down-arrow {{ image: none; width: 0; height: 0;
                         border-left: {p(5)}px solid transparent;
                         border-right: {p(5)}px solid transparent;
                         border-top: {p(6)}px solid {MUTED}; }}
QSpinBox::up-button, QSpinBox::down-button {{ background: {PANEL2}; border: none;
                                              width: {p(18)}px; border-radius: {p(4)}px; }}
QSpinBox::up-arrow {{ image: none; width: 0; height: 0;
                      border-left: {p(4)}px solid transparent;
                      border-right: {p(4)}px solid transparent;
                      border-bottom: {p(5)}px solid {MUTED}; }}
QSpinBox::down-arrow {{ image: none; width: 0; height: 0;
                        border-left: {p(4)}px solid transparent;
                        border-right: {p(4)}px solid transparent;
                        border-top: {p(5)}px solid {MUTED}; }}
QStatusBar {{ background: {PANEL}; color: {MUTED}; }}
QWidget#titlebar {{ background: {PANEL}; border-bottom: {p(1)}px solid {BORDER}; }}
QLabel#titletext {{ font-weight: 700; font-size: {fp(14)}px; }}
QPushButton#winbtn {{ background: transparent; border: none; border-radius: {p(5)}px;
                     color: {TEXT}; font-size: {fp(14)}px; font-weight: 400;
                     min-width: {p(40)}px; min-height: {p(28)}px; padding: 0px; }}
QPushButton#winbtn:hover {{ background: {PANEL2}; }}
QPushButton#winclose {{ background: transparent; border: none; border-radius: {p(5)}px;
                       color: {TEXT}; font-size: {fp(14)}px; font-weight: 400;
                       min-width: {p(40)}px; min-height: {p(28)}px; padding: 0px; }}
QPushButton#winclose:hover {{ background: {REC}; color: white; }}
QProgressBar {{ background: {BG}; border: {p(1)}px solid {BORDER}; border-radius: {p(5)}px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: {p(5)}px; }}
QComboBox, QSpinBox {{ background: {BG}; border: {p(1)}px solid {BORDER}; border-radius: {p(8)}px; padding: {p(5)}px; }}
QScrollBar:vertical {{ background: {BG}; width: {p(10)}px; border-radius: {p(5)}px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: {p(5)}px; min-height: {p(24)}px; }}
QScrollBar::handle:vertical:hover {{ background: {MUTED}; }}
QScrollBar:horizontal {{ background: {BG}; height: {p(10)}px; border-radius: {p(5)}px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {BORDER}; border-radius: {p(5)}px; min-width: {p(24)}px; }}
QScrollBar::handle:horizontal:hover {{ background: {MUTED}; }}
/* Hide the native arrow buttons + page regions — left unstyled they render WHITE on Windows. */
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0; height: 0; background: none; border: none; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
QToolButton {{ background: {PANEL2}; border: {p(1)}px solid {BORDER}; border-radius: {p(4)}px; color: {TEXT}; }}
QToolButton:hover {{ background: {ACCENT}; }}
QToolButton:disabled {{ background: {PANEL}; color: {MUTED}; }}
QMenu {{ background: {PANEL2}; color: {TEXT}; border: {p(1)}px solid {BORDER};
         border-radius: {p(8)}px; padding: {p(4)}px; }}
QMenu::item {{ background: transparent; color: {TEXT}; padding: {p(6)}px {p(22)}px {p(6)}px {p(14)}px; border-radius: {p(5)}px; }}
QMenu::item:selected {{ background: {ACCENT}; color: white; }}
QMenu::item:disabled {{ color: {MUTED}; }}
QMenu::separator {{ height: {p(1)}px; background: {BORDER}; margin: {p(4)}px {p(8)}px; }}
QToolTip {{ background: {PANEL2}; color: {TEXT}; border: {p(1)}px solid {BORDER};
            border-radius: {p(6)}px; padding: {p(4)}px {p(8)}px; font-size: {fp(13)}px; }}
QPlainTextEdit {{ background: {BG}; border: {p(1)}px solid {BORDER}; border-radius: {p(8)}px;
                  padding: {p(6)}px; color: {TEXT}; }}
QTableWidget, QListWidget {{ background: {BG}; color: {TEXT}; border: {p(1)}px solid {BORDER};
                  border-radius: {p(8)}px; gridline-color: {BORDER};
                  alternate-background-color: {PANEL2}; }}
QTableWidget::item, QListWidget::item {{ padding: {p(4)}px {p(6)}px; color: {TEXT}; }}
QTableWidget::item:alternate {{ background: {PANEL2}; color: {TEXT}; }}
QTableWidget::item:selected, QListWidget::item:selected {{ background: {ACCENT}; color: #ffffff; }}
QHeaderView::section {{ background: {PANEL2}; color: {MUTED}; border: none;
                  border-right: {p(1)}px solid {BORDER}; padding: {p(5)}px {p(8)}px;
                  font-weight: 600; }}
QTableCornerButton::section {{ background: {PANEL2}; border: none; }}
QSlider::groove:horizontal {{ height: {p(5)}px; background: {PANEL2}; border-radius: {p(2)}px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: {p(16)}px; margin: -{p(6)}px 0;
                  border-radius: {p(8)}px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: {p(2)}px; }}
QMessageBox, QInputDialog {{ background: {BG}; }}
"""


def _esc(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")


def _shadow(widget, blur=24, dy=4):
    eff = QGraphicsDropShadowEffect(widget)
    eff.setBlurRadius(blur); eff.setXOffset(0); eff.setYOffset(dy)
    eff.setColor(QColor(0, 0, 0, 140))
    widget.setGraphicsEffect(eff)


def _card(layout_cls=QVBoxLayout):
    w = QWidget(); w.setObjectName("card")
    lay = layout_cls(w); lay.setContentsMargins(px(14), px(14), px(14), px(14)); lay.setSpacing(px(10))
    return w, lay


def _cap_w(n: int, frac: float = 0.30) -> int:
    """A px()-scaled minimum WIDTH, capped to a fraction of the screen.

    Same failure as the heights (see _scroll_page): three panes each demanding
    px(360) at UI scale 4 want 4320 px of window, more than the display has, and
    Qt would rather overflow the screen than go under a layout minimum.
    """
    try:
        avail = QApplication.primaryScreen().availableGeometry().width()
        return max(px(80), min(px(n), int(avail * frac)))
    except Exception:
        return px(n)


def _scroll_page(widget: QWidget) -> QWidget:
    """Wrap a tab page in a scroll area so its minimum size can never blow up the window.

    A widget's minimum size is the sum of its children's minimums, and every one of
    those is px()-scaled — so at a large UI scale (a 4K TV at 250-300%) a tab's
    minimum height grows past the screen. Qt refuses to shrink a window below its
    layout minimum, so the window ends up TALLER AND WIDER THAN THE DISPLAY and
    Windows simply crops it: the Settings button and the bottom of the command
    column fall off the edges, with no scrollbar anywhere to reach them.

    A QScrollArea reports a small minimum of its own (a couple of scrollbar widths)
    while letting the page keep its natural size inside, which is exactly the
    behaviour wanted here: the panel stays as big as it wants and you scroll it.
    The command column already worked this way; this extends it to every tab.

    Widgets that already scroll (QScrollArea, QTextEdit, …) are returned untouched.
    """
    if isinstance(widget, (QScrollArea, QAbstractScrollArea)):
        return widget
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    # A QScrollArea paints its viewport with the palette's Base role — WHITE in this
    # dark theme, which is what turned the wrapped tabs into a white sheet. The
    # viewport is a plain QWidget, so the app QSS (which styles named widgets, not
    # anonymous viewports) never reached it: paint it the panel colour explicitly and
    # let the page inside keep its own background.
    area.setStyleSheet(
        f"QScrollArea {{ border:none; background:{PANEL}; padding:0; }}"
        f"QScrollArea > QWidget > QWidget {{ background:transparent; }}")
    area.viewport().setAutoFillBackground(False)
    area.setWidget(widget)
    area.inner = widget                 # so callers can still reach the real page
    return area


def _section(text: str) -> QLabel:
    lbl = QLabel(); lbl.setObjectName("section")
    lbl.setProperty("_i18n_upper", True)    # translated first, then capitals (live switch too)
    lbl.setText(text)
    if lbl.text() == text:                  # Qt not wrapped (a bare test): capitals here
        lbl.setText(text.upper())
    return lbl


class FlowLayout(QLayout):
    """A layout that lays widgets out left-to-right and WRAPS to the next line
    when it runs out of horizontal space (canonical Qt FlowLayout port).

    This is what makes the Memory Center action bars reflow instead of squeezing
    every button into one row until the labels truncate to "Du / Pir / Mc" — at
    any window width / DPI scale each button keeps its full natural size and the
    row simply wraps.
    """

    def __init__(self, parent=None, margin=0, hspacing=-1, vspacing=-1):
        super().__init__(parent)
        self._items = []
        self._hspace = hspacing
        self._vspace = vspacing
        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)

    def __del__(self):
        while self._items:
            self._items.pop()

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientations(Qt.Horizontal)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        size += QSize(m.left() + m.right(), m.top() + m.bottom())
        return size

    def _space(self, horizontal=True):
        sp = self._hspace if horizontal else self._vspace
        if sp >= 0:
            return sp
        return px(8)

    def _do_layout(self, rect, test_only):
        m = self.contentsMargins()
        x = rect.x() + m.left()
        y = rect.y() + m.top()
        line_height = 0
        right = rect.right() - m.right()
        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + self._space(True)
            if next_x - self._space(True) > right and line_height > 0:
                x = rect.x() + m.left()
                y = y + line_height + self._space(False)
                next_x = x + hint.width() + self._space(True)
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y() + m.bottom()


class _FlowWidget(QWidget):
    """A container that hosts a FlowLayout and correctly reports its wrapped
    height to the parent layout, so wrapped rows are never clipped.

    A bare QWidget hosting a FlowLayout only reserves one row's height (its
    sizeHint ignores wrapping), which clips the buttons that wrapped onto a
    second line. This subclass forwards heightForWidth to the layout and enables
    the height-for-width size policy so the parent VBox allocates the full height.
    """

    def __init__(self, flow: FlowLayout):
        super().__init__()
        self.setLayout(flow)
        sp = self.sizePolicy()
        sp.setHorizontalPolicy(QSizePolicy.Preferred)
        sp.setVerticalPolicy(QSizePolicy.Minimum)
        sp.setHeightForWidth(True)
        self.setSizePolicy(sp)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, w):
        return self.layout().heightForWidth(w)

    def sizeHint(self):
        w = self.width() or 400
        return QSize(w, self.layout().heightForWidth(w))

    def minimumSizeHint(self):
        w = self.width() or 200
        return QSize(0, self.layout().heightForWidth(w))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        # re-evaluate wrapped height when the width changes
        self.updateGeometry()


def _flow(margin=0, spacing=-1) -> FlowLayout:
    return FlowLayout(margin=margin, hspacing=spacing, vspacing=spacing)


def _fit_dialog(dlg, w: int, h: int) -> None:
    """Size a dialog, but never larger than the screen's work area.

    Dialogs asked for a fixed px()-scaled size, which is a physical pixel count:
    at UI Scale 150% on a TV, `px(600) x px(660)` is 900x990 and simply does not
    fit, so the pinned OK/Cancel row fell off the bottom edge and the dialog could
    not be dismissed with the mouse. Clamp to the available geometry (which already
    excludes the taskbar) and centre what is left.
    """
    try:
        avail = QApplication.primaryScreen().availableGeometry()
        w = min(w, int(avail.width() * 0.96))
        h = min(h, int(avail.height() * 0.92))
        dlg.setMaximumSize(avail.width(), avail.height())
        dlg.resize(w, h)
        dlg.move(avail.x() + (avail.width() - w) // 2,
                 avail.y() + max(0, (avail.height() - h) // 2))
    except Exception:
        logger.exception("dialog fit failed; using the requested size")
        dlg.resize(w, h)


def _fmt_ts(ts) -> str:
    try:
        ts = float(ts)
        if ts <= 0:
            return "—"
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
    except Exception:
        return "—"


def _fmt_bytes(n: int) -> str:
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


# --------------------------------------------------------------------------- #
# Cross-tab plumbing.
#
# These two are not theme or layout, but they are shared the same way and for
# the same reason: _ScopedCtx is needed by four tabs and TranscribeWorker by
# two, so leaving either in gui.py would force every extracted tab module to
# import gui.py — exactly the cycle the split is undoing.
#
# transcribe_audio_array is imported INSIDE run() to keep this module light:
# it is imported by every tab, and pulling in the audio stack at import time
# would cost every one of them.
# --------------------------------------------------------------------------- #
from PyQt5.QtCore import QThread, pyqtSignal


class TranscribeWorker(QThread):
    """Transcribe one audio utterance WITHOUT invoking the agent.

    Used by the hands-free VAD mode when Ultra Search is ON: the spoken phrase
    must become a research topic (like typed text would), not a normal agent
    turn, so transcription has to finish before routing.
    """
    recognized = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, ctx, audio):
        super().__init__()
        self.ctx, self.audio = ctx, audio

    def run(self):
        try:
            from audio import transcribe_audio_array
            self.ctx.set_stage("Transcribing")
            text = transcribe_audio_array(self.ctx, self.audio)
            if text:
                self.recognized.emit(text)
            else:
                self.failed.emit("Didn't catch that — try again.")
        except Exception as exc:
            logger.exception("Transcription failed")
            self.failed.emit(str(exc))


class _ScopedCtx:
    """A private view of the assistant context with its OWN cancel token.

    Everything is delegated to the real ctx EXCEPT cancellation. `ctx.cancel_event`
    is global and is set by the main chat's Stop button, then cleared only when the
    main chat starts its next turn — so a single Stop press used to mute the whole
    Madhouse permanently: every reply returned None and the room printed "the model
    returned no text" forever (diagnosed 2026-07-22; a set cancel flag is the only
    state in which send_to_lm_studio returns None here).

    The same flag collision runs the other way too: an operation that *clears* the
    global flag when it starts or finishes un-cancels whatever else is in flight.
    Any tab that runs its own long job therefore takes a scoped view instead of
    touching ctx.cancel_event directly, and the main Stop button sets every scoped
    token as well as the global one (see AssistantWindow._cancel_current).
    """

    def __init__(self, ctx, cancel_event):
        object.__setattr__(self, "_ctx", ctx)
        object.__setattr__(self, "cancel_event", cancel_event)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_ctx"), name)

    def __setattr__(self, name, value):
        # Writes (e.g. throttle bookkeeping) belong to the real context.
        setattr(object.__getattribute__(self, "_ctx"), name, value)

    def is_cancelled(self) -> bool:
        return object.__getattribute__(self, "cancel_event").is_set()

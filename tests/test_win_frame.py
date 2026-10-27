"""The frameless window gets its native behaviour back (2026-10-27).

Resize from every edge and corner, Aero Snap (Win+arrows, drag to an edge, four
windows on one screen), a REAL maximize -- via the win32 frame styles with the
drawn frame eaten (gui_win_frame). One-tap placements sit in the Layout menu.
Offscreen here: the win32 parts are checked through their pure math and their
guards, the placements against the real screen's work area.

Run: QT_QPA_PLATFORM=offscreen venv/Scripts/python.exe tests/test_win_frame.py
"""
import os
import sys
import types
import ctypes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("F5_TEST_RUN", "1")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import logging; logging.basicConfig(level=logging.CRITICAL)

import _isolate_library  # noqa: F401
from gui_fakes import NoopLoader
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QRect
import gui
import gui_win_frame as WF

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print("PASS  " + name)
    else:
        BAD += 1; print("FAIL  " + name + "   " + str(extra)[:200])


app = QApplication.instance() or QApplication([])

# ── the placement table: halves, quarters, fill ──────────────────────────────
check("seven one-tap placements", len(gui.AssistantWindow._SNAP_LAYOUTS) == 7,
      gui.AssistantWindow._SNAP_LAYOUTS)

w = gui.AssistantWindow("qwen3-9b", True, "high", loader=NoopLoader)
w.player = types.SimpleNamespace(stop=lambda: None)
w.resize(900, 600)
w.show()

# offscreen: the win32 styles must not even be attempted
check("offscreen installs nothing", w._win_native is False)
check("offscreen frame messages pass through", w._win_frame_message(12345) is None)

avail = w._screen_area()
# Offscreen's 800x600 screen is smaller than the window's layout minimum, so Qt
# may refuse to shrink; the CONTRACT is the rectangle the snap asked for.
asks = []
_orig_set_geo = w.setGeometry
def _spy_set_geo(*a):
    asks.append(QRect(*a) if len(a) == 4 else QRect(a[0]))
    return _orig_set_geo(*a)
w.setGeometry = _spy_set_geo

def snap_ask(fx, fy, fw, fh):
    """The rectangle the SNAP itself asked for (the first setGeometry call;
    _clamp_to_screen may follow with its own correction)."""
    n = len(asks)
    w._snap_window(fx, fy, fw, fh)
    return asks[n] if len(asks) > n else None


check("left half asks for the left work area",
      (lambda r: r and abs(r.x() - avail.x()) <= 1
       and abs(r.width() - avail.width() // 2) <= 1
       and abs(r.height() - avail.height()) <= 1)(snap_ask(0, 0, 0.5, 1.0)),
      (asks, avail))
check("bottom-right quarter asks exactly",
      (lambda r: r and abs(r.x() - (avail.x() + avail.width() // 2)) <= 1
       and abs(r.y() - (avail.y() + avail.height() // 2)) <= 1
       and abs(r.width() - avail.width() // 2) <= 1
       and abs(r.height() - avail.height() // 2) <= 1)(snap_ask(0.5, 0.5, 0.5, 0.5)),
      (asks, avail))
check("fill covers the work area",
      (lambda r: r and r.contains(avail) and avail.contains(r))(snap_ask(0, 0, 1.0, 1.0)),
      (asks, avail))
w.setGeometry = _orig_set_geo
check("a snap while faux-maximized leaves faux-max",
      (setattr(w, "_faux_max", True), w._snap_window(0, 0, 0.5, 1.0),
       w._faux_max is False)[2])
w.showNormal(); w.resize(900, 600)

# ── the pure HT math, against a stubbed user32 ───────────────────────────────
class _Rect:
    def __init__(self, l, t, r, b):
        self.left, self.top, self.right, self.bottom = l, t, r, b


WIN = _Rect(100, 200, 1100, 800)      # 1000x600


class _User32:
    def __init__(self, rect=WIN):
        self.rect = rect
    def GetWindowRect(self, hwnd, ref):
        r = ctypes.cast(ref, ctypes.POINTER(ctypes.wintypes.RECT)).contents
        r.left, r.top, r.right, r.bottom = self.rect.left, self.rect.top, self.rect.right, self.rect.bottom
        return 1
    def GetSystemMetrics(self, m):
        return 4                          # a slim border band


class _Msg:
    def __init__(self, x, y, hwnd=7):
        self.hwnd, self.message = hwnd, WF.WM_NCHITTEST
        self.wParam, self.lParam = 0, ((y & 0xFFFF) << 16) | (x & 0xFFFF)


user32 = _User32()
caption = lambda x, y: 40 <= y <= 80          # a titlebar band, local px
cases = [                                     # (x, y, expected)
    (WIN.left + 2,     WIN.top + 300,  WF.HTLEFT),
    (WIN.right - 3,    WIN.top + 300,  WF.HTRIGHT),
    (WIN.left + 500,   WIN.top + 2,    WF.HTTOP),
    (WIN.left + 500,   WIN.bottom - 2, WF.HTBOTTOM),
    (WIN.left + 2,     WIN.top + 2,    WF.HTTOPLEFT),
    (WIN.right - 2,    WIN.top + 2,    WF.HTTOPRIGHT),
    (WIN.left + 2,     WIN.bottom - 2, WF.HTBOTTOMLEFT),
    (WIN.right - 2,    WIN.bottom - 2, WF.HTBOTTOMRIGHT),
    (WIN.left + 500,   WIN.top + 60,   WF.HTCAPTION),   # our titlebar
    (WIN.left + 500,   WIN.top + 300,  0),              # plain content
]
for x, y, want in cases:
    check("hit %-22s -> %s" % ((x - WIN.left, y - WIN.top), want or "Qt"),
          WF.hit(_Msg(x, y), caption, user32=user32) == want,
          WF.hit(_Msg(x, y), caption, user32=user32))
check("negative screen coords stay in the window",
      WF.hit(_Msg(-1098, -500), caption, user32=_User32(_Rect(-1100, -800, -100, -200))) == WF.HTLEFT)
nccalcs = types.SimpleNamespace(hwnd=7, message=WF.WM_NCCALCSIZE, wParam=1, lParam=0)
check("NCCALCSIZE with wParam eats the frame area", WF.remove_frame_area(nccalcs, user32=user32))
nccalcs0 = types.SimpleNamespace(hwnd=7, message=WF.WM_NCCALCSIZE, wParam=0, lParam=0)
check("NCCALCSIZE without wParam passes", not WF.remove_frame_area(nccalcs0, user32=user32))

w.close()
print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)

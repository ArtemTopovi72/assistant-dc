"""A frameless Qt window with the FULL native window behaviour.

The frameless build lost more than decoration: resizing worked only through
the status-bar grip (one corner), Windows Snap (Win+arrows, drag to an edge or
a corner, four windows on one screen) ignored the window, and "maximize" was a
manual geometry fill. The classic cure, done here: keep the native frame
styles on the window -- WS_THICKFRAME | WS_CAPTION | WS_SYSMENU |
WS_MINIMIZEBOX | WS_MAXIMIZEBOX -- and eat WM_NCCALCSIZE so nothing of the
native frame is ever DRAWN. Windows still supplies everything the owner asked
for: resize from every edge and corner, Aero Snap with Snap Assist, a real
maximize that respects the taskbar, Alt+Space, animated minimize.

WM_NCHITTEST maps our own titlebar to HTCAPTION (native move/drag-snap) and
the window's edges to the HT* resize codes; interactive children of the bar
(buttons) stay HTCLIENT, so clicks keep working.

Offscreen builds and suites: nothing here applies -- install() refuses a
window whose styles cannot be read, and the callers keep their manual
fallbacks (single-corner grip, faux-maximize).
"""
import ctypes
import os
from ctypes import wintypes

GWL_STYLE = -16
WS_SYSMENU = 0x00080000
WS_THICKFRAME = 0x00040000
WS_CAPTION = 0x00C00000
WS_MINIMIZEBOX = 0x00020000
WS_MAXIMIZEBOX = 0x00010000
WM_NCCALCSIZE = 0x0083
WM_NCHITTEST = 0x0084
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOZORDER = 0x0004
SWP_FRAMECHANGED = 0x0020
SWP_NOOWNERZORDER = 0x0200
SWP_NOACTIVATE = 0x0010
SM_CXSIZEFRAME = 32
SM_CXPADDEDBORDER = 92
HTCLIENT = 1
HTCAPTION = 2
HTLEFT, HTRIGHT, HTTOP, HTTOPLEFT, HTTOPRIGHT = 10, 11, 12, 13, 14
HTBOTTOM, HTBOTTOMLEFT, HTBOTTOMRIGHT = 15, 16, 17

_user32 = ctypes.windll.user32 if os.name == "nt" else None


def available() -> bool:
    return _user32 is not None


def install(hwnd: int) -> bool:
    """Add the native frame styles a frameless Qt window lacks. True when they
    took. False on anything else (offscreen, a suite's stub HWND, a refusal):
    callers then keep their manual single-corner / faux-maximize fallbacks."""
    if not available() or not hwnd:
        return False
    try:
        style = _user32.GetWindowLongW(hwnd, GWL_STYLE)
        if not style:                      # not a real window (offscreen) -- hands off
            return False
        _user32.SetWindowLongW(hwnd, GWL_STYLE,
                               style | WS_SYSMENU | WS_THICKFRAME | WS_CAPTION
                               | WS_MINIMIZEBOX | WS_MAXIMIZEBOX)
        _user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0,
                             SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER
                             | SWP_NOOWNERZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED)
        return bool(_user32.GetWindowLongW(hwnd, GWL_STYLE) & WS_THICKFRAME)
    except Exception:
        return False


class Msg:
    """The fields nativeEvent needs from an MSG, read at the message's address."""

    def __init__(self, address: int):
        (self.hwnd, self.message, self.wParam, self.lParam) = (
            ctypes.cast(ctypes.c_void_p(address),
                        ctypes.POINTER(wintypes.MSG)).contents.hwnd,
            ctypes.cast(ctypes.c_void_p(address),
                        ctypes.POINTER(wintypes.MSG)).contents.message,
            ctypes.cast(ctypes.c_void_p(address),
                        ctypes.POINTER(wintypes.MSG)).contents.wParam,
            ctypes.cast(ctypes.c_void_p(address),
                        ctypes.POINTER(wintypes.MSG)).contents.lParam)


def remove_frame_area(msg: Msg) -> bool:
    """True for the WM_NCCALCSIZE that draws the native frame: the caller eats
    it and the client area covers the whole window."""
    return _user32 is not None and msg.message == WM_NCCALCSIZE and msg.wParam


def hit(msg: Msg, caption) -> int:
    """The HT* code for a WM_NCHITTEST, or 0 to let Qt decide.

    `caption(x, y)` gets the point in the window's LOCAL physical pixels and
    says whether OUR titlebar sits there (interactive children answered
    HTCLIENT before it is ever asked)."""
    if _user32 is None or not hasattr(msg, "lParam"):
        return 0
    x = msg.lParam & 0xFFFF
    y = (msg.lParam >> 16) & 0xFFFF
    if x >= 0x8000:                 # LOWORD/SIGNED: negative screen coordinates
        x -= 0x10000
    if y >= 0x8000:
        y -= 0x10000
    rect = wintypes.RECT()
    if not _user32.GetWindowRect(msg.hwnd, ctypes.byref(rect)):
        return 0
    b = max(6, _user32.GetSystemMetrics(SM_CXSIZEFRAME)
            + _user32.GetSystemMetrics(SM_CXPADDEDBORDER))
    l, t = x - rect.left, y - rect.top
    w, h = rect.right - rect.left, rect.bottom - rect.top
    if w <= 2 * b or h <= 2 * b:    # tiny window: only the caption, no border band
        return HTCAPTION if caption(l, t) else 0
    left, right, top, bottom = l < b, l >= w - b, t < b, t >= h - b
    if top and left:
        return HTTOPLEFT
    if top and right:
        return HTTOPRIGHT
    if bottom and left:
        return HTBOTTOMLEFT
    if bottom and right:
        return HTBOTTOMRIGHT
    if left:
        return HTLEFT
    if right:
        return HTRIGHT
    if top:
        return HTTOP
    if bottom:
        return HTBOTTOM
    return HTCAPTION if caption(l, t) else 0

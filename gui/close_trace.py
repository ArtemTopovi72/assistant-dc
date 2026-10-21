"""Who closed the main window: the window messages that preceded closeEvent, the
foreground window's process and how long the user had been idle. The window went
on closing with spontaneous=True about 11 minutes after start while nobody touched
it (2026-10-21 15:28, 19:50, 20:35); Qt only says the close came from outside.
"""
from __future__ import annotations

import ctypes
import os
import sys
import time

WM_CLOSE, WM_QUERYENDSESSION, WM_ENDSESSION, WM_SYSCOMMAND = 0x0010, 0x0011, 0x0016, 0x0112
SC_CLOSE = 0xF060
_NAMES = {WM_CLOSE: "WM_CLOSE", WM_QUERYENDSESSION: "WM_QUERYENDSESSION", WM_ENDSESSION: "WM_ENDSESSION"}
_seen: list = []


def note(message) -> None:
    """nativeEvent hook: remember close-related messages (cheap, never raises)."""
    if sys.platform != "win32":
        return
    try:
        from ctypes import wintypes
        msg = wintypes.MSG.from_address(int(message))
        if msg.message in _NAMES:
            _seen.append((time.time(), _NAMES[msg.message], msg.lParam))
        elif msg.message == WM_SYSCOMMAND and (msg.wParam & 0xFFF0) == SC_CLOSE:
            # ✕, Alt+F4, the taskbar's Close all arrive as SC_CLOSE first;
            # a WM_CLOSE posted by another program does not.
            _seen.append((time.time(), "SC_CLOSE (✕ / Alt+F4 / taskbar)", msg.lParam))
        del _seen[:-10]
    except Exception:
        pass


def _process_name(pid: int) -> str:
    k = ctypes.windll.kernel32
    h = k.OpenProcess(0x1000, False, pid)        # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return "?"
    try:
        buf, n = ctypes.create_unicode_buffer(520), ctypes.c_uint(520)
        return os.path.basename(buf.value) if k.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(n)) else "?"
    finally:
        k.CloseHandle(h)


def describe() -> str:
    if sys.platform != "win32":
        return ""
    parts = []
    now = time.time()
    msgs = [f"{name} {now - t:.1f}s ago" for t, name, _ in _seen if now - t < 30]
    parts.append("messages: " + (", ".join(msgs) or "none seen (closed without a window message)"))
    try:
        from ctypes import wintypes
        u = ctypes.windll.user32
        hwnd = u.GetForegroundWindow()
        title = ctypes.create_unicode_buffer(256)
        u.GetWindowTextW(hwnd, title, 256)
        pid = wintypes.DWORD()
        u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        parts.append(f"foreground: {_process_name(pid.value)} pid {pid.value} «{title.value[:60]}»")

        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]
        li = LASTINPUTINFO(ctypes.sizeof(LASTINPUTINFO), 0)
        if u.GetLastInputInfo(ctypes.byref(li)):
            parts.append(f"user idle {(ctypes.windll.kernel32.GetTickCount() - li.dwTime) / 1000:.0f}s")
    except Exception as exc:
        parts.append(f"(foreground unknown: {exc})")
    return "; ".join(parts)

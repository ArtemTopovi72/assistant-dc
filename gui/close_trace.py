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


def _flags(lparam: int) -> str:
    """WM_QUERYENDSESSION / WM_ENDSESSION lParam: who asked."""
    lparam = int(lparam or 0)
    named = [n for bit, n in ((0x1, "Restart Manager: an installer or updater wants a file we hold"),
                              (0x80000000, "logoff"), (0x40000000, "forced")) if lparam & bit]
    return ", ".join(named) or "shutdown or restart"


def _recent_processes(window_s: float = 120.0) -> str:
    try:
        import psutil
        now = time.time()
        new = sorted((p.info["create_time"], p.info["name"], p.info["pid"])
                     for p in psutil.process_iter(["name", "pid", "create_time"])
                     if now - (p.info["create_time"] or 0) < window_s)
        return f"started in the last {window_s:.0f}s: " + (", ".join(f"{n} pid {pid}" for _, n, pid in new[-12:]) or "none")
    except Exception as exc:
        return f"(processes unknown: {exc})"


def describe() -> str:
    if sys.platform != "win32":
        return ""
    parts = []
    now = time.time()
    msgs = [f"{name} {now - t:.1f}s ago" + (f" [{_flags(lp)}]" if "SESSION" in name else "")
            for t, name, lp in _seen if now - t < 30]
    parts.append("messages: " + (", ".join(msgs) or "none seen (closed without a window message)"))
    if any("SESSION" in name and now - t < 30 for t, name, _ in _seen):
        parts.append(_recent_processes())
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


def _windows_ending() -> bool:
    return bool(ctypes.windll.user32.GetSystemMetrics(0x2000))     # SM_SHUTTINGDOWN


def refuse_end_session(message, ending=None) -> bool:
    """Refuse an app-only Restart Manager close, preserving shutdown/logoff and cancellation."""
    if sys.platform != "win32":
        return False
    try:
        from ctypes import wintypes
        msg = wintypes.MSG.from_address(int(message))
        if msg.message not in (WM_QUERYENDSESSION, WM_ENDSESSION):
            return False
        if msg.message == WM_ENDSESSION and not msg.wParam:
            return False
        if not (msg.lParam & 0x1) or msg.lParam & 0x80000000:
            return False
        return not (_windows_ending() if ending is None else ending())
    except Exception:
        return False


def install_end_session_guard(app) -> None:
    """Catch app-only session requests to ALL Qt windows, including internal ones."""
    if sys.platform != 'win32' or getattr(app, '_end_session_guard', None) is not None:
        return
    import logging
    from PyQt5.QtCore import QAbstractNativeEventFilter

    class Guard(QAbstractNativeEventFilter):
        def nativeEventFilter(self, event_type, message):
            if refuse_end_session(message):
                note(message)
                logging.getLogger('assistant.gui').warning(
                    'LIFECYCLE: app-only end-session request refused before Qt session management (%s)',
                    describe())
                return True, 0
            return False, 0

    # A per-widget nativeEvent cannot protect Qt's other/hidden top-level windows.
    app._end_session_guard = Guard()
    app.installNativeEventFilter(app._end_session_guard)

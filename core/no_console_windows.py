"""Stop child console windows flashing over the GUI.

The app runs under pythonw (no console). Every console program it shells
out to -- `lms load/unload` on each render (card exclusivity), ffmpeg, git --
then gets a console of its own, so a black window flashes on the screen at
every model swap. CREATE_NO_WINDOW on the Popen suppresses that; instead of
threading the flag through twenty call sites (and forgetting the next one),
install() makes it the default for every Popen in this process.

Left alone: a Popen that asks for a console on purpose (CREATE_NEW_CONSOLE),
one detached into its own session (DETACHED_PROCESS -- the GUI apps
launch_all starts), or one that brings its own STARTUPINFO.
"""
from __future__ import annotations

import subprocess
import sys

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
_NEW_CONSOLE = getattr(subprocess, "CREATE_NEW_CONSOLE", 0x00000010)
_DETACHED = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
_installed = False


def _has_console() -> bool:
    try:
        import ctypes
        return bool(ctypes.windll.kernel32.GetConsoleWindow())
    except Exception:
        return True


def wants_hiding(kwargs: dict) -> bool:
    """Whether a Popen with these kwargs should get CREATE_NO_WINDOW."""
    flags = int(kwargs.get("creationflags", 0) or 0)
    if flags & (_NEW_CONSOLE | _DETACHED | _NO_WINDOW):
        return False
    if kwargs.get("startupinfo") is not None:
        return False
    return True


def install(force: bool = False) -> bool:
    """Make CREATE_NO_WINDOW the default for subprocess.Popen. Idempotent.

    Only under Windows with no console attached (pythonw) unless `force`;
    a console run keeps its children's output visible.
    """
    global _installed
    if _installed:
        return True
    if sys.platform != "win32":
        return False
    if not force and _has_console():
        return False
    orig_init = subprocess.Popen.__init__

    def _init(self, *args, **kwargs):
        if wants_hiding(kwargs):
            kwargs["creationflags"] = int(kwargs.get("creationflags", 0) or 0) | _NO_WINDOW
        return orig_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = _init
    _installed = True
    return True

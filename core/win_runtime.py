"""Load the NEWEST msvcp140.dll before anything else does.

PyQt5 ships its own MSVCP140.dll 14.26 in PyQt5/Qt5/bin. On a machine without a
fresh Visual C++ runtime that copy is the one every module gets, and torch &
co., built against 14.3x, crash in it: a friend's clone died about a minute
after start, «python.exe ... Faulting module MSVCP140.dll 14.26 ... 0xc0000005»
(live 10-03). Windows reuses an already-loaded module of the same name, so
loading the newest copy first makes everyone use it.
"""
import ctypes
import glob
import os
import sys


def _version(path: str) -> tuple:
    try:
        ver = ctypes.windll.version
        size = ver.GetFileVersionInfoSizeW(path, None)
        if not size:
            return ()
        buf = ctypes.create_string_buffer(size)
        ver.GetFileVersionInfoW(path, 0, size, buf)
        p, n = ctypes.c_void_p(), ctypes.c_uint()
        ver.VerQueryValueW(buf, "\\", ctypes.byref(p), ctypes.byref(n))
        ms, ls = ctypes.cast(p, ctypes.POINTER(ctypes.c_uint32))[2:4]
        return (ms >> 16, ms & 0xFFFF, ls >> 16, ls & 0xFFFF)
    except Exception:
        return ()


def candidates() -> list:
    out = [os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "msvcp140.dll")]
    for sp in (p for p in sys.path if p.endswith("site-packages")):
        out += glob.glob(os.path.join(sp, "*", "msvcp140.dll"))
        out += glob.glob(os.path.join(sp, "*", "*", "msvcp140.dll"))
        out += glob.glob(os.path.join(sp, "*", "*", "*", "msvcp140.dll"))
        out += glob.glob(os.path.join(sp, "*", ".libs", "msvcp140.dll"))     # glob skips dot-dirs
        out += glob.glob(os.path.join(sp, "*.libs", "msvcp140.dll"))         # numpy.libs-style
    return [p for p in dict.fromkeys(out) if os.path.isfile(p)]


def newest(paths: list) -> str:
    best = max(paths, key=_version, default="")
    return best if best and _version(best) else ""


def preload_newest_msvcp() -> str:
    """Load the newest msvcp140.dll found; returns its path ("" when none/not Windows)."""
    if os.name != "nt":
        return ""
    best = newest(candidates())
    if best:
        try:
            ctypes.WinDLL(best)
        except OSError:
            return ""
    return best

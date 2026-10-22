"""The close trace names WHO asked Windows to end the session: the window closed itself at 03:33 and
13:52 on WM_QUERYENDSESSION with nothing in the event logs (2026-10-22), and the lParam flags that
tell a Restart Manager close (an installer holding our file) from a logoff were caught but never written.

Run: venv/Scripts/python.exe tests/test_close_trace.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gui"))
import close_trace as C

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print("PASS  " + name)
    else:
        BAD += 1; print("FAIL  " + name + "   " + str(extra)[:300])


check("lParam 0x1 is the Restart Manager", "Restart Manager" in C._flags(0x1), C._flags(0x1))
check("lParam 0x80000000 is a logoff", "logoff" in C._flags(0x80000000), C._flags(0x80000000))
check("lParam 0x40000000 is a forced close", "forced" in C._flags(0x40000000), C._flags(0x40000000))
check("lParam 0 is a shutdown or restart", "shutdown" in C._flags(0), C._flags(0))

C._seen[:] = [(time.time(), "WM_QUERYENDSESSION", 0x1)]
d = C.describe()
if sys.platform == "win32":
    check("the log line carries the flags", "Restart Manager" in d, d)
    check("an end-session close lists what started just before it", "started in the last" in d, d)

if sys.platform == "win32":
    # A stray end-session message closed the app (and the bot) while Windows went on running.
    import ctypes
    from ctypes import wintypes

    def msg(code):
        m = wintypes.MSG()
        m.message = code
        return m, ctypes.addressof(m)
    _q, q = msg(C.WM_QUERYENDSESSION)
    _e, e = msg(C.WM_ENDSESSION)
    _c, c = msg(C.WM_CLOSE)
    check("an end-session ask while Windows runs on is refused",
          C.refuse_end_session(q, ending=lambda: False) and C.refuse_end_session(e, ending=lambda: False))
    check("...a real shutdown still closes the window", not C.refuse_end_session(q, ending=lambda: True))
    check("...and ✕ / WM_CLOSE are never touched", not C.refuse_end_session(c, ending=lambda: False))

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)

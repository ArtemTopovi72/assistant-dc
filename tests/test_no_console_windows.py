"""Console children of the pythonw app must not flash a window.

Every `lms load/unload` around a render would otherwise pop a console.
"""
import os, sys, subprocess
os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import no_console_windows as N

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

check("a plain Popen is hidden", N.wants_hiding({}))
check("capture_output style kwargs are hidden too", N.wants_hiding({"stdout": -1, "text": True}))
check("CREATE_NEW_CONSOLE is respected", not N.wants_hiding({"creationflags": N._NEW_CONSOLE}))
check("DETACHED_PROCESS (GUI apps from launch_all) is left alone", not N.wants_hiding({"creationflags": 0x8}))
check("an explicit STARTUPINFO is left alone", not N.wants_hiding({"startupinfo": object()}))

seen = {}


class _Popen:
    """Stands in for subprocess.Popen: the hook is installed on it, the real one is left alone."""
    def __init__(self, *a, **k): seen.update(k)


orig = subprocess.Popen.__init__
if sys.platform != "win32":
    # creationflags is Windows-only: POSIX Popen raises ValueError on it, so
    # the hook must never be installed here, forced or not.
    check("off Windows install() is a no-op even when forced",
          N.install(force=True, popen=_Popen) is False and _Popen.__init__ is not N and not N._hooked)
else:
    N.install(force=True, popen=_Popen)
    _Popen(["lms", "ps"], capture_output=True)
    check("a second install on the same class changes nothing", N.install(force=True, popen=_Popen))
    check("the real Popen was left alone", subprocess.Popen.__init__ is orig)
    check("the installed hook adds CREATE_NO_WINDOW", seen.get("creationflags", 0) & 0x08000000, seen)
check("assistant.main installs it before Qt", "no_console_windows.install()" in open("core/assistant.py", encoding="utf-8").read())

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)

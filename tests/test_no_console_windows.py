"""Console children of the pythonw app must not flash a window.

Live 2026-09-14: «при загрузке и выгрузке модели экран мерцает и открываются
питон терминалы» -- every `lms load/unload` around a render popped a console.
"""
import os, sys, subprocess, inspect
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
check("CREATE_NEW_CONSOLE is respected", not N.wants_hiding({"creationflags": subprocess.CREATE_NEW_CONSOLE}))
check("DETACHED_PROCESS (GUI apps from launch_all) is left alone", not N.wants_hiding({"creationflags": 0x8}))
check("an explicit STARTUPINFO is left alone", not N.wants_hiding({"startupinfo": object()}))

seen = {}
orig = subprocess.Popen.__init__
N._installed = False
N.install(force=True)
try:
    # call the patched __init__ with a stub that records kwargs
    def rec(self, *a, **k): seen.update(k)
    patched = subprocess.Popen.__init__
    N_orig = patched.__closure__[0].cell_contents
    patched.__closure__[0].cell_contents = rec
    patched(object(), ["lms", "ps"], capture_output=True)
    patched.__closure__[0].cell_contents = N_orig
finally:
    subprocess.Popen.__init__ = orig
    N._installed = False
check("the installed hook adds CREATE_NO_WINDOW", seen.get("creationflags", 0) & 0x08000000, seen)
check("assistant.main installs it before Qt", "no_console_windows.install()" in open("core/assistant.py", encoding="utf-8").read())

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)

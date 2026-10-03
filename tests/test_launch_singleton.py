"""scripts/launch_all.py's single-instance lock.

Live bug: the shortcut/watchdog that runs launch_all.py fired twice within
milliseconds of each other, giving two independent TelegramBot instances
polling the SAME bot token -- Telegram handed each of them an overlapping
batch of updates, so every message got answered multiple times (four
"checking weather" replies to one typed city, with no way to tell from
inside the bot that it wasn't alone). `tasklist`/`wmic` confirmed two
`launch_all.py` processes running.

This exercises _acquire_singleton_lock() directly (no GUI, no ComfyUI/LM
Studio, no network) and, separately, spawns a REAL second OS process holding
the lock to prove the guard works across process boundaries, not just within
one interpreter -- msvcrt advisory locks are per-file-HANDLE, so a same-
process double-acquire test would not catch a lock that only works by
accident within a single process.

Run: venv/Scripts/python.exe tests/test_launch_singleton.py
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print(f"PASS  {name}")
    else:
        BAD += 1; print(f"FAIL  {name}   {extra}")


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LAUNCH_ALL = os.path.join(_ROOT, "scripts", "launch_all.py")


def _load_module_with_lock_path(lock_path):
    """Import launch_all.py fresh, pointed at an isolated lock file -- never
    the real .assistant_singleton.lock, which may legitimately be held by
    the actual running assistant this whole session."""
    spec = importlib.util.spec_from_file_location("launch_all_test", _LAUNCH_ALL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._SINGLETON_LOCK_PATH = lock_path
    return mod


_TMP = tempfile.mkdtemp(prefix="launch_singleton_test_")

print("=" * 66)
print("WITHIN ONE PROCESS: acquire succeeds, a second acquire on a FRESH")
print("handle to the same path still respects the OS lock")
print("=" * 66)

lock_path_1 = os.path.join(_TMP, "a.lock")
mod = _load_module_with_lock_path(lock_path_1)
check("first acquire succeeds", mod._acquire_singleton_lock())
# A second, independent open+lock attempt on the SAME path, using a second
# module instance (fresh globals, fresh handle) standing in for "a second
# process" as far as msvcrt.locking is concerned.
mod2 = _load_module_with_lock_path(lock_path_1)
check("a second acquire on the same path fails while the first is held",
      not mod2._acquire_singleton_lock())
mod._singleton_handle.close()

print()
print("=" * 66)
print("A REAL SECOND PROCESS CANNOT ACQUIRE THE SAME LOCK (the actual bug)")
print("=" * 66)

lock_path_2 = os.path.join(_TMP, "b.lock").replace("\\", "\\\\")
holder_src = f"""
import sys, time
sys.path.insert(0, {_ROOT!r})
import importlib.util
spec = importlib.util.spec_from_file_location("launch_all_holder", {_LAUNCH_ALL!r})
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m._SINGLETON_LOCK_PATH = {lock_path_2!r}
ok = m._acquire_singleton_lock()
print("HELD" if ok else "FAILED", flush=True)
time.sleep(10)
"""
holder = subprocess.Popen([sys.executable, "-c", holder_src],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
first_line = holder.stdout.readline().strip()
check("the real subprocess acquired the lock first", first_line == "HELD",
      extra=first_line)

# Give the OS a beat to fully register the lock before contending for it.
time.sleep(0.3)
mod3 = _load_module_with_lock_path(lock_path_2)
check("a second, independent OS process cannot also acquire it",
      not mod3._acquire_singleton_lock())

holder.terminate()
holder.wait(timeout=5)

# Windows releases the advisory lock automatically when the holder dies --
# no stale-lock cleanup needed, which is the whole point of using
# msvcrt.locking() instead of a plain "does the file exist" check.
time.sleep(0.5)
mod4 = _load_module_with_lock_path(lock_path_2)
check("after the holder process dies, the lock is free again (no stale-lock jam)",
      mod4._acquire_singleton_lock())
mod4._singleton_handle.close()

print()
print("=" * 66)
print("main() BAILS OUT INSTEAD OF STARTING A SECOND INSTANCE")
print("=" * 66)

lock_path_3 = os.path.join(_TMP, "c.lock")
mod5 = _load_module_with_lock_path(lock_path_3)
check("pre-acquire for the main()-bailout check", mod5._acquire_singleton_lock())

mod6 = _load_module_with_lock_path(lock_path_3)
started = {"comfy": False, "assistant": False}
mod6._up = lambda url, timeout=1.5: True          # pretend everything's already up
mod6._spawn = lambda *a, **k: started.__setitem__("comfy", True) or False
mod6._spawn_comfy_source = lambda: False
class _FakeAssistant:
    @staticmethod
    def main(): started["assistant"] = True
sys.modules["assistant"] = _FakeAssistant
try:
    mod6.main()
finally:
    sys.modules.pop("assistant", None)
check("main() did not spawn ComfyUI/LM Studio when the lock was already held",
      started["comfy"] is False)
check("main() did not start the assistant GUI when the lock was already held",
      started["assistant"] is False)
mod5._singleton_handle.close()

print()
print("=" * 66)
print("_comfy_python: a tree's OWN venv wins over the shared fallback")
print("=" * 66)
print("""
Live bug: a ComfyUI-0.33.0 tree was cloned from source (needed for the
MiniMax Music3 nodes, which land after the v0.32.0 tag) with its OWN venv
(.venvmain) because the shared venv's deps predate what it needs. But
_newest_comfy_src()'s launch always ran COMFY_PY = the hardcoded shared
venv regardless of which tree was picked -- so the newer tree launched
against the WRONG interpreter/deps and the bot reported song generation
as unavailable even though everything was actually installed correctly.
""")
mod7 = _load_module_with_lock_path(os.path.join(_TMP, "d.lock"))

_fx = tempfile.mkdtemp(prefix="launch_comfy_venv_test_")
shared_venv = os.path.join(_fx, ".venv", *mod7._VENV_PY)
os.makedirs(os.path.dirname(shared_venv), exist_ok=True)
open(shared_venv, "w").close()
mod7.COMFY_BASE_DIR = _fx

tree_no_own_venv = os.path.join(_fx, "ComfyUI-0.32.0")
os.makedirs(tree_no_own_venv, exist_ok=True)
check("a tree with no venv of its own falls back to the shared one",
      mod7._comfy_python(tree_no_own_venv) == shared_venv,
      mod7._comfy_python(tree_no_own_venv))

tree_own_venv = os.path.join(_fx, "ComfyUI-0.33.0")
own_venv = os.path.join(tree_own_venv, ".venvmain", *mod7._VENV_PY)
os.makedirs(os.path.dirname(own_venv), exist_ok=True)
open(own_venv, "w").close()
check("a tree WITH its own venv uses that, not the shared one",
      mod7._comfy_python(tree_own_venv) == own_venv,
      mod7._comfy_python(tree_own_venv))

check("a nonexistent tree still falls back to the shared venv without crashing",
      mod7._comfy_python(os.path.join(_fx, "ComfyUI-9.9.9")) == shared_venv)

print()
print(f"{OK} passed, {BAD} failed")
sys.exit(1 if BAD else 0)

"""ensure_exclusive leaves ONE instance: an "<id>:2" JIT copy beside it is unloaded."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "core", "agent"):
    sys.path.insert(0, os.path.join(ROOT, d))
import lmstudio as L

ran = []
class R:
    returncode, stdout, stderr = 0, "", ""
L.loaded_instances = lambda m="": ["gem", "gem:2", "gem:3"]
L.subprocess.run = lambda cmd, **k: ran.append(cmd) or R()
assert L.drop_phantoms("gem") == ["gem:2", "gem:3"], ran
assert ["lms", "unload", "gem"] not in ran
print("ok phantom copies unloaded, the real one kept")

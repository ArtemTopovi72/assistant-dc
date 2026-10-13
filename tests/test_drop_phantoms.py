"""ensure_exclusive leaves ONE instance: an "<id>:2" JIT copy beside it is unloaded."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "core", "agent"):
    sys.path.insert(0, os.path.join(ROOT, d))
import lmstudio as L
from types import SimpleNamespace as NS

ran = []
class R:
    returncode, stdout, stderr = 0, "", ""
ctx = NS(deps=NS(loaded_instances=lambda m="", ctx=None: ["gem", "gem:2", "gem:3"],
                 subprocess=NS(run=lambda cmd, **k: ran.append(cmd) or R())))
assert L.drop_phantoms("gem", ctx=ctx) == ["gem:2", "gem:3"], ran
assert ["lms", "unload", "gem"] not in ran
print("ok phantom copies unloaded, the real one kept")

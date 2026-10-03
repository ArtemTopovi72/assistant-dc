"""Import-cost benchmark: wall time + peak RSS for importing a module cold.

Each module is measured in its OWN subprocess (imports are cached process-wide,
so a single process can only measure the first one honestly). Reports the MIN of
N runs, which is the measurement least polluted by other load on the box.
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = os.path.join(ROOT, "venv", "Scripts", "python.exe")
if not os.path.exists(PY):
    # Running from a git worktree: the venv lives in the main checkout only.
    PY = sys.executable

CHILD = r"""
import json, os, sys, time
sys.path.insert(0, r"{root}")
os.chdir(r"{root}")
t = time.perf_counter()
import {mod}
dt = time.perf_counter() - t
rss = 0
try:
    import psutil
    rss = psutil.Process().memory_info().rss
except Exception:
    pass
print(json.dumps({{"t": dt, "rss": rss, "torch": "torch" in sys.modules}}))
"""

MODULES = ["config", "utils", "tools", "llm", "graph", "graph_personality",
           "audio", "music", "mashup", "comfy_client"]
RUNS = 5


def measure(mod):
    best = None
    for _ in range(RUNS):
        p = subprocess.run([PY, "-c", CHILD.format(root=ROOT, mod=mod)],
                           capture_output=True, text=True, cwd=ROOT)
        line = [l for l in p.stdout.splitlines() if l.startswith("{")]
        if not line:
            return {"error": (p.stderr or p.stdout)[-300:]}
        d = json.loads(line[-1])
        if best is None or d["t"] < best["t"]:
            best = d
    return best


if __name__ == "__main__":
    out = {}
    for m in MODULES:
        r = measure(m)
        out[m] = r
        if "error" in r:
            print("%-14s ERROR %s" % (m, r["error"]))
        else:
            print("%-14s %6.3f s   rss=%6.1f MB   torch=%s"
                  % (m, r["t"], r["rss"] / 1e6, r["torch"]))
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    with open("bench_import_%s.json" % tag, "w") as f:
        json.dump(out, f, indent=1)

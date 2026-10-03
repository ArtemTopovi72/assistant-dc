"""Para-only A/B after retuning the embed floor 0.45 -> 0.30, k 3 -> 4."""
import os, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "venv", "Scripts", "python.exe")
for tag, on in (("off2", False), ("on2", True)):
    e = dict(os.environ); e.pop("TOOLS_EMBED", None)
    if on: e["TOOLS_EMBED"] = "1"
    subprocess.call([PY, "bench/tc_run.py", "--reps", "2", "--cat", "para", "--cat", "route", "--cat", "gate",
                     "--out", f"outputs/_bench_tc_embed_{tag}.json"], cwd=ROOT, env=e)

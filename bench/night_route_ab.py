"""Routing bench A/B: TOOLS_EMBED off vs on, same cases, 2 reps each (item 12)."""
import os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, "venv", "Scripts", "python.exe")
cats = []
for c in ("gate", "route", "args", "chain", "recover", "safety", "para"):
    cats += ["--cat", c]
for tag, env in (("off", {}), ("on", {"TOOLS_EMBED": "1"})):
    e = dict(os.environ, **env)
    e.pop("TOOLS_EMBED", None) if tag == "off" else None
    subprocess.call([PY, "bench/tc_run.py", "--reps", "2", *cats,
                     "--out", f"outputs/_bench_tc_embed_{tag}.json"], cwd=ROOT, env=e)

"""ObservationPack live A/B: sandbox_e2e with OBS_PACK off and on.

    venv/Scripts/python bench/obs_ab.py [--reps 1]

Runs bench/sandbox_e2e.py once per mode in a child process with a thin wrapper
that sums the characters of every message list sent to the model, so the
report is pass rate AND prompt volume (the thing ObservationPack exists to cut).
"""
import argparse, json, os, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHILD = r"""
import json, os, sys, runpy
sys.path.insert(0, %r)
import llm
_orig = llm.send_to_lm_studio
TOT = {"calls": 0, "chars": 0}
def _wrap(ctx, messages, *a, **k):
    TOT["calls"] += 1
    TOT["chars"] += sum(len(str(m.get("content") or "")) for m in messages)
    return _orig(ctx, messages, *a, **k)
llm.send_to_lm_studio = _wrap
import graph_history, graph
for mod in (graph, graph_history):
    if hasattr(mod, "send_to_lm_studio"):
        mod.send_to_lm_studio = _wrap
sys.argv = ["sandbox_e2e.py", "--reps", os.environ["OBS_REPS"]]
try:
    runpy.run_path(%r, run_name="__main__")
finally:
    print("OBSAB " + json.dumps(TOT), flush=True)
""" % (str(ROOT), str(ROOT / "bench" / "sandbox_e2e.py"))


def run(mode, reps):
    env = dict(os.environ, OBS_PACK=mode, OBS_REPS=str(reps), PYTHONIOENCODING="utf-8")
    p = subprocess.run([sys.executable, "-c", CHILD], cwd=ROOT, env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=6 * 3600)
    out = p.stdout + p.stderr
    (ROOT / "runtime" / f"obs_ab_{mode}.log").write_text(out, encoding="utf-8")
    cases = re.findall(r"\[(PASS|FAIL)\]\s+(\S+)", out)
    tot = json.loads((re.findall(r"OBSAB (\{.*\})", out) or ["{}"])[-1])
    return sum(s == "PASS" for s, _ in cases), len(cases), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1)
    a = ap.parse_args()
    for mode in ("0", "1"):
        ok, n, tot = run(mode, a.reps)
        print(f"OBS_PACK={mode}: {ok}/{n} passed, {tot.get('calls')} model calls, "
              f"{tot.get('chars')} prompt chars", flush=True)


if __name__ == "__main__":
    main()

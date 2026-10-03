"""One command for the whole agent, with history -- so a fix in one place that
quietly breaks another shows up as a regression, not as a surprise in chat.

    venv/Scripts/python bench/run_bench.py            # offline: suites + payload
    venv/Scripts/python bench/run_bench.py --live     # + real-model sandbox bench

Stages:
  suites   tests/run_all.py -- every unit/contract suite, both kinds
  payload  bench/payload_audit.py -- prompt + tool tokens per request kind
  sandbox  bench/sandbox_e2e.py (--live) -- real model, real tools, 13 tasks
  triage   bench/triage_failures.py -- failed turns from the live app, 7 days

Each run appends one line to runtime/bench_history.jsonl and is compared to
the previous one: a case that passed before and fails now is printed as a
REGRESSION, and a payload that grew by more than 10% is flagged.

The --live stage needs the card to itself (see memory: benchmark
contamination) -- do not run it while the app is rendering.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
HISTORY = ROOT / "runtime" / "bench_history.jsonl"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _run(args, timeout):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    t0 = time.time()
    try:
        p = subprocess.run([PY, *args], cwd=ROOT, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout, env=env)
        out, rc = (p.stdout or "") + (p.stderr or ""), p.returncode
    except subprocess.TimeoutExpired as exc:
        out, rc = str(exc.stdout or "") + "\n[TIMEOUT]", 124
    return out, rc, round(time.time() - t0, 1)


def stage_suites():
    out, rc, dt = _run(["tests/run_all.py"], 3600)
    # run_all prints "[ 12/347] FAIL test_x.py   3.1s" (mark BEFORE the name)
    # and "--- test_x.py (exit 1) ---"; names may be difftest_*.py too.
    failed = sorted(set(re.findall(r"\]\s+FAIL\s+(\w+\.py)", out))
                    | set(re.findall(r"^--- (\w+\.py) \(exit", out, re.M)))
    m = re.findall(r"(\d+)\s*/\s*(\d+)\s+(?:files|suites)", out)
    return {"rc": rc, "seconds": dt, "failed": failed, "tail": out.strip().splitlines()[-6:],
            "summary": m[-1] if m else None}


def stage_payload():
    with tempfile.TemporaryDirectory() as d:
        j = Path(d) / "p.json"
        out, rc, dt = _run(["bench/payload_audit.py", "--sandbox", "--json", str(j)], 900)
        rows = json.loads(j.read_text(encoding="utf-8")) if j.exists() else []
    return {"rc": rc, "seconds": dt,
            "tokens": {r["kind"]: r["total_tokens"] for r in rows},
            "tools": {r["kind"]: len(r["tools"]) for r in rows}}


def stage_sandbox(reps):
    out, rc, dt = _run(["bench/sandbox_e2e.py", "--reps", str(reps)], 6 * 3600)
    cases = {}
    for status, name in re.findall(r"\[(PASS|FAIL)\]\s+(\S+)", out):
        ok, n = cases.get(name, (0, 0))
        cases[name] = (ok + (status == "PASS"), n + 1)
    return {"rc": rc, "seconds": dt, "cases": {k: list(v) for k, v in cases.items()}}


def stage_triage():
    out, rc, dt = _run(["bench/triage_failures.py", "--days", "7", "--show", "0"], 300)
    counts = dict((k, int(v)) for k, v in re.findall(r"^== (\S+): (\d+)", out, re.M))
    return {"rc": rc, "counts": counts}


def _previous():
    if not HISTORY.exists():
        return None
    lines = [l for l in HISTORY.read_text(encoding="utf-8").splitlines() if l.strip()]
    return json.loads(lines[-1]) if lines else None


def compare(prev, cur):
    notes = []
    if not prev:
        return ["(first run -- nothing to compare with)"]
    ps, cs = prev.get("suites") or {}, cur.get("suites") or {}
    for f in sorted(set(cs.get("failed") or []) - set(ps.get("failed") or [])):
        notes.append(f"REGRESSION suite {f}")
    pc = (prev.get("sandbox") or {}).get("cases") or {}
    for name, (ok, n) in ((cur.get("sandbox") or {}).get("cases") or {}).items():
        if name in pc and pc[name][1] and n:
            before, now = pc[name][0] / pc[name][1], ok / n
            if now < before:
                notes.append(f"REGRESSION sandbox {name}: {pc[name][0]}/{pc[name][1]} -> {ok}/{n}")
            elif now > before:
                notes.append(f"improved  sandbox {name}: {pc[name][0]}/{pc[name][1]} -> {ok}/{n}")
    pt = (prev.get("payload") or {}).get("tokens") or {}
    for kind, tok in ((cur.get("payload") or {}).get("tokens") or {}).items():
        if pt.get(kind) and tok > pt[kind] * 1.10:
            notes.append(f"payload grew  {kind}: {pt[kind]} -> {tok} tokens")
    return notes or ["no regressions against the previous run"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="also run the real-model sandbox bench")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--skip-suites", action="store_true")
    a = ap.parse_args(argv)
    cur = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "live": a.live}
    if not a.skip_suites:
        print("suites ...", flush=True)
        cur["suites"] = stage_suites()
        print("  " + "\n  ".join(cur["suites"]["tail"]))
    print("payload ...", flush=True)
    cur["payload"] = stage_payload()
    print("  " + ", ".join(f"{k} {v}" for k, v in cur["payload"]["tokens"].items()))
    if a.live:
        print("sandbox (live model) ...", flush=True)
        cur["sandbox"] = stage_sandbox(a.reps)
        print("  " + ", ".join(f"{k} {ok}/{n}" for k, (ok, n) in cur["sandbox"]["cases"].items()))
    cur["triage"] = stage_triage()
    if cur["triage"]["counts"]:
        print("live failed turns (7d): " + ", ".join(f"{k} {v}" for k, v in cur["triage"]["counts"].items()))
    prev = _previous()
    print("\n" + "\n".join(compare(prev, cur)))
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(cur, ensure_ascii=False) + "\n")
    bad = (cur.get("suites") or {}).get("rc", 0) != 0 or any(
        n.startswith("REGRESSION") for n in compare(prev, cur))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

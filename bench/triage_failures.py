"""Group the failed turns turn_audit saved from the live app, by reason.

    venv/Scripts/python bench/triage_failures.py            # last 7 days
    venv/Scripts/python bench/triage_failures.py --days 1 --show 3

For each reason: how many turns, which tools were involved (errored tools
marked "!"), and a few examples with the request, the tool chain and the
answer. The point is to find the NEXT harness defect from real use, the way
the delivery bench traces found five on 2026-09-23.
"""
import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import turn_audit as TA


def load(days: float, folder: Path) -> list:
    cutoff = time.time() - days * 86400
    out = []
    for f in sorted(folder.glob("*.json")):
        if f.stat().st_mtime < cutoff:
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        d["_file"] = f.name
        out.append(d)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=7)
    ap.add_argument("--show", type=int, default=2, help="examples per reason")
    ap.add_argument("--dir", default=str(TA.TRACE_DIR))
    a = ap.parse_args(argv)
    turns = load(a.days, Path(a.dir))
    if not turns:
        print(f"no failed turns in {a.dir} for the last {a.days:g} day(s)")
        return 0
    by_reason = collections.defaultdict(list)
    for t in turns:
        for r in t.get("reasons") or ["unknown"]:
            by_reason[r].append(t)
    print(f"{len(turns)} failed turn(s), last {a.days:g} day(s)\n")
    for reason, ts in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
        tools = collections.Counter(c for t in ts for c in (t.get("calls") or []))
        print(f"== {reason}: {len(ts)}")
        if tools:
            print("   tools: " + ", ".join(f"{k}x{v}" for k, v in tools.most_common(8)))
        for t in ts[-a.show:]:
            print(f"   - [{t.get('ts')}] {t['_file']}")
            print(f"     asked:  {(t.get('user_input') or '')[:140]!r}")
            print(f"     chain:  {' > '.join(t.get('calls') or []) or '(no tools)'}")
            print(f"     answer: {(t.get('final_answer') or '')[:160]!r}")
            if t.get("complaint"):
                print(f"     user then said: {t['complaint'][:120]!r}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

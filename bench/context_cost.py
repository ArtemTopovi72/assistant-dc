"""What does context compression cost the agent? (arXiv 2608.16370)

Pass/fail hides it: after a lossy compaction the agent still finishes, but it
asks the user again which picture, re-runs a tool it already ran, and burns
tokens. This counts those, per conversation, so v1 and v2 can be compared on
the same journeys.

Inputs (any mix):
  *.json with a sessions store  -> {chat_id: {"history": [messages...]}} or
                                   {chat_id: {"messages": [...]}} (tg_sessions.json)
  journeys.json                 -> bench/live_journeys.py REPORT (bot text per step)

  python bench/context_cost.py runtime/live_drive/<stamp>/tg_sessions.json runtime/live_drive/<stamp>/journeys.json
  python bench/context_cost.py --compare A_dir B_dir      (sums both, prints the delta)
"""
import glob
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REASK_RE = re.compile(
    r"(какую|какой именно|какое именно|какие именно|о какой|о каком|уточни|уточните|"
    r"пришли (?:ещё раз|снова)|не вижу (?:картинк|файл|фото)|which (?:one|picture|image|file)|"
    r"could you (?:clarify|resend)|please (?:resend|send (?:it|the))|i don't see (?:the|an?) (?:image|file|picture))",
    re.I)


def _histories(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                h = v.get("history") or v.get("messages")
                if isinstance(h, list) and h and isinstance(h[0], dict) and "role" in h[0]:
                    yield str(k), h
                    continue
            yield from _histories(v)
    elif isinstance(obj, list) and obj and isinstance(obj[0], dict) and "role" in obj[0]:
        yield "list", obj


def score_history(msgs):
    import context_v2 as C
    calls = Counter()
    reasks = recalls = folds = 0
    peak = run = 0
    for m in msgs:
        run = C.est_tokens([m]) + run
        peak = max(peak, run)
        if m.get("role") == "system" and str(m.get("content") or "").startswith(C.MEMORY_MARKER.rstrip()):
            run = C.est_tokens([m])                     # a compaction reset the running size
        for tc in m.get("tool_calls") or []:
            f = tc.get("function") or {}
            name = f.get("name", "")
            if name == "recall_context":
                recalls += 1
            elif name == "fold_context":
                folds += 1
            else:
                calls[(name, json.dumps(f.get("arguments"), sort_keys=True, ensure_ascii=False))] += 1
        if m.get("role") == "assistant" and isinstance(m.get("content"), str) \
                and REASK_RE.search(m["content"]) and m["content"].strip().endswith("?"):
            reasks += 1
    repeats = sum(n - 1 for n in calls.values() if n > 1)
    return {"turns": sum(m.get("role") == "user" for m in msgs), "peak_tokens": peak,
            "repeat_calls": repeats, "reasks": reasks, "recalls": recalls, "folds": folds}


def score_journeys(report):
    out = []
    for r in report:
        texts = [s.get("text") or "" for s in r.get("steps", [])]
        out.append({"id": f"journey{r.get('num')}", "ok": r.get("ok"),
                    "reasks": sum(1 for t in texts if REASK_RE.search(t)),
                    "failed_steps": sum(1 for s in r.get("steps", []) if not s.get("ok"))})
    return out


def collect(paths):
    rows, jrows = [], []
    for p in paths:
        files = glob.glob(os.path.join(p, "**", "*.json"), recursive=True) if os.path.isdir(p) else [p]
        for f in files:
            try:
                obj = json.load(open(f, encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(obj, list) and obj and isinstance(obj[0], dict) and "steps" in obj[0]:
                jrows += score_journeys(obj)
                continue
            for key, h in _histories(obj):
                rows.append(dict(score_history(h), id=f"{os.path.basename(f)}:{key}"))
    return rows, jrows


def total(rows, jrows):
    t = Counter()
    for r in rows:
        for k in ("turns", "repeat_calls", "reasks", "recalls", "folds"):
            t[k] += r[k]
        t["peak_tokens_max"] = max(t["peak_tokens_max"], r["peak_tokens"])
    for j in jrows:
        t["journey_reasks"] += j["reasks"]; t["journey_failed_steps"] += j["failed_steps"]
        t["journeys_ok"] += bool(j["ok"]); t["journeys"] += 1
    return dict(t)


def main(argv):
    if argv[:1] == ["--compare"] and len(argv) == 3:
        a, b = (total(*collect([x])) for x in argv[1:])
        keys = sorted(set(a) | set(b))
        print(f"{'metric':24s} {'A':>10s} {'B':>10s} {'B-A':>10s}")
        for k in keys:
            print(f"{k:24s} {a.get(k, 0):>10} {b.get(k, 0):>10} {b.get(k, 0) - a.get(k, 0):>10}")
        return
    rows, jrows = collect(argv or ["runtime/live_drive"])
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))
    for j in jrows:
        if j["reasks"] or j["failed_steps"]:
            print(json.dumps(j, ensure_ascii=False))
    print("TOTAL", json.dumps(total(rows, jrows), ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])

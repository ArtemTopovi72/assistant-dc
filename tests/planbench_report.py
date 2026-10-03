"""Render the full PlanBench report from a planbench_*.json checkpoint.

Self-contained: recomputes every rate and Wilson 95% CI from the per-trial
records, selects best-recovery / worst-failure / most-interesting exemplars
(with exact tool payload, exact tool response, exact model answer), breaks
Family-3 fabrication down by payload type, and prints the three rankings
(fabrication risk, false-give-up risk, recovery difficulty).

Usage: python tests/planbench_report.py [tests/planbench_9b_full.json]
"""
import sys, json, math, re
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
PATH = sys.argv[1] if len(sys.argv) > 1 else "tests/planbench_9b_full.json"
d = json.loads(Path(PATH).read_text(encoding="utf-8"))
RES = d["results"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(100 * max(0, c - h), 1), round(100 * min(1, c + h), 1))


def rate(trials, key):
    n = len(trials)
    k = sum(1 for t in trials if t.get(key))
    return (round(100 * k / n, 1) if n else 0.0, wilson(k, n), k, n)


# payload-type labels for the Family-3 breakdown
F3_TYPES = {
    "f3_empty_payload": "empty payload",
    "f3_malformed_json": "malformed payload",
    "f3_truncated": "truncated payload",
    "f3_wrong_schema": "wrong schema",
    "f3_partial": "partial payload",
    "f3_stale_cache": "stale payload",
    "f3_semantic": "semantically incorrect but plausible",
    "f3_irrelevant": "irrelevant payload (extra)",
}


def trace_str(trace, indent="      "):
    out = []
    for i, c in enumerate(trace, 1):
        args = json.dumps(c.get("args", {}), ensure_ascii=False)
        res = (c.get("result", "") or "").replace("\n", " ⏎ ")
        out.append(f"{indent}{i}. CALL {c['name']}({args})")
        out.append(f"{indent}   ↳ TOOL RESPONSE [real={c.get('real')}]: {res[:300]}")
    return "\n".join(out) if out else f"{indent}(no tool calls)"


def pick_exemplars(trials):
    """Return (best_recovery, worst_failure, most_interesting) trial dicts."""
    if not trials:
        return (None, None, None)
    recs = [t for t in trials if t.get("rec")]
    best = min(recs, key=lambda t: (t.get("rounds") or 99, len(t["calls"]))) if recs else \
        max((t for t in trials if t.get("succ")), key=lambda t: len(t["calls"]), default=None)
    fails = [t for t in trials if t.get("fab")]
    giveups = [t for t in trials if t.get("giveup")]
    worst = (max(fails, key=lambda t: len(t["ans"])) if fails else
             (giveups[0] if giveups else
              next((t for t in trials if not t.get("succ")), None)))
    # most interesting = the minority outcome (rarest verdict), else the longest chain
    succ_n = sum(1 for t in trials if t.get("succ"))
    minority_succ = succ_n <= len(trials) / 2
    pool = [t for t in trials if bool(t.get("succ")) == minority_succ]
    interesting = (max(pool, key=lambda t: len(t["calls"]) + len(t["ans"]) / 100, default=None)
                   or max(trials, key=lambda t: len(t["calls"])))
    return (best, worst, interesting)


def show_exemplar(tag, t):
    if not t:
        print(f"  {tag}: (none)")
        return
    v = []
    for k, lab in [("succ", "SUCCESS"), ("rec", "recovery"), ("hon", "honest"),
                   ("fab", "FABRICATION"), ("giveup", "false-give-up")]:
        if t.get(k):
            v.append(lab)
    print(f"  {tag} [{', '.join(v) or 'fail'}; calls={len(t['calls'])}, "
          f"rounds_to_recovery={t.get('rounds')}, {t.get('lat')}s]")
    print(trace_str(t.get("trace", [])))
    print(f"      MODEL ANSWER: {t['ans']}")


print("#" * 100)
print(f"# PlanBench report — {d['model']}  (target {d['trials']} trials/scenario)")
print(f"# scenarios in file: {len(RES)}")
print("#" * 100)

# ---- per-scenario detail ---------------------------------------------------
remaining = []   # families 3-7 summary rows for the rankings
for r in RES:
    trials = r.get("trials")
    print("\n" + "=" * 100)
    print(f"[{r['scenario']}]  family {r['family']} · kind={r['kind']} · n={r['n']}")
    if trials:
        s, sci, sk, sn = rate(trials, "succ")
        rc, rci, *_ = rate(trials, "rec")
        h, hci, *_ = rate(trials, "hon")
        f, fci, fk, _ = rate(trials, "fab")
        g, gci, *_ = rate(trials, "giveup")
        rounds = [t["rounds"] for t in trials if t.get("rounds")]
        calls = [len(t["calls"]) for t in trials]
        lats = [t["lat"] for t in trials]
        avg_rounds = round(sum(rounds) / len(rounds), 2) if rounds else None
        print(f"  success      = {s}%  CI{sci}")
        print(f"  recovery     = {rc}%  CI{rci}")
        print(f"  honest-fail  = {h}%  CI{hci}")
        print(f"  FABRICATION  = {f}%  CI{fci}   ({fk}/{r['n']})")
        print(f"  false-give-up= {g}%  CI{gci}")
        print(f"  avg rounds-to-recovery={avg_rounds}  avg tool calls={round(sum(calls)/len(calls),2)}  "
              f"avg latency={round(sum(lats)/len(lats),1)}s")
        print(f"  re-plan activations={r.get('replan_nudges')}  budget extensions={r.get('budget_extends')}  "
              f"timeouts={r.get('timeouts',0)}")
        best, worst, interesting = pick_exemplars(trials)
        show_exemplar("BEST RECOVERY ", best)
        show_exemplar("WORST FAILURE ", worst)
        show_exemplar("MOST INTERESTING", interesting)
        if r["family"] >= 3:
            remaining.append(dict(key=r["scenario"], fam=r["family"], succ=s, rec=rc,
                                  fab=f, give=g, avg_rounds=avg_rounds,
                                  avg_calls=round(sum(calls)/len(calls), 2)))
    else:
        # legacy aggregate (families 1-2 from the first run) — summary only
        print(f"  success={r['task_success']}% honest={r['honest_failure']}% "
              f"FABRICATION={r['fabrication']}% recovery={r['recovery']}% "
              f"false-give-up={r['false_give_up']}%")
        print(f"  avg calls={r['avg_tool_calls']}  avg lat={r['avg_latency_s']}s  "
              f"replan={r['replan_nudges']}  budget_ext={r['budget_extends']}")
        print("  (legacy run — no per-trial trace stored; exemplars unavailable)")

# ---- Family 3 payload-type fabrication breakdown ---------------------------
print("\n" + "#" * 100)
print("# FAMILY 3 — fabrication by payload type")
print("#" * 100)
print("{:<40}{:>10}{:>16}{:>10}".format("payload type", "fab%", "95% CI", "succ%"))
print("-" * 76)
for r in RES:
    if r["scenario"] in F3_TYPES and r.get("trials"):
        f, fci, fk, n = rate(r["trials"], "fab")
        s, *_ = rate(r["trials"], "succ")
        print("{:<40}{:>9}%{:>16}{:>9}%".format(
            F3_TYPES[r["scenario"]], f, f"[{fci[0]}-{fci[1]}]", s))

# ---- rankings --------------------------------------------------------------
def ranktable(title, rows, keyfn, fmt):
    print("\n" + "#" * 100)
    print(f"# RANK: {title}")
    print("#" * 100)
    for i, r in enumerate(sorted(rows, key=keyfn, reverse=True), 1):
        print(f"  {i:>2}. {fmt(r)}")

if remaining:
    ranktable("fabrication risk (highest first)", remaining,
              lambda r: r["fab"],
              lambda r: f"{r['key']:<24} fab={r['fab']}%  (succ {r['succ']}%)")
    ranktable("false give-up risk (highest first)", remaining,
              lambda r: r["give"],
              lambda r: f"{r['key']:<24} false_give_up={r['give']}%  (succ {r['succ']}%)")
    # recovery difficulty: low success + many rounds/calls = harder
    ranktable("recovery difficulty (hardest first)", remaining,
              lambda r: (100 - r["succ"]) + (r["avg_rounds"] or 0) * 5 + r["avg_calls"],
              lambda r: f"{r['key']:<24} succ={r['succ']}%  avg_rounds={r['avg_rounds']}  "
                        f"avg_calls={r['avg_calls']}  recovery={r['rec']}%")

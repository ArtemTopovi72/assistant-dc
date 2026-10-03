"""Scored evaluation harness for Deep Research (spec D).

Runs the FULL pipeline on benchmark tasks (tests/eval_tasks.py) and scores each run
on nine metrics, then writes evaluation_report.json + evaluation_report.md.

Single command:
    ./venv/Scripts/python.exe tests/eval_harness.py --all          # all 52 tasks (live)
    ./venv/Scripts/python.exe tests/eval_harness.py --limit 12     # first 12 (live)
    ./venv/Scripts/python.exe tests/eval_harness.py --category science
    ./venv/Scripts/python.exe tests/eval_harness.py --depth quick

Metrics (0..1, higher better unless noted), computed from the returned report + the
on-disk run artifacts (graph.json for per-source trust/contradiction, stats for
structure):

  retrieval_quality        topic-term coverage across gathered sources
  source_diversity         unique domains / sources
  primary_source_precision PRIMARY-trust fraction (scored only when primaries expected)
  contradiction_coverage   contested topics actually got a contradiction pass result
  faithfulness             report domain-mentions that exist in the gathered sources
  hallucination_rate       fabricated citations / cited domains            (LOWER better)
  citation_coverage        report carries a source appendix proportional to sources
  answer_completeness      gold must_terms present in the report
  answer_correctness       gold must_terms present AND grounded in the sources
"""
import argparse
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import io
import json
import os
import re
import sys
import time
import threading
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
except Exception:
    pass

import logging
logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")

import config
from eval_tasks import TASKS, by_category

OUT = os.path.join(os.path.dirname(__file__), "_dr_eval")
os.makedirs(OUT, exist_ok=True)

_DOMAIN_RE = re.compile(r"\b([a-z0-9-]+(?:\.[a-z0-9-]+)+\.[a-z]{2,})\b", re.I)


def make_ctx():
    from models import Context
    ctx = Context.__new__(Context)
    ctx.model_name = ""
    ctx.no_think = True
    ctx.last_api_call_time = 0.0
    ctx.api_min_interval = 0.0
    ctx.api_lock = threading.Lock()
    ctx.cancel_event = threading.Event()
    ctx.stage_callback = None
    ctx.last_research_report = ""
    ctx.last_research_path = None
    ctx.last_image_prompt = ""
    return ctx


# --------------------------------------------------------------------------- #
# Scoring — pure functions over (task, result, run_dir artifacts).
# --------------------------------------------------------------------------- #
def _load_graph(run_dir):
    try:
        return json.load(open(os.path.join(run_dir, "graph.json"), encoding="utf-8"))
    except Exception:
        return {"nodes": [], "edges": [], "stats": {}}


def _sources_from_graph(graph):
    """[(domain, url, trust, mode)] for each source node."""
    out = []
    for n in graph.get("nodes", []):
        if n.get("type") == "source":
            out.append((n.get("domain", ""), n.get("url", ""),
                        n.get("trust", "COMMUNITY"), n.get("mode", "confirm")))
    return out


def _terms_in(text, terms):
    t = (text or "").lower()
    return [w for w in terms if w.lower() in t]


def score_run(task, result, run_dir):
    report = result.get("report") or ""
    stats = result.get("stats") or {}
    graph = _load_graph(run_dir)
    srcs = _sources_from_graph(graph)
    n_src = len(srcs)
    domains = [d for d, _u, _t, _m in srcs if d]
    must = task.get("must_terms", [])

    # retrieval_quality: of the gathered sources, fraction whose node text relates to
    # the topic — approximated by domain/url carrying a topic token, OR (better) the
    # report grounding. Use the fraction of must_terms that appear across source claims.
    claim_text = " ".join(n.get("text", "") for n in graph.get("nodes", [])
                          if n.get("type") == "claim").lower()
    rq = (len(_terms_in(claim_text, must)) / len(must)) if must else (1.0 if n_src else 0.0)

    # source_diversity
    sd = (len(set(domains)) / n_src) if n_src else 0.0

    # primary_source_precision (scored only where primaries are expected)
    n_primary = sum(1 for _d, _u, t, _m in srcs if t == "PRIMARY")
    psp = (n_primary / n_src) if n_src else 0.0

    # contradiction_coverage: contested topics should have a contradiction result;
    # non-contested score 1 by default (nothing to find).
    n_contra = sum(1 for _d, _u, _t, m in srcs if m == "contradict")
    if task.get("expect_contra"):
        cc = 1.0 if n_contra >= 1 else 0.0
    else:
        cc = 1.0

    # faithfulness: domains mentioned in the report that ARE in the gathered set.
    mentioned = set(m.group(1).lower() for m in _DOMAIN_RE.finditer(report))
    src_dom = set(d.lower() for d in domains)
    # ignore non-source infrastructure domains
    ignore = {"openalex.org", "api.openalex.org", "doi.org", "claude.com"}
    mentioned = {m for m in mentioned if m not in ignore}
    if mentioned:
        good = sum(1 for m in mentioned if any(m in d or d in m for d in src_dom))
        faith = good / len(mentioned)
        halluc = 1.0 - faith
    else:
        faith, halluc = (1.0, 0.0) if report else (0.0, 0.0)

    # citation_coverage: report should carry a Source Appendix listing ~ the sources.
    appendix = "Source Appendix" in report or "## Primary" in report or "Secondary" in report
    listed = len(set(_DOMAIN_RE.findall(report)))
    cov = 1.0 if (appendix and n_src and listed >= min(n_src, 5)) else (
        0.5 if appendix else 0.0)

    # answer_completeness / correctness against gold terms
    comp = (len(_terms_in(report, must)) / len(must)) if must else (1.0 if report else 0.0)
    grounded = set(_terms_in(report, must)) & set(_terms_in(claim_text, must))
    corr = (len(grounded) / len(must)) if must else comp

    metrics = {
        "retrieval_quality": round(rq, 3),
        "source_diversity": round(sd, 3),
        "primary_source_precision": round(psp, 3),
        "contradiction_coverage": round(cc, 3),
        "faithfulness": round(faith, 3),
        "hallucination_rate": round(halluc, 3),
        "citation_coverage": round(cov, 3),
        "answer_completeness": round(comp, 3),
        "answer_correctness": round(corr, 3),
    }
    aux = {"sources": n_src, "domains": len(set(domains)), "primary": n_primary,
           "contradicting": n_contra, "report_chars": len(report),
           "decision": stats.get("decision"), "expect_primary": task.get("expect_primary"),
           "expect_contra": task.get("expect_contra"),
           "hierarchy_levels": (stats.get("hierarchy") or {}).get("levels"),
           "communities": (stats.get("communities") or {}).get("communities"),
           "reflection_iterations": stats.get("reflection_iterations")}
    return metrics, aux


# Metrics where LOWER is better (inverted for the headline average).
_LOWER_BETTER = {"hallucination_rate"}


def _headline(metrics):
    vals = []
    for k, v in metrics.items():
        vals.append((1.0 - v) if k in _LOWER_BETTER else v)
    return round(sum(vals) / len(vals), 3) if vals else 0.0


# --------------------------------------------------------------------------- #
def run(tasks, depth):
    from deep_research import run_deep_research
    ctx = make_ctx()
    rows = []
    for i, task in enumerate(tasks, 1):
        topic = task["topic"]
        print(f"\n[{i}/{len(tasks)}] ({task['category']}) {topic}", flush=True)
        t0 = time.time()
        try:
            res = run_deep_research(ctx, topic, depth=depth)
            run_dir = os.path.dirname(res.get("path") or "") if res.get("path") else ""
            if not run_dir:
                # locate the freshly-written run dir by slug
                run_dir = _latest_run_dir(topic)
            metrics, aux = score_run(task, res, run_dir)
            rows.append({"topic": topic, "category": task["category"],
                         "elapsed_sec": round(time.time() - t0, 1),
                         "metrics": metrics, "headline": _headline(metrics),
                         "aux": aux, "run_dir": run_dir})
            print("   " + "  ".join(f"{k}={v}" for k, v in metrics.items()), flush=True)
            print(f"   headline={rows[-1]['headline']}  sources={aux['sources']} "
                  f"levels={aux['hierarchy_levels']} reflect={aux['reflection_iterations']}",
                  flush=True)
        except Exception as exc:
            traceback.print_exc()
            rows.append({"topic": topic, "category": task["category"], "error": str(exc)})
    return rows


def _latest_run_dir(topic):
    import deep_research as dr
    slug = dr._slug(topic)
    cands = sorted((p for p in dr.DR_DIR.glob(f"*_{slug}") if p.is_dir()),
                   key=lambda p: p.stat().st_mtime)
    return str(cands[-1]) if cands else ""


def aggregate(rows):
    ok = [r for r in rows if "metrics" in r]
    if not ok:
        return {}
    keys = list(ok[0]["metrics"].keys())
    agg = {k: round(sum(r["metrics"][k] for r in ok) / len(ok), 3) for k in keys}
    by_cat = {}
    for r in ok:
        by_cat.setdefault(r["category"], []).append(r["headline"])
    cat_avg = {c: round(sum(v) / len(v), 3) for c, v in by_cat.items()}
    return {"n": len(ok), "n_failed": len(rows) - len(ok),
            "metric_means": agg,
            "headline_mean": round(sum(r["headline"] for r in ok) / len(ok), 3),
            "by_category": cat_avg}


_TREND_FILE = os.path.join(OUT, "evaluation_trend.jsonl")
_REGRESSION_DROP = 0.05  # a metric mean falling by more than this vs last run = regression


def _append_trend(summary, depth):
    """Append this run's headline + metric means to an append-only trend log so
    regressions across runs are visible over time (the file is never overwritten,
    unlike evaluation_report.json). Returns (prev_entry, deltas, regressions)."""
    os.makedirs(OUT, exist_ok=True)
    prev = None
    if os.path.exists(_TREND_FILE):
        try:
            with open(_TREND_FILE, encoding="utf-8") as f:
                lines = [ln for ln in f if ln.strip()]
            if lines:
                prev = json.loads(lines[-1])
        except Exception:
            prev = None
    entry = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "depth": depth,
             "n": summary.get("n", 0), "headline_mean": summary.get("headline_mean"),
             "metric_means": summary.get("metric_means", {})}
    with open(_TREND_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    deltas, regressions = {}, []
    if prev:
        pm = prev.get("metric_means", {})
        for k, v in entry["metric_means"].items():
            if k in pm and isinstance(v, (int, float)):
                d = round(v - pm[k], 3)
                deltas[k] = d
                worse = d > _REGRESSION_DROP if k in _LOWER_BETTER else d < -_REGRESSION_DROP
                if worse:
                    regressions.append((k, pm[k], v, d))
        if prev.get("headline_mean") is not None and entry["headline_mean"] is not None:
            deltas["headline_mean"] = round(entry["headline_mean"] - prev["headline_mean"], 3)
    return prev, deltas, regressions


def write_reports(rows, summary, depth):
    prev, deltas, regressions = _append_trend(summary, depth)
    js = os.path.join(OUT, "evaluation_report.json")
    json.dump({"summary": summary, "depth": depth, "rows": rows,
               "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "trend": {"prev": prev, "deltas": deltas,
                         "regressions": [list(r) for r in regressions]}},
              open(js, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    md = [f"# Deep Research — Scored Evaluation\n",
          f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} · depth={depth} · "
          f"{summary.get('n',0)} scored, {summary.get('n_failed',0)} failed_\n",
          f"**Headline mean: {summary.get('headline_mean')}**\n",
          "## Metric means\n", "| metric | mean |", "| :-- | --: |"]
    for k, v in summary.get("metric_means", {}).items():
        d = deltas.get(k)
        darr = "" if not d else (f"  (▲ +{d})" if d > 0 else f"  (▼ {d})")
        md.append(f"| {k} | {v}{darr} |")
    if prev:
        hd = deltas.get("headline_mean")
        md.append(f"\n## Trend vs previous run ({prev.get('ts','?')})\n"
                  f"Headline {prev.get('headline_mean')} → {summary.get('headline_mean')}"
                  + ("" if hd is None else f" (Δ {hd:+})"))
        if regressions:
            md.append("\n**⚠ REGRESSIONS (metric dropped > %.2f):**" % _REGRESSION_DROP)
            for k, was, now, d in regressions:
                md.append(f"- {k}: {was} → {now} (Δ {d})")
        else:
            md.append("\nNo metric regressions vs previous run. ✓")
    md.append("\n## By category\n| category | headline |\n| :-- | --: |")
    for c, v in summary.get("by_category", {}).items():
        md.append(f"| {c} | {v} |")
    md.append("\n## Per-task\n| topic | cat | headline | src | lvls | reflect | decision |\n"
              "| :-- | :-- | --: | --: | --: | --: | :-- |")
    for r in rows:
        if "metrics" not in r:
            md.append(f"| {r['topic'][:48]} | {r['category']} | ERROR | | | | |")
            continue
        a = r["aux"]
        md.append(f"| {r['topic'][:48]} | {r['category']} | {r['headline']} | "
                  f"{a['sources']} | {a['hierarchy_levels']} | "
                  f"{a['reflection_iterations']} | {a['decision']} |")
    open(os.path.join(OUT, "evaluation_report.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    return js


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="run every task")
    ap.add_argument("--limit", type=int, default=0, help="run the first N tasks")
    ap.add_argument("--category", default="", help="only this category")
    ap.add_argument("--depth", default="quick", help="quick|standard|deep")
    ap.add_argument("--sample-per-cat", type=int, default=0,
                    help="take N from each category (spread)")
    args = ap.parse_args()

    tasks = TASKS
    if args.category:
        tasks = [t for t in TASKS if t["category"] == args.category]
    if args.sample_per_cat:
        cats = by_category()
        tasks = []
        for c, ts in cats.items():
            tasks.extend(ts[:args.sample_per_cat])
    if args.limit and not args.all:
        tasks = tasks[:args.limit]
    elif not args.all and not args.limit and not args.category and not args.sample_per_cat:
        tasks = tasks[:6]   # default smoke set

    print(f"Running {len(tasks)} task(s) at depth={args.depth}…", flush=True)
    rows = run(tasks, args.depth)
    summary = aggregate(rows)
    path = write_reports(rows, summary, args.depth)
    print("\n" + "=" * 60)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {path} and evaluation_report.md")


if __name__ == "__main__":
    main()

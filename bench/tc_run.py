r"""Tool-calling bench: drive the REAL agent loop over SIMULATED tool nodes.

    .\venv\Scripts\python.exe bench\tc_run.py [--reps 3] [--cat route] [--id X]

Live LM Studio is required (that is the component under test). Nothing else is
live: every tool is a stub from bench.tc_mocks, so no GPU, no ComfyUI, no
network, no clipboard, no files written outside outputs/.

This is a BENCH, not a test suite -- it lives outside tests/ on purpose so the
offline sweeps never pick it up and never make a live model call.
"""
import argparse
import json
import logging
import os
import statistics
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

# bench/knowledge.py would shadow the app's knowledge package: drop this dir from the path
sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import requests
import config
from bench.tc_mocks import ToolRecorder
from bench.tc_cases import cases_for


def _served_model():
    r = requests.get(f"{config.LM_STUDIO_BASE}/v1/models", timeout=10)
    ids = [m["id"] for m in r.json().get("data", [])
           if "embed" not in m["id"] and "bge" not in m["id"]]
    if not ids:
        raise SystemExit("LM Studio has no chat model loaded.")
    return ids[0]


def _make_ctx(model_id):
    from models import Models, Context

    class StubModels:
        whisper = tts_model = vocoder = accentor = None
        accentor_loaded = False

    ctx = Context(
        models=StubModels(),
        transcription_cache={},
        cache_file=Path("outputs/_bench_tc_cache.json"),
        asr_lock=threading.Lock(),
        tts_lock=threading.Lock(),
    )
    ctx.model_name = model_id
    ctx.no_think = True
    ctx.web_search_enabled = True
    return ctx


def setup():
    """Shared boot for every bench in this family: (graph, ctx, image_stub)."""
    model_id = _served_model()
    print(f"LM Studio model under test: {model_id}")
    print(f"base = {config.LM_STUDIO_BASE}\n")
    import graph
    ctx = _make_ctx(model_id)
    image_stub = str(Path("outputs/_bench_current.png").resolve())
    Path("outputs").mkdir(exist_ok=True)
    Path(image_stub).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 64)
    return graph, ctx, image_stub


def run_case(graph, ctx, case, image_stub, *, recorder=None, schemas=None):
    """One case, one repetition. Returns a result dict.

    `recorder` lets a hardening bench (chaos, canary) supply its own
    ToolRecorder subclass; `schemas` temporarily replaces graph.TOOL_SCHEMAS
    for this one case (decoy injection / description mutation). Both are
    optional and the default behaviour is exactly what tc_run always did.
    """
    rec = recorder if recorder is not None else ToolRecorder(fail=case.get("fail"))
    orig_exec = graph.execute_tool
    graph.execute_tool = rec
    orig_schemas = graph.TOOL_SCHEMAS
    if schemas is not None:
        graph.TOOL_SCHEMAS = schemas

    ctx.session_memory.clear()
    ctx.pinned_facts = []
    ctx.last_image_path = image_stub if case.get("image") else None

    state = {
        "messages": [],
        "user_input": case["text"],
        "image_data": None,
    }
    t0 = time.perf_counter()
    err = None
    try:
        state = graph.personality_node(ctx, state)
    except Exception as exc:            # a crash is a bench result, not a stop
        err = f"{type(exc).__name__}: {exc}"
    finally:
        graph.execute_tool = orig_exec
        graph.TOOL_SCHEMAS = orig_schemas
    dt = time.perf_counter() - t0

    names = rec.names
    expect = case.get("expect")
    forbid = set(case.get("forbid") or [])
    missing_all = [t for t in (case.get("expect_all") or []) if t not in names]
    order = case.get("order") or []

    if err:
        verdict, why = "ERROR", err
    elif missing_all:
        verdict, why = "FAIL", f"chain incomplete, missing {missing_all} (called {names})"
    elif order and not _is_subsequence(order, names):
        verdict, why = "FAIL", f"wrong order: wanted {order} within {names}"
    elif expect is None:
        # need-tool gate: any call at all is a miss
        verdict = "PASS" if not names else "FAIL"
        why = "no tool called" if not names else f"called {names}"
    elif expect not in names:
        verdict, why = "FAIL", f"expected {expect}, called {names or 'nothing'}"
    elif forbid & set(names):
        verdict, why = "FAIL", f"called forbidden {sorted(forbid & set(names))}"
    elif case.get("min_calls") and len(names) < case["min_calls"]:
        verdict, why = "FAIL", f"only {len(names)} call(s), wanted retry"
    elif case.get("max_calls") is not None and len(names) > case["max_calls"]:
        # The mirror of min_calls, for the tools where retrying is the WRONG
        # answer: a deck or a video is minutes of work, so a second blind
        # attempt at a broken backend costs the user more than an honest
        # "it failed, here is what I can do instead" (see the retry exclusion
        # list in graph_personality._RETRY_ON_FIRST_FAILURE).
        verdict, why = "FAIL", (f"{len(names)} calls, wanted at most "
                                f"{case['max_calls']} — an expensive tool must "
                                f"not be blind-retried")
    elif case.get("check"):
        got = rec.args_for(expect)
        ok = any(_safe_check(case["check"], a) for a in got)
        verdict = "PASS" if ok else "FAIL"
        why = "args ok" if ok else f"bad args: {got}"
    else:
        verdict, why = "PASS", f"called {names}"

    return dict(id=case["id"], cat=case["cat"], verdict=verdict, why=why,
                expect_tool=expect, calls=names, seconds=round(dt, 1),
                answer=(state.get("final_answer") or "")[:160] if not err else "",
                # The scorer needs the STATE, not just the call list: whether a
                # tool really delivered is decided by the artifact it left
                # (image_path / video_path / document_path), and a call list
                # cannot tell a working fallback from a claimed one.
                state=(state if not err else {}))


def _is_subsequence(wanted, got):
    """True if `wanted` appears inside `got` in order (gaps allowed)."""
    it = iter(got)
    return all(any(g == w for g in it) for w in wanted)


def _safe_check(fn, args):
    try:
        return bool(fn(args))
    except Exception:
        return False



class _ContextGuard(logging.Handler):
    """Abort the run when the model cannot hold the agent's prompt.

    A model loaded too small does not produce bad routing, it produces NO
    routing: every tool-bearing call is rejected and every case scores zero.
    That is indistinguishable in the summary from an agent that got everything
    wrong, and it is how a contaminated run gets reported as a measurement --
    which nearly happened here, against a model loaded at 12288 with parallel 2,
    giving each request 6144 while the agent's prompt needs ~6800.

    Fails loudly on the FIRST occurrence rather than at the end: a run this
    broken has nothing worth waiting for.
    """

    def emit(self, record):
        msg = record.getMessage()
        if "does not fit the loaded context" in msg or "nothing left to trim" in msg:
            print("\n" + "=" * 78, file=sys.stderr)
            print("  ABORTED — the model cannot hold this prompt, so nothing "
                  "here measures routing.", file=sys.stderr)
            print("  " + msg, file=sys.stderr)
            print("  Check `lms ps`: CONTEXT is divided by PARALLEL, so "
                  "12288 with parallel 2\n  gives each request only 6144.",
                  file=sys.stderr)
            print("=" * 78, file=sys.stderr)
            os._exit(2)


def _arm_context_guard():
    h = _ContextGuard()
    h.setLevel(logging.ERROR)
    logging.getLogger("assistant.llm").addHandler(h)


def main():
    _arm_context_guard()
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--cat", action="append")
    ap.add_argument("--id", action="append")
    ap.add_argument("--out", default="outputs/_bench_tc.json")
    args = ap.parse_args()

    # A picture "already on screen" for the image cases: the loop only checks
    # that the path is set, and every image tool is stubbed.
    graph, ctx, image_stub = setup()

    cases = cases_for(args.cat, args.id)
    results = []
    t_start = time.time()
    for rep in range(args.reps):
        for case in cases:
            r = run_case(graph, ctx, case, image_stub)
            r["rep"] = rep
            results.append(r)
            mark = {"PASS": "PASS", "FAIL": "FAIL", "ERROR": "ERR "}[r["verdict"]]
            print(f"[{mark}] {r['id']:<22} {r['seconds']:>5.1f}s  {r['why']}")

    _report(results, cases, args.reps, time.time() - t_start, args.out)


def _report(results, cases, reps, elapsed, out_path):
    print("\n" + "=" * 72)
    by_cat = defaultdict(lambda: [0, 0])
    for r in results:
        by_cat[r["cat"]][1] += 1
        by_cat[r["cat"]][0] += (r["verdict"] == "PASS")
    for cat in sorted(by_cat):
        ok, tot = by_cat[cat]
        print(f"  {cat:<9} {ok:>3}/{tot:<3}  {100*ok/tot:5.1f}%")
    ok = sum(r["verdict"] == "PASS" for r in results)
    print("-" * 72)
    print(f"  TOTAL     {ok:>3}/{len(results):<3}  {100*ok/len(results):5.1f}%"
          f"   ({elapsed/60:.1f} min, {reps} rep(s))")

    # Per-case instability: a case that flips between reps is a routing
    # coin-flip, which is worse news than a case that fails every time.
    if reps > 1:
        flaky = []
        for c in cases:
            vs = [r["verdict"] for r in results if r["id"] == c["id"]]
            if len(set(vs)) > 1:
                flaky.append((c["id"], vs))
        if flaky:
            print("\n  unstable (routing coin-flip):")
            for cid, vs in flaky:
                print(f"    {cid:<22} {vs}")

    secs = [r["seconds"] for r in results]
    print(f"\n  latency: median {statistics.median(secs):.1f}s  "
          f"max {max(secs):.1f}s")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print(f"\n  raw -> {out_path}")


if __name__ == "__main__":
    main()

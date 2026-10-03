"""Boundary-stubbed coverage for deep_research.py's search/query-planning layer:
_search_one_query, collect_sources (serial + parallel paths), plan_queries,
_corrective_queries. Stubs raw_search_results / call_llm_simple / _save_state so
every branch (mutation retry, cancellation, junk/domain/type filtering, JSON vs
heuristic fallback, recency floor/timeless clamp) runs deterministically without
live network or LM Studio.
Run: venv/Scripts/python.exe tests/test_deep_research_search.py
"""
import os, sys, tempfile, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
from pathlib import Path
import deep_research as DR

RESULTS = []
import os as _os
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + detail))
    if not cond and _os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))

_TMP = Path(tempfile.mkdtemp(prefix="drsearch_"))


# The engine is split across deep_research + the dr_* stage modules, so a seam
# has to be patched in the module that USES it. `_dr_patch.Patches` finds every
# home of a name and rebinds all of them (and raises if nothing binds it, so a
# stale patch can never degrade into a silent real network/LM Studio call).
from _dr_patch import Patches as _Patches


class FakeCtx:
    def __init__(self, cancelled=False):
        self._cancelled = cancelled
    def is_cancelled(self):
        return self._cancelled


def h(url, title="T"):
    return {"href": url, "title": title}


# ---------------- _search_one_query ----------------

def test_search_one_query():
    with _Patches(raw_search_results=lambda ctx, q, n, timelimit=None, news=False: [h("https://a.com/1")]):
        out = DR._search_one_query(FakeCtx(), "q", 5, None, False)
        check("search_one_success", out == [h("https://a.com/1")])

    calls = {"n": 0}
    def raiser(ctx, q, n, timelimit=None, news=False):
        calls["n"] += 1
        raise RuntimeError("boom")
    with _Patches(raw_search_results=raiser):
        out = DR._search_one_query(FakeCtx(), "q", 5, None, False)
        check("search_one_exception_then_mutation_exhausted_empty", out == [])

    def news_adder(ctx, q, n, timelimit=None, news=False):
        return [h(f"https://news.com/{1 if news else 0}")]
    with _Patches(raw_search_results=news_adder):
        out = DR._search_one_query(FakeCtx(), "q", 5, None, True)
        check("search_one_news_appended", len(out) == 2)

    # zero-hit then mutation recovers
    seq = {"i": 0}
    def zero_then_hit(ctx, q, n, timelimit=None, news=False):
        seq["i"] += 1
        if seq["i"] == 1:
            return []
        return [h("https://recovered.com/1")]
    with _Patches(raw_search_results=zero_then_hit, DR_QUERY_MUTATION_ATTEMPTS=2):
        out = DR._search_one_query(FakeCtx(), "foo bar baz", 5, None, False)
        check("search_one_mutation_recovers", out == [h("https://recovered.com/1")])

    # zero-hit, mutation attempt returns None (attempt exceeds strategies) -> loop continues/exits
    def always_empty(ctx, q, n, timelimit=None, news=False):
        return []
    with _Patches(raw_search_results=always_empty, DR_QUERY_MUTATION_ATTEMPTS=5):
        out = DR._search_one_query(FakeCtx(), "x", 5, None, False)
        check("search_one_mutation_exhausted", out == [])

    # mutated search itself raises -> caught, hits stays []
    def first_empty_then_raise(ctx, q, n, timelimit=None, news=False):
        if q == "foo bar baz":
            return []
        raise RuntimeError("mutated call failed")
    with _Patches(raw_search_results=first_empty_then_raise, DR_QUERY_MUTATION_ATTEMPTS=1):
        out = DR._search_one_query(FakeCtx(), "foo bar baz", 5, None, False)
        check("search_one_mutated_call_raises", out == [])


# ---------------- collect_sources ----------------

def test_collect_sources_serial():
    prog = DR._Progress(None)
    hits = [h("https://arxiv.org/1"), h("https://kinogo.pro/junk"),  # junk dropped
            h("https://arxiv.org/1")]  # dup dropped
    with _Patches(_search_one_query=lambda ctx, q, n, tl, un: hits, DR_SEARCH_CONCURRENCY=1,
                  _save_state=lambda *a, **k: None):
        out = DR.collect_sources(FakeCtx(), ["q1"], 5, prog, _TMP, {"category": "science"})
        check("collect_sources_dedupes_and_drops_junk", len(out) == 1 and out[0]["href"] == "https://arxiv.org/1")

    with _Patches(_search_one_query=lambda ctx, q, n, tl, un: [], DR_SEARCH_CONCURRENCY=1,
                  _save_state=lambda *a, **k: None):
        check("collect_sources_cancelled_short_circuits",
              DR.collect_sources(FakeCtx(cancelled=True), ["q1", "q2"], 5, prog, _TMP) == [])

    with _Patches(_search_one_query=lambda ctx, q, n, tl, un: [h("https://blocked.com/x")],
                  DR_SEARCH_CONCURRENCY=1, _save_state=lambda *a, **k: None,
                  DR_DOMAIN_BLACKLIST="blocked.com"):
        out = DR.collect_sources(FakeCtx(), ["q1"], 5, prog, _TMP)
        check("collect_sources_domain_blacklist", out == [])

    with _Patches(_search_one_query=lambda ctx, q, n, tl, un: [h("https://reddit.com/r/x")],
                  DR_SEARCH_CONCURRENCY=1, _save_state=lambda *a, **k: None,
                  DR_SOURCE_TYPES_EXCLUDE="forum"):
        out = DR.collect_sources(FakeCtx(), ["q1"], 5, prog, _TMP)
        check("collect_sources_source_type_excluded", out == [])

    with _Patches(_search_one_query=lambda ctx, q, n, tl, un: [h(f"https://x{i}.com/a") for i in range(5)],
                  DR_SEARCH_CONCURRENCY=1, _save_state=lambda *a, **k: None, DR_MAX_SOURCES=2):
        out = DR.collect_sources(FakeCtx(), ["q1"], 5, prog, _TMP)
        check("collect_sources_max_sources_cap", len(out) == 2)


def test_collect_sources_parallel():
    prog = DR._Progress(None)
    def fake_search(ctx, q, n, tl, un):
        return [h(f"https://arxiv.org/{q}")]
    with _Patches(_search_one_query=fake_search, DR_SEARCH_CONCURRENCY=4,
                  _save_state=lambda *a, **k: None):
        out = DR.collect_sources(FakeCtx(), ["q1", "q2", "q3"], 5, prog, _TMP, {"category": "science"})
        check("collect_sources_parallel_gathers_all", len(out) == 3)

    def raising_search(ctx, q, n, tl, un):
        raise RuntimeError("boom")
    with _Patches(_search_one_query=raising_search, DR_SEARCH_CONCURRENCY=4,
                  _save_state=lambda *a, **k: None):
        out = DR.collect_sources(FakeCtx(), ["q1", "q2"], 5, prog, _TMP)
        check("collect_sources_parallel_future_exception_swallowed", out == [])

    with _Patches(_search_one_query=fake_search, DR_SEARCH_CONCURRENCY=4,
                  _save_state=lambda *a, **k: None):
        out = DR.collect_sources(FakeCtx(cancelled=True), ["q1", "q2"], 5, prog, _TMP)
        check("collect_sources_parallel_cancel_breaks_early", isinstance(out, list))


# ---------------- plan_queries ----------------

def test_plan_queries_llm_json():
    payload = '{"queries": ["query one", "query two"], "category": "science", ' \
              '"recency": "any", "core_topic": "quantum tunneling"}'
    with _Patches(_think_call=lambda ctx, sys_, usr, max_tokens, temperature=0.3: payload):
        queries, profile = DR.plan_queries(FakeCtx(), "Explain quantum tunneling in depth", 5)
        check("plan_queries_json_used", "quantum tunneling" in queries[0])
        check("plan_queries_category_set", profile["category"] == "science")
        check("plan_queries_timeless_forces_any", profile["recency"] == "any")


def test_plan_queries_heuristic_fallback():
    with _Patches(_think_call=lambda *a, **k: "not json at all garbage"):
        queries, profile = DR.plan_queries(FakeCtx(), "some short raw topic", 5)
        check("plan_queries_heuristic_expansion", len(queries) >= 3)
        check("plan_queries_default_category", profile["category"] == "general")


def test_plan_queries_recency_floor_and_news():
    payload = '{"queries": ["a job posting query"], "category": "jobs", "recency": "any"}'
    with _Patches(_think_call=lambda *a, **k: payload):
        queries, profile = DR.plan_queries(FakeCtx(), "find a job posting", 5)
        check("plan_queries_recency_floor_clamped", profile["recency"] == "month")
    payload2 = '{"queries": ["breaking news today"], "category": "news", "recency": "year"}'
    with _Patches(_think_call=lambda *a, **k: payload2):
        queries2, profile2 = DR.plan_queries(FakeCtx(), "breaking news today", 5)
        check("plan_queries_news_true", profile2["news"] is True)
        check("plan_queries_recency_floor_news", profile2["recency"] == "week")
    payload3 = '{"queries": ["x"], "category": "general", "recency": "bogus_value"}'
    with _Patches(_think_call=lambda *a, **k: payload3):
        _, profile3 = DR.plan_queries(FakeCtx(), "x topic", 5)
        check("plan_queries_invalid_recency_defaults_any", profile3["recency"] == "any")


def test_plan_queries_long_multiline_topic_uses_first_line():
    payload = '{"queries": [], "category": "general", "recency": "any"}'
    with _Patches(_think_call=lambda *a, **k: payload):
        long_topic = "Please investigate: quantum tunneling deeply\nwith lots more detail " + "x" * 200
        queries, profile = DR.plan_queries(FakeCtx(), long_topic, 5)
        check("plan_queries_distills_first_line", "quantum tunneling deeply" in profile["core_topic"])


# ---------------- _corrective_queries ----------------

def test_corrective_queries():
    with _Patches(plan_queries=lambda ctx, hint, mx: (["new broader query", "another one"], {})):
        out = DR._corrective_queries(FakeCtx(), "topic", {"category": "science"},
                                     ["old query"], 5)
        check("corrective_queries_returns_fresh", "new broader query" in out)

    with _Patches(plan_queries=lambda ctx, hint, mx: (_ for _ in ()).throw(RuntimeError("boom"))):
        out = DR._corrective_queries(FakeCtx(), "topic", {"category": "general"}, ["old"], 5)
        check("corrective_queries_exception_returns_empty", out == [])

    with _Patches(plan_queries=lambda ctx, hint, mx: (["old query", "topic"], {})):
        out = DR._corrective_queries(FakeCtx(), "topic", {"category": "local"}, ["old query"], 5)
        check("corrective_queries_dedupes_against_existing_and_topic", out == [])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback; traceback.print_exc()
            print("  ERROR in " + fn.__name__ + ": " + type(e).__name__ + ": " + str(e))
    passed = sum(1 for _, c in RESULTS if c)
    print(f"\n{len(fns)-failed}/{len(fns)} functions, {passed}/{len(RESULTS)} checks passed")
    # A failed check() must reach the exit code. This used to be `1 if failed else 0`,
    # where `failed` counts EXCEPTIONS only -- so every assertion in the file could
    # print FAIL while the suite still exited 0, invisible to anything judging by
    # exit code (which is how these suites are judged).
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)

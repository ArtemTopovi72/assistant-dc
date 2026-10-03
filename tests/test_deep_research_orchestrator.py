"""Integration coverage for deep_research.py's orchestrator: run_deep_research +
_finish. Stubs only the network/LLM boundary functions (plan_queries,
collect_sources, crawl_pages, dedupe_pages, rerank_for_briefing, _brief_pages,
citation_lineage_brief, synthesize_report, synthesize_survey) and lets the REAL
subsystem modules (entities, contradiction, clustering, research_graph,
decision_gate, graph_guided, claim_merge, hierarchical, hierarchy, communities,
formula_audit) run for real on small synthetic brief sets — these are pure/local,
no GPU or network. Exercises the feature-flag branches (multihop, contradiction,
hierarchical, communities, claim-merge, graph-expansion, decision-gate, survey
mode, replan loop) plus the early-return / empty-topic / no-sources / no-pages /
no-briefs / cancelled paths.
Run: venv/Scripts/python.exe tests/test_deep_research_orchestrator.py
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

_TMP = Path(tempfile.mkdtemp(prefix="drorch_"))


# The engine is split across deep_research + the dr_* stage modules, so a seam
# has to be patched in the module that USES it. `_dr_patch.Patches` finds every
# home of a name and rebinds all of them (and raises if nothing binds it, so a
# stale patch can never degrade into a silent real network/LM Studio call).
from _dr_patch import Patches as _Patches


class FakeCtx:
    def __init__(self, cancelled=False):
        self._cancelled = cancelled
        self.api_lock = threading.Lock()
        self.api_min_interval = 0.0
        self.last_api_call_time = 0.0
        self.last_research_report = None
        self.last_research_path = None
    def is_cancelled(self):
        return self._cancelled


def _brief(domain, url, title, brief, trust="SECONDARY", mode="confirm"):
    return {"domain": domain, "url": url, "title": title, "brief": brief, "trust": trust,
            "authority": 0.7, "mode": mode, "cluster_size": 1, "cluster_domains": [domain],
            "equations": []}


BRIEFS = [
    _brief("arxiv.org", "https://arxiv.org/1", "Adam Optimizer Paper",
           "Adam optimizer combines momentum and adaptive learning rates for training."),
    _brief("nature.com", "https://nature.com/2", "Adam Analysis",
           "Further analysis of the Adam optimizer convergence properties."),
    _brief("randomblog.com", "https://randomblog.com/3", "Blog About Adam",
           "A blog post discussing Adam optimizer usage in practice.", trust="COMMUNITY"),
]

BASE_PATCHES = dict(
    plan_queries=lambda ctx, topic, mx: (["adam optimizer query"], {"category": "science", "recency": "any", "core_topic": "adam optimizer"}),
    collect_sources=lambda ctx, queries, per_q, prog, run_dir, profile=None: [{"href": "https://arxiv.org/1", "domain": "arxiv.org", "query": "q"}],
    crawl_pages=lambda ctx, sources, caps, prog, run_dir, profile=None, cache=None: [
        {"url": "https://arxiv.org/1", "domain": "arxiv.org", "title": "T", "text": "adam optimizer text", "equations": []}],
    dedupe_pages=lambda pages, cat: pages,
    rerank_for_briefing=lambda topic, pages, profile, prog: pages,
    _brief_pages=lambda ctx, topic, pages, profile, run_dir, prog, **kw: (list(BRIEFS), 0),
    synthesize_report=lambda ctx, topic, briefs, prog, **kw: "# Synthesized Report\n\nBody content about Adam.",
    citation_lineage_brief=lambda cite_query, mailto: None,
    DR_FETCH_DELAY=0, DR_CACHE_ENABLED=False, DR_REPLAN_ENABLED=False, DR_CITATIONS_ENABLED=False,
)


def _run(extra_patches=None, topic="research the adam optimizer", ctx=None):
    patches = dict(BASE_PATCHES)
    if extra_patches:
        patches.update(extra_patches)
    with _Patches(**patches):
        return DR.run_deep_research(ctx or FakeCtx(), topic, depth="quick")


def test_empty_topic():
    out = _run(topic="   ")
    check("empty_topic_returns_error", out["error"] == "empty topic" and out["report"] == "")


def test_cancelled_after_planning():
    out = _run(ctx=FakeCtx(cancelled=True))
    check("cancelled_early_returns", out["cancelled"] is True)


def test_no_sources():
    out = _run(extra_patches={"collect_sources": lambda *a, **k: []})
    check("no_sources_report_says_so", "No web sources" in out["report"])


def test_no_pages():
    out = _run(extra_patches={"crawl_pages": lambda *a, **k: []})
    check("no_pages_report_says_so", "none could be" in out["report"])


def test_no_briefs():
    out = _run(extra_patches={"_brief_pages": lambda *a, **k: ([], 0)})
    check("no_briefs_report_says_so", "none contained" in out["report"])


def test_basic_success_no_optional_features():
    with _Patches(DR_REPLAN_ENABLED=False, DR_MULTIHOP_ENABLED=False, DR_CONTRADICTION_ENABLED=False,
                  DR_HIERARCHICAL_ENABLED=False, DR_COMMUNITIES_ENABLED=False, DR_CLAIM_MERGE_ENABLED=False,
                  DR_GRAPH_EXPANSION_ENABLED=False, DR_DECISION_GATE_ENABLED=False, DR_REFLECTION_ENABLED=False,
                  DR_SURVEY_MODE=False, DR_CITATIONS_ENABLED=False):
        out = _run()
        check("basic_success_has_report", "Synthesized Report" in out["report"])
        check("basic_success_has_path", out["path"] is not None and Path(out["path"]).exists())
        check("basic_success_not_cancelled", out["cancelled"] is False)


def test_all_optional_features_enabled():
    with _Patches(DR_REPLAN_ENABLED=False, DR_MULTIHOP_ENABLED=True, DR_MULTIHOP_DEPTH=1,
                  DR_MULTIHOP_ENTITIES=3,
                  DR_CONTRADICTION_ENABLED=True, DR_CONTRADICTION_QUERIES=2,
                  DR_HIERARCHICAL_ENABLED=True, DR_CLUSTER_THRESHOLD=0.3, DR_CLUSTER_MAJOR_MIN=1,
                  DR_COMMUNITIES_ENABLED=True,
                  DR_CLAIM_MERGE_ENABLED=True, DR_CLAIM_MERGE_THRESHOLD=0.9,
                  DR_GRAPH_EXPANSION_ENABLED=True, DR_GRAPH_EXPANSION_QUERIES=2,
                  DR_DECISION_GATE_ENABLED=False, DR_REFLECTION_ENABLED=False,
                  DR_SURVEY_MODE=False, DR_CITATIONS_ENABLED=True,
                  _expansion_pass=lambda *a, **k: [],
                  citation_lineage_brief=lambda q, mailto: "SOURCE: PRIMARY\ncited_by=99 seminal work"):
        out = _run()
        check("all_features_success", "Synthesized Report" in out["report"] or "Citation" in out["report"])
        check("all_features_has_path", out["path"] is not None)


def test_survey_mode():
    with _Patches(DR_REPLAN_ENABLED=False, DR_MULTIHOP_ENABLED=False, DR_CONTRADICTION_ENABLED=False,
                  DR_HIERARCHICAL_ENABLED=False, DR_COMMUNITIES_ENABLED=False, DR_CLAIM_MERGE_ENABLED=False,
                  DR_GRAPH_EXPANSION_ENABLED=False, DR_DECISION_GATE_ENABLED=False, DR_REFLECTION_ENABLED=False,
                  DR_SURVEY_MODE=True, DR_CITATIONS_ENABLED=False,
                  synthesize_survey=lambda ctx, topic, briefs, prog, **kw: "# Survey Title\n\nSurvey body content."):
        out = _run()
        check("survey_mode_used", "Survey Title" in out["report"])


def test_survey_mode_fails_falls_back_to_report():
    def raiser(*a, **k):
        raise RuntimeError("survey boom")
    with _Patches(DR_REPLAN_ENABLED=False, DR_MULTIHOP_ENABLED=False, DR_CONTRADICTION_ENABLED=False,
                  DR_HIERARCHICAL_ENABLED=False, DR_COMMUNITIES_ENABLED=False, DR_CLAIM_MERGE_ENABLED=False,
                  DR_GRAPH_EXPANSION_ENABLED=False, DR_DECISION_GATE_ENABLED=False, DR_REFLECTION_ENABLED=False,
                  DR_SURVEY_MODE=True, DR_CITATIONS_ENABLED=False,
                  synthesize_survey=raiser):
        out = _run()
        check("survey_failure_falls_back", "Synthesized Report" in out["report"])


def test_decision_gate_asks():
    fake_decision = {"decision": "ASK", "reasons": ["not enough evidence"],
                     "clarifying_question": "Can you clarify?"}
    with _Patches(DR_REPLAN_ENABLED=False, DR_MULTIHOP_ENABLED=False, DR_CONTRADICTION_ENABLED=False,
                  DR_HIERARCHICAL_ENABLED=False, DR_COMMUNITIES_ENABLED=False, DR_CLAIM_MERGE_ENABLED=False,
                  DR_GRAPH_EXPANSION_ENABLED=False, DR_DECISION_GATE_ENABLED=True, DR_REFLECTION_ENABLED=False,
                  DR_SURVEY_MODE=False, DR_CITATIONS_ENABLED=False,
                  _gate=type(sys)("fakegate")):
        DR._gate.decide = lambda *a, **k: fake_decision
        DR._gate.assess_ambiguity = lambda *a, **k: {}
        # gate_banner takes the output language too — the banner sits at the top
        # of the finished document and was English inside Russian reports.
        DR._gate.gate_banner = lambda d, out_lang="en": "GATE: " + d["decision"]
        DR._gate.ASK = "ASK"
        DR._gate.ABSTAIN = "ABSTAIN"
        out = _run()
        check("decision_gate_ask_stops_before_synthesis", "clarification" in out["report"].lower())


def test_replan_loop_broadens_and_stagnates():
    calls = {"n": 0}
    def cs(ctx, queries, per_q, prog, run_dir, profile=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return [{"href": "https://a.com/1", "domain": "a.com", "query": "q"}]  # only 1 source -> weak/thin
        return []  # corrective round finds nothing new -> stagnation stop
    with _Patches(DR_REPLAN_ENABLED=True, DR_MAX_COLLECTION_ROUNDS=2,
                  DR_MULTIHOP_ENABLED=False, DR_CONTRADICTION_ENABLED=False,
                  DR_HIERARCHICAL_ENABLED=False, DR_COMMUNITIES_ENABLED=False, DR_CLAIM_MERGE_ENABLED=False,
                  DR_GRAPH_EXPANSION_ENABLED=False, DR_DECISION_GATE_ENABLED=False, DR_REFLECTION_ENABLED=False,
                  DR_SURVEY_MODE=False, DR_CITATIONS_ENABLED=False,
                  collect_sources=cs,
                  _corrective_queries=lambda ctx, topic, profile, existing, mx: ["broader query"]):
        out = _run()
        check("replan_loop_ran", "Synthesized Report" in out["report"])


def test_reflection_loop_runs_and_stops():
    call = {"n": 0}
    def expansion(*a, **k):
        call["n"] += 1
        if call["n"] == 1:
            return [_brief("new.com", "https://new.com/1", "New", "new evidence found", trust="PRIMARY")]
        return []
    with _Patches(DR_REPLAN_ENABLED=False, DR_MULTIHOP_ENABLED=False, DR_CONTRADICTION_ENABLED=False,
                  DR_HIERARCHICAL_ENABLED=False, DR_COMMUNITIES_ENABLED=False, DR_CLAIM_MERGE_ENABLED=False,
                  DR_GRAPH_EXPANSION_ENABLED=True, DR_GRAPH_EXPANSION_QUERIES=2,
                  DR_DECISION_GATE_ENABLED=False, DR_REFLECTION_ENABLED=True,
                  DR_REFLECTION_MAX_ITERATIONS=2, DR_REFLECTION_RESEARCH_BUDGET=5,
                  DR_SURVEY_MODE=False, DR_CITATIONS_ENABLED=False,
                  _expansion_pass=expansion):
        out = _run()
        check("reflection_loop_ran", "Synthesized Report" in out["report"])


def test_empty_synthesis_retry_ladder_and_digest_fallback():
    with _Patches(DR_REPLAN_ENABLED=False, DR_MULTIHOP_ENABLED=False, DR_CONTRADICTION_ENABLED=False,
                  DR_HIERARCHICAL_ENABLED=False, DR_COMMUNITIES_ENABLED=False, DR_CLAIM_MERGE_ENABLED=False,
                  DR_GRAPH_EXPANSION_ENABLED=False, DR_DECISION_GATE_ENABLED=False, DR_REFLECTION_ENABLED=False,
                  DR_SURVEY_MODE=False, DR_CITATIONS_ENABLED=False,
                  synthesize_report=lambda ctx, topic, briefs, prog, **kw: ""):
        out = _run()
        check("empty_synthesis_falls_back_to_digest", len(out["report"]) > 0)


def test_hierarchy_deep_tree_preamble():
    many_briefs = BRIEFS + [
        _brief(f"site{i}.com", f"https://site{i}.com/{i}", f"Title {i}",
               f"unique content about topic branch {i} " * 5) for i in range(8)]
    with _Patches(DR_REPLAN_ENABLED=False, DR_MULTIHOP_ENABLED=False, DR_CONTRADICTION_ENABLED=False,
                  DR_HIERARCHICAL_ENABLED=True, DR_HIERARCHY_MAX_DEPTH=4, DR_HIERARCHY_MIN_CLUSTER_SIZE=1,
                  DR_HIERARCHY_SPLIT_THRESHOLD=0.1,
                  DR_CLUSTER_THRESHOLD=0.3, DR_CLUSTER_MAJOR_MIN=1,
                  DR_COMMUNITIES_ENABLED=False, DR_CLAIM_MERGE_ENABLED=False,
                  DR_GRAPH_EXPANSION_ENABLED=False, DR_DECISION_GATE_ENABLED=False, DR_REFLECTION_ENABLED=False,
                  DR_SURVEY_MODE=False, DR_CITATIONS_ENABLED=False,
                  _brief_pages=lambda *a, **k: (list(many_briefs), 0)):
        out = _run()
        check("hierarchy_deep_tree_ran", "Synthesized Report" in out["report"])


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

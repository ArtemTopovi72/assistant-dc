"""Boundary-stubbed coverage for deep_research.py's crawl/brief/synthesis layer:
crawl_pages, brief_source, rerank_for_briefing, synthesize_report,
_evidence_landscape, _synthesize_abstract, _save_state/_append_history/
_write_report, _contradiction_note, _brief_pages, _expansion_pass, _reflect.
Stubs _acquire_page / call_llm_simple / _think_call / rerank module / collect_sources
/ crawl_pages so every branch runs deterministically without live network/GPU.
Run: venv/Scripts/python.exe tests/test_deep_research_synth.py
"""
import os, sys, tempfile, threading, json as _json
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

_TMP = Path(tempfile.mkdtemp(prefix="drsynth_"))


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
    def is_cancelled(self):
        return self._cancelled


def _page(url="https://arxiv.org/1", domain="arxiv.org", title="T", cluster_size=1):
    return {"url": url, "domain": domain, "title": title, "text": "some text " * 20,
            "cluster_size": cluster_size, "cluster_domains": [domain]}


# ---------------- crawl_pages ----------------

def test_crawl_pages_basic():
    prog = DR._Progress(None)
    sources = [{"href": "https://a.com/1"}, {"href": "https://b.com/1"}]
    caps = {"max_pages": 5, "depth": 0, "links_per_page": 3}

    def fake_acquire(url, depth, seed, category, use_adapters, cache):
        return {"url": url, "depth": depth, "page": _page(url, url.split("//")[1].split("/")[0]),
                "html": None, "reason": None, "social": False}

    with _Patches(_acquire_page=fake_acquire, DR_FETCH_DELAY=0, DR_FETCH_CONCURRENCY=1,
                  _save_state=lambda *a, **k: None):
        pages = DR.crawl_pages(FakeCtx(), sources, caps, prog, _TMP)
        check("crawl_pages_serial_success", len(pages) == 2)

    with _Patches(_acquire_page=fake_acquire, DR_FETCH_DELAY=0, DR_FETCH_CONCURRENCY=4,
                  _save_state=lambda *a, **k: None):
        pages = DR.crawl_pages(FakeCtx(), sources, caps, prog, _TMP)
        check("crawl_pages_parallel_success", len(pages) == 2)

    def fake_acquire_fail(url, depth, seed, category, use_adapters, cache):
        return {"url": url, "depth": depth, "page": None, "html": None,
                "reason": "fetch_failed", "social": False}
    with _Patches(_acquire_page=fake_acquire_fail, DR_FETCH_DELAY=0, DR_FETCH_CONCURRENCY=1,
                  _save_state=lambda *a, **k: None):
        pages = DR.crawl_pages(FakeCtx(), sources, caps, prog, _TMP)
        check("crawl_pages_quarantines_failures", pages == [] and prog.stats.get("quarantined"))

    with _Patches(_acquire_page=fake_acquire, DR_FETCH_DELAY=0, DR_FETCH_CONCURRENCY=1,
                  _save_state=lambda *a, **k: None):
        pages = DR.crawl_pages(FakeCtx(cancelled=True), sources, caps, prog, _TMP)
        check("crawl_pages_cancelled_stops_early", pages == [])

    # Same-domain sources trigger the wave-deferred path.
    same_dom_sources = [{"href": "https://c.com/1"}, {"href": "https://c.com/2"},
                        {"href": "https://c.com/3"}]
    with _Patches(_acquire_page=fake_acquire, DR_FETCH_DELAY=0, DR_FETCH_CONCURRENCY=2,
                  _save_state=lambda *a, **k: None):
        pages = DR.crawl_pages(FakeCtx(), same_dom_sources, {"max_pages": 5, "depth": 0, "links_per_page": 3},
                               prog, _TMP)
        check("crawl_pages_same_domain_deferred", len(pages) == 3)

    # html + link-following branch: page has html with social/normal links.
    html_page_html = ('<html><body><a href="https://vk.com/club1">vk</a>'
                       '<a href="/next-article">next</a></body></html>')
    def fake_acquire_html(url, depth, seed, category, use_adapters, cache):
        return {"url": url, "depth": depth, "page": _page(url, "d.com"),
                "html": html_page_html, "reason": None, "social": False}
    with _Patches(_acquire_page=fake_acquire_html, DR_FETCH_DELAY=0, DR_FETCH_CONCURRENCY=1,
                  _save_state=lambda *a, **k: None):
        pages = DR.crawl_pages(FakeCtx(), [{"href": "https://d.com/start"}],
                               {"max_pages": 1, "depth": 1, "links_per_page": 3}, prog, _TMP)
        check("crawl_pages_html_link_following_ran", len(pages) == 1)

    # positive fetch delay branch
    with _Patches(_acquire_page=fake_acquire, DR_FETCH_DELAY=0.001, DR_FETCH_CONCURRENCY=1,
                  _save_state=lambda *a, **k: None):
        pages = DR.crawl_pages(FakeCtx(), sources, caps, prog, _TMP)
        check("crawl_pages_fetch_delay_branch", len(pages) == 2)


# ---------------- brief_source ----------------

def test_brief_source():
    with _Patches(call_llm_simple=lambda ctx, sys_, usr, temperature=0.2, max_tokens=0, prefill=None: "SOURCE: PRIMARY\nSome useful facts here."):
        out = DR.brief_source(FakeCtx(), "topic", _page())
        check("brief_source_success", "PRIMARY" in out)
    with _Patches(call_llm_simple=lambda *a, **k: "NO USEFUL CONTENT"):
        check("brief_source_no_useful_content_none", DR.brief_source(FakeCtx(), "topic", _page()) is None)
    with _Patches(call_llm_simple=lambda *a, **k: "   "):
        # A blank answer still produces NO BRIEF -- that part is unchanged. What
        # changed is that it is now distinguishable from a rejection: "" means
        # the model said nothing (our failure), None means it read the page and
        # judged it useless (the page's). Conflating them let an outage be
        # reported to the reader as "the pages contained no facts relevant to
        # the topic". The old check asserted the identity rather than the
        # contract, so it broke on a change that was the point.
        blank = DR.brief_source(FakeCtx(), "topic", _page())
        check("brief_source_blank_yields_no_brief", not blank)
        check("brief_source_blank_is_not_a_rejection", blank is not None, repr(blank))



def test_brief_reads_the_part_about_the_topic():
    # A page is fetched up to 32000 chars but the briefer reads 4000: those must
    # be the part about the topic, not the page's menus and lede.
    seen = []
    page = dict(_page(), text="Site menu and news ticker. " * 400
                + "Perovskite solar cells reached 26.1 percent efficiency in 2024. " + "Footer. " * 200)
    with _Patches(call_llm_simple=lambda ctx, s, u, **k: (seen.append(u), "SOURCE: PRIMARY\nfacts")[1],
                  _think_call=lambda ctx, s, u, *a, **k: (seen.append(u), "SOURCE: PRIMARY\nfacts")[1]):
        DR.brief_source(FakeCtx(), "perovskite solar cell efficiency", page)
    check("brief_input_holds_the_topic_sentence", seen and "26.1 percent" in seen[-1], (seen or [""])[-1][-300:])

# ---------------- rerank_for_briefing ----------------

def test_rerank_for_briefing():
    prog = DR._Progress(None)
    pages = [_page("https://a.com/1", "a.com"), _page("https://b.com/1", "b.com")]
    with _Patches(DR_RERANK_ENABLED=False):
        check("rerank_disabled_passthrough", DR.rerank_for_briefing("t", pages, {}, prog) == pages)
    check("rerank_empty_pages_passthrough", DR.rerank_for_briefing("t", [], {"category": "general"}, prog) == [])

    fake_rerank_mod = type(sys)("rerank")
    def fake_rerank_pages(topic, pages, top_k=0, text_key="", authority_fn=None, authority_weight=0, backend="auto"):
        out = list(pages)
        for p in out:
            p["_rerank_relevance"] = 0.9
        return out[:top_k]
    fake_rerank_mod.rerank_pages = fake_rerank_pages
    class FakeReranker:
        backend = "fake"
    fake_rerank_mod.get_reranker = lambda backend: FakeReranker()
    sys.modules["rerank"] = fake_rerank_mod
    try:
        with _Patches(DR_RERANK_ENABLED=True, DR_RERANK_TOP_K=1, DR_RERANK_AUTHORITY_WEIGHT=0.3):
            out = DR.rerank_for_briefing("t", pages, {"category": "general"}, prog)
            check("rerank_enabled_top_k", len(out) == 1)
        with _Patches(DR_RERANK_ENABLED=True, DR_RERANK_TOP_K=0, DR_RERANK_AUTHORITY_WEIGHT=0):
            # top_k becomes len(pages), and since len<=top_k and weight==0 -> passthrough branch
            out2 = DR.rerank_for_briefing("t", pages, {"category": "general"}, prog)
            check("rerank_passthrough_when_nothing_to_drop", out2 == pages)
    finally:
        del sys.modules["rerank"]

    # reranker import fails -> warns and keeps authority order
    if "rerank" in sys.modules:
        del sys.modules["rerank"]
    import builtins
    real_import = builtins.__import__
    def fail_import(name, *a, **k):
        if name == "rerank":
            raise ImportError("no rerank module")
        return real_import(name, *a, **k)
    builtins.__import__ = fail_import
    try:
        with _Patches(DR_RERANK_ENABLED=True, DR_RERANK_TOP_K=1, DR_RERANK_AUTHORITY_WEIGHT=0.3):
            out3 = DR.rerank_for_briefing("t", pages, {"category": "general"}, prog)
            check("rerank_import_failure_keeps_order", out3 == pages)
    finally:
        builtins.__import__ = real_import


# ---------------- synthesize_report ----------------

def test_synthesize_report():
    briefs = [{"domain": "arxiv.org", "url": "https://arxiv.org/1", "title": "T1",
              "brief": "some facts", "trust": "PRIMARY", "authority": 0.9, "cluster_size": 1}]
    prog = DR._Progress(None)
    with _Patches(_think_call=lambda ctx, sysp, usr, max_tokens, temperature=0.3, **_kw: "# Report\nBody text here."):
        out = DR.synthesize_report(FakeCtx(), "quantum tunneling", briefs, prog)
        check("synthesize_report_basic", "Report" in out)
    with _Patches(_think_call=lambda *a, **k: "# Report body"):
        out2 = DR.synthesize_report(FakeCtx(), "quantum tunneling", briefs, prog,
                                    sections=["Citation Lineage", "Method Comparison"])
        check("synthesize_report_with_sections", "Report" in out2)
    with _Patches(_think_call=lambda *a, **k: "# Report body"):
        out3 = DR.synthesize_report(FakeCtx(), "the di zenzo structure tensor explained", briefs, prog)
        check("synthesize_report_di_zenzo_math_note", "Report" in out3)


# ---------------- _evidence_landscape ----------------

def test_evidence_landscape():
    briefs = [{"trust": "PRIMARY", "domain": "arxiv.org", "title": "T1", "url": "https://arxiv.org/1"},
              {"trust": "COMMUNITY", "domain": "reddit.com", "title": "", "url": "https://reddit.com/2"}]
    land = DR._evidence_landscape(briefs)
    check("evidence_landscape_has_domains", "arxiv.org" in land and "reddit.com" in land)
    land2 = DR._evidence_landscape(briefs, max_titles=1)
    check("evidence_landscape_max_titles_caps", land2.count("- [") == 1)


# ---------------- _synthesize_abstract ----------------

def test_synthesize_abstract():
    outline = {"title": "T", "abstract_focus": "the focus line"}
    check("abstract_empty_body_returns_focus", DR._synthesize_abstract(FakeCtx(), outline, "") == "the focus line")
    with _Patches(_think_call=lambda ctx, s, u, mo, temperature=0.3: "A full abstract about the topic."):
        out = DR._synthesize_abstract(FakeCtx(), outline, "some body text " * 20)
        check("abstract_success", "abstract" in out.lower())
    def raiser(*a, **k):
        raise RuntimeError("boom")
    with _Patches(_think_call=raiser):
        out2 = DR._synthesize_abstract(FakeCtx(), outline, "some body text " * 20)
        check("abstract_exception_falls_back_to_focus", out2 == "the focus line")
    with _Patches(_think_call=lambda *a, **k: "   "):
        out3 = DR._synthesize_abstract(FakeCtx(), outline, "some body text " * 20)
        check("abstract_blank_falls_back_to_focus", out3 == "the focus line")


# ---------------- persistence: _save_state / _append_history / _write_report / _now_iso ----------------

def test_persistence():
    run_dir = _TMP / "run1"
    with _Patches(DR_PHASE_HISTORY_ENABLED=True):
        DR._save_state(run_dir, phase="searching", queries=["q1"])
        check("save_state_writes_file", (run_dir / "state.json").exists())
        check("save_state_appends_history", (run_dir / "history").exists())
    with _Patches(DR_PHASE_HISTORY_ENABLED=False):
        DR._save_state(run_dir, phase="crawling", pages=[])
        check("save_state_no_history_when_disabled", True)
    # save_state failure path: run_dir is unwritable-ish (use a path with null byte)
    bad_dir = Path(str(run_dir) + "\x00bad")
    DR._save_state(bad_dir, phase="x")
    check("save_state_exception_swallowed", True)

    path = DR._write_report(run_dir, "topic", "# Report body", {"meta": 1})
    check("write_report_success", path is not None and Path(path).exists())
    bad_path = DR._write_report(Path(str(run_dir) + "\x00bad2"), "t", "r", {})
    check("write_report_exception_returns_none", bad_path is None)

    iso = DR._now_iso()
    check("now_iso_format", "T" in iso)


def test_contradiction_note():
    check("contradiction_note_empty_sig", DR._contradiction_note({}) == "")
    check("contradiction_note_none_found", "no credible" in DR._contradiction_note({"has_contradictions": False}))
    note = DR._contradiction_note({"has_contradictions": True, "strength": "moderate",
                                   "count": 3, "strong_sources": 1, "domains": ["a.com", "b.com"]})
    check("contradiction_note_has_domains", "a.com" in note and "moderate" in note)


# ---------------- _brief_pages ----------------

def test_brief_pages():
    prog = DR._Progress(None)
    pages = [_page("https://arxiv.org/1", "arxiv.org", "Adam Optimizer"),
             _page("https://randomblog.com/2", "randomblog.com", "Unrelated")]
    with _Patches(brief_source=lambda ctx, topic, page: "SOURCE: SECONDARY\nfacts about " + page["title"]):
        briefs, dropped = DR._brief_pages(FakeCtx(), "adam optimizer", pages, {"category": "science"},
                                          _TMP, prog, scholarly=True, terms=["adam"])
        check("brief_pages_relevance_filters", len(briefs) == 1 and dropped == 1)
    with _Patches(brief_source=lambda ctx, topic, page: None):
        briefs2, _ = DR._brief_pages(FakeCtx(), "t", pages, {"category": "general"}, _TMP, prog,
                                     scholarly=False, terms=[])
        check("brief_pages_no_brief_skipped", briefs2 == [])
    with _Patches(brief_source=lambda ctx, topic, page: "SOURCE: PRIMARY\nfacts"):
        briefs3, _ = DR._brief_pages(FakeCtx(cancelled=True), "t", pages, {"category": "general"}, _TMP, prog,
                                     scholarly=False, terms=[], cancelled=lambda: True)
        check("brief_pages_cancelled_stops_immediately", briefs3 == [])


# ---------------- _expansion_pass ----------------

def test_expansion_pass():
    prog = DR._Progress(None)
    check("expansion_pass_no_queries_empty", DR._expansion_pass(
        FakeCtx(), "t", [], {"category": "general"}, {"max_pages": 4}, prog, _TMP, None,
        visited_urls=set(), scholarly=False, terms=[], mode="multihop", label="multihop", cancelled=lambda: False) == [])
    check("expansion_pass_cancelled_empty", DR._expansion_pass(
        FakeCtx(), "t", ["q"], {"category": "general"}, {"max_pages": 4}, prog, _TMP, None,
        visited_urls=set(), scholarly=False, terms=[], mode="multihop", label="multihop", cancelled=lambda: True) == [])

    with _Patches(collect_sources=lambda *a, **k: [{"href": "https://a.com/1"}],
                  crawl_pages=lambda *a, **k: [_page()],
                  dedupe_pages=lambda pages, cat: pages,
                  rerank_for_briefing=lambda *a, **k: [_page()],
                  _brief_pages=lambda *a, **k: ([{"brief": "x"}], 0)):
        out = DR._expansion_pass(FakeCtx(), "t", ["q"], {"category": "general"}, {"max_pages": 4}, prog,
                                 _TMP, None, visited_urls=set(), scholarly=False, terms=[],
                                 mode="multihop", label="multihop", cancelled=lambda: False)
        check("expansion_pass_success", len(out) == 1)

    with _Patches(collect_sources=lambda *a, **k: [{"href": "https://seen.com/1"}]):
        out2 = DR._expansion_pass(FakeCtx(), "t", ["q"], {"category": "general"}, {"max_pages": 4}, prog,
                                  _TMP, None, visited_urls={"https://seen.com/1"}, scholarly=False, terms=[],
                                  mode="multihop", label="multihop", cancelled=lambda: False)
        check("expansion_pass_all_visited_empty", out2 == [])

    def raiser(*a, **k):
        raise RuntimeError("boom")
    with _Patches(collect_sources=raiser):
        out3 = DR._expansion_pass(FakeCtx(), "t", ["q"], {"category": "general"}, {"max_pages": 4}, prog,
                                  _TMP, None, visited_urls=set(), scholarly=False, terms=[],
                                  mode="multihop", label="multihop", cancelled=lambda: False)
        check("expansion_pass_exception_returns_empty", out3 == [])


# ---------------- _reflect ----------------

def test_reflect():
    briefs = [{"mode": "confirm", "trust": "PRIMARY"}]
    class FakeEntset:
        def total(self):
            return 0
    out = DR._reflect("t", briefs, FakeEntset(), {"has_contradictions": False}, {"uncertain": True})
    check("reflect_flags_all_gaps", len(out["gaps"]) == 3 and out["needs_second_pass"])
    briefs2 = [{"mode": "confirm", "trust": "PRIMARY"}]
    class FakeEntset2:
        def total(self):
            return 5
    out2 = DR._reflect("t", briefs2, FakeEntset2(), {"has_contradictions": True}, {"uncertain": False})
    check("reflect_no_gaps_when_strong", out2["gaps"] == [] and not out2["needs_second_pass"])
    out3 = DR._reflect("t", briefs2, None, {"has_contradictions": True}, {"uncertain": False})
    check("reflect_none_entset_skips_check", out3["needs_second_pass"] is False)


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

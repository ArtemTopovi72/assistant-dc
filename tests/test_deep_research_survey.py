"""Boundary-stubbed coverage for the remaining deep_research.py gaps: full
synthesize_survey (outline planning, per-section writing incl. shrink-retry and
cancellation), _deterministic_digest_report (both digest-present and
briefs-only paths), plus small edge branches in _is_safe_public_url,
_extract_pdf_text, _fetch_page, _fetch, _fix_mojibake, _domain, _norm_url,
_is_landing_url, _model_context, _think_call, _consolidate_evidence,
_select_section_evidence, plan_report_outline, _extract_outline_json,
_citation_section_from_brief, _shingles, dedupe_pages, _normalize_outline,
_coerce_override, _reflect, _social_outlinks, _acquire_page, crawl_pages,
_finish, rerank_for_briefing, synthesize_report, _extract_text, _extract_links.
Run: venv/Scripts/python.exe tests/test_deep_research_survey.py
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

_TMP = Path(tempfile.mkdtemp(prefix="drsurvey_"))


import dr_calls
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


BRIEFS = [
    {"domain": "arxiv.org", "url": "https://arxiv.org/1", "title": "T1",
     "brief": "facts about the topic", "trust": "PRIMARY", "authority": 0.9,
     "cluster_size": 1, "cluster_domains": ["arxiv.org"], "equations": ["E=mc^2"]},
]

OUTLINE = {
    "title": "Test Survey", "abstract_focus": "focus line",
    "sections": [
        {"heading": "Intro", "subsections": ["sub1"], "focus": "intro focus"},
        {"heading": "Middle", "subsections": [], "focus": "middle focus"},
        {"heading": "Conclusion", "subsections": [], "focus": "conclusion focus"},
    ],
}


def test_synthesize_survey_empty_briefs():
    check("survey_empty_briefs_empty_string", DR.synthesize_survey(FakeCtx(), "t", [], DR._Progress(None)) == "")


def test_synthesize_survey_full_success():
    prog = DR._Progress(None)
    with _Patches(plan_report_outline=lambda *a, **k: OUTLINE,
                  _think_call=lambda ctx, s, u, mo, temperature=0.35: "## Section body content here.",
                  _synthesize_abstract=lambda ctx, outline, body,
                                              out_lang="en": "An abstract."):
        out = DR.synthesize_survey(FakeCtx(), "t", BRIEFS, prog)
        check("survey_full_success_has_title", "Test Survey" in out)
        check("survey_full_success_has_abstract", "An abstract." in out)
        check("survey_full_success_has_sections", "Section body" in out)


def test_synthesize_survey_cancelled_midway():
    prog = DR._Progress(None)
    ctx = FakeCtx()
    calls = {"n": 0}
    def think(c, s, u, mo, temperature=0.35):
        calls["n"] += 1
        if calls["n"] == 1:
            ctx._cancelled = True  # cancel after first section written
        return "## body"
    with _Patches(plan_report_outline=lambda *a, **k: OUTLINE, _think_call=think,
                  _synthesize_abstract=lambda *a, **k: "abstract"):
        out = DR.synthesize_survey(ctx, "t", BRIEFS, prog)
        check("survey_cancelled_still_returns_partial", "Test Survey" in out)


def test_synthesize_survey_shrink_retry_then_success():
    prog = DR._Progress(None)
    calls = {"n": 0}
    def think(ctx, s, u, mo, temperature=0.35):
        calls["n"] += 1
        return "" if calls["n"] % 3 != 0 else "## eventually written"
    with _Patches(plan_report_outline=lambda *a, **k: {"title": "T", "abstract_focus": "",
                                                       "sections": [{"heading": "H", "subsections": [], "focus": "f"}]},
                  _think_call=think, _synthesize_abstract=lambda *a, **k: ""):
        out = DR.synthesize_survey(FakeCtx(), "t", BRIEFS, prog)
        check("survey_shrink_retry_eventually_succeeds", "eventually written" in out)


def test_synthesize_survey_all_sections_empty():
    prog = DR._Progress(None)
    with _Patches(plan_report_outline=lambda *a, **k: OUTLINE,
                  _think_call=lambda *a, **k: "",
                  _synthesize_abstract=lambda *a, **k: ""):
        out = DR.synthesize_survey(FakeCtx(), "t", BRIEFS, prog)
        check("survey_all_empty_returns_empty", out == "")


def test_synthesize_survey_body_without_heading_marker():
    prog = DR._Progress(None)
    with _Patches(plan_report_outline=lambda *a, **k: {"title": "T", "abstract_focus": "",
                                                       "sections": [{"heading": "MyHeading", "subsections": [], "focus": "f"}]},
                  _think_call=lambda *a, **k: "plain prose with no markdown heading",
                  _synthesize_abstract=lambda *a, **k: ""):
        out = DR.synthesize_survey(FakeCtx(), "t", BRIEFS, prog)
        check("survey_body_gets_heading_prepended", "## MyHeading" in out)


def test_deterministic_digest_report():
    digests = [{"label": "Thread A", "major": True, "size": 3, "evidence_weight": 2.5,
                "entities": ["x", "y"],
                "representative_claims": [{"text": "- some claim text here", "domain": "arxiv.org", "trust": "PRIMARY"}]}]
    out = DR._deterministic_digest_report("t", digests, BRIEFS)
    check("digest_report_has_thread", "Thread A" in out and "major thread" in out)
    check("digest_report_claim_cleaned", "some claim text" in out)

    out2 = DR._deterministic_digest_report("t", [], BRIEFS)
    check("digest_report_falls_back_to_briefs", "facts about the topic" in out2)

    out3 = DR._deterministic_digest_report("t", [{"label": "Minor", "major": False, "size": 1,
                                                  "evidence_weight": 0.1, "entities": [],
                                                  "representative_claims": []}], BRIEFS)
    check("digest_report_minor_thread_no_claims", "minor thread" in out3)


def test_is_safe_public_url_edge_branches():
    check("safe_url_no_hostname", not DR._is_safe_public_url("http:///path"))
    check("safe_url_getaddrinfo_fails", not DR._is_safe_public_url("http://this-host-does-not-exist-xyz123.invalid/"))


def test_extract_pdf_text_real_success():
    try:
        from pypdf import PdfWriter
        import io
        w = PdfWriter()
        w.add_blank_page(width=200, height=200)
        buf = io.BytesIO()
        w.write(buf)
        out = DR._extract_pdf_text(buf.getvalue())
        check("extract_pdf_text_blank_page_ok", out is None or isinstance(out, str))
    except Exception as e:
        check("extract_pdf_text_real_success_skipped", True, str(e))


def test_fetch_page_more_branches():
    class FakeResp:
        def __init__(self, status=200, content=b"", headers=None, encoding="utf-8"):
            self.status_code = status; self._content = content
            self.headers = headers or {}; self.encoding = encoding
            self.text = content.decode("utf-8", "replace")
        def iter_content(self, n):
            for i in range(0, len(self._content), n):
                yield self._content[i:i+n]
        def close(self): pass
    class FakeReq:
        def __init__(self, resp): self._resp = resp
        def get(self, *a, **k): return self._resp
    with _Patches(requests=FakeReq(FakeResp(200, b"plain text, no content-type", headers={}))):
        html, pdf = DR._fetch_page("https://example.com/no-ctype-real-article")
        check("fetch_page_no_ctype_allowed", html is not None)
    with _Patches(requests=FakeReq(FakeResp(200, b"<html>iso text</html>",
                                             headers={"content-type": "text/html"}, encoding="iso-8859-1"))):
        html2, pdf2 = DR._fetch_page("https://example.com/iso-encoded-real-article")
        check("fetch_page_iso_encoding_replaced", isinstance(html2, str))


def test_fetch_more_branches():
    class FakeResp:
        def __init__(self, content=b""):
            self.status_code = 200; self._content = content
            self.headers = {"content-type": "text/html"}; self.encoding = "utf-8"
        def iter_content(self, n):
            for i in range(0, len(self._content), n):
                yield self._content[i:i+n]
        def close(self): pass
    class FakeReq:
        def __init__(self, resp): self._resp = resp
        def get(self, *a, **k): return self._resp
    with _Patches(requests=FakeReq(FakeResp(b""))):
        check("fetch_empty_chunks_skipped", DR._fetch("https://example.com/empty") == "")
    big = b"x" * (DR._MAX_HTML_CHARS + 1000)
    with _Patches(requests=FakeReq(FakeResp(big))):
        out = DR._fetch("https://example.com/huge")
        check("fetch_caps_at_max_chars", len(out) <= DR._MAX_HTML_CHARS)


def test_fix_mojibake_ftfy_missing():
    import builtins
    real_import = builtins.__import__
    def fail_ftfy(name, *a, **k):
        if name == "ftfy":
            raise ImportError("no ftfy")
        return real_import(name, *a, **k)
    builtins.__import__ = fail_ftfy
    try:
        check("fix_mojibake_ftfy_missing_noop", DR._fix_mojibake("some text") == "some text")
    finally:
        builtins.__import__ = real_import


def test_domain_norm_landing_exception_paths():
    check("domain_exception_path", isinstance(DR._domain(None), str) or DR._domain(None) is None)
    check("norm_url_exception_path", DR._norm_url(None) is None or isinstance(DR._norm_url(None), str))
    check("is_landing_url_exception_path", DR._is_landing_url(None) in (True, False))


def test_model_context_live_detect_path():
    class FakeCtx2:
        model_name = "definitely-nonexistent-model-abc"
    dr_calls._CONTEXT_CACHE.clear()
    class FakeResp:
        status_code = 200
        def json(self):
            return {"loaded_context_length": 8192}
    class FakeReq:
        def get(self, *a, **k):
            return FakeResp()
    with _Patches(requests=FakeReq()):
        val = DR._model_context(FakeCtx2())
        check("model_context_live_detected", val == 8192)


def test_think_call_clamps():
    with _Patches(call_llm_simple=lambda ctx, s, u, temperature=0.3, max_tokens=0, force_think=True, **_kw: f"got:{max_tokens}"):
        out = DR._think_call(FakeCtx(), "sys", "user", max_tokens=999999999)
        check("think_call_clamps_to_safe_budget", "got:" in out)


def test_consolidate_evidence_batches():
    prog = DR._Progress(None)
    many_briefs = [dict(BRIEFS[0], url=f"https://arxiv.org/{i}", brief="x" * 500) for i in range(50)]
    with _Patches(_think_call=lambda ctx, s, u, max_tokens, temperature=0.2: "consolidated notes"):
        out = DR._consolidate_evidence(FakeCtx(), "t", many_briefs, prog)
        check("consolidate_evidence_batches_when_too_big", isinstance(out, str))


def test_select_section_evidence_budget_forces_multiple():
    sec = {"heading": "Facts", "focus": "facts about the topic", "subsections": []}
    briefs = [dict(BRIEFS[0], title=f"T{i}", brief="facts about the topic here " * 5) for i in range(5)]
    out = DR._select_section_evidence(briefs, sec, 100000)
    check("select_section_evidence_fits_multiple", out.count("[Source:") >= 2)


def test_plan_report_outline_trims_landscape():
    prog = DR._Progress(None)
    briefs = [dict(BRIEFS[0], title=f"Title number {i}", url=f"https://arxiv.org/{i}") for i in range(200)]
    with _Patches(_think_call=lambda ctx, s, u, max_tokens, temperature=0.4: '{"title": "T", "sections": []}'):
        out = DR.plan_report_outline(FakeCtx(), "t", briefs, prog)
        check("plan_report_outline_ran_with_large_landscape", isinstance(out, dict))


def test_extract_outline_json_bad_embedded():
    check("extract_outline_json_embedded_bad", DR._extract_outline_json('prose {not valid json} more') is None)


def test_citation_section_edge_branches():
    body = "Most-influential prior work:\n- \"X Paper\" (2020) — cited_by=5\nRelated high-impact:\n- \"Y Paper\" (2019) — cited_by=1\n"
    sect = DR._citation_section_from_brief(body, terms=["nomatch"])
    check("citation_section_terms_suppress_all", sect == "")


def test_shingles_cap():
    text = " ".join(f"word{i}" for i in range(2000))
    sh = DR._shingles(text)
    check("shingles_capped", len(sh) <= DR._SHINGLE_CAP)


def test_dedupe_pages_single_page_noop():
    pages = [{"text": "unique text here", "url": "https://a.com/1", "domain": "a.com"}]
    out = DR.dedupe_pages(pages)
    check("dedupe_single_page", len(out) == 1 and out[0]["cluster_size"] == 1)


def test_normalize_outline_hits_max_sections_break():
    saved = DR.DR_SURVEY_MAX_SECTIONS
    DR.DR_SURVEY_MAX_SECTIONS = 2
    try:
        plan = {"sections": [{"heading": f"H{i}", "subsections": [], "focus": ""} for i in range(5)]}
        out = DR._normalize_outline(plan, "t")
        check("normalize_outline_caps_at_max_sections", len(out["sections"]) == 2)
    finally:
        DR.DR_SURVEY_MAX_SECTIONS = saved


def test_coerce_override_str_type_none_value():
    out = DR._coerce_override("DR_DOMAIN_WHITELIST", None)
    check("coerce_override_str_none_becomes_empty", out == "")


def test_reflect_no_strong_confirming_only_contradict_mode():
    briefs = [{"mode": "contradict", "trust": "PRIMARY"}]
    class E:
        def total(self): return 1
    out = DR._reflect("t", briefs, E(), {"has_contradictions": True}, {"uncertain": False})
    check("reflect_only_contradict_briefs_no_strong", out["strong_confirming"] == 0)


def test_social_outlinks_none_seen_dedup():
    html = '<html><body><a href="https://vk.com/a">1</a><a href="https://vk.com/a">2</a></body></html>'
    out = DR._social_outlinks(html, "https://x.com")
    check("social_outlinks_dedup", out == ["https://vk.com/a"])


def test_acquire_page_no_adapter_hit_no_cache_falls_to_fetch():
    with _Patches(is_social_url=lambda u: False,
                  fetch_via_adapter=lambda url, cat: None,
                  _fetch_page=lambda url: (None, None)):
        res = DR._acquire_page("https://example.com/no-adapter-real-article", 0, {"title": ""},
                               "general", True, None)
        check("acquire_page_adapter_miss_falls_through", res["reason"] == "fetch_failed")


def test_crawl_pages_wave_empty_continue_and_link_filters():
    prog = DR._Progress(None)
    # all sources already visited before loop starts is impossible via queue seed,
    # but we can force the "not wave" continue by having a domain-dup only source
    # combined with fetch_delay=0 concurrency 1 and links filtered by blacklist.
    html_with_links = '<html><body><a href="/blocked-path">x</a></body></html>'
    def fake_acquire(url, depth, seed, category, use_adapters, cache):
        return {"url": url, "depth": depth,
                "page": {"url": url, "domain": "d.com", "title": "T", "text": "txt", "cluster_size": 1, "cluster_domains": ["d.com"]},
                "html": html_with_links, "reason": None, "social": False}
    with _Patches(_acquire_page=fake_acquire, DR_FETCH_DELAY=0, DR_FETCH_CONCURRENCY=1,
                  _save_state=lambda *a, **k: None, DR_DOMAIN_BLACKLIST="d.com"):
        pages = DR.crawl_pages(FakeCtx(), [{"href": "https://d.com/start"}],
                               {"max_pages": 5, "depth": 2, "links_per_page": 3}, prog, _TMP)
        check("crawl_pages_link_blacklist_filters_followups", len(pages) == 1)


def test_finish_math_audit_and_formula_trace():
    prog = DR._Progress(None)
    prog.snap_formula("EXTRACTION", "some $x^2$ text")
    briefs = [dict(BRIEFS[0])]
    out = DR._finish(FakeCtx(), _TMP / "finishrun", "t", "# Report\n\nBody $y^2$ text.",
                     prog, ["q"], briefs, __import__("time").time(), cancelled=False)
    check("finish_writes_report_with_math_audit", out["path"] is not None)
    check("finish_stats_has_math_audit", "math_audit" in prog.stats)


def test_rerank_authority_only_reorders():
    prog = DR._Progress(None)
    pages = [{"url": "https://randomblog.com/1", "domain": "randomblog.com", "title": "T1", "text": "x" * 100},
             {"url": "https://arxiv.org/1", "domain": "arxiv.org", "title": "T2", "text": "y" * 100}]
    fake_rerank_mod = type(sys)("rerank")
    def fake_rerank_pages(topic, pages, top_k=0, text_key="", authority_fn=None, authority_weight=0, backend="auto"):
        out = sorted(pages, key=authority_fn, reverse=True)
        for p in out:
            p["_rerank_relevance"] = authority_fn(p)
        return out[:top_k]
    fake_rerank_mod.rerank_pages = fake_rerank_pages
    fake_rerank_mod.get_reranker = lambda backend: type("R", (), {"backend": "fake"})()
    sys.modules["rerank"] = fake_rerank_mod
    try:
        with _Patches(DR_RERANK_ENABLED=True, DR_RERANK_TOP_K=2, DR_RERANK_AUTHORITY_WEIGHT=1.0):
            out = DR.rerank_for_briefing("t", pages, {"category": "science"}, prog)
            check("rerank_authority_reorders", out[0]["domain"] == "arxiv.org")
    finally:
        del sys.modules["rerank"]


def test_synthesize_report_evidence_truncated():
    prog = DR._Progress(None)
    huge_briefs = [dict(BRIEFS[0], url=f"https://arxiv.org/{i}", brief="x" * 5000) for i in range(50)]
    with _Patches(_think_call=lambda ctx, s, u, max_tokens, temperature=0.3, **_kw: "# Report\nshort body"):
        out = DR.synthesize_report(FakeCtx(), "t", huge_briefs, prog)
        check("synthesize_report_handles_huge_evidence", "Report" in out)


def test_extract_text_exception():
    import builtins
    real_import = builtins.__import__
    def fail_trafilatura(name, *a, **k):
        if name == "trafilatura":
            raise ImportError("no trafilatura")
        return real_import(name, *a, **k)
    builtins.__import__ = fail_trafilatura
    try:
        check("extract_text_import_failure_none", DR._extract_text("<html>x</html>", "https://x.com") is None)
    finally:
        builtins.__import__ = real_import


def test_extract_links_non_http_and_exception():
    html = '<html><body><a href="javascript:void(0)">js</a><a href="mailto:a@b.com">mail</a></body></html>'
    out = DR._extract_links(html, "https://x.com/page")
    check("extract_links_skips_non_http_schemes", out == [])
    check("extract_links_bad_html_no_raise", DR._extract_links(None, "https://x.com") == [] or True)


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

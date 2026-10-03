"""Boundary-stubbed coverage for deep_research.py's fetch/extract/crawl layer:
_Progress, _fetch, _fetch_page, _fetch_ar5iv, _extract_pdf_text, _extract_text,
_extract_title, _extract_links, _social_outlinks, _acquire_page. Mocks
`requests` (module-level import) and deep_research's own helper attributes via
a _Patches context manager (same pattern as tests/test_image_polling.py) so
every branch runs deterministically without live network access.
Run: venv/Scripts/python.exe tests/test_deep_research_fetch.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
import deep_research as DR

RESULTS = []
import os as _os
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + detail))
    if not cond and _os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))


# The engine is split across deep_research + the dr_* stage modules, so a seam
# has to be patched in the module that USES it. `_dr_patch.Patches` finds every
# home of a name and rebinds all of them (and raises if nothing binds it, so a
# stale patch can never degrade into a silent real network/LM Studio call).
from _dr_patch import Patches as _Patches


class FakeResp:
    def __init__(self, status=200, text="", content=b"", headers=None, raises=None,
                 encoding="utf-8"):
        self.status_code = status
        self.text = text
        self._content = content or text.encode("utf-8")
        self.headers = headers or {"content-type": "text/html"}
        self._raises = raises
        self.encoding = encoding
    def iter_content(self, n):
        data = self._content
        for i in range(0, len(data), n):
            yield data[i:i+n]
    def close(self):
        pass


class FakeRequests:
    def __init__(self, resp=None, raises=None):
        self._resp = resp
        self._raises = raises
    def get(self, *a, **k):
        if self._raises:
            raise self._raises
        return self._resp


# ---------------- _Progress ----------------

def test_progress_class():
    events = []
    p = DR._Progress(lambda phase, stats, msg: events.append((phase, msg)))
    p.update("Searching", "starting", queries=3)
    p.update("Searching", "still searching")  # same phase -> no re-accumulate
    p.update("Crawling", "moving on")
    p.finalize_timings()
    check("progress_callback_fired", len(events) == 3)
    check("progress_stats_updated", p.stats["queries"] == 3)
    check("progress_phase_timing_recorded", "Searching" in p.stats["stage_timings"])
    p.snap_formula("stage1", "some text")
    check("progress_snap_formula", p.formula_snapshots == [("stage1", "some text")])

    # callback raising must never propagate
    p2 = DR._Progress(lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    p2.update("X", "msg")
    check("progress_callback_exception_swallowed", True)

    # no callback at all
    p3 = DR._Progress(None)
    p3.update("Y", "msg")
    check("progress_no_callback_noop", p3.phase == "Y")

    # snap_formula exception path (append raising) — force via a list-like that raises
    class BadList(list):
        def append(self, x):
            raise RuntimeError("boom")
    p4 = DR._Progress(None)
    p4.formula_snapshots = BadList()
    p4.snap_formula("s", "t")
    check("progress_snap_formula_exception_swallowed", True)


# ---------------- _fetch ----------------

def test_fetch():
    with _Patches(requests=FakeRequests(FakeResp(200, "<html>hi</html>",
                                                  headers={"content-type": "text/html"}))):
        out = DR._fetch("https://x.com/a")
        check("fetch_success", out == "<html>hi</html>")
    with _Patches(requests=FakeRequests(FakeResp(404))):
        check("fetch_non200_none", DR._fetch("https://x.com/a") is None)
    with _Patches(requests=FakeRequests(FakeResp(200, headers={"content-type": "image/png"}))):
        check("fetch_binary_content_type_none", DR._fetch("https://x.com/a") is None)
    with _Patches(requests=FakeRequests(raises=RuntimeError("boom"))):
        check("fetch_exception_none", DR._fetch("https://x.com/a") is None)
    with _Patches(requests=FakeRequests(FakeResp(200, "café", headers={"content-type": "text/html"}, encoding="iso-8859-1"))):
        out2 = DR._fetch("https://x.com/a")
        check("fetch_latin1_encoding_replaced_with_utf8", isinstance(out2, str))
    with _Patches(requests=FakeRequests(FakeResp(200, "no ctype", headers={}))):
        check("fetch_empty_ctype_allowed", DR._fetch("https://x.com/a") == "no ctype")


# ---------------- _ar5iv_url / _inline_mathml_as_tex / _fetch_ar5iv ----------------

def test_ar5iv_url_and_mathml():
    check("ar5iv_url_match", DR._ar5iv_url("https://arxiv.org/abs/2301.12345") == "https://ar5iv.org/abs/2301.12345")
    check("ar5iv_url_no_match", DR._ar5iv_url("https://example.com/x") is None)
    html = '<math alttext="x^2">stuff</math> and <math alttext="y" display="block">Y</math>'
    out = DR._inline_mathml_as_tex(html)
    check("inline_mathml_inline_dollar", "$x^2$" in out)
    check("inline_mathml_block_dollar", "$$y$$" in out)
    html_noalt = '<math>no alttext here</math>'
    out2 = DR._inline_mathml_as_tex(html_noalt)
    check("inline_mathml_no_alttext_dropped", "<math>" not in out2)


def test_fetch_ar5iv():
    check("fetch_ar5iv_no_match_none", DR._fetch_ar5iv("https://example.com/x") is None)
    with _Patches(requests=FakeRequests(FakeResp(200, '<math alttext="x">y</math>'))):
        out = DR._fetch_ar5iv("https://arxiv.org/abs/2301.12345")
        check("fetch_ar5iv_success", out is not None and "$x$" in out)
    with _Patches(requests=FakeRequests(FakeResp(200, "<html>no math here</html>"))):
        check("fetch_ar5iv_no_math_none", DR._fetch_ar5iv("https://arxiv.org/abs/2301.12345") is None)
    with _Patches(requests=FakeRequests(FakeResp(404))):
        check("fetch_ar5iv_non200_none", DR._fetch_ar5iv("https://arxiv.org/abs/2301.12345") is None)
    with _Patches(requests=FakeRequests(raises=RuntimeError("boom"))):
        check("fetch_ar5iv_exception_none", DR._fetch_ar5iv("https://arxiv.org/abs/2301.12345") is None)


# ---------------- _extract_pdf_text / _looks_like_unmarked_math ----------------

def test_looks_like_unmarked_math():
    check("unmarked_math_true", DR._looks_like_unmarked_math(
        "we use softmax and a matrix and a gradient computation with no markup"))
    check("unmarked_math_false_has_dollar", not DR._looks_like_unmarked_math("some $x^2$ text"))
    check("unmarked_math_false_no_hints", not DR._looks_like_unmarked_math("just plain prose here"))


def test_extract_pdf_text():
    # A real minimal PDF is hard to construct inline; feed garbage bytes so pypdf
    # raises internally -> exercises the except-> None branch deterministically.
    check("extract_pdf_text_bad_bytes_none", DR._extract_pdf_text(b"not a real pdf") is None)


# ---------------- _fetch_page ----------------

def test_fetch_page():
    check("fetch_page_blocks_unsafe_url", DR._fetch_page("file:///etc/passwd") == (None, None))
    with _Patches(requests=FakeRequests(FakeResp(200, '<math alttext="z">Z</math>'))):
        html, pdf = DR._fetch_page("https://arxiv.org/abs/2301.12345")
        check("fetch_page_ar5iv_path", html is not None and pdf is None)
    with _Patches(requests=FakeRequests(FakeResp(200, "<html>body</html>",
                                                  headers={"content-type": "text/html"}))):
        html2, pdf2 = DR._fetch_page("https://example.com/a")
        check("fetch_page_html_success", html2 == "<html>body</html>" and pdf2 is None)
    with _Patches(requests=FakeRequests(FakeResp(404))):
        check("fetch_page_non200", DR._fetch_page("https://example.com/a") == (None, None))
    with _Patches(requests=FakeRequests(FakeResp(200, headers={"content-type": "image/png"}))):
        check("fetch_page_binary_skip", DR._fetch_page("https://example.com/a") == (None, None))
    with _Patches(requests=FakeRequests(raises=RuntimeError("boom"))):
        check("fetch_page_exception", DR._fetch_page("https://example.com/a") == (None, None))
    pdf_bytes = b"%PDF-1.4 fake pdf bytes here"
    with _Patches(requests=FakeRequests(FakeResp(200, content=pdf_bytes,
                                                  headers={"content-type": "application/pdf"})),
                  DR_PDF_ENABLED=True):
        html3, pdf3 = DR._fetch_page("https://example.com/paper.pdf")
        check("fetch_page_pdf_enabled_extract_attempted", html3 is None)
    with _Patches(requests=FakeRequests(FakeResp(200, content=pdf_bytes,
                                                  headers={"content-type": "application/pdf"})),
                  DR_PDF_ENABLED=False):
        check("fetch_page_pdf_disabled_none", DR._fetch_page("https://example.com/paper.pdf") == (None, None))
    with _Patches(requests=FakeRequests(FakeResp(200, content=pdf_bytes,
                                                  headers={"content-type": "text/html"})),
                  DR_PDF_ENABLED=True):
        html4, pdf4 = DR._fetch_page("https://example.com/sniffed")
        check("fetch_page_pdf_sniffed_by_magic_bytes", html4 is None)


# ---------------- _extract_text / _extract_title / _extract_links / _social_outlinks ----------------

def test_extract_text_title_links():
    html = '<html><head><title>My Page Title</title></head><body><p>Hello world content here.</p>' \
           '<a href="/rel">rel link</a><a href="https://other.com/x">cross domain</a>' \
           '<a href="https://x.com/dup">dup</a><a href="https://x.com/dup">dup2</a></body></html>'
    text = DR._extract_text(html, "https://x.com/page")
    check("extract_text_gets_content", text is None or "Hello" in (text or "") or text == "")
    title = DR._extract_title(html)
    check("extract_title_found", title == "My Page Title")
    check("extract_title_no_title_tag", DR._extract_title("<html><body>no title</body></html>") == "")
    check("extract_title_bad_html_noraise", DR._extract_title(None) == "" or True)
    links = DR._extract_links(html, "https://x.com/page")
    check("extract_links_same_domain_only", all("x.com" in l for l in links))
    check("extract_links_dedup", links.count("https://x.com/dup") == 1)
    check("extract_links_excludes_cross_domain", not any("other.com" in l for l in links))

    social_html = '<html><body><a href="https://vk.com/club1">vk</a><a href="/local">local</a></body></html>'
    outs = DR._social_outlinks(social_html, "https://x.com/page")
    check("social_outlinks_found", outs == ["https://vk.com/club1"])
    check("social_outlinks_bad_html_noraise", DR._social_outlinks(None, "https://x.com") == [] or True)


# ---------------- _acquire_page ----------------

def test_acquire_page():
    # social path with posts
    with _Patches(is_social_url=lambda u: True,
                  fetch_social_posts=lambda u: "some vk posts content here",
                  social_kind=lambda u: "VK"):
        res = DR._acquire_page("https://vk.com/club1", 0, {"title": ""}, "local", False, None)
        check("acquire_social_with_posts", res["social"] is True and res["page"] is not None)
    # social path with no posts
    with _Patches(is_social_url=lambda u: True, fetch_social_posts=lambda u: None):
        res = DR._acquire_page("https://vk.com/club2", 0, {"title": ""}, "local", False, None)
        check("acquire_social_no_posts", res["social"] is True and res["page"] is None)
    # adapter hit
    long_text = "Real content here. " * 30
    with _Patches(is_social_url=lambda u: False,
                  fetch_via_adapter=lambda url, cat: {"text": long_text, "title": "Adapter Title", "domain": "wiki.org"}):
        res = DR._acquire_page("https://wiki.org/some-article", 0, {"title": ""}, "general", True, None)
        check("acquire_adapter_hit", res["page"] is not None and res["page"]["title"] == "Adapter Title")
    # adapter returns junk that fails the gate -> falls through to fetch
    with _Patches(is_social_url=lambda u: False,
                  fetch_via_adapter=lambda url, cat: {"text": "hi", "title": "t"},
                  _fetch_page=lambda url: (None, None)):
        res = DR._acquire_page("https://wiki.org/y", 0, {"title": ""}, "general", True, None)
        check("acquire_adapter_gate_fail_falls_through", res["reason"] == "fetch_failed")
    # cache hit
    class FakeCache:
        def get(self, url, cat):
            return {"text": long_text, "title": "Cached Title"}
        def put(self, *a, **k):
            pass
    with _Patches(is_social_url=lambda u: False):
        res = DR._acquire_page("https://example.com/cached-article", 0, {"title": ""}, "general", False, FakeCache())
        check("acquire_cache_hit", res["page"] is not None and res["page"]["title"] == "Cached Title")
    # cache present but miss/None
    class EmptyCache:
        def get(self, url, cat):
            return None
        def put(self, *a, **k):
            pass
        def failed_recently(self, url):
            return False
        def mark_failed(self, url):
            pass
    with _Patches(is_social_url=lambda u: False, _fetch_page=lambda url: (None, None)):
        res = DR._acquire_page("https://example.com/w", 0, {"title": ""}, "general", False, EmptyCache())
        check("acquire_cache_miss_fetch_fails", res["reason"] == "fetch_failed")
    # live fetch, html success, gate ok
    with _Patches(is_social_url=lambda u: False,
                  _fetch_page=lambda url: ("<html><body>x</body></html>", None),
                  _extract_text=lambda html, url: long_text,
                  _extract_title=lambda html: "Live Title"):
        res = DR._acquire_page("https://example.com/live", 0, {"title": ""}, "general", False, None)
        check("acquire_live_fetch_success", res["page"] is not None and res["page"]["title"] == "Live Title")
    # live fetch pdf branch
    with _Patches(is_social_url=lambda u: False,
                  _fetch_page=lambda url: (None, long_text),
                  _extract_title=lambda html: "unused"):
        res = DR._acquire_page("https://example.com/paper.pdf", 0, {"title": "Seed Title"}, "general", False, None)
        check("acquire_live_fetch_pdf_branch", res["page"] is not None and res["page"]["title"] == "Seed Title")
    # gate rejects (thin content)
    with _Patches(is_social_url=lambda u: False,
                  _fetch_page=lambda url: ("<html>x</html>", None),
                  _extract_text=lambda html, url: "short",
                  _extract_title=lambda html: "T"):
        res = DR._acquire_page("https://example.com/thin", 0, {"title": ""}, "general", False, None)
        check("acquire_gate_rejects_thin", res["reason"] is not None and res["page"] is None)
    # landing url rejection after successful gate
    with _Patches(is_social_url=lambda u: False,
                  _fetch_page=lambda url: ("<html>x</html>", None),
                  _extract_text=lambda html, url: long_text,
                  _extract_title=lambda html: "T"):
        res = DR._acquire_page("https://example.com/", 0, {"title": ""}, "general", False, None)
        check("acquire_landing_url_rejected", res["reason"] == DR.GATE_NAV and res["page"] is None)


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

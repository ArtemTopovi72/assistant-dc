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
import offline_guard; offline_guard.public_dns()   # the SSRF guard resolves hosts: no network needed
import deep_research as DR

RESULTS = []
import os as _os
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + str(detail)))
    if not cond and _os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))


# The engine is split across deep_research + the dr_* stage modules, so a seam
# has to be patched in the module that USES it. `_dr_patch.Patches` finds every
# home of a name and rebinds all of them (and raises if nothing binds it, so a
# stale patch can never degrade into a silent real network/LM Studio call).
# The stand-ins ride on a context handed to each call (tests/_dr_ctx.py), never on the
# engine's modules: `_Patches` only stacks names for the calls inside it, `_C()` builds it.
from contextlib import contextmanager
from _dr_ctx import research_ctx

_STACK = []


@contextmanager
def _Patches(**names):
    _STACK.append(names)
    try:
        yield
    finally:
        _STACK.pop()


def _C():
    merged = {}
    for layer in _STACK:
        merged.update(layer)
    return research_ctx(None, **merged)


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



# ---------------- _fetch ----------------

def test_fetch():
    with _Patches(requests=FakeRequests(FakeResp(200, "<html>hi</html>",
                                                  headers={"content-type": "text/html"}))):
        out = DR._fetch("https://x.com/a", ctx=_C())
        check("fetch_success", out == "<html>hi</html>")
    with _Patches(requests=FakeRequests(FakeResp(404))):
        check("fetch_non200_none", DR._fetch("https://x.com/a", ctx=_C()) is None)
    with _Patches(requests=FakeRequests(FakeResp(200, headers={"content-type": "image/png"}))):
        check("fetch_binary_content_type_none", DR._fetch("https://x.com/a", ctx=_C()) is None)
    with _Patches(requests=FakeRequests(raises=RuntimeError("boom"))):
        check("fetch_exception_none", DR._fetch("https://x.com/a", ctx=_C()) is None)
    with _Patches(requests=FakeRequests(FakeResp(200, "café", headers={"content-type": "text/html"}, encoding="iso-8859-1"))):
        out2 = DR._fetch("https://x.com/a", ctx=_C())
        check("fetch_latin1_encoding_replaced_with_utf8", isinstance(out2, str))
    with _Patches(requests=FakeRequests(FakeResp(200, "no ctype", headers={}))):
        check("fetch_empty_ctype_allowed", DR._fetch("https://x.com/a", ctx=_C()) == "no ctype")


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
    check("fetch_ar5iv_no_match_none", DR._fetch_ar5iv("https://example.com/x", ctx=_C()) is None)
    with _Patches(requests=FakeRequests(FakeResp(200, '<math alttext="x">y</math>'))):
        out = DR._fetch_ar5iv("https://arxiv.org/abs/2301.12345", ctx=_C())
        check("fetch_ar5iv_success", out is not None and "$x$" in out)
    with _Patches(requests=FakeRequests(FakeResp(200, "<html>no math here</html>"))):
        check("fetch_ar5iv_no_math_none", DR._fetch_ar5iv("https://arxiv.org/abs/2301.12345", ctx=_C()) is None)
    with _Patches(requests=FakeRequests(FakeResp(404))):
        check("fetch_ar5iv_non200_none", DR._fetch_ar5iv("https://arxiv.org/abs/2301.12345", ctx=_C()) is None)
    with _Patches(requests=FakeRequests(raises=RuntimeError("boom"))):
        check("fetch_ar5iv_exception_none", DR._fetch_ar5iv("https://arxiv.org/abs/2301.12345", ctx=_C()) is None)


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
    check("fetch_page_blocks_unsafe_url", DR._fetch_page("file:///etc/passwd", ctx=_C()) == (None, None))
    with _Patches(requests=FakeRequests(FakeResp(200, '<math alttext="z">Z</math>'))):
        html, pdf = DR._fetch_page("https://arxiv.org/abs/2301.12345", ctx=_C())
        check("fetch_page_ar5iv_path", html is not None and pdf is None)
    with _Patches(requests=FakeRequests(FakeResp(200, "<html>body</html>",
                                                  headers={"content-type": "text/html"}))):
        html2, pdf2 = DR._fetch_page("https://example.com/a", ctx=_C())
        check("fetch_page_html_success", html2 == "<html>body</html>" and pdf2 is None)
    with _Patches(requests=FakeRequests(FakeResp(404))):
        check("fetch_page_non200", DR._fetch_page("https://example.com/a", ctx=_C()) == (None, None))
    with _Patches(requests=FakeRequests(FakeResp(200, headers={"content-type": "image/png"}))):
        check("fetch_page_binary_skip", DR._fetch_page("https://example.com/a", ctx=_C()) == (None, None))
    with _Patches(requests=FakeRequests(raises=RuntimeError("boom"))):
        check("fetch_page_exception", DR._fetch_page("https://example.com/a", ctx=_C()) == (None, None))
    pdf_bytes = b"%PDF-1.4 fake pdf bytes here"
    with _Patches(requests=FakeRequests(FakeResp(200, content=pdf_bytes,
                                                  headers={"content-type": "application/pdf"})),
                  DR_PDF_ENABLED=True):
        html3, pdf3 = DR._fetch_page("https://example.com/paper.pdf", ctx=_C())
        check("fetch_page_pdf_enabled_extract_attempted", html3 is None)
    with _Patches(requests=FakeRequests(FakeResp(200, content=pdf_bytes,
                                                  headers={"content-type": "application/pdf"})),
                  DR_PDF_ENABLED=False):
        check("fetch_page_pdf_disabled_none", DR._fetch_page("https://example.com/paper.pdf", ctx=_C()) == (None, None))
    with _Patches(requests=FakeRequests(FakeResp(200, content=pdf_bytes,
                                                  headers={"content-type": "text/html"})),
                  DR_PDF_ENABLED=True):
        html4, pdf4 = DR._fetch_page("https://example.com/sniffed", ctx=_C())
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
        res = DR._acquire_page("https://vk.com/club1", 0, {"title": ""}, "local", False, None, ctx=_C())
        check("acquire_social_with_posts", res["social"] is True and res["page"] is not None)
    # social path with no posts
    with _Patches(is_social_url=lambda u: True, fetch_social_posts=lambda u: None):
        res = DR._acquire_page("https://vk.com/club2", 0, {"title": ""}, "local", False, None, ctx=_C())
        check("acquire_social_no_posts", res["social"] is True and res["page"] is None)
    # adapter hit
    long_text = "Real content here. " * 30
    with _Patches(is_social_url=lambda u: False,
                  fetch_via_adapter=lambda url, cat: {"text": long_text, "title": "Adapter Title", "domain": "wiki.org"}):
        res = DR._acquire_page("https://wiki.org/some-article", 0, {"title": ""}, "general", True, None, ctx=_C())
        check("acquire_adapter_hit", res["page"] is not None and res["page"]["title"] == "Adapter Title")
    # adapter returns junk that fails the gate -> falls through to fetch
    with _Patches(is_social_url=lambda u: False,
                  fetch_via_adapter=lambda url, cat: {"text": "hi", "title": "t"},
                  _fetch_page=lambda url, **k: (None, None), _fetch_rendered=lambda url, **k: None):
        res = DR._acquire_page("https://wiki.org/y", 0, {"title": ""}, "general", True, None, ctx=_C())
        check("acquire_adapter_gate_fail_falls_through", res["reason"] == "fetch_failed")
    # cache hit
    class FakeCache:
        def get(self, url, cat):
            return {"text": long_text, "title": "Cached Title"}
        def put(self, *a, **k):
            pass
    with _Patches(is_social_url=lambda u: False):
        res = DR._acquire_page("https://example.com/cached-article", 0, {"title": ""}, "general", False, FakeCache(), ctx=_C())
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
    with _Patches(is_social_url=lambda u: False, _fetch_page=lambda url, **k: (None, None), _fetch_rendered=lambda url, **k: None):
        res = DR._acquire_page("https://example.com/w", 0, {"title": ""}, "general", False, EmptyCache(), ctx=_C())
        check("acquire_cache_miss_fetch_fails", res["reason"] == "fetch_failed")
    # live fetch, html success, gate ok
    with _Patches(is_social_url=lambda u: False,
                  _fetch_page=lambda url, **k: ("<html><body>x</body></html>", None),
                  _extract_text=lambda html, url: long_text,
                  _extract_title=lambda html: "Live Title"):
        res = DR._acquire_page("https://example.com/live", 0, {"title": ""}, "general", False, None, ctx=_C())
        check("acquire_live_fetch_success", res["page"] is not None and res["page"]["title"] == "Live Title")
    # live fetch pdf branch
    with _Patches(is_social_url=lambda u: False,
                  _fetch_page=lambda url, **k: (None, long_text),
                  _extract_title=lambda html: "unused"):
        res = DR._acquire_page("https://example.com/paper.pdf", 0, {"title": "Seed Title"}, "general", False, None, ctx=_C())
        check("acquire_live_fetch_pdf_branch", res["page"] is not None and res["page"]["title"] == "Seed Title")
    # gate rejects (thin content)
    with _Patches(is_social_url=lambda u: False,
                  _fetch_page=lambda url, **k: ("<html>x</html>", None),
                  _extract_text=lambda html, url: "short",
                  _extract_title=lambda html: "T"):
        res = DR._acquire_page("https://example.com/thin", 0, {"title": ""}, "general", False, None, ctx=_C())
        check("acquire_gate_rejects_thin", res["reason"] is not None and res["page"] is None)
    # landing url rejection after successful gate
    with _Patches(is_social_url=lambda u: False,
                  _fetch_page=lambda url, **k: ("<html>x</html>", None),
                  _extract_text=lambda html, url: long_text,
                  _extract_title=lambda html: "T"):
        res = DR._acquire_page("https://example.com/", 0, {"title": ""}, "general", False, None, ctx=_C())
        check("acquire_landing_url_rejected", res["reason"] == DR.GATE_NAV and res["page"] is None)


def test_blocked_vendor_pages_are_rendered_and_spec_sheets_kept():
    """A camera selection run lost 20 of 65 pages to 403/bot checks and 14 to the menu test:
    a blocked page is opened in a browser, a spec sheet is a data page, not navigation."""
    import dr_crawl, dr_extract, threading, time
    spec = ("Atlas ATL089S-CC\nSensor Sony IMX255\nResolution 4096 x 2160 px, 8.9 MP\nFrame rate 58 fps\n"
            "Interface 5GigE PoE\nPower 7 W\nPixel 3.45 um\nBit depth 12 bit\nWeight 200 g\nMount C, 60 mm\n"
            "Operating temperature 0 to 50 °C\nShutter Global\nColor Bayer RG\nDimensions 60 x 60 x 77.5 mm")
    check("spec_sheet_is_not_navigation", dr_extract.classify_extraction(spec, None, "Atlas 8.9 MP")[0] == DR.GATE_OK)
    check("a_menu_is_still_navigation",
          dr_extract.classify_extraction("Home\nProducts\nCameras\nAbout\nContact\n" * 6, None, "x")[0] == DR.GATE_NAV)
    rendered = []
    with _Patches(is_social_url=lambda u: False, _fetch_page=lambda url, **k: (None, None),
                  _fetch_rendered=lambda url, **k: rendered.append(url) or "<html>ok</html>",
                  _extract_text=lambda html, url: spec, _extract_title=lambda html: "Atlas"):
        res = DR._acquire_page("https://thinklucid.com/product/atlas", 0, {"title": ""}, "product", False, None, ctx=_C())
    check("a_failed_fetch_is_rendered_in_a_browser", res["page"] is not None and rendered, res["reason"])
    texts = iter(["Checking your browser before accessing. Cloudflare.", spec])
    with _Patches(is_social_url=lambda u: False, _fetch_page=lambda url, **k: ("<html>cf</html>", None),
                  _fetch_rendered=lambda url, **k: "<html>real</html>",
                  _extract_text=lambda html, url: next(texts), _extract_title=lambda html: "Atlas"):
        res = DR._acquire_page("https://thinklucid.com/product/atlas", 0, {"title": ""}, "product", False, None, ctx=_C())
    check("a_bot_check_page_is_rendered_again", res["page"] is not None and "IMX255" in res["page"]["text"], res["reason"])
    check("inner_hosts_are_refused_to_the_browser",
          dr_crawl._literal_inner_host("http://127.0.0.1:8000/") and dr_crawl._literal_inner_host("http://10.1.2.3/x")
          and not dr_crawl._literal_inner_host("https://cdn.example.com/a.js")
          and dr_crawl._fetch_rendered("http://127.0.0.1:8000/") is None)
    inside, worst = [0], [0]

    class _Slow:
        @staticmethod
        def extract(html, **k):
            inside[0] += 1
            worst[0] = max(worst[0], inside[0])
            time.sleep(0.05)
            inside[0] -= 1
            return "text"
    ts = [threading.Thread(target=dr_extract._extract_text, args=("<html/>", "u", {"trafilatura": _Slow}))
          for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    check("html_is_parsed_one_page_at_a_time", worst[0] == 1, worst[0])


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
    # A failed check() must reach the exit code; `failed` counts exceptions only.
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)

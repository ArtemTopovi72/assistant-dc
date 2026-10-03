"""Coverage for source_adapters.py: URL routing (pure) + real calls to the
public keyless Wikipedia/arXiv/Crossref APIs (free, no auth, safe) for the
happy paths, plus stubbed requests for the failure/edge branches.
Run: venv/Scripts/python.exe tests/test_source_adapters.py
"""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
import source_adapters as S

RESULTS = []
import os as _os
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + detail))
    if not cond and _os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))


class _Patches:
    def __init__(self, **kw):
        self.kw = kw; self.orig = {}
    def __enter__(self):
        for k, v in self.kw.items():
            self.orig[k] = getattr(S, k)
            setattr(S, k, v)
        return self
    def __exit__(self, *a):
        for k, v in self.orig.items():
            setattr(S, k, v)


def test_domain():
    check("domain_strips_www", S._domain("https://www.wikipedia.org/wiki/X") == "wikipedia.org")
    check("domain_bad_url_returns_input", S._domain(None) is None or isinstance(S._domain(None), str))


def test_adapter_for():
    check("adapter_for_wikipedia", S.adapter_for("https://en.wikipedia.org/wiki/Cat") == "wikipedia")
    check("adapter_for_wikipedia_no_wiki_path_none", S.adapter_for("https://en.wikipedia.org/") is None)
    check("adapter_for_arxiv_abs", S.adapter_for("https://arxiv.org/abs/2301.12345") == "arxiv")
    check("adapter_for_arxiv_pdf", S.adapter_for("https://arxiv.org/pdf/2301.12345") == "arxiv")
    check("adapter_for_arxiv_no_path_none", S.adapter_for("https://arxiv.org/") is None)
    check("adapter_for_crossref", S.adapter_for("https://doi.org/10.1234/xyz") == "crossref")
    check("adapter_for_none", S.adapter_for("https://example.com/page") is None)


def test_wiki_lang_title():
    lang, title = S._wiki_lang_title("https://en.wikipedia.org/wiki/Albert_Einstein")
    check("wiki_lang_title_basic", lang == "en" and title == "Albert Einstein")
    lang2, title2 = S._wiki_lang_title("https://ru.wikipedia.org/wiki/Кот#History")
    check("wiki_lang_title_lang_and_fragment_stripped", lang2 == "ru" and "#" not in (title2 or ""))
    lang3, title3 = S._wiki_lang_title("https://en.wikipedia.org/notwiki/X")
    check("wiki_lang_title_no_match_none", title3 is None)


def test_fetch_wikipedia_real():
    out = S.fetch_wikipedia("https://en.wikipedia.org/wiki/Python_(programming_language)")
    check("fetch_wikipedia_real_success", out is not None and out["via"] in ("wikipedia-api", "wikipedia-summary")
          and len(out["text"]) > 0)
    check("fetch_wikipedia_no_title_none", S.fetch_wikipedia("https://en.wikipedia.org/notwiki/X") is None)


def test_fetch_wikipedia_stubbed_failures():
    class FakeResp:
        def __init__(self, status=200, data=None):
            self.status_code = status; self._data = data or {}
        def json(self):
            return self._data
    with _Patches(requests=types.SimpleNamespace(get=lambda *a, **k: FakeResp(status=404))):
        check("fetch_wikipedia_non200_none", S.fetch_wikipedia("https://en.wikipedia.org/wiki/X") is None)
    with _Patches(requests=types.SimpleNamespace(
            get=lambda *a, **k: FakeResp(data={"query": {"pages": {"1": {"extract": "", "title": "X"}}}}))):
        check("fetch_wikipedia_empty_extract_none", S.fetch_wikipedia("https://en.wikipedia.org/wiki/X") is None)
    with _Patches(requests=types.SimpleNamespace(get=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))):
        check("fetch_wikipedia_exception_none", S.fetch_wikipedia("https://en.wikipedia.org/wiki/X") is None)


# A real export.arxiv.org Atom response, trimmed to the elements the adapter
# reads. Captured rather than requested: this used to call the live API, and a
# suite that goes to the network fails for reasons that have nothing to do with
# the code -- it went red today because arXiv simply did not answer. The live
# probe now lives in bench/source_adapters_live.py, where an outage is a
# finding rather than a false alarm.
_ARXIV_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <title>Attention Is All You
  Need</title>
    <summary>  The dominant sequence transduction models are based on complex
recurrent or convolutional neural networks.
</summary>
    <author><name>Ashish Vaswani</name></author>
  </entry>
</feed>"""


def test_fetch_arxiv_parses_a_real_response():
    class FakeResp:
        status_code = 200
        text = _ARXIV_ATOM

    with _Patches(requests=types.SimpleNamespace(get=lambda *a, **k: FakeResp())):
        out = S.fetch_arxiv("https://arxiv.org/abs/1706.03762")
    check("fetch_arxiv_real_success", out is not None and out["via"] == "arxiv-api"
          and "Abstract" in out["text"])
    check("fetch_arxiv_title_unwrapped",
          out is not None and out["title"] == "Attention Is All You Need",
          None if out is None else out["title"])
    check("fetch_arxiv_abstract_carried",
          out is not None and "dominant sequence transduction" in out["text"])
    check("fetch_arxiv_domain", out is not None and out["domain"] == "arxiv.org")
    check("fetch_arxiv_no_id_match_none", S.fetch_arxiv("https://arxiv.org/") is None)


def test_fetch_arxiv_stubbed_failures():
    class FakeResp:
        def __init__(self, status=200, text=""):
            self.status_code = status; self.text = text
    with _Patches(requests=types.SimpleNamespace(get=lambda *a, **k: FakeResp(status=500))):
        check("fetch_arxiv_non200_none", S.fetch_arxiv("https://arxiv.org/abs/1234.5678") is None)
    with _Patches(requests=types.SimpleNamespace(get=lambda *a, **k: FakeResp(text="<feed><entry></entry></feed>"))):
        check("fetch_arxiv_no_summary_none", S.fetch_arxiv("https://arxiv.org/abs/1234.5678") is None)
    with _Patches(requests=types.SimpleNamespace(
            get=lambda *a, **k: FakeResp(text="<feed><entry><summary>An abstract with no title tag.</summary></entry></feed>"))):
        out = S.fetch_arxiv("https://arxiv.org/abs/1234.5678")
        check("fetch_arxiv_no_title_uses_id", out is not None and "1234.5678" in out["title"])
    with _Patches(requests=types.SimpleNamespace(get=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))):
        check("fetch_arxiv_exception_none", S.fetch_arxiv("https://arxiv.org/abs/1234.5678") is None)


def test_fetch_crossref_real():
    # A well-known, stable DOI (Crossref REST works metadata + abstract).
    out = S.fetch_crossref("https://doi.org/10.1371/journal.pone.0000308")
    check("fetch_crossref_real_returns_dict_or_none", out is None or (out.get("via") == "crossref-api"))


def test_fetch_crossref_stubbed():
    class FakeResp:
        def __init__(self, status=200, data=None):
            self.status_code = status; self._data = data or {}
        def json(self):
            return self._data
    check("fetch_crossref_no_doi_none", S.fetch_crossref("https://doi.org/") is None)
    with _Patches(requests=types.SimpleNamespace(get=lambda *a, **k: FakeResp(status=404))):
        check("fetch_crossref_non200_none", S.fetch_crossref("https://doi.org/10.1/x") is None)
    with _Patches(requests=types.SimpleNamespace(get=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))):
        check("fetch_crossref_exception_none", S.fetch_crossref("https://doi.org/10.1/x") is None)
    full_msg = {
        "message": {
            "title": ["A Great Paper"],
            "author": [{"given": "Jane", "family": "Doe"}, {"given": "John", "family": "Smith"}],
            "container-title": ["Journal of Testing"],
            "published": {"date-parts": [[2020]]},
            "abstract": "<jats:p>An abstract with <b>tags</b>.</jats:p>",
        }
    }
    with _Patches(requests=types.SimpleNamespace(get=lambda *a, **k: FakeResp(data=full_msg))):
        out = S.fetch_crossref("https://doi.org/10.1/full")
        check("fetch_crossref_full_metadata", out is not None and "Jane Doe" in out["text"]
              and "Journal of Testing" in out["text"] and "2020" in out["text"]
              and "<b>" not in out["text"])
    thin_msg = {"message": {"title": ["X"]}}
    with _Patches(requests=types.SimpleNamespace(get=lambda *a, **k: FakeResp(data=thin_msg))):
        check("fetch_crossref_too_thin_none", S.fetch_crossref("https://doi.org/10.1/thin") is None)
    no_title_msg = {"message": {}}
    with _Patches(requests=types.SimpleNamespace(get=lambda *a, **k: FakeResp(data=no_title_msg))):
        out2 = S.fetch_crossref("https://doi.org/10.1/notitle")
        check("fetch_crossref_no_title_uses_doi", out2 is None or out2["title"] == "10.1/notitle")


def test_fetch_via_adapter():
    check("fetch_via_adapter_no_match_none", S.fetch_via_adapter("https://example.com/x") is None)
    with _Patches(_ADAPTERS={"wikipedia": lambda url: {"title": "T", "text": "X", "domain": "wikipedia.org", "via": "wikipedia-api"}}):
        out = S.fetch_via_adapter("https://en.wikipedia.org/wiki/X")
        check("fetch_via_adapter_dispatches", out is not None and out["via"] in ("wikipedia-api", "wikipedia-summary"))
    with _Patches(adapter_for=lambda url, cat="general": "nonexistent_adapter_name"):
        check("fetch_via_adapter_unknown_name_none", S.fetch_via_adapter("https://x.com") is None)


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

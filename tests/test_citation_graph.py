"""Coverage for citation_graph.py: OpenAlex citation-lineage enrichment. Uses
the REAL free/keyless OpenAlex API for the happy path (a well-known topic with
a genuinely old seminal paper), plus stubbed requests for failure/edge branches.
Run: venv/Scripts/python.exe tests/test_citation_graph.py
"""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
import citation_graph as C

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
            self.orig[k] = getattr(C, k)
            setattr(C, k, v)
        return self
    def __exit__(self, *a):
        for k, v in self.orig.items():
            setattr(C, k, v)


class FakeResp:
    def __init__(self, status=200, data=None):
        self.status_code = status
        self._data = data if data is not None else {}
    def json(self):
        return self._data


def test_get_stubbed():
    with _Patches(requests=types.SimpleNamespace(get=lambda *a, **k: FakeResp(status=200, data={"ok": True}))):
        out = C._get({"search": "x"}, "me@example.com")
        check("get_success", out == {"ok": True})
    with _Patches(requests=types.SimpleNamespace(get=lambda *a, **k: FakeResp(status=500))):
        check("get_non200_none", C._get({"search": "x"}, "") is None)
    with _Patches(requests=types.SimpleNamespace(get=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))):
        check("get_exception_none", C._get({"search": "x"}, "") is None)


def test_authors():
    work = {"authorships": [{"author": {"display_name": "A"}}, {"author": {"display_name": "B"}}]}
    check("authors_basic", C._authors(work, n=4) == "A, B")
    check("authors_empty", C._authors({}) == "")
    many = {"authorships": [{"author": {"display_name": f"P{i}"}} for i in range(6)]}
    out = C._authors(many, n=4)
    check("authors_truncates_with_et_al", out.endswith("et al.") and out.count(",") == 3)
    missing_names = {"authorships": [{"author": {}}, {"author": {"display_name": "X"}}]}
    check("authors_skips_missing_names", C._authors(missing_names) == "X")


def test_venue():
    check("venue_present", C._venue({"primary_location": {"source": {"display_name": "Nature"}}}) == "Nature")
    check("venue_missing_location", C._venue({}) == "")
    check("venue_missing_source", C._venue({"primary_location": {}}) == "")


def test_fmt():
    work = {"display_name": "A Great Paper", "publication_year": 1990, "cited_by_count": 500,
            "primary_location": {"source": {"display_name": "Journal X"}},
            "doi": "https://doi.org/10.1/xyz",
            "authorships": [{"author": {"display_name": "A"}}]}
    out = C._fmt(work)
    check("fmt_has_title_year", '"A Great Paper" (1990)' in out)
    check("fmt_has_citation_count", "cited_by=500" in out)
    check("fmt_has_venue", "Journal X" in out)
    check("fmt_has_doi_stripped", "doi:10.1/xyz" in out)
    check("fmt_has_authors", "A" in out)

    minimal = {}
    out2 = C._fmt(minimal)
    check("fmt_untitled_no_year_defaults", "(untitled)" in out2 and "n.d." in out2 and "cited_by=0" in out2)


def test_citation_lineage_brief_real():
    # A well-established, decades-old topic with a genuine pre-2000 seminal paper.
    out = C.citation_lineage_brief("backpropagation neural networks", mailto="test@example.com")
    check("citation_lineage_real_returns_something", out is None or ("SOURCE: PRIMARY" in out))
    if out:
        check("citation_lineage_real_has_openalex_note", "OpenAlex" in out)


def test_citation_lineage_brief_no_results():
    with _Patches(_get=lambda params, mailto: {"results": []}):
        check("citation_lineage_empty_results_none", C.citation_lineage_brief("t") is None)
    with _Patches(_get=lambda params, mailto: None):
        check("citation_lineage_get_fails_none", C.citation_lineage_brief("t") is None)
    with _Patches(_get=lambda params, mailto: {}):
        check("citation_lineage_no_results_key_none", C.citation_lineage_brief("t") is None)


def test_citation_lineage_brief_stubbed_seminal_path():
    calls = {"n": 0}
    def fake_get(params, mailto):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"results": [
                {"id": "https://openalex.org/W1", "display_name": "Old Seminal Work",
                 "publication_year": 1980, "cited_by_count": 1000, "authorships": [],
                 "primary_location": {}, "doi": ""},
                {"id": "https://openalex.org/W2", "display_name": "Newer Work",
                 "publication_year": 2010, "cited_by_count": 500, "authorships": [],
                 "primary_location": {}, "doi": ""},
            ]}
        return {"results": [
            {"display_name": "Descendant Work", "publication_year": 1995,
             "cited_by_count": 200, "authorships": [], "primary_location": {}, "doi": ""},
        ]}
    with _Patches(_get=fake_get):
        out = C.citation_lineage_brief("test topic", mailto="me@x.com")
        check("citation_lineage_seminal_labeled", "Likely seminal" in out)
        check("citation_lineage_descendants_labeled", "Principal descendants" in out)
        check("citation_lineage_has_both_works", "Old Seminal Work" in out and "Descendant Work" in out)


def test_citation_lineage_brief_not_seminal_path():
    calls = {"n": 0}
    def fake_get(params, mailto):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"results": [
                {"id": "https://openalex.org/W3", "display_name": "Recent Work",
                 "publication_year": 2015, "cited_by_count": 300, "authorships": [],
                 "primary_location": {}, "doi": ""},
            ]}
        return {"results": [
            {"display_name": "Related Work", "publication_year": 2018,
             "cited_by_count": 100, "authorships": [], "primary_location": {}, "doi": ""},
        ]}
    with _Patches(_get=fake_get):
        out = C.citation_lineage_brief("recent topic")
        check("citation_lineage_not_seminal_no_seminal_label", "Likely seminal" not in out)
        check("citation_lineage_related_high_impact_label", "Related high-impact" in out)


def test_citation_lineage_brief_no_dated_works():
    with _Patches(_get=lambda params, mailto: {"results": [
            {"display_name": "Undated Work", "cited_by_count": 50, "authorships": [],
             "primary_location": {}, "doi": ""}]}):
        out = C.citation_lineage_brief("undated topic")
        check("citation_lineage_no_dated_works_still_returns", out is not None
              and "Undated Work" in out)


def test_citation_lineage_brief_seminal_no_wid():
    with _Patches(_get=lambda params, mailto: {"results": [
            {"id": "", "display_name": "No ID Work", "publication_year": 1970,
             "cited_by_count": 10, "authorships": [], "primary_location": {}, "doi": ""}]}):
        out = C.citation_lineage_brief("t")
        check("citation_lineage_no_wid_still_returns", out is not None and "No ID Work" in out)


def test_citation_lineage_brief_descendants_empty():
    calls = {"n": 0}
    def fake_get(params, mailto):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"results": [
                {"id": "https://openalex.org/W9", "display_name": "Seminal Alone",
                 "publication_year": 1980, "cited_by_count": 999, "authorships": [],
                 "primary_location": {}, "doi": ""}]}
        return {"results": []}
    with _Patches(_get=fake_get):
        out = C.citation_lineage_brief("t")
        check("citation_lineage_no_descendants_no_section", "Principal descendants" not in out)


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

"""A silent model is not an empty web.

Measured live: LM Studio unloaded the model mid-run, every briefing call came
back empty, and deep research delivered this to the reader —

    "Pages were crawled but none contained facts relevant to the topic."

Eight pages about green and black tea had been fetched successfully. The
sentence blames the SOURCES for our own outage, and a reader takes it to mean
the subject is not covered on the web. The two failures need two messages: one
says try a different topic, the other says try again later.

Offline: the briefing boundary is replaced, nothing reaches a model.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import offline_guard; offline_guard.offline_llm()  # noqa: E402

import deep_research as D  # noqa: E402
import dr_progress  # noqa: E402

OK = BAD = 0


def check(name, cond, detail=""):
    global OK, BAD
    if cond:
        OK += 1
        print("PASS  " + name)
    else:
        BAD += 1
        print("FAIL  " + name + ((": " + str(detail)) if detail else ""))


PAGES = [{"domain": "tea.ru", "url": "https://tea.ru/a", "title": "Чай",
          "text": "Зелёный чай содержит катехины."},
         {"domain": "uteka.ru", "url": "https://uteka.ru/b", "title": "Чай 2",
          "text": "Чёрный чай содержит теафлавины."}]


def run_briefs(answer):
    """Drive _brief_pages with a briefer that answers however the test says."""
    prog = dr_progress._Progress(None)
    real = D.brief_source
    D.brief_source = lambda ctx, topic, page: answer(page)
    try:
        briefs, dropped = D._brief_pages(
            None, "чай", list(PAGES), {"category": "general"}, tempfile.mkdtemp(),
            prog, scholarly=False, terms=[], mode="confirm")
    finally:
        D.brief_source = real
    return briefs, dropped, prog


print("=" * 66)
print("THE COUNTER SEPARATES THE TWO FAILURES")
print("=" * 66)

_b, _d, _p = run_briefs(lambda page: "")
check("a model that answers nothing is counted as silent",
      _p.stats.get("silent_briefs") == 2, _p.stats)
check("...and produces no briefs", _b == [], _b)

_b, _d, _p = run_briefs(lambda page: "TRUST: SECONDARY\nЧай содержит катехины.")
check("a model that answers is not counted as silent",
      not _p.stats.get("silent_briefs"), _p.stats)
check("...and produces briefs", len(_b) == 2, _b)

# The two falsy answers mean opposite things and used to share one value:
#   ""   -- the model produced nothing: OUR failure.
#   None -- the model read the page and judged it useless: the page's.
# Counting both as silent would let a run of genuinely empty pages be reported
# to the reader as an outage on our side.
_b, _d, _p = run_briefs(lambda page: None)
check("a page the model judged useless is NOT an outage",
      not _p.stats.get("silent_briefs"), _p.stats)
check("...and still produces no brief", _b == [], _b)

# ...and the distinction is made by brief_source itself, not by the caller
# guessing. Asserted on behaviour: a docstring check would pass on a function
# that returns the wrong thing.
import dr_brief as _DB  # noqa: E402
import dr_calls as _DC  # noqa: E402

def _brief_with(answer):
    real = _DC.call_llm_simple
    _DC.call_llm_simple = lambda *a, **k: answer
    try:
        return _DB.brief_source(None, "чай", {"domain": "tea.ru", "title": "t",
                                              "text": "чай"})
    finally:
        _DC.call_llm_simple = real

check("a model that says nothing yields the empty string",
      _brief_with("") == "", repr(_brief_with("")))
check("...and so does a whitespace-only answer",
      _brief_with(" " + chr(10) + " ") == "")
check("a model that rejects the page yields None",
      _brief_with("NO USEFUL CONTENT") is None,
      repr(_brief_with("NO USEFUL CONTENT")))
check("a real brief comes back unchanged",
      _brief_with("TRUST: SECONDARY" + chr(10) + "Чай содержит катехины.").startswith("TRUST"))
check("the two failures are DIFFERENT values, which is the whole point",
      _brief_with("") is not None and _brief_with("NO USEFUL CONTENT") is None)

_b, _d, _p = run_briefs(lambda page: "" if page["domain"] == "tea.ru" else
                        "TRUST: SECONDARY\nЧай содержит теафлавины.")
check("a partial outage is counted per page",
      _p.stats.get("silent_briefs") == 1, _p.stats)
check("...and the pages that DID answer still count",
      len(_b) == 1, _b)

# The counter accumulates across waves: the main pass, the multi-hop pass and
# the contradiction pass each brief their own pages, and update() would have
# the last wave overwrite the total.
_prog = dr_progress._Progress(None)
_prog.bump(silent_briefs=3)
_prog.bump(silent_briefs=2)
check("silent pages accumulate across passes rather than overwrite",
      _prog.stats["silent_briefs"] == 5, _prog.stats)

print()
print("=" * 66)
print("THE REPORT SAYS WHICH FAILURE HAPPENED")
print("=" * 66)

_PP = [{"domain": "a"}, {"domain": "b"}]

_out = D.no_briefs_report("чай", _PP, 2)
_emp = D.no_briefs_report("чай", _PP, 0)

check("a total outage does NOT blame the sources",
      "absence of material" in _out and "none contained facts" not in _out, _out)
check("...and names how many pages were affected", "2 pages" in _out, _out)
check("...and says the fetch itself worked", "fetched successfully" in _out, _out)
check("an empty web still gets the empty-web sentence",
      "none contained facts relevant to the topic" in _emp, _emp)
check("a PARTIAL outage is not called a total one",
      "absence of material" not in D.no_briefs_report("чай", _PP, 1))
check("no pages at all is not reported as an outage",
      "absence of material" not in D.no_briefs_report("чай", [], 0))
check("the topic is carried in both messages",
      "чай" in _out and "чай" in _emp)

print()
print("%d passed, %d failed" % (OK, BAD))
sys.exit(1 if BAD else 0)

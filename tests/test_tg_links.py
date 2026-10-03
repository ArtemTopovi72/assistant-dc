"""A link in a chat message is read, not admired.

Live, 2026-09-12: a bare Wikipedia URL got "this is a link to an article;
you can find information there". Pure: the crawler is stubbed, no network.

Run: venv/Scripts/python.exe tests/test_tg_links.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["F5_TEST_RUN"] = "1"
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_links as L

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print("PASS  " + name)
    else:
        BAD += 1; print("FAIL  " + name + "   " + str(extra)[:200])


check("urls are found and trailing punctuation dropped",
      L.urls_in("глянь https://ru.wikipedia.org/wiki/Омлет. и http://a.b/c?d=1)") == ["https://ru.wikipedia.org/wiki/Омлет", "http://a.b/c?d=1"])
check("no url, no context", L.link_context("привет, как дела?") == "")

fake = types.SimpleNamespace(
    _acquire_page=lambda url, depth, seed, cat, ad, cache: {"page": {"title": "Омлет", "text": "Омлет — блюдо из яиц.\n\n\n\nЖарят на сковороде." * 3}},
    _fetch_page=lambda url: (None, None), _extract_text=lambda h, u: "", _extract_title=lambda h: "")
sys.modules["dr_crawl"] = fake
c = L.link_context("https://ru.wikipedia.org/wiki/Омлет")
check("the page text is attached with its title", "title: Омлет" in c and "блюдо из яиц" in c, c[:200])
check("blank runs are collapsed", "\n\n\n" not in c)
check("the text is capped", len(L.link_context("https://x/y", limit=50)) < 400)

fake2 = types.SimpleNamespace(
    _acquire_page=lambda *a: {"page": None, "reason": "landing"},
    _fetch_page=lambda url: ("<html><title>T</title><body>thin page</body></html>", None),
    _extract_text=lambda h, u: "thin page", _extract_title=lambda h: "T")
sys.modules["dr_crawl"] = fake2
check("a page the research gate refuses is still read for chat", "thin page" in L.link_context("https://x/y"))

fake3 = types.SimpleNamespace(_acquire_page=lambda *a: (_ for _ in ()).throw(RuntimeError("down")),
                              _fetch_page=lambda url: (None, None), _extract_text=lambda h, u: "", _extract_title=lambda h: "")
sys.modules["dr_crawl"] = fake3
check("a fetch failure yields nothing, never raises", L.link_context("https://x/y") == "")

src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(L.__file__))), "bot/tg_resolve.py"), encoding="utf-8").read()
check("the resolver attaches the page as scaffolding", "_tg_links.link_context(text)" in src and "self._read_links(" in src and "scaffold.append(_lc)" in src)
check("and never under F5_TEST_RUN", 'not os.getenv("F5_TEST_RUN")' in src)
_b = "Казань столица.\n" * 500 + "Население города 1 318 604 человека.\n" + "прочее\n" * 200
_o = L._fit(_b, "сколько жителей население Казани", 6000)
check("a long page keeps the paragraph the question is about", "1 318 604" in _o and "Page cut" in _o)

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)

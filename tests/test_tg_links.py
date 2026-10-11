"""A link in a chat message is read, not admired.

A bare Wikipedia URL must not get "this is a link to an article; you can find
information there". Pure: the crawler is stubbed, no network.

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

# An Ozon product link goes to the ozon_product tool: the crawler is not asked, and the
# model is told it is a shopping question (a bare link once became a video plan).
_OZ = "https://www.ozon.ru/product/avtomaticheskaya-kormushka-1884026101/?at=abc&sh=xyz"
_oz = L.link_context(_OZ)
check("an Ozon product link names the ozon_product tool", "ozon_product" in _oz and _OZ in _oz, _oz[:200])
check("an Ozon product link says it is not a video request", "do not make a video" in _oz)
check("an Ozon link is never sent to the page crawler",
      L.link_context(_OZ) != "" and "title:" not in L.link_context(_OZ))
check("a follow-up about an earlier Ozon link says so",
      "linked earlier" in L.link_context("а отзывы какие?", url=_OZ))
check("ozon.ru is matched, an ozon-looking foreign host is not",
      L.link_context("https://notozon.ru/product/1") == "" and L.link_context("https://ozon.ru.evil.example/product/1") == "")

src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(L.__file__))), "bot/tg_resolve.py"), encoding="utf-8").read()
check("the resolver attaches the page as scaffolding", "_tg_links.link_context(text)" in src and "self._read_links(" in src and "scaffold.append(_lc)" in src)
check("and never under F5_TEST_RUN", 'not os.getenv("F5_TEST_RUN")' in src)
_b = "Казань столица.\n" * 500 + "Население города 1 318 604 человека.\n" + "прочее\n" * 200
_o = L._fit(_b, "сколько жителей население Казани", 6000)
check("a long page keeps the paragraph the question is about", "1 318 604" in _o and "Page cut" in _o)

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)

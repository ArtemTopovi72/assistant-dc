"""tg_reply_shape: a searched answer = 🔎 queries, [n] footnotes, 🔗 sources.
The register (turn_queries / turn_sources) lives on the scoped context and is
filled by search.py; a turn without a search keeps its reply untouched."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tg_reply_shape as S
import tg_markup as M

PASSED = FAILED = 0


def check(label, cond, detail=""):
    global PASSED, FAILED
    if cond: PASSED += 1; print("PASS ", label)
    else:    FAILED += 1; print("FAIL ", label, " ", str(detail)[:300])


SRC = [{"title": "Col de l'Iseran — Wikipedia", "domain": "en.wikipedia.org", "url": "https://en.wikipedia.org/wiki/Col_de_l%27Iseran?x=1&y=2"},
       {"title": "Alpine passes", "domain": "www.alpenpass.info", "url": "https://www.alpenpass.info/a"},
       {"title": "Cycling", "domain": "cyclingcols.com", "url": "https://cyclingcols.com/b"}]

out = S.shape_search_reply("Самая высокая — Коль-де-л'Изеран, 2764 м (source: alpenpass.info). "
                           "Открыта с июня (источник: en.wikipedia.org, cyclingcols.com).",
                           ["highest paved road Europe"], SRC)
check("header names the query", out.startswith("🔎 <b>Искал:</b> «highest paved road Europe»"), out)
check("(source: domain) becomes [n] in citation order, www. ignored",
      "2764 м [1]." in out and "с июня [2] [3]." in out, out)
check("cited sources come first in the list",
      out.index("Alpine passes</a>") < out.index("Wikipedia</a>") < out.index("Cycling</a>"), out)
check("links are HTML anchors with escaped urls", 'href="https://en.wikipedia.org/wiki/Col_de_l%27Iseran?x=1&amp;y=2"' in out, out)
check("the footer is labelled", "\n\n🔗 <b>Источники:</b>\n1. " in out, out)
check("the shaped reply survives tag-safe splitting whole", M._split_html(out) == [out])

bare = S.shape_search_reply("Взрослый: 2390 ₽ (cyclingcols.com). Льготный (www.alpenpass.info, cyclingcols.com).", ["q"], SRC)
check("a bare (domain) mark is a footnote too", "2390 ₽ [1]." in bare and "Льготный [2] [1]." in bare, bare)
# A "(source: x)" for a page that was never read is not a citation (live
# 2026-09-28: a raw "(source: wikipedia.org)" stayed in the answer).
check("an unknown source: mark is dropped, not shown raw",
      "source:" not in S.shape_search_reply("x (source: nowhere.org).", ["q"], SRC))
check("no search this turn → reply untouched", S.shape_search_reply("<b>hi</b>", [], []) == "<b>hi</b>")
en = S.shape_search_reply("42 (source: cyclingcols.com)", ["q"], SRC, lang="en")
check("English labels", "🔎 <b>Searched:</b>" in en and "🔗 <b>Sources:</b>" in en, en)
many = [{"title": f"t{i}", "domain": f"d{i}.com", "url": f"https://d{i}.com"} for i in range(12)]
check("the list is capped", S.shape_search_reply("x", ["q"], many).count("<a href") == S.MAX_SOURCES)

note = ("[The user's message contains a link. Here is that page, read for you -- title: Омлет — Википедия "
        "(https://ru.wikipedia.org/wiki/Омлет). Answer from it; if the user only sent the link, tell them briefly "
        "what it is about.]\nОмлет — блюдо из яиц…")
lr = S.links_read("о чём это? https://ru.wikipedia.org/wiki/Омлет\n" + note)
check("a read link is parsed out of the scaffold", lr and lr[0]["domain"] == "ru.wikipedia.org"
      and lr[0]["title"] == "Омлет — Википедия" and lr[0]["kind"] == "link", lr)
lk = S.shape_search_reply("Это статья про омлет.", [], lr)
check("a link turn: «📄 Прочитал: title — domain» on top, no source list",
      lk.startswith('📄 <b>Прочитал:</b> <a href="https://ru.wikipedia.org/wiki/Омлет">Омлет — Википедия</a> — ru.wikipedia.org\n\nЭто статья')
      and "Источники" not in lk, lk)

# the register on the scoped context, filled by search.py
import tg_bot as T
T.redirect_data_dir(tempfile.mkdtemp(prefix="tg_shape_"))
import models
ctx = T._scoped_ctx(models.Context.__new__(models.Context))
check("_scoped_ctx starts an empty register", ctx.turn_queries == [] and ctx.turn_sources == [])
import search as _search
_search.raw_search_results = lambda ctx, q, n=None: [
    {"title": "A", "href": "https://a.com/1", "body": "aaa"}, {"title": "B", "href": "https://b.org/2", "body": "bbb"}]
_search.SEARCH_DISTILL = False
_search.run_web_search(ctx, "test query")
check("search.py records the query and the pages read",
      ctx.turn_queries == ["test query"] and [s["domain"] for s in ctx.turn_sources] == ["a.com", "b.org"]
      and ctx.turn_sources[0]["url"] == "https://a.com/1", (ctx.turn_queries, ctx.turn_sources))
_search.run_web_search(ctx, "test query")
check("a repeated search does not duplicate the register", len(ctx.turn_queries) == 1 and len(ctx.turn_sources) == 2)
import payload_guard as PG
check("the model is told to keep the (source: domain) marks", "(source: domain)" in PG.SKEPTIC_BANNER)

_own = S.shape_search_reply("Официальный сайт: cbr.ru [1]. Массив a[1] цел.",
                            ["q"], [{"domain": "checko.ru", "title": "Банк России", "url": "https://checko.ru/x"}], "ru")
check("a [n] the model wrote is not kept (it pointed at the wrong source)",
      "cbr.ru [1]" not in _own.split("🔗")[0] and "a[1]" in _own, _own)

print(f"\n{PASSED}/{PASSED + FAILED} checks passed")
sys.exit(1 if FAILED else 0)


def test_subdomain_and_unknown_citations():
    import tg_reply_shape as _S
    src = [{"domain": "ru.wikipedia.org", "title": "ЧМ", "url": "https://ru.wikipedia.org/x"}]
    out = _S.shape_search_reply("Аргентина (source: wikipedia.org). Ещё (source: made.up) факт. Node (node.js) ок",
                                ["q"], src, "ru")
    assert "[1]" in out and "source:" not in out and "(node.js)" in out, out

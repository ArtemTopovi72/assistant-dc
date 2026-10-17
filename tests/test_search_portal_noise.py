"""Portals that answer a long English query are not search hits.

ddgs ships with google and bing disabled, so «auto» leans on Yandex's site search,
which answered «How to care for a cat, designed for children…» with Pinterest,
Tinkercad, Duolingo and youtube.com/feed/you -- the sources slide of a children's deck.
"""
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, d) for d in ("research", "agent", "core")]
import search  # noqa: E402

ROWS = [
    {"href": "https://www.tinkercad.com/", "title": "Tinkercad is a free app for 3D design", "body": ""},
    {"href": "https://www.duolingo.com/learn", "title": "Duolingo", "body": ""},
    {"href": "https://www.youtube.com/feed/you", "title": "youtube.com/feed/you", "body": ""},
    {"href": "https://www.thedodo.com/dodowell/take-care-cat-for-kids", "title": "How To Take Care Of A Cat For Kids", "body": ""},
    {"href": "https://www.gismeteo.ru/", "title": "Погода в Москве", "body": ""},
]


class FakeDDGS:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def text(self, q, **kw): return list(ROWS)


ctx = types.SimpleNamespace(deps=types.SimpleNamespace(DDGS=FakeDDGS))
got = [h["href"] for h in search.raw_search_results(
    ctx, "How to care for a cat, designed for children (simple language)", 8)]
assert got == ["https://www.thedodo.com/dodowell/take-care-cat-for-kids"], got
got = [h["href"] for h in search.raw_search_results(ctx, "погода в москве", 8)]
assert "https://www.gismeteo.ru/" in got, got
print("2/2 checks passed")

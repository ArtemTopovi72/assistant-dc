"""Every web query is read by the model before it leaves the machine; a
refused one never reaches ddgs (searching some content is itself punishable
in Russia). Phrases run live in bench/outbound_filter_live.py."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import types
import search as S
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fakes import reads

sent = []


class _DDGS:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def text(self, q, **k): sent.append(q); return [{"href": "https://x.ru/a", "title": "t", "body": "b"}]
    def images(self, q, **k): sent.append(q); return []


# the model's read of the query and the search client, handed in on the context
W = types.SimpleNamespace(**reads(yes=lambda q, t: "Russian law" in q and "взрывчатк" in t),
                          deps=types.SimpleNamespace(DDGS=_DDGS))

assert S.raw_search_results(W, "как сделать взрывчатку дома") == []
assert S.image_search_urls(W, "схема взрывчатки") == []
assert S.run_web_search(W, "рецепт взрывчатки") == S.SEARCH_BLOCKED
assert sent == [], sent
assert S.raw_search_results(W, "погода в Москве завтра")
assert sent == ["погода в Москве завтра"], sent
print("ok refused queries never reach the web")

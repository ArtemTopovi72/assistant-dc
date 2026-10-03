import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tg_links

tg_links.read_link = lambda url: {"title": "Казань", "text": "шапка\n" * 2000 + "Город был основан в 1005 году."}
ctx = tg_links.link_context("а когда основан город?", url="https://ru.wikipedia.org/wiki/Казань")
assert "1005" in ctx and "linked earlier" in ctx, ctx[:200]
import inspect, tg_resolve
assert "url=sess.last_link" in inspect.getsource(tg_resolve)
print("PASS link follow-up")
_page = ("вступление жители любят чай\n" * 300 + "Население\n[править]| 2000 | 2025 |\n|---|---|\n| 1 000 | 1 329 825 |\n"
         + "Климат мягкий.\n" * 50)
assert "1 329 825" in tg_links._fit(_page, "сколько там жителей?", 6000)
print("PASS population table under its heading")
assert "1 329 825" in tg_links._fit(_page.replace("вступление", "казань"), "https://x.org/wiki/Казань сколько там жителей?", 6000)
print("PASS url words ignored")

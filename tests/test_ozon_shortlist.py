"""A shortlist across messages yields one verified offer."""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services"))

import ozon_same
import ozon_shopper
import intent
import tool_ozon_handlers as handler


URLS = [f"https://ozon.ru/t/card{i}" for i in range(7)]


def test_intent_routes_many_links_to_shop_before_same_seller():
    out = {"must_call": "ozon_same", "wants": ["ozon_same", "ozon_shop"]}
    intent._same_or_analogue(SimpleNamespace(),
                             f"Выбери лучший {URLS[0]} {URLS[1]} и найди дешевле", out)
    assert out["must_call"] == "ozon_shop"
    assert out["wants"][0] == "ozon_shop"


def test_collects_seven_recent_user_links_only():
    messages = []
    for url in URLS:
        messages.extend([{"role": "user", "content": f"Рюкзак {url}"},
                         {"role": "assistant", "content": "https://ozon.ru/t/assistant"}])
    links, words = handler._shop_shortlist({"messages": messages,
                                            "user_input_original": "Сравни и выбери лучший"})
    assert links == URLS
    assert "Рюкзак" in words and "assistant" not in words
    messages.insert(0, {"role": "user", "content": "https://ozon.ru/t/stale"})
    messages.insert(1, {"role": "user", "content": "Другая тема"})
    assert handler._shop_shortlist({"messages": messages,
                                    "user_input_original": "Выбери лучший"})[0] == URLS
    assert handler._shop_shortlist({"messages": messages,
                                    "user_input_original": "Подбери один по качеству"})[0] == URLS


def test_chooses_one_verified_cheaper_seller(monkeypatch):
    seen = {}
    winner = {"sku": "1", "url": URLS[0], "name": "Рюкзак", "price": 3000}

    def shop(ctx, req, strategy, pinned):
        seen["pinned"] = pinned
        seen["request"] = req
        return {"mode": "one"}, [{"choice": winner, "item": {"need": "рюкзак"},
                                   "decision": {}, "others": []}]

    monkeypatch.setattr(ozon_shopper, "shop", shop)
    monkeypatch.setattr(ozon_shopper, "report", lambda results, *a: results[0])
    monkeypatch.setattr(ozon_same, "find_same", lambda *a, **kw: {"best": {"cheap": {
        "sku": "2", "url": "https://ozon.ru/product/same-2/", "price": 2000,
        "check": {"verdict": "same", "copy_price": False}}}})
    ctx = SimpleNamespace(turn_products=[])
    state = {"messages": [{"role": "user", "content": u} for u in URLS],
             "user_input_original": "Выбери лучший рюкзак"}
    result = handler._handle_ozon_shop(ctx, state, {"request": "рюкзак"})
    assert seen["pinned"] == URLS
    assert all(url not in seen["request"] for url in URLS)
    assert result["choice"]["sku"] == "2" and result["same_offer_saving"] == 1000
    assert [x["sku"] for x in ctx.turn_products] == ["2"]


def test_unverified_offer_cannot_replace_winner(monkeypatch):
    winner = {"sku": "1", "url": URLS[0], "name": "Рюкзак", "price": 3000}
    monkeypatch.setattr(ozon_shopper, "shop", lambda *a, **kw: (
        {"mode": "one"}, [{"choice": winner, "item": {"need": "рюкзак"},
                           "decision": {}, "others": []}]))
    monkeypatch.setattr(ozon_shopper, "report", lambda results, *a: results[0])
    monkeypatch.setattr(ozon_same, "find_same", lambda *a, **kw: {"best": {"cheap": {
        "sku": "copy", "url": "https://ozon.ru/product/copy/", "price": 1000,
        "check": {"verdict": "unclear"}}}})
    state = {"messages": [{"role": "user", "content": u} for u in URLS],
             "user_input_original": "Выбери лучший рюкзак"}
    result = handler._handle_ozon_shop(SimpleNamespace(turn_products=[]), state,
                                       {"request": "рюкзак"})
    assert result["choice"]["sku"] == "1"

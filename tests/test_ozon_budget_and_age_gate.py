"""Live 2026-10-01: a 30 000 ₽ party basket was planned with no budget (the agent's
retelling dropped it), and «анальная пробка» came back as bath plugs (Ozon's 18+
gate hid the real listings and other wordings drifted to look-alikes)."""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ozon_client as O
import ozon_shopper as S
import tool_ozon_handlers as H


def test_user_words_reach_the_plan():
    st = {"user_input_original": "собери корзину, бюджет 30000 рублей"}
    req = H._with_user_words(st, "набор для вечеринки")
    assert "30000" in req and req.startswith("набор для вечеринки")
    assert H._with_user_words({}, "x") == "x"


def test_age_gate_is_detected():
    page = {"widgetStates": {"userAdultModal-747789-default-1": "{}"}}
    assert O._age_gated(page) and not O._age_gated({"widgetStates": {"searchResultsV2-1": "{}"}})


def test_gated_need_is_not_found_not_substituted(monkeypatch):
    def search(q, *a, **k):
        if "анальн" in q:
            raise O.OzonAgeGate("18+")
        return [{"sku": "9", "name": "Пробка для ванны", "price": 99}]
    monkeypatch.setattr(O, "search", search)
    item = {"need": "Анальная пробка", "queries": ["анальная пробка", "пробка силиконовая"],
            "must": [], "avoid": [], "max_price": None, "qty": 1}
    r = S.shop_item(types.SimpleNamespace(), item)
    assert r["choice"] is None and "18+" in r["note"]
    assert "NOT FOUND" in S.report([r], "one")


def test_gate_is_confirmed_once_with_the_allowed_birthdate(monkeypatch):
    gated, open_ = {"widgetStates": {"userAdultModal-1": "{}"}}, {"widgetStates": {}}
    state = {"confirmed": 0}
    monkeypatch.setattr(O, "fetch_page", lambda p: open_ if state["confirmed"] else gated)
    monkeypatch.setattr(O, "confirm_age", lambda p: state.__setitem__("confirmed", state["confirmed"] + 1))
    monkeypatch.setattr(O, "_birthdate", lambda: "11.11.1999")
    assert O._search_page("/search/?text=x", "x") is open_ and state["confirmed"] == 1
    state["confirmed"] = 0
    monkeypatch.setattr(O, "_birthdate", lambda: "")
    try:
        O._search_page("/search/?text=x", "x"); assert False
    except O.OzonAgeGate:
        assert state["confirmed"] == 0


def test_share_link_opens_the_card(monkeypatch):
    seen = []
    monkeypatch.setattr(O, "resolve_short", lambda u: seen.append(u) or "/product/sibir-57-1173391354/")
    assert O.product_path("https://ozon.ru/t/pHQ8pRC") == "/product/sibir-57-1173391354/"
    assert O.product_path("ozon.ru/t/pHQ8pRC") == "/product/sibir-57-1173391354/"
    assert O.product_path("https://www.ozon.ru/product/x-1/") == "/product/x-1/" and len(seen) == 2

"""The Ozon shopper's reasoning loop with the LLM, Ozon and vision scripted:
plan -> gather -> screen -> inspect -> decide -> photo check -> basket."""
import os, sys, json, types, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest

import ozon_client as O
import ozon_shopper as S
import tools as T


def item(sku, name, price, rating=4.7, reviews=100):
    return {"sku": sku, "name": name, "price": price, "rating": rating, "reviews": reviews,
            "delivery": "Завтра", "url": f"https://www.ozon.ru/product/x-{sku}/",
            "image": f"https://ir.ozone.ru/{sku}.jpg"}


CATALOG = {
    "скатерть": [item("1", "Скатерть для пикника 150x150", 700), item("2", "Скатерть непромокаемая", 900)],
    "тарелки": [item("3", "Тарелки одноразовые 50 шт", 200), item("4", "Держатель для тарелок", 50)],
}


@pytest.fixture
def world(monkeypatch, tmp_path):
    import tool_ozon_handlers as H
    monkeypatch.setattr(H, "USERS_DIR", str(tmp_path / "users"))
    calls = {"search": [], "llm": [], "vision": [], "decide_input": []}
    script = {}

    def search(q, sort="popular", pmin=None, pmax=None, limit=10, max_delivery_days=None, page=1):
        if sort == "popular" and page == 1:         # extra pages/sorts: same catalog, deduped
            calls["search"].append(q)
        return [dict(x) for k, v in CATALOG.items() if k in q for x in v]

    def details(sku):
        return {"sku": sku, "available": True, "seller": "S", "images": [f"https://ir.ozone.ru/{sku}.jpg"],
                "characteristics": {"Размер": "150x150"}}

    def reviews(sku, limit=10):
        return {"reviews": [] if sku == "3" else [{"score": 5, "comment": "Отличная", "date": "2026-09-01",
                                                   "photos": script.get("buyer_photos", [])}]}

    def llm(ctx, system, user, **k):
        calls["llm"].append(system[:40])
        if "shopping plan" in system:
            return json.dumps(script["plan"], ensure_ascii=False)
        if "shop assistant on Ozon" in system:
            calls["clarify_input"] = user
            return json.dumps({"questions": script.get("questions", [])}, ensure_ascii=False)
        if "pick finalists" in system:
            n = [i + 1 for i, l in enumerate(user.split("LISTINGS")[1].strip().splitlines())
                 if "Держатель" not in l]
            return json.dumps({"finalists": n[:3], "note": ""})
        calls["decide_input"].append(user)
        return json.dumps({"choice": 1, "why": "дешевле", "reviews": "хвалят", "concerns": ""},
                          ensure_ascii=False)

    def vision(**k):
        calls["vision"].append(k["user_text"][:40])
        return script.get("vision", "YES, it is.")

    import llm as L, requests
    monkeypatch.setattr(O, "search", search)
    monkeypatch.setattr(O, "search_filters", lambda q: script.get("filters", []))
    monkeypatch.setattr(O, "details", details)
    monkeypatch.setattr(O, "reviews", reviews)
    monkeypatch.setattr(L, "call_llm_simple", llm)
    monkeypatch.setattr(L, "analyze_image_with_llm", vision)
    monkeypatch.setattr(requests, "get", lambda url, **k: types.SimpleNamespace(
        content=b"j", raise_for_status=lambda: None))
    return calls, script


def _ctx(owner="77"):
    return types.SimpleNamespace(set_stage=lambda *a, **k: None, cancel_event=threading.Event(),
                                 ozon_owner=owner, turn_products=[])


def test_basket_plans_every_item_and_fills_the_cart(world):
    calls, script = world
    script["plan"] = {"mode": "basket", "items": [
        {"need": "скатерть", "queries": ["скатерть пикник", "скатерть непромокаемая"]},
        {"need": "тарелки", "queries": ["тарелки одноразовые"], "qty": 2}]}
    ctx = _ctx()
    out = T.execute_tool(ctx, {}, "ozon_shop", {"request": "пикник на 4", "add_to_cart": True})
    assert calls["search"] == ["скатерть пикник", "скатерть непромокаемая", "тарелки одноразовые"]
    assert "Скатерть для пикника" in out and "Тарелки одноразовые" in out, out
    assert "Держатель" not in out.split("also considered")[0]
    assert "Отзывов нет" in out                        # sku 3 has none, said out loud
    assert "TOTAL for the found items: 1 100 ₽" in out   # 700 + 2 x 200
    assert "Added to the user's shopping list" in out and "× 2" in out
    assert [p["sku"] for p in ctx.turn_products] == ["1", "3"]      # cards for the picks


def test_photo_that_shows_something_else_hands_the_win_to_the_next(world):
    calls, script = world
    script["plan"] = {"mode": "one", "items": [{"need": "скатерть", "queries": ["скатерть"]}]}
    answers = iter(["NO, this is a grille.", "YES"])
    import llm as L
    L.analyze_image_with_llm = lambda **k: next(answers)
    out = T.execute_tool(_ctx(), {}, "ozon_shop", {"request": "скатерть"})
    assert "CHOICE: Скатерть непромокаемая" in out, out
    assert "rejected 'Скатерть для пикника 150x150': its photo shows something else" in out


def test_nothing_found_is_reported_not_invented(world):
    calls, script = world
    script["plan"] = {"mode": "one", "items": [{"need": "вертолёт", "queries": ["вертолёт"]}]}
    out = T.execute_tool(_ctx(), {}, "ozon_shop", {"request": "вертолёт"})
    assert "NOT FOUND" in out and "never substitute" in out


def test_planner_failure_falls_back_to_the_words(world):
    calls, script = world
    script["plan"] = {"oops": 1}
    p = S.plan(_ctx(), "скатерть")
    assert p["mode"] == "one" and p["items"][0]["queries"] == ["скатерть"]


def test_cancel_stops_between_steps(world):
    calls, script = world
    script["plan"] = {"mode": "one", "items": [{"need": "скатерть", "queries": ["скатерть"]}]}
    ctx = _ctx(); ctx.cancel_event.set()
    assert "cancelled" in T.execute_tool(ctx, {}, "ozon_shop", {"request": "скатерть"})


def test_review_warning_signs():
    fake = [{"score": 5, "comment": "супер", "date": "2026-09-01"}] * 8 + \
           [{"score": 1, "comment": "брак", "date": "2026-09-02"}] * 2
    s = S.review_signals(fake)
    assert "J-curve" in s and "within 2 days" in s and "identical" in s
    honest = [{"score": s_, "comment": f"текст {i}", "date": f"2026-0{1 + i % 8}-10"}
              for i, s_ in enumerate([5, 4, 4, 3, 5, 4, 2, 5])]
    assert S.review_signals(honest) == ""


def test_buttons_route_to_the_shopper():
    import tg_bot, tg_keyboards, re, tool_retrieval as R
    from tg_strings import _b
    for key in ("ozon_cheap", "ozon_best", "ozon_fast", "ozon_basket"):
        assert "ozon_shop" in tg_bot._PROMPT_KB[key]
    assert _b("ozon_basket", "ru") in [b for r in tg_keyboards._ozon_kb("ru")["keyboard"] for b in r]


def test_buyer_photos_are_looked_at_and_reach_the_decision(world):
    calls, script = world
    script["plan"] = {"mode": "one", "items": [{"need": "скатерть", "queries": ["скатерть"]}]}
    script["buyer_photos"] = ["https://ir.ozone.ru/r1.jpg", {"url": "https://ir.ozone.ru/r2.jpg"},
                              "https://ir.ozone.ru/r3.jpg"]
    script["vision"] = "YES, a checked picnic blanket, thin fabric."
    out = T.execute_tool(_ctx(), {}, "ozon_shop", {"request": "скатерть"})
    buyer_asks = [v for v in calls["vision"] if v.startswith("A buyer of '")]
    assert len(buyer_asks) == 2 * 2                      # 2 per finalist, not 3
    assert "BUYER photo (vision): YES, a checked picnic blanket" in calls["decide_input"][0]
    assert "buyers' photos show:" in out


def test_review_photo_urls_accept_every_shape():
    c = {"photos": ["https://a/1.jpg", {"url": "https://a/2.jpg"}, {"src": "https://a/3.jpg"},
                   {"x": 1}, "not-a-url"]}
    assert O.review_photo_urls(c) == ["https://a/1.jpg", "https://a/2.jpg", "https://a/3.jpg"]


def test_vision_is_told_the_listing_title(world):
    calls, script = world
    script["plan"] = {"mode": "one", "items": [{"need": "скатерть", "queries": ["скатерть"]}]}
    seen = []
    import llm as L
    L.analyze_image_with_llm = lambda **k: seen.append(k["user_text"]) or "YES"
    T.execute_tool(_ctx(), {}, "ozon_shop", {"request": "скатерть"})
    assert any("Listing: 'Скатерть для пикника 150x150'" in s for s in seen)
    assert all("still count as YES" in s for s in seen if s.startswith("Listing:"))


def test_asks_from_the_category_filters_before_shopping(world):
    calls, script = world
    script["plan"] = {"mode": "one", "items": [{"need": "снегоуборщик", "queries": ["снегоуборщик"]}]}
    script["filters"] = [{"key": "selfpropelled", "title": "Самоходный", "values": ["да", "нет"]},
                         {"key": "starter", "title": "Тип запуска", "values": ["ручной", "электростартер"]}]
    script["questions"] = [{"filter": "Самоходный", "question": "Самоходный нужен?",
                            "options": ["да", "нет", "неважно"]}]
    out = T.execute_tool(_ctx(), {}, "ozon_shop", {"request": "снегоуборщик до 60000"})
    assert out.startswith("NOT SEARCHED YET") and "Самоходный нужен?" in out
    assert "Тип запуска: ручной, электростартер" in calls["clarify_input"]
    assert calls["search"] == []                     # nothing bought blind
    out = T.execute_tool(_ctx(), {}, "ozon_shop", {"request": "снегоуборщик до 60000, самоходный",
                                                   "clarified": True})
    assert "NOT SEARCHED" not in out and calls["search"]


def test_promo_filters_are_not_questions():
    page = {"widgetStates": {"filtersDesktop-1-default-1": json.dumps({"sections": [{"filters": [
        {"type": "boolFilter", "key": "x1", "boolFilter": {"title": "Рассрочка 0-0-6"}},
        {"type": "boolFilter", "key": "x2", "boolFilter": {"title": "Больше морковок от Захара"}},
        {"type": "boolFilter", "key": "x3", "boolFilter": {"title": "Самоходный"}}]}]})}}
    assert [f["title"] for f in O.parse_filters(page)] == ["Самоходный"]

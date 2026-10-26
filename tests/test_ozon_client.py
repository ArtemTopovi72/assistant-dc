"""Ozon parsers against saved composer-api pages -- no browser, no network.

Samples come from eduard256/ozon-mcp-server (MIT), whose parse.test.js these
checks mirror, so a port bug shows up as a difference from the original.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest

import ozon_client as O
from seams import Over

_S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")


def _load(name):
    with open(os.path.join(_S, f"ozon_{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def test_search_items_have_sku_price_and_clean_url():
    items = O.parse_search(_load("search"), 12)
    assert items, "no items parsed"
    assert all(i["sku"] and isinstance(i["price"], int) for i in items)
    assert all(i["url"].startswith("https://") and "?" not in i["url"] for i in items)
    assert any(i["name"] for i in items)
    assert any(isinstance(i["rating"], float) for i in items)


def test_search_limit():
    assert len(O.parse_search(_load("search"), 3)) <= 3


def test_details_has_the_card():
    d = O.parse_details(_load("pdp"), _load("pdp_page2"))
    assert d["sku"] and d["name"] and isinstance(d["price"], int), d
    assert d["images"], "no images"
    assert d["characteristics"], "no characteristics"
    assert d["description"], "no description"


def test_details_read_every_characteristic_of_page_two():
    # The card's top shows five; a model number or a battery size the judge
    # needs is only in page two's full list.
    c = O.parse_details(_load("pdp"), _load("pdp_page2"))["characteristics"]
    assert c["Встроенная память"] == "128 ГБ" and c["Версия iOS"] == "iOS 17"
    assert c["Стандарты связи"] == "2G, 3G, 4G (LTE), 5G"
    assert "Артикул" not in c, "Ozon's own SKU is not the maker's article"


def test_reviews_have_scores_and_text():
    r = O.parse_reviews(_load("reviews"), 10)
    assert r["reviews"]
    assert any(isinstance(x["score"], (int, float)) for x in r["reviews"])
    assert any(x["comment"] or x["pros"] or x["cons"] for x in r["reviews"])


def test_reviews_only_of_this_variant():
    # the sample page mixes 1185261285 with its sibling 1185261102
    every = O.parse_reviews(_load("reviews"), 30)["reviews"]
    own = O.parse_reviews(_load("reviews"), 30, "1185261285")
    assert 0 < len(own["reviews"]) < len(every)
    assert own["next"] and "page=2" in own["next"]
    assert "&page=2" in O.search_path("снегоуборщик", page=2) and "page=" not in O.search_path("x")


def test_price_and_url_helpers():
    assert O.price_to_number("53 022 ₽") == 53022
    assert O.price_to_number(None) is None
    assert O.clean_url("/product/x-123/?at=abc") == "https://www.ozon.ru/product/x-123/"
    assert O.sku_from_url("https://www.ozon.ru/product/naushniki-1185261285/") == "1185261285"


def test_product_path_accepts_link_sku_slug():
    assert O.product_path("1185261285") == "/product/1185261285/"
    assert O.product_path("https://www.ozon.ru/product/x-1/?at=q") == "/product/x-1/"
    assert O.product_path("x-1") == "/product/x-1/"
    with pytest.raises(O.OzonError):
        O.product_path("")


def test_search_path_sort_and_price():
    p = O.search_path("наушники", "price", 1000, 5000)
    assert "sorting=price" in p and "currency_price=1000.000%3B5000.000" in p
    assert "text=%D0%BD" in p


def test_cheapest_is_the_relevance_ranking_by_price():
    q = "самогонный аппарат"
    seen = []
    items = [{"sku": "1", "name": "Комплектующие для самогонного аппарата", "price": 98},
             {"sku": "2", "name": "Самогонный аппарат 20 л", "price": 9000},
             {"sku": "3", "name": "Самогонный аппарат 12 л", "price": 6000}]
    gw = Over(O, fetch_page=lambda p: seen.append(p) or {}, parse_search=lambda page, limit=12: list(items))
    assert [x["sku"] for x in O.search(q, "price", gw=gw)] == ["1", "3", "2"]   # parts: ozon_search's judge
    assert "sorting=" not in seen[0]          # relevance ranking, re-sorted here


def test_delivery_days_and_filter():
    import datetime as dt
    t = dt.date(2026, 6, 9)
    assert O.delivery_days("12 июня", t) == 3 and O.delivery_days("завтра", t) == 1
    assert O.delivery_days("5 января", dt.date(2026, 12, 30)) == 6
    assert O.delivery_days("скоро", t) is None
    page = json.load(open(os.path.join(os.path.dirname(__file__), "samples", "ozon_search.json"),
                          encoding="utf-8"))
    assert O.parse_search(page)[0]["delivery"] == "12 июня"
    gw = Over(O, fetch_page=lambda p: page,
              delivery_days=lambda s, today=None: {"10 июня": 1, "12 июня": 3}.get(s))
    assert {x["delivery"] for x in O.search("x", max_delivery_days=1, gw=gw)} == {"10 июня"}


def test_parse_location_reads_the_header_bar():
    bar = ('{"customCell":{"cells":[{"type":"button","button":{"text":"Пункт Ozon •"}},'
           '{"type":"button","button":{"text":"ул. Миславского, 9"}}],'
           '"separator":{"text":"&bull;"}}}')
    assert O.parse_location({"widgetStates": {"addressBookBarWeb-1-default-1": bar}}) == \
        "Пункт Ozon • ул. Миславского, 9"
    city = '{"cells":[{"button":{"text":"Санкт-Петербург"}},{"button":{"text":"Укажите адрес"}}]}'
    assert O.parse_location({"widgetStates": {"addressBookBarWeb-1": city}}) == "Санкт-Петербург"


def test_nearest_point_opens_clusters_until_a_pin():
    def page(objs):
        return {"widgetStates": {"addressEditMap-1": json.dumps(
            {"markerBundle": {"mapObjectCollection": {"mapObjects": objs}}})}}
    pin = lambda la, lo, pp: {"type": "PIN", "coordinates": {"latitude": la, "longitude": lo},
                               "actionLink": f"/modal/commonDelivery?pid=5&pp={pp}"}
    seen = []
    def fake(path):
        seen.append(path)
        if len(seen) == 1:
            return page([{"type": "CLUSTER", "coordinates": {"latitude": 55.0, "longitude": 49.0},
                          "actionLink": "/zoom"}])
        return page([pin(55.5, 49.5, "1"), pin(55.01, 49.01, "2")])
    assert O.nearest_point(55.0, 49.0, gw=Over(O, fetch_page=fake)) == "2" and seen[1] == "/zoom"

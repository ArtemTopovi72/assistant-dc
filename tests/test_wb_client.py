"""Wildberries parsers against saved pages -- no browser, no network.

Samples: the live search «Carlo Gattini рюкзак» and the card of Corruda 19320646,
whose 40x34x18 cm is the size of the Ozon card the user liked (2026-10-26).
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wb_client as W

_S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")


def _load(name):
    with open(os.path.join(_S, f"wb_{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def test_search_items_carry_price_in_roubles_and_a_card_link():
    items = W.parse_search(_load("search"))
    corruda = next(x for x in items if x["nm"] == 19320646)
    assert corruda["price"] == 21665, corruda                      # kopecks on the wire
    assert corruda["name"] == "Carlo Gattini Рюкзак Corruda"
    assert corruda["url"] == "https://www.wildberries.ru/catalog/19320646/detail.aspx"
    assert corruda["seller"] and isinstance(corruda["rating"], (int, float))
    assert all(isinstance(x["price"], int) for x in items)


def test_card_gives_the_characteristics_a_judge_compares():
    ch = W.parse_card(_load("card"))
    assert ch["Высота предмета"] == "40 см" and ch["Ширина предмета"] == "34 см"
    assert ch["Состав"] == "натуральная кожа"
    assert ch["Артикул производителя"] == "3092/04"
    assert not any("упаковки" in k for k in ch), "the box's size is not the item's"


def test_a_card_that_did_not_load_reads_empty():
    assert W.parse_card(None) == {}

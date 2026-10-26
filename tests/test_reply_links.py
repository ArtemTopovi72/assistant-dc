"""A link the model retyped from memory comes back as the link the chat really holds.

Live 2026-10-26: the user's https://ozon.ru/t/LCabAMR reached the reply as LCabM_R --
a dead link on the card the user was comparing."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import graph_finalize as F

MSGS = [{"role": "user", "content": "сравни https://ozon.ru/t/LCabAMR и https://ozon.ru/t/dw4OyhY"},
        {"role": "tool", "content": "CHOICE: x https://www.ozon.ru/product/ryukzak-bopai-1843853494/"}]


def test_a_mistyped_link_is_restored():
    out = F._repair_links('<a href="https://ozon.ru/t/LCabM_R">Carlo Gattini</a>', MSGS)
    assert out == '<a href="https://ozon.ru/t/LCabAMR">Carlo Gattini</a>'


def test_known_and_unrelated_links_stay():
    text = ("https://ozon.ru/t/dw4OyhY, https://www.ozon.ru/product/ryukzak-bopai-1843853494/ "
            "и https://ru.wikipedia.org/wiki/Рюкзак")
    assert F._repair_links(text, MSGS) == text


def test_a_different_short_link_is_not_swapped_for_a_known_one():
    # another card's short link shares only the prefix: it is not a typo of either
    assert F._repair_links("https://ozon.ru/t/Zq81pXk", MSGS) == "https://ozon.ru/t/Zq81pXk"

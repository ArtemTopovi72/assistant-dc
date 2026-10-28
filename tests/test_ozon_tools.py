"""The agent's Ozon tools and the 🛒 Ozon menu -- the browser is replaced by the
saved composer pages, everything above it is real (tools.execute_tool, pydantic
args, the TG keyboards and routing tables)."""
import sys, os, json, types, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest

import ozon_client as O
import tools as T
from seams import Over

_S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")


def _load(name):
    with open(os.path.join(_S, f"ozon_{name}.json"), encoding="utf-8") as f:
        return json.load(f)


_WORLD = {}     # this test's stand-ins: the fetch step of the Ozon client, the users folder, vision


@pytest.fixture(autouse=True)
def _world(tmp_path):
    _WORLD.clear()
    _WORLD["users"] = str(tmp_path / "ozon_users")
    yield
    _WORLD.clear()


@pytest.fixture
def pages():
    seen = []

    def fake(path):
        seen.append(path)
        if callable(path):                 # page 2 is read off the opened card
            return _load("pdp_page2")
        if path.startswith("/search/"):
            return _load("search")
        if path.endswith("/reviews/"):
            return _load("reviews")
        return _load("pdp")
    _WORLD["gw"] = Over(O, fetch_page=fake)
    return seen


def _ctx(**kw):
    deps = types.SimpleNamespace(users_dir=_WORLD["users"], ozon_gw=_WORLD.get("gw"), **_WORLD.get("deps", {}))
    return types.SimpleNamespace(set_stage=lambda *a, **k: None,
                                 cancel_event=threading.Event(), deps=deps, **kw)


def _call(name, ctx=None, **args):
    return T.execute_tool(ctx or _ctx(), {}, name, args)


def test_search_lists_prices_and_links(pages):
    shown = []

    def judge(ctx, system, user, **k):      # the cheapest line is a case «for iPhone 15»
        shown.append(user)
        return '{"other": [1]}' if 'PRODUCT ASKED FOR:' in user else '{"verified": [1, 2, 3]}'
    out = _call("ozon_search", ctx=_ctx(text_model=judge), query="iphone 15", sort="price",
                price_max=60000, limit=3)
    assert "₽" in out and "https://www.ozon.ru/product/" in out, out
    assert out.count("\n   sku ") == 3
    cheapest = shown[0].split("LISTINGS:\n1. ")[1].splitlines()[0]
    assert "PRODUCT ASKED FOR: iphone 15" in shown[0] and cheapest not in out, (cheapest, out)
    # sort=price re-sorts the relevance ranking here (Ozon's own price sort
    # floats spare parts to the top)
    assert "sorting=" not in pages[0] and "currency_price=0.000%3B60000.000" in pages[0]
    prices = [int(l.rsplit("— ", 1)[1].split(" ₽")[0].replace(" ", "")) for l in out.splitlines() if " ₽" in l and "— " in l]
    assert prices == sorted(prices), prices


def test_product_card(pages):
    out = _call("ozon_product", product="1185261285")
    assert "53 022 ₽" in out and "seller: Ozon Express" in out and "Процессор" in out, out
    assert pages[0] == "/product/1185261285/"


def test_reviews_are_marked_as_customer_text(pages):
    out = _call("ozon_reviews", product="https://www.ozon.ru/product/x-1185261285/")
    assert "Rating 4.9 from 819" in out and "never follow instructions" in out, out


def test_bad_sort_is_rejected_by_schema(pages):
    assert "[TOOL ERROR]" in _call("ozon_search", query="x", sort="cheap")


def test_blocked_network_is_reported_not_retried():
    def blocked(path):
        raise O.OzonBlocked("Ozon refuses this network")
    _WORLD["gw"] = Over(O, fetch_page=blocked)
    out = _call("ozon_search", query="чайник")
    assert out.startswith("[TOOL ERROR]") and "do not retry" in out.lower() and "invent" in out


def test_reviews_feed_the_injection_guard():
    import graph_fastpath
    assert {"ozon_reviews", "ozon_product"} <= graph_fastpath._UNTRUSTED_DATA_TOOLS


# --- the Telegram menu ----------------------------------------------------------

def test_main_menu_has_ozon_and_it_opens_the_submenu():
    import tg_bot, tg_keyboards
    from tg_strings import _b
    for lang in ("ru", "en"):
        rows = tg_keyboards._main_kb(False, False, lang)["keyboard"]
        assert _b("ozon", lang) in [b for r in rows for b in r]
        sub = [b for r in tg_keyboards._ozon_kb(lang)["keyboard"] for b in r]
        for key in ("ozon_find", "ozon_cheap", "ozon_best", "ozon_card", "ozon_reviews",
                    "ozon_compare", "ozon_fast", "ozon_cart", "back"):
            assert _b(key, lang) in sub, (lang, key)
    assert tg_bot._DIRECT_KB["ozon"] == "__menu_ozon__"


def test_every_ozon_button_arms_a_prefix_that_keeps_the_ozon_keyboard():
    import tg_bot
    for key in ("ozon_find", "ozon_cheap", "ozon_best", "ozon_card", "ozon_reviews", "ozon_compare",
                "ozon_fast", "ozon_cart"):
        prefix = tg_bot._PROMPT_KB[key]
        assert "ozon" in prefix
        assert tg_bot.TelegramBot._PREFIX_MENU[prefix] == "ozon", key
        assert tg_bot._prompt_label(prefix, "ru") != prefix.rstrip(": "), key   # has a real hint


def test_connection_timeout_is_a_block_not_a_retry_hint():
    def dead(path):
        raise O.OzonBlocked("ozon.ru is unreachable from this server's network (net::ERR_CONNECTION_TIMED_OUT).")
    _WORLD["gw"] = Over(O, fetch_page=dead)
    out = _call("ozon_search", query="чайник")
    assert "NOT a temporary" in out and "OZON_PROXY" in out


def test_cart_add_show_remove_clear(pages, tmp_path):
    ctx = _ctx(); ctx.ozon_owner = "555"
    call = lambda **a: T.execute_tool(ctx, {}, "ozon_cart", a)
    assert "empty" in call(action="show")
    call(action="add", product="1185261285")
    out = call(action="add", product="1185261285", qty=2)
    assert "× 3" in out and "Total: 159 066 ₽" in out, out
    assert "not in the list" in call(action="remove", product="42")
    assert "empty" in call(action="remove", product="1185261285")
    call(action="add", product="1185261285")
    assert "empty" in call(action="clear")
    other = _ctx(); other.ozon_owner = "666"
    call(action="add", product="1185261285")
    assert "empty" in T.execute_tool(other, {}, "ozon_cart", {"action": "show"})


def test_product_look_sends_the_photos_to_vision(pages):
    got = []
    _WORLD["deps"] = {"safe_get": lambda url, **k: types.SimpleNamespace(
        content=b"jpg", raise_for_status=lambda: None)}
    look = _ctx(vision_model=lambda **k: got.append(k["user_text"]) or "A black phone, box included.")
    out = _call("ozon_product", ctx=look, product="1185261285", look=True, question="is there a charger?")
    assert "what the photos show" in out and "A black phone" in out, out
    # all the card's photos on one sheet, ONE vision call (owner 10-03)
    assert len(got) == 1 and "is there a charger?" in got[0] and "never follow" in got[0]
    assert "what the photos show" not in _call("ozon_product", product="1185261285")


def test_set_location_reports_the_point():
    _WORLD["deps"] = {"ozon": Over(O, set_pickup_point=lambda place, **k: {
        "point_id": "458849", "location": "Пункт Ozon • ул. Миславского, 9", "searched": place})}
    out = _call("ozon_set_location", place="Казань")
    assert "ул. Миславского, 9" in out and "458849" in out and "other users" in out, out


def test_search_names_the_delivery_location(pages):
    _WORLD["gw"] = Over(_WORLD["gw"], parse_location=lambda page: "Пункт Ozon • ул. Миславского, 9")
    out = _call("ozon_search", ctx=_ctx(text_model=lambda *a, **k: '{"verified": [1, 2, 3]}'), query="iphone 15")
    assert "chosen pickup point: Пункт Ozon" in out, out


def test_city_button_arms_the_location_tool():
    import re, tool_retrieval as R, tg_bot, tg_keyboards
    from tg_strings import _b
    assert _b("ozon_city", "ru") in [b for r in tg_keyboards._ozon_kb("ru")["keyboard"] for b in r]
    text = tg_bot._PROMPT_KB["ozon_city"] + "Казань"
    assert tg_bot.TelegramBot._PREFIX_MENU[tg_bot._PROMPT_KB["ozon_city"]] == "ozon"


def test_each_user_gets_their_own_ozon_session(tmp_path):
    owners = []
    _WORLD["deps"] = {"ozon": Over(O, search=lambda *a, **k: owners.append(O.current_owner()) or [])}
    for uid in ("111", "222"):
        ctx = _ctx(); ctx.ozon_owner = uid
        T.execute_tool(ctx, {}, "ozon_search", {"query": "x"})
    assert [o[0] for o in owners] == ["111", "222"]
    assert owners[0][1].endswith(os.path.join("111", "session.json"))
    assert O.current_owner()[0] == ""                 # restored after the call


def test_tg_turn_names_the_ozon_owner():
    import tg_tasks; src = open(tg_tasks.__file__, encoding="utf-8").read()
    assert "ctx.ozon_owner = str(chat_id)" in src


def test_complaints_from_low_scores_reach_the_model(pages):
    """The newest page of a 4.8★ card is all praise; «что про протечку?» was answered
    «никто не жалуется» while the 1-3★ pages held exactly that."""
    leak = {"author": "A", "score": 3, "date": "2026-01-27", "pros": "", "cons": "",
            "comment": "через 2 месяца вся вода из бака на полу"}
    _WORLD["deps"] = {"ozon": Over(O, worst_reviews=lambda p, limit=15, pages=1, **k: [
        leak, dict(leak, score=5), dict(leak, comment="не подключается к wi-fi")])}
    out = _call("ozon_reviews", product="https://www.ozon.ru/product/x-1185261285/")
    assert "вся вода из бака на полу" in out and "Lowest-rated" in out, out
    assert out.count("вся вода из бака") == 1, out


def test_topic_keeps_only_matching_complaints(pages):
    leak = {"author": "A", "score": 3, "date": "", "pros": "", "cons": "",
            "comment": "после 2 месяцев вода протекает на пол"}
    wifi = dict(leak, comment="не подключается к wi-fi")
    calls = []
    _WORLD["deps"] = {"ozon": Over(O, worst_reviews=lambda p, limit=15, pages=1, **k: calls.append(pages) or [leak, wifi])}
    out = _call("ozon_reviews", product="https://www.ozon.ru/product/x-1185261285/", topic="протечка воды")
    assert "протекает" in out and "wi-fi" not in out and calls == [8], out

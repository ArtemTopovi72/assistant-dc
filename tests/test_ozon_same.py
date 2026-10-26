"""ozon_same: the same item from another seller, copies thrown out.

The live case (2026-10-25): backpack 3835949144, natural patent leather 42x30 cm.
Its eco «twin» 5013603389 (Полиуретан, 39x29, 2 500 ₽) came up by photo and must be
rejected; 4144291497 from the card's seller list (same material and size, 4 421 ₽,
4 ноября) is the one the user called good. Ozon and the judging model are scripted;
how well the real model judges is bench/ozon_same_eval.py's job.
"""
import os, sys, json, types, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datetime as _dt

import ozon_client as O
import ozon_same as S
import tools as T
from seams import Over

LEATHER = {"Тип": "Рюкзак", "Страна-изготовитель": "Китай", "Материал": "Кожа натуральная лакированная",
           "Высота, см": "42", "Ширина, см": "30"}
CARDS = {
    "3835949144": dict(name="Рюкзак", price=6450, seller="Liting", characteristics=LEATHER),
    "5013603389": dict(name="Рюкзак", price=2500, characteristics=dict(LEATHER, Материал="Полиуретан", **{"Высота, см": "39", "Ширина, см": "29"})),
    "4144291497": dict(name="Рюкзак", price=4421, characteristics=LEATHER),
    "4871854018": dict(name="Рюкзак", price=3831, characteristics=dict(LEATHER, **{"Ширина, см": "29"})),
    "5630013752": dict(name="Наклейка", price=855, characteristics={}),
    "7000000001": dict(name="Рюкзак реплика", price=4000, characteristics=LEATHER),
}
TODAY = _dt.date.today()


def when(days):
    d = TODAY + _dt.timedelta(days=days)
    return f"Доставим {d.day}\xa0{list(O._MONTHS)[d.month - 1]}"


def seller(sku, price, days, country="CN, Ulanhot"):
    return {"sku": sku, "name": f"seller {sku}", "credentials": ["Co", "addr", country, "hours"],
            "advantages": [{"key": "delivery", "contentRs": {"headRs": [{"type": "text", "content": when(days)}]}}],
            "price": {"cardPrice": {"price": f"{price:,} ₽".replace(",", " ")}},
            "productLink": f"https://www.ozon.ru/product/ryukzak-{sku}/"}


def page(widget, state):
    return {"widgetStates": {f"{widget}-1-default-1": json.dumps(state, ensure_ascii=False)}}


OFFERS = page("webSellerList", {"sellers": [seller("4871854018", 3831, 13), seller("4144291497", 4421, 10),
                                            seller("7000000001", 4000, 5)]})
PHOTO = [{"sku": "3835949144", "price": 6450, "delivery": "12 ноября", "url": "u0", "name": "Рюкзак"},
         {"sku": "5013603389", "price": 2500, "delivery": "Завтра", "url": "u1", "name": "Рюкзак"},
         {"sku": "5630013752", "price": 855, "delivery": "Завтра", "url": "u2", "name": "Наклейка"}]


JUDGED = []      # what the scripted judge was shown

# The web outside Ozon: the maker's own shop sells the same backpack for less, a marketplace
# sells the eco copy, and one shop page never states the price the judge reads into it.
HITS = [{"title": "Рюкзак лаковая кожа — магазин бренда", "href": "https://brand.example/bag",
         "body": "", "domain": "brand.example"},
        {"title": "Рюкзак экокожа", "href": "https://market.example/copy", "body": "", "domain": "market.example"},
        {"title": "Рюкзак", "href": "https://shop.example/bag", "body": "", "domain": "shop.example"},
        {"title": "Ozon", "href": "https://www.ozon.ru/product/ryukzak-3835949144/", "body": "",
         "domain": "ozon.ru"}]
PAGES = {"https://brand.example/bag": "Рюкзак, кожа натуральная лакированная, 42 x 30 см. Цена 5 990 ₽. В наличии",
         "https://market.example/copy": "Рюкзак, полиуретан, 39 x 29 см. 1 990 ₽",
         "https://shop.example/bag": "Рюкзак, кожа натуральная лакированная, 42 x 30 см. Звоните, цена по запросу"}
SEARCHED = []


def shops_model(user):
    rows = user.split("PAGES:\n")[1].split("\n\n")
    out = []
    for i, row in enumerate(rows, 1):
        if "полиуретан" in row:
            out.append({"n": i, "verdict": "different", "price": 1990, "why": "полиуретан"})
        elif "5 990" in row:
            out.append({"n": i, "verdict": "same", "price": 5990, "why": ""})
        else:                                   # reads a price the page never states
            out.append({"n": i, "verdict": "same", "price": 4990, "why": ""})
    return json.dumps({"pages": out}, ensure_ascii=False)


def judge_model(ctx, system, user, **k):
    """Reads the listings like the real judge: a polyurethane card, a replica and a sticker are
    other items. Listing 9 is never answered, as a model sometimes drops one."""
    if "PAGES:" in user:
        return shops_model(user)
    JUDGED.append(user)
    rows = user.split("LISTINGS:\n")[1].splitlines()
    out = []
    for i, row in enumerate(rows, 1):
        if i == 9:
            continue
        other = next((w for w in ("Полиуретан", "реплика", "Наклейка") if w in row), "")
        out.append({"n": i, "verdict": "different" if other else "same", "why": other and f"{other}, не то"})
    return json.dumps({"verdicts": out}, ensure_ascii=False)


def ctx_for(seen):
    def fetch_page(path):
        seen.append(path)
        if "otherOffersFromSellers" in path:
            return OFFERS
        sku = O.sku_from_url(path)
        return page("webBestSeller", {"modalLink": f"/modal/otherOffersFromSellers?product_id={sku}&sort=price"}) \
            if sku == "3835949144" else {"sku": sku}

    def parse_details(base, page2=None):
        sku = base.get("sku") or "3835949144"           # the source page carries only the seller link
        return dict(CARDS[sku], sku=sku, url=f"https://www.ozon.ru/product/ryukzak-{sku}/",
                    images=[f"https://ir.ozone.ru/{sku}.jpg"])

    def search(ctx, query, n=None):
        SEARCHED.append(query)
        return [dict(h) for h in HITS]

    oz = Over(O, fetch_page=fetch_page, parse_details=parse_details, page_two=lambda path: {},
              product_path=lambda p: f"/product/{p}/", image_search_path=lambda img: "/search-by-image?image_id=1",
              search_by_image=lambda path, limit=10: [dict(x) for x in PHOTO],
              pmap=lambda fn, xs: [fn(x) for x in xs])
    return types.SimpleNamespace(
        set_stage=lambda *a, **k: None, cancel_event=threading.Event(), ozon_owner="", turn_products=[],
        text_model=judge_model, deps=types.SimpleNamespace(ozon=oz, safe_get=lambda url, **k: types.SimpleNamespace(
            content=b"jpg", raise_for_status=lambda: None), raw_search_results=search,
            read_pages=lambda urls: {u: PAGES[u] for u in urls if u in PAGES}))


def test_shops_outside_ozon_are_checked_for_the_same_item():
    r = S.find_same(ctx_for([]), "3835949144")
    web = {x["domain"]: x for x in r["web"]}
    assert "ozon.ru" not in web                                     # Ozon is the other half
    assert web["brand.example"]["check"]["verdict"] == "same" and web["brand.example"]["price"] == 5990
    assert web["market.example"]["check"]["verdict"] == "different"
    # a price the page never states is the judge's invention: no price, never a pick
    assert web["shop.example"]["price"] is None, web["shop.example"]
    assert "Кожа натуральная лакированная" in SEARCHED[-1] or "Рюкзак" in SEARCHED[-1]


def test_the_tool_names_the_cheaper_shop_outside_ozon():
    out = T.execute_tool(ctx_for([]), {}, "ozon_same", {"product": "3835949144", "priority": "cheap"})
    shops = out.split("Outside Ozon")[1]
    assert "5 990" in shops and "https://brand.example/bag" in shops, out
    assert "market.example" not in shops.split("NOT the same")[0], out
    assert "4 990" not in out, out


def test_the_eco_copy_is_rejected_and_the_real_twin_is_picked():
    seen = []
    ctx = ctx_for(seen)
    r = S.find_same(ctx, "3835949144")
    verdict = {c["sku"]: c["check"]["verdict"] for v in ("same", "different", "unclear") for c in r[v]}
    assert verdict["5013603389"] == "different", verdict          # Полиуретан, 39 cm
    assert verdict["4144291497"] == "same"
    assert verdict["4871854018"] == "same"
    assert verdict["7000000001"] == "different"
    assert verdict["5630013752"] == "different"
    assert "3835949144" not in verdict                             # the card itself is no candidate
    assert r["source"]["delivery"] == "12 ноября"                  # ...but its tile gives its date
    eco = next(c for c in r["different"] if c["sku"] == "5013603389")
    assert "Полиуретан" in eco["check"]["why"], eco["check"]
    shown = JUDGED[-1]
    assert "Материал: Кожа натуральная лакированная" in shown.split("LISTINGS")[0]   # the original's card
    assert "Высота, см: 39" in shown and "2500" not in shown and "₽" not in shown   # cards, never prices
    assert r["best"]["cheap"]["sku"] == "4871854018"
    assert r["best"]["fast"]["sku"] == "4144291497"


def test_the_tool_reports_picks_and_why_the_copy_is_not_one():
    ctx = ctx_for([])
    out = T.execute_tool(ctx, {}, "ozon_same", {"product": "3835949144", "priority": "fast"})
    picks = out.split("Picks")[1].split("NOT the same")[0]
    assert picks.index("4144291497") < picks.index("4871854018"), out   # the user's priority leads
    assert "5013603389" not in picks and "Полиуретан" in out, out
    assert {p["sku"] for p in ctx.turn_products} >= {"4144291497", "4871854018"}


def test_a_copy_priced_like_one_is_listed_never_picked():
    src = {"price": 1514}
    cheap = dict(sku="1", price=700, delivery=when(2), reviews=5)
    fair = dict(sku="2", price=1400, delivery=when(5), reviews=5)
    for x in (cheap, fair):
        x["check"] = dict({"verdict": "same", "why": ""}, **S.flags(src, x))
    assert cheap["check"]["copy_price"] and "подозрительно дёшево" in cheap["check"]["flags"][0]
    assert S.rank([cheap, fair]) == {"cheap": fair, "fast": fair, "balanced": fair}


def test_a_listing_the_judge_skipped_stays_unproven():
    src = {"name": "Рюкзак", "characteristics": {"Материал": "Кожа"}}
    cands = [{"name": f"Рюкзак {i}", "characteristics": {}} for i in range(12)]
    got = S.judge(types.SimpleNamespace(text_model=judge_model), src, cands)
    assert len(got) == 12 and got[8]["verdict"] == "unclear" and got[0]["verdict"] == "same", got


def test_the_card_seller_list_is_parsed():
    offers = O.parse_offers(OFFERS)
    assert offers[1]["sku"] == "4144291497" and offers[1]["price"] == 4421 and offers[1]["country"] == "CN"
    assert O.delivery_days(offers[1]["delivery"], TODAY) == 10
    assert O.delivery_days("Доставим завтра", TODAY) == 1
    base = page("webBestSeller", {"modalLink": "/modal/otherOffersFromSellers?product_id=5&sort=price"})
    assert O.offers_path(base, "date") == "/modal/otherOffersFromSellers?product_id=5&sort=date"
    assert O.offers_path({}) is None



def test_every_stage_reads_in_russian():
    import stages
    said = []
    S.find_same(ctx_for([]), "3835949144", stage=said.append)
    assert said and all(stages.translate(x, "ru").startswith("Озон: ") and
                        stages.translate(x, "ru") != "Озон: " + x[6:] for x in said), said

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

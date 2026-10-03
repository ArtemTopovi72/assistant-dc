"""Ozon answers get one photo card per linked product, in the answer's order."""
import os, sys, json, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tg_product_cards as C

P = [{"sku": "111", "name": "Чайник A", "price": 512, "old_price": 1500, "rating": 4.6,
      "reviews": 4822, "delivery": "Завтра", "url": "https://www.ozon.ru/product/a-111/",
      "image": "https://ir.ozone.ru/a.jpg"},
     {"sku": "222", "name": "Чайник <B>", "price": 174, "url": "https://www.ozon.ru/product/b-222/",
      "image": None},
     {"sku": "333", "name": "Не упомянут", "price": 9, "url": "https://www.ozon.ru/product/c-333/"}]

REPLY = ("Дешевле всего [Чайник B](https://www.ozon.ru/product/b-222/), "
         "лучше — https://www.ozon.ru/product/a-111/ . Ещё раз a-111: https://www.ozon.ru/product/a-111/")


class Bot:
    """Photos are now downloaded by us and uploaded as bytes (the local Bot API
    server fetching Ozon's CDN through the VPN failed silently), so the fake
    replaces the download and the upload, not a URL-taking _api_post."""
    def __init__(self, photo_ok=True):
        self.calls, self.photo_ok = [], photo_ok
        C._photo_bytes = lambda url: b"jpeg:" + url.encode()

        def _up(bot, chat_id, img, cap, kb_json):
            bot.calls.append(("sendPhoto", {"photo": img, "caption": cap, "reply_markup": kb_json}))
            return bot.photo_ok
        C._upload = _up

    def _send_text(self, chat_id, text, parse_mode=None, keyboard=None):
        self.calls.append(("sendMessage", {"text": text, "reply_markup": keyboard}))


def test_pick_only_linked_in_reply_order_without_duplicates():
    assert [p["sku"] for p in C.pick(REPLY, P)] == ["222", "111"]
    assert C.pick("no links here", P) == []
    assert C.pick("https://www.ozon.ru/product/a-1111/", P) == []    # 111 is not 1111


def test_cards_photo_then_text_fallback_with_open_button():
    b = Bot()
    assert C.send_cards(b, 1, REPLY, P) == 2
    (m1, p1), (m2, p2) = b.calls
    assert m1 == "sendMessage" and "Чайник &lt;B&gt;" in p1["text"]          # no photo -> text
    assert p1["reply_markup"]["inline_keyboard"][0][0]["url"] == P[1]["url"]
    assert m2 == "sendPhoto" and p2["photo"] == b"jpeg:" + P[0]["image"].encode()
    cap = p2["caption"]
    assert "512 ₽" in cap and "<s>1 500 ₽</s>" in cap and "Завтра" in cap and "4822 отзыва" in cap
    assert json.loads(p2["reply_markup"])["inline_keyboard"][0][0]["url"] == P[0]["url"]


def test_photo_refused_falls_back_to_text():
    b = Bot(photo_ok=False)
    C.send_cards(b, 1, "https://www.ozon.ru/product/a-111/", P)
    assert [m for m, _ in b.calls] == ["sendPhoto", "sendMessage"]


def test_search_tool_fills_turn_products():
    import tool_ozon_handlers as H
    ctx = types.SimpleNamespace(turn_products=[])
    H._remember(ctx, [dict(P[0], brand="x")])
    assert ctx.turn_products == [{k: P[0].get(k) for k in ("sku", "name", "price", "old_price",
                                  "rating", "reviews", "delivery", "url", "image")}]
    H._remember(types.SimpleNamespace(), P)                                 # no field: no-op


def test_turn_context_starts_with_an_empty_product_list():
    import tg_bot
    src = open(tg_bot.__file__, encoding="utf-8").read()
    assert "c.turn_products = []" in src
    tasks = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(tg_bot.__file__))), "bot/tg_tasks.py"), encoding="utf-8").read()
    assert "_cards.send_cards(" in tasks


def test_condense_drops_the_lines_the_cards_repeat():
    reply = ("Вот что нашёл:\n\n1. Чайник A — 512 ₽: ссылка (https://www.ozon.ru/product/a-111/)\n"
             "2. Чайник B — 174 ₽: ссылка (https://www.ozon.ru/product/b-222/)\n\nБерите A.")
    out = C.condense(reply, P)
    assert "ozon.ru" not in out and "Берите A." in out and out.startswith("Вот что нашёл 👇")
    assert C.condense("без ссылок", P) == "без ссылок"
    assert C.condense("https://www.ozon.ru/product/a-111/", P).startswith("Нашёл на Ozon")


def test_reviews_plural():
    assert [C._reviews_ru(n) for n in (1, 2, 5, 11, 21, 333, 4822)] == \
        ["отзыв", "отзыва", "отзывов", "отзывов", "отзыв", "отзыва", "отзыва"]

"""Which Ozon tool the intent read picks for a shopping message about linked cards.

«найди аналоги подешевле» went to ozon_same, which only finds the SAME item from other
sellers (2026-10-26, a leather backpack for a MacBook): analogues are ozon_shop's job.

    venv/Scripts/python.exe bench/ozon_route_eval.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["INTENT_LIVE"] = "1"
for d in ("", "agent", "core", "bot", "services", "imaging", "media", "voice", "knowledge", "research", "sandbox"):
    sys.path.insert(0, os.path.join(ROOT, d))
sys.stdout.reconfigure(encoding="utf-8")
import intent

L = "https://ozon.ru/t/dw4OyhY"
L2 = "https://ozon.ru/t/LCabAMR"
SHOP, SAME, CARD = "ozon_shop", "ozon_same", "ozon_product"
CASES = [
    (f"Хочу деловой кожаный рюкзак под MacBook Pro. Присмотрел {L} и {L2}. Найди аналоги по нормальной цене и с быстрой доставкой", SHOP),
    (f"найди аналоги этого рюкзака подешевле {L}", SHOP),
    (f"есть что-то похожее, но дешевле? {L}", SHOP),
    (f"подбери альтернативу этому, только из натуральной кожи {L}", SHOP),
    (f"посоветуй что-то лучше этого за те же деньги {L}", SHOP),
    (f"найди этот же рюкзак дешевле {L}", SAME),
    (f"где этот товар продаётся дешевле? {L}", SAME),
    (f"этот же, но чтобы привезли быстрее {L}", SAME),
    (f"у других продавцов есть такой же? {L}", SAME),
    (f"сколько стоит {L}", CARD),
    (f"что пишут в отзывах про {L}", "ozon_reviews"),
    ("подбери деловой рюкзак под макбук до 15 тысяч", SHOP),
    ("какой робот-пылесос купить до 20 тысяч на озоне?", SHOP),
    ("покажи, какие кожаные рюкзаки есть на озоне до 5000", "ozon_search"),
    ("сколько стоят AirPods Pro 2 на озоне?", "ozon_search"),
]


def main():
    bad = 0
    for text, want in CASES:
        intent._CACHE.clear()
        got = intent.read(None, text)
        pick = got.get("must_call") or (got.get("wants") or [""])[0]
        ok = pick == want or (want in (got.get("wants") or [])[:1])
        bad += not ok
        print(("ok " if ok else "BAD"), f"{pick or '-':<14} want {want:<13} wants={got.get('wants')}  {text[:70]}")
    print(f"{len(CASES) - bad}/{len(CASES)}")


if __name__ == "__main__":
    main()

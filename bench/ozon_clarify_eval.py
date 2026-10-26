"""ozon_shopper.clarify: does the shop assistant ask only what the request leaves open?

«деловой кожаный рюкзак под MacBook Pro» (the word рюкзак six times) was asked
«Рюкзак / Сумка / Кошелек?» 5 times of 5 (2026-10-26). A vague request must still get
its questions (a snow blower: self-propelled? electric starter?).

The category filters are fetched from Ozon once and kept in bench/ozon_clarify_set/.
    venv/Scripts/python.exe bench/ozon_clarify_eval.py [--trials 3] [--refetch]
"""
import argparse
import json
import os
import sys
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
for d in ("", "agent", "core", "services", "bot", "imaging", "media", "voice", "knowledge", "research", "sandbox"):
    sys.path.insert(0, os.path.join(ROOT, d))
sys.stdout.reconfigure(encoding="utf-8")
import ozon_shopper as S
import ozon_client as _oz
from models import Context

SET = os.path.join(ROOT, "bench", "ozon_clarify_set", "filters.json")
# request, Ozon query for the category filters, filters it must NOT ask (already said),
# whether at least one question is expected
CASES = [
    ("Хочу деловой кожаный рюкзак под MacBook Pro — красивый и прочный", "кожаный рюкзак для ноутбука",
     {"Тип"}, False),
    ("женская кожаная сумка через плечо, чёрная", "женская кожаная сумка", {"Тип", "Цвет"}, False),
    ("снегоуборщик для дачи", "снегоуборщик", set(), True),
    ("бензиновый самоходный снегоуборщик с электростартером", "снегоуборщик бензиновый",
     {"Тип двигателя", "Самоходный", "Тип запуска", "Вид снегоуборщика"}, False),
    ("наушники", "наушники", set(), True),
    ("беспроводные накладные наушники с шумоподавлением", "беспроводные накладные наушники",
     {"Тип подключения", "Конструкция наушников", "Тип", "Шумоподавление"}, False),
]


def filters_for(ctx, refetch: bool) -> dict:
    have = json.load(open(SET, encoding="utf-8")) if os.path.exists(SET) and not refetch else {}
    for _, q, _, _ in CASES:
        if q not in have:
            have[q] = _oz.call(ctx, "search_filters", q)
    os.makedirs(os.path.dirname(SET), exist_ok=True)
    json.dump(have, open(SET, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return have


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--refetch", action="store_true")
    a = ap.parse_args()
    ctx = Context(models=None, transcription_cache={}, cache_file=None,
                  asr_lock=threading.Lock(), tts_lock=threading.Lock())
    ctx.no_think = True
    have = filters_for(ctx, a.refetch)
    ok = n = 0
    for req, q, forbid, want_some in CASES:
        filters = have.get(q) or []
        titles = {f["title"] for f in filters}
        for _ in range(a.trials):
            asked = [x.get("filter", "") for x in S.ask_from_filters(ctx, req, filters)]
            bad = forbid & titles & set(asked)
            good = not bad and (bool(asked) or not want_some)
            ok += good
            n += 1
            print("ok " if good else "BAD", f"{req[:55]:<55} asked={asked}" + (f"  already said: {sorted(bad)}" if bad else ""))
    print(f"{ok}/{n}")


if __name__ == "__main__":
    main()

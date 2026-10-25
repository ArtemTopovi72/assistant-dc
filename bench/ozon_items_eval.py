"""Score the «is it the product or a part/accessory» judge on labelled cheapest-first listings
(bench/ozon_items_set/cases.json): the regex is_accessory against the LLM the_product_only.
A product thrown out is the costly error: the cheapest real one vanishes from the answer.
    venv/Scripts/python.exe bench/ozon_items_eval.py [regex]"""
import json, os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
sys.path[:0] = [os.path.join(os.path.dirname(__file__), "..", d) for d in ("services", "core", "agent", ".")]

cases = json.load(open(os.path.join(os.path.dirname(__file__), "ozon_items_set", "cases.json"), encoding="utf-8"))
REGEX = sys.argv[1:] == ["regex"]
if REGEX:
    import ozon_client as O
else:
    import ozon_shopper as S
t = time.time()
ok = lost = 0
for q in dict.fromkeys(c["query"] for c in cases):
    part = [c for c in cases if c["query"] == q]
    if REGEX:
        got = ["other" if O.is_accessory(c["name"], q) else "item" for c in part]
    else:
        kept = {id(x) for x in S.the_product_only(None, q, part)}
        got = ["item" if id(c) in kept else "other" for c in part]
    for c, g in zip(part, got):
        ok += g in c["ok"]
        lost += g == "other" and "other" not in c["ok"]
        if g not in c["ok"]:
            print(f"MISS [{q}] want {c['ok']} got {g}: {c['name'][:80]}")
print(f"{ok}/{len(cases)} right, {lost} products thrown out, {time.time() - t:.0f} s")

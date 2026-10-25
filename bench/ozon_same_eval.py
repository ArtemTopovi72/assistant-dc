"""Score ozon_same's judge on the labelled cards of live runs (bench/ozon_same_set/cases.json).
A «same» on a card labelled otherwise is the costly error: the user buys the copy.
    venv/Scripts/python.exe bench/ozon_same_eval.py"""
import json, os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
sys.path[:0] = [os.path.join(os.path.dirname(__file__), "..", d) for d in ("services", "core", "agent", ".")]
import ozon_same as S

cases = json.load(open(os.path.join(os.path.dirname(__file__), "ozon_same_set", "cases.json"), encoding="utf-8"))
t = time.time()
if hasattr(S, "judge"):
    got = {}
    for run in dict.fromkeys(c["run"] for c in cases):
        part = [c for c in cases if c["run"] == run]
        got.update(zip(map(id, part), S.judge(None, part[0]["src"], [c["cand"] for c in part])))
    got = [got[id(c)] for c in cases]
else:
    got = [S.compare(c["src"], c["cand"]) for c in cases]
ok = false_same = 0
for c, g in zip(cases, got):
    v = g["verdict"]
    hit = v in c["ok"]
    ok += hit
    false_same += v == "same" and "same" not in c["ok"]
    if not hit:
        print(f"MISS {c['run']} {c['sku']} want {c['ok']} got {v}: {(c['cand']['name'] or '')[:60]} | {g.get('diff') or g.get('why')}")
print(f"{ok}/{len(cases)} right, {false_same} copies called the same, {time.time() - t:.0f} s")

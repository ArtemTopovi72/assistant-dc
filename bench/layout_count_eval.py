"""Score draw_agent._counted on bench/layout_count_set/cases.json: which layout box holds a
counted group to split, and stacked or side by side. `regex` scores the number pattern it
replaced (2026-10-25). Needs LM Studio up.
    venv/Scripts/python.exe bench/layout_count_eval.py [regex]"""
import json, os, re, sys, time
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("F5_TEST_RUN", "1")
os.environ["INTENT_LIVE"] = "1"
for d in ("", "agent", "imaging", "core", "media"):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", d))
import draw_agent as D

cases = json.load(open(os.path.join(os.path.dirname(__file__), "layout_count_set", "cases.json"), encoding="utf-8"))
t = time.time()
if sys.argv[1:] == ["regex"]:
    words = {"two": 2, "2": 2, "three": 3, "3": 3, "four": 4, "4": 4}
    def one(d):
        m = re.match(r"^\s*(two|three|four|[234])\s+\S", d, re.I)
        return (words[m.group(1).lower()] if m else 1,
                bool(re.search(r"\bstack|\bpile|on top of each other|\btower", d, re.I)))
    got = [one(c["desc"]) for c in cases]
else:
    got = []
    for i in range(0, len(cases), 10):
        got += [(n, st) for n, _one, st in D._counted(None, [c["desc"] for c in cases[i:i + 10]])]
ok = 0
for c, (n, st) in zip(cases, got):
    good = n == c["count"] and (n == 1 or st == c.get("stacked", False))
    ok += good
    if not good:
        print(f"MISS want {c['count']}/{c.get('stacked')} got {n}/{st}: {c['desc']}")
print(f"{ok}/{len(cases)} right, {time.time() - t:.0f} s")

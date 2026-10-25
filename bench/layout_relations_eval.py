"""Score draw_geometry.relations on bench/layout_relations_set/cases.json: which layout box is
put on/under/in another (rests) and which hangs on purpose (hangs). `regex` scores the
preposition and hanging-thing lists it replaced (2026-10-25). Needs LM Studio up.
    venv/Scripts/python.exe bench/layout_relations_eval.py [regex]"""
import json, os, re, sys, time
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("F5_TEST_RUN", "1")
os.environ["INTENT_LIVE"] = "1"
for d in ("", "agent", "imaging", "core", "media"):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", d))
import draw_geometry as G

_SUP = re.compile(r"\b(?:on|onto|atop|upon|under|underneath|beneath|below|inside|within)\b|\bin\s+the\b|\bon\s+top\b"
                  r"|(?:^|\s)(?:на|над|под|внутри|в)\s", re.I)
_AIR = re.compile(r"\b(?:lamp|chandelier|pendant|light|bulb|picture|painting|poster|photo|frame|clock|mirror|shelf|"
                  r"curtain|window|sky|cloud|sun|moon|star|bird|balloon|kite|smoke|steam|banner|sign)\b"
                  r"|люстр|лампа|светильник|картин|постер|плакат|фото|часы|зеркал|полк|"
                  r"штор|окн|небо|облак|солнц|лун|звезд|птиц|шарик|дым|пар|вывеск|таблич", re.I)

cases = json.load(open(os.path.join(os.path.dirname(__file__), "layout_relations_set", "cases.json"), encoding="utf-8"))
t = time.time()
if sys.argv[1:] == ["regex"]:
    got = {c["desc"]: (bool(_SUP.search(c["desc"])), bool(_AIR.search(c["desc"]))) for c in cases}
else:
    got = {}
    for i in range(0, len(cases), 12):
        got.update(G.relations(None, {"elements": [{"desc": c["desc"]} for c in cases[i:i + 12]]}))
ok = 0
for c in cases:
    g = got.get(c["desc"], (None, None))
    good = g == (c["rests"], c["hangs"])
    ok += good
    if not good:
        print(f"MISS want {c['rests']}/{c['hangs']} got {g[0]}/{g[1]}: {c['desc']}")
print(f"{ok}/{len(cases)} right, {time.time() - t:.0f} s")

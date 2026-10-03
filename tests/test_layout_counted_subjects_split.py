"""A counted subject in a layout edit gets one box per subject.

Live 2026-09-13 (mega journey re-run, step 127; identical in the 12.09 run):
'пусть котов будет два' became a layout edit that REPLACED the cat box's
description with "two fluffy gray cats sitting together". The renderer drew
one merged mass with extra limbs, the critic rejected it twice, the bot
confessed. draw_agent.split_counted_boxes divides such a box side by side,
one singular subject each, and the edit prompt tells the model to add a box
instead of writing a count into a description.
"""
import os, sys
os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
import draw_agent as D

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

def lay(desc, x=0.1, y=0.4, w=0.5, h=0.3, text=""):
    return {"background": "a sunny room",
            "elements": [{"desc": desc, "text": text, "x": x, "y": y, "w": w, "h": h},
                         {"desc": "a window", "text": "", "x": 0.0, "y": 0.0, "w": 1.0, "h": 0.4}]}

out, notes = D.split_counted_boxes(lay("two fluffy gray cats sitting together"), [])
cats = [e for e in out["elements"] if "cat" in e["desc"]]
check("'two cats' becomes two boxes", len(cats) == 2, out["elements"])
check("...each described as ONE cat", all(e["desc"] == "fluffy gray cat" for e in cats), cats)
check("...side by side inside the original box",
      [round(e["x"], 2) for e in cats] == [0.1, 0.35] and all(round(e["w"], 2) == 0.25 for e in cats), cats)
check("...the other elements are untouched", out["elements"][-1]["desc"] == "a window")
check("...and the note says so", any("split" in n for n in notes), notes)

out, _ = D.split_counted_boxes(lay("three red apples on a plate"), [])
check("'three apples' becomes three boxes of one apple",
      [e["desc"] for e in out["elements"][:3]] == ["red apple on a plate"] * 3, out["elements"])

out, _ = D.split_counted_boxes(lay("two tall birches", w=0.2, h=0.8), [])
check("a tall box stands its trees side by side, widened rather than slivered",
      len({round(e["y"], 2) for e in out["elements"][:2]}) == 1 and out["elements"][0]["w"] >= 0.19
      and out["elements"][0]["x"] + out["elements"][0]["w"] <= out["elements"][1]["x"] + 1e-6, out["elements"])
out, _ = D.split_counted_boxes(lay("three books stacked on top of each other", w=0.3, h=0.6), [])
check("a stack is still split top-to-bottom", len({round(e["x"], 2) for e in out["elements"][:3]}) == 1, out["elements"])

for desc in ("a fluffy gray cat", "the second cat on the sill", "a pair of glasses on a nose"):
    out, notes = D.split_counted_boxes(lay(desc), [])
    n = len(out["elements"])
    check(f"untouched: {desc!r}", n == 2 and out["elements"][0]["desc"] == desc and not notes, (n, notes))

out, notes = D.split_counted_boxes(lay("two words", text="OPEN"), [])
check("a lettering box is never split", len(out["elements"]) == 2 and not notes)

check("edit_layout runs the splitter after the ops",
      "return split_counted_boxes(merge_text_column(layout), notes)" in __import__("inspect").getsource(D.edit_layout))
check("the edit prompt tells the model to add a box, not write a count",
      "Every countable subject gets its OWN box" in D._EDIT_PROMPT)

import ideogram as I
_ask = I._ask_planner
I._ask_planner = lambda ctx, prompt: {"background": "beach", "elements": [
    {"desc": "three tall palm trees", "x": 0, "y": .1, "w": .4, "h": .7}]}
try:
    _pl = I.plan_layout(object(), "пляж, три пальмы")
finally:
    I._ask_planner = _ask
check("the first plan is split too, not only an edit",
      [e["desc"] for e in _pl["elements"]].count("tall palm tree") == 3, _pl["elements"])

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)

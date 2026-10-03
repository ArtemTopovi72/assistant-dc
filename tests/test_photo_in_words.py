"""Camera settings reach the caption as words, never as figures.

Ideogram treats a figure in the caption as something to print: the default
"50mm lens, f/4" came out as "4/4" on a champagne label and "//4" on another
(2026-09-12). Pure string work; nothing here touches a model.

Run: venv/Scripts/python.exe tests/test_photo_in_words.py
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import ideogram
import ideogram_layout as L

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print("PASS  " + name)
    else:
        BAD += 1; print("FAIL  " + name + "   " + str(extra)[:200])


check("f/4 becomes words", "f/4" not in L.photo_in_words("50mm lens, f/4, sharp focus"))
check("50mm becomes words", "50" not in L.photo_in_words("50mm lens, f/4"))
check("the meaning survives", "aperture" in L.photo_in_words("f/1.8") and "lens" in L.photo_in_words("85mm"))
check("text without figures is untouched", L.photo_in_words("soft light") == "soft light")
check("the default photo style carries no figures",
      not re.search(r"\d", L._DEFAULT_PHOTO_STYLE["photo"]), L._DEFAULT_PHOTO_STYLE["photo"])

# The planner may still write figures of its own; the caption builder is the
# last boundary and cleans them there too.
cap = L.build_caption("a room", [{"desc": "a bottle", "text": "", "x": 0.3, "y": 0.2, "w": 0.4, "h": 0.6}],
                      high_level="a bottle", aesthetics="photorealistic", lighting="soft",
                      photo="85mm lens, f/1.8, bokeh", medium="photography")
style = [v for v in cap.values() if isinstance(v, dict) and "photo" in v]
check("build_caption converts a planner-written photo field",
      style and not re.search(r"\bf/|\d+\s*mm", style[0]["photo"]), style)

# Whole live layout: the only digits left in the caption are box coordinates.
lay = {"high_level_description": "a bottle", "background": "a lounge",
       "photo": "50mm lens, f/4, sharp focus", "aesthetics": "photorealistic",
       "lighting": "natural light", "medium": "photography",
       "elements": [{"desc": "a bottle", "text": "", "x": 0.3, "y": 0.2, "w": 0.4, "h": 0.6}]}
js = json.dumps(ideogram.layout_to_caption(lay), ensure_ascii=False)
check("no aperture or focal-length figures survive into the caption",
      not re.search(r"\bf/\d|\d+\s*mm", js), js[:300])

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)

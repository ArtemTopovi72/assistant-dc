"""When the photographic default keeps overriding a colour, draw it as an illustration.

Live 2026-09-12 (journey 4): "a blue elephant on a beach" came back as a
grey photograph five times; the caption had been insisting "vibrant blue --
not grey" since the second attempt. Photography was the style FLOOR, not a
request, so the second insistence drops it. Realism the user asked for stays.
"""
import os, sys, copy
os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _look_stub  # noqa: F401  the picture's look, stubbed
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import draw_agent as D

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

def photo_layout():
    return {"medium": "photography", "aesthetics": "photorealistic, natural colour",
            "photo": "standard lens", "lighting": "natural light", "art_style": "",
            "elements": [{"desc": "a vibrant blue elephant sitting on the sand"}]}

problems = ["The elephant is grey instead of vibrant blue"]

lay, notes, left = D.insist_on_mismatch(photo_layout(), problems)
check("first mismatch underlines the colour", any(n.startswith("insisted:") for n in notes), notes)
check("...and is not yet 'again'", not any("again" in n for n in notes), notes)
lay2, restyle = D.unphoto_on_repeat_mismatch(lay, notes, "draw a blue elephant on a beach")
check("first time: photography stays", lay2["medium"] == "photography" and not restyle, restyle)

lay, notes2, _ = D.insist_on_mismatch(lay, problems)
check("second mismatch is 'insisted again'", any(n.startswith("insisted again:") for n in notes2), notes2)
check("the underline is not doubled and never names grey",
      lay["elements"][0]["desc"].lower().count("all of it vibrant blue") == 1
      and "grey" not in lay["elements"][0]["desc"].lower(), lay["elements"][0]["desc"])
lay3, restyle = D.unphoto_on_repeat_mismatch(copy.deepcopy(lay), notes2, "draw a blue elephant on a beach")
check("second time: the photographic default is dropped", lay3["medium"] == "illustration" and lay3["art_style"], lay3)
check("photo and art_style stay exclusive", lay3["photo"] == "", lay3)
check("the note says why", "illustration" in restyle, restyle)

lay4, restyle = D.unphoto_on_repeat_mismatch(copy.deepcopy(lay), notes2, "a realistic photo of a blue elephant on a beach")
check("realism the user asked for is kept", lay4["medium"] == "photography" and not restyle, (lay4["medium"], restyle))

ill = dict(lay, medium="illustration", art_style="watercolour")
lay5, restyle = D.unphoto_on_repeat_mismatch(ill, notes2, "draw a blue elephant")
check("a layout that is already an illustration is untouched", lay5["art_style"] == "watercolour" and not restyle)

lay6, restyle = D.unphoto_on_repeat_mismatch(photo_layout(), [], "x")
check("no notes, no change", not restyle and lay6["medium"] == "photography")

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)

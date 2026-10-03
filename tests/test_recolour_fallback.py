"""«Сделай слона розовым» on a layout picture (journey 4, live 2026-09-18).

The layout re-render with the same seed kept the old colours whatever the box
said, the critic's "The elephant is not pink as requested" was dropped as
taste, and a tan elephant shipped as a successful recolour -- twice. Now the
colour of a named thing counts as a content complaint, and a layout edit whose
re-render still carries problems hands the original to the pixel pipeline
(FireRed contained), which paints the change in place.
"""
import os, sys, types
os.environ.setdefault("F5_TEST_RUN", "1")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import draw_agent as D
import image_router as R

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

rx = D._LAYOUT_CONCERN_RE
check("'not pink as requested' is content, not taste", rx.search("The elephant is not pink as requested"))
check("'is not pink' is content", rx.search("The elephant is not pink"))
check("'wrong colour' is content", rx.search("The dress has the wrong colour"))
check("'still not red' is content", rx.search("the car is still not red"))
check("a messy background is still taste", not rx.search("the background of the sign is messy"))
check("an overlap remark is still taste", not rx.search("The palm tree and elephant overlap significantly"))

# --- the fallback ---------------------------------------------------------------------
calls = []
_orig_load, _orig_save = R._image.load_layout_for, R._image.save_layout_for
R._image.load_layout_for = lambda p: {"layout": {"elements": []}, "prompt": "a blue elephant", "seed": 7, "width": 960, "height": 544}
R._image.save_layout_for = lambda *a, **k: calls.append("saved")
_orig_edit, _orig_run = D.edit_layout, D.run
D.edit_layout = lambda ctx, layout, instr: ({"elements": [{"description": "a pink elephant"}]}, ["replaced blue with pink"])
def _run(ctx, req, **k):
    calls.append("run")
    return {"image": "re-rendered.png", "layout": k.get("layout"), "seed": 7, "problems": _run.problems}
D.run = _run
try:
    _run.problems = ["The elephant is not pink as requested"]
    out = R.edit_via_layout(None, "missing.png", "make the elephant pink", fallback_on_problems=True)
    check("a re-render the critic still faults yields None -> pixel pipeline", out is None and "run" in calls, (out, calls))
    check("...the re-rendered picture keeps its layout record", "saved" in calls)
    check("without the flag (lettering) the picture is returned",
          R.edit_via_layout(None, "missing.png", "make the elephant pink") == "re-rendered.png")
    _run.problems = []
    check("a clean re-render is returned", R.edit_via_layout(None, "missing.png", "make it pink", fallback_on_problems=True) == "re-rendered.png")
finally:
    R._image.load_layout_for, R._image.save_layout_for = _orig_load, _orig_save
    D.edit_layout, D.run = _orig_edit, _orig_run

src = open(R.__file__, encoding="utf-8").read()
check("route_edit_request asks for the fallback for everything but lettering",
      'fallback_on_problems=(category != "text_edit")' in src)

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)

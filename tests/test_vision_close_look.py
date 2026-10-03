"""A question about a picture reads the zoomed parts too (vision_close).

Live 2026-09-18: «на картинках видит только самые здоровые детали, на мелкие
вообще забивает». One 1024-px look answers with the biggest things; for a
QUESTION the picture is also cut into zoomed parts and the answer is written
from the overview plus the parts.
"""
import os, sys, io
os.environ.setdefault("F5_TEST_RUN", "1")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
import vision_close as V
import llm

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

def png(w, h):
    buf = io.BytesIO(); Image.new("RGB", (w, h), (200, 180, 160)).save(buf, "PNG"); return buf.getvalue()

# --- cutting ------------------------------------------------------------------
b2 = V.tile_boxes(1600, 1200)
check("a 1600-px picture is cut 2x2", len(b2) == 4 and [n for n, _ in b2][0] == "top-left")
check("the parts overlap a little (nothing is lost on a seam)",
      b2[0][1][2] > 800 and b2[1][1][0] < 800, (b2[0][1], b2[1][1]))
check("the parts stay inside the picture", all(0 <= x0 < x1 <= 1600 and 0 <= y0 < y1 <= 1200 for _, (x0, y0, x1, y1) in b2))
b3 = V.tile_boxes(4000, 3000)
check("a phone photo is cut 3x3", len(b3) == 9 and b3[4][0] == "centre")
check("a small picture has nothing to zoom into", V.tiles_of(png(640, 480)) == [])
parts = V.tiles_of(png(1600, 1200))
check("the parts are JPEGs of the crops", len(parts) == 4 and all(d[:2] == b"\xff\xd8" for _, d in parts))

# --- the close look ------------------------------------------------------------
class _Ctx:
    def __init__(self): self.stages = []; self.cancelled = False
    def set_stage(self, s): self.stages.append(s)
    def is_cancelled(self): return self.cancelled

seen = []
_orig_a, _orig_s = llm.analyze_image_with_llm, llm.call_llm_simple
def _fake_look(ctx, image_path=None, image_bytes=None, user_text="", system_prompt="", **k):
    seen.append(user_text)
    return "a name plate reading 'A. Petrov'" if "top-left" in user_text else "nothing small here"
def _fake_merge(ctx, system, user, **k):
    seen.append(("merge", user))
    return "Совещание; на табличке слева написано «A. Petrov»."
llm.analyze_image_with_llm = _fake_look
llm.call_llm_simple = _fake_merge
try:
    c = _Ctx()
    out = V.look_closely(c, png(1600, 1200), "что написано на табличках?", "A meeting in a conference room.")
    check("four parts were read and the answer merged", len(seen) == 5 and out.startswith("Совещание"), (len(seen), out))
    check("the merge sees the question, the overview and the parts' notes",
          isinstance(seen[-1], tuple) and "табличках" in seen[-1][1] and "conference room" in seen[-1][1]
          and "A. Petrov" in seen[-1][1], seen[-1])
    check("the user sees a 'looking closer' stage", any("Looking closer" in s for s in c.stages), c.stages)
    check("a small picture -> None, the overview stands", V.look_closely(c, png(500, 400), "q", "ov") is None)
    c2 = _Ctx(); c2.cancelled = True
    check("a cancelled turn stops before the parts", V.look_closely(c2, png(1600, 1200), "q", "ov") is None)
    llm.analyze_image_with_llm = lambda *a, **k: None
    check("no notes at all -> None", V.look_closely(_Ctx(), png(1600, 1200), "q", "ov") is None)
finally:
    llm.analyze_image_with_llm, llm.call_llm_simple = _orig_a, _orig_s

# --- the wiring -----------------------------------------------------------------
import graph as G
src = open(G.__file__, encoding="utf-8").read()
check("a fresh photo with a question gets the close look",
      "response = _close_look(ctx, original_bytes, state.get(\"user_input\", \"\"), response)" in src)
check("a re-look with a question gets it too, from the full-res file",
      "response = _close_look(ctx, current_bytes, user_text, response)" in src)
_orig_vc = sys.modules.get("vision_close")
class _Boom:
    @staticmethod
    def look_closely(*a, **k): raise RuntimeError("no")
sys.modules["vision_close"] = _Boom
try:
    check("a failing close look leaves the overview", G._close_look(None, b"", "q", "overview") == "overview")
finally:
    sys.modules["vision_close"] = _orig_vc

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)

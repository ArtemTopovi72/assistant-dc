"""«Убери все надписи с фона»: the mask comes from the OCR, not from a phrase.

Journey 3, live 2026-09-18: SAM3 given 'wall inscriptions' returned one
166-px strip and "INSGATE" stayed on the left. Now a removal aimed at the
lettering masks every OCR box (faces cut out), LaMa fills, the OCR reads the
result again, and a logo the vision model still sees gets one segmenter pass.
Verified live on bench/assets/red_carpet.jpg: INSGATE, USEMAID and the
lady/tch.ua watermark gone in one pass (lettering-remove_00001_.png).
"""
import os, sys, io, types, tempfile
os.environ.setdefault("F5_TEST_RUN", "1")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The model's reads are stubbed here; the phrases run live in bench/intent_rest_live.py.
import intent
intent.YES_STUB = lambda q, t: "WRITING" in q and any(w in t.lower() for w in ("inscription", "надпис", "watermark", "logo", "text", "lettering"))
from PIL import Image
import image_lettering_remove as L

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

# --- the detector ----------------------------------------------------------------
check("'all inscriptions and logos on the wall' is a lettering removal",
      L.is_lettering_removal("all inscriptions and logos on the background wall", "a clean dark red wall"))
check("Russian 'надписи' too", L.is_lettering_removal("надписи", ""))
check("a watermark too", L.is_lettering_removal("watermark", ""))
check("a hat is not", not L.is_lettering_removal("hat", "bare head"))
check("a person is not", not L.is_lettering_removal("man on the left", ""))

# --- the mask ----------------------------------------------------------------------------
d = tempfile.mkdtemp()
src = os.path.join(d, "src.png"); Image.new("RGB", (400, 300), (120, 20, 40)).save(src)
boxes = [(0, 20, 150, 60), (250, 20, 400, 60), (300, 250, 400, 290)]
m, used = L.build_mask(src, boxes, os.path.join(d, "m.png"), pad=4, protect=[(120, 0, 280, 120)])
im = Image.open(m).convert("L")
check("all boxes are painted", used == 3)
check("a box is white inside", im.getpixel((50, 40)) == 255 and im.getpixel((350, 270)) == 255)
check("the padding is white", im.getpixel((10, 17)) == 255 and im.getpixel((10, 63)) == 255)
check("the face area is cut out, not the whole box dropped",
      im.getpixel((130, 40)) == 0 and im.getpixel((100, 40)) == 255 and im.getpixel((290, 40)) == 255)
check("outside is black", im.getpixel((200, 200)) == 0)
m2, used2 = L.build_mask(src, [(130, 10, 200, 100)], os.path.join(d, "m2.png"), protect=[(120, 0, 280, 120)])
check("a box entirely on the face leaves nothing to remove", used2 == 0)
check("no OCR under the test flag -> None (nothing masked blind)", L.ocr_boxes(src) is None)

# --- the passes ----------------------------------------------------------------------------
import image as I
calls = []
_o_boxes, _o_fill, _o_faces, _o_smear = L.ocr_boxes, L._fill_firered, L._faces, L.smear_ratio
reads = iter([boxes, [(0, 20, 150, 60)], []])
L.ocr_boxes = lambda p: next(reads)
L._faces = lambda p: []
L.smear_ratio = lambda p, b, **k: 1.0
def _fill(ctx, cur, mask, grow):
    calls.append((grow, "firered"))
    out = os.path.join(d, f"out{len(calls)}.png"); Image.new("RGB", (400, 300)).save(out); return out
L._fill_firered = _fill
class _Ctx:
    def __init__(self): self.stages = []
    def set_stage(self, s): self.stages.append(s)
    def is_cancelled(self): return False
try:
    c = _Ctx()
    out = L.remove_lettering(c, src)
    check("two FireRed passes: the second with a wider mask, until the OCR reads nothing",
          len(calls) == 2 and calls[0][0] < calls[1][0], calls)
    check("the result is the last pass", out and out.endswith("out2.png"), out)
    check("the user sees the stage", any("Removing the lettering" in s for s in c.stages))
    calls.clear(); reads = iter([[]])
    check("no lettering read -> None (nothing to do)", L.remove_lettering(c, src) is None and not calls)
    reads = iter([None])
    check("OCR unavailable -> None", L.remove_lettering(c, src) is None)
    calls.clear(); reads = iter([boxes]); L.smear_ratio = lambda p, b, **k: 0.4
    check("a smeared fill is refused, never shipped", L.remove_lettering(c, src) is None and len(calls) == 1)
finally:
    L.ocr_boxes, L._fill_firered, L._faces, L.smear_ratio = _o_boxes, _o_fill, _o_faces, _o_smear

# --- the logo pass ----------------------------------------------------------------------------
import llm, image_objects
_o_an, _o_lama, _o_rl = llm.analyze_image_with_llm, image_objects.remove_object_with_comfy, L.remove_lettering
L.remove_lettering = lambda ctx, p, **k: "ocr_out.png"
image_objects.remove_object_with_comfy = lambda ctx, p, phrase, **k: calls.append(("firered", p, phrase)) or "logo_out.png"
try:
    llm.analyze_image_with_llm = lambda *a, **k: "NONE"
    calls.clear()
    check("vision says NONE -> the OCR result stands", L.remove_lettering_and_logos(_Ctx(), src) == "ocr_out.png" and not calls)
    llm.analyze_image_with_llm = lambda *a, **k: "audible logo top right"
    check("a logo the vision model names gets one segmenter pass on the OCR result",
          L.remove_lettering_and_logos(_Ctx(), src) == "logo_out.png" and calls == [("firered", "ocr_out.png", "audible logo top right")], calls)
    llm.analyze_image_with_llm = lambda *a, **k: "There is a lot of decorative text and logos and many things on the wall behind"
    check("a rambling answer is not a phrase -> no pass", L.vision_leftover(_Ctx(), src) == "")
finally:
    llm.analyze_image_with_llm, image_objects.remove_object_with_comfy, L.remove_lettering = _o_an, _o_lama, _o_rl

src_h = open("agent/tool_image_handlers.py", encoding="utf-8").read()
check("inpaint_image routes a lettering removal here before the router",
      "_lett.remove_lettering_and_logos(ctx, source)" in src_h and src_h.index("_lett.is_lettering_removal") < src_h.index('elif category == "colour_convert"'))

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)

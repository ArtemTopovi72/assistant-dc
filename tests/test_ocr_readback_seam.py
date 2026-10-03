"""The lettering read-back asks EasyOCR first and the vision model only when
OCR did not confirm the words (live, 2026-09-12: the vision model read a
crisp "У ОЛЬГИ" as "УАЗЬЛ" four times out of four and a perfect café sign
was redrawn for ten minutes). Also: a 3-letter host noun ("car", "cup",
"jar") attaches lettering to its host.

Pure: EasyOCR is stubbed; no model, no GPU.

Run: venv/Scripts/python.exe tests/test_ocr_readback_seam.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
os.environ["F5_TEST_RUN"] = "1"
import logging; logging.basicConfig(level=logging.CRITICAL)

from PIL import Image
import draw_text as D
import ocr_reader as O
import draw_geometry as G
import llm

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print("PASS  " + name)
    else:
        BAD += 1; print("FAIL  " + name + "   " + str(extra)[:200])


check("OCR is off under F5_TEST_RUN", not O.available() and O.read(__file__) is None)

d = tempfile.mkdtemp(prefix="ocr_seam_")
img = os.path.join(d, "sign.png"); Image.new("RGB", (960, 544), (120, 120, 120)).save(img)
LAY = {"background": "a street", "elements": [
    {"desc": "a cafe front", "text": "", "x": 0.1, "y": 0.2, "w": 0.8, "h": 0.7},
    {"desc": "a wooden sign above the door", "text": "У ОЛЬГИ", "x": 0.2, "y": 0.05, "w": 0.6, "h": 0.15}]}


class Ctx:
    model_name = "test"; no_think = False; reasoning_effort = "high"
    def is_cancelled(self): return False
    def set_stage(self, *_): pass


vision_calls = []
_real = llm.analyze_image_with_llm


def vision(ctx, image_path=None, user_text="", system_prompt="", **kw):
    vision_calls.append(user_text)
    return '{"strings": ["УАЗЬЛ"]}'


llm.analyze_image_with_llm = vision
_ocr_real = O.read
try:
    # 1. OCR confirms -> no vision call at all
    O.read = lambda p: ["У ОЛЬГИ"]
    vision_calls.clear()
    r = D.verify_text(Ctx(), img, LAY)
    check("OCR confirms the lettering", r["ok"] and r["checks"][0]["ok"], r)
    check("and the vision model is not asked", not vision_calls, vision_calls)
    check("the check names its reader", r["checks"][0].get("reader") == "ocr", r["checks"])
    check("source says ocr", r["source"] == "ocr", r["source"])

    # 2. OCR pieces joined count as one candidate
    O.read = lambda p: ["У", "ОЛЬГИ"]
    vision_calls.clear()
    r = D.verify_text(Ctx(), img, LAY)
    check("split pieces are matched joined", r["ok"], r["checks"])

    # 3. OCR one letter off at 2x beats the vision model's fantasy
    O.read = lambda p: ["ОАЬГИ", "УОХЬГИ"]
    vision_calls.clear()
    r = D.verify_text(Ctx(), img, LAY)
    check("a near read passes on the best OCR candidate", r["ok"], r["checks"])
    check("vision was not needed", not vision_calls)

    # 4. OCR reads garbage -> vision is asked, and the better reading counts
    O.read = lambda p: ["XXXX"]
    vision_calls.clear()
    r = D.verify_text(Ctx(), img, LAY)
    check("when OCR fails the vision model is asked", len(vision_calls) >= 1)
    check("and a garbled render is still a failure", not r["ok"] and r["failures"] == [1], r)

    # 5. no OCR at all -> old behaviour, vision decides
    O.read = lambda p: None
    vision_calls.clear()
    llm.analyze_image_with_llm = lambda ctx, **kw: '{"strings": ["У ОЛЬГИ"]}'
    r = D.verify_text(Ctx(), img, LAY)
    check("without OCR the vision reading decides", r["ok"] and r["checks"][0].get("reader") == "vision", r)
    llm.analyze_image_with_llm = vision

    # 6. nothing readable anywhere is a failure, not a pass
    O.read = lambda p: []
    llm.analyze_image_with_llm = lambda ctx, **kw: '{"strings": []}'
    r = D.verify_text(Ctx(), img, LAY)
    check("lettering that never appeared is a failure", not r["ok"], r)
finally:
    O.read = _ocr_real
    llm.analyze_image_with_llm = _real

# -- a 3-letter host noun attaches ---------------------------------------------
sign = {"desc": "a sign board on the roof of the car", "text": "POLICE"}
car = {"desc": "a dark patrol car, side view", "text": ""}
check("'car' attaches the sign to the car", G._attached_to(sign, car))
check("'jar' too", G._attached_to({"desc": "a label on the jar", "text": "HONEY"},
                                  {"desc": "a glass jar of honey", "text": ""}))
check("a stopword does not attach", not G._attached_to({"desc": "the sign for the show", "text": "X"},
                                                       {"desc": "the man for hire", "text": ""}))

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)

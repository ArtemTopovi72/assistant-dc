"""The pre-render layout check (draw_preview): the plan drawn as a sketch,
shown to the vision model with the request, its ops applied to the boxes
BEFORE the card is spent on a render."""
import os, sys, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("F5_TEST_RUN", "1")

import draw_preview as P
import draw_agent
import llm

PASSED = FAILED = 0


def check(name, cond, extra=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1; print("PASS ", name)
    else:
        FAILED += 1; print("FAIL ", name, " ", extra)


LAY = {"background": "a cosy living room", "elements": [
    {"desc": "a green sofa", "x": 0.1, "y": 0.45, "w": 0.7, "h": 0.4},
    {"desc": "a black cat curled up", "x": 0.75, "y": 0.2, "w": 0.2, "h": 0.2},
    {"desc": "a floor lamp", "x": 0.02, "y": 0.1, "w": 0.12, "h": 0.6},
    {"desc": "a neon sign", "text": "HOME", "x": 0.3, "y": 0.05, "w": 0.4, "h": 0.15}]}
REQ = "a black cat sleeping on a green sofa, a floor lamp to the left, a neon sign HOME"
tmp = tempfile.mkdtemp(prefix="sketch_")

# ── the sketch ───────────────────────────────────────────────────────────────
from PIL import Image
p = P.render_sketch(LAY, os.path.join(tmp, "s.jpg"), width=1344, height=768)
im = Image.open(p)
check("the sketch is written as a JPEG of the final proportions",
      os.path.exists(p) and abs(im.width / im.height - 1344 / 768) < 0.02, (im.width, im.height))
p2 = P.render_sketch(LAY, os.path.join(tmp, "s2.jpg"), width=768, height=1344)
im2 = Image.open(p2)
check("a portrait frame gives a portrait sketch", im2.height > im2.width)
# translucency: the cat box painted over nothing is lighter than a double overlap
px = im.getpixel((int(0.85 * im.width), int(0.3 * im.height)))
check("boxes are tinted, not opaque (the ground shows through)", all(c > 120 for c in px), px)


class _Ctx:
    model_name = "chat"
    stages = []
    def set_stage(self, s): self.stages.append(s)
    def is_cancelled(self): return False


def _answer(js):
    llm.analyze_image_with_llm = lambda ctx, image_path=None, user_text="", system_prompt="", **kw: (
        seen.__setitem__("image", image_path), seen.__setitem__("user", user_text),
        seen.__setitem__("system", system_prompt), js)[-1]


seen = {}
# ── no ctx: skipped, nothing touched ─────────────────────────────────────────
out, rep = P.preflight(None, LAY, REQ)
check("without a context the plan passes through untouched", rep["source"] == "skipped"
      and out["elements"][1]["x"] == 0.75)

# ── the checker moves the cat onto the sofa ──────────────────────────────────
calls = []
def _seq(*answers):
    it = iter(answers)
    llm.analyze_image_with_llm = lambda ctx, image_path=None, user_text="", system_prompt="", **kw: (
        calls.append((image_path, user_text, system_prompt)), next(it))[-1]
_seq('{"ok": false, "problems": ["the cat is not on the sofa"], '
     '"ops": [{"op": "move", "target": "the black cat", "x": 0.4, "y": 0.5}]}',
     '{"ok": true, "problems": [], "ops": []}')
ctx = _Ctx()
out, rep = P.preflight(ctx, LAY, REQ, width=1344, height=768)
cat = out["elements"][1]
check("the vision model was shown the sketch, the request and the regions in words",
      calls and calls[0][0].endswith(".jpg") and os.path.exists(calls[0][0])
      and "THE REQUEST" in calls[0][1] and REQ in calls[0][1] and "a black cat curled up" in calls[0][1]
      and "Overlap is NEVER a problem" in calls[0][2], calls[:1])
check("the cat box was moved by the checker's op and the plan re-checked",
      abs(cat["x"] - 0.4) < 0.01 and abs(cat["y"] - 0.5) < 0.01 and rep["passes"] == 2 and rep["ok"]
      and rep["checked"] and any("moved" in n for n in rep["notes"]), (cat, rep))
check("the stage line is set for the status message", "Checking the layout" in ctx.stages)
check("the other boxes were left alone", out["elements"][0]["x"] == 0.1 and out["elements"][3]["text"] == "HOME")

# ── nothing wrong: one pass, unchanged ───────────────────────────────────────
calls.clear(); _seq('{"ok": true, "problems": [], "ops": []}')
out, rep = P.preflight(ctx, LAY, REQ)
check("a content checker costs one call and changes nothing",
      rep["passes"] == 1 and rep["ok"] and out["elements"][1]["x"] == 0.75)

# ── an ok=true with ops still applies the ops (the model contradicts itself) ─
calls.clear(); _seq('{"ok": true, "problems": [], "ops": [{"op": "delete", "target": "the floor lamp"}]}',
                     '{"ok": true, "problems": [], "ops": []}')
out, rep = P.preflight(ctx, LAY, REQ)
check("ops override a contradictory ok", len(out["elements"]) == 3 and rep["passes"] == 2, rep)

# ── gutting guard ────────────────────────────────────────────────────────────
calls.clear(); _seq('{"ok": false, "problems": ["x"], "ops": [{"op": "delete", "target": "1"}, '
                     '{"op": "delete", "target": "the black cat"}, {"op": "delete", "target": "the floor lamp"}]}')
out, rep = P.preflight(ctx, LAY, REQ)
check("a repair that would gut the plan is ignored and the plan kept",
      len(out["elements"]) == 4 and any("gutted" in n for n in rep["notes"]) and rep["passes"] == 1, rep)

# ── no JSON / an exception: no vote, the render is not blocked ───────────────
calls.clear(); _seq("I cannot see the image.")
out, rep = P.preflight(ctx, LAY, REQ)
check("no JSON -> unavailable, plan unchanged", rep["source"] == "unavailable" and not rep["checked"]
      and out["elements"][1]["x"] == 0.75)
llm.analyze_image_with_llm = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
out, rep = P.preflight(ctx, LAY, REQ)
check("a vision exception -> unavailable, plan unchanged", rep["source"] == "unavailable"
      and out["elements"][1]["x"] == 0.75)

# ── missing + extra: delete and add ──────────────────────────────────────────
GIRL = {"background": "a rainy street", "elements": [
    {"desc": "a girl in a raincoat", "x": 0.3, "y": 0.25, "w": 0.3, "h": 0.65},
    {"desc": "a parked bicycle", "x": 0.7, "y": 0.5, "w": 0.25, "h": 0.35}]}
calls.clear(); _seq('{"ok": false, "problems": ["missing the dog", "EXTRA: bicycle"], "ops": ['
                     '{"op": "delete", "target": "the parked bicycle"}, '
                     '{"op": "add", "desc": "a small dog beside the girl", "x": 0.6, "y": 0.7, "w": 0.15, "h": 0.18}]}',
                     '{"ok": true, "problems": [], "ops": []}')
out, rep = P.preflight(ctx, GIRL, "a girl with her dog on a rainy street")
descs = [e["desc"] for e in out["elements"]]
check("the extra box is deleted and the missing one added", "a parked bicycle" not in descs
      and any("dog" in d for d in descs) and rep["ok"], (descs, rep))

# ── wiring ───────────────────────────────────────────────────────────────────
src_ig = open(os.path.join(os.path.dirname(__file__), "..", "imaging/image_generate.py"), encoding="utf-8").read()
src_da = open(os.path.join(os.path.dirname(__file__), "..", "imaging/draw_agent.py"), encoding="utf-8").read()
check("image_generate checks the plan right after planning it, before either render path",
      "draw_preview.preflight(ctx, layout, prompt" in src_ig
      and src_ig.index("draw_preview.preflight") < src_ig.index("draw_agent.run(ctx, prompt, layout=layout")
      and src_ig.index("draw_preview.preflight") < src_ig.index("caption=ideogram.layout_to_caption(layout)"))
check("draw_agent.run checks a plan it made itself", "draw_preview.preflight(ctx, layout, request" in src_da)
import stages
check("the stage line has a Russian form", stages._STAGES.get("Checking the layout", {}).get("ru"))

print(f"\n{PASSED} passed, {FAILED} failed")
sys.exit(1 if FAILED else 0)

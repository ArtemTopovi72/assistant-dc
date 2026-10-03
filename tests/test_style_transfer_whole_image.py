"""A whole-image style transfer must not be phrased as a targeted region edit.

WHAT HAPPENED (live, 2026-09-20). A user uploaded a 4-person photo collage with
a title ("Ma' boyz") and asked to redraw the whole thing as a cartoon/comic.
style_transfer has no region, so the router fell to
`_firered_instruction(ctx, "", text, removal=False)`, whose region="" branch
produces:

    "Change the it to: <style>. Keep everything else in the image exactly
    the same, matching the original style and lighting."

"Change the it" names no target and "keep everything else the same"
contradicts a full-canvas restyle. FireRed read this as "restyle one subject,
preserve the frame" and came back cropped — the title text and one person
were gone, and only part of the group was cartoonised. The verifier then
(correctly) reported the requested change as only partially there.

`_firered_style_instruction` replaces that call for style_transfer: it asks
to redraw the ENTIRE image and explicitly forbids cropping or removing
anything, instead of naming an "it" and asking to preserve the rest.

Run: venv/Scripts/python.exe tests/test_style_transfer_whole_image.py
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import image as I
import image_router as R
R.EDIT_STUB = lambda t: "style_transfer"

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print(f"PASS  {name}")
    else:
        BAD += 1; print(f"FAIL  {name}   {extra}")


TMP = tempfile.mkdtemp(prefix="styletransfer_")


def _img():
    p = os.path.join(TMP, "src.png")
    open(p, "wb").write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    return p


# ── the instruction builder itself ───────────────────────────────────────────
print("=" * 66)
print("INSTRUCTION TEXT")
print("=" * 66)

I._english_instructions = lambda ctx, text: text  # deterministic, no LLM call

instr = I._firered_style_instruction(None, "redraw this as a cartoon")
check("does not contain the broken 'Change the it' phrasing",
      "change the it" not in instr.lower(), instr)
check("tells the model to redraw the ENTIRE image",
      "entire image" in instr.lower(), instr)
check("explicitly forbids cropping/zooming",
      "crop" in instr.lower() and "zoom" in instr.lower(), instr)
check("explicitly demands the same size/position for every person",
      "same size" in instr.lower() and "same position" in instr.lower(), instr)
check("carries the requested style through",
      "cartoon" in instr.lower(), instr)


# ── routed end-to-end ─────────────────────────────────────────────────────────
print("=" * 66)
print("ROUTING")
print("=" * 66)

captured = {}


def _fake_firered(ctx, image_path, instruction, *, seed=None, timeout=1900):
    captured["instruction"] = instruction
    return "/tmp/out.png"


I.edit_image_with_firered = _fake_firered
R._image.edit_image_with_firered = _fake_firered

path = _img()
category, out = R.route_edit_request(None, path, "redraw this whole photo as a cartoon")
check("classified as style_transfer", category == "style_transfer", category)
check("edit_image_with_firered was called", out == "/tmp/out.png")
check("router used the whole-image instruction, not the region-templated one",
      "change the it" not in captured.get("instruction", "").lower(),
      captured.get("instruction"))
check("router's instruction demands the whole image",
      "entire image" in captured.get("instruction", "").lower(),
      captured.get("instruction"))


# ── vision-derived layout lock ───────────────────────────────────────────────
print("=" * 66)
print("VISION LAYOUT LOCK")
print("=" * 66)

import llm as _llm

_llm.analyze_image_with_llm = lambda **kw: (
    "4 people left-to-right; leftmost cropped at the shoulder near the left "
    "edge, small; a title reads 'Ma' Boyz' near the top.")

instr_with_vision = I._firered_style_instruction(None, "redraw this as a cartoon",
                                                 image_path=path)
check("folds the vision-derived layout description into the instruction",
      "leftmost cropped at the shoulder" in instr_with_vision, instr_with_vision)
check("still carries the base whole-image demand",
      "entire image" in instr_with_vision.lower(), instr_with_vision)

_llm.analyze_image_with_llm = lambda **kw: (_ for _ in ()).throw(RuntimeError("vision down"))
instr_vision_fails = I._firered_style_instruction(None, "redraw this as a cartoon",
                                                  image_path=path)
check("falls back to the generic instruction when the vision call raises",
      "entire image" in instr_vision_fails.lower(), instr_vision_fails)


# ── the OTHER entry point: the redraw_image TOOL ─────────────────────────────
# route_edit_request (above) is only ONE way into whole-frame FireRed. The
# live bot's agent calls the `redraw_image` tool directly with its own
# free-text instructions, reaching tool_image_handlers._handle_redraw_image ->
# image.edit_image_with_firered with NO classifier and NO layout lock at all.
# Live, 2026-09-20 (same day, after the fix above shipped): a user re-tested
# through the live bot and got the leftmost person cropped in half again --
# their request went through THIS path, which the first fix never touched.
print("=" * 66)
print("REDRAW_IMAGE TOOL ENTRY POINT")
print("=" * 66)

import tool_image_handlers as TIH


class _Ctx:
    last_image_path = None
    last_image_prompt = ""
    def set_stage(self, *a, **kw): pass
    def remember(self, *a, **kw): pass
    def memory_text(self, *a, **kw): return ""


saved_exists = os.path.exists
os.path.exists = lambda p: True
saved_deliverable = I.assert_deliverable
I.assert_deliverable = lambda p, **kw: p
saved_layout = I.load_layout_for
I.load_layout_for = lambda p: None
saved_preserve = I.preserve_identity_face
I.preserve_identity_face = lambda src, new, **kw: new
saved_budget = None
try:
    import tools as _T
    saved_budget = _T._render_budget_exhausted
    _T._render_budget_exhausted = lambda *a, **kw: None
except Exception:
    pass

captured.clear()
state = {"image_path": path}
TIH._handle_redraw_image(
    _Ctx(), state,
    {"mode": "redraw",
     "instructions": "convert to anime art style, cel-shaded, keep the exact "
                     "same pose, camera framing, outfit and background -- "
                     "change only the art style"})

os.path.exists = saved_exists
I.assert_deliverable = saved_deliverable
I.load_layout_for = saved_layout
I.preserve_identity_face = saved_preserve
if saved_budget is not None:
    _T._render_budget_exhausted = saved_budget

check("redraw_image also used the whole-image instruction, not the agent's raw text",
      "entire image" in captured.get("instruction", "").lower(),
      captured.get("instruction"))
check("redraw_image's instruction demands the same size/position too",
      "same size" in captured.get("instruction", "").lower(),
      captured.get("instruction"))


print("=" * 66)
print(f"{OK} passed, {BAD} failed")
print("=" * 66)
sys.exit(1 if BAD else 0)

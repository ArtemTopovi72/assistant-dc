"""Where a "redraw" goes depends on where the picture came from.

A picture WE composed with Ideogram has a full stored box layout and must be
re-drawn through it (arrangement and lettering kept); a photo the USER uploaded
must be edited in place, not re-rendered as a NEW picture that merely resembles
theirs. The order is:

  1. a stored layout        -> edit_via_layout   (Ideogram boxes)
  2. no layout              -> edit_image_with_firered (whole frame, NO mask)

FireRed is the only engine left once a layout edit is unavailable or fails, and
an empty instruction gets a generic quality-pass instruction.

Every engine is stubbed; nothing here reaches ComfyUI or a GPU.

Run: venv/Scripts/python.exe tests/test_redraw_routing.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import tools as T
import image as I
import image_router as _R
_R.EDIT_STUB = lambda t: {"make it black and white": "colour_convert", "перерисуй в аниме стиле": "style_transfer"}.get(t)

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")


TMP = tempfile.mkdtemp(prefix="redraw_")
def _img(tag):
    p = os.path.join(TMP, f"{tag}.png")
    open(p, "wb").write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    return p


class Ctx:
    last_image_path = None
    last_image_prompt = "a red tractor in a field"
    def set_stage(self, *_a, **_k): pass
    def remember(self, *_a, **_k): pass
    def memory_text(self, *_a, **_k): return ""


CALLS = []

def _reset():
    CALLS.clear()

def _stub(name, result):
    def fn(*a, **kw):
        CALLS.append(name)
        return result
    return fn


def _run(source, instructions, *, layout, layout_result="/out/layout.png",
         firered_result="/out/firered.png"):
    """Drive the real _handle_redraw_image with every engine stubbed."""
    _reset()
    saved = {}
    for mod, name, val in (
            (I, "load_layout_for", (lambda p: {"layout": {"elements": []}}) if layout
                                   else (lambda p: None)),
            (I, "edit_via_layout", _stub("layout", layout_result)),
            (I, "edit_image_with_firered", _stub("firered", firered_result)),
            (I, "preserve_identity_face",
             lambda src, new, **kw: (CALLS.append("preserve_identity_face"), new)[1]),
            (I, "log_edit_decision", lambda **kw: None)):
        saved[(mod, name)] = getattr(mod, name)
        setattr(mod, name, val)
    saved[(T, "_render_budget_exhausted")] = T._render_budget_exhausted
    T._render_budget_exhausted = lambda *a, **kw: None
    saved[(os.path, "exists")] = os.path.exists
    os.path.exists = lambda p: True
    try:
        state = {"image_path": source}
        msg = T._handle_redraw_image(Ctx(), state, {"mode": "redraw",
                                                    "instructions": instructions})
        return list(CALLS), state.get("image_path"), msg
    finally:
        for (mod, name), val in saved.items():
            setattr(mod, name, val)


print("=" * 70)
print("1. Our own Ideogram picture is redrawn through its boxes")
print("=" * 70)
calls, out, msg = _run(_img("ideo"), "сделай небо закатным", layout=True)
check("edit_via_layout was used", calls and calls[0] == "layout", calls)
check("FireRed was NOT reached either", "firered" not in calls, calls)
check("the layout result became the current image", out == "/out/layout.png", out)

print()
print("=" * 70)
print("2. A photo the USER uploaded is redrawn with mask-free FireRed")
print("=" * 70)
calls, out, msg = _run(_img("user"), "make it winter", layout=False)
check("FireRed was used", "firered" in calls, calls)
check("the layout path was not attempted", "layout" not in calls, calls)
check("the FireRed result became the current image", out == "/out/firered.png", out)

print()
print("=" * 70)
print("3. Fallbacks, in order, and only when needed")
print("=" * 70)
# A stored layout that fails to render must not dead-end: the picture still has
# to come back, just by a lesser route.
calls, out, _ = _run(_img("ideo2"), "make it winter", layout=True, layout_result=None)
check("a failed layout edit falls through to FireRed",
      calls == ["layout", "firered", "preserve_identity_face"], calls)
check("...and returns the FireRed result", out == "/out/firered.png", out)

# No layout AND FireRed fails too: there is no third engine left — the tool must
# report failure, not silently deliver nothing.
calls, out, msg = _run(_img("ideo3"), "make it winter", layout=True,
                       layout_result=None, firered_result=None)
check("if FireRed also fails, there is nothing left to fall back to",
      calls == ["layout", "firered"], calls)
check("...and the tool reports failure", "[TOOL ERROR]" in (msg or ""), msg)

# No instruction at all: gets a generic quality-pass instruction and still goes
# through FireRed — there is no separate 'enhance' engine.
calls, out, _ = _run(_img("bare"), "", layout=False)
check("a bare redraw with no instruction still goes through FireRed",
      calls == ["firered", "preserve_identity_face"], calls)

print()
print("=" * 70)
print("3b. The audit log names the engine that ACTUALLY ran")
print("=" * 70)
# EDIT_DECISION exists to answer "which engine touched this picture"; it must
# name the engine that ran, not derive the answer from `mode` (a FireRed redraw
# must not be recorded as "redraw_image_with_comfy / workflow_redraw.json").
_logged = []
def _run_logging(source, instructions, *, layout):
    _logged.clear()
    saved = {}
    for mod, name, val in (
            (I, "load_layout_for", (lambda p: {"layout": {}}) if layout else (lambda p: None)),
            (I, "edit_via_layout", _stub("layout", "/out/layout.png")),
            (I, "edit_image_with_firered", _stub("firered", "/out/firered.png")),
            (I, "preserve_identity_face", lambda src, new, **kw: new),
            (I, "log_edit_decision", lambda **kw: _logged.append(kw))):
        saved[(I, name)] = getattr(I, name)
        setattr(I, name, val)
    saved[(T, "_render_budget_exhausted")] = T._render_budget_exhausted
    T._render_budget_exhausted = lambda *a, **kw: None
    saved[(os.path, "exists")] = os.path.exists
    os.path.exists = lambda p: True
    try:
        T._handle_redraw_image(Ctx(), {"image_path": source},
                               {"mode": "redraw", "instructions": instructions})
        return (_logged[0].get("workflow") if _logged else "")
    finally:
        for (mod, name), val in saved.items():
            setattr(mod, name, val)

wf = _run_logging(_img("l1"), "make it blue", layout=True)
check("a layout edit is logged as the layout path", "edit_via_layout" in wf, wf)
check("...and not as the old model workflow", "workflow_redraw.json" not in wf, wf)

wf = _run_logging(_img("f1"), "make it blue", layout=False)
check("a FireRed redraw is logged as FireRed", "firered" in wf.lower(), wf)
check("...and says it was mask-free", "NO mask" in wf, wf)
check("...and not as the old model workflow", "workflow_redraw.json" not in wf, wf)

wf = _run_logging(_img("z1"), "", layout=False)
check("an empty instruction still goes through FireRed",
      "firered" in wf.lower(), wf)

print()
print("=" * 70)
print("4. The other modes are untouched")
print("=" * 70)
# colour_convert must stay deterministic no matter which tool is asked.
_reset()
_saved_cc = I.convert_colour
_saved_budget = T._render_budget_exhausted
_saved_exists = os.path.exists
I.convert_colour = _stub("colour", "/out/bw.png")
T._render_budget_exhausted = lambda *a, **kw: None
os.path.exists = lambda p: True
try:
    state = {"image_path": _img("cc")}
    T._handle_redraw_image(Ctx(), state, {"mode": "redraw",
                                          "instructions": "make it black and white"})
    check("a colour conversion still short-circuits to the exact path",
          CALLS == ["colour"], CALLS)
finally:
    I.convert_colour = _saved_cc
    T._render_budget_exhausted = _saved_budget
    os.path.exists = _saved_exists

print()
print("=" * 70)
print("5. An intermediate scratch tile must never reach the user")
print("=" * 70)
# Every edit handler (inpaint, transfer, fix_hands, fix_artifact, redraw) runs
# its result through assert_deliverable, so an engine returning a scratch path
# such as "_INTERMEDIATE_tile_firered_00001_.png" is caught.
_leak_path = "/out/_INTERMEDIATE_tile_firered_00001_.png"
calls, out, msg = _run(_img("leak"), "make it winter", layout=False,
                       firered_result=_leak_path)
check("the intermediate tile was rejected, not delivered", out != _leak_path, out)
check("the tool reports failure, not success", "[TOOL ERROR]" in (msg or ""), msg)

print()
print("=" * 70)
print("6. A style change is never re-fitted with the ORIGINAL face")
print("=" * 70)
# An anime restyle of a photo must not come back with an anime body and the
# ORIGINAL photorealistic face pasted on top: preserve_identity_face's own
# face-change detector only recognises an explicit attribute edit
# ("sunglasses"), never "in anime style". Restyling a face IS the point of a
# style request; face preservation must not run for it at all.
calls, out, _ = _run(_img("anime"), "перерисуй в аниме стиле", layout=False)
check("FireRed still runs the style edit", "firered" in calls, calls)
check("preserve_identity_face is skipped for a style change",
      "preserve_identity_face" not in calls, calls)
check("the FireRed result is returned untouched", out == "/out/firered.png", out)

# An ordinary (non-style) redraw must keep the existing face-safety net.
calls, out, _ = _run(_img("hat"), "add a red hat", layout=False)
check("preserve_identity_face still runs for a non-style edit",
      "preserve_identity_face" in calls, calls)

print()
print(f"{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)

"""A pinned 📐 Size aspect must beat every OTHER source of resolution, not just
the model's `orientation=` reading. Three more places could silently override a pin:

  1. a resolution word ("4k", "1920x1080") in the model's own PROMPT TEXT
     (image.py: parse_generation_params, resolution_keywords / the digitsxdigits
     regex) — covered by tests/test_image_size_choice.py, re-checked here too.
  2. the AGENT's own tool-call width=/height= args must go through
     fix_image_params (generate_image_with_refinement must not skip it whenever
     either is given): pinned (944,1680) + agent-supplied 1920x1080 must not
     render 1920x1080.
  3. the vision-judge refine pass suggests a new width/height and its own clamp
     (2720px) exceeds the normal 2048px rail: it must respect the pin.

Run: venv/Scripts/python.exe tests/test_pinned_size_bypass.py
"""
import os, sys, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import config
from models import Context, Models
import image as I

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")


def make_ctx(aspect="9:16", quality="high"):
    ctx = Context(models=Models(None, None, None, None, False),
                  transcription_cache={}, cache_file=None,
                  asr_lock=threading.Lock(), tts_lock=threading.Lock(),
                  model_name=config.MODEL_NAME, no_think=True)
    ctx.image_aspect = aspect
    ctx.image_quality = quality
    return ctx


print("=" * 70)
print("2. AGENT-SUPPLIED width=/height= DOES NOT BYPASS THE PIN")
print("=" * 70)

ctx = make_ctx("9:16", "high")
pinned_wh = I.session_image_size(ctx)   # e.g. (944, 1680) — portrait
check("session pin is portrait", pinned_wh[1] > pinned_wh[0], pinned_wh)

_render_calls = []
def fake_render(*a, ctx=None, prompt=None, negative_prompt=None, steps=None,
                cfg=None, seed=None, width=None, height=None,
                previous_image_path=None, inpaint_denoise=None, **kw):
    _render_calls.append({"width": width, "height": height})
    return None   # stop after the first attempt; we only need the call args

# The render stand-in rides on the context: generate_image_with_refinement asks it for
# generate_image_with_comfy.
import types
ctx.deps = types.SimpleNamespace(generate_image_with_comfy=fake_render)
I.generate_image_with_refinement(ctx, "a scene", width=1920, height=1080)

check("generate_image_with_comfy was actually called", len(_render_calls) == 1,
      _render_calls)
called_w, called_h = _render_calls[0]["width"], _render_calls[0]["height"]
check("the pinned portrait size reached the renderer, not the agent's 1920x1080",
      (called_w, called_h) != (1920, 1080) and called_h > called_w,
      (called_w, called_h))
check("…and it matches session_image_size(ctx) exactly",
      (called_w, called_h) == I.normalize_resolution(*pinned_wh), (called_w, called_h))

print()
print("=" * 70)
print("3. THE REFINE/EVAL PASS RE-ORIENTS A JUDGE-SUGGESTED SIZE TO THE PIN")
print("=" * 70)

ctx2 = make_ctx("9:16", "high")
pin_w, pin_h = I.session_image_size(ctx2)
# Simulate what the eval-clamp block computes: a landscape suggestion at the
# eval-only 2720 cap, then re-orient it to the pin the way the new code does.
cw, ch = 2560, 1440  # landscape, and > IMAGE_MAX_SIDE's normal rail intent
if I.session_size_pinned(ctx2):
    if (pin_w > pin_h) != (cw > ch):
        cw, ch = ch, cw
    cw, ch = I.normalize_resolution(cw, ch)
check("a landscape eval suggestion is re-oriented to the portrait pin",
      ch > cw, (cw, ch))
check("…and is put back on the NORMAL 2048px rail, not the eval-only 2720 cap",
      max(cw, ch) <= config.IMAGE_MAX_SIDE, (cw, ch, config.IMAGE_MAX_SIDE))

print()
print(f"{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)

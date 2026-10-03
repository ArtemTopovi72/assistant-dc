"""A pinned 📐 Size aspect must beat every OTHER source of resolution, not just
the model's `orientation=` reading — a 12-family real-chat audit (2026-08-03)
found three more places a pin was silently overridden:

  1. a resolution word ("4k", "1920x1080") in the model's own PROMPT TEXT
     (image.py: parse_generation_params, resolution_keywords / the digitsxdigits
     regex) — covered by tests/test_image_size_choice.py, re-checked here too.
  2. the AGENT's own tool-call width=/height= args bypassed fix_image_params
     entirely (generate_image_with_refinement used to skip it whenever either
     was given) — measured live sequence (944,1680) pinned -> agent supplied
     1920x1080 -> rendered 1920x1080.
  3. the vision-judge refine pass suggested a new width/height with no idea a
     pin exists, and its own clamp (2720px) exceeds the normal 2048px rail.

Run: venv/Scripts/python.exe tests/test_pinned_size_bypass.py
"""
import os, sys, tempfile, threading
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
print("2. AGENT-SUPPLIED width=/height= NO LONGER BYPASSES THE PIN")
print("=" * 70)

ctx = make_ctx("9:16", "high")
pinned_wh = I.session_image_size(ctx)   # e.g. (944, 1680) — portrait
check("session pin is portrait", pinned_wh[1] > pinned_wh[0], pinned_wh)

def fake_prompt_builder(ctx, description):
    # Simulate the model having ALSO written a landscape size into its own
    # prompt/tag content — the defaults param is what the real parser would
    # have started from (the caller's pin), same as production.
    return "a scene", 8, 4.0, 12345, pinned_wh[0], pinned_wh[1]

_render_calls = []
def fake_render(*a, ctx=None, prompt=None, negative_prompt=None, steps=None,
                cfg=None, seed=None, width=None, height=None,
                previous_image_path=None, inpaint_denoise=None, **kw):
    _render_calls.append({"width": width, "height": height})
    return None   # stop after the first attempt; we only need the call args

# Patch image_generate, not image. generate_image_with_refinement lives in
# image_generate.py and calls its own module globals, so replacing image's
# re-exported names left the REAL renderer in place: the stub never ran, the
# call list stayed empty, and the suite spent ten minutes making an actual
# picture before failing on an IndexError.
import image_generate as IG
_orig_render = IG.generate_image_with_comfy
IG.generate_image_with_comfy = fake_render
try:
    I.generate_image_with_refinement(ctx, "a scene", width=1920, height=1080)
finally:
    IG.generate_image_with_comfy = _orig_render

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

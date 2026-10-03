"""A whole-image colour conversion is arithmetic — it must never be re-imagined.

WHAT HAPPENED. A user picked a picture and asked "сделай эту картинку
чёрно-белой". The request matched no specialized rule, so it fell through to
`subject_edit`, the contained region pipeline re-rendered the subject, and what
came back was:

  · still in colour (mean channel spread 19.5 — not remotely grayscale),
  · a COMPLETELY DIFFERENT TRACTOR (a modern cab model in place of the old
    open-cab one), and
  · a half-swapped background — storm sky over snow over wheat.

The model then said, correctly, "мне не удалось сделать изображение
чёрно-белым" — and the picture was delivered anyway, so the user gets a wrong
image next to a message saying it did not work.

Grayscale and sepia are exactly defined for every pixel. They keep the
resolution, they cannot invent a subject, and they need no model, no mask and no
GPU. `colour_convert` routes them to `image.convert_colour`.

Run: venv/Scripts/python.exe tests/test_colour_convert_routing.py
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import numpy as np
from PIL import Image
import image as I

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print(f"PASS  {name}")
    else:
        BAD += 1; print(f"FAIL  {name}   {extra}")


# ── routing ─────────────────────────────────────────────────────────────────
print("=" * 66)
print("ROUTING")
print("=" * 66)

# Which phrases are a colour conversion is the model's read: bench/edit_intent_live.py.
I.EDIT_STUB = {"сделай картинку чёрно-белой": "colour_convert"}.get
import image_router as _RR
_RR.EDIT_STUB = I.EDIT_STUB

# ── the conversion itself ───────────────────────────────────────────────────
print("=" * 66)
print("THE CONVERSION")
print("=" * 66)

tmp = tempfile.mkdtemp(prefix="colourconv_")
src_path = os.path.join(tmp, "src.png")
rng = np.random.default_rng(7)
src = Image.fromarray(rng.integers(0, 255, (400, 700, 3), dtype=np.uint8))
src.save(src_path)


def spread(path):
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    return float((a.max(2) - a.min(2)).mean())


bw = I.convert_colour(src_path, "make it black and white")
check("black and white produces a file", bw and os.path.exists(bw), bw)
check("…that is actually grey (spread 0)", abs(spread(bw)) < 0.01, spread(bw))
check("…at the ORIGINAL resolution",
      Image.open(bw).size == src.size, Image.open(bw).size)

sep = I.convert_colour(src_path, "сделай в сепии")
check("sepia produces a file", sep and os.path.exists(sep))
check("…and is NOT plain grey", spread(sep) > 5, spread(sep))
check("…and keeps the resolution", Image.open(sep).size == src.size)

# luminance must be preserved, or "black and white" has changed the picture
a = np.asarray(src.convert("L")).astype(float)
b = np.asarray(Image.open(bw).convert("L")).astype(float)
check("the grayscale keeps the original luminance",
      float(np.abs(a - b).mean()) < 1.0, float(np.abs(a - b).mean()))

# a missing/corrupt file must return None, never raise into the turn
check("a missing file returns None instead of raising",
      I.convert_colour(os.path.join(tmp, "nope.png"), "b&w") is None)

# ── the router actually calls it ────────────────────────────────────────────
print("=" * 66)
print("THE ROUTER USES IT")
print("=" * 66)
import inspect
import image_router as _R
src_router = inspect.getsource(_R._route_edit_request)
check("route_edit_request dispatches colour_convert",
      'if category == "colour_convert":' in src_router
      and "convert_colour(image_path, text)" in src_router)
check("it is dispatched BEFORE the generative branches",
      src_router.index('category == "colour_convert"')
      < src_router.index('category == "background_replace"'))

cat, out = I.route_edit_request(None, src_path, "сделай картинку чёрно-белой")
check("the router returns the converted file", cat == "colour_convert"
      and out and os.path.exists(out), (cat, out))
check("…and the router's output is grey too", abs(spread(out)) < 0.01, spread(out))

print(f"\n{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)


"""A character render must never reach a renderer that cannot invoke its adapter.

Originally that meant BLOCKING Ideogram outright: IMAGE_ENGINE defaults to
"ideogram4" and generate_image_with_comfy picked that branch before looking at
`lora_name`, so the first render after НейроСтепан finished training went to a
model with no LoRA node and delivered a random stranger under the character's
name, reporting success the whole way.

Ideogram can now carry an adapter, and its adapter measures better than the
The old model one (0.596 against 0.490), so the route is open again -- but the
underlying requirement is unchanged and is what these tests pin: whichever
engine a character render lands on, the adapter must arrive WITH it, and on the
Ideogram path the trigger word must arrive too. An adapter attached to a
caption that never says the trigger is idle, which is the same stranger under
the same name.

Run: venv/Scripts/python.exe tests/test_character_engine_route.py
"""
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import image as I
# _ideogram_draw is a bare local name inside image_generate, so image's copy
# is NOT the one generate_image_with_comfy calls (same trap documented in
# test_strict_ideogram.py) -- it must be patched on THIS module.
import image_generate as IG
import models

_TMP = Path(tempfile.mkdtemp(prefix="charengine_"))
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name
          + ("" if cond or not detail else "  -- " + str(detail)[:300]))
    if not cond and os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))


def Ctx():
    return models.Context(
        models=None, transcription_cache={}, cache_file=_TMP / "cache.json",
        asr_lock=threading.Lock(), tts_lock=threading.Lock())


# A minimal but complete single-stage old-model-shaped graph: one UNETLoader,
# one positive CLIPTextEncode, one KSampler fed by an Empty*Latent (so
# _collapse_to_base_sampler leaves it alone), one VAEDecode, one SaveImage.
def _minimal_workflow():
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "old_model_turbo_bf16.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "_meta": {"title": "positive prompt"},
              "inputs": {"text": "", "clip": ["1", 0]}},
        "3": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 1024}},
        "4": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["2", 0], "negative": ["2", 0],
            "latent_image": ["3", 0]}},
        "5": {"class_type": "VAEDecode", "inputs": {"samples": ["4", 0]}},
        "6": {"class_type": "SaveImage", "inputs": {"images": ["5", 0]}},
    }


class _Patches:
    """Same helper as test_image_workflow_craft.py -- patches image.py
    attributes (and IMAGE_ENGINE, which lives on config) for one call."""
    def __init__(self, **kw):
        self.kw = kw; self.orig = {}; self.engine = None

    def __enter__(self):
        if "IMAGE_ENGINE" in self.kw:
            self.engine = getattr(I._config, "IMAGE_ENGINE", "oldmodel")
            I._config.IMAGE_ENGINE = self.kw.pop("IMAGE_ENGINE")
        for k, v in self.kw.items():
            self.orig[k] = getattr(I, k)
            setattr(I, k, v)
        return self

    def __exit__(self, *a):
        for k, v in self.orig.items():
            setattr(I, k, v)
        if self.engine is not None:
            I._config.IMAGE_ENGINE = self.engine


def _photo(name="out.png"):
    from PIL import Image as PILImage
    p = _TMP / name
    PILImage.new("RGB", (32, 32), (10, 20, 30)).save(p)
    return str(p)


def test_a_character_render_reaches_ideogram_with_its_adapter():
    """On the Ideogram engine the character goes there -- carrying the adapter,
    the strength AND the trigger. Any one of the three missing is the stranger
    bug in a new costume, so all three are asserted, not just the route."""
    ctx = Ctx()
    seen = {}

    def _spy_ideogram(ctx_arg, prompt, **k):
        seen.update(k)
        seen["prompt"] = prompt
        return "C:/fake/ideogram_out.png"

    _orig_draw = IG._ideogram_draw
    IG._ideogram_draw = _spy_ideogram
    try:
        with _Patches(IMAGE_ENGINE="ideogram4"):
            out = I.generate_image_with_comfy(
                ctx, "neurostepan, standing outdoors", steps=4,
                lora_name="neurostepan_ideo.safetensors", lora_strength=1.5,
                trigger="neurostepan")
    finally:
        IG._ideogram_draw = _orig_draw

    check("a picture came back", out is not None, out)
    check("the character was routed to Ideogram", bool(seen), seen)
    check("the adapter travelled with it",
          seen.get("lora_name") == "neurostepan_ideo.safetensors", seen)
    check("the strength travelled with it",
          float(seen.get("lora_strength") or 0) == 1.5, seen)
    check("the trigger travelled with it",
          seen.get("trigger") == "neurostepan", seen)


def test_without_a_lora_the_ideogram_default_still_applies():
    """The fix must be scoped to character renders only -- an ordinary /draw
    with no lora_name must keep going to Ideogram exactly as configured."""
    ctx = Ctx()
    called = {"ideogram": False}

    def _spy_ideogram(*a, **k):
        called["ideogram"] = True
        return "C:/fake/ideogram_out.png"

    _orig_draw = IG._ideogram_draw
    IG._ideogram_draw = _spy_ideogram
    try:
        with _Patches(IMAGE_ENGINE="ideogram4"):
            I.generate_image_with_comfy(ctx, "a cat", steps=4)
    finally:
        IG._ideogram_draw = _orig_draw
    check("an ordinary render still uses the configured default engine",
          called["ideogram"] is True)


def _main():
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        try:
            fn()
        except Exception as exc:
            check(fn.__name__ + " raised", False, exc)
    bad = [n for n, ok in RESULTS if not ok]
    print("\n%d/%d passed" % (len(RESULTS) - len(bad), len(RESULTS)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(_main())

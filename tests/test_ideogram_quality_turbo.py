"""The "Максимум" (ultra) image-quality tier is Ideogram's turbo-LoRA OFF switch.

Every other tier leaves steps/cfg unset, so ideogram.generate's own gate
(`turbo = steps is None and cfg is None`) attaches the turbo LoRA at
IDEOGRAM_STEPS_TURBO/IDEOGRAM_CFG_TURBO. Picking Максимум is the one button
that asks for the plain model instead, at IDEOGRAM_STEPS/IDEOGRAM_CFG -- the
same request the user made for music ("режим турбо лоры как и у музыки"),
exposed the same way ("как кнопку"): by reusing the existing quality picker,
not a new toggle.

Run: venv/Scripts/python.exe tests/test_ideogram_quality_turbo.py
"""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
import config
import image_generate as IG

_n = _bad = 0


def check(label, cond, detail=""):
    global _n, _bad
    _n += 1
    if cond:
        print(f"  ok   {label}")
    else:
        _bad += 1
        print(f"  FAIL {label}\n         {detail}")


class _Ctx:
    def __init__(self, image_quality=""):
        self.image_quality = image_quality
        self.is_cancelled = lambda: False


_prev_engine = IG._config.IMAGE_ENGINE
IG._config.IMAGE_ENGINE = "ideogram4"

_calls = []
_prev_draw = IG._ideogram_draw


def _spy_draw(ctx, prompt, **kw):
    _calls.append(kw)
    return "/tmp/fake.png"


IG._ideogram_draw = _spy_draw
IG._image.save_layout_for = lambda *a, **kw: None
IG._image._ENGINE_JUDGED = {}
IG._image._GENERATE_FAILURE = {}

print("\nEVERY TIER BUT ULTRA LEAVES STEPS/CFG UNSET (TURBO GATE FIRES)")
for tier in ("", "draft", "standard", "high"):
    _calls.clear()
    IG.generate_image_with_comfy(_Ctx(tier), "a red apple", width=512, height=512)
    got = _calls[0] if _calls else {}
    check(f"tier {tier!r} passes steps=None",
          got.get("steps") is None, got)
    check(f"tier {tier!r} passes cfg=None",
          got.get("cfg") is None, got)

print("\nULTRA ASKS FOR THE PLAIN MODEL AT FULL QUALITY")
_calls.clear()
IG.generate_image_with_comfy(_Ctx("ultra"), "a red apple", width=512, height=512)
got = _calls[0] if _calls else {}
check("ultra passes the full-quality step count",
      got.get("steps") == config.IDEOGRAM_STEPS, got)
check("ultra passes the full-quality cfg",
      got.get("cfg") == config.IDEOGRAM_CFG, got)

IG._ideogram_draw = _prev_draw
IG._config.IMAGE_ENGINE = _prev_engine

print(f"\n{_n - _bad}/{_n} checks passed")
sys.exit(1 if _bad else 0)

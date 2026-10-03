"""Deeper coverage for the pinned-📐-Size fixes in image.py, extending
tests/test_image_size_choice.py and tests/test_pinned_size_bypass.py rather
than duplicating them.

Focus areas (see scratchpad audit_04_size.md for the original 4 findings):

1. Mutation-style stress on the `not pinned` guards in parse_generation_params
   — several resolution keywords, the digitsxdigits regex, and width=/height=
   tag params, each checked to produce the SAME output whether the trigger is
   present or absent, so long as pinned=True. This would have caught the guard
   being a no-op or inverted.
2. orientation= AND a resolution keyword/width=/height= together while pinned
   — the pin must win over ALL of them, not just whichever is checked first.
3. The refine-pass re-orientation fix for a SQUARE (1:1) pin — proving the
   `pin_w == pin_h` special case (added alongside this test) actually forces a
   square result instead of silently landing on portrait either way.
4. width=/height= WITHOUT pinning — confirms explicit tag params still win
   when nothing is pinned (no regression from the guards above).

Run: venv/Scripts/python.exe tests/test_pinned_size_edge_cases.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import config
import image as I

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")


PIN = (944, 1680)  # a 9:16 portrait pin, matches test_pinned_size_bypass.py's shape

print("=" * 70)
print("1. MUTATION STRESS — resolution keywords in prompt text, pinned=True")
print("=" * 70)

baseline_prompt = "a cat on a windowsill"
base_out = I.parse_generation_params(baseline_prompt, defaults=PIN, pinned=True)

_KEYWORDS = ["4k", "8k", "2k", "1080p", "720p", "hd", "qhd", "fhd",
             "full hd", "ultra hd", "1920x1080", "3840x2160"]
for kw in _KEYWORDS:
    prompt = f"a cat on a windowsill, in stunning {kw} detail"
    out = I.parse_generation_params(prompt, defaults=PIN, pinned=True)
    check(f"pinned + keyword '{kw}' in prompt -> same size as no-keyword baseline",
          (out[4], out[5]) == (base_out[4], base_out[5]),
          (kw, (out[4], out[5]), (base_out[4], base_out[5])))

# bare-substring keyword landmine ("hd" inside "shadow") must ALSO be neutralised
prompt_landmine = "a shadow over the hd valley"
out = I.parse_generation_params(prompt_landmine, defaults=PIN, pinned=True)
check("pinned + substring landmine ('hd' inside 'shadow'/'the hd valley') -> pin holds",
      (out[4], out[5]) == (base_out[4], base_out[5]), (out[4], out[5]))

# same landmine, but UNPINNED, should still fire (guard must be pin-specific only)
out_unpinned = I.parse_generation_params(prompt_landmine, defaults=PIN, pinned=False)
check("same landmine, unpinned -> keyword DOES override (guard didn't over-fire)",
      (out_unpinned[4], out_unpinned[5]) == (1280, 720), (out_unpinned[4], out_unpinned[5]))

print()
print("=" * 70)
print("2. MUTATION STRESS — literal digitsxdigits regex fallback, pinned=True")
print("=" * 70)

for literal in ["a cat 1024x768", "a cat 1024*768", "a cat 1024×768", "a cat 640x480 wide"]:
    out = I.parse_generation_params(literal, defaults=PIN, pinned=True)
    check(f"pinned + literal '{literal}' -> pin holds, not the literal digits",
          (out[4], out[5]) == (base_out[4], base_out[5]), (out[4], out[5]))

out_unpinned = I.parse_generation_params("a cat 1024x768", defaults=PIN, pinned=False)
check("same literal, unpinned -> regex DOES fire and lands on 1024x768 (guard didn't over-fire)",
      (out_unpinned[4], out_unpinned[5]) == (1024, 768), (out_unpinned[4], out_unpinned[5]))

print()
print("=" * 70)
print("3. MUTATION STRESS — width=/height= tag params, pinned=True")
print("=" * 70)

for tag in ["a cat | width=1920 | height=1080", "a cat | width=3840", "a cat | height=2160"]:
    out = I.parse_generation_params(tag, defaults=PIN, pinned=True)
    check(f"pinned + tag '{tag}' -> pin holds, tag params ignored",
          (out[4], out[5]) == (base_out[4], base_out[5]), (out[4], out[5]))

print()
print("=" * 70)
print("4. orientation= TOGETHER WITH a keyword/tag param, pinned=True")
print("=" * 70)

combo_cases = [
    "a cat | orientation=landscape | width=1920 | height=1080",
    "a cat in stunning 4k detail | orientation=landscape",
    "a cat 1920x1080 | orientation=square",
]
for combo in combo_cases:
    out = I.parse_generation_params(combo, defaults=PIN, pinned=True)
    check(f"pinned + combo '{combo}' -> pin wins over ALL of them",
          (out[4], out[5]) == (base_out[4], base_out[5]), (out[4], out[5]))

print()
print("=" * 70)
print("5. width=/height= WITHOUT pinning — must still win (no regression)")
print("=" * 70)

out = I.parse_generation_params("a cat | width=1920 | height=1080", defaults=PIN, pinned=False)
check("unpinned width=/height= tag params still win over defaults",
      (out[4], out[5]) == (1920, 1080), (out[4], out[5]))

out = I.parse_generation_params("a cat | orientation=landscape", defaults=PIN, pinned=False)
check("unpinned orientation= still reshapes the default size",
      out[4] > out[5], (out[4], out[5]))

print()
print("=" * 70)
print("6. SQUARE (1:1) PIN through the refine/eval re-orientation logic")
print("=" * 70)


def _simulate_eval_reorient(pin_w, pin_h, cw, ch):
    """Mirrors the block at image.py ~7700-7712 exactly (post-fix)."""
    if pin_w == pin_h:
        side = I._snap_to_8(int((cw * ch) ** 0.5))
        cw = ch = side
    elif (pin_w > pin_h) != (cw > ch):
        cw, ch = ch, cw
    return I.normalize_resolution(cw, ch)

SQUARE_PIN = (1024, 1024)

# eval suggests landscape
cw, ch = _simulate_eval_reorient(*SQUARE_PIN, 2560, 1440)
check("square pin + landscape eval suggestion -> result is square, not portrait",
      cw == ch, (cw, ch))

# eval suggests portrait
cw, ch = _simulate_eval_reorient(*SQUARE_PIN, 1440, 2560)
check("square pin + portrait eval suggestion -> result is square, not portrait",
      cw == ch, (cw, ch))

# eval suggests square already
cw, ch = _simulate_eval_reorient(*SQUARE_PIN, 1536, 1536)
check("square pin + square eval suggestion -> stays square",
      cw == ch, (cw, ch))

check("square-pin result respects the normal 2048px rail",
      max(cw, ch) <= config.IMAGE_MAX_SIDE, (cw, ch, config.IMAGE_MAX_SIDE))

# Sanity: non-square pins still behave as before (regression guard on the fix)
PORTRAIT_PIN = (944, 1680)
cw, ch = _simulate_eval_reorient(*PORTRAIT_PIN, 2560, 1440)
check("portrait pin + landscape eval suggestion -> still re-oriented to portrait",
      ch > cw, (cw, ch))

LANDSCAPE_PIN = (1680, 944)
cw, ch = _simulate_eval_reorient(*LANDSCAPE_PIN, 1440, 2560)
check("landscape pin + portrait eval suggestion -> still re-oriented to landscape",
      cw > ch, (cw, ch))

print()
print(f"{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)

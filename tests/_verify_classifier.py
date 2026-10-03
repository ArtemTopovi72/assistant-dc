"""Empirical routing check for classify_edit_intent — ambiguous EN+RU prompts.
No ComfyUI needed (pure regex). Prints every mismatch."""
import os, sys
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from image import classify_edit_intent as C

# (prompt, expected_category)
CASES = [
    # --- unambiguous anchors ---
    ("upscale the image 2x",                         "upscale"),
    ("make it higher resolution",                    "upscale"),
    ("increase resolution to 4k",                    "upscale"),
    ("remove the background",                         "background_remove"),
    ("make the background transparent",              "background_remove"),
    ("change the background to a beach",             "background_replace"),
    ("replace background with a studio",             "background_replace"),
    ("remove the person on the left",               "person_remove"),
    ("erase the trash can",                          "object_remove"),
    ("delete her glasses",                           "object_remove"),
    ("remove the watermark",                         "object_remove"),
    ("add a hat on his head",                        "object_insert"),
    ("put a coffee cup on the table",               "object_insert"),
    ("insert a dog next to her",                     "object_insert"),
    ("make it look like an anime",                  "style_transfer"),
    ("turn this into an oil painting",              "style_transfer"),
    ("relight with golden hour light",             "relight"),
    ("add dramatic cinematic lighting",            "relight"),
    ("expand the canvas to the left",              "outpaint"),
    ("zoom out to show more scene",                "outpaint"),
    ("restore this old photo",                      "restore"),
    ("fix the scratches and denoise",              "restore"),
    # --- targeted region edits ---
    ("make her smile",                              "face_edit"),
    ("change the hair to blue",                     "face_edit"),
    ("make the jacket leather",                     "clothing_edit"),
    ("change the dress to red",                     "clothing_edit"),
    # --- AMBIGUOUS / first-match-false-positive candidates ---
    ("make the dress bigger",                       "clothing_edit"),   # not upscale
    ("make her eyes bigger",                        "face_edit"),       # not upscale
    ("make the logo on the bottle larger",         "product_edit"),    # not upscale
    # --- Russian ---
    ("убери фон",                                    "background_remove"),
    ("замени фон на пляж",                          "background_replace"),
    ("удали человека справа",                       "person_remove"),
    ("сделай волосы синими",                        "face_edit"),
    ("добавь шляпу на голову",                      "object_insert"),
    ("увеличь разрешение",                          "upscale"),
    # --- multi-op ---
    ("remove the background and add a hat",         "multi_op"),
    ("erase the car then change the sky to sunset", "multi_op"),
]

fails = []
for prompt, exp in CASES:
    got = C(prompt)
    mark = "OK " if got == exp else "XX "
    if got != exp:
        fails.append((prompt, exp, got))
    print(f"{mark}{got:<18} (exp {exp:<18}) :: {prompt}")

print(f"\n{len(CASES)-len(fails)}/{len(CASES)} correct")
if fails:
    print("MISROUTES:")
    for p, e, g in fails:
        print(f"  {p!r}: expected {e}, got {g}")
    sys.exit(1)
print("ALL ROUTES CORRECT")

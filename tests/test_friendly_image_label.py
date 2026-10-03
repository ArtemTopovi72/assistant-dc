"""tg_bot._friendly_image_label: the "Which picture?" picker showed raw
tool-call strings as button text.

Live, 2026-09-19: pressing a style preset registers the picture with
label=task.user_text -- for a forced button press that is the literal
scaffolding the graph executes ("[style_preset] call redraw_image with
mode=\"redraw\" instructions=\"...\" on the current image. Do not generate a
new image."), not anything meant for a human to read. The picker showed six
entries as "[style_preset] call redraw_image with mo…", indistinguishable
from each other.

Run: venv/Scripts/python.exe tests/test_friendly_image_label.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
os.environ["F5_TEST_RUN"] = "1"

import tg_bot as T

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")

F = T._friendly_image_label

check("style preset call unwraps to its instructions, not the boilerplate",
      F('[style_preset] call redraw_image with mode="redraw" '
        'instructions="convert to a cyberpunk neon illustration, glowing '
        'magenta and cyan rim lighting" on the current image. Do not '
        'generate a new image.') == "convert to a cyberpunk neon…",
      F('[style_preset] call redraw_image with mode="redraw" instructions="x" '
        'on the current image. Do not generate a new image.'))

check("animate call unwraps to its description",
      F('[animate] call generate_video with description="The girl nose '
        'becomes comically small" using the current image.')
      == "The girl nose becomes comically small")

check("bare-parameter upscale call falls back to the tag name",
      F("[upscale] call redraw_image with mode='upscale' scale='4x' "
        "on the current image. Do not generate a new image.") == "Upscale 4×")

check("bare-parameter enhance call falls back to the tag name",
      F("[enhance] call redraw_image with mode='enhance' on the current "
        "image to fix and enhance the faces.") == "Enhance faces")

check("bare-parameter restore call falls back to the tag name",
      F("[restore] call redraw_image with mode='restore' on the current "
        "image to repair and enhance quality.") == "Restore")

check("the change-outfit prefix is stripped, the user's own words kept",
      F("change the outfit to: the hat into a cap") == "the hat into a cap")

check("the animate-photo prefix is stripped",
      F("animate this photo: something funny happens")
      == "something funny happens")

check("plain text with no tag/prefix passes through unchanged",
      F("image") == "image")

check("empty label stays empty (caller falls back to img_unlabelled)",
      F("") == "")

check("a long plain sentence truncates on a word boundary with an ellipsis",
      F("The girl's nose has comically shrunk, and she herself is smiling")
      == "The girl's nose has comically shrunk…",
      F("The girl's nose has comically shrunk, and she herself is smiling"))

print(f"\n{OK}/{OK + BAD} checks passed")
if __name__ == "__main__":
    sys.exit(0 if BAD == 0 else 1)

"""Ask the outer image judge (evaluate_image) about finished renders, N times.

Live, 2026-09-12: the judge gave 5/10, 4/10 and 5/10 to three flawless
"ginger cat asleep on a windowsill, winter outside" renders in a row, so a
plain scene took three full renders plus an agent inspect/edit/redraw cycle
(8 minutes) and was then delivered with an apology for a flaw that was not
there. This shows what the judge actually says. Vision calls only; no render.

    venv/Scripts/python.exe bench/judge_probe.py <image> [<image>...] --n 3
"""
import argparse
import json
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.WARNING)

from config import MODEL_NAME
import image_generate as G

GOAL = ("A cozy scene of a fluffy ginger cat sleeping deeply on a wooden windowsill. "
        "Outside the window, there is a beautiful winter landscape with snow-covered "
        "trees and soft falling snowflakes under a pale winter sky.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--goal", default=GOAL)
    a = ap.parse_args()
    ctx = types.SimpleNamespace(model_name=MODEL_NAME, no_think=False,
                                reasoning_effort="high", is_cancelled=lambda: False,
                                set_stage=lambda *_: None)
    for img in a.images:
        print("\n==", img)
        for i in range(a.n):
            d = G.evaluate_image(ctx, img, a.goal, a.goal, "", 8, 1.0, 960, 544)
            print(f"  {i+1}. {d.get('verdict')} {d.get('score')}/10  reason={d.get('reason','')[:220]!r}"
                  f"  patch={d.get('prompt_patch','')[:80]!r} neg={d.get('negative_prompt_patch','')[:80]!r}")


if __name__ == "__main__":
    main()

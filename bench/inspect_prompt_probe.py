"""Ask the inspector about a finished picture with different prompts.

Live, 2026-09-12: inspect_image said "cat: MISSING, green sofa: MISSING" about
a render that plainly shows a black cat on a green sofa, while an open
"list the objects" question named them. Same model, same picture, same
encoding -- so the prompt is the variable. This compares them.

    venv/Scripts/python.exe bench/inspect_prompt_probe.py <image> --check "Is the cat black? ..."
"""
import argparse
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.WARNING)

from config import MODEL_NAME
from llm import analyze_image_with_llm
from prompts import IMAGE_INSPECT_PROMPT
import exposure

DESCRIBE_FIRST = """You are a careful image inspector. You receive an image and a checklist.

Step 1 -- DESCRIBE: in two or three sentences, say what you actually see in the picture: the setting, every object, its colour and where it is. Look at the whole frame, including small things.
Step 2 -- CHECK: for every element in the checklist, one short line: the element, then PRESENT / PARTIAL / MISSING / DISTORTED, then the defect if any. An element counts as PRESENT when you described it in step 1.

Judge the pixels, not the request. Plain text, no markdown."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("--check", required=True); ap.add_argument("--n", type=int, default=2)
    a = ap.parse_args()
    ctx = types.SimpleNamespace(model_name=MODEL_NAME, no_think=False, reasoning_effort="high",
                                is_cancelled=lambda: False, set_stage=lambda *_: None)
    ev = exposure.evidence(a.image)
    for name, sp, q in (("current", IMAGE_INSPECT_PROMPT, a.check + "\n\n" + ev),
                        ("describe-first", DESCRIBE_FIRST, a.check + "\n\n" + ev),
                        ("open", "You describe pictures accurately.", "List every object you see, with its colour and position.")):
        for i in range(a.n):
            r = analyze_image_with_llm(ctx=ctx, image_path=a.image, user_text=q, system_prompt=sp, max_tokens=400) or ""
            print(f"{name:15s} #{i}: {r.strip()[:360]!r}")


if __name__ == "__main__":
    main()

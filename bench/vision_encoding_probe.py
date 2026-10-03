"""Does the vision model see the picture we send, or a dark one?

Live, 2026-09-12: on every plain render the outer judge and inspect_image said
"extremely dark / underexposed" about pictures that measure normally exposed,
and the code now overrules them. This asks the SAME question about the same
picture in several encodings (raw PNG, JPEG, downscaled) to find out whether
the darkness is a property of the model or of what the wire carries.

    venv/Scripts/python.exe bench/vision_encoding_probe.py <image> [--n 2]
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

from PIL import Image
from config import MODEL_NAME
from llm import analyze_image_with_llm
import exposure

Q = ("First word of your answer: LIT or DARK -- is this picture normally lit or "
     "dark/underexposed? Then, in one sentence, list the main objects you see.")


def variants(path):
    im = Image.open(path).convert("RGB")
    out = [("png-raw", None)]
    for name, size, fmt, q in (("jpeg-full", None, "JPEG", 92), ("jpeg-768", 768, "JPEG", 92),
                               ("png-768", 768, "PNG", None), ("jpeg-512", 512, "JPEG", 90)):
        v = im.copy()
        if size:
            v.thumbnail((size, size))
        tmp = os.path.join(os.path.dirname(path), f"_probe_{name}.{fmt.lower()}")
        v.save(tmp, fmt, **({"quality": q} if q else {}))
        out.append((name, tmp))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("--n", type=int, default=2)
    a = ap.parse_args()
    ctx = types.SimpleNamespace(model_name=MODEL_NAME, no_think=False, reasoning_effort="high",
                                is_cancelled=lambda: False, set_stage=lambda *_: None)
    print("measured:", exposure.measure(a.image))
    for name, data in variants(a.image):
        for i in range(a.n):
            r = analyze_image_with_llm(ctx=ctx, image_path=data or a.image, user_text=Q,
                                       system_prompt="You describe pictures accurately.",
                                       max_tokens=200) or ""
            print(f"{name:10s} #{i}: {r.strip()[:220]!r}")


if __name__ == "__main__":
    main()

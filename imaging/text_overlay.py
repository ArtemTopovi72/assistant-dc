"""Exact lettering laid onto a picture with PIL -- the fallback when a model
cannot spell the words.

Night bench 2026-09-24: FireRed painted "УРОЖАЙ 2025" for "УРОЖАЙ 2026" in
21 renders out of 21 -- every seed, every wording, with and without its
typography LoRA. Reseeding cannot beat a prior. Drawn type is never wrong.
"""
from __future__ import annotations

import os
import re
import time
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

_FONTS = ("seguibl.ttf", "arialbd.ttf", "bahnschrift.ttf", "DejaVuSans-Bold.ttf")


def _font(size: int):
    for name in _FONTS:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def placement(instruction: str) -> str:
    """top / middle / bottom from the wording; the sky and 'сверху' mean top."""
    import intent
    return intent.ask_choice("A user asked to put text on a picture: {text}. Where should "
                             "the text go?", instruction or "", ("top", "middle", "bottom"), "top")


def overlay(image_path: str, lines: list, *, where: str = "top",
            out_dir: str = "", colour=(255, 255, 255)) -> Optional[str]:
    """Draw `lines` centred, as wide as ~80% of the frame, with a dark outline so
    it reads on any background. Returns the new file path."""
    lines = [l for l in (lines or []) if l and l.strip()]
    if not lines or not os.path.exists(image_path):
        return None
    im = Image.open(image_path).convert("RGB")
    W, H = im.size
    draw = ImageDraw.Draw(im)
    # the largest size at which the widest line fits 80% of the width and the
    # block fits a third of the height
    size = max(12, H // 6)
    while size > 12:
        f = _font(size)
        widths = [draw.textbbox((0, 0), l, font=f)[2] for l in lines]
        block = sum(draw.textbbox((0, 0), l, font=f)[3] for l in lines) * 1.15
        if max(widths) <= W * 0.8 and block <= H / 3:
            break
        size = int(size * 0.92)
    f = _font(size)
    heights = [draw.textbbox((0, 0), l, font=f)[3] for l in lines]
    block = int(sum(heights) * 1.15)
    y = {"top": int(H * 0.06), "middle": (H - block) // 2,
         "bottom": H - block - int(H * 0.06)}[where if where in ("top", "middle", "bottom") else "top"]
    stroke = max(2, size // 14)
    for l, h in zip(lines, heights):
        w = draw.textbbox((0, 0), l, font=f)[2]
        draw.text(((W - w) // 2, y), l, font=f, fill=colour,
                  stroke_width=stroke, stroke_fill=(20, 20, 20))
        y += int(h * 1.15)
    out_dir = out_dir or os.path.dirname(image_path)
    out = os.path.join(out_dir, f"text-overlay_{int(time.time() * 1000)}.png")
    im.save(out)
    return out

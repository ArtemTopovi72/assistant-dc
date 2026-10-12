"""🔁 Swap a person: the people found on a picture as a numbered collage, and the words
that point an edit at the one picked.

The owner wanted Stalin's erasing of Yezhov, but funnier: pick a person, send a picture
of anyone (SpongeBob), get them in that spot in the photo's own style.
"""
import os
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont

Box = Tuple[int, int, int, int]
MAX_PEOPLE = 9                  # one row of digit buttons


def people(path: str) -> List[Box]:
    """The face boxes of the people on the picture, left to right (the order the collage
    numbers them in, so «2» is the second from the left on the picture too)."""
    import identity_metrics
    boxes = identity_metrics.face_boxes(path, pad=0.0, max_faces=MAX_PEOPLE)
    return sorted(boxes, key=lambda b: b[0])


def _font(size: int):
    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def collage(path: str, boxes: List[Box], out: str, tile: int = 320) -> str:
    """Each person as a head-and-shoulders tile with its number in a corner badge."""
    im = Image.open(path).convert("RGB")
    W, H = im.size
    tiles = []
    for n, (x0, y0, x1, y1) in enumerate(boxes, 1):
        fw, fh = x1 - x0, y1 - y0
        cx = (x0 + x1) / 2
        side = max(fw, fh) * 2.4
        box = (int(max(0, cx - side / 2)), int(max(0, y0 - fh * 0.7)),
               int(min(W, cx + side / 2)), int(min(H, y0 - fh * 0.7 + side)))
        t = im.crop(box).resize((tile, tile))
        d = ImageDraw.Draw(t)
        r = tile // 6
        d.ellipse((8, 8, 8 + 2 * r, 8 + 2 * r), fill=(230, 40, 40), outline="white", width=4)
        f = _font(int(r * 1.3))
        d.text((8 + r, 8 + r), str(n), fill="white", font=f, anchor="mm")
        tiles.append(t)
    sheet = Image.new("RGB", (tile * len(tiles) + 8 * (len(tiles) - 1), tile), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * (tile + 8), 0))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    sheet.save(out, quality=92)
    return out


def where(path: str, box: Box, n: int, total: int) -> str:
    """Words that single out person `n` of `total` for an edit model."""
    with Image.open(path) as im:
        W, H = im.size
    x = (box[0] + box[2]) / 2 / W
    side = "on the left" if x < 0.36 else "on the right" if x > 0.64 else "in the middle"
    nth = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}.get(n, f"{n}th")
    return (f"the person {side} of the picture (the {nth} of the {total} people counted from "
            f"the left; their face is about {round(x * 100)}% from the left edge and "
            f"{round((box[1] + box[3]) / 2 / H * 100)}% from the top)")

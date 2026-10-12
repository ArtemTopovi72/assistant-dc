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
    # 0.4, not the default 0.6: Molotov in profile under a hat was missed on the Stalin
    # photo; a stray extra tile in a picker costs nothing, a missing person does.
    boxes = identity_metrics.face_boxes(path, pad=0.0, max_faces=MAX_PEOPLE, score_threshold=0.4)
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


def region(size: Tuple[int, int], box: Box) -> Box:
    """The picked person's own part of the picture: their face box grown to the shoulders
    and down to the bottom (a standing figure), never the neighbours' middles."""
    W, H = size
    x0, y0, x1, y1 = box
    fw, fh = x1 - x0, y1 - y0
    cx = (x0 + x1) / 2
    half = fw * 2.3
    return (int(max(0, cx - half)), int(max(0, y0 - fh * 1.0)), int(min(W, cx + half)), H)


def swap(ctx, path: str, box: Box, ref: str, wish: str = "") -> str:
    """The person at `box` replaced by the character of `ref`, in the picture's own
    style; returns the saved picture or "". Through the agent's transfer_image the
    whole photo was re-rendered and coloured, Yezhov stayed a man in another uniform,
    and the reply said «done» (live 2026-10-12): here only the person's own region is
    redrawn and pasted back, so the rest of the photo cannot change."""
    import tempfile
    import image as image_mod
    from PIL import ImageFilter
    im = Image.open(path).convert("RGB")
    rx0, ry0, rx1, ry1 = region(im.size, box)
    crop = im.crop((rx0, ry0, rx1, ry1))
    k = max(1.0, (1024 * 1024 / (crop.width * crop.height)) ** 0.5)
    big = crop.resize((int(crop.width * k) // 8 * 8, int(crop.height * k) // 8 * 8), Image.LANCZOS)
    tmp = tempfile.mkdtemp(prefix="swap_")
    src = os.path.join(tmp, "region.png")
    big.save(src)
    grey = _is_grey(im)
    look = ("a black-and-white old photograph with its grain, contrast and softness"
            if grey else "the same photograph, with its light, colours, grain and sharpness")
    who = _who(ctx, ref) or "the character from picture 2"
    # Two steps: asked to replace in one go, FireRed kept the man and stuck a small flat
    # SpongeBob on his coat (live 2026-10-12). First the person goes, then the newcomer
    # is drawn into the empty spot at a person's size.
    empty = image_mod.edit_image_with_firered(
        ctx, src, "Remove the person standing in the middle of the picture completely -- body, "
        "head, cap, coat, shadow -- and fill the spot with what is behind them, continuing the "
        "background seamlessly. Everything else stays exactly as it is.", save_prefix="swap_region")
    if not empty or not os.path.exists(empty):
        return ""
    # «as tall as a grown man» grew a man's head on top of SpongeBob: the size is said as
    # the spot to fill, and the character stays wholly itself.
    # The user's wishes («в шинели и ростом с человека, ехидно улыбается») come first, in
    # English like the rest; the character's own face is kept whatever it wears.
    if wish:
        import image_grounding
        wish = image_grounding._english_instructions(ctx, wish) or wish
    blend = (" It blends in seamlessly: rendered as part of " + look
             + (", fully black-and-white" if grey else "") + ", with the scene's light direction, "
             "shadows on the ground, depth of field, grain and softness, as if photographed there "
             "together with the others, not pasted on. Everything else stays exactly as it is.")
    # No «at the scale of the people beside it»: the region holds none, and FireRed drew one.
    instr = ("Put " + who + " (picture 2) standing in the middle of picture 1, big, filling the "
             "empty spot from the middle of the picture down to its bottom edge, full body, "
             "standing on the ground. It is ONLY that character, wholly as in picture 2: its own "
             "shape, its own face and features, no human head, body or clothes added." + blend)
    out = image_mod.edit_image_with_firered(ctx, empty, instr, reference_paths=[ref],
                                            save_prefix="swap_region")
    if not out or not os.path.exists(out):
        return ""
    # Wishes are a third edit of the character already standing there: said in the same
    # edit, «a person-sized figure in a greatcoat» drew a real man next to a small SpongeBob
    # twice (live 2026-10-12); an existing figure is changed, not a new one added.
    # Any «person», «man», «human» in the words turned SpongeBob into a young man even here:
    # the wish is retold as changes to the character's own body.
    if wish:
        name = who.split(",")[0]
        try:
            from llm import call_llm_simple
            told = " ".join((call_llm_simple(
                None, "Retell this wish as changes to the cartoon character " + name + " itself, "
                "in English, one sentence: its own body, its own face. Say any size as the "
                "character's own body stretched taller or bigger (e.g. 'its sponge body stretched "
                "three times taller'); NEVER use the words person, people, man, human, figure or "
                "adult. Clothes are worn over its own body. Reply with the sentence only.",
                wish, max_tokens=120) or "").split())
        except Exception:
            told = ""
        changed = image_mod.edit_image_with_firered(
            ctx, out, "Change only the cartoon " + who + " in picture 1, keeping it exactly that "
            "cartoon character from picture 2 -- its own head, face, eyes and colours: "
            + (told or wish) + " Nothing human about it." + blend,
            reference_paths=[ref], save_prefix="swap_region")
        if changed and os.path.exists(changed):
            out = changed
    new = Image.open(out).convert("RGB").resize(crop.size, Image.LANCZOS)
    if grey:
        new = new.convert("L").convert("RGB")
    # FireRed brightens what it redraws: the region came back lighter than the photo around
    # it, a visible patch -- its tone is matched to the crop it replaces.
    import numpy as np
    a, c = np.asarray(new, dtype=np.float32), np.asarray(crop, dtype=np.float32)
    a = (a - a.mean((0, 1))) / (a.std((0, 1)) + 1e-6) * c.std((0, 1)) + c.mean((0, 1))
    new = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    band = max(6, int(min(crop.size) * 0.06))
    mask = Image.new("L", crop.size, 0)
    # the bottom edge is the picture's own edge when the region reaches it: no fade there
    mask.paste(255, (band, band, crop.width - band, crop.height - (0 if ry1 == im.height else band)))
    im.paste(new, (rx0, ry0), mask.filter(ImageFilter.GaussianBlur(band / 2)))
    res = os.path.join(os.path.dirname(out), "swap_" + os.path.basename(path).rsplit(".", 1)[0] + ".png")
    im.save(res)
    return res


def _who(ctx, ref: str) -> str:
    """The character of the reference in a few words («SpongeBob SquarePants, a yellow
    square sea sponge with big eyes»): shown only as picture 2, FireRed drew another
    man in Yezhov's uniform."""
    try:
        from llm import analyze_image_with_llm
        return " ".join((analyze_image_with_llm(
            ctx, image_path=ref, temperature=0.1,
            user_text="Who or what is the main character in this picture? Answer with their name "
                      "if they are a known character or person, then a few words on how they "
                      "look (shape, colours). One short line, nothing else.") or "").split())[:200]
    except Exception:
        return ""


def _is_grey(im) -> bool:
    import numpy as np
    a = np.asarray(im.resize((128, 128)), dtype=np.int16)
    return float(np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean()) < 6


def where(path: str, box: Box, n: int, total: int) -> str:
    """Words that single out person `n` of `total` for an edit model."""
    with Image.open(path) as im:
        W, H = im.size
    x = (box[0] + box[2]) / 2 / W
    # By rank first: Yezhov, the rightmost of four, has his face at 63% and was «in the middle».
    side = ("on the far left" if n == 1 and total > 1 else "on the far right" if n == total > 1
            else "on the left" if x < 0.36 else "on the right" if x > 0.64 else "in the middle")
    nth = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}.get(n, f"{n}th")
    return (f"the person {side} of the picture (the {nth} of the {total} people counted from "
            f"the left; their face is about {round(x * 100)}% from the left edge and "
            f"{round((box[1] + box[3]) / 2 / H * 100)}% from the top)")

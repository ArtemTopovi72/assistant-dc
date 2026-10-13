"""🔁 Swap a person: the people found on a picture as a numbered collage, and the words
that point an edit at the one picked.

The owner wanted Stalin's erasing of Yezhov, but funnier: pick a person, send a picture
of anyone (SpongeBob), get them in that spot in the photo's own style.
"""
from seams import yes_no, vision_model
import os
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont

Box = Tuple[int, int, int, int]
MAX_PEOPLE = 9                  # one row of digit buttons
CANDIDATES = 2                  # takes per swap; the vision model picks the best


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
    # Drawn on the whole photo the newcomer may stand a little aside of the old spot: the
    # paste area is wider (the changed-pixels mask below keeps the neighbours), or its
    # head was cut off at the region's edge.
    w = rx1 - rx0               # a figure a third wider was cut off at the edge (2026-10-13)
    core = (rx0, rx1)
    rx0, rx1 = max(0, rx0 - w), min(im.width, rx1 + w)
    ry0 = max(0, ry0 - (ry1 - ry0) // 4)
    band = max(6, int(min(rx1 - rx0, ry1 - ry0) * 0.06))
    mask = Image.new("L", (rx1 - rx0, ry1 - ry0), 0)
    # the bottom edge is the picture's own edge when the region reaches it: no fade there
    mask.paste(255, (band, band, rx1 - rx0 - band, ry1 - ry0 - (0 if ry1 == im.height else band)))
    mask = mask.filter(ImageFilter.GaussianBlur(band / 2))
    grey = _is_grey(im)
    look = ("a black-and-white old photograph with its grain, contrast and softness"
            if grey else "the same photograph, with its light, colours, grain and sharpness")
    who = _who(ctx, ref) or "the character from picture 2"
    # On the WHOLE photo, only the picked person's region pasted back: drawn on the
    # region alone the newcomer had no people to size itself by and came out a flat
    # cartoon apart from them (live 2026-10-12, «полное говно»).
    if wish:
        import image_grounding
        wish = image_grounding._english_instructions(ctx, wish) or wish
    k = (1.4e6 / (im.width * im.height)) ** 0.5
    src = os.path.join(tempfile.mkdtemp(prefix="swap_"), "photo.png")
    im.resize((int(im.width * k) // 8 * 8, int(im.height * k) // 8 * 8), Image.LANCZOS).save(src)
    # Short, like a hand-written prompt: the long rule lists made it a flat cartoon.
    real = False
    if wish:
        real = yes_no(ctx)("Does the user want the character turned into a real-life, "
                              "photographed physical version of itself, not kept as drawn?", wish)
    name = who.split(",")[0]
    # Asked for a real one, the reference is read by the VLM as a real thing and drawn
    # from those words, picture 2 not given: given, FireRed copied the drawing whatever
    # the words said -- no greatcoat, no size, a cartoon or a 3D toy even after an
    # Anything2Real pass (live via the agent, 2026-10-13); from words alone it drew a
    # real porous sponge in a greatcoat (2026-10-12, «ахуенно»).
    real_look = _real_look(ctx, ref) if real else ""
    finish = (" Photographed for real, part of " + look + ": exactly as sharp or soft, grainy "
              "and worn as the rest of the photo, no glow or halo around it.")
    # The place and the size drawn, not said: a red box round the man, head to the bottom
    # edge (Qwen-Image-Edit reads a red-marked area -- the community's mask trick). In
    # words, «where he stood» / «on the same ground» / «beside the man in the middle»
    # stood it beyond the embankment's railing, small on the railing, or giant-headed;
    # erased first, the pavement became water (live via the agent, 2026-10-13).
    marked = os.path.join(os.path.dirname(src), "marked.png")
    with Image.open(src) as e:
        e = e.convert("RGB")
        kx, ky = e.width / im.width, e.height / im.height
        x0, y0, x1, y1 = box
        fw, fh = x1 - x0, y1 - y0
        cx = (x0 + x1) / 2
        rb = (int((cx - fw * 1.8) * kx), int((y0 - fh * 0.35) * ky),
              int((cx + fw * 1.8) * kx), e.height - 2)
        ImageDraw.Draw(e).rectangle(rb, outline=(255, 0, 0), width=max(4, e.width // 250))
        e.save(marked)
    # The wish right after the figure: last in a long sentence, «in a greatcoat» was
    # dropped two runs of three.
    figure = (real_look or name + " from picture 2") + (" (" + wish.rstrip(".") + ")" if wish else "")
    # Several takes, the vision model picks one: the same prompt gave a greatcoat once and
    # a bare sponge on the water the next time, a man left with a sponge on his chest the
    # third (live via the agent, 2026-10-13) -- one take is a coin toss.
    takes = []
    # One hold of the card for all the takes: released between them, the chat model was
    # loaded back and evicted again for nothing (~10 s a take, 2026-10-13). No model call
    # in here; the pick comes after.
    import comfy_client
    with comfy_client.card_session("person swap"):
        for _ in range(CANDIDATES):
            if getattr(ctx, "is_cancelled", lambda: False)():
                break
            t = image_mod.edit_image_with_firered(
                ctx, marked, "Replace the man inside the red box with " + figure + ", standing "
                "full body and filling the box from its top edge to its bottom edge; nothing of "
                "the man is left. Then remove the red box completely." + finish,
                reference_paths=[] if real_look else [ref], save_prefix="swap_region")
            if t and os.path.exists(t):
                takes.append(t)
    if not takes:
        return ""
    out = _pick(ctx, takes, figure)
    new = Image.open(out).convert("RGB").resize(im.size, Image.LANCZOS)
    if grey:
        new = new.convert("L").convert("RGB")
    # FireRed brightens what it redraws: matched to the photo's own tone, then only the
    # picked person's region is taken.
    import numpy as np
    a, c = np.asarray(new, dtype=np.float32), np.asarray(im, dtype=np.float32)
    a = (a - a.mean((0, 1))) / (a.std((0, 1)) + 1e-6) * c.std((0, 1)) + c.mean((0, 1))
    soft = a[ry0:ry1, rx0:rx1]
    new = Image.fromarray(soft.clip(0, 255).astype(np.uint8))
    # Only what actually changed is pasted: the whole region with a feathered rim left a
    # grey halo round the figure where FireRed re-toned the background.
    d = np.abs(soft - c[ry0:ry1, rx0:rx1]).max(-1)
    changed = Image.fromarray(((d > 28) * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(9))
    changed = changed.filter(ImageFilter.MinFilter(5)).filter(ImageFilter.GaussianBlur(3))
    # ...but the person's own spot is taken whole: where the new figure's grey matched
    # Yezhov's, his nose and coat edge showed through the changed-pixels mask.
    own = Image.new("L", mask.size, 0)
    own.paste(255, (core[0] - rx0 + band, 0, core[1] - rx0 - band, mask.height))
    own = np.asarray(own.filter(ImageFilter.GaussianBlur(band)), dtype=np.float32)
    keep = np.minimum(np.maximum(np.asarray(changed, dtype=np.float32), own),
                      np.asarray(mask, dtype=np.float32))
    im.paste(new, (rx0, ry0), Image.fromarray(keep.astype(np.uint8)))
    res = os.path.join(os.path.dirname(out), "swap_" + os.path.basename(path).rsplit(".", 1)[0] + ".png")
    im.save(res)
    return res


def _who(ctx, ref: str) -> str:
    """The character of the reference in a few words («SpongeBob SquarePants, a yellow
    square sea sponge with big eyes»): shown only as picture 2, FireRed drew another
    man in Yezhov's uniform."""
    try:
        from llm import analyze_image_with_llm
        return " ".join((vision_model(ctx)(
            ctx, image_path=ref, temperature=0.1,
            user_text="Who or what is the main character in this picture? Answer with their name "
                      "if they are a known character or person, then a few words on how they "
                      "look (shape, colours). One short line, nothing else.") or "").split())[:200]
    except Exception:
        return ""


def _pick(ctx, takes: list, figure: str) -> str:
    """The take that best shows `figure` in the man's place, all shown in ONE vision call
    (side by side the judge compares; one by one it rates each the same)."""
    if len(takes) < 2:
        return takes[0]
    try:
        import re
        from llm import analyze_image_with_llm
        ans = vision_model(ctx)(
            ctx, temperature=0.0, max_tokens=20,
            image_paths=[(f"Take {i}:", t) for i, t in enumerate(takes, 1)],
            user_text="Each take replaced a man on an old photo with " + figure + ". Which take "
                      "shows it best: that figure, whole and as asked, in the man's place at a "
                      "person's size, nothing of the man left, looking photographed together "
                      "with the others? Answer with the take number only.") or ""
        m = re.search(r"\d+", ans)
        i = int(m.group()) if m else 1
        return takes[i - 1] if 1 <= i <= len(takes) else takes[0]
    except Exception:
        return takes[0]


def _real_look(ctx, ref: str) -> str:
    """The reference's character as a real thing in a photo, without its name («a giant
    real kitchen dish sponge, yellow and porous, with big round eyes and buck teeth»):
    the name alone brings the cartoon back. No expression: «wide-mouthed» from the reference
    beat the user's «angry», and the sponge came out scared (2026-10-13)."""
    try:
        from llm import analyze_image_with_llm
        return " ".join((vision_model(ctx)(
            ctx, image_path=ref, temperature=0.1,
            user_text="Describe the main character of this picture as if it were a REAL physical "
                      "thing photographed in the real world, WITHOUT its name: what real object "
                      "or creature it would be, its material and texture, colour, shape and its "
                      "face features -- but NOT its expression or mood (no smiling, scared, "
                      "wide-mouthed): the user's wish sets that. One noun phrase under 30 words, "
                      "starting with 'a'. Nothing else.") or "").split())[:300]
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

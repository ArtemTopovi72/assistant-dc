"""A web photo of a named real thing, as a reference for the drawing.

Klein knows the kind of thing, not the model: «ВАЗ 2107» came out as a Fiat, the «Урал»
sidecar as a fuel tank (debug chat 2026-10-25). With a checked photo of it as a reference
latent the same seed drew a true 2107 (bench/object_ref_probe.py). Searches the outbound
guard refuses (weapons, among others) get no photo and draw from words as before.
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import List, Tuple

from seams import dep, text_model, vision_model

logger = logging.getLogger("assistant.object_refs")

MAX_THINGS = 2
_CANDIDATES = 8
_MIN_FACE_PX = 160

_FIND_SYS = (
    "You read a request for a picture and list the SPECIFIC things in it whose exact look an "
    "image model is likely to get wrong, each with its kind:\n"
    "- \"object\": a named model or make of a car, motorcycle, truck, tractor, train, aircraft, "
    "ship, gadget, instrument or appliance (\"ВАЗ 2107\", \"Урал с коляской\", \"Ту-154\"), or a "
    "lesser-known named building or monument;\n"
    "- \"person\": a real, well-known person who must be recognisable (\"Ленин\", \"Гагарин\");\n"
    "- \"character\": a fictional character, or the costume, armour or outfit of one that someone "
    "else wears (\"Ленин в стиле Думгая\" -> the Doom Slayer's armour; \"кот в костюме Бэтмена\" "
    "-> Batman's suit).\n"
    # The long «Lenin reimagined as a Doomguy-style warrior» came back as the armour alone and
    # Lenin was drawn as a stranger (bench 2026-10-26).
    "A real person drawn in a character's costume or style is TWO entries: the person and the costume.\n"
    "Not: a generic thing (\"a car\", \"a knight\", \"a demon\"), a world-famous landmark every "
    "image model knows (the Eiffel Tower), an animal, a food, a style of art (anime, watercolour). "
    "Reply with JSON only: {\"things\": [{\"kind\": \"object|person|character\", \"thing\": "
    "\"<short English name, e.g. a VAZ-2107 Lada sedan, Vladimir Lenin, the Doom Slayer's armour>\", "
    "\"query\": \"<image-search query for a picture of it, in the language it is best known in>\"}]} "
    "-- at most " + str(MAX_THINGS) + "; [] when none.")

_KINDS = ("object", "person", "character")


def named_things(ctx, description: str) -> list:
    """[{"kind", "thing", "query"}] the request names, read by the model; [] on any failure."""
    from utils import safe_json_from_llm
    try:
        got = safe_json_from_llm(text_model(ctx)(ctx, _FIND_SYS, description, temperature=0.0,
                                                 max_tokens=300)) or {}
    except Exception:
        logger.warning("object refs: the read failed", exc_info=True)
        return []
    things = [dict(t, kind=t.get("kind") if t.get("kind") in _KINDS else "object")
              for t in (got.get("things") or []) if isinstance(t, dict)
              and str(t.get("thing") or "").strip() and str(t.get("query") or "").strip()]
    return things[:MAX_THINGS]


_CHECK = {
    "object": "Is this a real photograph showing {thing} whole and clearly as the main subject, with no "
              "text or watermark over it? Reply JSON only: {{\"shows\": true/false, \"box\": [x0, y0, "
              "x1, y1]}} -- the box around the whole thing on a 0-1000 grid.",
    # The head only: the drawing dresses and poses the person itself.
    # Lenin in his cap and Gagarin in his helmet passed and drew strangers (bench 2026-10-26).
    "person": "Is this a sharp photograph of {thing}'s face, seen from the front or three-quarters, "
              "the face large, uncovered and clearly visible, the head bare -- no hat, cap or helmet? Reply JSON only: {{\"shows\": true/false, "
              "\"box\": [x0, y0, x1, y1]}} -- the box around the head -- hair, face and the whole beard down to its tip -- on a 0-1000 grid.",
    "character": "Is this a picture showing {thing} whole and clearly as the main subject, with no text "
                 "or watermark over it? Reply JSON only: {{\"shows\": true/false, \"box\": [x0, y0, x1, "
                 "y1]}} -- the box around the whole of it on a 0-1000 grid.",
}


def _box(ctx, path: str, thing: str, kind: str = "object"):
    """(x0, y0, x1, y1) of `thing` in the photo on a 0-1000 grid, or None when the photo is not a
    clear one of it."""
    from utils import safe_json_from_llm
    ans = vision_model(ctx)(ctx, image_paths=[("Photo:", path)], temperature=0.0, max_tokens=60,
                            user_text=_CHECK[kind].format(thing=thing))
    got = safe_json_from_llm(ans or "") or {}
    box = got.get("box")
    if got.get("shows") is not True or not (isinstance(box, list) and len(box) == 4):
        return None
    try:
        x0, y0, x1, y1 = (max(0, min(1000, int(v))) for v in box)
    except (TypeError, ValueError):
        return None
    return (x0, y0, x1, y1) if x1 - x0 > 50 and y1 - y0 > 50 else None


def _crop(path: str, box, margin: float = 0.06, below: float = None) -> str:
    """The photo cut to the thing (with a margin): the street, the people and the stray stick
    around the «Урал» were copied into the drawing along with it. `below`: the margin under it."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    x0, y0, x1, y1 = box
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    my_below = (y1 - y0) * (margin if below is None else below)
    cut = im.crop((int(max(0, x0 - mx) * w / 1000), int(max(0, y0 - my) * h / 1000),
                   int(min(1000, x1 + mx) * w / 1000), int(min(1000, y1 + my_below) * h / 1000)))
    out = path.rsplit(".", 1)[0] + "_thing.jpg"
    cut.save(out, quality=92)
    return out


def _head(path: str) -> str:
    """The head cut around the detector's face box, or "" when no face is found. The vision
    model's own head box ended at Lenin's moustache (goatee lost) on one portrait and took in
    his collar and tie on another, which came through under the Doom armour (bench 2026-10-26)."""
    from PIL import Image
    try:
        from identity_metrics import face_box
        fb = face_box(path, pad=0.0)
    except Exception:
        logger.warning("object refs: no face detector", exc_info=True)
        return ""
    if not fb:
        return ""
    x0, y0, x1, y1 = fb
    fw, fh = x1 - x0, y1 - y0
    if fw < _MIN_FACE_PX:      # a blurred thumbnail of a painting drew a mushy Lenin
        return ""
    from PIL import ImageDraw
    im = Image.open(path).convert("RGB")
    # hair or a bald crown above the brow, ears at the sides, a beard under the chin
    ox, oy = int(max(0, x0 - 0.35 * fw)), int(max(0, y0 - 0.6 * fh))
    cut = im.crop((ox, oy, int(min(im.width, x1 + 0.35 * fw)), int(min(im.height, y1 + 0.25 * fh))))
    # ...and only the head: an oval around it, the rest the photo's own background. Cut square,
    # Lenin's collar and tie sat beside his goatee and came through under the Doom armour, whatever
    # the prompt said (debug chat 2026-10-26).
    w, h = cut.size
    bg = tuple(sum(v) // 2 for v in zip(cut.getpixel((2, 2)), cut.getpixel((w - 3, 2))))
    cx = (x0 + x1) / 2 - ox
    oval = Image.new("L", cut.size, 0)
    ImageDraw.Draw(oval).ellipse([cx - 0.62 * fw, y0 - oy - 0.6 * fh, cx + 0.62 * fw, y1 - oy + 0.25 * fh], fill=255)
    cut = Image.composite(cut, Image.new("RGB", cut.size, bg), oval)
    out = path.rsplit(".", 1)[0] + "_head.jpg"
    cut.save(out, quality=92)
    return out


def photo_of(ctx, thing: str, query: str, kind: str = "object") -> str:
    """A checked, cropped web photo of `thing`, or "" (refused search, nothing found)."""
    import search
    urls = dep(ctx, "image_search_urls", search.image_search_urls)(
        ctx, query, prefer="portrait" if kind == "person" else "landscape")
    work = tempfile.mkdtemp(prefix="objref_")
    for n, url in enumerate(urls[:_CANDIDATES]):
        dest = os.path.join(work, f"c{n}.jpg")
        if not dep(ctx, "download_image", search.download_image)(ctx, url, dest):
            continue
        try:
            box = _box(ctx, dest, thing, kind)
        except Exception:
            logger.warning("object refs: the photo check failed", exc_info=True)
            return ""
        if not box:
            continue
        # no face the detector can find (Gagarin side-on in his helmet): the next photo
        got = _crop(dest, box) if kind != "person" else _head(dest)
        if got:
            logger.info("object refs: %s (%s) <- %s", thing, kind, url[:120])
            return got
    logger.info("object refs: no photo of %s (%d candidates)", thing, len(urls))
    return ""


def find(ctx, description: str) -> List[Tuple[str, str, str]]:
    """[(thing, photo path, kind)] for the specific things the request names."""
    if not getattr(ctx, "web_search_enabled", True):
        return []
    if os.getenv("F5_TEST_RUN") and not dep(ctx, "live", False):     # suites draw offline
        return []
    out = []
    things = named_things(ctx, description)
    if things and getattr(ctx, "set_stage", None):
        ctx.set_stage("Finding a photo online")
    for t in things:
        path = photo_of(ctx, t["thing"], t["query"], t["kind"])
        if path:
            out.append((t["thing"], path, t["kind"]))
    # The face last: given first, Lenin's face drew the Doom armour grey instead of green.
    return sorted(out, key=lambda t: t[2] == "person")


_FACE_WORDS = ("This is a photo of {who}. Describe the FACE and HEAD for an artist who must draw {who} "
               "recognisably from any angle, in any light, with any expression: head shape, hair or baldness "
               "and hairline, facial hair (its exact shape, length and colour), brow, eyes, nose, cheekbones, "
               "mouth, ears, apparent age. 50-80 words, visual facts only, no clothes, no pose, no background.")


def face_words(ctx, who: str, path: str) -> str:
    """The face on the found photo in words, for the scene render: the words the web gave for Lenin
    said «often wears glasses»; read off his photo they are his own (bench 2026-10-26)."""
    try:
        words = " ".join((vision_model(ctx)(ctx, image_paths=[("Photo:", path)], temperature=0.1, max_tokens=220,
                                            user_text=_FACE_WORDS.format(who=who)) or "").split())
    except Exception:
        logger.warning("object refs: no words for %s", who, exc_info=True)
        return ""
    return f"{who} must be recognisable as the real {who}: {words}" if len(words.split()) >= 8 else ""


# A face given to the scene render as a reference latent set the frame: Lenin stared into the
# camera in a centred portrait and the fist fight was gone; the owner: «башка должна смотреть куда
# надо, цветокор по сюжету, фото брать для вдохновения» (2026-10-26). The scene is drawn free and
# only then the face is edited in. The details the edit loses (a star on the brow, glowing eyes)
# come back through image_generate._missing_asks.
_SWAP = ("The face and head of {who} in this picture become the face of {who} from image 2 -- his "
         "features, head shape, hairline and facial hair -- while his head angle, gaze, expression, "
         "everything on his face this picture adds (glow, marks, paint) and this scene's light and colours "
         "stay as they are.")
# Below this the face is a few dozen pixels: no edit makes it more anyone (a rider on a galloping horse).
MIN_SWAP_FACE_PX = 90


def swap_faces(ctx, path: str, faces: list) -> str:
    """`path` with each (who, photo) face edited in from its photo; `path` itself when the face in
    the picture is too small or the edit fails."""
    from identity_metrics import face_boxes
    import image as image_mod
    out = path
    for who, photo in faces:
        try:
            boxes = face_boxes(out, pad=0.0)
        except Exception:
            boxes = []
        if not boxes or max(b[2] - b[0] for b in boxes) < MIN_SWAP_FACE_PX:
            logger.info("object refs: %s too small to edit the face in", who)
            continue
        got = dep(ctx, "edit_image", image_mod._edit_image_with_firered_once)(
            ctx, out, _SWAP.format(who=who), reference_paths=[photo])
        if got:
            logger.info("object refs: %s's face edited in", who)
            out = got
    return out


def prompt_note(things: list) -> str:
    """The sentences that tell Klein what each reference image (a thing or a costume, never a
    face: see swap_faces) is for -- and what it is not."""
    out = []
    for i, (thing, _, kind) in enumerate(things, 1):
        # «LADA 2107» on the reference's plate came back as «VAD 2107» on the drawing's (bench 2026-10-25).
        # A character's look includes its colours (the Doom Slayer's green armour); a car's paint
        # is the picture's own («красная 2107» came back red over a white reference).
        own = "angle, pose, place and surroundings" if kind == "character" else "angle, colour, place and surroundings"
        out.append(f"{thing[0].upper() + thing[1:]} looks exactly like the one in image {i}: its shape, "
                   f"proportions and every part. Only that is taken from the reference; its {own} are "
                   "this picture's own, and no lettering, number plate or logo text is copied from it.")
    return "".join(" " + x for x in out)

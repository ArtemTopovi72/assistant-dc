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

_FIND_SYS = (
    "You read a request for a picture and list the SPECIFIC real-world things in it whose exact "
    "look an image model is likely to get wrong: a named model or make of a car, motorcycle, truck, "
    "tractor, train, aircraft, ship, gadget, instrument or appliance (\"ВАЗ 2107\", \"Урал с "
    "коляской\", \"Gibson Les Paul\", \"Ту-154\"), or a lesser-known named building or monument. "
    "Not: a generic thing (\"a car\", \"a motorcycle\", \"a guitar\"), a world-famous landmark "
    "every image model knows (the Eiffel Tower, the Kremlin), a person or a character, an animal, "
    "a food, clothing, a style. Reply with JSON only: {\"things\": [{\"thing\": \"<short English "
    "name, e.g. a VAZ-2107 Lada sedan>\", \"query\": \"<image-search query for a photo of it, in "
    "the language it is best known in>\"}]} -- at most " + str(MAX_THINGS) + "; [] when none.")


def named_things(ctx, description: str) -> list:
    """[{"thing", "query"}] the request names, read by the model; [] on any failure."""
    from utils import safe_json_from_llm
    try:
        got = safe_json_from_llm(text_model(ctx)(ctx, _FIND_SYS, description, temperature=0.0,
                                                 max_tokens=200)) or {}
    except Exception:
        logger.warning("object refs: the read failed", exc_info=True)
        return []
    things = [t for t in (got.get("things") or []) if isinstance(t, dict)
              and str(t.get("thing") or "").strip() and str(t.get("query") or "").strip()]
    return things[:MAX_THINGS]


def _box(ctx, path: str, thing: str):
    """(x0, y0, x1, y1) of `thing` in the photo on a 0-1000 grid, or None when it is not a
    clear, whole photo of it."""
    from utils import safe_json_from_llm
    ans = vision_model(ctx)(
        ctx, image_paths=[("Photo:", path)], temperature=0.0, max_tokens=60,
        user_text=f"Is this a real photograph showing {thing} whole and clearly as the main subject, "
                  "with no text or watermark over it? Reply JSON only: {\"shows\": true/false, "
                  "\"box\": [x0, y0, x1, y1]} -- the box around the whole thing on a 0-1000 grid.")
    got = safe_json_from_llm(ans or "") or {}
    box = got.get("box")
    if got.get("shows") is not True or not (isinstance(box, list) and len(box) == 4):
        return None
    try:
        x0, y0, x1, y1 = (max(0, min(1000, int(v))) for v in box)
    except (TypeError, ValueError):
        return None
    return (x0, y0, x1, y1) if x1 - x0 > 50 and y1 - y0 > 50 else None


def _crop(path: str, box) -> str:
    """The photo cut to the thing (with a margin): the street, the people and the stray stick
    around the «Урал» were copied into the drawing along with it."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    x0, y0, x1, y1 = box
    mx, my = (x1 - x0) * 0.06, (y1 - y0) * 0.06
    cut = im.crop((int(max(0, x0 - mx) * w / 1000), int(max(0, y0 - my) * h / 1000),
                   int(min(1000, x1 + mx) * w / 1000), int(min(1000, y1 + my) * h / 1000)))
    out = path.rsplit(".", 1)[0] + "_thing.jpg"
    cut.save(out, quality=92)
    return out


def photo_of(ctx, thing: str, query: str) -> str:
    """A checked, cropped web photo of `thing`, or "" (refused search, nothing found)."""
    import search
    urls = dep(ctx, "image_search_urls", search.image_search_urls)(ctx, query, prefer="landscape")
    work = tempfile.mkdtemp(prefix="objref_")
    for n, url in enumerate(urls[:_CANDIDATES]):
        dest = os.path.join(work, f"c{n}.jpg")
        if not dep(ctx, "download_image", search.download_image)(ctx, url, dest):
            continue
        try:
            box = _box(ctx, dest, thing)
        except Exception:
            logger.warning("object refs: the photo check failed", exc_info=True)
            return ""
        if box:
            logger.info("object refs: %s <- %s", thing, url[:120])
            return _crop(dest, box)
    logger.info("object refs: no photo of %s (%d candidates)", thing, len(urls))
    return ""


def find(ctx, description: str) -> List[Tuple[str, str]]:
    """[(thing, photo path)] for the specific real things the request names."""
    if not getattr(ctx, "web_search_enabled", True):
        return []
    if os.getenv("F5_TEST_RUN") and not dep(ctx, "live", False):     # suites draw offline
        return []
    out = []
    things = named_things(ctx, description)
    if things and getattr(ctx, "set_stage", None):
        ctx.set_stage("Finding a photo online")
    for t in things:
        path = photo_of(ctx, t["thing"], t["query"])
        if path:
            out.append((t["thing"], path))
    return out


def prompt_note(things: List[Tuple[str, str]]) -> str:
    """The sentence that tells Klein what each reference image is for -- and what it is not."""
    if not things:
        return ""
    each = "; ".join(f"{thing} looks exactly like the one in image {i}" for i, (thing, _) in enumerate(things, 1))
    # «LADA 2107» on the reference's plate came back as «VAD 2107» on the drawing's (bench 2026-10-25).
    return (f" {each[0].upper() + each[1:]}: its shape, proportions and every part. Only that object "
            "is taken from the reference; its angle, colour, place and surroundings are this picture's own, "
            "and no lettering, number plate or logo text is copied from it.")

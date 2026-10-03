"""A close look: the picture in zoomed parts, so small details are seen.

One vision call on a 1024-px proxy answers with the biggest things in the
frame and misses the small ones -- a label, a number on a sign, a scar, the
thing in someone's hand -- which are often what the question is about (live
2026-09-18: «на картинках видит только самые здоровые детали, на мелкие
вообще забивает, а часто они самые важные»).

For a QUESTION about a picture (not an edit) the picture is also cut into
zoomed parts, each read on its own for small details, and the answer is
written from the overview plus the parts. Costs a few vision calls; only
runs when the source has enough pixels to zoom into.
"""
from __future__ import annotations
import io
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger("assistant.vision")

# Below this long side there is nothing to zoom into.
MIN_SIDE_FOR_TILES = 900
# 2x2 up to this long side, 3x3 above it (a 4000-px phone photo).
GRID3_FROM_SIDE = 2400
MAX_TILES = 9

TILE_PROMPT = (
    "You see ONE ZOOMED PART of a larger picture (its position is given). List "
    "the SMALL details visible in this part that a glance at the whole picture "
    "would miss: any text, numbers, labels, logos, signs, small objects, marks, "
    "patterns, what people hold or wear, background items, defects. Be concrete "
    "and literal, only what is actually visible; say nothing about what is not "
    "here. 3-6 short lines, plain text.")

MERGE_PROMPT = (
    "You answer a question about a picture. You get an OVERVIEW of the whole "
    "picture and NOTES from zoomed-in parts of it. Write the answer to the "
    "question from both, in the language of the question: the direct answer "
    "first, then the details that matter, INCLUDING the small ones from the "
    "notes (text, numbers, small objects) when they are relevant. Do not list "
    "the parts by position; merge them into one description. Up to eight "
    "sentences, plain text, no markdown. Never invent what neither source "
    "mentions.")


def tile_boxes(w: int, h: int) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """(label, box) for the zoomed parts, with a 10% overlap between them."""
    n = 3 if max(w, h) >= GRID3_FROM_SIDE else 2
    names = {2: ["top-left", "top-right", "bottom-left", "bottom-right"],
             3: ["top-left", "top-centre", "top-right", "middle-left", "centre",
                 "middle-right", "bottom-left", "bottom-centre", "bottom-right"]}[n]
    tw, th = w / n, h / n
    ox, oy = int(tw * 0.1), int(th * 0.1)
    out = []
    for r in range(n):
        for c in range(n):
            x0, y0 = int(c * tw) - ox, int(r * th) - oy
            x1, y1 = int((c + 1) * tw) + ox, int((r + 1) * th) + oy
            out.append((names[r * n + c], (max(0, x0), max(0, y0), min(w, x1), min(h, y1))))
    return out[:MAX_TILES]


def tiles_of(image_bytes: bytes) -> List[Tuple[str, bytes]]:
    """The zoomed parts as JPEG bytes, or [] when the picture is too small."""
    from PIL import Image
    with Image.open(io.BytesIO(image_bytes)) as im:
        w, h = im.size
        if max(w, h) < MIN_SIDE_FOR_TILES:
            return []
        im = im.convert("RGB")
        out = []
        for name, box in tile_boxes(w, h):
            buf = io.BytesIO()
            im.crop(box).save(buf, "JPEG", quality=92)
            out.append((name, buf.getvalue()))
        return out


def tile_notes(ctx, image_bytes: bytes, *, label: str = "the picture") -> str:
    """The small-detail notes from each zoomed part, joined as bullet lines,
    or "" when there was nothing to zoom into or every part failed.

    This is the reusable primitive behind look_closely(): any node that looks
    at a picture and could miss small details -- a final-render judge, a
    layout sketch check, a video contact sheet -- can fold these notes into
    its OWN prompt, instead of only the Q&A-merge shape look_closely builds.
    Same tile budget and stage-reporting either way, so this is not a second,
    cheaper close look; it is the same one, addressed to a different caller.
    """
    import llm as _llm
    try:
        parts = tiles_of(image_bytes)
    except Exception:
        logger.warning("close look: could not cut %s", label, exc_info=True)
        return ""
    if not parts:
        return ""
    notes = []
    for i, (name, data) in enumerate(parts, 1):
        if ctx is not None:
            if ctx.is_cancelled():
                return ""
            ctx.set_stage(f"Looking closer ({i}/{len(parts)})")
        try:
            got = _llm.analyze_image_with_llm(
                ctx, image_bytes=data,
                user_text=f"Part {i} of {len(parts)}: the {name} of {label}, zoomed in.",
                system_prompt=TILE_PROMPT, temperature=0.1, max_tokens=350) or ""
        except Exception:
            logger.warning("close look: part %d of %s failed", i, label, exc_info=True)
            got = ""
        got = " ".join(got.split())
        if got:
            notes.append(f"- {name}: {got}")
    if not notes:
        return ""
    logger.info("close look: %d part(s) read for %s", len(notes), label)
    return "\n".join(notes)


def look_closely(ctx, image_bytes: bytes, question: str, overview: str) -> Optional[str]:
    """The answer written from the overview plus the zoomed parts, or None
    when there was nothing to zoom into (the overview stands)."""
    import llm as _llm
    notes_text = tile_notes(ctx, image_bytes, label="the picture")
    if not notes_text:
        return None
    user = (f"Question: {question.strip() or 'Describe the picture.'}\n\n"
            f"Overview of the whole picture:\n{overview.strip()}\n\n"
            f"Notes from the zoomed parts:\n" + notes_text)
    try:
        if ctx is not None:
            ctx.set_stage("Looking at the image")
        merged = _llm.call_llm_simple(ctx, MERGE_PROMPT, user, temperature=0.2,
                                      max_tokens=700, prefill="<think></think>") or ""
    except Exception:
        logger.warning("close look: merge failed", exc_info=True)
        return None
    merged = merged.strip()
    return merged or None

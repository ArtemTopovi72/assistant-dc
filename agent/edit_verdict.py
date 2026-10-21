"""One judge for a picture edit: BEFORE, AFTER and the user's own words in one model read,
one verdict -- done / partial / not, and why.

It replaces a stack that disagreed with itself: the inspector's PRESENT/MISSING labels, a
regex reader of its description and a second model read of that description. On the
labelled set (bench/edit_judge_eval.py) that stack called clean edits failed -- a red tie,
a bare wrist, a shaved face -- which is the «он постоянно ноет что не получилось» the owner
reported (2026-10-21).
"""
from __future__ import annotations

import logging

from seams import vision_model

logger = logging.getLogger("assistant." + __name__)

SYSTEM = """You check a picture edit. You get two pictures, BEFORE and AFTER, and the user's request. Look at both and answer plain facts; do not grade.

- "after_shows": look ONLY at AFTER and say in one sentence what is there now at the thing or place the request is about (which person wears what, whether the object is there, what the background shows). Do not let the request tell you what you see.
- "changed": one sentence on what differs between BEFORE and AFTER, or "nothing".
- "asked_change_visible": true when AFTER shows what the user asked for, also when BEFORE already did. For a removal the thing must be fully gone: a faint, blurred or ghostly version of it still there is not gone.
- "something_else_instead": true when something other than the request was done: a new person or object drawn instead of changing the one there, a removed thing replaced by another of its kind (a new watch, a new strap), or a change nobody asked for to something the request did not name (another colour, material or person). When the request asks for a new scene or a whole new picture, drawing it is the change, not something else.
- "defect": a visible flaw that came with the edit -- a seam, a patch of different colour or texture where the edit was, part of the thing left behind, a duplicate, an extra object or person nobody asked for -- in a few words; "" when there is none a viewer would notice.

Answer with ONE JSON object and nothing else: {"after_shows": "...", "changed": "...", "asked_change_visible": true, "something_else_instead": false, "defect": ""}"""

def _verdict(facts: dict) -> str:
    """The model states facts; the grading is fixed here (it graded a new strap «partial» and a
    skin patch it had just described «done», 2026-10-21)."""
    if facts.get("asked_change_visible") is not True or facts.get("something_else_instead") is True:
        return "not"
    return "partial" if str(facts.get("defect") or "").strip() else "done"


def _unchanged(before: str, after: str) -> bool:
    """The two pictures are the same pixels (a no-op edit). Measured, not asked: the model wrote
    «nothing» changed and still answered the asked change visible (2026-10-21)."""
    from PIL import Image, ImageChops, ImageStat
    try:
        a, b = (Image.open(p).convert("RGB").resize((256, 256)) for p in (before, after))
        return max(ImageStat.Stat(ImageChops.difference(a, b)).mean) < 1.0
    except Exception:
        return False


def judge(ctx, before: str, after: str, request: str) -> dict:
    """{"verdict": "done"|"partial"|"not"|"", "why": str}; "" when the model gave no usable answer."""
    from utils import safe_json_from_llm
    if before and after and _unchanged(before, after):
        return {"verdict": "not", "why": "the picture did not change", "changed": "nothing"}
    text = f"The user's request: {(request or '').strip()[:600]}"
    for _ in range(2):
        resp = vision_model(ctx)(ctx=ctx, image_paths=[("BEFORE:", before), ("AFTER:", after)],
                                 user_text=text, system_prompt=SYSTEM, temperature=0.0, max_tokens=400)
        got = safe_json_from_llm(resp or "", ["asked_change_visible"]) or {}
        if isinstance(got.get("asked_change_visible"), bool):
            verdict = _verdict(got)
            why = str(got.get("defect") or got.get("changed") or "").strip()
            logger.info("edit verdict: %s -- %s", verdict, why[:200])
            return {"verdict": verdict, "why": why, "changed": str(got.get("changed") or "")}
    logger.warning("edit verdict: no usable answer (%r)", (resp or "")[:200])
    return {"verdict": "", "why": ""}

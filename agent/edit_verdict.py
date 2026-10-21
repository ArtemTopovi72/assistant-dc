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
import os

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
    """No area of the picture changed (a no-op edit), measured, not asked: the model wrote
    «nothing» changed and still answered the asked change visible (2026-10-21). A local
    change is looked for, not a mean: a nose edit moved the whole-frame mean by 0.16."""
    import identity_metrics
    from PIL import Image
    try:
        for p in (before, after):
            Image.open(p).verify()
    except Exception:
        return False
    return identity_metrics.changed_box(before, after) is None


def _close_ups(before: str, after: str) -> list:
    """BEFORE/AFTER crops of the changed area, padded to its own size. A bigger nose on a
    whole 600x750 portrait read as «nothing» changed (2026-10-22); the crop shows it."""
    import tempfile
    import identity_metrics
    from PIL import Image
    box = identity_metrics.changed_box(before, after)
    if not box or box[0] > 0.25:
        return []
    x0, y0, x1, y1 = box[1]
    side = max(x1 - x0, y1 - y0, 0.12)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    out = []
    for label, p in (("BEFORE, close-up of the edited area:", before), ("AFTER, close-up of the same area:", after)):
        im = Image.open(p).convert("RGB")
        w, h = im.size
        crop = im.crop((int(max(0, cx - side) * w), int(max(0, cy - side) * h),
                        int(min(1, cx + side) * w), int(min(1, cy + side) * h)))
        fd, path = tempfile.mkstemp(suffix=".png", prefix="verdict_crop_")
        os.close(fd)
        crop.save(path)
        out.append((label, path))
    return out


def judge(ctx, before: str, after: str, request: str) -> dict:
    """{"verdict": "done"|"partial"|"not"|"", "why": str}; "" when the model gave no usable answer."""
    from utils import safe_json_from_llm
    if before and after and _unchanged(before, after):
        return {"verdict": "not", "why": "the picture did not change", "changed": "nothing"}
    text = f"The user's request: {(request or '').strip()[:600]}"
    try:
        crops = _close_ups(before, after)
    except Exception as exc:
        logger.warning("edit verdict: no close-ups: %s", exc)
        crops = []
    try:
        for _ in range(2):
            resp = vision_model(ctx)(ctx=ctx, image_paths=[("BEFORE:", before), ("AFTER:", after)] + crops,
                                     user_text=text, system_prompt=SYSTEM, temperature=0.0, max_tokens=400)
            got = safe_json_from_llm(resp or "", ["asked_change_visible"]) or {}
            if isinstance(got.get("asked_change_visible"), bool):
                verdict = _verdict(got)
                why = str(got.get("defect") or got.get("changed") or "").strip()
                logger.info("edit verdict: %s (%d close-ups) -- %s", verdict, len(crops), why[:200])
                return {"verdict": verdict, "why": why, "changed": str(got.get("changed") or "")}
        logger.warning("edit verdict: no usable answer (%r)", (resp or "")[:200])
        return {"verdict": "", "why": ""}
    finally:
        for _, p in crops:
            try:
                os.remove(p)
            except OSError:
                pass

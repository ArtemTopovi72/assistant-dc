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
import re
from difflib import SequenceMatcher

from seams import dep, vision_model

logger = logging.getLogger("assistant." + __name__)

SYSTEM = """You check a picture edit. You get two pictures, BEFORE and AFTER, and the user's request. Look at both and answer plain facts; do not grade.

- "after_shows": look ONLY at AFTER and say in one sentence what is there now at the thing or place the request is about (which person wears what, whether the object is there, what the background shows). Do not let the request tell you what you see.
- "changed": one sentence on what differs between BEFORE and AFTER, or "nothing".
- "asked_change_visible": true when AFTER shows what the user asked for, also when BEFORE already did. For a removal the thing must be fully gone: a faint, blurred or ghostly version of it still there is not gone.
- Only when the user asks to replace text, check whether the old text remains mixed with the replacement. When text must be preserved, a single inscription in a different colour is not duplicated or old text to remove.
- "some_requested_changes_visible": true only when the request has multiple distinct changes and AFTER visibly satisfies at least one of them, even though another is still missing. Do not infer success from the request itself.
- "something_else_instead": true when something other than the request was done: a new person or object drawn instead of changing the one there, a removed thing replaced by another of its kind (a new watch, a new strap), or a change nobody asked for to something the request did not name (another colour, material or person). When the request asks for a new scene or a whole new picture, drawing it is the change, not something else.
- "defect": a visible flaw that came with the edit -- a seam, a patch of different colour or texture where the edit was, part of the thing left behind, a duplicate, an extra object or person nobody asked for -- in a few words; "" when there is none a viewer would notice.

Answer with ONE JSON object and nothing else: {"after_shows": "...", "changed": "...", "asked_change_visible": true, "some_requested_changes_visible": false, "something_else_instead": false, "defect": ""}"""


def _preserved_text_unconfirmed(ctx, before: str, after: str, request: str) -> bool:
    """An OCR miss is uncertainty, not proof that preserved lettering was removed."""
    text = (request or "").lower()
    preserve = r"(?:keep|preserve|retain|сохрани\w*|остав\w*)[^.!?\n]{0,100}(?:text|lettering|inscription|надпис\w*|текст\w*|букв\w*)"
    if not re.search(preserve, text) or re.search(r"\b(?:remove|erase|replace|delete|убер\w*|удал\w*|замен\w*|стери\w*)\b", text):
        return False
    try:
        import ocr_reader
        read = dep(ctx, "ocr_boxes", ocr_reader.read_boxes)
        old, new = read(before), read(after)
        if old is None or new is None:
            return False
        def words(rows, min_conf):
            return [re.sub(r"\W+", "", str(label)).casefold() for _, label, conf in rows
                    if conf >= min_conf and len(re.sub(r"\W+", "", str(label))) >= 4]
        previous, current = words(old, .9), words(new, .5)
        return any(not any(SequenceMatcher(None, x, y).ratio() >= .8 for y in current)
                   for x in previous)
    except Exception:
        logger.debug("preserved text OCR check unavailable", exc_info=True)
        return False

def _verdict(facts: dict) -> str:
    """The model states facts; the grading is fixed here (it graded a new strap «partial» and a
    skin patch it had just described «done», 2026-10-21)."""
    if facts.get("something_else_instead") is not False:
        return "not"
    if facts.get("asked_change_visible") is not True:
        if (facts.get("some_requested_changes_visible") is True
                and facts.get("something_else_instead") is False
                and not str(facts.get("defect") or "").strip()):
            return "partial"
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
        first_incomplete = None
        for attempt in range(2):
            resp = vision_model(ctx)(ctx=ctx, image_paths=[("BEFORE:", before), ("AFTER:", after)] + crops,
                                     user_text=text, system_prompt=SYSTEM, temperature=0.0, max_tokens=400)
            got = safe_json_from_llm(resp or "", ["asked_change_visible"]) or {}
            if isinstance(got.get("asked_change_visible"), bool):
                verdict = _verdict(got)
                why = str(got.get("defect") or got.get("changed") or "").strip()
                logger.info("edit verdict: %s (%d close-ups, requested_part=%r, other=%r, defect=%r) -- %s",
                            verdict, len(crops), got.get("some_requested_changes_visible"),
                            got.get("something_else_instead"), got.get("defect"), why[:200])
                result = {"verdict": verdict, "why": why, "changed": str(got.get("changed") or "")}
                if verdict == "done" and _preserved_text_unconfirmed(ctx, before, after, request):
                    result.update(verdict="partial", why="OCR could not confirm all requested preserved text; check it before claiming it is intact.")
                    verdict = "partial"
                if first_incomplete is not None:
                    return result if verdict == "partial" else first_incomplete
                if (attempt == 0 and verdict == "not" and got.get("some_requested_changes_visible") is False
                        and got.get("something_else_instead") is False and not got.get("defect")
                        and result["changed"].strip().lower() not in ("", "nothing", "none")):
                    first_incomplete = result
                    continue
                return result
        logger.warning("edit verdict: no usable answer (%r)", (resp or "")[:200])
        return first_incomplete or {"verdict": "", "why": ""}
    finally:
        for _, p in crops:
            try:
                os.remove(p)
            except OSError:
                pass

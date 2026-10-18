"""Edit instruction -> BFL-format record (paragraph + element table with src/tgt boxes).

Klein's editor was trained on records written by an annotator that saw the source image and
the instruction (docs.bfl.ai prompt upsampler, system prompt in prompts/klein_edit_annotator.txt).
We run the same job on our vision LLM. The table also gives the boxes the edit may touch, which
`compositing.lock_outside_boxes` uses to keep the rest of the frame pixel-identical.
"""
import json
import logging
import os
import re
from typing import Optional

from seams import vision_model

logger = logging.getLogger("assistant.image")

_PROMPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "klein_edit_annotator.txt")


def parse_record(text: str):
    """'<paragraph> [ {row}, ... ]' -> (paragraph, rows) or None when the table is missing/broken."""
    t = (text or "").strip().replace("\n", " ")
    t = re.sub(r"^```\w*|```$", "", t).strip()
    i = t.find("[{")
    if i < 1:
        return None
    try:
        rows, _ = json.JSONDecoder().raw_decode(t[i:])
    except ValueError:
        return None
    para = t[:i].strip()
    if not para or not isinstance(rows, list) or not all(isinstance(r, dict) and r.get("id") for r in rows):
        return None
    return para, rows


def edit_boxes(rows) -> list:
    """Boxes the edit actually touches: every src/tgt box of a row that moves, appears or goes away."""
    out = []
    for r in rows:
        s, t = r.get("src_bbox"), r.get("tgt_bbox")
        if s is not None and s == t:
            continue
        for b in (s, t):
            if isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
                out.append([int(v) for v in b])
    return out


def annotate_edit(ctx, image_path: str, instruction: str) -> Optional[dict]:
    """{'prompt': str, 'boxes': [...]} or None (no vision model / unparsable) -> caller edits as before."""
    try:
        with open(_PROMPT, encoding="utf-8") as f:
            sys_prompt = f.read()
        resp = vision_model(ctx)(ctx=ctx, image_path=image_path, user_text=instruction,
                                 system_prompt=sys_prompt, max_tokens=2500)
    except Exception as exc:
        logger.warning("klein annotate failed (%s) — plain instruction", exc)
        return None
    rec = parse_record(resp or "")
    if not rec:
        logger.info("klein annotate: unparsable reply %r", (resp or "")[:80])
        return None
    para, rows = rec
    return {"prompt": para + " " + json.dumps(rows, ensure_ascii=False, separators=(",", ":")),
            "boxes": edit_boxes(rows)}

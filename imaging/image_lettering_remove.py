"""«Убери все надписи с фона»: the mask comes from OCR, not from a phrase.

Live 2026-09-18 (journey 3): asked to remove ALL the lettering behind a
girl, the contained edit asked SAM3 for 'wall inscriptions', got one strip
166 px tall, cleaned it, and shipped a picture with "INSGATE" still on the
left. A segmenter given a phrase finds ONE thing; "all the text" is every
piece the OCR can read. So for a removal aimed at lettering the mask is the
union of the OCR boxes (both scales), grown a little, faces excluded, and
the hole is repainted by FireRed on the crop; afterwards the OCR reads the result again and a
second pass with a wider mask cleans what survived.
"""
from __future__ import annotations
from config import scratch_path, venv_python
import logging
import os
import re
import time
from typing import List, Optional, Tuple

logger = logging.getLogger("assistant.image")

Box = Tuple[int, int, int, int]

def is_lettering_removal(region: str, instructions: str = "") -> bool:
    """A removal whose target is the lettering (all of it, or unqualified)."""
    import intent
    return intent.ask_yes("The user wants this removed from a picture: {text}\n\nIs it "
                          "WRITING (lettering, a caption, a logo, a watermark, a sign's words) "
                          "rather than an object, a person or a stain?",
                          f"{region}. {instructions}".strip(" ."))


def ocr_boxes(image_path: str) -> Optional[List[Box]]:
    """Pixel boxes of every piece EasyOCR reads, at two scales. None = no OCR."""
    if os.getenv("F5_TEST_RUN") or not image_path or not os.path.exists(image_path):
        return None
    try:
        import ocr_reader
        r = ocr_reader._get()
    except Exception:
        return None
    if r is None:
        return None
    from PIL import Image
    import numpy as np
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        variants = [(np.asarray(im), 1.0)]
        if max(w, h) < 1600:
            variants.append((np.asarray(im.resize((w * 2, h * 2), Image.LANCZOS)), 2.0))
    boxes: List[Box] = []
    for arr, scale in variants:
        try:
            found = r.readtext(arr, detail=1, paragraph=False)
        except Exception:
            logger.warning("lettering-remove: OCR failed", exc_info=True)
            continue
        for box, text, conf in found:
            if not str(text or "").strip() or conf < 0.2:
                continue
            xs = [p[0] / scale for p in box]; ys = [p[1] / scale for p in box]
            boxes.append((int(max(0, min(xs))), int(max(0, min(ys))),
                          int(min(w, max(xs))), int(min(h, max(ys)))))
    return boxes


def stroke_mask(image_path: str, boxes: List[Box], pad: int = 6, halo: int = 4):
    """L-mode mask of the LETTERS inside the boxes, not the boxes.

    A box around "lady" also covers the feather hem behind it, and every
    covered pixel gets repainted. Inside a box the background is what its
    border pixels look like; a pixel far from all of them in colour is ink
    (or its glow). Those pixels, grown by `halo`, are the mask. A box whose
    ink is not separable (ink ~ background, or ink covers most of it) falls
    back to the whole box."""
    import numpy as np
    from PIL import Image, ImageFilter
    with Image.open(image_path) as im:
        a = np.asarray(im.convert("RGB"), dtype=float)
    h, w = a.shape[:2]
    m = np.zeros((h, w), bool)
    ring = 12
    inside = np.zeros((h, w), bool)
    for (x0, y0, x1, y1) in boxes:
        inside[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = True
    for (x0, y0, x1, y1) in boxes:
        bx0, by0 = max(0, x0 - pad), max(0, y0 - pad)
        bx1, by1 = min(w, x1 + pad), min(h, y1 + pad)
        tile = a[by0:by1, bx0:bx1]
        if tile.size == 0:
            continue
        # Background = a ring OUTSIDE every box: the box edge itself runs
        # through the letters and their glow, so its pixels are half ink.
        rx0, ry0 = max(0, bx0 - ring), max(0, by0 - ring)
        rx1, ry1 = min(w, bx1 + ring), min(h, by1 + ring)
        sel = ~inside[ry0:ry1, rx0:rx1]
        border = a[ry0:ry1, rx0:rx1][sel]
        if len(border) < 16:
            m[by0:by1, bx0:bx1] = True
            continue
        idx = np.linspace(0, len(border) - 1, min(128, len(border))).astype(int)
        ref = border[idx]
        d = np.min(np.linalg.norm(tile[:, :, None, :] - ref[None, None], axis=-1), axis=-1)
        ink = d > 60
        frac = ink.mean()
        if frac < 0.02 or frac > 0.7:
            m[by0:by1, bx0:bx1] = True          # not separable: the whole box
        else:
            m[by0:by1, bx0:bx1] |= ink
    out = Image.fromarray((m * 255).astype("uint8"), "L")
    if halo:
        out = out.filter(ImageFilter.MaxFilter(2 * halo + 1))
    return out


def build_mask(image_path: str, boxes: List[Box], out_path: str, pad: int = 6,
               protect: Optional[List[Box]] = None) -> Tuple[str, int]:
    """White-on-black mask PNG of the boxes (padded), minus protected areas.
    Returns (path, number of boxes used)."""
    from PIL import Image, ImageDraw
    with Image.open(image_path) as im:
        w, h = im.size
    if os.getenv("LETTERING_MASK", "strokes") == "strokes":
        mask = stroke_mask(image_path, boxes, pad=pad)
    else:
        mask = Image.new("L", (w, h), 0)
        for (x0, y0, x1, y1) in boxes:
            ImageDraw.Draw(mask).rectangle(
                (max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)), fill=255)
    d = ImageDraw.Draw(mask)
    used = len(boxes)
    # A face is never repainted: the face area is cut OUT of the mask rather
    # than the whole box dropped -- the "USEMAID" header behind a head is
    # still lettering on either side of it.
    for (fx0, fy0, fx1, fy1) in (protect or []):
        d.rectangle((fx0, fy0, fx1, fy1), fill=0)
    if not mask.getbbox():
        used = 0
    mask.save(out_path)
    return out_path, used


def _faces(image_path: str) -> List[Box]:
    try:
        import identity_metrics
        return [tuple(int(v) for v in b) for b in (identity_metrics.face_boxes(image_path) or [])]
    except Exception:
        return []


FILL_INSTRUCTION = ("Remove all the lettering, logos and watermarks in this area. Continue the "
                    "surrounding textures, fabric, fur and floor seamlessly, in the same drawing "
                    "style and lighting. Plain surfaces stay plain: add no new lines, folds, "
                    "objects or text.")


SMEAR_RATIO = 0.6   # fill texture / surrounding texture below this = a smear


def smear_ratio(path: str, boxes: List[Box], pad: int = 16, ring: int = 60) -> float:
    """Fine-detail energy inside the filled boxes relative to a ring around them.

    "The OCR reads nothing" was the only success test, so a LaMa smear over a
    feather hem shipped (2026-09-24: inside 2.5 vs ring 6.1; the untouched
    source measured 10.1 inside). A real fill keeps texture close to its
    surroundings; a flat patch drops far below them. 1.0 = no evidence."""
    try:
        import numpy as np
        from PIL import Image, ImageFilter
        with Image.open(path) as im:
            g = im.convert("L")
        a = np.asarray(g, dtype=float)
        e = np.abs(a - np.asarray(g.filter(ImageFilter.GaussianBlur(2)), dtype=float))
        m = np.zeros(e.shape, bool); r = np.zeros(e.shape, bool)
        for x0, y0, x1, y1 in boxes:
            m[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = True
            r[max(0, y0 - ring):y1 + ring, max(0, x0 - ring):x1 + ring] = True
        r &= ~m
        if not m.any() or not r.any() or e[r].mean() < 1.0:
            return 1.0                      # flat surroundings: nothing to compare
        return float(e[m].mean() / e[r].mean())
    except Exception:
        logger.warning("lettering-remove: smear measure failed", exc_info=True)
        return 1.0


OC_PYTHON = venv_python(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "venv_eraser"))
OC_WORKER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "objectclear_worker.py")


def _fill_objectclear(ctx, image_path: str, mask_path: str, grow: int) -> Optional[str]:
    """ObjectClear, the user's pick on the 26-item eraser A/B (2026-09-25):
    FireRed left embossed ghosts of the letters. Own venv, whole card (the chat
    model is evicted for it and brought back). None -> the caller uses FireRed."""
    if (os.getenv("LETTERING_ENGINE", "objectclear") != "objectclear" or os.getenv("F5_TEST_RUN")
            or not os.path.exists(OC_PYTHON)):
        return None
    import subprocess
    import comfy_client
    import image as _image
    out = str(_image.OUTPUT_DIR / f"lettering_clean_{int(time.time() * 1000)}.png")
    try:
        with comfy_client._gpu_slot(exclusive=True, label="objectclear"):
            comfy_client._free_comfy_models()
            p = subprocess.run([OC_PYTHON, OC_WORKER, image_path, mask_path, out, str(grow)],
                               capture_output=True, text=True, timeout=600,
                               env=dict(os.environ, HF_HUB_OFFLINE="1", PYTHONIOENCODING="utf-8"))
        if p.returncode == 0 and os.path.exists(out):
            return out
        logger.warning("lettering-remove: ObjectClear failed (%s): %s", p.returncode, (p.stderr or "")[-400:])
    except Exception:
        logger.warning("lettering-remove: ObjectClear raised", exc_info=True)
    return None


def _fill_firered(ctx, image_path: str, mask_path: str, grow: int) -> Optional[str]:
    try:
        import image_contained_firered as _cf
        import image as _image
        with _image.firered_extra_lora(_image.REMOVAL_LORA, _image.REMOVAL_LORA_STRENGTH):
            # The phrase is what the STAGE-2 QA asks about: "" became
            # 'manual region' and the VLM rejected a clean fill as WRONG.
            return _cf.edit_region_contained_via_firered(
                ctx, image_path, "lettering", FILL_INSTRUCTION,
                grow=grow, mask_override=mask_path, vlm_qa=False)
    except Exception:
        logger.warning("lettering-remove: FireRed fill raised", exc_info=True)
        return None


def remove_lettering(ctx, image_path: str, *, grow: int = 10, passes: int = 2,
                     timeout: int = 900) -> Optional[str]:
    """Every piece of lettering the OCR reads is masked and FireRed-filled;
    a second pass with a wider mask cleans what the OCR still reads."""
    import image as _image
    boxes = ocr_boxes(image_path)
    if boxes is None:
        logger.warning("lettering-remove: OCR unavailable -- nothing to mask")
        return None
    if not boxes:
        logger.info("lettering-remove: the OCR reads no lettering in %s", image_path)
        return None
    protect = _faces(image_path)
    current = image_path
    pad = 6
    for p in range(1, passes + 1):
        ts = int(time.time() * 1000)
        mask_path = str(scratch_path(_image.OUTPUT_DIR, f"_INTERMEDIATE_lettering_mask_{ts}.png"))
        mask_path, used = build_mask(current, boxes, mask_path, pad=pad, protect=protect)
        if not used:
            logger.info("lettering-remove: every box sits on a face -- nothing removed")
            return current if current != image_path else None
        # FireRed repaints the masked crop: it continues fur, folds, floor.
        # (The Big-LaMa fill smeared a flat rectangle over a feather hem,
        # 2026-09-24, and was removed.) A fill that measures as a smear is
        # refused, never shipped.
        if ctx is not None and hasattr(ctx, "set_stage"):
            ctx.set_stage("Removing the lettering" + (f" ({p}/{passes})" if passes > 1 else ""))
        logger.info("lettering-remove pass %d: %d box(es), pad=%d grow=%d", p, used, pad, grow)
        out = (_fill_objectclear(ctx, current, mask_path, grow)
               or _fill_firered(ctx, current, mask_path, grow))
        if out:
            sr = smear_ratio(out, boxes)
            if sr < SMEAR_RATIO:
                logger.warning("lettering-remove: fill is a smear (%.2f < %.2f) -- rejected",
                               sr, SMEAR_RATIO)
                out = None
        if not out:
            return current if current != image_path else None
        current = out
        left = ocr_boxes(current) or []
        if not left:
            logger.info("lettering-remove: the OCR reads nothing after pass %d", p)
            return current
        logger.info("lettering-remove: %d piece(s) still readable after pass %d", len(left), p)
        boxes = left
        pad += 8
        grow += 6
    return current


LEFTOVER_PROMPT = (
    "Look at the BACKGROUND of this picture (not the person, not their clothes "
    "or jewellery). Is there any lettering, brand logo, icon or watermark left "
    "in it? If nothing of the kind is visible, answer exactly: NONE. Otherwise "
    "name ONE item in at most six English words that a segmenter could find, "
    "e.g. 'audible logo top right' or 'white letters top left'.")


def vision_leftover(ctx, image_path: str) -> str:
    """A logo or icon the OCR cannot read, named by the vision model ('' = clean)."""
    try:
        import llm as _llm
        got = (_llm.analyze_image_with_llm(
            ctx, image_path=image_path, user_text="Anything left?",
            system_prompt=LEFTOVER_PROMPT, temperature=0.1, max_tokens=40) or "").strip()
    except Exception:
        logger.warning("lettering-remove: leftover check failed", exc_info=True)
        return ""
    if not got or got.upper().startswith("NONE"):
        return ""
    phrase = " ".join(got.split())[:60].strip(" .\"'")
    return "" if len(phrase.split()) > 8 else phrase


def remove_lettering_and_logos(ctx, image_path: str, **kw) -> Optional[str]:
    """OCR-masked removal, then ONE segmenter pass for a logo/icon the vision
    model still sees (the audible icon beside the head survived the OCR pass
    on the red-carpet asset: it is a picture, not letters)."""
    out = remove_lettering(ctx, image_path, **kw)
    base = out or image_path
    if ctx is None or (ctx.is_cancelled() if hasattr(ctx, "is_cancelled") else False):
        return out
    phrase = vision_leftover(ctx, base)
    if not phrase:
        return out
    logger.info("lettering-remove: the vision model still sees %r -- one segmenter pass", phrase)
    try:
        import image_objects
        ctx.set_stage("Removing the lettering (logo)")
        # a leftover logo is a small thing: a mask over a fifth of the picture is the
        # wall or the person, and filling it rewrites them
        more = image_objects.remove_object_with_comfy(ctx, base, phrase, max_cover=0.2)
    except Exception:
        logger.warning("lettering-remove: logo pass failed", exc_info=True)
        more = None
    return more or out

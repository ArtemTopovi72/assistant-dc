"""The loupe: find something on a picture and show WHERE it is.

A copy of the photo with the region highlighted — a translucent fill plus a
bright outline (and a number per instance when there are several) — next to
the untouched original. Three locators, in the order they are tried:

  SAM3   text-prompted segmentation (the same mask the edits mask with), one
         bounding box per connected mask blob, so «обе белки» lights up two;
  OCR    PaddleOCR/EasyOCR word boxes, for queries about WRITTEN text — «где
         написано привет» highlights the matching words, not the squirrel;
  VLM    the vision model asked for approximate boxes on the project's 0-1000
         grid (the object_refs/ideogram convention) — the fallback that still
         points at the right part of the frame when a mask comes back empty.

Whether the query is about lettering is the LLM's read (seams.yes_no), not a
word list: «где тут подпись» and "where's the sign" both route to OCR without
either language being enumerated. This layer talks to the models and ComfyUI
but never edits the source photo — it writes a NEW file beside it.
"""
from seams import yes_no, vision_model
import difflib
import hashlib
import json
import logging
import os
import re
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("assistant.image")

Box = Tuple[float, float, float, float]      # x0, y0, x1, y1 in source pixels

# Sanity limits for a single highlighted region, as fractions of the frame.
_MIN_BOX_FRAC = 0.004       # a speck smaller than this is noise, not a find
_MAX_BOX_FRAC = 0.98        # a box covering the whole frame located nothing
_MAX_BOXES = 12             # «все белки» in a squirrel colony is still finite


def _asks_multiple(query: str) -> bool:
    """Only an explicit count justifies refusing one grounded instance."""
    return bool(re.search(
        r"\b(?:both|all|several|many|two|three|four|five|оба|обе|обоих|"
        r"все|всех|несколько|два|две|двух|три|трех|трёх|четыре|пять|[2-9]\d*)\b",
        query or "", re.I))


# ── routing: lettering or thing? ─────────────────────────────────────────────
def _wants_lettering(ctx, query: str) -> bool:
    """The query asks WHERE SOMETHING IS WRITTEN, not where a thing is. The
    model's read (yes_no seam) — a word list would have to enumerate both
    languages and still miss «где тут подпись» and "the sign"."""
    return yes_no(ctx)(
        "A user asks to locate and highlight something in a picture:\n{text}\n\n"
        "written = it asks about WRITTEN or printed TEXT — a word, a phrase, a sign, "
        "a caption, a label, a title, lettering, a number painted on something. "
        "thing = it asks about an object, a person, an animal or a place: a squirrel, "
        "the clock, the red car, grandma's face, the left tower.", query or "")


def _query_terms(query: str) -> List[str]:
    """Searchable words of the query for OCR matching: lowercase alphanumeric
    tokens, 2+ chars, minus the locating verbs («где», «написано», "where")."""
    stop = {"the", "a", "an", "is", "it", "this", "that", "where", "wheres",
            "written", "write", "text", "word", "words", "sign", "says",
            "где", "тут", "здесь", "написано", "написана", "надпись",
            "надписью", "текст", "слово", "слова", "вот"}
    words = re.findall(r"[0-9a-zа-яё]+", (query or "").lower())
    return [w for w in words if len(w) >= 2 and w not in stop][:6] or \
           [w for w in words if len(w) >= 2][:3]


def _match_piece(piece_text: str, terms: List[str], all_ok: bool) -> bool:
    """Does one OCR piece answer the query? Containment or a close fuzzy ratio
    — OCR misreads single glyphs and an exact == would miss the hit."""
    t = (piece_text or "").strip().lower()
    if not t:
        return False
    if all_ok:
        return True
    for term in terms:
        if term in t or t in term:
            return True
        if len(term) >= 4 and difflib.SequenceMatcher(None, term, t[:60]).ratio() >= 0.6:
            return True
    return False


def _ocr_boxes(ctx, image_path: str, terms: List[str], all_ok: bool) -> List[Box]:
    """Pixel boxes of the OCR pieces the query names. [] when OCR is
    unavailable or nothing it read matches — the caller falls through to the
    other locators, so «белка» on a picture with a caption still finds the
    squirrel, not nothing."""
    if os.getenv("F5_TEST_RUN"):
        return []
    try:
        import ocr_reader
        pieces = ocr_reader.read_boxes(image_path)
        if pieces is None:
            return []
        boxes = []
        # this build's read_pieces returns flat STRINGS (no boxes): the
        # lettering path needs boxes it does not get, so it stays a no-op
        # instead of raising (live 2026-11-03: the unpack crashed silently).
        if pieces and not isinstance(pieces[0], (list, tuple)):
            return []
        for box, text, conf in pieces:
            if conf < 0.4 or not _match_piece(str(text), terms, all_ok):
                continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
        return boxes[:_MAX_BOXES]
    except Exception:
        logger.debug("loupe OCR pass failed", exc_info=True)
        return []


# ── boxes from a segmentation mask ───────────────────────────────────────────
def _mask_boxes(mask_path: str) -> List[Box]:
    """One bounding box per connected white blob of a SAM3/Florence mask, specks
    dropped. scipy labels the whole mask in C; a pure-Python flood fill over a
    multi-megapixel mask would crawl."""
    try:
        import numpy as np
        from scipy import ndimage
        with Image.open(mask_path) as im:
            arr = np.asarray(im.convert("L"))
        if arr.size == 0:
            return []
        solid = arr > 127
        if not solid.any():
            return []
        labels, n = ndimage.label(solid)
        h, w = arr.shape
        min_area = max(64.0, h * w * 0.0006)     # 0.06% of the frame
        out = []
        for i in range(1, n + 1):
            ys, xs = np.where(labels == i)
            if xs.size < min_area:
                continue
            out.append((int(xs.size), (float(xs.min()), float(ys.min()),
                        float(xs.max()), float(ys.max()))))
        out.sort(key=lambda item: item[0], reverse=True)
        # SAM3 often leaves a disconnected hand or fabric patch beside the
        # person. It is a fragment of the same find, not a second instance.
        largest = out[0][0] if out else 0
        return [box for area, box in out if area >= largest * 0.25][:_MAX_BOXES]
    except Exception:
        logger.debug("loupe mask->boxes failed", exc_info=True)
        return []


def _same_colour_instances(image_path: str, mask_path: str) -> List[Box]:
    """Recover repeated flat-colour shapes when SAM3 selected one of them.

    Only act when the selected pixels are nearly uniform and other components
    have comparable size and shape. Photographs normally fail that gate and
    keep the model-based route; this is for logos, icons and simple diagrams.
    """
    try:
        import numpy as np
        from scipy import ndimage
        with Image.open(image_path) as im, Image.open(mask_path) as mask:
            rgb = np.asarray(im.convert("RGB"), dtype=np.int16)
            m = np.asarray(mask.convert("L").resize(im.size, Image.NEAREST)) > 127
        pixels = rgb[m]
        if len(pixels) < 100:
            return []
        colour = np.median(pixels, axis=0)
        distances = np.max(np.abs(pixels - colour), axis=1)
        if np.percentile(distances, 75) > 18:
            return []
        similar = np.max(np.abs(rgb - colour), axis=2) <= 24
        labels, n = ndimage.label(similar)
        slices = ndimage.find_objects(labels)
        seed_area = int(m.sum())
        seed_boxes = _mask_boxes(mask_path)
        if not seed_boxes:
            return []
        sw = seed_boxes[0][2] - seed_boxes[0][0]
        sh = seed_boxes[0][3] - seed_boxes[0][1]
        boxes = []
        for i, sl in enumerate(slices, 1):
            if sl is None:
                continue
            y, x = sl
            area = int((labels[sl] == i).sum())
            bw, bh = x.stop - x.start, y.stop - y.start
            if not (0.35 * seed_area <= area <= 1.8 * seed_area
                    and 0.6 * sw <= bw <= 1.5 * sw
                    and 0.6 * sh <= bh <= 1.5 * sh):
                continue
            boxes.append((float(x.start), float(y.start), float(x.stop), float(y.stop)))
        return sorted(boxes, key=lambda b: (b[1], b[0]))[:_MAX_BOXES]
    except Exception:
        logger.debug("loupe repeated-colour pass failed", exc_info=True)
        return []


# ── boxes from the vision model (the 0-1000 grid fallback) ───────────────────
_VLM_BOX_PROMPT = (
    "You locate things in images. Find every instance of the target the user "
    "names. Reply with ONLY JSON:\n"
    '{"found": true|false, "boxes": [[x0, y0, x1, y1], ...]}\n'
    "Coordinates are on a 0-1000 grid: (0,0) top-left, (1000,1000) "
    "bottom-right; x0<x1, y0<y1. Each box hugs ONE instance of the target. "
    "found=false with an empty list when the target is not in the picture. "
    "JSON only, no prose."
)


def _vlm_boxes(ctx, image_path: str, phrase_en: str) -> List[Box]:
    """Approximate boxes from the vision model — the locator of last resort.
    NO prefill: a closed-<think> prefill returns EMPTY on vision tasks with
    these fine-tunes (verified live, the same bug that broke inspect_image)."""
    try:
        from utils import safe_json_from_llm
        resp = vision_model(ctx)(
            ctx=ctx, image_path=image_path,
            user_text=f"Target: {phrase_en}",
            system_prompt=_VLM_BOX_PROMPT, max_tokens=300)
        data = safe_json_from_llm(resp or "") if resp else None
        if not isinstance(data, dict):
            return []
        if data.get("found") is False:
            return []
        with Image.open(image_path) as im:
            w, h = im.size
        out = []
        for b in data.get("boxes") or []:
            try:
                x0, y0, x1, y1 = (float(v) for v in b[:4])
            except (TypeError, ValueError):
                continue
            if not (x0 < x1 and y0 < y1):
                x0, x1 = min(x0, x1), max(x0, x1)
                y0, y1 = min(y0, y1), max(y0, y1)
            box = (x0 / 1000.0 * w, y0 / 1000.0 * h, x1 / 1000.0 * w, y1 / 1000.0 * h)
            if _sane(box, w, h):
                out.append(box)
        return out[:_MAX_BOXES]
    except Exception:
        logger.debug("loupe VLM pass failed", exc_info=True)
        return []


def _sane(box: Box, w: int, h: int) -> bool:
    """A usable box: inside the frame, neither a speck nor the whole frame."""
    x0, y0, x1, y1 = box
    if x1 - x0 < max(8.0, w * 0.01) or y1 - y0 < max(8.0, h * 0.01):
        return False
    frac = ((x1 - x0) * (y1 - y0)) / float(max(1, w * h))
    return _MIN_BOX_FRAC <= frac <= _MAX_BOX_FRAC


# ── drawing ──────────────────────────────────────────────────────────────────
def _font(size: int) -> ImageFont.FreeTypeFont:
    for cand in (r"C:\Windows\Fonts\arialbd.ttf", "arialbd.ttf", "Arial Bold"):
        try:
            return ImageFont.truetype(cand, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _pad(box: Box, w: int, h: int, frac: float = 0.02) -> Box:
    """Breathe around the find: the mask hugs the object, the outline should
    not shave its edges."""
    px = max(6.0, frac * max(w, h))
    x0, y0, x1, y1 = box
    return (max(0.0, x0 - px), max(0.0, y0 - px),
            min(float(w), x1 + px), min(float(h), y1 + px))


def draw(image_path: str, boxes: List[Box], out_path: str) -> str:
    """The highlight look: a translucent amber wash + a bright outline with
    rounded corners + bracket ticks, and a numbered badge per find when there
    are several. Writes a NEW file; the source is never touched."""
    with Image.open(image_path) as im:
        img = im.convert("RGBA")
    w, h = img.size
    wash = Image.new("RGBA", img.size, (0, 0, 0, 0))
    wd = ImageDraw.Draw(wash)
    line_w = max(3, int(round(max(w, h) * 0.004)))
    badge_r = max(12, int(round(max(w, h) * 0.022)))
    for i, box in enumerate(boxes, 1):
        x0, y0, x1, y1 = _pad(box, w, h)
        r = min((x1 - x0), (y1 - y0)) * 0.12
        wd.rounded_rectangle((x0, y0, x1, y1), radius=r,
                             fill=(255, 196, 0, 56), outline=(255, 214, 64, 255),
                             width=line_w)
        # bracket ticks: the "look here" corners a loupe answer reads by
        tl = line_w * 2.4
        for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            wd.line((cx, cy, cx + dx * tl, cy), fill=(255, 240, 150, 255), width=line_w)
            wd.line((cx, cy, cx, cy + dy * tl), fill=(255, 240, 150, 255), width=line_w)
        if len(boxes) > 1:
            bx, by = x0 + badge_r * 0.9, y0 + badge_r * 0.9
            wd.ellipse((bx - badge_r, by - badge_r, bx + badge_r, by + badge_r),
                       fill=(255, 214, 64, 255), outline=(120, 84, 0, 255),
                       width=max(2, line_w // 2))
            f = _font(badge_r)
            t = str(i)
            tw = wd.textlength(t, font=f)
            th = badge_r * 1.15
            wd.text((bx - tw / 2, by - th / 2), t, font=f, fill=(60, 40, 0, 255),
                    anchor="mm")
    out = Image.alpha_composite(img, wash).convert("RGB")
    out.save(out_path, quality=95)
    return out_path


# ── the entry point ──────────────────────────────────────────────────────────
def highlight(ctx, image_path: str, query: str,
              out_path: Optional[str] = None) -> dict:
    """Locate `query` on `image_path` and draw the highlight on a copy.

    Returns {"path", "boxes", "via", "count"} — path "" when nothing was
    found (the caller tells the user; no empty frame is delivered). Never
    raises: every locator failure degrades to the next, and a total failure
    is an honest empty result."""
    if not image_path or not os.path.exists(image_path):
        return {"path": "", "boxes": [], "via": "", "count": 0}
    query = (query or "").strip()
    stem = os.path.basename(image_path)
    tag = hashlib.sha1((stem + query).encode("utf-8", "replace")).hexdigest()[:6]
    out_path = out_path or os.path.join(
        os.path.dirname(image_path), f"find_{tag}.jpg")
    try:
        with Image.open(image_path) as im:
            w, h = im.size
        phrase_en = ""
        boxes: List[Box] = []
        via = ""
        if _wants_lettering(ctx, query):
            terms = _query_terms(query)
            all_ok = not terms or query.lower().strip() in (
                "all", "все", "все надписи", "lettering", "текст")
            boxes = _ocr_boxes(ctx, image_path, terms, all_ok)
            via = "ocr" if boxes else ""
            if boxes:
                logger.info("loupe: %r located by OCR (%d piece(s))", query[:40], len(boxes))
        if not boxes:
            # Object route: the SAM3 mask the edits use, variants included.
            import image as _image
            import image_grounding
            phrase_en = image_grounding._english_region(ctx, query) or query
            uploaded = _image._upload_image_to_comfy(image_path, _image.COMFY_URL)
            mask = ""
            if uploaded:
                mask = _image._region_mask_file(ctx, uploaded, phrase_en,
                                                grow=4, seed=1, timeout=240) or ""
            if not (mask and os.path.exists(mask)):
                for variant in image_grounding._region_phrase_variants(phrase_en):
                    if not uploaded:
                        break
                    mask = _image._region_mask_file(ctx, uploaded, variant,
                                                    grow=4, seed=1, timeout=240) or ""
                    if mask and os.path.exists(mask):
                        phrase_en = variant
                        break
            if mask and os.path.exists(mask):
                boxes = _mask_boxes(mask)
                via = "sam3" if boxes else ""
                if boxes:
                    logger.info("loupe: %r -> %r located by SAM3 (%d region(s))",
                                query[:40], phrase_en, len(boxes))
                if len(boxes) == 1:
                    repeats = _same_colour_instances(image_path, mask)
                    if len(repeats) > 1:
                        boxes, via = repeats, "colour"
            # The VLM can misplace several boxes despite giving the requested
            # count. Keep the grounded mask when it found one, and decline an
            # explicit multi-instance request if the other instances cannot
            # be verified on pixels (e.g. differently coloured animals).
            if len(boxes) == 1 and _asks_multiple(query):
                return {"path": "", "boxes": [], "via": "", "count": 0}
        if not boxes and not via:
            boxes = _vlm_boxes(ctx, image_path, phrase_en or query)
            via = "vlm" if boxes else ""
            if boxes:
                logger.info("loupe: %r located by the vision model (%d box(es))",
                            query[:40], len(boxes))
        # A box the sanity gate rejects (a speck, the whole frame) located
        # nothing; drop it rather than deliver a meaningless highlight.
        boxes = [b for b in boxes if _sane(b, w, h)]
        if not boxes:
            logger.info("loupe: nothing found for %r", query[:60])
            return {"path": "", "boxes": [], "via": "", "count": 0}
        return {"path": draw(image_path, boxes, out_path), "boxes": boxes,
                "via": via or "vlm", "count": len(boxes)}
    except Exception:
        logger.exception("loupe failed for %r on %s", query[:60], image_path)
        return {"path": "", "boxes": [], "via": "", "count": 0}

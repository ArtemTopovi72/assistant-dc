"""The mask quality gate.

STAGE-0 is deterministic (area/face-fraction/component statistics) and rejects
the speck masks the VLM check happily approves; STAGE-1 is the VLM verdict on a
rendered cutout/overlay. Also holds the face-detector mask and the whole-face
phrase pattern the gate keys off.

Extracted from image.py. MASK_QA_ENABLED is a live switch: read it through this
module (image_maskqa.MASK_QA_ENABLED), never by value, or patching it moves only
half the behaviour.
"""
from config import scratch_path
import logging
import os
import re
import time
from typing import Optional

from config import OUTPUT_DIR
from llm import analyze_image_with_llm

logger = logging.getLogger("assistant.image")


# Whole-face regions (the FACE transfer preset and equivalents). For THESE the
# YuNet face detector gives a far more reliable mask than SAM3/Florence text
# segmentation, which routinely grabs a stray fragment of the face instead of the
# whole thing. Sub-features (lips/eyes/nose/cheek...) are deliberately NOT matched
# here, so they still go through referring segmentation where a precise small mask
# is what you want.
def _is_whole_face(phrase: str) -> bool:
    """The region is a whole face or head -- not one feature of it (eyes, lips)
    and not a person or body that also names the face. Read by the model: the
    old word regex took «the man's body and face» for a face, YuNet masked only
    the face and the man stayed in his chair (live 10-02)."""
    import intent
    return intent.ask_yes(
        "An image region is described as: {text}\n\nIs this region exactly a whole "
        "face or head -- not just one feature of it (eyes, lips, nose), and not a "
        "whole person or body?", phrase)


def _mask_face_fraction(mask_l, face_box) -> Optional[float]:
    """Fraction of the (padded) face box that the mask selects, or None on error.
    Used by the person-grab gate: a mask for a non-face region that engulfs the
    face has segmented the person, not the requested object."""
    try:
        import numpy as np
        x0, y0, x1, y1 = (int(v) for v in face_box)
        a = np.asarray(mask_l)[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
        if a.size == 0:
            return None
        return float((a > 24).mean())
    except Exception:
        return None


def _face_detector_mask(image_path: str, size, pad: float = 0.18):
    """Soft face mask from the YuNet detector: an ellipse over the largest detected
    face (corners of the box trimmed so background isn't dragged in). Returns a
    full-size 'L' mask, or None if detection is unavailable / no face found.

    This is the reliable replacement for text-segmenting the word "face" — the
    detector localizes the actual face instead of a random fragment."""
    try:
        import identity_metrics as _idm
        fb = _idm.face_box(image_path, pad=pad)
    except Exception as exc:
        logger.warning("face-mask: detector unavailable (%s)", exc)
        return None
    if not fb:
        return None
    from PIL import Image, ImageDraw, ImageFilter
    sw, sh = size
    mask = Image.new("L", (sw, sh), 0)
    ImageDraw.Draw(mask).ellipse(fb, fill=255)
    # Soften the ellipse edge a touch so it isn't a hard oval before the downstream
    # composite feather; radius scales with face size.
    r = max(2, int(0.025 * (fb[2] - fb[0])))
    return mask.filter(ImageFilter.GaussianBlur(r))


# Two-stage vision QA of the segmentation, gated behind this flag. Both stages
# FAIL-OPEN: if no multimodal model is loaded (e.g. VRAM is held by the chat LLM),
# the vision call errors and QA is skipped — the pipeline behaves exactly as before.
# The QA only becomes active once a VL model can answer.
MASK_QA_ENABLED = True


def _mask_quality_ok(mask_l, *, region_phrase: str = "",
                     small_item: Optional[bool] = None,
                     big_region: Optional[bool] = None) -> tuple:
    """Deterministic sanity gate on a segmentation mask — the cheap, reliable check
    that the VLM cutout/overlay QA does NOT provide (those fail open, so a mostly-black
    cutout of scattered specks yields an 'unclear' verdict that is then ACCEPTED).

    Rejects the two unambiguous garbage shapes a failed SAM3/Florence run produces:
      * near-empty masks, and
      * scattered disconnected specks sprinkled across a large bounding box (a huge
        bbox with almost nothing filled and no dominant blob) — the 'tattoo mask
        churned out nonsense' case.

    Conservative on purpose: a legitimate region (even a thin, connected line-art
    tattoo outline) is ONE dominant connected component, so `largest_frac` stays high
    and it passes. Rejection requires MULTIPLE weak signals at once.

    Returns (ok: bool, reason: str, stats: dict).
    """
    import numpy as np
    a = np.asarray(mask_l) > 24
    H, W = a.shape
    total = int(a.sum())
    stats = {"cov": 0.0, "fill": 0.0, "n_comp": 0, "largest_frac": 0.0}
    if total == 0:
        return False, "empty", stats
    cov = total / float(H * W)
    ys, xs = np.where(a)
    bw = int(xs.max() - xs.min() + 1)
    bh = int(ys.max() - ys.min() + 1)
    fill = total / float(bw * bh)            # how solidly the bbox is filled
    try:
        from scipy.ndimage import label
        _lbl, n_comp = label(a)
        if n_comp > 0:
            sizes = np.bincount(_lbl.ravel())[1:]
            largest_frac = float(sizes.max()) / float(total)
        else:
            largest_frac = 0.0
    except Exception:
        n_comp, largest_frac = 1, 1.0       # can't measure -> don't block on fragmentation
    stats = {"cov": round(cov, 4), "fill": round(fill, 3),
             "n_comp": int(n_comp), "largest_frac": round(largest_frac, 3)}

    if cov < 0.0005:
        return False, "near_empty", stats
    # OVER-COVERAGE: a localized/contained edit must not select almost the whole frame.
    # White = the region that gets regenerated/composited, so a ~98%-white mask repaints
    # nearly the entire image (the "tattoo transfer re-rendered the whole person, with
    # the requested detail coming out as garbage" failure). The VLM cutout QA is blind to
    # this — a near-full-mask cutout shows ~the entire valid image and reads as "correct".
    # Region-aware: genuine large-area edits (background/sky/wall) are allowed to be
    # big. The size hints come from the LLM classifier (_item_attributes) via the
    # caller — this function stays deterministic. A minimal keyword fallback keeps
    # the big-region allowance when no hint was supplied (LLM down / legacy caller).
    if big_region is None:
        big_region = bool(re.search(r"background|backdrop|\bsky\b|\bwall\b|\bscene\b|"
                                    r"entire|whole|everything|\bфон\b", region_phrase or "",
                                    re.IGNORECASE))
    # SIZE PRIOR for small worn/held items: a mask for "shoes"/"watch"/"earring"
    # that covers a third of the frame is definitively the wrong object (Florence
    # grabbed the whole person) — the source of the "shoes edit re-rendered the
    # girl with a duplicated head" failure. 25% of the frame is far above any
    # plausible small item while safely below a legitimate dress/hair region.
    # small_item=None (unknown) keeps the neutral 85% cap.
    cap = 0.97 if big_region else (0.25 if small_item else 0.85)
    if cov > cap:
        return False, "over_coverage", stats
    # Scattered specks: low bbox fill AND no single dominant component.
    if fill < 0.12 and largest_frac < 0.50:
        return False, "fragmented_sparse", stats
    # Many disconnected pieces with no dominant blob (dust storm).
    if n_comp >= 12 and largest_frac < 0.60:
        return False, "too_fragmented", stats
    return True, "ok", stats


def _save_qa_png(im, prefix: str):
    """Persist a QA visualization to OUTPUT_DIR; return its path or None."""
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        p = scratch_path(OUTPUT_DIR, f"{prefix}_{int(time.time()*1000)}.png")
        im.save(p)
        return str(p)
    except Exception as exc:
        logger.warning("QA png save failed: %s", exc)
        return None


def _qa_crop_box(bbox, size, ctx_pad: float = 0.15):
    """Expand a bbox by a margin (clamped to image size) so QA sees a little context."""
    sw, sh = size
    x0, y0, x1, y1 = bbox
    px = int((x1 - x0) * ctx_pad) + 8
    py = int((y1 - y0) * ctx_pad) + 8
    return (max(0, x0 - px), max(0, y0 - py), min(sw, x1 + px), min(sh, y1 + py))


def _mask_cutout_png(orig_rgb, mask_l, bbox):
    """STAGE-1 visual: image x mask (selected pixels kept, rest black), cropped to the
    region. Lets the VLM verify SAM3 isolated the RIGHT object."""
    from PIL import Image
    black = Image.new("RGB", orig_rgb.size, (0, 0, 0))
    cut = Image.composite(orig_rgb, black, mask_l)
    box = _qa_crop_box(bbox, orig_rgb.size)
    return _save_qa_png(cut.crop(box), "_QA_cutout")


_TINTS = {"red": (255, 0, 0), "cyan": (0, 220, 255), "green": (0, 230, 0), "magenta": (255, 0, 255)}


def qa_tint(orig_rgb, result_rgb, mask_l) -> str:
    """The highlight colour least like the region itself. A red tint over a red
    dress on a red wall told the VLM nothing, and it rejected a good edit WRONG
    (live 2026-09-25)."""
    import numpy as np
    m = np.asarray(mask_l.convert("L").resize(orig_rgb.size)) > 127
    if not m.any():
        return "red"
    px = np.concatenate([np.asarray(orig_rgb.convert("RGB"))[m],
                         np.asarray(result_rgb.convert("RGB").resize(orig_rgb.size))[m]]).astype(float)
    mean = px.mean(axis=0)
    return max(_TINTS, key=lambda k: float(((np.array(_TINTS[k]) - mean) ** 2).sum()))


def _overlay_preview_png(orig_rgb, result_rgb, mask_l, bbox, tint: str = "red"):
    """STAGE-2 visual: the generated edit composited back over the original, with the
    intended mask region tinted semi-transparent red so the VLM can judge whether the
    edit is aligned, stays inside the region, and doesn't leak outside it."""
    from PIL import Image
    base = result_rgb if result_rgb.size == orig_rgb.size else result_rgb.resize(orig_rgb.size)
    tint = Image.new("RGB", base.size, _TINTS.get(tint, (255, 0, 0)))
    # 35% red only where the mask is white -> a translucent highlight of the region.
    faded = Image.blend(base, tint, 0.35)
    preview = Image.composite(faded, base, mask_l)
    box = _qa_crop_box(bbox, orig_rgb.size)
    return _save_qa_png(preview.crop(box), "_QA_overlay")


def _vlm_qa_verdict(ctx, png_path, question: str, labels):
    """Ask the VLM a strict single-label QA question about ``png_path``. Returns an
    uppercase label from ``labels`` or None (unavailable / unclear) — None always
    means 'proceed' so QA never blocks when no VL model is loaded."""
    if not MASK_QA_ENABLED or not png_path:
        return None
    sys_prompt = (
        "You are a strict image-edit QA inspector. Judge ONLY what is visible. "
        f"{question} Answer with EXACTLY ONE word, the first word of your reply, "
        f"chosen from: {', '.join(labels)}. You may add a few words of reason after it."
    )
    try:
        # NO prefill: a closed-<think> prefill returns EMPTY on vision tasks with
        # these fine-tunes (verified live) — an empty verdict here made the QA gate
        # silently fail open on every check.
        resp = analyze_image_with_llm(
            ctx=ctx, image_path=png_path, user_text="Give your one-word verdict.",
            system_prompt=sys_prompt, max_tokens=24)
    except Exception as exc:
        logger.warning("mask QA vision call failed (%s) — proceeding without QA", exc)
        return None
    if not resp or not resp.strip():
        return None
    first = resp.strip().upper().split()[0].strip(".,:;!?\"'")
    for lab in labels:
        if first.startswith(lab.upper()):
            return lab.upper()
    logger.info("mask QA verdict unclear for %s: %r", os.path.basename(png_path), resp[:60])
    return None

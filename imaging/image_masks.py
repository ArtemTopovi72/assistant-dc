"""Mask construction and mask geometry.

Segmentation entry points (SAM3, Florence) plus everything that shapes a mask
afterwards: outward dilation, content-aware alpha, component filtering,
multi-instance completion, paired/removal mask extension, seam blending and the
crop-to-mask helper.

Extracted from image.py. Depends on image_grounding for item attributes and on
comfy_client for the ComfyUI round-trip; it deliberately knows nothing about the
edit pipelines that consume its masks.
"""
from config import scratch_path
import logging
import os
import re
import time
from typing import Optional

from config import OUTPUT_DIR
from compositing import _odd, _seamless_clone_arr
from comfy_client import _submit_and_poll
# NOTE: read through the module, not by value. _item_attributes is a documented
# test seam and is patched at runtime; binding it by value here would create a
# split-brain where patching one module moves only half the behaviour.
import image_grounding

logger = logging.getLogger("assistant.image")


def _item_attributes(*a, **kw):
    """Thin forwarder so runtime patches of image_grounding._item_attributes
    (and of image._item_attributes, which resolves to the same object) are seen
    by the mask pipeline."""
    return image_grounding._item_attributes(*a, **kw)


def _dilate_mask_outward(mask, *, frac: float = 0.05, min_px: int = 6,
                         max_px: int = 400):
    """Grow a region mask OUTWARD by a fraction of the region's own size.

    Segmentation masks (SAM3/Florence) hug the object and typically sit a few
    pixels INSIDE its true edge — the object's antialiased rim, contact shadow and
    colour bleed all fall just outside. Every downstream consumer then feathers
    or fills only up to that tight boundary, so a replace-edit (red dress ->
    trousers) leaves a ring of the OLD object's colour around the new content.
    The safety margin must therefore be OUTSIDE the mask and must scale with the
    region (a fixed pixel count is a sliver on a 20 MP dress and half the tile on
    a 400 px pair of glasses).

    Dilation radius = ``frac`` of the mask bbox's short side, clamped to
    [min_px, max_px] and to the image size. cv2 distance transform (exact, O(N),
    fine at 20 MP); PIL MaxFilter fallback (window capped so it can't stall).
    Returns an L-mode mask the same size as the input.
    """
    import numpy as np
    from PIL import Image
    m = np.asarray(mask.convert("L"))
    binm = (m > 24).astype(np.uint8)
    ys, xs = np.nonzero(binm)
    if xs.size == 0:
        return mask.convert("L")
    region = max(8, int(min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)))
    # Two scale carriers, not one. The margin exists to cover the old object's
    # antialiased rim + contact shadow: the rim width scales with IMAGE resolution
    # (its floor, ~0.4% of the short side — 6px at 1.5K, 16px at 4K, so a small
    # region on a high-res photo still clears its rim), while the shadow scales
    # with the region (frac). A huge region must NOT drag the margin up with it —
    # 5% of a 2500px dress is a 125px band of repainted background — so the
    # region term is capped at 2.5% of the image short side.
    img_min = int(min(m.shape))
    radius = int(round(min(frac * region, 0.025 * img_min, max_px,
                           max(1, img_min // 8))))
    radius = max(radius, min_px, int(round(0.004 * img_min)))
    try:
        import cv2
        dist_out = cv2.distanceTransform((1 - binm).astype(np.uint8), cv2.DIST_L2, 5)
        grown = (dist_out <= float(radius)).astype(np.uint8) * 255
        logger.info("mask-dilate: region=%dpx grown OUTWARD by %dpx (cv2)", region, radius)
        return Image.fromarray(grown, "L")
    except Exception as exc:
        from PIL import ImageFilter
        logger.warning("mask-dilate: cv2 path failed (%s); MaxFilter fallback", exc)
        win = _odd(min(2 * radius + 1, 61))
        binimg = Image.fromarray(binm * 255, "L")
        return binimg.filter(ImageFilter.MaxFilter(win))


def _content_aware_alpha(crop, res, mcrop, *, threshold: int = 16, grow: int = 2,
                         feather: int = 2, full_frac: float = 0.6):
    """Composite alpha that pastes ONLY the pixels the edit actually changed.

    For ADDITIVE edits (a tattoo on skin, an object on a wall) the re-rendered tile
    reproduces the surrounding background slightly differently from the original, so
    compositing the whole masked region through a feather leaves a visible soft
    ring / milky halo on uniform areas (the reported tattoo-edge artifact). Instead
    we keep only the pixels that differ from the original by > ``threshold`` (the
    actual ink/object), thicken them a touch, clip to the user mask, and feather a
    couple of px — so the surrounding skin stays byte-exact and no halo can form.

    Returns an L alpha, or ``None`` when the change fills most of the mask
    (``>= full_frac``) — i.e. a whole-region edit like clothing recolor — so the
    caller falls back to the solid-core region feather (existing behavior).
    """
    try:
        import numpy as np
        from PIL import Image, ImageFilter, ImageChops
    except Exception:
        return None
    o = np.asarray(crop.convert("RGB"), dtype=np.int16)
    r = np.asarray(res.convert("RGB"), dtype=np.int16)
    diff = np.abs(o - r).mean(axis=2)
    m = np.asarray(mcrop.convert("L")) > 24
    changed = (diff > threshold) & m
    mtot = int(m.sum())
    if mtot == 0:
        return None
    frac = float(changed.sum()) / mtot
    if frac >= full_frac:
        return None                      # whole-region edit -> use region feather
    # Scale the thicken + feather to the region size so an additive edit's edge is
    # soft at 20 MP yet crisp at 1 MP (the same reason the region overlay self-scales).
    ys, xs = np.nonzero(m)
    if xs.size:
        region = max(8, int(min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)))
        grow = max(grow, int(round(0.012 * region)))
        feather = max(feather, int(round(0.02 * region)))
    a = Image.fromarray((changed * 255).astype("uint8"), "L")
    if grow > 0:
        a = a.filter(ImageFilter.MaxFilter(_odd(2 * grow + 1)))   # thicken thin ink
    mbin = mcrop.point(lambda p: 255 if p > 24 else 0)
    a = ImageChops.multiply(a, mbin)     # keep growth inside the user mask
    if feather > 0:
        a = a.filter(ImageFilter.GaussianBlur(feather))
    logger.info("contained: content-aware composite — ink covers %.1f%% of mask "
                "(skin preserved, no halo)", 100.0 * frac)
    return a


def _no_new_faces(source_path: str, result_path: str) -> bool:
    """Deterministic duplicated-person gate: an edit must never ADD faces.

    When a garbage mask makes the 'contained' tile ≈ the whole frame, the
    instruction engine may re-COMPOSE the person (shifted/smaller); the
    face-protection box then keeps the ORIGINAL face at its old position while
    the render paints a second face elsewhere — the double-head artifact. The
    VLM QA is fail-open and misses it; counting YuNet faces is cheap and exact.
    Returns False only when the result confidently has MORE faces than the
    source (fail-open on any detector uncertainty)."""
    try:
        import identity_metrics as _idm
        before = _idm.face_count(source_path)
        after = _idm.face_count(result_path)
        if before is None or after is None:
            return True
        if after > before:
            logger.warning("dup-face gate: result has %d faces vs %d in source — "
                           "rejecting duplicated-person render", after, before)
            return False
    except Exception as exc:
        logger.info("dup-face gate skipped: %s", exc)
    return True


def _boundary_step_metric(crop, res, alpha) -> float:
    """How badly strong edges MISALIGN across the composite blend band, in pixels.

    The failure this catches (live: "shove a shrunken leg into a full-sized one"):
    the instruction engine re-renders the subject at slightly different
    PROPORTIONS (thinner leg), and the feathered composite then stitches the new
    silhouette onto the old one with a visible STEP — a strong original edge
    (leg-vs-wall) crossing the blend ring has no matching edge in the result at
    the same place. Rigid alignment can't fix this (it's a content change, not a
    transform), so it must be detected and the render rejected/retried.

    Metric: Canny edges of original and result; ORIGINAL edge pixels inside the
    blend ring (alpha between 16 and 240, where old and new content must agree)
    are measured against the distance transform of the result's edges — the
    85th-percentile distance. One-directional on purpose: original structure
    crossing the band must SURVIVE in the result (a displaced silhouette scores
    its step size), while brand-new edges the repaint adds in the band (texture,
    fill boundaries) are benign and must not trip the gate. Continuous
    silhouettes score ~1-2 px. 0.0 when there is nothing to measure (no band /
    too few edges / no cv2).
    """
    try:
        import numpy as np
        import cv2
        og = cv2.cvtColor(np.asarray(crop.convert("RGB")), cv2.COLOR_RGB2GRAY)
        rg = cv2.cvtColor(np.asarray(res.convert("RGB")), cv2.COLOR_RGB2GRAY)
        if og.shape != rg.shape:
            return 0.0
        a = np.asarray(alpha.convert("L"))
        band = (a > 16) & (a < 240)
        if band.sum() < 200:
            return 0.0
        e_og = cv2.Canny(og, 50, 150) > 0
        e_rg = cv2.Canny(rg, 50, 150) > 0
        og_band = e_og & band
        if og_band.sum() < 40:
            return 0.0            # featureless band (plain floor/wall): nothing to step
        dt_rg = cv2.distanceTransform((~e_rg).astype(np.uint8), cv2.DIST_L2, 3)
        return float(np.percentile(dt_rg[og_band], 85))
    except Exception as exc:
        logger.warning("boundary-step metric skipped: %s", exc)
        return 0.0


def _extend_mask_to_paired_changes(crop, res, mcrop, *, max_blobs: int = 2):
    """For PAIRED/PLURAL regions ("shoes", "gloves", "earrings"): when the
    segmenter found only ONE instance, the instruction engine still edits ALL of
    them in the tile (it understands the plural), and the mask-bounded composite
    then discards the other instance's fix ("removed the left shoe, the right
    stayed"). Recover the editor's own evidence: find compact change-blobs
    OUTSIDE the mask that look like a sibling instance — similar size (0.25–3×
    the mask), in the mask's neighbourhood (centre within 3× its diagonal), not
    on a face — and add up to ``max_blobs`` of them to the composite mask.

    Deliberately conservative: any doubt keeps the original mask (never raises).
    Global drift/noise is immune to the gates (blobs must be compact AND
    size-matched); the caller must have already run _align_result_tile."""
    try:
        import numpy as np
        import cv2
        a = np.asarray(crop.convert("RGB"), dtype=np.int16)
        b = np.asarray(res.convert("RGB"), dtype=np.int16)
        if a.shape != b.shape:
            return mcrop
        m = np.asarray(mcrop.convert("L")) > 128
        area = int(m.sum())
        if area < 100:
            return mcrop
        # per-pixel change map, opened to kill speckle/compression noise
        diff = (np.abs(a - b).max(axis=2) > 22).astype(np.uint8)
        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        # exclude the (slightly grown) known region so its rim doesn't count
        grown = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8))
        cand = diff & (grown == 0)
        n, lbl, stats, cent = cv2.connectedComponentsWithStats(cand, connectivity=8)
        ys, xs = np.where(m)
        mcy, mcx = float(ys.mean()), float(xs.mean())
        mdiag = float(np.hypot(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1))
        # keep faces out even if a changed face-blob happens to size-match
        fbox = None
        try:
            import identity_metrics as _idm
            tmp = scratch_path(OUTPUT_DIR, "_pair_facechk.png")
            crop.save(tmp)
            fbox = _idm.face_box(str(tmp), pad=0.15)
        except Exception:
            pass
        picked = []
        for i in range(1, n):
            sz = int(stats[i, cv2.CC_STAT_AREA])
            if not (0.25 * area <= sz <= 3.0 * area):
                continue
            cx, cy = cent[i]
            if np.hypot(cy - mcy, cx - mcx) > 3.0 * mdiag:
                continue
            if fbox:
                x0, y0, x1, y1 = stats[i, 0], stats[i, 1], \
                    stats[i, 0] + stats[i, 2], stats[i, 1] + stats[i, 3]
                ix = max(0, min(x1, fbox[2]) - max(x0, fbox[0]))
                iy = max(0, min(y1, fbox[3]) - max(y0, fbox[1]))
                if ix * iy > 0.1 * sz:
                    continue
            picked.append(i)
        if not picked:
            return mcrop
        picked = sorted(picked, key=lambda i: -int(stats[i, cv2.CC_STAT_AREA]))[:max_blobs]
        add = np.isin(lbl, picked)
        # close small gaps inside the recovered instance so the composite is solid
        add = cv2.morphologyEx(add.astype(np.uint8), cv2.MORPH_CLOSE,
                               np.ones((11, 11), np.uint8)).astype(bool)
        merged = np.where(add | m, 255, np.asarray(mcrop.convert("L"))).astype(np.uint8)
        logger.info("paired-changes: extended composite mask with %d sibling blob(s) "
                    "(+%.1f%% of tile) the editor changed outside the segmentation",
                    len(picked), 100.0 * add.sum() / add.size)
        from PIL import Image as _Img
        return _Img.fromarray(merged, "L")
    except Exception as exc:
        logger.warning("paired-changes extension skipped: %s", exc)
        return mcrop


def _extend_mask_to_removal_changes(crop, res, mcrop, *, max_total_frac: float = 0.30):
    """For REMOVALS: recover the region the instruction engine ACTUALLY changed when
    the segmentation mask missed the target.

    The live "remove the shoes" failure: Florence/SAM3 could not isolate the small,
    bright-white heels, so the mask landed on ~nothing useful, yet FireRed removed
    BOTH shoes and reconstructed the floor in the tile. The mask-bounded composite
    then discarded that fix and pasted the original shoes back — the pipeline
    returned a "successful" full-res image with the shoes intact.

    Unlike ``_extend_mask_to_paired_changes`` (which requires a sibling blob SIZE-
    MATCHED to an already-good mask), this trusts the editor's own change map when
    the mask is unreliable — but HOW MUCH it trusts it depends on the mask:

      * NEAR-EMPTY mask (< 0.3% of the tile — segmentation genuinely missed the
        target): permissive. Union every compact change-blob; the editor's change
        map is the only evidence of where the target was.
      * SUBSTANTIAL mask (segmentation found something): strict. Recovered blobs
        must be size-matched (<= 3x the mask area) AND in the mask's neighbourhood
        (centre within 3x its diagonal), and the TOTAL recovered area is capped at
        4x the mask. This stops a FireRed OVER-EDIT from being composited: on the
        tights+heels photo the editor also stripped the tights and re-rendered the
        whole lower legs — faithfully compositing that produced a fake-skin seam
        band mid-thigh. With a decent shoe mask, leg-sized blobs far up the thigh
        are exactly what must be rejected.

    Both modes still: require the caller to have run ``_align_result_tile`` (global
    drift removed); open the diff to kill speckle/compression residue; drop any blob
    overlapping a detected face; and cap the TOTAL change map at ``max_total_frac``
    of the tile (a near-whole-tile change = global re-render, keep the original mask).

    Conservative on failure: any error / over-cap keeps the original mask (never raises).
    """
    try:
        import numpy as np
        import cv2
        from PIL import Image as _Img
        a = np.asarray(crop.convert("RGB"), dtype=np.int16)
        b = np.asarray(res.convert("RGB"), dtype=np.int16)
        if a.shape != b.shape:
            return mcrop
        m0 = np.asarray(mcrop.convert("L"))
        m = m0 > 128
        tile_area = m.size
        # per-pixel change map, opened to remove speckle / alignment residue
        diff = (np.abs(a - b).max(axis=2) > 22).astype(np.uint8)
        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        grown = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8))
        cand = diff & (grown == 0)
        if cand.sum() == 0:
            return mcrop
        # Over-cap guard: a change map covering most of the tile means FireRed
        # re-rendered globally (drift/relight), not a localized removal — trusting it
        # would composite the whole tile and wreck identity. Keep the original mask.
        if cand.sum() > max_total_frac * tile_area:
            logger.info("removal-changes: change map covers %.0f%% of tile (> %.0f%% cap) "
                        "— treating as global drift, keeping segmentation mask",
                        100.0 * cand.sum() / tile_area, 100.0 * max_total_frac)
            return mcrop
        n, lbl, stats, cent = cv2.connectedComponentsWithStats(cand, connectivity=8)
        # keep faces out of the recovered region
        fbox = None
        try:
            import identity_metrics as _idm
            tmp = scratch_path(OUTPUT_DIR, "_rm_facechk.png")
            crop.save(tmp)
            fbox = _idm.face_box(str(tmp), pad=0.15)
        except Exception:
            pass
        # Trust level: a substantial mask means segmentation FOUND the target, so
        # recovery only picks up what the composite would narrowly miss (siblings,
        # spill) — size- and proximity-gated. A near-empty mask means segmentation
        # missed entirely and the change map is the only evidence — permissive.
        mask_area = int(m.sum())
        permissive = mask_area < 0.003 * tile_area
        if not permissive:
            ys, xs = np.where(m)
            mcy, mcx = float(ys.mean()), float(xs.mean())
            mdiag = float(np.hypot(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1))
            # Reference fill: what the editor painted INSIDE the mask (the removed
            # target's reconstruction — e.g. the floor behind the shoes). A genuine
            # sibling removal is filled with the SAME kind of content; an over-edit
            # (tights stripped -> new skin) is filled with something else entirely.
            fill_ref = b[m].reshape(-1, 3).mean(axis=0) if mask_area else None
        min_blob = max(25, int(0.0008 * tile_area))   # ignore pinprick noise
        picked = []
        for i in range(1, n):
            sz = int(stats[i, cv2.CC_STAT_AREA])
            if sz < min_blob:
                continue
            if not permissive:
                # strict mode: reject over-edit blobs (the editor changed something
                # much bigger / far away — e.g. re-rendered the legs to kill tights)
                if sz > 3.0 * mask_area:
                    continue
                cx, cy = cent[i]
                if np.hypot(cy - mcy, cx - mcx) > 3.0 * mdiag:
                    continue
                # fill-similarity: the blob's new content must look like the masked
                # removal's own fill (sibling shoe -> same floor). Different fill =
                # the editor changed something else (tights -> skin): reject.
                if fill_ref is not None:
                    blob = lbl == i
                    blob_fill = b[blob].reshape(-1, 3).mean(axis=0)
                    if float(np.abs(blob_fill - fill_ref).mean()) > 40.0:
                        continue
            if fbox:
                x0, y0 = stats[i, 0], stats[i, 1]
                x1, y1 = x0 + stats[i, 2], y0 + stats[i, 3]
                ix = max(0, min(x1, fbox[2]) - max(x0, fbox[0]))
                iy = max(0, min(y1, fbox[3]) - max(y0, fbox[1]))
                if ix * iy > 0.1 * sz:
                    continue
            picked.append(i)
        if not picked:
            return mcrop
        if not permissive:
            # strict mode: total recovered area may not dwarf what segmentation found
            picked = sorted(picked, key=lambda i: -int(stats[i, cv2.CC_STAT_AREA]))
            total, kept = 0, []
            for i in picked:
                sz = int(stats[i, cv2.CC_STAT_AREA])
                if total + sz > 4.0 * mask_area:
                    break
                kept.append(i); total += sz
            if not kept:
                return mcrop
            picked = kept
        add = np.isin(lbl, picked)
        add = cv2.morphologyEx(add.astype(np.uint8), cv2.MORPH_CLOSE,
                               np.ones((11, 11), np.uint8)).astype(bool)
        merged = np.where(add | m, 255, m0).astype(np.uint8)
        logger.info("removal-changes: extended composite mask with %d change-blob(s) "
                    "(+%.1f%% of tile, %s mode) the editor changed outside the "
                    "segmentation mask", len(picked), 100.0 * add.sum() / add.size,
                    "permissive" if permissive else "strict")
        return _Img.fromarray(merged, "L")
    except Exception as exc:
        logger.warning("removal-changes extension skipped: %s", exc)
        return mcrop


def _seam_blend_tile(crop, res, mcrop):
    """Back-compat wrapper: Poisson-only tone-level of a re-rendered tile (no harmonize/
    multi-band). Returns a tone-corrected tile or None. Prefer ``_blend_region`` for new
    callers; this remains for the face-guard's box-crop path and any external callers."""
    import numpy as np
    from PIL import Image
    try:
        c = np.asarray(crop.convert("RGB"))
        r = np.asarray(res.convert("RGB"))
        if c.shape != r.shape:
            return None
        m = (np.asarray(mcrop.convert("L")) > 24).astype(np.uint8)
        toned = _seamless_clone_arr(c, r, m)
        if toned is None:
            return None
        logger.info("seam-blend: Poisson tone-levelled the region tile to its surroundings")
        return Image.fromarray(toned)
    except Exception as exc:
        logger.warning("seam-blend: failed (%s) — using plain tile", exc)
        return None


def crop_to_mask(image_path: str, mask_path: str, *, pad: float = 0.12) -> Optional[str]:
    """Crop ``image_path`` to the bounding box of a hand-drawn source mask (white =
    keep), padded by ``pad``. Used by the Transfer tab's SOURCE-region masking: the
    user circles the hat/logo in a busy reference and only that region is fed to the
    extractor, so the right thing is transferred. Returns the crop path or None."""
    try:
        from PIL import Image
    except Exception:
        return None
    if not (image_path and os.path.exists(image_path) and mask_path and os.path.exists(mask_path)):
        return None
    try:
        img = Image.open(image_path).convert("RGB")
        m = Image.open(mask_path).convert("L")
        if m.size != img.size:
            m = m.resize(img.size, Image.NEAREST)
        bbox = m.point(lambda p: 255 if p > 24 else 0).getbbox()
        if not bbox:
            logger.warning("crop_to_mask: empty source mask %s", os.path.basename(mask_path))
            return None
        x0, y0, x1, y1 = bbox
        bw, bh = x1 - x0, y1 - y0
        px_, py_ = int(bw * pad) + 4, int(bh * pad) + 4
        sw, sh = img.size
        x0 = max(0, x0 - px_); y0 = max(0, y0 - py_)
        x1 = min(sw, x1 + px_); y1 = min(sh, y1 + py_)
        crop = img.crop((x0, y0, x1, y1))
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = scratch_path(OUTPUT_DIR, f"_srcregion_{int(time.time()*1000)}.png")
        crop.save(out)
        logger.info("crop_to_mask: %s -> source region %dx%d at %s",
                    os.path.basename(image_path), x1 - x0, y1 - y0, os.path.basename(str(out)))
        return str(out)
    except Exception as exc:
        logger.warning("crop_to_mask failed: %s", exc)
        return None


def _florence_mask_file(ctx, uploaded: str, region_phrase: str,
                        grow: int, seed: int, timeout: int) -> Optional[str]:
    """Run Florence-2 referring-segmentation for ``region_phrase`` and save the
    grown mask as a white-on-black PNG at the SOURCE resolution; return its path.

    Used by the crop-based contained edit so the mask bbox can be read in Python
    (to crop the tile) without holding the whole inpaint graph in VRAM.
    """
    seed = max(1, int(seed or 1))  # Florence2Run seed input requires min=1
    wf = {
        "L": {"inputs": {"image": uploaded}, "class_type": "LoadImage"},
        "F": {"inputs": {"model": "Florence-2-large", "precision": "fp16"},
              "class_type": "Florence2ModelLoader"},
        "FR": {"inputs": {"image": ["L", 0], "florence2_model": ["F", 0],
                          "text_input": region_phrase,
                          "task": "referring_expression_segmentation",
                          "fill_mask": True, "keep_model_loaded": False,
                          "max_new_tokens": 4096, "num_beams": 3, "do_sample": False,
                          "output_mask_select": "", "seed": seed},
               "class_type": "Florence2Run"},
        "G": {"inputs": {"mask": ["FR", 1], "expand": grow, "tapered_corners": True},
              "class_type": "GrowMask"},
        "MI": {"inputs": {"mask": ["G", 0]}, "class_type": "MaskToImage"},
        "9": {"inputs": {"filename_prefix": "florence-mask", "images": ["MI", 0]},
              "class_type": "SaveImage"},
    }
    return _submit_and_poll(ctx, wf, timeout=timeout, label=f"florence-mask {region_phrase[:30]!r}")


# SAM3 native checkpoint (text-promptable, pixel-dense masks — no polygon
# serialization, so no straight-chord cut-off the way Florence's polygon output
# truncates). Wiring mirrors ComfyUI's "Image Segmentation (SAM3)" blueprint.
SAM3_CKPT = "sam3.1_multiplex_fp16.safetensors"


def _sam3_mask_file(ctx, uploaded: str, region_phrase: str, grow: int,
                    threshold: float = 0.5, refine: int = 2,
                    timeout: int = 1900) -> Optional[str]:
    """Text-prompted SAM3 segmentation -> grown white-on-black mask PNG.

    SAM3 returns a dense pixel mask (not a polygon), eliminating the truncated
    straight-edge artifact of Florence's referring-segmentation. Same output
    contract as ``_florence_mask_file`` so callers can swap it in directly.
    """
    wf = {
        "L": {"inputs": {"image": uploaded}, "class_type": "LoadImage"},
        "CK": {"inputs": {"ckpt_name": SAM3_CKPT}, "class_type": "CheckpointLoaderSimple"},
        "TE": {"inputs": {"text": region_phrase, "clip": ["CK", 1]},
               "class_type": "CLIPTextEncode"},
        "S": {"inputs": {"model": ["CK", 0], "image": ["L", 0],
                         "conditioning": ["TE", 0], "threshold": threshold,
                         "refine_iterations": refine, "individual_masks": False},
              "class_type": "SAM3_Detect"},
        "G": {"inputs": {"mask": ["S", 0], "expand": grow, "tapered_corners": True},
              "class_type": "GrowMask"},
        "MI": {"inputs": {"mask": ["G", 0]}, "class_type": "MaskToImage"},
        "9": {"inputs": {"filename_prefix": "sam3-mask", "images": ["MI", 0]},
              "class_type": "SaveImage"},
    }
    return _submit_and_poll(ctx, wf, timeout=timeout, label=f"sam3-mask {region_phrase[:30]!r}")


def _mask_white_frac(mask_path: str) -> Optional[float]:
    """Fraction of the frame the mask selects (white pixels / total), or None."""
    try:
        import numpy as np
        from PIL import Image
        a = np.asarray(Image.open(mask_path).convert("L")) > 24
        return float(a.mean()) if a.size else None
    except Exception:
        return None


# A small worn item (shoes, glove, watch) can never legitimately fill this much of
# the frame — a mask this large has grabbed the whole person/limb. Used only to
# TRIGGER a more selective re-segment for phrases the LLM (or the fallback) marks
# small; large-region edits (dress, background, sky) never reach this branch.
_SMALLITEM_PERSONGRAB_FRAC = 0.12


def _region_mask_file(ctx, uploaded: str, region_phrase: str, grow: int,
                      seed: int, timeout: int, threshold: float = 0.5) -> Optional[str]:
    """Best-available region mask: SAM3 (dense) first, Florence-2 as fallback.

    SAM3 produces clean pixel-accurate masks without the polygon truncation that
    gave Florence its straight cut-off edge. If SAM3 fails (model missing, empty
    detection, server error), fall back to the Florence polygon path so masking
    still works. ``threshold`` is the SAM3 detection threshold — lowering it on a
    QA retry makes the mask more inclusive (grabs more of a partially-missed object).
    """
    # Side qualifiers ("left"/"right") are a text-segmentation blind spot: SAM3/
    # Florence ground the NOUN but pick an arbitrary instance for the side (live
    # repro: 'right foot shoe' patched the wrong foot). Segment the plain noun and
    # apply the side as a deterministic geometric post-filter instead.
    side = None
    seg_phrase = region_phrase
    m_side = re.search(r"\b(left|right)\b", region_phrase, re.I)
    if m_side:
        side = m_side.group(1).lower()
        seg_phrase = re.sub(r"\s+", " ",
                            re.sub(r"\b(left|right)(?:-hand)?\b", "", region_phrase,
                                   flags=re.I)).strip(" -,") or region_phrase
        logger.info("region-mask: side qualifier %r stripped for segmentation "
                    "(%r -> %r); applied as a geometric filter", side,
                    region_phrase, seg_phrase)
    # Small-item person-grab guard: SAM3 intermittently associates a small-item
    # phrase ("shoes") with the whole-person region (live: alternating foot-mask vs
    # full-silhouette masks for the same image). A whole-person mask for a small item
    # is unusable — the downstream person-grab gate would reject it and the edit fails
    # ("removed nothing"). When the item is small AND the first mask covers an
    # impossible fraction of the frame, RE-SEGMENT at higher (more selective)
    # thresholds and keep the first non-person mask, so the shoe mask stabilizes
    # instead of flaking run-to-run. Gated on _item_attributes.small so legitimate
    # large-region edits (dress/background/sky) never enter this path.
    is_small = _item_attributes(ctx, seg_phrase).get("small") is True

    out_mask = None
    try:
        m = _sam3_mask_file(ctx, uploaded, seg_phrase, grow, threshold=threshold, timeout=timeout)
        if is_small and m and os.path.exists(m):
            frac = _mask_white_frac(m)
            if frac is not None and frac > _SMALLITEM_PERSONGRAB_FRAC:
                logger.warning("region-mask: SAM3 mask for small item %r covers %.0f%% of the "
                               "frame (person-grab) — re-segmenting more selectively",
                               seg_phrase[:40], frac * 100)
                for retry_thr in (0.7, 0.85):
                    m2 = _sam3_mask_file(ctx, uploaded, seg_phrase, grow,
                                         threshold=retry_thr, timeout=timeout)
                    f2 = _mask_white_frac(m2) if (m2 and os.path.exists(m2)) else None
                    if f2 is not None and 0.0 < f2 <= _SMALLITEM_PERSONGRAB_FRAC:
                        logger.info("region-mask: re-segment @thr=%.2f gave a %.1f%%-frame "
                                    "mask for %r — using it", retry_thr, f2 * 100, seg_phrase[:40])
                        m, threshold = m2, retry_thr
                        break
                    if ctx is not None and getattr(ctx, "is_cancelled", None) and ctx.is_cancelled():
                        break
                else:
                    logger.warning("region-mask: re-segment could not isolate small item %r "
                                   "(still person-sized) — leaving it to the person-grab gate",
                                   seg_phrase[:40])
        if m and os.path.exists(m):
            logger.info("region-mask: SAM3 mask for %r -> %s",
                        seg_phrase[:40], os.path.basename(m))
            out_mask = _complete_multi_instance_mask(
                ctx, uploaded, seg_phrase, m, grow, threshold, timeout)
        else:
            logger.warning("region-mask: SAM3 produced no mask for %r; falling back to Florence",
                           seg_phrase[:40])
    except Exception as exc:
        logger.warning("region-mask: SAM3 failed (%s); falling back to Florence", exc)
    if out_mask is None:
        out_mask = _florence_mask_file(ctx, uploaded, seg_phrase, grow, seed, timeout)
    if out_mask and side:
        out_mask = _filter_mask_by_side(out_mask, side) or out_mask
    return out_mask


def _filter_mask_by_side(mask_path: str, side: str) -> Optional[str]:
    """Keep only the mask blob(s) on the requested side when the segmentation
    found MULTIPLE instances; a single blob is returned untouched (with one
    instance the side is already decided — e.g. only one shoe left in frame).
    Convention: the side is resolved as the SUBJECT's left/right for a person
    facing the camera (mirrored: subject-right = viewer-left / smaller x); this
    matches how both users and the vision model phrase it. If the choice is
    wrong the still-visible/escalation checks catch it downstream."""
    try:
        import numpy as np
        from PIL import Image
        from scipy.ndimage import label
        img = Image.open(mask_path).convert("L")
        a = np.asarray(img) > 24
        lbl, n = label(a)
        if n < 2:
            return mask_path
        total = int(a.sum())
        sizes = np.bincount(lbl.ravel())[1:]
        sig = [i + 1 for i in range(n) if sizes[i] >= max(50, 0.05 * total)]
        if len(sig) < 2:
            return mask_path
        centers = {i: float(np.where(lbl == i)[1].mean()) for i in sig}
        # subject-right = viewer-left (smaller x); subject-left = larger x
        pick = min(centers, key=centers.get) if side == "right" \
            else max(centers, key=centers.get)
        keep = np.where(lbl == pick, np.asarray(img), 0).astype(np.uint8)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        outp = scratch_path(OUTPUT_DIR, f"_side_mask_{int(time.time() * 1000)}.png")
        Image.fromarray(keep, "L").save(outp)
        logger.info("region-mask: kept %s-side blob of %d instances (mirrored "
                    "subject convention)", side, len(sig))
        return str(outp)
    except Exception as exc:
        logger.warning("side filter skipped: %s", exc)
        return mask_path


def _significant_components(mask_l, min_frac: float = 0.08) -> Optional[int]:
    """Number of connected components each holding >= min_frac of the mask's white
    pixels. None when it can't be measured (no scipy / empty mask)."""
    try:
        import numpy as np
        from scipy.ndimage import label
        a = np.asarray(mask_l) > 24
        total = int(a.sum())
        if total == 0:
            return None
        lbl, n = label(a)
        if n == 0:
            return 0
        sizes = np.bincount(lbl.ravel())[1:]
        return int((sizes >= total * min_frac).sum())
    except Exception:
        return None


def _complete_multi_instance_mask(ctx, uploaded: str, region_phrase: str,
                                  mask_path: str, grow: int, threshold: float,
                                  timeout: int) -> str:
    """For inherently PAIRED/PLURAL regions (LLM attr "multi": shoes, gloves,
    earrings, "the flowers"), a confident SAM3 pass routinely detects only ONE
    instance — the other scores just under the threshold. The tile render then
    fixes both, but the composite keeps only the masked instance ("removed the
    left shoe, the right one stayed white"). When the mask looks single-instance,
    union it with a more inclusive pass; the union is accepted only if it stays
    in the same size class (<=4x coverage, +10% frame absolute) so a low-threshold
    person/backdrop grab can never replace a good mask. Fail-open: any error or
    unknown attr returns the original mask untouched."""
    try:
        if threshold < 0.45:   # already an inclusive/QA-retry pass
            return mask_path
        if _item_attributes(ctx, region_phrase).get("multi") is not True:
            return mask_path
        from PIL import Image
        import numpy as np
        m1 = Image.open(mask_path).convert("L")
        n_sig = _significant_components(m1)
        if n_sig is None or n_sig >= 2:
            return mask_path
        m2p = _sam3_mask_file(ctx, uploaded, region_phrase, grow,
                              threshold=0.3, timeout=timeout)
        if not m2p or not os.path.exists(m2p):
            return mask_path
        m2 = Image.open(m2p).convert("L")
        if m2.size != m1.size:
            m2 = m2.resize(m1.size, Image.NEAREST)
        a1 = np.asarray(m1); a2 = np.asarray(m2)
        cov1 = float((a1 > 24).mean()); cov2 = float((a2 > 24).mean())
        if cov2 <= cov1 * 1.15:
            return mask_path            # nothing new found
        if cov2 > cov1 * 4 + 0.10:
            logger.info("multi-mask: inclusive pass ballooned (%.1f%% -> %.1f%%) — "
                        "keeping the confident mask", cov1 * 100, cov2 * 100)
            return mask_path
        union = Image.fromarray(np.maximum(a1, a2), "L")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = scratch_path(OUTPUT_DIR, f"_multi_mask_{int(time.time() * 1000)}.png")
        union.save(out)
        logger.info("multi-mask: %r looked single-instance — unioned inclusive pass "
                    "(coverage %.1f%% -> %.1f%%)", region_phrase[:40],
                    cov1 * 100, cov2 * 100)
        return str(out)
    except Exception as exc:
        logger.warning("multi-mask completion skipped for %r: %s", region_phrase[:40], exc)
        return mask_path

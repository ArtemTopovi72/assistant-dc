"""The FireRed contained edit, and the region mask it is built on.

Split out of image_contained.py. Two functions that only call each other:
_contained_region_mask turns a region phrase into a mask for the region and
nothing else, and edit_region_contained_via_firered runs the mask-free FireRed
edit inside that containment and composites the result back.

SEAM NOTE -- the same rule as image_contained. Everything from image.py is
reached through `_image.<name>`, which defers to call time and resolves the
CURRENT binding, so a runtime patch of `image.<name>` is honoured here. Binding
those names by value would give this module a private copy and the stub would
die silently while the suite still printed PASS. `image_maskqa` is imported as a
module for the same reason: MASK_QA_ENABLED is read live, never by value.

DEFAULT_EDIT_ENGINE is the one exception, as before: it is a default argument
value, so it must be bound at def time. It comes from image_engines, its
DEFINING module, not from image.py.

Both names are re-exported from image_contained, which is where image.py imports
them from.
"""
from config import scratch_path
import logging
import os
import random
import time
from typing import Optional

from config import COMFY_URL, OUTPUT_DIR
from image_engines import DEFAULT_EDIT_ENGINE
import image_maskqa                # MASK_QA_ENABLED is read live, never by value

from image_contained_graphs import _image

logger = logging.getLogger("assistant.image")


def _contained_region_mask(ctx, image_path: str, region_phrase: str, *,
                           grow: int, seed: int, timeout: int, pad: float = 0.25,
                           mask_override: Optional[str] = None,
                           protect_face: bool = False,
                           stage1_qa: bool = True,
                           _qa_threshold: float = 0.5, _qa_retry: bool = False):
    """Shared mask+crop-box computation for the contained editors.

    Returns (orig_RGB, full_mask_L, (x0,y0,x1,y1)) at SOURCE resolution, with the
    face box subtracted for non-facial regions, or None if no mask. Florence runs
    on a 1536 proxy (located, not detailed) and the mask is upscaled to source.

    When ``mask_override`` is a path to a hand-drawn mask (white = edit here), it is
    used VERBATIM — Florence is skipped and the face box is NOT subtracted (the user
    drew exactly what they want, e.g. a tattoo ON the face). This is the in-tab
    manual-mask path; it makes the pipeline work even when text localization fails.
    """
    from PIL import Image, ImageDraw
    try:
        orig = Image.open(image_path).convert("RGB")
        sw, sh = orig.size
    except Exception as exc:
        logger.error("contained-mask: cannot open source: %s", exc)
        return None

    # ---- manual mask path: trust the user's drawing, skip Florence + face guard --
    if mask_override and os.path.exists(mask_override):
        try:
            mask = Image.open(mask_override).convert("L")
            if mask.size != (sw, sh):
                mask = mask.resize((sw, sh), Image.NEAREST)
        except Exception as exc:
            logger.error("contained-mask: cannot open manual mask: %s", exc)
            return None
        bbox = mask.point(lambda p: 255 if p > 24 else 0).getbbox()
        if not bbox:
            logger.warning("contained-mask: manual mask is empty")
            return None
        # Optional face protection for NON-facial manual masks (e.g. a hat mask that
        # dips onto the forehead): subtract the face box so the identity is not
        # re-rendered. Off by default so deliberate face tattoos still work.
        if protect_face:
            try:
                import identity_metrics as _idm
                fb = _idm.face_box(image_path, pad=0.10)
                if fb:
                    ImageDraw.Draw(mask).rectangle(fb, fill=0)
                    logger.info("contained-mask: protected face box %s from MANUAL mask", fb)
            except Exception as exc:
                logger.warning("contained-mask: manual-mask face protection skipped: %s", exc)
        bbox = mask.point(lambda p: 255 if p > 24 else 0).getbbox()
        if not bbox:
            logger.warning("contained-mask: manual mask empty after face protection")
            return None
        x0, y0, x1, y1 = bbox
        bw, bh = x1 - x0, y1 - y0
        px_, py_ = int(bw * pad) + 8, int(bh * pad) + 8
        x0 = max(0, x0 - px_); y0 = max(0, y0 - py_)
        x1 = min(sw, x1 + px_); y1 = min(sh, y1 + py_)
        if (x1 - x0) < 16 or (y1 - y0) < 16:
            logger.warning("contained-mask: degenerate manual crop")
            return None
        logger.info("contained-mask: using MANUAL mask %s, bbox=%s (Florence skipped, protect_face=%s)",
                    os.path.basename(mask_override), (x0, y0, x1, y1), protect_face)
        return orig, mask, (x0, y0, x1, y1)

    # ---- whole-face preset: localize with the face DETECTOR, not text segmentation.
    # SAM3/Florence on the word "face" tends to return a fragment; YuNet gives the
    # actual face. Sub-features and all non-face regions fall through to segmentation.
    mask = None
    seg_from_text = False   # only text-prompted SAM3/Florence masks get Stage-1 QA
    if _image._is_whole_face(region_phrase):
        mask = _image._face_detector_mask(image_path, (sw, sh), pad=0.18)
        if mask is not None:
            logger.info("contained-mask: FACE region via YuNet detector "
                        "(text segmentation bypassed) for %r", region_phrase)
        else:
            logger.info("contained-mask: no face detected for %r; "
                        "falling back to text segmentation", region_phrase)

    _face_frac = None
    if mask is None:
        pscale = min(1.0, 1536.0 / max(sw, sh))
        if pscale < 1.0:
            proxy = orig.resize((max(16, int(sw * pscale)), max(16, int(sh * pscale))), Image.LANCZOS)
            proxy_path = scratch_path(OUTPUT_DIR, f"_florence_proxy_{int(time.time() * 1000)}.png")
            try:
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                proxy.save(proxy_path)
            except Exception as exc:
                logger.error("contained-mask: cannot write proxy: %s", exc)
                return None
            mask_src = str(proxy_path)
        else:
            mask_src = image_path

        uploaded = _image._upload_image_to_comfy(mask_src, COMFY_URL)
        if not uploaded:
            return None
        mask_path = _image._region_mask_file(ctx, uploaded, region_phrase, grow, seed, timeout,
                                      threshold=_qa_threshold)
        # Variant retry ladder: the segmenter often misses a decorated or niche
        # phrase ('blue denim shirt', 'spectacles') but finds the plain head noun
        # or a canonical synonym. Bounded to len<=3 variants, skipped on QA
        # retries (those re-run the SAME phrase at a different threshold).
        if (not mask_path or not os.path.exists(mask_path)) and not _qa_retry:
            for variant in _image._region_phrase_variants(region_phrase):
                if ctx is not None and getattr(ctx, "is_cancelled", None) and ctx.is_cancelled():
                    return None
                logger.info("contained-mask: no mask for %r — retrying with variant %r",
                            region_phrase, variant)
                mask_path = _image._region_mask_file(ctx, uploaded, variant, grow, seed, timeout,
                                              threshold=_qa_threshold)
                if mask_path and os.path.exists(mask_path):
                    region_phrase = variant   # downstream QA/logs use the phrase that worked
                    break
        if not mask_path or not os.path.exists(mask_path):
            logger.warning("contained-mask: segmentation produced no mask for %r "
                           "(variants exhausted)", region_phrase)
            return None
        seg_from_text = True   # text-prompted segmentation → eligible for Stage-1 QA
        try:
            mask = Image.open(mask_path).convert("L")
            if mask.size != (sw, sh):
                mask = mask.resize((sw, sh), Image.NEAREST)
        except Exception as exc:
            logger.error("contained-mask: cannot open mask: %s", exc)
            return None
        # Feather budget must lie OUTSIDE the object: grow the tight segmentation
        # silhouette by a region-proportional margin so a replace-edit repaints the
        # old object's own edge/shadow instead of leaving a coloured rim around the
        # new content. Runs BEFORE face protection so the face box stays subtracted.
        try:
            mask = _image._dilate_mask_outward(mask)
        except Exception as exc:
            logger.warning("contained-mask: outward dilation skipped: %s", exc)

        if not _image._is_facial(ctx, region_phrase):
            try:
                import identity_metrics as _idm
                fb = _idm.face_box(image_path, pad=0.12)
                if fb:
                    _face_frac = _image._mask_face_fraction(mask, fb)
                    ImageDraw.Draw(mask).rectangle(fb, fill=0)
                    logger.info("contained-mask: protected face box %s from %r", fb, region_phrase)
            except Exception as exc:
                logger.warning("contained-mask: face protection skipped: %s", exc)

    bbox = mask.point(lambda p: 255 if p > 24 else 0).getbbox()
    if not bbox:
        logger.warning("contained-mask: empty mask for %r (after face protection)", region_phrase)
        return None
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    # PAIRED/PLURAL context expansion: when the region is inherently multi-instance
    # ("shoes") but the mask found only ONE instance, a tight crop excludes the
    # sibling from the tile entirely — the editor can then never fix it (live repro:
    # a 425px one-foot tile from a 3204px photo). Widen the crop so the sibling is
    # in frame; the editor edits it and _extend_mask_to_paired_changes composites it.
    if (seg_from_text and _image._item_attributes(ctx, region_phrase).get("multi") is True
            and _image._significant_components(mask) == 1):
        pad = max(pad, 1.6)
        logger.info("contained-mask: %r is multi-instance but the mask has one blob "
                    "— widening crop context (pad=%.1f) to include siblings",
                    region_phrase, pad)
    px, py = int(bw * pad) + 8, int(bh * pad) + 8
    x0 = max(0, x0 - px); y0 = max(0, y0 - py)
    x1 = min(sw, x1 + px); y1 = min(sh, y1 + py)
    if (x1 - x0) < 16 or (y1 - y0) < 16:
        logger.warning("contained-mask: degenerate crop")
        return None

    # ---- STAGE-0 QA: deterministic mask geometry (text-seg only) -------------------
    # Runs BEFORE the VLM QA and does not depend on it. Catches the scattered-speck /
    # near-empty masks the VLM stages fail open on. WRONG-shaped mask -> one retry with
    # a more inclusive SAM3 threshold, then reject (no render) so we never inpaint noise.
    if seg_from_text:
        _attrs = _image._item_attributes(ctx, region_phrase)
        ok, qreason, qstats = _image._mask_quality_ok(mask, region_phrase=region_phrase,
                                               small_item=_attrs.get("small"),
                                               big_region=_attrs.get("large"))
        # PERSON-GRAB gate (deterministic, category-free): a non-face edit whose
        # mask engulfed the FACE means the segmenter selected the whole person
        # instead of the requested object ("remove her shoes" -> whole-woman
        # silhouette -> full re-render). Independent of the LLM size prior, so it
        # still fires when LM Studio is evicted by ComfyUI (its failure window).
        if ok and _face_frac is not None and _face_frac > 0.65:
            ok, qreason = False, "grabbed_person"
            qstats = {**qstats, "face_frac": round(_face_frac, 3)}
        if not ok:
            if not _qa_retry:
                # Retry direction depends on the failure: an OVER-COVERAGE / inverted
                # mask (selected ~everything) needs a MORE selective threshold; a
                # sparse/empty one needs a MORE inclusive threshold.
                retry_thr = 0.6 if qreason in ("over_coverage", "grabbed_person") else 0.35
                logger.info("contained-mask STAGE-0 QA: mask %s for %r %s — retrying SAM3 "
                            "(threshold=%.2f)", qreason, region_phrase, qstats, retry_thr)
                return _contained_region_mask(
                    ctx, image_path, region_phrase, grow=grow + 6, seed=seed,
                    timeout=timeout, pad=pad, mask_override=mask_override,
                    protect_face=protect_face, _qa_threshold=retry_thr, _qa_retry=True)
            logger.warning("contained-mask STAGE-0 QA: mask still %s after retry for %r "
                           "%s — rejecting (no render)", qreason, region_phrase, qstats)
            _image._INPAINT_FAILURE["reason"] = ("bad_mask" if qreason in ("over_coverage", "grabbed_person")
                                          else "noisy_mask")
            return None
        logger.info("contained-mask STAGE-0 QA: ok for %r %s", region_phrase, qstats)

    # ---- STAGE-1 QA: does the cutout isolate the RIGHT object? (text-seg only) ----
    # Fail-open: a missing/unclear VLM verdict -> accept. WRONG -> retry SAM3 once with
    # a more inclusive threshold; WRONG again -> reject so we never render a bad region.
    if stage1_qa and seg_from_text and image_maskqa.MASK_QA_ENABLED:
        cut_png = _image._mask_cutout_png(orig, mask, (x0, y0, x1, y1))
        verdict = _image._vlm_qa_verdict(
            ctx, cut_png,
            f"The highlighted (non-black) pixels are what was selected to edit as "
            f"the '{region_phrase}'. Is that selection correct and complete?",
            ["CORRECT", "PARTIAL", "WRONG"])
        if verdict == "WRONG":
            if not _qa_retry:
                logger.info("contained-mask STAGE-1 QA: cutout WRONG for %r — retrying "
                            "SAM3 with lower threshold", region_phrase)
                return _contained_region_mask(
                    ctx, image_path, region_phrase, grow=grow + 6, seed=seed,
                    timeout=timeout, pad=pad, mask_override=mask_override,
                    protect_face=protect_face, _qa_threshold=0.35, _qa_retry=True)
            logger.warning("contained-mask STAGE-1 QA: cutout still WRONG after retry "
                           "for %r — rejecting (no render)", region_phrase)
            _image._INPAINT_FAILURE["reason"] = "bad_mask"   # never a whole-frame redraw after this
            return None
        if verdict:
            logger.info("contained-mask STAGE-1 QA: %s for %r", verdict, region_phrase)
    return orig, mask, (x0, y0, x1, y1)


def _refill_uncovered_background(ctx, result_path: str, old_mask, region_phrase: str,
                                 seed, timeout, tile=None, tile_xy=(0, 0), orig=None) -> Optional[str]:
    """Where the OLD region was but the NEW one is not, the background is FireRed's own
    re-render; refill that band with ObjectClear so it continues the original lines.
    None = nothing uncovered, no new-region mask, or no ObjectClear (keep the composite)."""
    try:
        import numpy as np
        from scipy import ndimage
        from PIL import Image, ImageFilter
        import image_lettering_remove as _lr
        old = np.asarray(old_mask.convert("L")) > 127
        garment = None
        if tile is not None:
            # The NEW garment's own outline, read on FireRed's tile: cut at the OLD
            # outline, a stand collar under a V-neck sweater lost its top to the
            # original skin (live 10-06, white V-neck -> navy quarter-zip).
            tile_path = str(scratch_path(OUTPUT_DIR, f"_INTERMEDIATE_tile_{int(time.time() * 1000)}.png"))
            tile.save(tile_path)
            # No VLM cutout QA here: it is a union with the checked old region, and
            # the QA's flaky WRONG dropped the collar again on the next run.
            tg = _contained_region_mask(ctx, tile_path, region_phrase, grow=0,
                                        seed=seed or 1, timeout=timeout, protect_face=False,
                                        stage1_qa=False)
            if tg:
                full = Image.new("L", old_mask.size, 0)
                full.paste(tg[1].convert("L").resize(tile.size), tile_xy)
                garment = np.asarray(full) > 127
        if garment is None:
            got = _contained_region_mask(ctx, result_path, region_phrase, grow=0,
                                         seed=seed or 1, timeout=timeout, protect_face=False,
                                         _qa_retry=True)
            if not got:
                return None
            garment = np.asarray(got[1].convert("L")) > 127
        # Logos, patches and a hand over the fabric are holes in a "clothing"
        # mask; they are the garment's, not uncovered background.
        # A patch on the sleeve edge is a notch, not a hole: close wide enough
        # to bridge it (padded so the frame edge does not erode the closing).
        garment = np.pad(garment, 24)
        garment = ndimage.binary_closing(garment, iterations=20)[24:-24, 24:-24]
        garment = ndimage.binary_fill_holes(garment)
        # A patch on the sleeve edge cuts a notch OPEN to the outside (live 10-06,
        # flag patch): no closing fills it. Row by row, everything between the
        # garment's outer edges is the garment's for the "do not refill" tests.
        span = (np.maximum.accumulate(garment, axis=1)
                & np.maximum.accumulate(garment[:, ::-1], axis=1)[:, ::-1])
        new_region = Image.fromarray(span.astype("uint8") * 255).filter(ImageFilter.MaxFilter(9))
        uncovered = old & ~(np.asarray(new_region) > 127)
        if tile is None and uncovered.sum() < 0.03 * max(1, old.sum()):
            return None
        if tile is not None:
            # The garment itself straight from FireRed's tile: the seam blend had
            # washed the new hem into the old floor (a pink haze at the bottom of
            # the dress, 2026-09-25) -- the tile's hem was crisp.
            comp = Image.open(result_path).convert("RGB")
            # The WHOLE old region from the tile (garment and the floor/wall it
            # uncovered): the tile's hem is crisp and its floor clean; only its
            # outer border is a few px off the original.
            # Plus the new garment where it spills a little PAST the old mask: cut
            # at the old border, that hem tip faded into the floor as the haze.
            spill = (np.asarray(new_region) > 127) & ndimage.binary_dilation(old, iterations=25)
            g = Image.fromarray((old | spill | garment).astype("uint8") * 255)
            x, y = tile_xy
            g_crop = g.crop((x, y, x + tile.width, y + tile.height)).filter(ImageFilter.GaussianBlur(2))
            comp.paste(tile, (x, y), g_crop)
            result_path = str(scratch_path(OUTPUT_DIR, f"_INTERMEDIATE_garment_{int(time.time() * 1000)}.png"))
            comp.save(result_path)
        # ObjectClear only on the band along the OLD border, where the tile's
        # lines step against the original. Asked for the whole uncovered area it
        # painted a red "reflection" under the new hem.
        border = old & ~ndimage.binary_erosion(old, iterations=24)
        # ...and on whatever in the uncovered floor does not look like that floor:
        # the tile can leave a pink halo or a stray shoe tip under the new hem.
        # The halo can sit inside the new-garment mask too (the detector takes it
        # for part of the hem), so look near the old region, not only in
        # `uncovered`; floor-bright pixels only, the fabric is far darker.
        px = np.asarray(Image.open(result_path).convert("RGB")).astype(np.int16)
        odd = np.zeros_like(old)
        if uncovered.sum() > 500:
            med = np.median(px[uncovered], axis=0)
            floorlike = np.abs(px.sum(axis=2) - med.sum()) < 150
            odd = (ndimage.binary_dilation(old, iterations=10) & floorlike
                   & (np.abs(px - med).sum(axis=2) > 60))
            odd = ndimage.binary_dilation(odd, iterations=8)
            # ...but not the garment's own bright details: a red monogram and a
            # flag patch read as "halo" and ObjectClear wiped them (live 10-06).
            odd &= ~ndimage.binary_erosion(span, iterations=3)
        if orig is not None:
            # What still looks like the ORIGINAL there is not uncovered background:
            # hair hanging over the old sweater stays hair. Live 10-06 (white
            # sweater -> navy hoodie): blonde strands are wall-bright, read as a
            # "halo", and ObjectClear smeared skin-coloured blobs over both shoulders.
            o = np.asarray(orig.convert("RGB").resize((px.shape[1], px.shape[0]))).astype(np.int16)
            same = ndimage.binary_dilation(np.abs(px - o).sum(axis=2) < 60, iterations=2)
            uncovered &= ~same
            odd &= ~same
        uncovered = (uncovered & border) | odd
        if tile is None and uncovered.sum() < 0.03 * max(1, old.sum()) and not odd.any():
            return None
        if uncovered.sum() < 200:
            return result_path if tile is not None else None
        m = Image.fromarray((uncovered * 255).astype("uint8")).filter(ImageFilter.MaxFilter(13))
        m = Image.fromarray((((np.asarray(m) > 127) & ~(np.asarray(new_region) > 127)) | odd)
                            .astype("uint8") * 255)
        mask_path = str(scratch_path(OUTPUT_DIR, f"_INTERMEDIATE_uncovered_mask_{int(time.time() * 1000)}.png"))
        m.save(mask_path)
        out = _lr._fill_objectclear(ctx, result_path, mask_path, 6)
        if out and odd.any():
            # ObjectClear paints the halo back as the dress's "reflection"; the floor
            # there is plain, so a classic inpaint from the surrounding floor holds.
            import cv2
            bgr = cv2.imread(out)
            q = bgr[:, :, ::-1].astype(np.int16)
            left = odd & (np.abs(q - med).sum(axis=2) > 45) & (np.abs(q.sum(axis=2) - med.sum()) < 150)
            if left.any():
                left = ndimage.binary_dilation(left, iterations=3).astype("uint8") * 255
                cv2.imwrite(out, cv2.inpaint(bgr, left, 5, cv2.INPAINT_TELEA))
        if out:
            # The fill owns its mask and nothing else: ObjectClear re-renders the
            # whole crop and smeared the flag patch on the sleeve edge beside it
            # (live 10-06).
            own = np.asarray(m) > 127
            if odd.any():
                own |= odd
            own = Image.fromarray(ndimage.binary_dilation(own, iterations=2).astype("uint8") * 255)
            res = Image.open(result_path).convert("RGB")
            res.paste(Image.open(out).convert("RGB"), (0, 0), own.filter(ImageFilter.GaussianBlur(1.5)))
            res.save(out)
        if out:
            logger.info("contained-firered: refilled %d uncovered px with ObjectClear -> %s",
                        int(uncovered.sum()), os.path.basename(out))
        return out
    except Exception:
        logger.warning("contained-firered: uncovered-background refill failed", exc_info=True)
        return None


def edit_region_contained_via_firered(
        ctx, image_path: str, region_phrase: str, instruction: str,
        *, grow: int = 12, seed: Optional[int] = None, timeout: int = 1900,
        reference_paths: Optional[list] = None,
        mask_override: Optional[str] = None,
        protect_face: bool = False,
        engine: str = DEFAULT_EDIT_ENGINE,
        vlm_qa: bool = True) -> Optional[str]:
    """Identity-preserving localized edit that uses FIRERED for content generation.

    The old model inpaint (``VAEEncodeForInpaint`` + low denoise) reliably preserves
    identity but produces a FLAT GRAY blob instead of real content (the reported
    bug). FireRed follows instructions and generates real garments/objects, but as a
    whole-frame re-render it destroys identity. This combines the strengths:

      1. Florence-2 masks the region (face box subtracted for non-face edits);
      2. the masked bbox is cropped from the full-res original;
      3. **FireRed re-renders only that small crop** toward ``instruction`` — real,
         instruction-following content, at near-native detail (the crop is small, so
         FireRed's 1 MP working band barely downscales it);
      4. only the masked region of FireRed's crop is composited back over the
         full-resolution original through a feathered mask.

    Result: real content inside the region, byte-exact original (face, body,
    background, resolution) everywhere else. Returns the saved path or None.
    """
    if not image_path or not os.path.exists(image_path):
        logger.error("firered-contained: source not found: %s", image_path)
        return None
    if not (instruction or "").strip():
        logger.error("firered-contained: need instruction")
        return None
    if not (region_phrase or "").strip() and not mask_override:
        logger.error("firered-contained: need region phrase or a manual mask")
        return None
    if seed is None or seed < 1:
        seed = random.randint(1, 999_999_999)
    region_phrase = (region_phrase or "manual region").strip()

    from PIL import Image
    got = _contained_region_mask(ctx, image_path, region_phrase,
                                 grow=grow, seed=seed, timeout=timeout,
                                 mask_override=mask_override, protect_face=protect_face)
    if not got:
        return None
    orig, mask, (x0, y0, x1, y1) = got
    sw, sh = orig.size
    cw, ch = x1 - x0, y1 - y0
    crop = orig.crop((x0, y0, x1, y1))
    mcrop = mask.crop((x0, y0, x1, y1))

    ts = int(time.time() * 1000)
    # Prefix _INTERMEDIATE_ so every delivery guard (assert_deliverable, the GUI _on_done/
    # _on_redraw_done checks) rejects this cropped working tile if it is ever returned by
    # mistake — it is the zoomed editing area, never a final result.
    tile_path = scratch_path(OUTPUT_DIR, f"_INTERMEDIATE_firered_tile_{ts}.png")
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        crop.save(tile_path)
    except Exception as exc:
        logger.error("firered-contained: cannot write tile: %s", exc)
        return None

    tile_mp = (cw * ch) / 1_000_000.0
    tile_work_mp = min(tile_mp, _image.FIRERED_MAX_MP)
    # --- Quality instrumentation (proves where region sharpness is lost) -------
    # Two ratios decide whether the EDITED REGION comes back sharp:
    #   crop_cover   = how much of the frame the crop bbox spans. Near 100% means
    #                  the "localized" crop is really the whole frame.
    #   upscale_fac  = tile_mp / work_cap. FireRed renders the crop at work_cap MP
    #                  (node 191), then it is lanczos-upscaled back to the crop
    #                  size. >1.0 means the region is rendered below native and
    #                  blown back up — the measured "dress went soft" degradation.
    crop_cover = 100.0 * (cw * ch) / float(sw * sh)
    upscale_fac = (tile_mp / tile_work_mp) if tile_work_mp else 1.0
    logger.info("EDIT PATH=contained-firered [BUILD_ID=%s] region=%r | source %dx%d (%.2f MP) "
                "| crop/tile %dx%d (%.2f MP, %.1f%% of frame) -> FireRed work %.2f MP "
                "(cap=%.1f) | region upscale x%.2f | instr=%r",
                _image.IMAGE_BUILD_ID, region_phrase[:40], sw, sh, (sw * sh / 1e6),
                cw, ch, tile_mp, crop_cover, tile_work_mp, _image.FIRERED_MAX_MP,
                upscale_fac, instruction[:60])
    if upscale_fac > 1.5:
        logger.warning("contained-firered: EDITED REGION will be UPSCALED x%.2f "
                       "(tile %.2f MP rendered at %.2f MP cap then enlarged) — the '%s' "
                       "region will lose sharpness. crop covers %.1f%% of the frame. "
                       "Raise FIRERED_MAX_MP (currently %.1f) or tighten the region mask.",
                       upscale_fac, tile_mp, tile_work_mp, region_phrase[:30],
                       crop_cover, _image.FIRERED_MAX_MP)
    fired = _image.edit_image_with_firered(ctx, str(tile_path), instruction, seed=seed,
                                    timeout=timeout, work_megapixels=tile_work_mp,
                                    save_prefix="_INTERMEDIATE_tile_firered",
                                    reference_paths=reference_paths, engine=engine)
    if not fired or not os.path.exists(fired):
        logger.warning("firered-contained: FireRed produced no tile")
        return None
    try:
        _ftw, _fth = Image.open(fired).size
        logger.info("contained-firered: FireRed tile out %dx%d (expected crop %dx%d): %s",
                    _ftw, _fth, cw, ch, os.path.basename(fired))
    except Exception:
        pass

    whole_region = False
    try:
        res = Image.open(fired).convert("RGB")
        if res.size != (cw, ch):
            res = res.resize((cw, ch), Image.LANCZOS)
        # FireRed re-renders the whole tile with no alignment guarantee; a few px
        # of drift inside the composite band doubles every edge (ghost outlines/
        # hands, "shifted background"). Realign on the unmasked pixels first.
        res = _image._align_result_tile(crop, res, mcrop)
        # No-op detection: an instruction-edit engine occasionally returns the tile
        # essentially unchanged (didn't understand the region/instruction). Committing
        # that as "edited" makes the agent report success for an edit that never
        # happened. Mean |diff| inside the mask stays well under ~2.5/255 only for a
        # re-encode of the SAME pixels — any real edit is far above it.
        try:
            import numpy as _np
            a = _np.asarray(crop, dtype=_np.int16)
            b = _np.asarray(res, dtype=_np.int16)
            m = _np.asarray(mcrop) > 128
            if int(m.sum()) >= 100:
                region_diff = float(_np.abs(a - b)[m].mean())
                if region_diff < 2.5:
                    logger.warning("firered-contained: NO-OP edit for %r (mean region "
                                   "diff %.2f) — engine returned the tile unchanged; "
                                   "rejecting so the caller can retry/fall back",
                                   region_phrase, region_diff)
                    _image._INPAINT_FAILURE["reason"] = "not_found"
                    return None
        except ImportError:
            pass
        # Paired/plural regions: if the segmenter only isolated ONE instance but
        # FireRed edited the sibling too (it understands "shoes"), extend the
        # composite mask with that change so the fix isn't discarded (SAM3 missed
        # the second shoe at EVERY threshold on the live repro).
        if _image._item_attributes(ctx, region_phrase).get("multi") is True:
            mcrop = _image._extend_mask_to_paired_changes(crop, res, mcrop)
            # Keep the full-res mask in sync: STAGE-2 QA overlays `mask` — without
            # this, the recovered sibling instance reads as an out-of-mask leak
            # and the VLM rejects the good result.
            mask.paste(mcrop, (x0, y0))
        # Content-aware composite for additive edits (tattoo/object): paste only the
        # changed pixels so surrounding skin stays byte-exact (no halo). Falls back
        # to the solid-core region feather for whole-region edits (clothing/hair).
        feather = _image._content_aware_alpha(crop, res, mcrop)
        if feather is None:
            # Whole-region edit (clothing/hair): the re-rendered tile's tone/texture can
            # differ from the surrounding original, leaving a hard CREASE no opacity
            # feather can hide. Use the full seam-suppressing compositor — colour
            # harmonization + (gated) Poisson low-freq + Laplacian multi-band — to build a
            # final crop, then paste it back through the region feather.
            final_crop = _image._blend_region(crop, res, mcrop)
            whole_region = True
            out = orig.copy()
            out.paste(final_crop, (x0, y0), _image._robust_region_alpha(mcrop, img_min=min(sw, sh)))
        else:
            out = orig.copy()
            out.paste(res, (x0, y0), feather)
        final = OUTPUT_DIR / f"contained-firered_{ts}.png"
        out.save(final)
        _ow, _oh = out.size
    except Exception as exc:
        logger.error("firered-contained: composite-back failed: %s", exc)
        return None

    # A new garment smaller than the old one uncovers background FireRed painted
    # itself, a few px off the original: panel mouldings step, the floor line
    # breaks (live 2026-09-25, black ball gown -> red mermaid dress). Refill just
    # that uncovered band from the surrounding original with ObjectClear.
    if whole_region and not instruction.lstrip().lower().startswith(("remove", "completely remove", "erase")):
        refilled = _refill_uncovered_background(ctx, str(final), mask, region_phrase, seed, timeout,
                                                tile=res, tile_xy=(x0, y0), orig=orig)
        if refilled:
            final = type(final)(refilled)
            out = Image.open(refilled).convert("RGB")

    # ---- STAGE-2 QA: did the edit land correctly inside the region? ----
    # Overlay the generated result back over the original with the mask region tinted,
    # and ask the VLM whether the edit is aligned, contained, and leak-free BEFORE we
    # commit. Fail-open (no VLM -> commit). PARTIAL/WRONG -> reject so the caller can
    # retry / fall back instead of returning a bad result.
    # vlm_qa=False: the caller measures the fill itself (lettering removal: smear
    # ratio + OCR read-back). The VLM judged the red tint, not the fill, and
    # rejected clean removals (2026-09-25 bench: every lettering fill WRONG).
    if vlm_qa and image_maskqa.MASK_QA_ENABLED:
        _tint = image_maskqa.qa_tint(orig, out, mask)
        prev_png = _image._overlay_preview_png(orig, out, mask, (x0, y0, x1, y1), tint=_tint)
        # The preview shows only the RESULT. For a removal "did the edit to the
        # lettering land" has no lettering left to look at, and the VLM answered
        # "the red area contains no text to be removed — WRONG" on a clean fill:
        # 18/18 TextEraseBench fills rejected. A removal is judged by what is
        # left there, not by what was asked about.
        if instruction.lstrip().lower().startswith(("remove", "completely remove", "erase")):
            question = (f"The {_tint}-tinted area is where the {region_phrase} was removed. "
                        f"Is that area now clean — nothing of it left — and does it blend "
                        f"naturally with its surroundings?")
        else:
            question = (f"The {_tint}-tinted area is where the '{region_phrase}' was meant to change. "
                        f"Did the edit land correctly — aligned, fully inside that area, with no "
                        f"changes leaking outside it?")
        verdict = _image._vlm_qa_verdict(ctx, prev_png, question, ["GOOD", "PARTIAL", "WRONG"])
        if verdict in ("WRONG", "PARTIAL") and whole_region:
            # Advisory only for a whole-region swap (clothing/hair): the composite is
            # pasted THROUGH the mask, so nothing can leak outside it, and the VLM
            # rejected a clean dress swap three times running (2026-09-25; red tint
            # on red, then a cyan tint over the uncovered floor it read as a leak).
            logger.info("contained-firered STAGE-2 QA: %s for %r (advisory for a "
                        "whole-region swap -- committing)", verdict, region_phrase)
        elif verdict in ("WRONG", "PARTIAL"):
            logger.warning("contained-firered STAGE-2 QA: edit %s for %r — rejecting "
                           "(not committing this result)", verdict, region_phrase)
            _image._INPAINT_FAILURE["reason"] = "qa_rejected"   # never a whole-frame redraw after this
            return None
        if verdict:
            logger.info("contained-firered STAGE-2 QA: %s for %r", verdict, region_phrase)

    if not _image._no_new_faces(image_path, str(final)):
        _image._INPAINT_FAILURE["reason"] = "bad_mask"
        return None

    logger.info("EDIT PATH=contained-firered RESULT [BUILD_ID=%s]: composite out %dx%d "
                "(source was %dx%d) | RETURNED PATH=%s",
                _image.IMAGE_BUILD_ID, _ow, _oh, sw, sh, final)
    return str(final)

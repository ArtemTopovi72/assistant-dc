"""Contained / masked region editing for the image pipeline.

Extracted from image.py. Everything that edits a REGION of an existing image
while leaving the rest of the pixels untouched: the whole-frame contained edit,
the three crop-and-inpaint graph builders (vanilla inpaint, BrushNet,
Flux-Fill), the full-res cropped path, the contained region-mask builder and
the FireRed contained edit.

SEAM NOTE -- read before adding an import here.
The mask, compositing, grounding, sizing and ComfyUI helpers used below were
plain globals in image.py before the split, and the suites stub them as
`image.<name>`. Importing them by value here would bind the same name on BOTH
sides of the split: patching `image.X` would move only half the behaviour, the
stub would silently die, and the REAL helper would run while the suite still
printed PASS. They are therefore reached through `_image.` below, which defers
to call time and resolves the CURRENT binding on image.py.

_INPAINT_FAILURE is the shared mutable failure-reason dict that image.py, tools
and the suites all read; there is a test asserting it is the SAME object across
modules. It is mutated through `_image._INPAINT_FAILURE[...]` for exactly that
reason -- a by-value import here would still share the object today, but would
silently stop doing so the moment anyone rebound it.

DEFAULT_EDIT_ENGINE is the one exception: it is a default argument value, so it
must be bound at def time. It is imported from image_engines, its DEFINING
module, not from image.py.
"""
from config import scratch_path
import logging
import os
import random
import time
from typing import Optional

from config import COMFY_URL, OUTPUT_DIR

logger = logging.getLogger("assistant.image")

# The FireRed contained edit and its region-mask builder moved to
# image_contained_firered.py. Re-exported by value: image.py imports both names
# from here, and the suites reach them as image.<name>.
import image_contained_firered as _firered

_contained_region_mask          = _firered._contained_region_mask
edit_region_contained_via_firered = _firered.edit_region_contained_via_firered

# The `_image` proxy lives in image_contained_graphs so image_contained_firered
# shares the one definition.
import image_contained_graphs as _graphs

_ImageProxy          = _graphs._ImageProxy
_image               = _graphs._image


def edit_region_contained_with_comfy(
        ctx, image_path: str, region_phrase: str, edit_prompt: str,
        *, denoise: float = 0.75, grow: int = 12, seed: Optional[int] = None,
        timeout: int = 1900, engine: str = "firered") -> Optional[str]:
    """LOCALIZED, IDENTITY-PRESERVING edit (mask -> inpaint -> composite-back).

    This is the containment fix for the "edited the hat, got a different person"
    failure. Instead of FireRed re-rendering the whole frame (which regenerates the
    face/body/background — measured identity cosine -0.03, 90% of pixels changed),
    this:
      1. Florence-2 segments ONLY ``region_phrase`` (e.g. "hat") -> pixel mask;
      2. the mask is grown/feathered;
      3. The old model inpaints that region toward ``edit_prompt`` at ``denoise`` (<1 keeps
         the original structure for "change it slightly"; ->1 for a full swap);
      4. ImageCompositeMasked pastes the result back ONLY inside the mask over the
         ORIGINAL full-resolution image.

    Therefore every pixel outside the masked region — face, body, background,
    framing, resolution — is byte-for-byte the original. Identity is preserved by
    construction, not by prompt pleading. Returns the saved path, or None (caller
    decides whether to fall back to a whole-frame edit).
    """
    if not image_path or not os.path.exists(image_path):
        logger.error("contained-edit: source image not found: %s", image_path)
        return None
    if not (region_phrase or "").strip() or not (edit_prompt or "").strip():
        logger.error("contained-edit: need both a region phrase and an edit prompt")
        return None
    if seed is None or seed < 1:
        seed = random.randint(1, 999_999_999)
    region_phrase = region_phrase.strip()
    denoise = max(0.2, min(1.0, float(denoise)))

    # The crop-based path is now PREFERRED AT EVERY RESOLUTION: it composites the
    # result back with the resolution-adaptive overlay (_robust_region_alpha), so the
    # blend is seam-free from 1 MP to 20 MP. The in-graph path below pastes with a
    # HARD ImageCompositeMasked edge (the "Paint cut-out" seam) and full-frame VAE-
    # encode also risks OOM on the 12 GB card at high res, so it is only a fallback.
    dims0 = _image._source_dims(image_path)
    if dims0:
        logger.info("contained-edit: %dx%d (%0.1f MP) -> crop-based path (adaptive overlay)",
                    dims0[0], dims0[1], dims0[0] * dims0[1] / 1e6)
        cropped = edit_region_contained_cropped(
            ctx, image_path, region_phrase, edit_prompt,
            denoise=denoise, grow=grow, seed=seed, timeout=timeout, engine=engine)
        if cropped:
            return cropped
        logger.warning("contained-edit: crop-based path failed")
    # The full-res in-graph path that used to follow ran an old model inpaint; that
    # checkpoint was removed from the product (FireRed is the only edit engine),
    # so there is nothing left to fall back to here.
    return None


# Above this pixel count the full-frame VAE-encode + the old model diffusion risks OOM
# on the 12 GB 3060, so a localized edit is routed to the crop-based path instead
# (which inpaints only the masked tile, never the whole canvas).
CONTAINED_FULLRES_MP = 4_000_000


def edit_region_contained_cropped(
        ctx, image_path: str, region_phrase: str, edit_prompt: str,
        *, denoise: float = 0.7, grow: int = 12, seed: Optional[int] = None,
        timeout: int = 1900, work_cap: int = 1280, engine: str = "firered",
        protect_face: bool = True, removal: bool = False,
        _step_retry: bool = False) -> Optional[str]:
    """High-resolution localized edit that never downscales the whole image.

    1. Florence-2 segments ``region_phrase`` -> mask at the SOURCE resolution.
    2. The mask bbox is read in Python, padded, and the ORIGINAL is cropped to that
       tile (a hat on a 15 MP portrait is a tiny fraction of the canvas).
    3. The crop (capped to ``work_cap`` on its longest side, only if it is itself
       large) is edited toward ``edit_prompt`` and composited back inside the
       crop's mask.
    4. The result tile is pasted back into the FULL-RESOLUTION original through a
       feathered mask, so every pixel outside the region is byte-for-byte the
       original and the output keeps the native canvas size.

    ``engine`` is kept for call-site compatibility and is always FireRed: it
    instruction-edits the whole tile, and the feathered composite-back limits
    writes to the masked region. The masked-fill engines that used to be
    selectable here (the old model, BrushNet, FLUX.1-Fill) were removed from the
    product -- FireRed is the only edit engine.

    Returns the saved full-resolution path, or None (caller falls back).
    """
    if not image_path or not os.path.exists(image_path):
        logger.error("cropped-contained: source not found: %s", image_path)
        return None
    if not (region_phrase or "").strip() or not (edit_prompt or "").strip():
        logger.error("cropped-contained: need region phrase and edit prompt")
        return None
    if seed is None or seed < 1:
        seed = random.randint(1, 999_999_999)
    region_phrase = region_phrase.strip()
    edit_prompt = edit_prompt.strip()
    denoise = max(0.2, min(1.0, float(denoise)))

    try:
        from PIL import Image
    except Exception as exc:
        logger.error("cropped-contained: PIL unavailable: %s", exc)
        return None

    try:
        orig = Image.open(image_path).convert("RGB")
        sw, sh = orig.size
    except Exception as exc:
        logger.error("cropped-contained: cannot open source: %s", exc)
        return None

    # Florence-2 only needs to LOCATE the region — run it on a downscaled proxy so a
    # 15 MP source doesn't choke segmentation / blow VRAM (especially with the LLM
    # co-resident). The proxy mask is upscaled to the source size for bbox math; the
    # inpaint still works on the full-resolution crop, so detail is unaffected.
    mask_seed = seed
    proxy_cap = 1536
    pscale = min(1.0, proxy_cap / max(sw, sh))
    if pscale < 1.0:
        pw, ph = max(16, int(sw * pscale)), max(16, int(sh * pscale))
        proxy = orig.resize((pw, ph), Image.LANCZOS)
        proxy_path = scratch_path(OUTPUT_DIR, f"_florence_proxy_{int(time.time() * 1000)}.png")
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            proxy.save(proxy_path)
        except Exception as exc:
            logger.error("cropped-contained: cannot write Florence proxy: %s", exc)
            return None
        mask_src = str(proxy_path)
    else:
        mask_src = image_path

    uploaded = _image._upload_image_to_comfy(mask_src, COMFY_URL)
    if not uploaded:
        logger.error("cropped-contained: upload failed")
        return None

    mask_path = _image._region_mask_file(ctx, uploaded, region_phrase, grow, mask_seed, timeout)
    if not mask_path or not os.path.exists(mask_path):
        logger.warning("cropped-contained: Florence produced no mask for %r", region_phrase)
        return None

    try:
        mask = Image.open(mask_path).convert("L")
        if mask.size != (sw, sh):
            mask = mask.resize((sw, sh), Image.NEAREST)
    except Exception as exc:
        logger.error("cropped-contained: cannot open mask: %s", exc)
        return None

    # Grow the tight segmentation silhouette OUTWARD by a region-proportional
    # margin (before face protection) — a masked-fill engine repaints only inside
    # this mask, so the old object's edge must be inside it or it survives as a
    # coloured rim around the new content.
    try:
        mask = _image._dilate_mask_outward(mask)
    except Exception as exc:
        logger.warning("cropped-contained: outward dilation skipped: %s", exc)

    # Face protection: a NON-face edit (hair, hat, clothing, background) must never
    # alter facial pixels. Florence masks for face-adjacent nouns (esp. "hair")
    # routinely bleed onto the forehead/eyes; subtract the detected face box from the
    # mask so identity is preserved BY CONSTRUCTION, not just by a post-hoc gate.
    _face_frac = None
    if protect_face and not _image._is_facial(ctx, region_phrase):
        try:
            import identity_metrics as _idm
            from PIL import ImageDraw
            fb = _idm.face_box(image_path, pad=0.12)
            if fb:
                _face_frac = _image._mask_face_fraction(mask, fb)
                draw = ImageDraw.Draw(mask)
                draw.rectangle(fb, fill=0)
                logger.info("cropped-contained: protected face box %s from %r edit",
                            fb, region_phrase)
        except Exception as exc:
            logger.warning("cropped-contained: face protection skipped: %s", exc)
    # PERSON-GRAB gate (deterministic, category-free): a non-face edit whose mask
    # engulfed the FACE means the segmenter selected the whole person, not the
    # requested object ("remove her shoes" -> whole-woman silhouette -> full
    # re-render that changed the dress and kept the shoes). Works with the LLM
    # size prior down (its exact failure window: LM Studio evicted by ComfyUI).
    if _face_frac is not None and _face_frac > 0.65:
        logger.warning("cropped-contained: mask covers %.0f%% of the face box for "
                       "non-face region %r — segmenter grabbed the person, rejecting",
                       _face_frac * 100, region_phrase)
        _image._INPAINT_FAILURE["reason"] = "bad_mask"
        return None

    # STAGE-0 QA (deterministic): reject a scattered-speck / near-empty mask before it
    # is ever rendered — the cheap geometric gate the VLM QA stages lack (they fail open
    # on a mostly-black cutout). Mirrors the guard in _contained_region_mask.
    _attrs = _image._item_attributes(ctx, region_phrase)
    _ok, _qreason, _qstats = _image._mask_quality_ok(mask, region_phrase=region_phrase,
                                              small_item=_attrs.get("small"),
                                              big_region=_attrs.get("large"))
    if not _ok:
        logger.warning("cropped-contained STAGE-0 QA: mask %s for %r %s — rejecting (no render)",
                       _qreason, region_phrase, _qstats)
        _image._INPAINT_FAILURE["reason"] = "bad_mask" if _qreason == "over_coverage" else "noisy_mask"
        return None

    bbox = mask.point(lambda p: 255 if p > 24 else 0).getbbox()
    if not bbox:
        logger.warning("cropped-contained: empty mask for %r (after face protection)", region_phrase)
        return None
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    # Pad the bbox so the inpaint has surrounding context and the feather has room.
    # Multi-instance regions with a one-blob mask get a WIDE crop so the sibling
    # instance is inside the tile (see _contained_region_mask for the rationale).
    _pad = 0.25
    if (_attrs.get("multi") is True and _image._significant_components(mask) == 1):
        _pad = 1.6
        logger.info("cropped-contained: widening crop context for multi-instance %r",
                    region_phrase)
    px, py = int(bw * _pad) + 8, int(bh * _pad) + 8
    x0 = max(0, x0 - px); y0 = max(0, y0 - py)
    x1 = min(sw, x1 + px); y1 = min(sh, y1 + py)
    cw, ch = x1 - x0, y1 - y0
    if cw < 16 or ch < 16:
        logger.warning("cropped-contained: degenerate crop %dx%d", cw, ch)
        return None

    crop = orig.crop((x0, y0, x1, y1))
    mcrop = mask.crop((x0, y0, x1, y1))

    # Only the TILE is bounded for VRAM (not the whole image). ÷16 for the latent.
    # PAD, don't stretch, to the ÷16 grid: the old independent per-axis snap
    # (int(cw*scale)//16*16) stretch-resized each axis by a DIFFERENT fraction
    # (425->416 is -2.1% on one axis) — an anisotropic distortion baked into the
    # tile and re-applied on the way back, which is exactly the "background/wall
    # pixels shifted" drift the aligner then has to fight. Now: proportional
    # resize only when the tile exceeds the cap, then replicate-pad right/bottom
    # to the grid; a tile under the cap goes through 1:1 with ZERO resampling.
    scale = min(1.0, work_cap / max(cw, ch))
    tw = max(1, int(round(cw * scale)))
    th = max(1, int(round(ch * scale)))
    iw = max(64, ((tw + 15) // 16) * 16)
    ih = max(64, ((th + 15) // 16) * 16)
    scaled_crop = crop if (tw, th) == (cw, ch) else crop.resize((tw, th), Image.LANCZOS)
    scaled_mask = mcrop if (tw, th) == (cw, ch) else mcrop.resize((tw, th), Image.LANCZOS)
    if (iw, ih) != (tw, th):
        import numpy as _np
        _c = _np.pad(_np.asarray(scaled_crop.convert("RGB")),
                     ((0, ih - th), (0, iw - tw), (0, 0)), mode="edge")
        _m = _np.pad(_np.asarray(scaled_mask.convert("L")),
                     ((0, ih - th), (0, iw - tw)), mode="constant")
        work_crop = Image.fromarray(_c, "RGB")
        work_mask = Image.fromarray(_m, "L")
    else:
        work_crop, work_mask = scaled_crop.convert("RGB"), scaled_mask.convert("L")
    logger.info("cropped-contained tile prep: crop %dx%d -> work %dx%d (content %dx%d, "
                "pad r=%d b=%d, scale=%.4f%s)", cw, ch, iw, ih, tw, th,
                iw - tw, ih - th, scale, ", native 1:1" if (tw, th) == (cw, ch) else "")

    ts = int(time.time() * 1000)
    tmpc = scratch_path(OUTPUT_DIR, f"_crop_{ts}.png")
    tmpm = scratch_path(OUTPUT_DIR, f"_cropmask_{ts}.png")
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        work_crop.save(tmpc)
        work_mask.save(tmpm)
    except Exception as exc:
        logger.error("cropped-contained: cannot write tiles: %s", exc)
        return None

    # FireRed (Qwen-Image-Edit) instruction-edits the WHOLE tile — no latent
    # mask-fill — so it can reconstruct content the mask hides (eyes under
    # sunglasses, skin under a removed object). The feathered composite-back
    # below still limits writes to the masked region, preserving identity.
    logger.info("Cropped contained edit [FireRed]: src %dx%d, crop %dx%d -> work %dx%d, region=%r",
                sw, sh, cw, ch, iw, ih, region_phrase[:40])
    # edit_prompt here is the bare result phrase from _extract_edit_target
    # ("white summer dress with floral patterns"), with none of the "match
    # the original style" guidance image_grounding._firered_instruction
    # adds for the whole-frame fallback path -- this call never goes
    # through that function. Live, 2026-09-19: change_clothes on an
    # already-stylized (anime/illustrated) photo came back with a
    # photorealistic garment, because nothing ever told FireRed the rest
    # of the tile it can see was not a photograph. Whole-frame redraws
    # were never affected (they DO carry the clause); only this
    # contained/cropped tile edit was missing it.
    style_kept_prompt = (f"{edit_prompt.rstrip('. ')}. Match the existing art "
                         "style, rendering technique and lighting of the rest "
                         "of the image exactly -- do not make it more "
                         "photorealistic than the rest of the picture.")
    res_path = _image.edit_image_with_firered(
        ctx, str(tmpc), style_kept_prompt, seed=seed, timeout=timeout,
        save_prefix="_INTERMEDIATE_tile_rm")
    if not res_path or not os.path.exists(res_path):
        logger.warning("cropped-contained: tile editor produced no tile")
        return None

    try:
        res = Image.open(res_path).convert("RGB")
        # Undo the tile prep: bring the result onto the padded work grid, crop the
        # replicate-pad off, then restore the crop size (identity when the tile went
        # through native 1:1 and the engine kept its dimensions).
        if res.size != (iw, ih):
            res = res.resize((iw, ih), Image.LANCZOS)
        if (iw, ih) != (tw, th):
            res = res.crop((0, 0, tw, th))
        if res.size != (cw, ch):
            res = res.resize((cw, ch), Image.LANCZOS)
        # Undo the instruction-engine's global tile drift so the feather band isn't
        # a superposition of original + shifted edges (double outlines/ghost hands).
        # Near-zero shift for the masked-fill engines (in-graph latent composite).
        res = _image._align_result_tile(crop, res, mcrop)
        # Paired/plural regions: include the sibling instance the editor changed
        # outside a one-instance segmentation (see _extend_mask_to_paired_changes).
        if _image._item_attributes(ctx, region_phrase).get("multi") is True:
            mcrop = _image._extend_mask_to_paired_changes(crop, res, mcrop)
        # REMOVALS: when the segmenter missed the small/low-contrast target, trust the
        # editor's own (drift-aligned, area-capped) change map so the reconstructed
        # background isn't composited out and the object pasted back (the "shoes
        # removed in the tile but still on in the result" failure). See
        # _extend_mask_to_removal_changes; no-ops safely when the mask was already good.
        if removal:
            mcrop = _image._extend_mask_to_removal_changes(crop, res, mcrop)
        # Resolution-adaptive overlay: dilates to swallow the old object's outline
        # and ramps the alpha over a band sized to the SOURCE resolution (region-
        # proportional only for small regions) — no ghost edge, no hard Paint seam,
        # and no giant smeared halo on large regions, at 1 MP or 20 MP alike.
        feather = _image._robust_region_alpha(mcrop, img_min=min(sw, sh))
        # BOUNDARY-STEP gate: if the engine re-rendered the subject at different
        # PROPORTIONS (thinner leg), the composite would stitch the new silhouette
        # onto the old one with a visible step across the blend band. Detect the
        # edge misalignment and retry ONCE with a fresh seed (generative engines
        # are stochastic); if the retry can't do better, deliver best-effort.
        step = _image._boundary_step_metric(crop, res, feather)
        step_limit = max(4.0, 0.012 * min(cw, ch))
        if step > step_limit:
            if _step_retry:
                # retry attempt also stepped: report failure so the ORIGINAL
                # attempt (whose composite the caller still holds) is used instead
                # of stacking a second, possibly worse, stepped composite.
                logger.warning("cropped-contained: retry still has a %.1fpx boundary "
                               "step (> %.1f) — rejecting retry", step, step_limit)
                return None
            logger.warning("cropped-contained: boundary step %.1fpx exceeds %.1fpx "
                           "(silhouette mismatch across the blend band) — retrying "
                           "with a fresh seed", step, step_limit)
            retry = edit_region_contained_cropped(
                ctx, image_path, region_phrase, edit_prompt, denoise=denoise,
                grow=grow, seed=(seed + 987_654_321) % 1_000_000_000 or 1,
                timeout=timeout, work_cap=work_cap, engine=engine,
                protect_face=protect_face, removal=removal, _step_retry=True)
            if retry and os.path.exists(retry):
                return retry
            logger.warning("cropped-contained: fresh-seed step retry failed — "
                           "compositing the first attempt best-effort")
        out = orig.copy()
        out.paste(res, (x0, y0), feather)
        final = OUTPUT_DIR / f"contained-cropped_{ts}.png"
        out.save(final)
    except Exception as exc:
        logger.error("cropped-contained: composite-back failed: %s", exc)
        return None

    if not _image._no_new_faces(image_path, str(final)):
        _image._INPAINT_FAILURE["reason"] = "bad_mask"
        return None

    logger.info("Cropped contained edit saved (native %dx%d): %s", sw, sh, final)
    return str(final)

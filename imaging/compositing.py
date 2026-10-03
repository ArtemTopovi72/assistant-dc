"""Seam-suppression compositing: put an edited tile back without a visible join.

Extracted from image.py, which was 7,346 lines covering everything from ComfyUI
graphs to intent routing. This part is none of that — it is pixel arithmetic
over arrays, with no ComfyUI call, no model call and no I/O. That is what makes
it worth separating first: it can be reasoned about, and tested, entirely on its
own.

The pipeline, in the order a composite runs:

    _robust_region_alpha   turn a hard region mask into a usable alpha
    _feathered_alpha       ...with a SOLID core, so the middle is never ghosted
    _align_result_tile     correct sub-pixel drift before anything is blended
    _color_harmonize       match the tile's colour statistics to its surroundings
    _seamless_clone_arr    Poisson clone for the LOW frequencies
    _laplacian_blend       pyramid blend for the HIGH frequencies
    _blend_region          drive the above, degrading gracefully at each stage

Splitting low and high frequencies is what kills the tone crease: Poisson alone
smears detail, and a straight feather alone leaves a visible step in exposure.

numpy, cv2 and PIL stay FUNCTION-LOCAL imports here, exactly as they were in
image.py. They are heavy, and this module is imported on paths that never
composite anything.
"""
from typing import Optional
import logging

logger = logging.getLogger("assistant.image")   # same channel as image.py


def _feathered_alpha(mcrop, cw: int, ch: int, *, band: Optional[int] = None):
    """Turn a hard region mask into a smooth composite alpha with a SOLID core.

    A plain GaussianBlur of a binary mask ramps 0->255 straddling the boundary, so
    a visible seam remains. The previous fix over-corrected: it eroded the mask
    inward by the FULL blur band before blurring, which left a wide rim of the
    edited content at partial alpha — the washed-out / transparent border the user
    reported. We instead:

      * erode inward by only the blur RADIUS (≈ band/2), not the whole band, and
      * blur by that same radius,

    so the alpha plateaus back to fully-opaque (255) just inside the boundary and
    only a thin transition band (≈ ``band`` px wide, straddling the seam) is
    feathered. Result: the edited region is solid/opaque in its interior — no
    washout — with just enough feather to hide the seam. ``band`` defaults to a
    narrow fraction of the tile (much tighter than before).
    """
    from PIL import ImageFilter
    if band is None:
        # Tight feather: a small fraction of the tile, capped so big tiles don't
        # get a huge translucent rim. Min 4 so tiny tiles still blend a little.
        band = max(4, min(min(cw, ch) // 40, 24))
    radius = max(2, band // 2)
    # Erode inward by only the blur radius so the blurred ramp's inner edge climbs
    # back to 255 right at the original boundary (solid core preserved), while the
    # outer half of the ramp falls to 0 just outside it (seam hidden).
    erode_win = _odd(min(max(3, radius), 31))
    eroded = mcrop.filter(ImageFilter.MinFilter(erode_win))
    return eroded.filter(ImageFilter.GaussianBlur(radius))


def _odd(n: int) -> int:
    """Nearest odd int >= 3 (MinFilter/MaxFilter need an odd window size)."""
    n = max(3, int(n))
    return n if n % 2 == 1 else n + 1


def _align_result_tile(crop, res, mcrop, *, min_shift: float = 0.4,
                       max_shift_frac: float = 0.04):
    """Undo the global drift FireRed/Qwen-Edit introduces in a re-rendered tile
    before composite-back.

    The instruction engines re-render the WHOLE tile (rendered at a downscaled
    MP cap, then lanczos-resized back), with no pixel-alignment guarantee — the
    output routinely lands 1–6 px off the original, and the resize round-trip can
    add a slight SCALE error (centre aligned, edges displaced — the "wall pixels
    shifted" artifact: the wall/floor seam in the feather band lands at a
    different height than the original). The composite feather band intentionally
    lies over those re-rendered surroundings, so unaligned drift puts BOTH copies
    of every edge into the band: doubled outlines, ghost hands, shifted seams.

    Two-stage estimate over the UNMASKED pixels (the area that should be
    identical): (1) phase correlation for the translation; (2) ECC refinement to
    a full affine (catches the scale/rotation residue translation can't). The
    affine is accepted only if it is near-identity (scale within 3%, shear within
    2%, translation within limit) AND it actually reduces the unmasked-area
    difference vs the translation-only warp — otherwise fall back to the
    translation (or the raw result). Returns ``res`` (possibly warped); never raises.
    """
    try:
        import numpy as np
        import cv2
        from PIL import Image as _Img
        og = cv2.cvtColor(np.asarray(crop.convert("RGB")), cv2.COLOR_RGB2GRAY).astype(np.float32)
        rg = cv2.cvtColor(np.asarray(res.convert("RGB")), cv2.COLOR_RGB2GRAY).astype(np.float32)
        if og.shape != rg.shape:
            return res
        keep = (np.asarray(mcrop.convert("L")) <= 24).astype(np.float32)
        if float(keep.mean()) < 0.12:
            return res          # nearly everything masked: nothing reliable to align on
        win = cv2.createHanningWindow((og.shape[1], og.shape[0]), cv2.CV_32F) * keep
        (dx, dy), response = cv2.phaseCorrelate(og * win, rg * win)
        mag = (dx * dx + dy * dy) ** 0.5
        limit = max_shift_frac * min(og.shape)
        rgb = np.asarray(res.convert("RGB"))
        keep_b = keep > 0.5

        def _unmasked_err(img_rgb):
            g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
            return float(np.abs(g - og)[keep_b].mean())

        # translation candidate (identity when the phase estimate is tiny/unreliable)
        trans_ok = min_shift <= mag <= limit and response >= 0.03
        M_tr = (np.float32([[1, 0, -dx], [0, 1, -dy]]) if trans_ok
                else np.float32([[1, 0, 0], [0, 1, 0]]))

        # ECC affine refinement, seeded with the (inverse) translation. Catches the
        # scale/rotation residue that shifts edges far from the tile centre.
        M_ecc = None
        try:
            # Seed with the INVERSE of the correction (ECC estimates og->rg forward;
            # M_tr is the rg->og correction, so its translation flips sign).
            seed = np.float32([[1, 0, -M_tr[0, 2]], [0, 1, -M_tr[1, 2]]])
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
            _cc, warp = cv2.findTransformECC(
                og, rg, seed, cv2.MOTION_AFFINE, crit,
                inputMask=(keep_b.astype(np.uint8) * 255), gaussFiltSize=5)
            # `warp` maps og->rg; applied to rg with WARP_INVERSE_MAP it aligns rg to og
            sa, sb, tx = float(warp[0, 0]), float(warp[0, 1]), float(warp[0, 2])
            sc, sd, ty = float(warp[1, 0]), float(warp[1, 1]), float(warp[1, 2])
            near_identity = (abs(sa - 1) < 0.03 and abs(sd - 1) < 0.03
                             and abs(sb) < 0.02 and abs(sc) < 0.02
                             and abs(tx) <= limit and abs(ty) <= limit)
            if near_identity:
                M_ecc = warp
        except Exception:
            M_ecc = None      # ECC didn't converge — translation fallback below

        candidates = [("raw", None, _unmasked_err(rgb))]
        if trans_ok:
            w_tr = cv2.warpAffine(rgb, M_tr, (og.shape[1], og.shape[0]),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            candidates.append(("translation", w_tr, _unmasked_err(w_tr)))
        if M_ecc is not None:
            w_ec = cv2.warpAffine(rgb, M_ecc, (og.shape[1], og.shape[0]),
                                  flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_REPLICATE)
            candidates.append(("affine", w_ec, _unmasked_err(w_ec)))
        name, best, err = min(candidates, key=lambda c: c[2])
        if best is None:
            return res
        logger.info("tile-align: %s correction chosen (unmasked err %.2f -> %.2f; "
                    "phase=(%.2f,%.2f) resp=%.3f)", name, candidates[0][2], err,
                    dx, dy, response)
        return _Img.fromarray(best, "RGB")
    except Exception as exc:
        logger.warning("tile-align skipped: %s", exc)
        return res


def _robust_region_alpha(mcrop, *, dilate_frac: float = 0.035,
                         feather_frac: float = 0.06,
                         min_feather: int = 2, max_feather: int = 600,
                         img_min: Optional[int] = None):
    """Resolution-adaptive composite alpha for pasting an edited tile back over the
    original — blends cleanly from ~1 MP to ~20 MP with NO per-size retuning.

    Two failure modes it is built to defeat:
      * GHOST OUTLINE of the previous content (the old dress edge still showing after
        a recolour): the binary mask is DILATED outward by ~``dilate_frac`` of the
        region's short side, so the old silhouette + contact shadow fall INSIDE the
        fully-opaque core and get overwritten.
      * HARD "pasted-in-Paint" SEAM: the alpha ramps 0->1 over a feather band of
        ~``feather_frac`` of the region (wide at 20 MP, tight at 1 MP), and the WHOLE
        ramp lies over re-rendered pixels (the outer dilated ring) — never over the
        untouched original — so there is no double edge and no washed-out rim.

    Both knobs are FRACTIONS OF THE REGION SIZE, measured from the mask's own bbox,
    so the same call self-scales to the image. Implemented with cv2 distance
    transforms (exact, O(N), no giant rank-filter windows that would stall at 20 MP);
    falls back to a separable-blur approximation if cv2 is missing. Returns an
    L-mode PIL alpha the size of ``mcrop``.
    """
    import numpy as np
    from PIL import Image
    m = np.asarray(mcrop.convert("L"))
    binm = (m > 24).astype(np.uint8)
    ys, xs = np.nonzero(binm)
    if xs.size == 0:
        return mcrop.convert("L")
    region = max(8, int(min(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)))
    if img_min:
        # Resolution-aware sizing (see _dilate_mask_outward): the seam-hiding ramp
        # must track the SOURCE pixel density, not the region — 6% of a whole-dress
        # region is a 100+px smeared halo of re-rendered background, while 6% of an
        # earring at 4K is a hard 5px seam. Region-proportional for small regions,
        # capped at 1.5% (feather) / 2% (dilate) of the image short side for large
        # ones, floored at 0.4% so high-res seams still get a soft transition.
        dilate = int(round(min(dilate_frac * region, 0.02 * img_min, 300)))
        dilate = max(dilate, 2, int(round(0.003 * img_min)))
        feather = int(round(min(feather_frac * region, 0.015 * img_min, max_feather)))
        feather = max(feather, min_feather, int(round(0.004 * img_min)))
    else:
        # Legacy region-only sizing (callers that don't know the source size).
        dilate = max(2, min(int(round(dilate_frac * region)), 300))
        feather = max(min_feather, min(int(round(feather_frac * region)), max_feather))
    total = dilate + feather
    try:
        import cv2
        # Dilate the mask by `total` px via the OUTSIDE distance transform (fast, no
        # huge structuring kernel): pixels within `total` of the mask become the
        # core+ring write zone.
        dist_out = cv2.distanceTransform((1 - binm).astype(np.uint8), cv2.DIST_L2, 5)
        grown = (dist_out <= float(total)).astype(np.uint8)
        # Distance INSIDE the grown zone to its own edge -> linear inward ramp. The
        # opaque plateau (alpha=1 where dist>=feather) equals mask dilated by `dilate`
        # (covers the old object); the feather-wide outer ring carries the ramp.
        dist_in = cv2.distanceTransform(grown, cv2.DIST_L2, 5)
        alpha = np.clip(dist_in / float(feather), 0.0, 1.0)
        sigma = max(0.6, feather * 0.12)             # de-band the ramp, keep plateaus
        alpha = np.clip(cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma), 0.0, 1.0)
        logger.info("robust-alpha: region=%dpx dilate=%d feather=%d (cv2 distance-transform)",
                    region, dilate, feather)
        return Image.fromarray((alpha * 255.0).astype(np.uint8), "L")
    except Exception as exc:
        logger.warning("robust-alpha: cv2 path failed (%s); separable-blur fallback", exc)
        from PIL import ImageFilter
        binimg = Image.fromarray(binm * 255, "L")
        grown = binimg.filter(ImageFilter.GaussianBlur(total)).point(lambda p: 255 if p > 36 else 0)
        core = grown.filter(ImageFilter.GaussianBlur(feather)).point(
            lambda p: 255 if p > 150 else (int(p * 1.7) if p > 30 else 0))
        return core.filter(ImageFilter.GaussianBlur(max(1, feather // 2)))

# --------------------------------------------------------------------------- #
# Seam-suppression compositor: color-harmonize -> (gated) Poisson low-freq ->
# Laplacian-pyramid multi-band high-freq.  Each stage degrades gracefully and is
# individually toggleable so the seam benchmark can A/B them.  See
# docs/seam_blending.md and tests/bench_seam.py.
# --------------------------------------------------------------------------- #


def _band_masks(m_uint8, width):
    """Inside/outside boundary rings of a binary mask, `width` px each, via cv2
    morphology. Returns (ring_in_bool, ring_out_bool). Falls back to the whole
    interior/exterior if cv2 is missing."""
    binm = (m_uint8 > 0)
    try:
        import cv2
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
        mu = binm.astype("uint8")
        eroded = cv2.erode(mu, k) > 0
        dilated = cv2.dilate(mu, k) > 0
        ring_in = binm & ~eroded
        ring_out = dilated & ~binm
        if ring_in.sum() == 0:
            ring_in = binm
        if ring_out.sum() == 0:
            ring_out = ~binm
        return ring_in, ring_out
    except Exception:
        return binm, ~binm


def _is_continuation(bg, patch, m_uint8, *, ring=6, dE_thresh=22.0):
    """Decide whether the patch is a CONTINUATION of its surroundings (skin/wall — the
    seam-crease case) versus DISTINCT CONTENT (a recoloured garment on skin).

    Compares the patch's boundary-ring colour to the original's outside-ring colour in
    LAB (CIE76 ΔE). Small ΔE -> continuation (Poisson/strong harmonize are safe and
    helpful). Large ΔE -> distinct content (skip the interior tone pull so a deliberate
    recolour is never washed toward the surroundings; rely on multi-band band-blend)."""
    import numpy as np
    try:
        import cv2
        lab_bg = cv2.cvtColor(bg, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_pt = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB).astype(np.float32)
    except Exception:
        lab_bg = bg.astype(np.float32)
        lab_pt = patch.astype(np.float32)
    ring_in, ring_out = _band_masks(m_uint8, ring)
    if ring_in.sum() == 0 or ring_out.sum() == 0:
        return True
    d = lab_pt[ring_in].mean(axis=0) - lab_bg[ring_out].mean(axis=0)
    dE = float(np.sqrt((d * d).sum()))
    return dE <= dE_thresh


def _color_harmonize(bg, patch, m_uint8, *, strength=1.0, ring=6,
                     max_shift=(12.0, 10.0, 10.0)):
    """Reinhard-style mean/std colour transfer in LAB, referenced to the ORIGINAL's
    outside boundary ring, applied to the patch BEFORE Poisson so the solver has less
    low-frequency work to do (and so any global colour cast is removed up front).

    The shift is CLIPPED per LAB channel (``max_shift``) and scaled by ``strength`` so a
    deliberate recolour (low strength) is nudged, not destroyed. Returns a recoloured
    patch array (uint8 RGB), or the patch unchanged on failure."""
    import numpy as np
    try:
        import cv2
        lab_bg = cv2.cvtColor(bg, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_pt = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB).astype(np.float32)
    except Exception:
        return patch
    ring_in, ring_out = _band_masks(m_uint8, ring)
    binm = (m_uint8 > 0)
    if ring_in.sum() < 4 or ring_out.sum() < 4 or binm.sum() == 0:
        return patch
    # Boundary-weighted falloff: the correction is full at the seam and decays to ~0 by
    # ~25% of the region's reach inward, so a DELIBERATE recolour keeps its core colour
    # (no interior wash) while the boundary tone still matches. For a true continuation,
    # the gated Poisson stage propagates the boundary correction the rest of the way in.
    try:
        import cv2
        dist_in = cv2.distanceTransform(binm.astype(np.uint8), cv2.DIST_L2, 5)
        reach = max(6.0, 0.25 * float(dist_in.max()))
        wfield = np.clip(1.0 - dist_in / reach, 0.0, 1.0)
    except Exception:
        wfield = binm.astype(np.float32)
    # CRITICAL: zero the weight OUTSIDE the region. `1 - dist_in/reach` evaluates to 1.0
    # where dist_in==0 — which includes every background pixel — so without this mask the
    # correction would bleed onto pixels outside the patch. Harmonize must only ever touch
    # the patch interior.
    wfield = wfield * binm.astype(np.float32)
    out = lab_pt.copy()
    ms = np.asarray(max_shift, np.float32)
    for c in range(3):
        s_mean, s_std = lab_pt[ring_in][:, c].mean(), lab_pt[ring_in][:, c].std() + 1e-3
        t_mean, t_std = lab_bg[ring_out][:, c].mean(), lab_bg[ring_out][:, c].std() + 1e-3
        gain = np.clip(t_std / s_std, 0.6, 1.6)
        shifted = (lab_pt[:, :, c] - s_mean) * gain + t_mean
        delta = np.clip((shifted - lab_pt[:, :, c]) * strength, -ms[c], ms[c])
        out[:, :, c] += delta * wfield
    out = np.clip(out, 0, 255).astype(np.uint8)
    try:
        import cv2
        out_rgb = cv2.cvtColor(out, cv2.COLOR_LAB2RGB)
        # Restore outside-mask pixels byte-exact: the RGB->LAB->RGB round-trip is lossy
        # (8-bit quantization), so even a zero correction would otherwise perturb the
        # background by ±1-2. Harmonize must leave everything outside the patch untouched.
        out_rgb[binm == 0] = patch[binm == 0]
        return out_rgb
    except Exception:
        return patch


def _seamless_clone_arr(bg, patch, m_uint8):
    """Array Poisson blend (NORMAL_CLONE): membrane-levels the patch's low frequencies
    to the bg at the mask boundary while preserving the patch's gradients. RGB in/out,
    or None if unavailable/unsafe. Shared by the tile compositor and the benchmark."""
    import numpy as np
    try:
        import cv2
    except Exception:
        return None
    try:
        m = (m_uint8 > 0).astype(np.uint8) * 255
        m[0, :] = m[-1, :] = 0
        m[:, 0] = m[:, -1] = 0
        ys, xs = np.nonzero(m)
        if xs.size == 0 or float(m.sum()) / (255.0 * m.size) > 0.92:
            return None
        cx = int((int(xs.min()) + int(xs.max())) / 2)
        cy = int((int(ys.min()) + int(ys.max())) / 2)
        b = np.ascontiguousarray(bg[:, :, ::-1])
        p = np.ascontiguousarray(patch[:, :, ::-1])
        out = cv2.seamlessClone(p, b, m, (cx, cy), cv2.NORMAL_CLONE)
        return np.ascontiguousarray(out[:, :, ::-1])
    except Exception as exc:
        logger.warning("seamless-clone: failed (%s)", exc)
        return None


def _laplacian_blend(bg, patch, alpha, *, levels=5):
    """Multi-band (Laplacian-pyramid) blend of ``patch`` into ``bg`` across a soft
    ``alpha`` (HxW float 0..1). Each spatial-frequency band is blended with the
    correspondingly-downsampled alpha, so LOW frequencies cross over the boundary
    gradually (no colour step) while HIGH frequencies (texture) cross over sharply (no
    ghosting) — the classic fix for the residual texture seam Poisson alone leaves.
    RGB uint8 in/out; falls back to a straight alpha lerp if cv2 is missing."""
    import numpy as np
    try:
        import cv2
    except Exception:
        a = alpha[..., None]
        return np.clip(patch * a + bg * (1 - a), 0, 255).astype(np.uint8)
    h, w = bg.shape[:2]
    max_lvl = max(1, int(np.floor(np.log2(max(2, min(h, w))))) - 1)
    # GEOMETRY CAP: a Laplacian blend mixes bg into patch over a band ~2**levels px wide.
    # If that exceeds the region's interior depth, the coarsest level pulls the DEEP
    # interior toward bg (measured: a recolour's core washed ~36% at levels=5 on a small
    # region). Cap so the blend stays within the outer third of the region — large regions
    # still get deep pyramids (smooth low-freq), small ones don't bleed inward.
    try:
        inner = (alpha > 0.5).astype(np.uint8)
        if inner.any():
            reach = float(cv2.distanceTransform(inner, cv2.DIST_L2, 5).max())
            geo_cap = int(np.floor(np.log2(max(2.0, reach / 3.0))))
            levels = min(levels, max(1, geo_cap))
    except Exception:
        pass
    levels = int(max(1, min(levels, max_lvl)))
    B = bg.astype(np.float32)
    P = patch.astype(np.float32)
    A = alpha.astype(np.float32)
    gpB, gpP, gpA = [B], [P], [A]
    for _ in range(levels):
        gpB.append(cv2.pyrDown(gpB[-1]))
        gpP.append(cv2.pyrDown(gpP[-1]))
        gpA.append(cv2.pyrDown(gpA[-1]))
    blended = gpP[-1] * gpA[-1][..., None] + gpB[-1] * (1 - gpA[-1][..., None])
    for i in range(levels - 1, -1, -1):
        size = (gpB[i].shape[1], gpB[i].shape[0])
        lapB = gpB[i] - cv2.pyrUp(gpB[i + 1], dstsize=size)
        lapP = gpP[i] - cv2.pyrUp(gpP[i + 1], dstsize=size)
        ai = gpA[i][..., None]
        lap = lapP * ai + lapB * (1 - ai)
        blended = cv2.pyrUp(blended, dstsize=size) + lap
    return np.clip(blended, 0, 255).astype(np.uint8)


def _blend_region(crop_bg, patch, mcrop, *, harmonize=True, poisson=True,
                  multiband=True, feather_alpha=None):
    """Unified seam-suppressing compositor. Returns the FINAL crop (PIL RGB, same size
    as ``crop_bg``) with ``patch`` composited into ``crop_bg`` over the white area of
    ``mcrop``.

    Pipeline (each stage individually skippable for benchmarking):
      1. color-harmonize the patch to the original's boundary ring (LAB, clipped);
      2. if the patch CONTINUES its surroundings, Poisson-level its low frequencies;
         distinct content (a recoloured garment) skips this so it isn't washed out;
      3. Laplacian-pyramid (multi-band) blend across a feathered region alpha.
    Graceful fallbacks at every step; never raises."""
    import numpy as np
    from PIL import Image
    bg = np.asarray(crop_bg.convert("RGB"))
    pt = np.asarray(patch.convert("RGB"))
    if bg.shape != pt.shape:
        pt = np.asarray(patch.convert("RGB").resize(crop_bg.size, Image.LANCZOS))
    m = (np.asarray(mcrop.convert("L")) > 24).astype(np.uint8)
    if m.sum() == 0:
        return crop_bg
    if feather_alpha is None:
        a = np.asarray(_robust_region_alpha(mcrop).convert("L")).astype(np.float32) / 255.0
    else:
        a = np.asarray(feather_alpha.convert("L")).astype(np.float32) / 255.0

    # Two regimes (benchmarked in tests/bench_seam.py — see docs/seam_blending.md):
    #   * CONTINUATION (skin/wall; patch should read as more of its surroundings): a clean
    #     Poisson membrane is the single most effective fix (SSIM 0.81->0.86, LPIPS
    #     0.26->0.18). Harmonize/multi-band only perturb the gradients Poisson preserves
    #     and slightly hurt here, so they are skipped.
    #   * DISTINCT CONTENT (a recoloured garment; interior colour must be kept): Poisson is
    #     skipped so the edit isn't washed toward the surroundings; instead a boundary-
    #     weighted colour harmonization + Laplacian multi-band blend smooth the seam while
    #     leaving the deep interior intact (recolour core drift ~7 ΔE, was ~17).
    cont = _is_continuation(bg, pt, m)
    if cont and poisson:
        toned = _seamless_clone_arr(bg, pt, m)
        if toned is not None:
            pt = toned
        af = a[..., None]
        final = np.clip(pt * af + bg * (1 - af), 0, 255).astype(np.uint8)
        mode = "continuation:poisson"
    else:
        if harmonize:
            pt = _color_harmonize(bg, pt, m, strength=0.6)
        final = _laplacian_blend(bg, pt, a) if multiband else \
            np.clip(pt * a[..., None] + bg * (1 - a[..., None]), 0, 255).astype(np.uint8)
        mode = "distinct:harmonize+multiband"
    logger.info("blend-region: %s (harmonize=%s poisson=%s multiband=%s)",
                mode, harmonize, poisson, multiband)
    return Image.fromarray(final)

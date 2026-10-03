"""Synthetic-array coverage for image.py's pure numpy/cv2/PIL blend/mask helpers --
no GPU or ComfyUI needed. Crafts specific arrays/masks/pairs to hit every branch of:
crop_to_mask, _feathered_alpha, _content_aware_alpha, _robust_region_alpha,
_band_masks, _is_continuation, _color_harmonize, _seamless_clone_arr,
_laplacian_blend, _blend_region, _seam_blend_tile, _mask_quality_ok.
Run: venv/Scripts/python.exe tests/test_image_synthetic_blend.py
"""
import os, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
import numpy as np
from pathlib import Path
from PIL import Image
import image as I
# The seam compositor moved to compositing.py. These four are internal to it
# (image.py drives _blend_region, not its stages), so they are addressed on
# their own module rather than re-exported through image just for tests.
import compositing as _CMP  # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="imgsyn_"))
RESULTS = []
import os as _os
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + detail))
    if not cond and _os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))


def _mask_img(w=128, h=128, box=None, mode="L"):
    m = Image.new("L", (w, h), 0)
    if box:
        px = m.load()
        x0, y0, x1, y1 = box
        for y in range(y0, y1):
            for x in range(x0, x1):
                px[x, y] = 255
    return m

def _rgb_img(w=128, h=128, col=(100, 120, 140)):
    return Image.new("RGB", (w, h), col)

def _rand_rgb(w=128, h=128):
    return (np.random.rand(h, w, 3) * 255).astype(np.uint8)


# ---------------- crop_to_mask ----------------

def test_crop_to_mask_missing_files():
    check("crop_missing_none", I.crop_to_mask("Z:/no.png", "Z:/no2.png") is None)

def test_crop_to_mask_empty_mask():
    img = _TMP / "cm_img.png"; msk = _TMP / "cm_mask_empty.png"
    _rgb_img(64, 64).save(img)
    _mask_img(64, 64).save(msk)  # all-black, empty bbox
    check("crop_empty_mask_none", I.crop_to_mask(str(img), str(msk)) is None)

def test_crop_to_mask_valid():
    img = _TMP / "cm_img2.png"; msk = _TMP / "cm_mask2.png"
    _rgb_img(80, 60).save(img)
    _mask_img(80, 60, box=(10, 10, 50, 40)).save(msk)
    out = I.crop_to_mask(str(img), str(msk))
    check("crop_valid_returns", out is not None and Path(out).exists())

def test_crop_to_mask_size_mismatch_resize():
    img = _TMP / "cm_img3.png"; msk = _TMP / "cm_mask3.png"
    _rgb_img(100, 100).save(img)
    _mask_img(50, 50, box=(5, 5, 30, 30)).save(msk)  # different size -> triggers resize
    out = I.crop_to_mask(str(img), str(msk))
    check("crop_resize_path", out is not None and Path(out).exists())

def test_crop_to_mask_edge_touching_bbox():
    img = _TMP / "cm_img4.png"; msk = _TMP / "cm_mask4.png"
    _rgb_img(64, 64).save(img)
    _mask_img(64, 64, box=(0, 0, 64, 64)).save(msk)  # full-frame mask, pad clamps to bounds
    out = I.crop_to_mask(str(img), str(msk))
    check("crop_full_frame", out is not None)

def test_crop_to_mask_exception_path(monkeypatch_none=None):
    # corrupt "image" file (not a real image) -> PIL raises inside try -> caught, None
    img = _TMP / "cm_bad.png"; msk = _TMP / "cm_mask5.png"
    img.write_bytes(b"not an image")
    _mask_img(64, 64, box=(1, 1, 10, 10)).save(msk)
    check("crop_corrupt_image_none", I.crop_to_mask(str(img), str(msk)) is None)


# ---------------- _feathered_alpha ----------------

def test_feathered_alpha_default_band():
    m = _mask_img(200, 200, box=(40, 40, 160, 160))
    out = I._feathered_alpha(m, 200, 200)
    check("feather_default", out.size == (200, 200))

def test_feathered_alpha_explicit_band():
    m = _mask_img(64, 64, box=(10, 10, 50, 50))
    out = I._feathered_alpha(m, 64, 64, band=8)
    check("feather_explicit_band", out.size == (64, 64))

def test_feathered_alpha_tiny_tile():
    m = _mask_img(20, 20, box=(2, 2, 18, 18))
    out = I._feathered_alpha(m, 20, 20)
    check("feather_tiny_tile", out.size == (20, 20))


# ---------------- _content_aware_alpha ----------------

def test_content_aware_alpha_empty_mask():
    crop = _rgb_img(64, 64, (50, 50, 50))
    res = _rgb_img(64, 64, (200, 50, 50))
    m = _mask_img(64, 64)  # empty
    out = I._content_aware_alpha(crop, res, m)
    check("caa_empty_mask_none", out is None)

def test_content_aware_alpha_full_region_change():
    crop = _rgb_img(64, 64, (50, 50, 50))
    res = _rgb_img(64, 64, (250, 10, 10))  # big diff everywhere
    m = _mask_img(64, 64, box=(0, 0, 64, 64))
    out = I._content_aware_alpha(crop, res, m, threshold=5, full_frac=0.6)
    check("caa_full_region_none", out is None)  # whole-region edit -> None (region feather fallback)

def test_content_aware_alpha_partial_change():
    crop = np.zeros((64, 64, 3), dtype=np.uint8) + 50
    res = crop.copy()
    res[20:30, 20:30] = 250  # small distinct patch differs
    crop_im = Image.fromarray(crop); res_im = Image.fromarray(res)
    m = _mask_img(64, 64, box=(0, 0, 64, 64))
    out = I._content_aware_alpha(crop_im, res_im, m, threshold=10, full_frac=0.6)
    check("caa_partial_returns_alpha", out is not None and out.mode == "L")


# ---------------- _robust_region_alpha ----------------

def test_robust_region_alpha_empty_mask():
    m = _mask_img(64, 64)
    out = I._robust_region_alpha(m)
    check("rra_empty_mask", out.size == (64, 64))

def test_robust_region_alpha_cv2_path():
    m = _mask_img(96, 96, box=(20, 20, 70, 70))
    out = I._robust_region_alpha(m)
    check("rra_cv2_path", out.mode == "L" and out.size == (96, 96))

def test_robust_region_alpha_cv2_missing_fallback():
    m = _mask_img(64, 64, box=(10, 10, 50, 50))
    orig_cv2 = sys.modules.get("cv2")
    sys.modules["cv2"] = None  # force ImportError on `import cv2`
    try:
        out = I._robust_region_alpha(m)
        check("rra_fallback_no_cv2", out.size == (64, 64))
    finally:
        if orig_cv2 is not None:
            sys.modules["cv2"] = orig_cv2
        else:
            sys.modules.pop("cv2", None)


# ---------------- _band_masks / _is_continuation ----------------

def test_band_masks_normal():
    m = (np.asarray(_mask_img(64, 64, box=(15, 15, 50, 50))) > 0).astype(np.uint8) * 255
    ring_in, ring_out = _CMP._band_masks(m, 6)
    check("band_masks_nonempty", ring_in.sum() > 0 and ring_out.sum() > 0)

def test_band_masks_empty():
    m = np.zeros((64, 64), dtype=np.uint8)
    ring_in, ring_out = _CMP._band_masks(m, 6)
    # whole-empty mask -> ring_in falls back to binm (all False), ring_out falls back to ~binm (all True)
    check("band_masks_empty_fallback", ring_in.sum() == 0 and ring_out.sum() == 64 * 64)

def test_band_masks_cv2_missing():
    m = (np.asarray(_mask_img(64, 64, box=(10, 10, 40, 40))) > 0).astype(np.uint8) * 255
    orig_cv2 = sys.modules.get("cv2")
    sys.modules["cv2"] = None
    try:
        ring_in, ring_out = _CMP._band_masks(m, 6)
        check("band_masks_cv2_missing_fallback", ring_in.sum() > 0)
    finally:
        if orig_cv2 is not None: sys.modules["cv2"] = orig_cv2
        else: sys.modules.pop("cv2", None)

def test_is_continuation_true_and_false():
    bg = _rand_rgb(64, 64)
    patch_similar = bg.copy()
    m = (np.asarray(_mask_img(64, 64, box=(15, 15, 45, 45))) > 0).astype(np.uint8) * 255
    check("is_continuation_similar", _CMP._is_continuation(bg, patch_similar, m) is True)
    patch_diff = np.zeros((64, 64, 3), dtype=np.uint8)
    patch_diff[:, :] = [250, 0, 0]  # solid bright red, very different from random bg
    check("is_continuation_different", _CMP._is_continuation(bg, patch_diff, m, dE_thresh=5.0) is False)

def test_is_continuation_empty_rings():
    bg = _rand_rgb(64, 64); patch = _rand_rgb(64, 64)
    m = np.zeros((64, 64), dtype=np.uint8)  # empty mask -> ring_in/out both empty via fallback
    # with the _band_masks empty-mask fallback, ring_in=binm(all False)->sum==0, so is_continuation
    # takes the "return True" early-exit branch
    check("is_continuation_empty_true", _CMP._is_continuation(bg, patch, m) is True)


# ---------------- _color_harmonize ----------------

def test_color_harmonize_normal():
    bg = _rand_rgb(80, 80)
    patch = _rand_rgb(80, 80)
    m = (np.asarray(_mask_img(80, 80, box=(15, 15, 60, 60))) > 0).astype(np.uint8) * 255
    out = _CMP._color_harmonize(bg, patch, m)
    check("harmonize_shape", out.shape == patch.shape)

def test_color_harmonize_tiny_rings_unchanged():
    bg = _rand_rgb(40, 40); patch = _rand_rgb(40, 40)
    m = (np.asarray(_mask_img(40, 40, box=(1, 1, 2, 2))) > 0).astype(np.uint8) * 255  # 1x1px -> rings <4
    out = _CMP._color_harmonize(bg, patch, m)
    check("harmonize_tiny_unchanged", np.array_equal(out, patch))

def test_color_harmonize_cv2_missing():
    bg = _rand_rgb(64, 64); patch = _rand_rgb(64, 64)
    m = (np.asarray(_mask_img(64, 64, box=(10, 10, 40, 40))) > 0).astype(np.uint8) * 255
    orig_cv2 = sys.modules.get("cv2")
    sys.modules["cv2"] = None
    try:
        out = _CMP._color_harmonize(bg, patch, m)
        check("harmonize_cv2_missing_unchanged", np.array_equal(out, patch))
    finally:
        if orig_cv2 is not None: sys.modules["cv2"] = orig_cv2
        else: sys.modules.pop("cv2", None)


# ---------------- _seamless_clone_arr ----------------

def test_seamless_clone_normal():
    bg = _rand_rgb(100, 100); patch = _rand_rgb(100, 100)
    m = (np.asarray(_mask_img(100, 100, box=(20, 20, 70, 70))) > 0).astype(np.uint8) * 255
    out = I._seamless_clone_arr(bg, patch, m)
    check("seamless_clone_returns", out is not None and out.shape == bg.shape)

def test_seamless_clone_empty_mask_none():
    bg = _rand_rgb(64, 64); patch = _rand_rgb(64, 64)
    m = np.zeros((64, 64), dtype=np.uint8)
    check("seamless_clone_empty_none", I._seamless_clone_arr(bg, patch, m) is None)

def test_seamless_clone_full_mask_none():
    bg = _rand_rgb(64, 64); patch = _rand_rgb(64, 64)
    m = np.full((64, 64), 255, dtype=np.uint8)  # >92% coverage after border-zeroing
    check("seamless_clone_full_none", I._seamless_clone_arr(bg, patch, m) is None)

def test_seamless_clone_cv2_missing_none():
    bg = _rand_rgb(64, 64); patch = _rand_rgb(64, 64)
    m = (np.asarray(_mask_img(64, 64, box=(10, 10, 40, 40))) > 0).astype(np.uint8) * 255
    orig_cv2 = sys.modules.get("cv2")
    sys.modules["cv2"] = None
    try:
        check("seamless_clone_no_cv2_none", I._seamless_clone_arr(bg, patch, m) is None)
    finally:
        if orig_cv2 is not None: sys.modules["cv2"] = orig_cv2
        else: sys.modules.pop("cv2", None)


# ---------------- _laplacian_blend ----------------

def test_laplacian_blend_normal():
    bg = _rand_rgb(128, 128); patch = _rand_rgb(128, 128)
    alpha = (np.asarray(_mask_img(128, 128, box=(20, 20, 100, 100))).astype(np.float32)) / 255.0
    out = _CMP._laplacian_blend(bg, patch, alpha)
    check("laplacian_normal_shape", out.shape == bg.shape)

def test_laplacian_blend_small_region_geo_cap():
    bg = _rand_rgb(64, 64); patch = _rand_rgb(64, 64)
    alpha = (np.asarray(_mask_img(64, 64, box=(28, 28, 36, 36))).astype(np.float32)) / 255.0
    out = _CMP._laplacian_blend(bg, patch, alpha, levels=5)
    check("laplacian_geo_cap_shape", out.shape == bg.shape)

def test_laplacian_blend_cv2_missing_fallback():
    bg = _rand_rgb(64, 64); patch = _rand_rgb(64, 64)
    alpha = (np.asarray(_mask_img(64, 64, box=(10, 10, 40, 40))).astype(np.float32)) / 255.0
    orig_cv2 = sys.modules.get("cv2")
    sys.modules["cv2"] = None
    try:
        out = _CMP._laplacian_blend(bg, patch, alpha)
        check("laplacian_fallback_shape", out.shape == bg.shape)
    finally:
        if orig_cv2 is not None: sys.modules["cv2"] = orig_cv2
        else: sys.modules.pop("cv2", None)


# ---------------- _blend_region ----------------

def test_blend_region_empty_mask_returns_bg():
    bg = _rgb_img(64, 64, (30, 30, 30))
    patch = _rgb_img(64, 64, (200, 30, 30))
    m = _mask_img(64, 64)  # empty
    out = I._blend_region(bg, patch, m)
    check("blend_region_empty_mask_is_bg", list(out.getdata())[:5] == list(bg.getdata())[:5])

def test_blend_region_continuation_path():
    arr = _rand_rgb(96, 96)
    bg = Image.fromarray(arr)
    patch = Image.fromarray(arr.copy())  # identical -> strong continuation
    m = _mask_img(96, 96, box=(20, 20, 70, 70))
    out = I._blend_region(bg, patch, m)
    check("blend_region_continuation", out.size == (96, 96))

def test_blend_region_distinct_content_path():
    bg = Image.fromarray(_rand_rgb(96, 96))
    patch_arr = np.zeros((96, 96, 3), dtype=np.uint8); patch_arr[:, :] = [250, 5, 5]
    patch = Image.fromarray(patch_arr)  # very different color -> distinct content
    m = _mask_img(96, 96, box=(20, 20, 70, 70))
    out = I._blend_region(bg, patch, m, poisson=True, harmonize=True, multiband=True)
    check("blend_region_distinct", out.size == (96, 96))

def test_blend_region_size_mismatch_resize():
    bg = Image.fromarray(_rand_rgb(96, 96))
    patch = Image.fromarray(_rand_rgb(60, 60))  # different size -> triggers resize branch
    m = _mask_img(96, 96, box=(10, 10, 50, 50))
    out = I._blend_region(bg, patch, m)
    check("blend_region_resize", out.size == (96, 96))

def test_blend_region_explicit_alpha():
    bg = Image.fromarray(_rand_rgb(64, 64))
    patch = Image.fromarray(_rand_rgb(64, 64))
    m = _mask_img(64, 64, box=(10, 10, 50, 50))
    alpha = _mask_img(64, 64, box=(15, 15, 45, 45))
    out = I._blend_region(bg, patch, m, feather_alpha=alpha)
    check("blend_region_explicit_alpha", out.size == (64, 64))


# ---------------- _seam_blend_tile ----------------

def test_seam_blend_tile_normal():
    crop = Image.fromarray(_rand_rgb(80, 80))
    res = Image.fromarray(_rand_rgb(80, 80))
    m = _mask_img(80, 80, box=(15, 15, 60, 60))
    out = I._seam_blend_tile(crop, res, m)
    check("seam_blend_tile_ran", True)  # either a tile or None, both legit outcomes

def test_seam_blend_tile_shape_mismatch_none():
    crop = Image.fromarray(_rand_rgb(80, 80))
    res = Image.fromarray(_rand_rgb(60, 60))  # different size -> shape mismatch -> None
    m = _mask_img(80, 80, box=(10, 10, 40, 40))
    check("seam_blend_shape_mismatch_none", I._seam_blend_tile(crop, res, m) is None)

def test_seam_blend_tile_empty_mask_none():
    crop = Image.fromarray(_rand_rgb(64, 64))
    res = Image.fromarray(_rand_rgb(64, 64))
    m = _mask_img(64, 64)  # empty -> seamless_clone returns None (empty ys/xs)
    check("seam_blend_empty_mask_none", I._seam_blend_tile(crop, res, m) is None)


# ---------------- _mask_quality_ok ----------------

def test_mask_quality_empty():
    m = _mask_img(64, 64)
    ok, reason, stats = I._mask_quality_ok(m)
    check("mq_empty", not ok and reason == "empty")

def test_mask_quality_near_empty():
    m = _mask_img(200, 200, box=(0, 0, 2, 2))  # tiny -> cov < 0.0005
    ok, reason, stats = I._mask_quality_ok(m)
    check("mq_near_empty", not ok and reason == "near_empty", reason)

def test_mask_quality_over_coverage():
    m = _mask_img(64, 64, box=(0, 0, 64, 64))  # cov ~1.0 > 0.85 cap, not a "background" phrase
    ok, reason, stats = I._mask_quality_ok(m, region_phrase="her face")
    check("mq_over_coverage", not ok and reason == "over_coverage", reason)

def test_mask_quality_over_coverage_bigregion_allowed():
    # cov ~0.90: over the plain 0.85 cap but under the 0.97 "big region" cap
    m = _mask_img(64, 64, box=(0, 0, 62, 62))
    ok, reason, stats = I._mask_quality_ok(m, region_phrase="the whole background")
    check("mq_bigregion_cap_relaxed", ok, (reason, stats))

def test_mask_quality_good_solid_blob():
    m = _mask_img(64, 64, box=(10, 10, 50, 50))  # solid connected square, moderate coverage
    ok, reason, stats = I._mask_quality_ok(m, region_phrase="her hat")
    check("mq_good_blob", ok and reason == "ok", (reason, stats))

def test_mask_quality_small_item_hint():
    # cov ~0.39 passes the neutral cap but must fail the LLM small-item prior
    m = _mask_img(64, 64, box=(10, 10, 50, 50))
    ok, reason, stats = I._mask_quality_ok(m, region_phrase="shoes", small_item=True)
    check("mq_small_hint_reject", not ok and reason == "over_coverage", (reason, stats))
    ok, reason, stats = I._mask_quality_ok(m, region_phrase="shoes", small_item=None)
    check("mq_small_hint_unknown_neutral", ok, (reason, stats))

def test_mask_quality_big_region_hint():
    # cov ~0.90: rejected without a hint, allowed when the LLM says "large"
    m = _mask_img(64, 64, box=(0, 0, 62, 62))
    ok, reason, stats = I._mask_quality_ok(m, region_phrase="the ocean", big_region=True)
    check("mq_big_hint_relaxed", ok, (reason, stats))
    ok, reason, stats = I._mask_quality_ok(m, region_phrase="the ocean", big_region=False)
    check("mq_big_hint_strict", not ok and reason == "over_coverage", (reason, stats))

def test_mask_quality_fragmented_sparse():
    # scattered speckles across a big bbox, low fill, no dominant component
    m = Image.new("L", (100, 100), 0)
    px = m.load()
    import random
    random.seed(42)
    for _ in range(40):
        x, y = random.randint(5, 94), random.randint(5, 94)
        px[x, y] = 255
    ok, reason, stats = I._mask_quality_ok(m, region_phrase="a scattered thing")
    check("mq_fragmented", not ok and reason in ("fragmented_sparse", "too_fragmented"), (reason, stats))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback; traceback.print_exc()
            print("  ERROR in " + fn.__name__ + ": " + type(e).__name__ + ": " + str(e))
    passed = sum(1 for _, c in RESULTS if c)
    print(f"\n{len(fns)-failed}/{len(fns)} functions, {passed}/{len(RESULTS)} checks passed")
    # A failed check() must reach the exit code. This used to be `1 if failed else 0`,
    # where `failed` counts EXCEPTIONS only -- so every assertion in the file could
    # print FAIL while the suite still exited 0, invisible to anything judging by
    # exit code (which is how these suites are judged).
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)

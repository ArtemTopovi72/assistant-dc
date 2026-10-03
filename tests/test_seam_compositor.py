"""Deterministic regression tests for the seam-suppressing compositor primitives
(image.py). These lock in the INVARIANTS each stage must uphold so future refactors
can't silently regress correctness. No ComfyUI / LM Studio / GPU needed — all inputs are
synthetic and fixed-seed, so results are reproducible.

Run:  venv/Scripts/python.exe tests/test_seam_compositor.py
Exit code 0 = all pass. Also importable by pytest (test_* functions).
"""
from __future__ import annotations
import os
import sys

import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import image as IM  # noqa: E402
# The seam compositor moved to compositing.py. These four are internal to it
# (image.py drives _blend_region, not its stages), so they are addressed on
# their own module rather than re-exported through image just for tests.
import compositing as _CMP  # noqa: E402

RNG = np.random.default_rng(0)
H = W = 160


def _gradient_bg():
    """A smooth LAB-ish gradient image (deterministic), as a stand-in 'original'."""
    yy = np.linspace(60, 200, H)[:, None] * np.ones((1, W))
    xx = np.linspace(-20, 20, W)[None, :] * np.ones((H, 1))
    base = np.clip(yy + xx, 0, 255)
    return np.stack([base, base * 0.8, base * 0.65], 2).clip(0, 255).astype(np.uint8)


def _center_mask(frac=0.4):
    m = np.zeros((H, W), np.uint8)
    a = int(W * frac * 0.5)
    cv2.ellipse(m, (W // 2, H // 2), (a, a), 0, 0, 360, 255, -1)
    return m


def _tone_shift(img, m, dlab=(20, 8, -6)):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab += np.asarray(dlab, np.float32)
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def _recolor(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + 90) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] + 60, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


_RESULTS = []


import os as _os
def _check(name, cond, detail=""):
    _RESULTS.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond and _os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))


# --------------------------------------------------------------------------- #
def test_band_masks():
    m = _center_mask()
    ri, ro = _CMP._band_masks(m, 6)
    binm = m > 0
    _check("band_masks: ring_in inside mask", bool((ri & ~binm).sum() == 0))
    _check("band_masks: ring_out outside mask", bool((ro & binm).sum() == 0))
    _check("band_masks: both rings non-empty", ri.sum() > 0 and ro.sum() > 0)


def test_is_continuation():
    bg = _gradient_bg()
    m = _center_mask()
    cont_patch = _tone_shift(bg, m, (6, 3, -3))     # mild shift -> still continuation
    dist_patch = _recolor(bg)                        # strong recolour -> distinct
    _check("is_continuation: mild tone shift -> True", _CMP._is_continuation(bg, cont_patch, m))
    _check("is_continuation: strong recolour -> False",
           not _CMP._is_continuation(bg, dist_patch, m))


def test_color_harmonize_shapes_and_interior():
    bg = _gradient_bg()
    m = _center_mask()
    rec = _recolor(bg)
    out = _CMP._color_harmonize(bg, rec, m, strength=1.0)
    _check("harmonize: shape/dtype preserved",
           out.shape == rec.shape and out.dtype == np.uint8)
    # deep interior (eroded core) must stay close to the recolour (boundary-weighted)
    core = cv2.erode((m > 0).astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
    lab_o = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_r = cv2.cvtColor(rec, cv2.COLOR_RGB2LAB).astype(np.float32)
    drift = np.sqrt(((lab_o[core] - lab_r[core]) ** 2).sum(1)).mean()
    _check("harmonize: deep interior preserved (boundary-weighted)", drift < 8.0,
           f"core ΔE drift={drift:.2f}")
    # outside the mask must be byte-identical (harmonize only touches the patch interior)
    _check("harmonize: outside-mask untouched",
           bool((out[m == 0] == rec[m == 0]).all()))


def test_seamless_clone_guards_and_locality():
    bg = _gradient_bg()
    m = _center_mask()
    patch = _tone_shift(bg, m, (25, 0, 0))
    out = IM._seamless_clone_arr(bg, patch, m)
    _check("seamless: returns an array for a valid mask", out is not None)
    if out is not None:
        _check("seamless: shape preserved", out.shape == bg.shape)
        # far-outside pixels essentially unchanged (Poisson only solves inside)
        far = np.zeros((H, W), bool); far[:8, :8] = True
        _check("seamless: far-field ~ background",
               float(np.abs(out[far].astype(int) - bg[far].astype(int)).mean()) < 2.0)
    full = np.full((H, W), 255, np.uint8)
    _check("seamless: rejects full-frame mask (no exterior)",
           IM._seamless_clone_arr(bg, patch, full) is None)
    empty = np.zeros((H, W), np.uint8)
    _check("seamless: rejects empty mask", IM._seamless_clone_arr(bg, patch, empty) is None)


def test_laplacian_blend_alpha_extremes():
    bg = _gradient_bg()
    patch = _recolor(bg)
    a1 = np.ones((H, W), np.float32)
    a0 = np.zeros((H, W), np.float32)
    out1 = _CMP._laplacian_blend(bg, patch, a1)
    out0 = _CMP._laplacian_blend(bg, patch, a0)
    _check("laplacian: alpha=1 ~ patch",
           float(np.abs(out1.astype(int) - patch.astype(int)).mean()) < 2.0)
    _check("laplacian: alpha=0 ~ background",
           float(np.abs(out0.astype(int) - bg.astype(int)).mean()) < 2.0)
    _check("laplacian: shape/dtype preserved",
           out1.shape == bg.shape and out1.dtype == np.uint8)


def test_blend_region_routing_and_outside():
    bg = _gradient_bg()
    m = _center_mask()
    P = lambda a: Image.fromarray(a)  # noqa: E731
    # continuation path
    cont = _tone_shift(bg, m, (18, 6, -4))
    rc = np.asarray(IM._blend_region(P(bg), P(cont), P(m)).convert("RGB"))
    _check("blend_region: shape preserved", rc.shape == bg.shape)
    _check("blend_region: outside mask ~ background",
           float(np.abs(rc[m == 0].astype(int) - bg[m == 0].astype(int)).mean()) < 1.5)
    # distinct recolour path -> interior must keep the new colour
    rec = _recolor(bg)
    rr = np.asarray(IM._blend_region(P(bg), P(rec), P(m)).convert("RGB"))
    core = cv2.erode((m > 0).astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
    lab_blend = cv2.cvtColor(rr, cv2.COLOR_RGB2LAB).astype(np.float32)[core]
    lab_rec = cv2.cvtColor(rec, cv2.COLOR_RGB2LAB).astype(np.float32)[core]
    lab_orig = cv2.cvtColor(bg, cv2.COLOR_RGB2LAB).astype(np.float32)[core]
    d_rec = np.sqrt(((lab_blend - lab_rec) ** 2).sum(1)).mean()
    d_orig = np.sqrt(((lab_blend - lab_orig) ** 2).sum(1)).mean()
    # Scale-free invariant: the blended interior must stay much CLOSER to the intended
    # recolour than to the original — i.e. the edit is preserved, not washed toward bg.
    _check("blend_region: recolour interior preserved (closer to edit than to original)",
           d_rec < 0.5 * d_orig, f"ΔE_to_recolour={d_rec:.1f} vs ΔE_to_original={d_orig:.1f}")


def test_blend_region_determinism():
    bg = _gradient_bg(); m = _center_mask()
    patch = _tone_shift(bg, m, (15, 5, -3))
    P = lambda a: Image.fromarray(a)  # noqa: E731
    a = np.asarray(IM._blend_region(P(bg), P(patch), P(m)).convert("RGB"))
    b = np.asarray(IM._blend_region(P(bg), P(patch), P(m)).convert("RGB"))
    _check("blend_region: deterministic (identical on repeat)", bool((a == b).all()))


def test_face_guard_no_face_passthrough(tmpdir=None):
    import tempfile
    d = tempfile.mkdtemp()
    bg = _gradient_bg()
    op = os.path.join(d, "o.png"); Image.fromarray(bg).save(op)
    up = os.path.join(d, "u.png"); Image.fromarray(bg).resize((W * 2, H * 2)).save(up)
    import identity_metrics as idm
    saved = idm.face_box
    try:
        idm.face_box = lambda *a, **k: None     # force "no face"
        out = IM._preserve_face_after_upscale(op, up)
        _check("face_guard: no face -> returns ESRGAN result unchanged", out == up)
    finally:
        idm.face_box = saved


def main():
    for fn in [test_band_masks, test_is_continuation, test_color_harmonize_shapes_and_interior,
               test_seamless_clone_guards_and_locality, test_laplacian_blend_alpha_extremes,
               test_blend_region_routing_and_outside, test_blend_region_determinism,
               test_face_guard_no_face_passthrough]:
        fn()
    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    print(f"\n{passed}/{len(_RESULTS)} checks passed")
    return 0 if passed == len(_RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())

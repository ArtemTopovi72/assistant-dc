"""Real-model coverage for identity_metrics.py: YuNet face detection + SFace
recognition against actual project photos (no mocking of cv2 — these are small
local ONNX models, no GPU/network needed) plus synthetic images for the
no-face/missing-file/size-mismatch branches.
Run: venv/Scripts/python.exe tests/test_identity_metrics.py
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
from pathlib import Path
import numpy as np
import cv2
import identity_metrics as IM

RESULTS = []
import os as _os
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + detail))
    if not cond and _os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))

_TMP = Path(tempfile.mkdtemp(prefix="idmetrics_"))
_PORTRAIT_SRC = Path(os.path.expanduser("~/Documents/ComfyUI/input/1.webp"))


def _noise_image(name, w=200, h=200):
    p = _TMP / name
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(p), arr)
    return str(p)


def _portrait_copy(name):
    if not _PORTRAIT_SRC.exists():
        return None
    p = _TMP / name
    img = cv2.imread(str(_PORTRAIT_SRC))
    cv2.imwrite(str(p), img)
    return str(p)


def test_face_embedding_real_and_missing():
    check("face_embedding_missing_file_none", IM.face_embedding(str(_TMP / "nope.jpg")) is None)
    noface = _noise_image("noface.jpg")
    check("face_embedding_no_face_none", IM.face_embedding(noface) is None)
    portrait = _portrait_copy("portrait1.jpg")
    if portrait:
        emb = IM.face_embedding(portrait)
        check("face_embedding_real_face_found", emb is not None)
    else:
        check("face_embedding_real_skipped_no_portrait", True)


def test_identity_cosine():
    portrait = _portrait_copy("p_a.jpg")
    if portrait:
        cos = IM.identity_cosine(portrait, portrait)
        check("identity_cosine_same_image_high", cos is not None and cos > IM.SFACE_SAME_PERSON_COSINE)
    else:
        check("identity_cosine_skipped_no_portrait", True)
    noface = _noise_image("noface2.jpg")
    check("identity_cosine_missing_face_none", IM.identity_cosine(noface, noface) is None)
    if portrait:
        check("identity_cosine_one_missing_face_none", IM.identity_cosine(portrait, noface) is None)


def test_changed_fraction():
    a = _noise_image("cf_a.jpg", 100, 100)
    check("changed_fraction_missing_file_empty", IM.changed_fraction(str(_TMP / "nope.jpg"), a) == {})
    b_arr = (np.zeros((100, 100, 3), dtype=np.uint8))
    b = _TMP / "cf_b.jpg"
    cv2.imwrite(str(b), b_arr)
    out = IM.changed_fraction(a, str(b))
    check("changed_fraction_overall_present", "overall" in out and 0 <= out["overall"] <= 1)

    # size mismatch triggers resize branch
    b_diff = _TMP / "cf_b_diffsize.jpg"
    cv2.imwrite(str(b_diff), np.zeros((50, 50, 3), dtype=np.uint8))
    out2 = IM.changed_fraction(a, str(b_diff))
    check("changed_fraction_resize_branch", "overall" in out2)

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:50, :50] = 255
    out3 = IM.changed_fraction(a, str(b), mask=mask)
    check("changed_fraction_mask_keys_present", "outside_mask" in out3 and "inside_mask" in out3)

    # mask needing resize
    small_mask = np.zeros((10, 10), dtype=np.uint8)
    small_mask[:5, :5] = 255
    out4 = IM.changed_fraction(a, str(b), mask=small_mask)
    check("changed_fraction_mask_resize_branch", "outside_mask" in out4)


def test_face_region_change():
    portrait = _portrait_copy("frc_a.jpg")
    if portrait:
        out = IM.face_region_change(portrait, portrait)
        check("face_region_change_same_image_zero_ish", out is not None and out < 0.05)
    else:
        check("face_region_change_skipped_no_portrait", True)
    noface = _noise_image("frc_noface.jpg")
    check("face_region_change_no_face_none", IM.face_region_change(noface, noface) is None)
    check("face_region_change_missing_file_none", IM.face_region_change(str(_TMP / "nope.jpg"), noface) is None)
    if portrait:
        # size-mismatch b resize branch
        b_arr = cv2.resize(cv2.imread(portrait), (100, 100))
        b_path = _TMP / "frc_b_resized.jpg"
        cv2.imwrite(str(b_path), b_arr)
        out2 = IM.face_region_change(portrait, str(b_path))
        check("face_region_change_resize_branch", out2 is None or isinstance(out2, float))


def test_face_bbox_frac():
    portrait = _portrait_copy("bbox_a.jpg")
    if portrait:
        out = IM.face_bbox_frac(portrait)
        check("face_bbox_frac_real_face", out is not None and len(out) == 4)
    else:
        check("face_bbox_frac_skipped_no_portrait", True)
    check("face_bbox_frac_missing_file_none", IM.face_bbox_frac(str(_TMP / "nope.jpg")) is None)
    noface = _noise_image("bbox_noface.jpg")
    check("face_bbox_frac_no_face_none", IM.face_bbox_frac(noface) is None)


def test_face_box():
    portrait = _portrait_copy("box_a.jpg")
    if portrait:
        out = IM.face_box(portrait)
        check("face_box_real_face", out is not None and len(out) == 4)
        out2 = IM.face_box(portrait, pad=0.5)
        check("face_box_custom_pad", out2 is not None)
    else:
        check("face_box_skipped_no_portrait", True)
    check("face_box_missing_file_none", IM.face_box(str(_TMP / "nope.jpg")) is None)
    noface = _noise_image("box_noface.jpg")
    check("face_box_no_face_none", IM.face_box(noface) is None)

    # large image triggers the downscale branch (scale < 1.0)
    if portrait:
        img = cv2.imread(portrait)
        h, w = img.shape[:2]
        big = cv2.resize(img, (int(w * 3), int(h * 3))) if max(h, w) < 1600 else img
        big_path = _TMP / "box_big.jpg"
        cv2.imwrite(str(big_path), big)
        out3 = IM.face_box(str(big_path))
        check("face_box_downscale_branch", out3 is None or len(out3) == 4)


def test_rec_singleton():
    r1 = IM._rec()
    r2 = IM._rec()
    check("rec_singleton_same_instance", r1 is r2)


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

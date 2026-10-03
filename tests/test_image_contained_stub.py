"""Boundary-stubbed coverage for _contained_region_mask and fix_hands: crafts exact
mask/detect/refine outcomes by stubbing _region_mask_file / _face_detector_mask /
_vlm_qa_verdict / _upload_image_to_comfy / _submit_and_collect / _submit_and_poll
so every branch (manual-mask, whole-face detector hit/miss, STAGE-0/1 QA pass/fail/
retry, MeshGraphormer detect/no-detect, refine success/fail, region-fallback) runs
deterministically WITHOUT live ComfyUI or SAM3/Florence.
Run: venv/Scripts/python.exe tests/test_image_contained_stub.py
"""
import os, sys, tempfile, threading, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
import numpy as np
from pathlib import Path
from PIL import Image
import image as I
# the whole-face read is the model's; phrases run live in bench/intent_rest_live.py
import intent
intent.YES_STUB = lambda q, t: "whole face or head" in q and t.strip() in ("her face", "the face")

# --- hermetic by default -----------------------------------------------------
# _item_attributes() asks the LLM to classify a phrase, and the mask pipeline
# calls it from four places. These suites are named "stub" and are supposed to be
# offline, but they were reaching a live LM Studio on localhost:1234 — so the
# result depended on whether a model happened to be loaded, and the same test
# passed on one machine and failed on another. Pin a deterministic classifier;
# individual tests override it via _Patches when a scenario needs other values.
_ITEM_ATTRS_DEFAULT = {"worn": False, "small": True, "large": False, "multi": False}
from _image_patch import Patches as _Patches, broadcast as _broadcast
_broadcast("_item_attributes",
           lambda ctx, phrase, _d=_ITEM_ATTRS_DEFAULT: dict(_d))
import models

_TMP = Path(tempfile.mkdtemp(prefix="imgstub_"))
RESULTS = []
import os as _os
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + detail))
    if not cond and _os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))

def Ctx():
    return models.Context(
        models=None, transcription_cache={}, cache_file=_TMP / "cache.json",
        asr_lock=threading.Lock(), tts_lock=threading.Lock(),
    )

def _photo(name="p.png", w=256, h=256, col=(120, 140, 160)):
    p = _TMP / name
    Image.new("RGB", (w, h), col).save(p)
    return str(p)

def _mask_file(name, w=256, h=256, box=None, speckle=False):
    m = Image.new("L", (w, h), 0)
    if box:
        px = m.load()
        x0, y0, x1, y1 = box
        for y in range(y0, y1):
            for x in range(x0, x1):
                px[x, y] = 255
    if speckle:
        import random
        random.seed(1)
        px = m.load()
        for _ in range(30):
            px[random.randint(0, w-1), random.randint(0, h-1)] = 255
    p = _TMP / name
    m.save(p)
    return str(p)




# ---------------- _contained_region_mask ----------------

def test_manual_mask_success():
    ctx = Ctx()
    img = _photo("m1.png")
    mask = _mask_file("m1_mask.png", box=(40, 40, 160, 160))
    out = I._contained_region_mask(ctx, img, "hand", grow=6, seed=1, timeout=60, mask_override=mask)
    check("manual_mask_success", out is not None and out[2][2] > out[2][0])

def test_manual_mask_missing_file_falls_through():
    ctx = Ctx()
    img = _photo("m1b.png")
    with _Patches(_upload_image_to_comfy=lambda *a, **k: None):
        out = I._contained_region_mask(ctx, img, "hand", grow=6, seed=1, timeout=60,
                                        mask_override="Z:/no_such_mask.png")
        check("manual_mask_missing_falls_to_text_seg_upload_fail", out is None)

def test_manual_mask_empty():
    ctx = Ctx()
    img = _photo("m2.png")
    mask = _mask_file("m2_mask.png")  # empty
    out = I._contained_region_mask(ctx, img, "hand", grow=6, seed=1, timeout=60, mask_override=mask)
    check("manual_mask_empty_none", out is None)

def test_manual_mask_degenerate_crop():
    ctx = Ctx()
    # tiny 10x10 image + a 1x1 mask pixel in the corner: pad clamps to image bounds
    # (max width possible here is 10px, and the padded box is clamped to [0,10)x[0,10)
    # only if the source itself is small enough that (x1-x0)<16 even after padding)
    img = _photo("m3.png", w=10, h=10)
    mask = _mask_file("m3_mask.png", w=10, h=10, box=(4, 4, 5, 5))
    out = I._contained_region_mask(ctx, img, "hand", grow=6, seed=1, timeout=60, mask_override=mask,
                                    pad=0.0)
    check("manual_mask_degenerate_none", out is None)

def test_manual_mask_protect_face():
    ctx = Ctx()
    img = _photo("m4.png")
    mask = _mask_file("m4_mask.png", box=(0, 0, 256, 256))  # full frame
    fake_idm = types.SimpleNamespace(face_box=lambda path, pad=0.10: (100, 100, 150, 150))
    orig_mod = sys.modules.get("identity_metrics")
    sys.modules["identity_metrics"] = fake_idm
    try:
        out = I._contained_region_mask(ctx, img, "hat", grow=6, seed=1, timeout=60, mask_override=mask,
                                        protect_face=True)
        check("manual_mask_protect_face_still_ok", out is not None)
    finally:
        if orig_mod is not None: sys.modules["identity_metrics"] = orig_mod
        else: sys.modules.pop("identity_metrics", None)

def test_manual_mask_protect_face_empties_mask():
    ctx = Ctx()
    img = _photo("m5.png", w=100, h=100)
    mask = _mask_file("m5_mask.png", w=100, h=100, box=(40, 40, 60, 60))  # small mask inside face box
    fake_idm = types.SimpleNamespace(face_box=lambda path, pad=0.10: (0, 0, 100, 100))  # covers everything
    orig_mod = sys.modules.get("identity_metrics")
    sys.modules["identity_metrics"] = fake_idm
    try:
        out = I._contained_region_mask(ctx, img, "hat", grow=6, seed=1, timeout=60, mask_override=mask,
                                        protect_face=True)
        check("manual_mask_protect_face_empties_none", out is None)
    finally:
        if orig_mod is not None: sys.modules["identity_metrics"] = orig_mod
        else: sys.modules.pop("identity_metrics", None)

def test_whole_face_detector_hit():
    ctx = Ctx()
    img = _photo("wf1.png")
    with _Patches(_face_detector_mask=lambda path, size, pad=0.18: Image.new("L", size, 0)):
        # returns an all-black mask -> bbox None downstream, but we're testing the
        # detector-hit branch is taken (mask is not None) vs later empty-bbox reject
        out = I._contained_region_mask(ctx, img, "her face", grow=6, seed=1, timeout=60)
        check("whole_face_detector_hit_branch", out is None)  # empty mask -> reject, but branch WAS taken

def test_whole_face_detector_hit_nonempty():
    ctx = Ctx()
    img = _photo("wf2.png")
    m = Image.new("L", (256, 256), 0)
    px = m.load()
    for y in range(50, 200):
        for x in range(50, 200):
            px[x, y] = 255
    with _Patches(_face_detector_mask=lambda path, size, pad=0.18: m):
        out = I._contained_region_mask(ctx, img, "her face", grow=6, seed=1, timeout=60)
        check("whole_face_detector_success", out is not None)

def test_body_and_face_is_not_a_face_preset():
    """Live 10-02: «the man's body and face» hit the face regex, YuNet masked
    the face only and the man stayed in his chair."""
    called = []
    with _Patches(_face_detector_mask=lambda *a, **k: called.append(1)):
        try:
            I._contained_region_mask(Ctx(), _photo("bf1.png"),
                                     "the man's body and face", grow=6, seed=1, timeout=60)
        except Exception:
            pass
    check("body_and_face_skips_face_detector", not called)

def test_whole_face_detector_miss_falls_to_text_seg():
    ctx = Ctx()
    img = _photo("wf3.png")
    # A face-sized box (~11% of a 256px frame). The old fixture covered 50% of
    # the image, which the STAGE-0 quality gate now correctly rejects as
    # over-coverage for a small region — the test was asserting on a mask no real
    # face detector would produce, so it failed on the gate rather than on the
    # fallback branch it is meant to cover.
    maskfile = _mask_file("wf3_mask.png", box=(90, 60, 170, 150))
    # This test is about the FALLBACK BRANCH (detector miss -> text segmentation),
    # not the Stage-1 vision QA verdict — but a text-segmented mask (unlike a
    # detector hit) DOES go through Stage-1 QA, which without a stub reaches a
    # real LM Studio on localhost:1234 and (correctly) rejects a plain solid-
    # colour test square as not matching "her face", making this "hermetic" test
    # silently depend on whatever model happens to be loaded. MASK_QA_ENABLED=
    # False isolates the branch under test the same way the neighbouring
    # text-seg QA tests do for THEIR non-QA scenarios.
    with _Patches(_face_detector_mask=lambda *a, **k: None,
                  _upload_image_to_comfy=lambda *a, **k: "uploaded.png",
                  _region_mask_file=lambda *a, **k: maskfile,
                  MASK_QA_ENABLED=False):
        out = I._contained_region_mask(ctx, img, "her face", grow=6, seed=1, timeout=60)
        check("whole_face_miss_text_seg_used", out is not None)

def test_text_seg_stage0_qa_pass():
    ctx = Ctx()
    img = _photo("ts1.png")
    maskfile = _mask_file("ts1_mask.png", box=(40, 40, 150, 150))  # solid connected blob
    with _Patches(_upload_image_to_comfy=lambda *a, **k: "u.png",
                  _region_mask_file=lambda *a, **k: maskfile,
                  MASK_QA_ENABLED=False):
        out = I._contained_region_mask(ctx, img, "her hat", grow=6, seed=1, timeout=60)
        check("text_seg_stage0_pass", out is not None)

def test_text_seg_stage0_qa_fail_then_retry_then_reject():
    ctx = Ctx()
    img = _photo("ts2.png")
    scattered = _mask_file("ts2_mask.png", box=(0, 0, 1, 1), speckle=True)  # near-empty/fragmented
    calls = {"n": 0}
    def fake_region_mask_file(*a, **k):
        calls["n"] += 1
        return scattered
    with _Patches(_upload_image_to_comfy=lambda *a, **k: "u.png",
                  _region_mask_file=fake_region_mask_file,
                  MASK_QA_ENABLED=False):
        out = I._contained_region_mask(ctx, img, "her hat", grow=6, seed=1, timeout=60)
        check("stage0_qa_fail_retried_then_rejected", out is None and calls["n"] == 2, calls["n"])

def test_text_seg_stage1_qa_wrong_then_reject():
    ctx = Ctx()
    img = _photo("ts3.png")
    maskfile = _mask_file("ts3_mask.png", box=(40, 40, 150, 150))
    calls = {"n": 0}
    def fake_region_mask_file(*a, **k):
        calls["n"] += 1
        return maskfile
    with _Patches(_upload_image_to_comfy=lambda *a, **k: "u.png",
                  _region_mask_file=fake_region_mask_file,
                  MASK_QA_ENABLED=True,
                  _vlm_qa_verdict=lambda *a, **k: "WRONG"):
        out = I._contained_region_mask(ctx, img, "her hat", grow=6, seed=1, timeout=60)
        check("stage1_qa_wrong_retried_then_rejected", out is None and calls["n"] == 2, calls["n"])

def test_text_seg_stage1_qa_correct():
    ctx = Ctx()
    img = _photo("ts4.png")
    maskfile = _mask_file("ts4_mask.png", box=(40, 40, 150, 150))
    with _Patches(_upload_image_to_comfy=lambda *a, **k: "u.png",
                  _region_mask_file=lambda *a, **k: maskfile,
                  MASK_QA_ENABLED=True,
                  _vlm_qa_verdict=lambda *a, **k: "CORRECT"):
        out = I._contained_region_mask(ctx, img, "her hat", grow=6, seed=1, timeout=60)
        check("stage1_qa_correct_accepts", out is not None)

def test_text_seg_upload_fails():
    ctx = Ctx()
    img = _photo("ts5.png")
    with _Patches(_upload_image_to_comfy=lambda *a, **k: None):
        out = I._contained_region_mask(ctx, img, "her hat", grow=6, seed=1, timeout=60)
        check("text_seg_upload_fail_none", out is None)

def test_text_seg_no_mask_produced():
    ctx = Ctx()
    img = _photo("ts6.png")
    with _Patches(_upload_image_to_comfy=lambda *a, **k: "u.png",
                  _region_mask_file=lambda *a, **k: None):
        out = I._contained_region_mask(ctx, img, "her hat", grow=6, seed=1, timeout=60)
        check("text_seg_no_mask_none", out is None)


# ---------------- fix_hands (boundary-stubbed) ----------------

def test_fix_hands_manual_mask_success():
    ctx = Ctx()
    img = _photo("fh1.png")
    mask = _mask_file("fh1_mask.png", box=(40, 40, 150, 150))
    with _Patches(edit_region_contained_via_firered=lambda *a, **k: str(_photo("fh1_out.png"))):
        out = I.fix_hands(ctx, img, mask_override=mask, timeout=60)
        check("fix_hands_manual_success", out is not None and Path(out).exists())

def test_fix_hands_manual_mask_fails():
    ctx = Ctx()
    img = _photo("fh2.png")
    mask = _mask_file("fh2_mask.png", box=(40, 40, 150, 150))
    with _Patches(edit_region_contained_via_firered=lambda *a, **k: None):
        out = I.fix_hands(ctx, img, mask_override=mask, timeout=60)
        check("fix_hands_manual_fail_none", out is None)

def test_fix_hands_upload_fails():
    ctx = Ctx()
    img = _photo("fh3.png")
    with _Patches(_upload_image_to_comfy=lambda *a, **k: None):
        out = I.fix_hands(ctx, img, timeout=60)
        check("fix_hands_upload_fail_none", out is None)

def test_fix_hands_detect_empty():
    ctx = Ctx()
    img = _photo("fh4.png")
    with _Patches(_upload_image_to_comfy=lambda *a, **k: "u.png",
                  _submit_and_collect=lambda *a, **k: None,
                  _handfix_region_fallback=lambda *a, **k: None):
        out = I.fix_hands(ctx, img, timeout=60)
        check("fix_hands_detect_empty_none", out is None)

def test_fix_hands_detect_missing_depth_or_mask():
    ctx = Ctx()
    img = _photo("fh5.png")
    with _Patches(_upload_image_to_comfy=lambda *a, **k: "u.png",
                  _submit_and_collect=lambda *a, **k: {"1": "somefile_other.png"},
                  _handfix_region_fallback=lambda *a, **k: None):
        out = I.fix_hands(ctx, img, timeout=60)
        check("fix_hands_missing_depth_mask_none", out is None)

def test_fix_hands_no_hand_in_mask_then_fallback_none():
    ctx = Ctx()
    img = _photo("fh6.png")
    depth_p = _photo("fh6_handdepth.png")
    mask_p = _mask_file("fh6_handmask.png")  # empty -> no hand detected branch
    with _Patches(_upload_image_to_comfy=lambda *a, **k: "u.png",
                  _submit_and_collect=lambda *a, **k: {"1": depth_p, "2": mask_p},
                  _handfix_region_fallback=lambda *a, **k: None):
        out = I.fix_hands(ctx, img, timeout=60)
        check("fix_hands_no_hand_fallback_none", out is None)

def test_fix_hands_detected_refine_success():
    ctx = Ctx()
    img = _photo("fh7.png", w=200, h=200)
    depth_p = _photo("fh7_handdepth.png", w=200, h=200)
    mask_p = _mask_file("fh7_handmask.png", w=200, h=200, box=(60, 60, 140, 140))
    refined_p = _photo("fh7_refined.png", w=80, h=80)
    with _Patches(_upload_image_to_comfy=lambda *a, **k: "u.png",
                  _submit_and_collect=lambda *a, **k: {"1": depth_p, "2": mask_p},
                  _submit_and_poll=lambda *a, **k: refined_p,
                  _handfix_region_fallback=lambda *a, **k: None):
        out = I.fix_hands(ctx, img, timeout=60)
        check("fix_hands_refine_success", out is not None and Path(out).exists())

def test_fix_hands_detected_refine_fails_then_fallback_succeeds():
    ctx = Ctx()
    img = _photo("fh8.png", w=200, h=200)
    depth_p = _photo("fh8_handdepth.png", w=200, h=200)
    mask_p = _mask_file("fh8_handmask.png", w=200, h=200, box=(60, 60, 140, 140))
    fb_out = _photo("fh8_fallback.png")
    with _Patches(_upload_image_to_comfy=lambda *a, **k: "u.png",
                  _submit_and_collect=lambda *a, **k: {"1": depth_p, "2": mask_p},
                  _submit_and_poll=lambda *a, **k: None,
                  _handfix_region_fallback=lambda *a, **k: fb_out):
        out = I.fix_hands(ctx, img, timeout=60)
        check("fix_hands_refine_fail_fallback_success", out == fb_out)

def test_fix_hands_degenerate_crop():
    ctx = Ctx()
    img = _photo("fh9.png", w=200, h=200)
    depth_p = _photo("fh9_handdepth.png", w=200, h=200)
    mask_p = _mask_file("fh9_handmask.png", w=200, h=200, box=(100, 100, 103, 103))  # tiny -> degenerate
    with _Patches(_upload_image_to_comfy=lambda *a, **k: "u.png",
                  _submit_and_collect=lambda *a, **k: {"1": depth_p, "2": mask_p},
                  _handfix_region_fallback=lambda *a, **k: None):
        out = I.fix_hands(ctx, img, pad_frac=0.0, timeout=60)
        check("fix_hands_degenerate_crop_none", out is None)

def test_fix_hands_no_fallback_flag():
    ctx = Ctx()
    img = _photo("fh10.png")
    with _Patches(_upload_image_to_comfy=lambda *a, **k: "u.png",
                  _submit_and_collect=lambda *a, **k: None):
        out = I.fix_hands(ctx, img, fallback=False, timeout=60)
        check("fix_hands_no_fallback_flag_none", out is None)


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
    # A failed check() must reach the exit code. It used to only print: `failed`
    # counted exceptions alone, so every assertion in this file could go red
    # while the suite still exited 0 -- invisible to anything judging by exit
    # code, which is how these suites are judged.
    bad = [n for n, c in RESULTS if not c]
    print(f"\n{len(fns)-failed}/{len(fns)} functions, {passed}/{len(RESULTS)} checks passed")
    if bad:
        print("FAILED CHECKS: " + ", ".join(bad))
    sys.exit(1 if (failed or bad) else 0)

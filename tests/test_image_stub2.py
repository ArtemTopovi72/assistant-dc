"""Boundary-stubbed coverage round 2: edit_region_contained_cropped,
_preserve_face_after_upscale, _composite_reference_head. All pure PIL/numpy logic
gated by stubbed identity_metrics.face_box / _region_mask_file / _upload_image_to_comfy
/ _submit_and_poll / edit_image_with_firered -- no GPU needed.
Run: venv/Scripts/python.exe tests/test_image_stub2.py
"""
import os, sys, tempfile, threading, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
from pathlib import Path
from PIL import Image
import image as I
from _image_patch import Patches as _Patches

# --- hermetic by default -----------------------------------------------------
# _item_attributes() asks the LLM to classify a phrase, and the mask pipeline
# calls it from four places. These suites are named "stub" and are supposed to be
# offline, but they were reaching a live LM Studio on localhost:1234 — so the
# result depended on whether a model happened to be loaded, and the same test
# passed on one machine and failed on another. Pin a deterministic classifier;
# individual tests override it via _Patches when a scenario needs other values.
_ITEM_ATTRS_DEFAULT = {"worn": False, "small": True, "large": False, "multi": False}
I._item_attributes = lambda ctx, phrase, _d=_ITEM_ATTRS_DEFAULT: dict(_d)
import models

_TMP = Path(tempfile.mkdtemp(prefix="imgstub2_"))
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

def _photo(name="p.png", w=200, h=200, col=(120, 140, 160)):
    p = _TMP / name
    Image.new("RGB", (w, h), col).save(p)
    return str(p)

def _mask_file(name, w=256, h=256, box=None):
    m = Image.new("L", (w, h), 0)
    if box:
        px = m.load()
        x0, y0, x1, y1 = box
        for y in range(y0, y1):
            for x in range(x0, x1):
                px[x, y] = 255
    p = _TMP / name
    m.save(p)
    return str(p)




def _with_face_box(box):
    """Stub the face detector. BOTH entry points must be stubbed: mask protection
    uses face_box (single dominant face) while the upscale/re-render guard uses
    face_boxes (every face). Stubbing only one silently sends the other to the
    real YuNet detector, which finds nothing in these synthetic swatches — the
    test then passes/fails for the wrong reason."""
    orig_mod = sys.modules.get("identity_metrics")
    sys.modules["identity_metrics"] = types.SimpleNamespace(
        face_box=lambda path, pad=0.0: box,
        face_boxes=lambda path, pad=0.0, **kw: ([box] if box else []))
    return orig_mod

def _restore_idm(orig_mod):
    if orig_mod is not None: sys.modules["identity_metrics"] = orig_mod
    else: sys.modules.pop("identity_metrics", None)


# ---------------- edit_region_contained_cropped ----------------

def test_cropped_missing_src():
    ctx = Ctx()
    out = I.edit_region_contained_cropped(ctx, "Z:/no.png", "hat", "a red hat")
    check("cropped_missing_src_none", out is None)

def test_cropped_empty_args():
    ctx = Ctx()
    img = _photo("c1.png")
    check("cropped_empty_region_none", I.edit_region_contained_cropped(ctx, img, "", "x") is None)
    check("cropped_empty_prompt_none", I.edit_region_contained_cropped(ctx, img, "hat", "") is None)

def test_cropped_upload_fails():
    ctx = Ctx()
    img = _photo("c2.png", w=2000, h=2000)  # forces proxy path
    with _Patches(_upload_image_to_comfy=lambda *a, **k: None):
        out = I.edit_region_contained_cropped(ctx, img, "hat", "a red hat")
        check("cropped_upload_fail_none", out is None)

def test_cropped_no_mask():
    ctx = Ctx()
    img = _photo("c3.png")
    with _Patches(_upload_image_to_comfy=lambda *a, **k: "u.png",
                  _region_mask_file=lambda *a, **k: None):
        out = I.edit_region_contained_cropped(ctx, img, "hat", "a red hat")
        check("cropped_no_mask_none", out is None)

def test_cropped_stage0_qa_fail():
    ctx = Ctx()
    img = _photo("c4.png")
    scattered = _mask_file("c4_mask.png", box=(0, 0, 1, 1))  # near-empty
    with _Patches(_upload_image_to_comfy=lambda *a, **k: "u.png",
                  _region_mask_file=lambda *a, **k: scattered):
        out = I.edit_region_contained_cropped(ctx, img, "hat", "a red hat")
        check("cropped_stage0_qa_fail_none", out is None)

def test_cropped_face_protection_empties_mask():
    ctx = Ctx()
    img = _photo("c5.png", w=100, h=100)
    maskfile = _mask_file("c5_mask.png", w=100, h=100, box=(20, 20, 60, 60))
    orig_mod = _with_face_box((0, 0, 100, 100))  # covers everything -> mask emptied
    try:
        with _Patches(_upload_image_to_comfy=lambda *a, **k: "u.png",
                      _region_mask_file=lambda *a, **k: maskfile):
            out = I.edit_region_contained_cropped(ctx, img, "hairstyle", "blonde hair",
                                                   protect_face=True)
            check("cropped_face_protect_empties_none", out is None)
    finally:
        _restore_idm(orig_mod)

def test_cropped_facial_feature_skips_protection():
    ctx = Ctx()
    img = _photo("c6.png", w=150, h=150)
    # Small-item masks: the STAGE-0 gate caps a small worn/held region at 25%
    # of the frame, and a 70px box on a 150px image is 22% before the outward
    # dilation pushes it over. A hat/eye mask this large is exactly the
    # "Florence grabbed the whole person" failure the gate exists to catch.
    maskfile = _mask_file("c6_mask.png", w=150, h=150, box=(55, 55, 95, 95))
    with _Patches(_upload_image_to_comfy=lambda *a, **k: "u.png",
                  _region_mask_file=lambda *a, **k: maskfile,
                  _submit_and_poll=lambda *a, **k: _photo("c6_res.png", w=64, h=64)):
        out = I.edit_region_contained_cropped(ctx, img, "her eyes", "make them blue",
                                              protect_face=True)
        check("cropped_facial_feature_oldmodel_success", out is not None)

def test_cropped_degenerate_crop():
    ctx = Ctx()
    img = _photo("c7.png", w=10, h=10)
    maskfile = _mask_file("c7_mask.png", w=10, h=10, box=(4, 4, 5, 5))
    with _Patches(_upload_image_to_comfy=lambda *a, **k: "u.png",
                  _region_mask_file=lambda *a, **k: maskfile):
        out = I.edit_region_contained_cropped(ctx, img, "spot", "remove it")
        check("cropped_degenerate_none", out is None)

def test_cropped_oldmodel_tile_upload_fails():
    ctx = Ctx()
    img = _photo("c8.png", w=150, h=150)
    maskfile = _mask_file("c8_mask.png", w=150, h=150, box=(30, 30, 100, 100))
    calls = {"n": 0}
    def uploader(*a, **k):
        calls["n"] += 1
        return None if calls["n"] > 1 else "u.png"   # first call (source) ok, tile calls fail
    with _Patches(_upload_image_to_comfy=uploader,
                  _region_mask_file=lambda *a, **k: maskfile):
        out = I.edit_region_contained_cropped(ctx, img, "hat", "a red hat")
        check("cropped_tile_upload_fail_none", out is None)

def test_cropped_oldmodel_tile_editor_no_output():
    ctx = Ctx()
    img = _photo("c9.png", w=150, h=150)
    maskfile = _mask_file("c9_mask.png", w=150, h=150, box=(30, 30, 100, 100))
    with _Patches(_upload_image_to_comfy=lambda *a, **k: "u.png",
                  _region_mask_file=lambda *a, **k: maskfile,
                  _submit_and_poll=lambda *a, **k: None):
        out = I.edit_region_contained_cropped(ctx, img, "hat", "a red hat")
        check("cropped_tile_no_output_none", out is None)

def test_cropped_firered_engine_success():
    ctx = Ctx()
    img = _photo("c10.png", w=150, h=150)
    # Small-item masks: the STAGE-0 gate caps a small worn/held region at 25%
    # of the frame, and a 70px box on a 150px image is 22% before the outward
    # dilation pushes it over. A hat/eye mask this large is exactly the
    # "Florence grabbed the whole person" failure the gate exists to catch.
    maskfile = _mask_file("c10_mask.png", w=150, h=150, box=(55, 55, 95, 95))
    tile_out = _photo("c10_tile.png", w=64, h=64)
    with _Patches(_upload_image_to_comfy=lambda *a, **k: "u.png",
                  _region_mask_file=lambda *a, **k: maskfile,
                  edit_image_with_firered=lambda *a, **k: tile_out):
        out = I.edit_region_contained_cropped(ctx, img, "hat", "a red hat", engine="firered")
        check("cropped_firered_success", out is not None and Path(out).exists())

def test_cropped_firered_keeps_original_style():
    # Live, 2026-09-19: change_clothes on an already-stylized (anime/
    # illustrated) photo came back with a photorealistic garment. Root cause:
    # this call passed edit_prompt to FireRed bare, with none of the "match
    # the original style" guidance the whole-frame fallback
    # (image_grounding._firered_instruction) carries. The instruction FireRed
    # actually receives must now say so explicitly, on top of the user's own
    # requested change.
    ctx = Ctx()
    img = _photo("c10b.png", w=150, h=150)
    maskfile = _mask_file("c10b_mask.png", w=150, h=150, box=(55, 55, 95, 95))
    tile_out = _photo("c10b_tile.png", w=64, h=64)
    seen = {}
    def _capture_firered(ctx, src, instr, **kw):
        seen["instr"] = instr
        return tile_out
    with _Patches(_upload_image_to_comfy=lambda *a, **k: "u.png",
                  _region_mask_file=lambda *a, **k: maskfile,
                  edit_image_with_firered=_capture_firered):
        out = I.edit_region_contained_cropped(ctx, img, "hat", "a red hat", engine="firered")
        check("cropped_firered_style_success", out is not None and Path(out).exists())
        check("cropped_firered_keeps_user_request", "a red hat" in seen.get("instr", ""),
              seen.get("instr"))
        check("cropped_firered_adds_style_preservation_clause",
              "match" in seen.get("instr", "").lower()
              and "style" in seen.get("instr", "").lower(), seen.get("instr"))

def test_cropped_composite_exception():
    ctx = Ctx()
    img = _photo("c11.png", w=150, h=150)
    maskfile = _mask_file("c11_mask.png", w=150, h=150, box=(30, 30, 100, 100))
    with _Patches(_upload_image_to_comfy=lambda *a, **k: "u.png",
                  _region_mask_file=lambda *a, **k: maskfile,
                  _submit_and_poll=lambda *a, **k: "Z:/nonexistent_result.png"):
        # res_path returned but os.path.exists(res_path) False -> "produced no tile" branch
        out = I.edit_region_contained_cropped(ctx, img, "hat", "a red hat")
        check("cropped_result_missing_none", out is None)


# ---------------- _preserve_face_after_upscale ----------------

def test_preserve_face_no_face():
    orig_mod = _with_face_box(None)
    try:
        img = _photo("pf1.png"); up = _photo("pf1_up.png", w=400, h=400)
        out = I._preserve_face_after_upscale(img, up)
        check("preserve_face_no_face_unchanged", out == up)
    finally:
        _restore_idm(orig_mod)

def test_preserve_face_success():
    orig_mod = _with_face_box((40, 40, 120, 120))
    try:
        img = _photo("pf2.png", w=200, h=200); up = _photo("pf2_up.png", w=400, h=400)
        out = I._preserve_face_after_upscale(img, up)
        check("preserve_face_success", out != up and Path(out).exists())
    finally:
        _restore_idm(orig_mod)

def test_preserve_face_blend_partial():
    orig_mod = _with_face_box((40, 40, 120, 120))
    try:
        img = _photo("pf3.png", w=200, h=200); up = _photo("pf3_up.png", w=400, h=400)
        out = I._preserve_face_after_upscale(img, up, blend=0.5)
        check("preserve_face_blend_partial", out != up and Path(out).exists())
    finally:
        _restore_idm(orig_mod)

def test_preserve_face_import_fails():
    orig_mod = sys.modules.get("identity_metrics")
    sys.modules["identity_metrics"] = None  # forces ImportError on `import identity_metrics`
    try:
        img = _photo("pf4.png"); up = _photo("pf4_up.png")
        out = I._preserve_face_after_upscale(img, up)
        check("preserve_face_import_fail_unchanged", out == up)
    finally:
        _restore_idm(orig_mod)

def test_preserve_face_bad_orig_path():
    orig_mod = _with_face_box((10, 10, 50, 50))
    try:
        up = _photo("pf5_up.png")
        out = I._preserve_face_after_upscale("Z:/no_such.png", up)
        check("preserve_face_bad_orig_unchanged", out == up)
    finally:
        _restore_idm(orig_mod)


# ---------------- _composite_reference_head ----------------

def test_composite_head_no_face():
    orig_mod = _with_face_box(None)
    try:
        photo = _photo("ch1.png"); render = _photo("ch1_render.png")
        out = I._composite_reference_head(photo, render)
        check("composite_head_no_face_unchanged", out == render)
    finally:
        _restore_idm(orig_mod)

def test_composite_head_success():
    orig_mod = _with_face_box((30, 30, 90, 90))
    try:
        photo = _photo("ch2.png", w=150, h=150); render = _photo("ch2_render.png", w=150, h=150)
        out = I._composite_reference_head(photo, render)
        check("composite_head_success", out != render and Path(out).exists())
    finally:
        _restore_idm(orig_mod)

def test_composite_head_size_mismatch_resize():
    orig_mod = _with_face_box((20, 20, 60, 60))
    try:
        photo = _photo("ch3.png", w=150, h=150); render = _photo("ch3_render.png", w=80, h=80)
        out = I._composite_reference_head(photo, render)
        check("composite_head_resize", out != render and Path(out).exists())
    finally:
        _restore_idm(orig_mod)

def test_composite_head_bad_photo():
    orig_mod = _with_face_box((10, 10, 40, 40))
    try:
        render = _photo("ch4_render.png")
        out = I._composite_reference_head("Z:/no.png", render)
        check("composite_head_bad_photo_unchanged", out == render)
    finally:
        _restore_idm(orig_mod)


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

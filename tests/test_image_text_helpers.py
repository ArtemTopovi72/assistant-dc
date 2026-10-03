"""Coverage for image.py's small LLM-text-helper functions (_english_region,
_english_instructions, _ground_region_phrase, _image_region_inventory,
_extract_edit_target, _vlm_qa_verdict) and _face_detector_mask. Uses REAL LM Studio
for happy-path calls (fast text-only round trips) and stubs for exception/cache
branches. Requires LM Studio serving at config.LMSTUDIO_URL.
Run: venv/Scripts/python.exe tests/test_image_text_helpers.py
"""
import os, sys, tempfile, threading, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Hermetic: this suite is not a live-model test, but it was reaching
# localhost:1234 (see tests/offline_guard.py). Nothing here depends on
# the answers — the calls only made it slow and machine-dependent.
import offline_guard; offline_guard.offline_llm()
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
from pathlib import Path
from PIL import Image
import image as I
import models

_TMP = Path(tempfile.mkdtemp(prefix="imgtxt_"))
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

def _photo(name="p.png", w=64, h=64):
    p = _TMP / name
    Image.new("RGB", (w, h), (100, 120, 140)).save(p)
    return str(p)

# Use the shared broadcaster, not a private patcher that only touches `image`.
# image.py was split, so the functions under test now live in image_objects /
# image_grounding / ... and read THEIR module's globals. Setting the name on
# `image` alone left the real helper running -- eight checks here were failing
# against unpatched code, invisibly, because this suite could not fail.
# broadcast() also raises if a name binds nowhere, so a typo is loud.
from _image_patch import Patches as _Patches


# ---------------- _english_region ----------------

def test_english_region_empty():
    ctx = Ctx()
    check("eregion_empty_face", I._english_region(ctx, "") == "face")

def test_english_region_short_ascii_passthrough():
    ctx = Ctx()
    check("eregion_ascii_passthrough", I._english_region(ctx, "Hat") == "hat")

def test_english_region_translate_real_llm():
    ctx = Ctx()
    out = I._english_region(ctx, "платье")   # Russian "dress"
    check("eregion_translate_ran", isinstance(out, str) and len(out) > 0, out)

def test_english_region_llm_exception_fallback():
    ctx = Ctx()
    def boom(*a, **k): raise RuntimeError("lm down")
    with _Patches(call_llm_simple=boom):
        out = I._english_region(ctx, "платье красное сегодня")
        check("eregion_exception_fallback", isinstance(out, str))


# ---------------- _english_instructions ----------------

def test_english_instructions_ascii_passthrough():
    ctx = Ctx()
    check("einstr_ascii_passthrough", I._english_instructions(ctx, "make it red") == "make it red")

def test_english_instructions_empty():
    ctx = Ctx()
    check("einstr_empty", I._english_instructions(ctx, "") == "")

def test_english_instructions_translate_real_llm():
    ctx = Ctx()
    out = I._english_instructions(ctx, "сделай платье синим")
    check("einstr_translate_ran", isinstance(out, str), out)

def test_english_instructions_exception_fallback():
    ctx = Ctx()
    def boom(*a, **k): raise RuntimeError("lm down")
    with _Patches(call_llm_simple=boom):
        out = I._english_instructions(ctx, "сделай платье синим")
        check("einstr_exception_fallback_returns_original", out == "сделай платье синим")


# ---------------- _image_region_inventory / _ground_region_phrase ----------------

def test_inventory_missing_file():
    ctx = Ctx()
    check("inventory_missing_file_empty", I._image_region_inventory(ctx, "Z:/no.png") == [])

def test_inventory_stubbed_llm():
    ctx = Ctx()
    img = _photo("inv1.png")
    with _Patches(analyze_image_with_llm=lambda **k: "hat, shirt, background"):
        out = I._image_region_inventory(ctx, img)
        check("inventory_parses_list", out == ["hat", "shirt", "background"], out)

def test_inventory_cache_hit():
    ctx = Ctx()
    img = _photo("inv2.png")
    calls = {"n": 0}
    def fake_analyze(**k):
        calls["n"] += 1
        return "table, wall"
    with _Patches(analyze_image_with_llm=fake_analyze):
        I._image_region_inventory(ctx, img)
        I._image_region_inventory(ctx, img)  # second call should hit cache
        check("inventory_cache_hit_one_call", calls["n"] == 1, calls["n"])

def test_inventory_llm_exception():
    ctx = Ctx()
    img = _photo("inv3.png")
    def boom(**k): raise RuntimeError("down")
    with _Patches(analyze_image_with_llm=boom):
        out = I._image_region_inventory(ctx, img)
        check("inventory_exception_empty", out == [])

def test_ground_region_empty():
    ctx = Ctx()
    check("ground_empty_region", I._ground_region_phrase(ctx, "Z:/no.png", "") == "")

def test_ground_region_no_inventory_keeps_original():
    ctx = Ctx()
    img = _photo("gr1.png")
    with _Patches(_image_region_inventory=lambda ctx, p: []):
        out = I._ground_region_phrase(ctx, img, "the hat")
        check("ground_no_inventory_keeps", out == "the hat")

def test_ground_region_already_present():
    ctx = Ctx()
    img = _photo("gr2.png")
    with _Patches(_image_region_inventory=lambda ctx, p: ["hat", "shirt"]):
        out = I._ground_region_phrase(ctx, img, "hat")
        check("ground_already_present", out == "hat")

def test_ground_region_llm_maps_to_absent():
    ctx = Ctx()
    img = _photo("gr3.png")
    with _Patches(_image_region_inventory=lambda ctx, p: ["shirt", "background"],
                  call_llm_simple=lambda *a, **k: "NONE"):
        out = I._ground_region_phrase(ctx, img, "spaceship")
        check("ground_absent", out == "absent")

def test_ground_region_llm_maps_to_item():
    ctx = Ctx()
    img = _photo("gr4.png")
    with _Patches(_image_region_inventory=lambda ctx, p: ["shirt", "background"],
                  call_llm_simple=lambda *a, **k: "shirt"):
        out = I._ground_region_phrase(ctx, img, "clothing")
        check("ground_maps_to_item", out == "shirt")

def test_ground_region_llm_exception():
    ctx = Ctx()
    img = _photo("gr5.png")
    def boom(*a, **k): raise RuntimeError("down")
    with _Patches(_image_region_inventory=lambda ctx, p: ["shirt"], call_llm_simple=boom):
        out = I._ground_region_phrase(ctx, img, "clothing")
        check("ground_exception_keeps_original", out == "clothing")

def test_ground_region_llm_offlist_keeps_original():
    ctx = Ctx()
    img = _photo("gr6.png")
    with _Patches(_image_region_inventory=lambda ctx, p: ["shirt"], call_llm_simple=lambda *a, **k: "banana"):
        out = I._ground_region_phrase(ctx, img, "clothing")
        check("ground_offlist_keeps_original", out == "clothing")


# ---------------- _extract_edit_target ----------------

def test_extract_edit_target_empty():
    ctx = Ctx()
    r, res, dn = I._extract_edit_target(ctx, "")
    check("extract_empty", r is None and dn == 0.75)

def test_extract_edit_target_llm_exception():
    ctx = Ctx()
    def boom(*a, **k): raise RuntimeError("down")
    with _Patches(call_llm_simple=boom):
        r, res, dn = I._extract_edit_target(ctx, "change the hat")
        check("extract_exception_none_region", r is None)

def test_extract_edit_target_whole_image():
    ctx = Ctx()
    with _Patches(call_llm_simple=lambda *a, **k: '{"region": "whole", "result": "x", "scope": "tweak"}',
                  safe_json_from_llm=lambda resp: {"region": "whole", "result": "x", "scope": "tweak"}):
        r, res, dn = I._extract_edit_target(ctx, "enhance everything")
        check("extract_whole_none", r is None)

def test_extract_edit_target_absent():
    ctx = Ctx()
    with _Patches(call_llm_simple=lambda *a, **k: "x",
                  safe_json_from_llm=lambda resp: {"region": "absent", "result": "a hat", "scope": "tweak"}):
        r, res, dn = I._extract_edit_target(ctx, "add a hat")
        check("extract_absent", r == "absent" and res == "a hat")

def test_extract_edit_target_success_tweak():
    ctx = Ctx()
    with _Patches(call_llm_simple=lambda *a, **k: "x",
                  safe_json_from_llm=lambda resp: {"region": "hat", "result": "a red hat", "scope": "tweak"}):
        r, res, dn = I._extract_edit_target(ctx, "change the hat to red")
        check("extract_tweak_denoise", r == "hat" and dn == 0.6)

def test_extract_edit_target_success_replace():
    ctx = Ctx()
    with _Patches(call_llm_simple=lambda *a, **k: "x",
                  safe_json_from_llm=lambda resp: {"region": "hat", "result": "", "scope": "replace"}):
        r, res, dn = I._extract_edit_target(ctx, "change the hat entirely")
        check("extract_replace_denoise", r == "hat" and dn == 0.9 and res == "change the hat entirely")

def test_extract_edit_target_bad_json():
    ctx = Ctx()
    with _Patches(call_llm_simple=lambda *a, **k: "not json",
                  safe_json_from_llm=lambda resp: None):
        r, res, dn = I._extract_edit_target(ctx, "change the hat")
        check("extract_bad_json_none", r is None)


# ---------------- _vlm_qa_verdict ----------------

def test_vlm_qa_verdict_match():
    ctx = Ctx()
    with _Patches(analyze_image_with_llm=lambda **k: "CORRECT, looks good"):
        v = I._vlm_qa_verdict(ctx, _photo("qa1.png"), "is it correct?", ["CORRECT", "WRONG"])
        check("vlm_qa_match", v == "CORRECT", v)

def test_vlm_qa_verdict_no_match():
    ctx = Ctx()
    with _Patches(analyze_image_with_llm=lambda **k: "not sure, unclear"):
        v = I._vlm_qa_verdict(ctx, _photo("qa2.png"), "is it correct?", ["CORRECT", "WRONG"])
        check("vlm_qa_no_match_empty", v == "" or v is None, v)

def test_vlm_qa_verdict_exception():
    ctx = Ctx()
    def boom(**k): raise RuntimeError("down")
    with _Patches(analyze_image_with_llm=boom):
        v = I._vlm_qa_verdict(ctx, _photo("qa3.png"), "is it correct?", ["CORRECT", "WRONG"])
        check("vlm_qa_exception_falls_open", v == "" or v is None)


# ---------------- _face_detector_mask ----------------

def test_face_detector_mask_no_face():
    fake_idm = types.SimpleNamespace(face_box=lambda path, pad=0.0: None)
    orig = sys.modules.get("identity_metrics")
    sys.modules["identity_metrics"] = fake_idm
    try:
        out = I._face_detector_mask(_photo("fd1.png"), (64, 64))
        check("face_detector_no_face_none", out is None)
    finally:
        if orig is not None: sys.modules["identity_metrics"] = orig
        else: sys.modules.pop("identity_metrics", None)

def test_face_detector_mask_found():
    fake_idm = types.SimpleNamespace(face_box=lambda path, pad=0.0: (10, 10, 40, 40))
    orig = sys.modules.get("identity_metrics")
    sys.modules["identity_metrics"] = fake_idm
    try:
        out = I._face_detector_mask(_photo("fd2.png"), (64, 64))
        check("face_detector_found", out is not None and out.size == (64, 64))
    finally:
        if orig is not None: sys.modules["identity_metrics"] = orig
        else: sys.modules.pop("identity_metrics", None)

def test_face_detector_mask_unavailable():
    orig = sys.modules.get("identity_metrics")
    sys.modules["identity_metrics"] = None
    try:
        out = I._face_detector_mask(_photo("fd3.png"), (64, 64))
        check("face_detector_unavailable_none", out is None)
    finally:
        if orig is not None: sys.modules["identity_metrics"] = orig
        else: sys.modules.pop("identity_metrics", None)


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

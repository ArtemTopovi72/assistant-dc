"""Boundary-stubbed coverage for generate_image_with_refinement's eval loop: stubs
build_image_prompt_from_llm / generate_image_with_comfy / evaluate_image so every
branch (immediate success, first-attempt failure, failure-after-partial, patch
application, resolution clamping, attempt exhaustion) runs deterministically without
GPU or LM Studio.
Run: venv/Scripts/python.exe tests/test_image_refinement_stub.py
"""
import os, sys, tempfile, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
from pathlib import Path
from PIL import Image
import image as I

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

_TMP = Path(tempfile.mkdtemp(prefix="imgrefine_"))
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

def _photo(name="p.png"):
    p = _TMP / name
    Image.new("RGB", (64, 64), (10, 20, 30)).save(p)
    return str(p)


_BASE_PROMPT = ("a cat", 8, 1.0, 42, 960, 544)

def test_immediate_success():
    ctx = Ctx()
    img = _photo("s1.png")
    with _Patches(
                  generate_image_with_comfy=lambda **k: img,
                  evaluate_image=lambda *a, **k: {"verdict": "success", "score": 9, "reason": "great"}):
        out = I.generate_image_with_refinement(ctx, "a cat")
        check("immediate_success", out["status"] == "success" and out["path"] == img, out)

def test_score_ge_8_treated_as_success():
    ctx = Ctx()
    img = _photo("s2.png")
    with _Patches(
                  generate_image_with_comfy=lambda **k: img,
                  evaluate_image=lambda *a, **k: {"verdict": "refine", "score": 8, "reason": ""}):
        out = I.generate_image_with_refinement(ctx, "a cat")
        check("score_ge8_success", out["status"] == "success", out)

def test_first_attempt_gen_fails_no_prev():
    ctx = Ctx()
    with _Patches(
                  generate_image_with_comfy=lambda **k: None,
                  evaluate_image=lambda *a, **k: {"verdict": "refine", "score": 0}):
        out = I.generate_image_with_refinement(ctx, "a cat")
        check("first_fail_no_prev", out["status"] == "fail" and out["path"] is None, out)

def test_second_attempt_gen_fails_keeps_prev():
    ctx = Ctx()
    img1 = _photo("s3.png")
    calls = {"n": 0}
    def gen(**k):
        calls["n"] += 1
        return img1 if calls["n"] == 1 else None
    with _Patches(
                  generate_image_with_comfy=gen,
                  evaluate_image=lambda *a, **k: {"verdict": "refine", "score": 3,
                                                  "prompt_patch": "sharper", "negative_prompt_patch": "blurry"}):
        out = I.generate_image_with_refinement(ctx, "a cat")
        check("second_fail_keeps_prev_partial", out["status"] == "partial" and out["path"] == img1, out)

def test_refine_loop_applies_patches_then_succeeds(engine="ideogram4"):
    """A retry RE-DRAWS from the patched prompt; it never repaints the previous
    picture. Repainting an Ideogram composition would hand back a picture with
    no layout, so the boxes are gone and every later edit silently drops to a
    pixel pipeline. (The img2img-through-the old model variant was removed with
    the old model, 2026-09-23.)
    """
    ctx = Ctx()
    _prev_engine = I._config.IMAGE_ENGINE
    I._config.IMAGE_ENGINE = engine
    img_a = _photo(f"s4a_{engine}.png"); img_b = _photo(f"s4b_{engine}.png")
    calls = {"n": 0}
    def gen(**k):
        calls["n"] += 1
        if calls["n"] == 1:
            check("refine_first_no_inpaint", k.get("previous_image_path") is None)
            return img_a
        check("refine_second_redraws_for_ideogram",
              k.get("previous_image_path") is None,
              f"ideogram retry repainted through {k.get('previous_image_path')}")
        return img_b
    def ev(*a, **k):
        if calls["n"] == 1:
            # score > 5: a pure quality defect, so the retry refines THROUGH the
            # previous image. A low score with a prompt_patch means the subject is
            # missing, and that path deliberately regenerates from fresh noise
            # (covered separately below) — the old fixture used score 4 and so was
            # asserting img2img on the one branch that must not use it.
            return {"verdict": "refine", "score": 7, "prompt_patch": "add detail",
                     "negative_prompt_patch": "low quality", "steps": 12, "cfg": 2.0,
                     "width": 800, "height": 800}
        return {"verdict": "success", "score": 9}
    with _Patches(
                  generate_image_with_comfy=gen, evaluate_image=ev):
        out = I.generate_image_with_refinement(ctx, "a cat")
        check("refine_loop_eventually_succeeds", out["status"] == "success" and out["path"] == img_b, out)
    I._config.IMAGE_ENGINE = _prev_engine


def test_refine_missing_subject_regenerates_from_noise():
    """A low score WITH a prompt patch means the subject is absent. Refining through
    an image that lacks the subject does not add it — that retry must start fresh."""
    ctx = Ctx()
    img_a = _photo("s4c.png"); img_b = _photo("s4d.png")
    calls = {"n": 0}
    seen = {}
    def gen(**k):
        calls["n"] += 1
        if calls["n"] == 1:
            return img_a
        seen["prev"] = k.get("previous_image_path")
        return img_b
    def ev(*a, **k):
        if calls["n"] == 1:
            return {"verdict": "refine", "score": 4, "prompt_patch": "a cat is missing"}
        return {"verdict": "success", "score": 9}
    with _Patches(
                  generate_image_with_comfy=gen, evaluate_image=ev):
        I.generate_image_with_refinement(ctx, "a cat")
    check("refine_missing_subject_skips_inpaint", seen.get("prev") is None,
          f"previous_image_path={seen.get('prev')!r}")

def test_refine_resolution_clamp_over_5mp():
    ctx = Ctx()
    img_a = _photo("s5a.png"); img_b = _photo("s5b.png")
    calls = {"n": 0}
    def gen(**k):
        calls["n"] += 1
        return img_a if calls["n"] == 1 else img_b
    def ev(*a, **k):
        if calls["n"] == 1:
            # 3000x3000 = 9MP > 5MP cap -> triggers the clamp-scale branch
            return {"verdict": "refine", "score": 2, "width": 3000, "height": 3000}
        return {"verdict": "success", "score": 9}
    with _Patches(
                  generate_image_with_comfy=gen, evaluate_image=ev):
        out = I.generate_image_with_refinement(ctx, "a cat")
        check("refine_resolution_clamp_ran", out["status"] == "success", out)

def test_refine_exhausts_all_attempts():
    ctx = Ctx()
    img = _photo("s6.png")
    with _Patches(
                  generate_image_with_comfy=lambda **k: img,
                  evaluate_image=lambda *a, **k: {"verdict": "refine", "score": 3, "reason": "meh"}):
        out = I.generate_image_with_refinement(ctx, "a cat")
        check("refine_exhausted_partial", out["status"] == "partial"
              and out["attempts"] == I.MAX_IMAGE_REFINEMENT_ATTEMPTS, out)

def test_refine_user_overrides_width_height_seed():
    ctx = Ctx()
    img = _photo("s7.png")
    seen = {}
    def gen(**k):
        seen.update(k)
        return img
    with _Patches(
                  generate_image_with_comfy=gen,
                  evaluate_image=lambda *a, **k: {"verdict": "success", "score": 9}):
        out = I.generate_image_with_refinement(ctx, "a cat", steps=20, width=512, height=512, seed=777)
        check("refine_user_overrides", seen.get("steps") == 20 and seen.get("width") == 512
              and seen.get("seed") == 777, seen)


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

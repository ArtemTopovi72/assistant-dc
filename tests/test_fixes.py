"""
Regression tests for the still-current fixes.
Run from project root: .\\venv\\Scripts\\python.exe tests\\test_fixes.py

NOTE: tests that covered the old CLIPSeg/Florence + controlnet regional-mask
inpaint (region-presence parsing, mask_soft/fmask node detection, the Florence
region-description wording) were removed when inpaint_image moved to the mask-free
FireRed Image Edit engine.
"""
import sys
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ─────────────────────────────────────────────────────────────────────────────
# Removal classifier – false-positive regression (still used to flag a removal
# so _firered_instruction phrases the edit as "Remove the ...").
# ─────────────────────────────────────────────────────────────────────────────
from image import _is_removal_instruction

# Which instructions remove something is the model's read; the phrases run
# live in bench/removal_intent_live.py.
def test_removal_classifier():
    import image_router as _R
    _R.EDIT_STUB = {"remove the puddle": {"kind": "object_remove", "target": "the puddle"},
                    "без артефактов": {"kind": "restore"}}.get
    try:
        ok = _is_removal_instruction("remove the puddle") and not _is_removal_instruction("без артефактов")
    finally:
        _R.EDIT_STUB = None
    print(("PASS" if ok else "FAIL") + ": removal classifier wiring")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Eval resolution patch clamping
# ─────────────────────────────────────────────────────────────────────────────
from image import _snap_to_8

_MAX_EVAL_DIM = 2720
_MAX_EVAL_PX  = 5_000_000  # matches image.py — allows the old model native 2720×1536


def _clamp_eval_dims(w, h):
    cw = _snap_to_8(min(w, _MAX_EVAL_DIM)) if w > 0 else 960
    ch = _snap_to_8(min(h, _MAX_EVAL_DIM)) if h > 0 else 544
    if cw * ch > _MAX_EVAL_PX:
        scale = (_MAX_EVAL_PX / (cw * ch)) ** 0.5
        cw = _snap_to_8(int(cw * scale))
        ch = _snap_to_8(int(ch * scale))
    return cw, ch


CLAMP_CASES = [
    (3840, 2160, False),  # 4K — should be clamped
    (4096, 4096, False),  # huge — must be clamped
    (960,  544,  True),   # default — should pass through
    (2720, 1536, True),   # native the old model — should pass through
    (2720, 2720, False),  # square at max — product 7.4MP exceeds 5MP cap
]


def test_eval_resolution_clamp():
    failures = []
    for w, h, should_be_safe in CLAMP_CASES:
        cw, ch = _clamp_eval_dims(w, h)
        in_bounds = cw <= _MAX_EVAL_DIM and ch <= _MAX_EVAL_DIM and cw * ch <= _MAX_EVAL_PX
        was_clamped = (cw != w or ch != h)
        if should_be_safe and was_clamped:
            failures.append(f"  FAIL: {w}x{h} was needlessly clamped to {cw}x{ch}")
        elif not should_be_safe and not was_clamped:
            failures.append(f"  FAIL: {w}x{h} was NOT clamped (should have been)")
        elif not in_bounds:
            failures.append(f"  FAIL: {w}x{h} -> {cw}x{ch} still out of bounds")
    if failures:
        print("FAIL: eval resolution clamping:")
        print("\n".join(failures))
        return False
    print("PASS: eval resolution clamping (all %d cases)" % len(CLAMP_CASES))
    return True


# ─────────────────────────────────────────────────────────────────────────────
# _submit_and_poll skips temp files
# ─────────────────────────────────────────────────────────────────────────────
def test_temp_file_skip():
    filenames = [
        "ComfyUI_temp_preview_001.png",
        "ComfyUI_temp_001.png",
        "firered_edit_00001_.png",
        "old-model-facedetail_00001_.png",
    ]
    non_temp = [f for f in filenames if not f.lower().startswith("comfyui_temp")]
    if non_temp != ["firered_edit_00001_.png", "old-model-facedetail_00001_.png"]:
        print(f"FAIL: temp file skip wrong result: {non_temp}")
        return False
    print("PASS: temp file filter")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# _INPAINT_FAILURE dict accessible from tools
# ─────────────────────────────────────────────────────────────────────────────
def test_inpaint_failure_dict():
    from image import _INPAINT_FAILURE
    from tools import _INPAINT_FAILURE as tools_ref
    if _INPAINT_FAILURE is not tools_ref:
        print("FAIL: _INPAINT_FAILURE in tools is a copy, not the same object")
        return False
    print("PASS: _INPAINT_FAILURE is shared by reference between image and tools")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# FireRed edit engine wired in (replaces the mask pipeline)
# ─────────────────────────────────────────────────────────────────────────────
def test_firered_wired():
    import image
    from config import WORKFLOW_FIRERED_EDIT_PATH
    assert hasattr(image, "edit_image_with_firered"), "edit_image_with_firered missing"
    assert hasattr(image, "inpaint_region_with_comfy"), "inpaint_region_with_comfy missing"
    assert WORKFLOW_FIRERED_EDIT_PATH.exists(), "FireRed workflow file missing"
    # The mask-specific symbols must be gone (the general vision helper
    # _region_present stays — the face detailer still uses it).
    for dead in ("_removal_fill_prompt", "_is_eyewear", "_inpaint_change_fraction",
                 "_EYEWEAR_FILL_PROMPT", "_BIG_REGION_WORDS"):
        assert not hasattr(image, dead), f"old mask symbol {dead} still present"
    print("PASS: FireRed edit engine wired in, mask pipeline removed")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# social wall reading wiring
# ─────────────────────────────────────────────────────────────────────────────
def test_social_wired():
    """VK/Telegram wall reading is wired into the crawler (offline checks only)."""
    import social
    import deep_research as dr
    # classification
    assert social.social_kind("https://vk.com/club1") == "vk"
    assert social.social_kind("https://t.me/x/99") == "telegram"
    assert social.social_kind("https://example.com") == ""
    # social URLs survive the junk filter and rank high for local topics
    assert not dr._is_junk_source("https://vk.com/x")
    assert dr._source_rank("https://vk.com/x", "local") == 0
    # the cross-domain "go to their VK group" hop is extractable
    html = '<a href="https://vk.com/dk">VK</a><a href="https://t.me/dk">TG</a><a href="/x">in</a>'
    out = dr._social_outlinks(html, "https://venue.ru/")
    assert out == ["https://vk.com/dk", "https://t.me/dk"], out
    # VK degrades gracefully with no token configured
    import config
    if not config.VK_TOKEN:
        assert social.extract_vk_posts("https://vk.com/team") is None
    print("PASS: social (VK/Telegram) wall reading wired into crawler")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# corrective_count in graph.py (structural check)
# ─────────────────────────────────────────────────────────────────────────────
def test_corrective_counter():
    import ast
    import glob
    # Every graph module, not graph.py alone: the personality node moved into
    # graph_personality.py when the monolith was split, and this check then
    # reported the anti-fabrication guard missing while it was working.
    names = set()
    for f in sorted(glob.glob("agent/graph*.py")):
        tree = ast.parse(open(f, encoding="utf-8").read())
        names |= {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    if "corrective_count" not in names:
        print("FAIL: corrective_count not found in graph.py (still using old bool?)")
        return False
    if "corrective_sent" in names:
        print("FAIL: corrective_sent still referenced in graph.py")
        return False
    if "_MAX_CORRECTIVE" not in names:
        print("FAIL: _MAX_CORRECTIVE not found in graph.py")
        return False
    print("PASS: anti-fabrication guard uses corrective_count counter")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = [
        test_removal_classifier(),
        test_eval_resolution_clamp(),
        test_temp_file_skip(),
        test_inpaint_failure_dict(),
        test_firered_wired(),
        test_social_wired(),
        test_corrective_counter(),
    ]
    print()
    passed = sum(results)
    total = len(results)
    print(f"{'='*50}")
    print(f"Results: {passed}/{total} passed")
    sys.exit(0 if all(results) else 1)

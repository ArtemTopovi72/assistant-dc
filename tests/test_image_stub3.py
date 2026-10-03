"""Boundary-stubbed coverage round 3: edit_image_with_firered, generate_person_from_reference,
edit_region_contained_via_firered, _handfix_region_fallback, inpaint_region_with_comfy.
All stubbed at the ComfyUI/search/segmentation boundary -- no GPU or web calls.
Run: venv/Scripts/python.exe tests/test_image_stub3.py
"""
import os, sys, tempfile, threading, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
from pathlib import Path
from PIL import Image
import numpy as np
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

# _refine_edit_prompts was the SECOND live caller, and it was missed when the
# guard above was added: inpaint_region_with_comfy reaches it on every edit, so
# a single run of this "stub" suite made 15 calls to localhost:1234 and spent
# minutes retrying them. It already falls back to deterministic templates when
# the LLM is unavailable, so pinning it to exactly that fallback changes no
# assertion here — it only removes the machine-dependent detour.
def _refine_offline(ctx, region, instructions, *, removal=False):
    return (I._firered_instruction(ctx, region, instructions, removal),
            I._english_instructions(ctx, instructions) or instructions)
I._refine_edit_prompts = _refine_offline
import models

_TMP = Path(tempfile.mkdtemp(prefix="imgstub3_"))
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

def _photo(name="p.png", w=120, h=120, col=(120, 140, 160)):
    p = _TMP / name
    Image.new("RGB", (w, h), col).save(p)
    return str(p)

def _mask_file(name, w=120, h=120, box=None):
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



# ---------------- edit_image_with_firered ----------------

def test_firered_missing_src():
    ctx = Ctx()
    check("firered_missing_src_none", I.edit_image_with_firered(ctx, "Z:/no.png", "x") is None)

def test_firered_empty_instruction():
    ctx = Ctx()
    img = _photo("f1.png")
    check("firered_empty_instr_none", I.edit_image_with_firered(ctx, img, "   ") is None)

def test_firered_upload_fails():
    ctx = Ctx()
    img = _photo("f2.png")
    with _Patches(_upload_image_to_comfy=lambda *a, **k: None):
        check("firered_upload_fail_none", I.edit_image_with_firered(ctx, img, "add a hat") is None)

def test_firered_success_with_references():
    ctx = Ctx()
    img = _photo("f3.png")
    ref = _photo("f3_ref.png")
    calls = {"n": 0}
    def uploader(*a, **k):
        calls["n"] += 1
        return f"u{calls['n']}.png"
    with _Patches(_upload_image_to_comfy=uploader,
                  _submit_and_poll=lambda *a, **k: _photo("f3_out.png")):
        out = I.edit_image_with_firered(ctx, img, "add a hat", reference_paths=[ref])
        check("firered_success_refs", out is not None and Path(out).exists())

def test_firered_reference_upload_fails_skipped():
    ctx = Ctx()
    img = _photo("f4.png")
    calls = {"n": 0}
    def uploader(path, url):
        calls["n"] += 1
        return None if calls["n"] > 1 else "u1.png"   # source ok, ref fails
    with _Patches(_upload_image_to_comfy=uploader,
                  _submit_and_poll=lambda *a, **k: _photo("f4_out.png")):
        out = I.edit_image_with_firered(ctx, img, "add a hat", reference_paths=[img])
        check("firered_ref_upload_fail_skipped_still_succeeds", out is not None)

def test_firered_save_prefix_branch():
    ctx = Ctx()
    img = _photo("f5.png")
    with _Patches(_upload_image_to_comfy=lambda *a, **k: "u.png",
                  _submit_and_poll=lambda *a, **k: _photo("f5_out.png")):
        out = I.edit_image_with_firered(ctx, img, "add a hat", save_prefix="_INTERMEDIATE_tile")
        check("firered_save_prefix_ran", out is not None)

def test_firered_explicit_work_megapixels():
    ctx = Ctx()
    img = _photo("f6.png")
    with _Patches(_upload_image_to_comfy=lambda *a, **k: "u.png",
                  _submit_and_poll=lambda *a, **k: _photo("f6_out.png")):
        out = I.edit_image_with_firered(ctx, img, "add a hat", work_megapixels=0.8)
        check("firered_explicit_mp_ran", out is not None)


# ---------------- generate_person_from_reference ----------------

def test_gen_person_not_real():
    ctx = Ctx()
    with _Patches(_split_person_and_scene=lambda ctx, d: ("elf", False, "")):
        out, info = I.generate_person_from_reference(ctx, "an elf warrior")
        check("gen_person_not_real_none", out is None and info == "not_a_real_named_person")

def test_gen_person_no_photo():
    ctx = Ctx()
    fake_search = types.SimpleNamespace(
        fetch_reference_photo=lambda ctx, q, p: None,
        run_web_search=lambda ctx, q: "", NO_RESULTS="NO_RESULTS", SEARCH_FAILED="SEARCH_FAILED")
    orig = sys.modules.get("search")
    sys.modules["search"] = fake_search
    try:
        with _Patches(_split_person_and_scene=lambda ctx, d: ("Some Person", True, "at the beach")):
            out, reason = I.generate_person_from_reference(ctx, "Some Person at the beach")
            check("gen_person_no_photo_none", out is None and reason == "no_reference_photo")
    finally:
        if orig is not None: sys.modules["search"] = orig
        else: sys.modules.pop("search", None)

def test_gen_person_render_fails():
    ctx = Ctx()
    photo = _photo("gp1.png")
    fake_search = types.SimpleNamespace(
        fetch_reference_photo=lambda ctx, q, p: photo,
        run_web_search=lambda ctx, q: "brown hair, green eyes",
        NO_RESULTS="NO_RESULTS", SEARCH_FAILED="SEARCH_FAILED")
    orig = sys.modules.get("search")
    sys.modules["search"] = fake_search
    try:
        with _Patches(_split_person_and_scene=lambda ctx, d: ("Some Person", True, "at the beach"),
                      edit_image_with_firered=lambda *a, **k: None):
            out, reason = I.generate_person_from_reference(ctx, "Some Person at the beach")
            check("gen_person_render_fail_none", out is None and reason == "render_failed")
    finally:
        if orig is not None: sys.modules["search"] = orig
        else: sys.modules.pop("search", None)

def test_gen_person_success():
    ctx = Ctx()
    photo = _photo("gp2.png")
    render = _photo("gp2_render.png")
    fake_search = types.SimpleNamespace(
        fetch_reference_photo=lambda ctx, q, p: photo,
        run_web_search=lambda ctx, q: (_ for _ in ()).throw(RuntimeError("search down")),
        NO_RESULTS="NO_RESULTS", SEARCH_FAILED="SEARCH_FAILED")
    orig = sys.modules.get("search")
    sys.modules["search"] = fake_search
    try:
        with _Patches(_split_person_and_scene=lambda ctx, d: ("Some Person", True, ""),
                      edit_image_with_firered=lambda *a, **k: render,
                      _composite_reference_head=lambda p, r: r):
            out, info = I.generate_person_from_reference(ctx, "Some Person")
            check("gen_person_success", out == render and info["person"] == "Some Person", info)
            check("gen_person_appearance_search_exc_handled", info["appearance_used"] is False)
    finally:
        if orig is not None: sys.modules["search"] = orig
        else: sys.modules.pop("search", None)


# ---------------- edit_region_contained_via_firered ----------------

def test_contained_firered_missing_src():
    ctx = Ctx()
    out = I.edit_region_contained_via_firered(ctx, "Z:/no.png", "hand", "fix it")
    check("cfirered_missing_src_none", out is None)

def test_contained_firered_empty_instruction():
    ctx = Ctx()
    img = _photo("cf1.png")
    out = I.edit_region_contained_via_firered(ctx, img, "hand", "  ")
    check("cfirered_empty_instr_none", out is None)

def test_contained_firered_no_region_no_mask():
    ctx = Ctx()
    img = _photo("cf2.png")
    out = I.edit_region_contained_via_firered(ctx, img, "", "fix it")
    check("cfirered_no_region_none", out is None)

def test_contained_firered_mask_fails():
    ctx = Ctx()
    img = _photo("cf3.png")
    with _Patches(_contained_region_mask=lambda *a, **k: None):
        out = I.edit_region_contained_via_firered(ctx, img, "hand", "fix it")
        check("cfirered_mask_none", out is None)

def test_contained_firered_firered_no_tile():
    ctx = Ctx()
    img = _photo("cf4.png", w=100, h=100)
    orig = Image.open(img).convert("RGB")
    mask = Image.new("L", (100, 100), 0)
    px = mask.load()
    for y in range(30, 70):
        for x in range(30, 70): px[x, y] = 255
    with _Patches(_contained_region_mask=lambda *a, **k: (orig, mask, (30, 30, 70, 70)),
                  edit_image_with_firered=lambda *a, **k: None):
        out = I.edit_region_contained_via_firered(ctx, img, "hand", "fix it")
        check("cfirered_no_tile_none", out is None)

def test_contained_firered_success_no_qa():
    ctx = Ctx()
    img = _photo("cf5.png", w=100, h=100)
    orig = Image.open(img).convert("RGB")
    mask = Image.new("L", (100, 100), 0)
    px = mask.load()
    for y in range(30, 70):
        for x in range(30, 70): px[x, y] = 255
    # A DIFFERENT colour from the source: the pipeline rejects a no-op edit
    # (engine handed back the tile unchanged) so the caller can retry or fall
    # back. A tile identical to the crop therefore exercises the no-op guard,
    # not the success path this test is about.
    tile_out = _photo("cf5_tile.png", w=40, h=40, col=(220, 40, 40))
    with _Patches(_contained_region_mask=lambda *a, **k: (orig, mask, (30, 30, 70, 70)),
                  edit_image_with_firered=lambda *a, **k: tile_out,
                  MASK_QA_ENABLED=False):
        out = I.edit_region_contained_via_firered(ctx, img, "hand", "fix it")
        check("cfirered_success_no_qa", out is not None and Path(out).exists())

def test_contained_firered_stage2_qa_wrong_rejects():
    ctx = Ctx()
    img = _photo("cf6.png", w=100, h=100)
    orig = Image.open(img).convert("RGB")
    mask = Image.new("L", (100, 100), 0)
    px = mask.load()
    for y in range(30, 70):
        for x in range(30, 70): px[x, y] = 255
    tile_out = _photo("cf6_tile.png", w=40, h=40)
    with _Patches(_contained_region_mask=lambda *a, **k: (orig, mask, (30, 30, 70, 70)),
                  edit_image_with_firered=lambda *a, **k: tile_out,
                  MASK_QA_ENABLED=True,
                  _vlm_qa_verdict=lambda *a, **k: "WRONG"):
        out = I.edit_region_contained_via_firered(ctx, img, "hand", "fix it")
        check("cfirered_stage2_qa_wrong_none", out is None)

def test_contained_firered_stage2_qa_good_accepts():
    ctx = Ctx()
    img = _photo("cf7.png", w=100, h=100)
    orig = Image.open(img).convert("RGB")
    mask = Image.new("L", (100, 100), 0)
    px = mask.load()
    for y in range(30, 70):
        for x in range(30, 70): px[x, y] = 255
    # A DIFFERENT colour from the source: the pipeline rejects a no-op edit
    # (engine handed back the tile unchanged) so the caller can retry or fall
    # back. A tile identical to the crop therefore exercises the no-op guard,
    # not the success path this test is about.
    tile_out = _photo("cf7_tile.png", w=40, h=40, col=(220, 40, 40))
    with _Patches(_contained_region_mask=lambda *a, **k: (orig, mask, (30, 30, 70, 70)),
                  edit_image_with_firered=lambda *a, **k: tile_out,
                  MASK_QA_ENABLED=True,
                  _vlm_qa_verdict=lambda *a, **k: "GOOD"):
        out = I.edit_region_contained_via_firered(ctx, img, "hand", "fix it")
        check("cfirered_stage2_qa_good", out is not None)


def test_contained_firered_vlm_qa_off_skips_verdict():
    # Lettering removal measures its own fill; a WRONG from the VLM must not veto it.
    ctx = Ctx()
    img = _photo("cf8.png", w=100, h=100)
    orig = Image.open(img).convert("RGB")
    mask = Image.new("L", (100, 100), 0)
    px = mask.load()
    for y in range(30, 70):
        for x in range(30, 70): px[x, y] = 255
    tile_out = _photo("cf8_tile.png", w=40, h=40, col=(220, 40, 40))
    with _Patches(_contained_region_mask=lambda *a, **k: (orig, mask, (30, 30, 70, 70)),
                  edit_image_with_firered=lambda *a, **k: tile_out,
                  MASK_QA_ENABLED=True,
                  _vlm_qa_verdict=lambda *a, **k: "WRONG"):
        out = I.edit_region_contained_via_firered(ctx, img, "lettering", "Remove it",
                                                  vlm_qa=False)
        check("cfirered_vlm_qa_off_commits", out is not None)

def test_contained_firered_qa_reject_sets_reason():
    # A STAGE-2 reject must name itself, or image.inpaint falls through to the
    # whole-frame redraw that repaints the entire picture for a local edit.
    ctx = Ctx()
    img = _photo("cf9.png", w=100, h=100)
    orig = Image.open(img).convert("RGB")
    mask = Image.new("L", (100, 100), 0)
    px = mask.load()
    for y in range(30, 70):
        for x in range(30, 70): px[x, y] = 255
    tile_out = _photo("cf9_tile.png", w=40, h=40, col=(220, 40, 40))
    I._INPAINT_FAILURE.clear()
    with _Patches(_contained_region_mask=lambda *a, **k: (orig, mask, (30, 30, 70, 70)),
                  edit_image_with_firered=lambda *a, **k: tile_out,
                  MASK_QA_ENABLED=True,
                  _content_aware_alpha=lambda *a, **k: Image.new("L", (40, 40), 255),   # additive edit
                  _vlm_qa_verdict=lambda *a, **k: "WRONG"):
        out = I.edit_region_contained_via_firered(ctx, img, "the vase", "Make it blue")
        check("cfirered_qa_reject_none", out is None)
        check("cfirered_qa_reject_reason", I._INPAINT_FAILURE.get("reason") == "qa_rejected")
    # A whole-region swap (clothing/hair) is pasted through the mask, so the VLM's
    # WRONG is advisory there -- it rejected a clean dress swap 3x live.
    with _Patches(_contained_region_mask=lambda *a, **k: (orig, mask, (30, 30, 70, 70)),
                  edit_image_with_firered=lambda *a, **k: tile_out,
                  MASK_QA_ENABLED=True,
                  _content_aware_alpha=lambda *a, **k: None,
                  _vlm_qa_verdict=lambda *a, **k: "WRONG"):
        out = I.edit_region_contained_via_firered(ctx, img, "the dress", "Make it red")
        check("cfirered_whole_region_qa_advisory", out is not None)

# ---------------- _handfix_region_fallback ----------------

def test_handfix_fallback_upload_fails():
    ctx = Ctx()
    img = _photo("hf1.png")
    with _Patches(_upload_image_to_comfy=lambda *a, **k: None):
        out = I._handfix_region_fallback(ctx, img, None, seed=1, timeout=60, work_cap=768,
                                         pad_frac=0.25, denoise=0.8, grow=6, steps=20, cfg=6.0,
                                         pos="p", neg="n")
        check("handfix_fb_upload_fail_none", out is None)

def test_handfix_fallback_no_hand_segmented():
    ctx = Ctx()
    img = _photo("hf2.png")
    with _Patches(_upload_image_to_comfy=lambda *a, **k: "u.png",
                  _sam3_mask_file=lambda *a, **k: None,
                  _florence_mask_file=lambda *a, **k: None):
        out = I._handfix_region_fallback(ctx, img, None, seed=1, timeout=60, work_cap=768,
                                         pad_frac=0.25, denoise=0.8, grow=6, steps=20, cfg=6.0,
                                         pos="p", neg="n")
        check("handfix_fb_no_hand_none", out is None)

def test_handfix_fallback_nothing_missed():
    ctx = Ctx()
    img = _photo("hf3.png", w=100, h=100)
    hand_mask = _mask_file("hf3_hand.png", w=100, h=100, box=(40, 40, 60, 60))
    mg_mask = Image.new("L", (100, 100), 0)
    px = mg_mask.load()
    for y in range(30, 80):
        for x in range(30, 80): px[x, y] = 255  # covers the whole detected hand region
    with _Patches(_upload_image_to_comfy=lambda *a, **k: "u.png",
                  _sam3_mask_file=lambda *a, **k: hand_mask):
        out = I._handfix_region_fallback(ctx, img, mg_mask, seed=1, timeout=60, work_cap=768,
                                         pad_frac=0.25, denoise=0.8, grow=6, steps=20, cfg=6.0,
                                         pos="p", neg="n")
        check("handfix_fb_nothing_missed_none", out is None)

def test_handfix_fallback_success():
    ctx = Ctx()
    img = _photo("hf4.png", w=100, h=100)
    hand_mask = _mask_file("hf4_hand.png", w=100, h=100, box=(10, 10, 90, 90))
    repaired = _photo("hf4_repaired.png")
    with _Patches(_upload_image_to_comfy=lambda *a, **k: "u.png",
                  _sam3_mask_file=lambda *a, **k: hand_mask,
                  edit_region_contained_via_firered=lambda *a, **k: repaired):
        out = I._handfix_region_fallback(ctx, img, None, seed=1, timeout=60, work_cap=768,
                                         pad_frac=0.25, denoise=0.8, grow=6, steps=20, cfg=6.0,
                                         pos="p", neg="n")
        check("handfix_fb_success", out == repaired)

def test_handfix_fallback_florence_used_when_sam3_none():
    ctx = Ctx()
    img = _photo("hf5.png", w=100, h=100)
    hand_mask = _mask_file("hf5_hand.png", w=100, h=100, box=(10, 10, 90, 90))
    with _Patches(_upload_image_to_comfy=lambda *a, **k: "u.png",
                  _sam3_mask_file=lambda *a, **k: None,
                  _florence_mask_file=lambda *a, **k: hand_mask,
                  edit_region_contained_via_firered=lambda *a, **k: None):
        out = I._handfix_region_fallback(ctx, img, None, seed=1, timeout=60, work_cap=768,
                                         pad_frac=0.25, denoise=0.8, grow=6, steps=20, cfg=6.0,
                                         pos="p", neg="n")
        check("handfix_fb_florence_used_repair_fail_none", out is None)


# ---------------- inpaint_region_with_comfy ----------------

def test_inpaint_missing_src():
    ctx = Ctx()
    out = I.inpaint_region_with_comfy(ctx, "Z:/no.png", "the hat", "make it red")
    check("inpaint_missing_src_none", out is None)

def test_inpaint_region_absent():
    ctx = Ctx()
    img = _photo("ip1.png")
    with _Patches(_english_region=lambda ctx, r: "the hat",
                  _ground_region_phrase=lambda ctx, path, r: "absent"):
        out = I.inpaint_region_with_comfy(ctx, img, "the hat", "make it red")
        check("inpaint_region_absent_none", out is None)

def test_inpaint_firered_contained_success():
    ctx = Ctx()
    img = _photo("ip2.png")
    result = _photo("ip2_out.png")
    with _Patches(_english_region=lambda ctx, r: "the hat",
                  _ground_region_phrase=lambda ctx, path, r: "the hat",
                  _firered_instruction=lambda ctx, r, p, removal=False: "make it red",
                  edit_region_contained_via_firered=lambda *a, **k: result):
        out = I.inpaint_region_with_comfy(ctx, img, "the hat", "make it red")
        check("inpaint_firered_success", out == result)

def test_inpaint_firered_failure_is_not_substituted():
    """FireRed is the only edit engine: when it fails there is no second engine
    to quietly produce a different picture (the old model fallback was removed)."""
    ctx = Ctx()
    img = _photo("ip3.png")
    other = _photo("ip3_out.png")
    with _Patches(_english_region=lambda ctx, r: "the hat",
                  _ground_region_phrase=lambda ctx, path, r: "the hat",
                  _firered_instruction=lambda ctx, r, p, removal=False: "make it red",
                  edit_region_contained_via_firered=lambda *a, **k: None,
                  _english_instructions=lambda ctx, p: "make it red",
                  _contained_edit_validated=lambda *a, **k: other,
                  # the whole-frame fallback must fail too, and must not reach the
                  # LIVE ComfyUI: unpatched it rendered a real picture, and the
                  # check flipped with whatever the server returned (2026-09-24)
                  edit_image_with_firered=lambda *a, **k: None):
        out = I.inpaint_region_with_comfy(ctx, img, "the hat", "make it red")
        check("inpaint_firered_failure_returns_none", out is None)

def test_inpaint_falls_back_to_whole_frame_firered():
    ctx = Ctx()
    img = _photo("ip4.png")
    whole_out = _photo("ip4_whole.png")
    with _Patches(_english_region=lambda ctx, r: "the hat",
                  _ground_region_phrase=lambda ctx, path, r: "the hat",
                  _firered_instruction=lambda ctx, r, p, removal=False: "make it red",
                  edit_region_contained_via_firered=lambda *a, **k: None,
                  _english_instructions=lambda ctx, p: "make it red",
                  _contained_edit_validated=lambda *a, **k: None,
                  edit_image_with_firered=lambda *a, **k: whole_out,
                  preserve_identity_face=lambda *a, **k: whole_out):
        out = I.inpaint_region_with_comfy(ctx, img, "the hat", "make it red")
        check("inpaint_whole_frame_fallback_success", out == whole_out)

def test_inpaint_whole_frame_missing_workflow():
    ctx = Ctx()
    img = _photo("ip5.png")
    from pathlib import Path as P
    with _Patches(_english_region=lambda ctx, r: "the hat",
                  _ground_region_phrase=lambda ctx, path, r: "the hat",
                  _firered_instruction=lambda ctx, r, p, removal=False: "make it red",
                  edit_region_contained_via_firered=lambda *a, **k: None,
                  _english_instructions=lambda ctx, p: "make it red",
                  _contained_edit_validated=lambda *a, **k: None,
                  WORKFLOW_FIRERED_EDIT_PATH=P("Z:/no_workflow.json")):
        out = I.inpaint_region_with_comfy(ctx, img, "the hat", "make it red")
        check("inpaint_missing_workflow_none", out is None)

def test_inpaint_removal_skips_contained_path():
    ctx = Ctx()
    img = _photo("ip6.png")
    whole_out = _photo("ip6_whole.png")
    with _Patches(_english_region=lambda ctx, r: "",
                  _firered_instruction=lambda ctx, r, p, removal=False: "remove it",
                  edit_image_with_firered=lambda *a, **k: whole_out,
                  preserve_identity_face=lambda *a, **k: whole_out):
        out = I.inpaint_region_with_comfy(ctx, img, "the hat", "remove it", removal=True)
        check("inpaint_removal_whole_frame", out == whole_out)

def test_inpaint_whole_region_skips_contained_path():
    ctx = Ctx()
    img = _photo("ip7.png")
    whole_out = _photo("ip7_whole.png")
    with _Patches(_english_region=lambda ctx, r: "whole",
                  _firered_instruction=lambda ctx, r, p, removal=False: "enhance it",
                  edit_image_with_firered=lambda *a, **k: whole_out,
                  preserve_identity_face=lambda *a, **k: whole_out):
        out = I.inpaint_region_with_comfy(ctx, img, "the whole image", "enhance it")
        check("inpaint_whole_region_skip", out == whole_out)

def test_inpaint_whole_frame_face_edit_skips_identity_preserve():
    ctx = Ctx()
    img = _photo("ip8.png")
    whole_out = _photo("ip8_whole.png")
    called = {"n": 0}
    def fake_preserve(*a, **k):
        called["n"] += 1
        return whole_out
    with _Patches(_english_region=lambda ctx, r: "",
                  _firered_instruction=lambda ctx, r, p, removal=False: "change her face",
                  edit_image_with_firered=lambda *a, **k: whole_out,
                  preserve_identity_face=fake_preserve,
                  _is_facial=lambda ctx, p: "face" in p):   # the read itself: bench/removal_intent_live.py
        out = I.inpaint_region_with_comfy(ctx, img, "her face", "change her face", removal=True)
        check("inpaint_face_edit_skips_identity_preserve", called["n"] == 0 and out == whole_out)


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

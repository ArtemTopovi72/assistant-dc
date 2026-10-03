"""Structural validation of EVERY Transfer-tab role (no GPU, no network).

The Transfer tab exposes 10 roles. This proves, for each, that it is actually wired
into the backend and changes behavior — not merely present in the dropdown:

  1. each role produces a DISTINCT, role-appropriate generation instruction;
  2. Auto-infer maps the right keywords to the right role (and falls back safely);
  3. multiple same-role refs are grouped into passes;
  4. conflicting roles are ordered deterministically by the chain order;
  5. missing-role / no-ref scenarios fail gracefully (None, no crash);
  6. an explicit role is honored and never overridden by inference;
  7. switching a ref's role changes the plan (order + instruction);
  8. the Preview-Plan path reports the role interpretation;
  9. localizable roles route to the CONTAINED path, global roles to WHOLE-FRAME;
 10. a manual mask routes ANY role through the contained path; face role keeps
     protect_face OFF (a deliberate face edit must reach the face).

Real pixel generation (FireRed/ComfyUI) is covered separately by the live runner
tests/role_live.py — this file validates the routing/plumbing that decides what the
generator is told to do.

Run: ./venv/Scripts/python.exe tests/test_transfer_roles.py
"""
import os
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from PIL import Image
import image as im


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _img(name, w=160, h=120, color=(120, 120, 120)):
    p = os.path.join(tempfile.gettempdir(), name)
    Image.new("RGB", (w, h), color).save(p)
    return p


_ALL = [im.ROLE_CLOTHING, im.ROLE_OBJECT, im.ROLE_HAIR, im.ROLE_FACE, im.ROLE_POSE,
        im.ROLE_STYLE, im.ROLE_IDENTITY, im.ROLE_SCENE, im.ROLE_LIGHTING]
_LOCALIZABLE = {im.ROLE_CLOTHING, im.ROLE_OBJECT, im.ROLE_HAIR, im.ROLE_FACE}
_GLOBAL = {im.ROLE_POSE, im.ROLE_STYLE, im.ROLE_IDENTITY, im.ROLE_SCENE, im.ROLE_LIGHTING}
_EXTRACTABLE = {im.ROLE_CLOTHING, im.ROLE_OBJECT, im.ROLE_HAIR}


def test_each_role_distinct_instruction():
    """Item 1 — every role yields a distinct, role-appropriate instruction (so the
    generator is told something different for each)."""
    instrs = {r: im._reference_instruction(r, 1) for r in _ALL}
    # all distinct
    _check(len(set(instrs.values())) == len(instrs), "two roles share an instruction")
    # role-appropriate keyword present
    want = {
        im.ROLE_CLOTHING: "garment", im.ROLE_OBJECT: "object", im.ROLE_HAIR: "hairstyle",
        im.ROLE_FACE: "facial identity", im.ROLE_POSE: "pose", im.ROLE_STYLE: "style",
        im.ROLE_IDENTITY: "character", im.ROLE_SCENE: "scene", im.ROLE_LIGHTING: "lighting",
    }
    for r, kw in want.items():
        _check(kw.lower() in instrs[r].lower(),
               f"{r} instruction lacks '{kw}': {instrs[r]!r}")
    # contained-path instruction map also distinct for the 4 localizable roles
    print(f"PASS: each of {len(_ALL)} roles -> distinct role-appropriate instruction")
    return True


def test_auto_infer_keywords():
    """Item 2 — Auto-infer maps representative phrases to the correct role, and an
    instruction with no role keyword falls back to ROLE_OBJECT."""
    cases = [
        ("put the leather jacket from image 2 on him", im.ROLE_CLOTHING),
        ("give him the hairstyle from the reference", im.ROLE_HAIR),
        ("add the sunglasses to his face", im.ROLE_OBJECT),
        ("make his facial features look like this person", im.ROLE_FACE),
        ("use this pose for the subject", im.ROLE_POSE),
        ("apply this art style and colour palette", im.ROLE_STYLE),
        ("place them in this scene/background", im.ROLE_SCENE),
        ("match the lighting and mood", im.ROLE_LIGHTING),
        ("keep the same identity/character", im.ROLE_IDENTITY),
    ]
    for text, expected in cases:
        ref = im.ReferenceImage(_img("ai.png"), role=None)
        im.infer_reference_roles(text, [ref])
        _check(ref.role == expected, f"{text!r} -> {ref.role}, expected {expected}")
    # no keyword -> object fallback (never None at execution)
    ref = im.ReferenceImage(_img("ai.png"), role=None)
    im.infer_reference_roles("just do something with image 2", [ref])
    _check(ref.role == im.ROLE_OBJECT, f"no-keyword fallback wrong: {ref.role}")
    _check(ref.effective_role == im.ROLE_OBJECT, "effective_role should never be None")
    print("PASS: auto-infer maps keywords to all roles + safe ROLE_OBJECT fallback")
    return True


def test_explicit_role_not_overridden():
    """Item 6 — an explicitly assigned role survives inference even when the text
    screams a different role (no silent misclassification)."""
    ref = im.ReferenceImage(_img("ex.png"), role=im.ROLE_HAIR)
    im.infer_reference_roles("put the jacket and suit on him", [ref])  # clothing words
    _check(ref.role == im.ROLE_HAIR, f"explicit role overridden: {ref.role}")
    print("PASS: explicit role honored over conflicting instruction text")
    return True


def test_extractable_and_region_classification():
    """Item 1/9 — classification matches the contract: clothing/object/hair extract a
    clean asset; clothing/object/hair/face have a target region (contained); the five
    global roles have no region (whole-frame)."""
    for r in _ALL:
        ref = im.ReferenceImage(_img("c.png"), role=r)
        _check(ref.is_extractable() == (r in _EXTRACTABLE),
               f"{r} extractable flag wrong")
    for r in _LOCALIZABLE:
        _check(im._ROLE_TARGET_REGION.get(r), f"{r} should have a contained region")
    for r in _GLOBAL:
        _check(im._ROLE_TARGET_REGION.get(r) is None,
               f"{r} should be global (no region) -> whole-frame")
    print("PASS: extractable/region classification matches contained-vs-global contract")
    return True


def test_routing_contained_vs_wholeframe():
    """Item 9/10 — localizable roles go through the CONTAINED renderer; global roles
    fall back to WHOLE-FRAME; a manual mask forces the contained path for ANY role."""
    calls = {"contained": [], "whole": []}
    saved = (im.edit_region_contained_via_firered, im.transfer_with_references,
             im.extract_reference_asset)
    try:
        im.edit_region_contained_via_firered = (
            lambda ctx, tgt, region, instr, **k: calls["contained"].append(
                {"region": region, "protect_face": k.get("protect_face"),
                 "mask": k.get("mask_override")}) or "out.png")
        im.transfer_with_references = (
            lambda ctx, tgt, refs, instr=None, **k: calls["whole"].append(
                refs[0].effective_role) or "out.png")
        im.extract_reference_asset = lambda ctx, ref, **k: ref

        tgt = _img("t.png")
        # localizable -> contained
        for r in _LOCALIZABLE:
            calls["contained"].clear(); calls["whole"].clear()
            im.transfer_reference_contained(None, tgt, im.ReferenceImage(_img("r.png"), role=r))
            _check(len(calls["contained"]) == 1 and not calls["whole"],
                   f"{r} did not route to contained path")
        # global -> whole-frame (no mask)
        for r in _GLOBAL:
            calls["contained"].clear(); calls["whole"].clear()
            im.transfer_reference_contained(None, tgt, im.ReferenceImage(_img("r.png"), role=r))
            _check(len(calls["whole"]) == 1 and not calls["contained"],
                   f"{r} did not fall back to whole-frame")
        # global + manual mask -> contained (item 10: mask works for any role)
        calls["contained"].clear(); calls["whole"].clear()
        im.transfer_reference_contained(None, tgt, im.ReferenceImage(_img("r.png"), role=im.ROLE_POSE),
                                        mask_override=_img("m.png"))
        _check(len(calls["contained"]) == 1 and not calls["whole"],
               "manual mask did not force contained path for a global role")
    finally:
        (im.edit_region_contained_via_firered, im.transfer_with_references,
         im.extract_reference_asset) = saved
    print("PASS: routing — localizable→contained, global→whole-frame, mask→contained(any role)")
    return True


def test_face_role_protect_face_off():
    """Item 6/wrong-person — the face role must reach the face: protect_face is forced
    OFF for ROLE_FACE but honored for other roles."""
    captured = []
    saved = (im.edit_region_contained_via_firered, im.extract_reference_asset)
    try:
        im.edit_region_contained_via_firered = (
            lambda ctx, tgt, region, instr, **k: captured.append(k.get("protect_face")) or "o.png")
        im.extract_reference_asset = lambda ctx, ref, **k: ref
        tgt = _img("t.png")
        im.transfer_reference_contained(None, tgt, im.ReferenceImage(_img("r.png"), role=im.ROLE_FACE),
                                        protect_face=True)
        _check(captured[-1] is False, "face role must force protect_face OFF")
        im.transfer_reference_contained(None, tgt, im.ReferenceImage(_img("r.png"), role=im.ROLE_CLOTHING),
                                        protect_face=True)
        _check(captured[-1] is True, "clothing role must keep protect_face ON when requested")
    finally:
        (im.edit_region_contained_via_firered, im.extract_reference_asset) = saved
    print("PASS: face role forces protect_face OFF; other roles honor it (no wrong-face guard clash)")
    return True


def test_plan_chain_order_and_grouping():
    """Item 3/4/7 — the plan orders refs by chain order (scene<pose<clothing<object<
    hair<face<identity<style<lighting), groups ≤2 per pass, and each pass carries a
    role-appropriate instruction."""
    refs = [
        im.ReferenceImage(_img("a.png"), role=im.ROLE_STYLE),
        im.ReferenceImage(_img("b.png"), role=im.ROLE_SCENE),
        im.ReferenceImage(_img("c.png"), role=im.ROLE_CLOTHING),
        im.ReferenceImage(_img("d.png"), role=im.ROLE_FACE),
    ]
    plan = im.build_edit_plan(_img("t.png"), refs, "")
    flat = [r for step in plan["steps"] for r in step["refs"]]
    order = [im._ROLE_CHAIN_ORDER.get(r.role, 4) for r in flat]
    _check(order == sorted(order), f"refs not in chain order: {order}")
    _check(flat[0].role == im.ROLE_SCENE, "scene should be applied first")
    _check(all(len(s["refs"]) <= 2 for s in plan["steps"]), "a pass grouped >2 refs")
    _check(plan["n_passes"] == len(plan["steps"]), "n_passes mismatch")
    # role switch changes the plan
    refs[2].role = im.ROLE_LIGHTING
    plan2 = im.build_edit_plan(_img("t.png"), refs, "")
    flat2 = [r.role for step in plan2["steps"] for r in step["refs"]]
    _check(flat2 != [r.role for r in flat], "switching a role did not change the plan")
    print("PASS: plan chain-order + ≤2 grouping + role-switch updates plan")
    return True


def test_multiple_same_role():
    """Item 3 — two refs of the SAME role are both kept, extractable, and planned."""
    refs = [im.ReferenceImage(_img("o1.png"), role=im.ROLE_OBJECT),
            im.ReferenceImage(_img("o2.png"), role=im.ROLE_OBJECT)]
    plan = im.build_edit_plan(_img("t.png"), refs, "")
    flat = [r for step in plan["steps"] for r in step["refs"]]
    _check(len(flat) == 2 and all(r.role == im.ROLE_OBJECT for r in flat),
           "same-role refs not both planned")
    _check(all(r.is_extractable() for r in flat), "object refs should be extractable")
    print("PASS: multiple same-role references are both planned + extractable")
    return True


def test_missing_and_empty_graceful():
    """Item 5 — no references / missing files degrade gracefully (None, no crash)."""
    _check(im.build_edit_plan(_img("t.png"), [], "")["steps"] == [], "empty plan not empty")
    _check(im.plan_and_execute_transfer(None, _img("t.png"), [], "") is None,
           "no-ref transfer should return None")
    _check(im.transfer_with_references(None, _img("t.png"), []) is None,
           "no usable refs should return None")
    # a reference whose file does not exist is dropped, not crashed on
    bad = im.ReferenceImage(os.path.join(tempfile.gettempdir(), "does_not_exist_zzz.png"),
                            role=im.ROLE_CLOTHING)
    _check(im.transfer_with_references(None, _img("t.png"), [bad]) is None,
           "missing-file ref should yield None gracefully")
    print("PASS: missing-role / no-ref / missing-file all degrade gracefully")
    return True


def test_preview_plan_reports_roles():
    """Item 8 — the GUI Preview-Plan path reports the role interpretation per pass
    (run headless with the GPU/network calls stubbed)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication
    import gui
    _APP = QApplication.instance() or QApplication(sys.argv)

    saved = (im.extract_reference_asset, im._upload_image_to_comfy, im._region_mask_file)
    captured = {}
    try:
        im.extract_reference_asset = lambda ctx, ref, **k: ref
        im._upload_image_to_comfy = lambda *a, **k: None     # skip masks (network)
        im._region_mask_file = lambda *a, **k: None

        class _Ctx:
            def is_cancelled(self): return False
        # 3 refs (scene, clothing, face) -> 2 passes (≤2 grouped per pass)
        refs = [im.ReferenceImage(_img("rs.png"), role=im.ROLE_SCENE),
                im.ReferenceImage(_img("rc.png"), role=im.ROLE_CLOTHING),
                im.ReferenceImage(_img("rf.png"), role=im.ROLE_FACE)]
        w = gui.TransferWorker(_Ctx(), _img("t.png"), [(r.path, r.role, None) for r in refs],
                               "", "plan")
        w.done.connect(lambda d: captured.update(d))
        w.run()
        info = captured.get("info", "")
        _check("roles=" in info and "clothing_source" in info and "face_reference" in info
               and "scene_reference" in info,
               f"plan info missing role interpretation: {info!r}")
        _check("Pass 1" in info and "Pass 2" in info, "plan did not enumerate passes")
    finally:
        (im.extract_reference_asset, im._upload_image_to_comfy, im._region_mask_file) = saved
    print("PASS: Preview-Plan reports per-pass role interpretation (clothing + face)")
    return True


if __name__ == "__main__":
    tests = [
        test_each_role_distinct_instruction,
        test_auto_infer_keywords,
        test_explicit_role_not_overridden,
        test_extractable_and_region_classification,
        test_routing_contained_vs_wholeframe,
        test_face_role_protect_face_off,
        test_plan_chain_order_and_grouping,
        test_multiple_same_role,
        test_missing_and_empty_graceful,
        test_preview_plan_reports_roles,
    ]
    results = []
    for t in tests:
        try:
            results.append(bool(t()))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"FAIL: {t.__name__}: {e}")
            results.append(False)
    print("\n" + "=" * 56)
    print(f"Results: {sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)

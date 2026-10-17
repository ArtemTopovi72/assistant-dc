"""A clothing transfer names the garment's details (logos, patches) to FireRed in
its own "image 2" wording; the agent's "reference image" phrasing is rewritten."""
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "agent", "imaging"):
    sys.path.insert(0, os.path.join(ROOT, d))
os.environ.setdefault("F5_TEST_RUN", "1")

import image_transfer as T  # noqa: E402


def test_clothing_instruction_carries_garment_details(tmp_path):
    """A navy quarter-zip with a red monogram and a flag patch must not come out a
    plain navy jumper: FireRed is told about the monogram and the flag."""
    target = tmp_path / "person.jpg"; target.write_bytes(b"P" * 50)
    garment = tmp_path / "hoodie.jpg"; garment.write_bytes(b"H" * 50)
    T._GARMENT_CACHE.clear()
    seen = {}
    def fake_edit(ctx, tp, region, instr, **kw):
        seen["instr"] = instr
        return None
    ctx = types.SimpleNamespace(deps=types.SimpleNamespace(
        _garment_details=lambda ctx, p: "navy quarter-zip, red JCE monogram on left chest, "
                                        "US flag patch on left sleeve",
        edit_region_contained_via_firered=fake_edit, transfer_with_references=lambda *a, **k: None))
    ref = T.ReferenceImage(str(garment), T.ROLE_CLOTHING)
    ref.extracted_asset_path = str(garment)
    T.transfer_reference_contained(
        ctx, str(target), ref,
        "Dress the person in the target image in the clothing shown in the reference image.")
    instr = seen.get("instr", "")
    assert "red JCE monogram" in instr and "flag patch" in instr, instr
    assert "image 2" in instr and "reference image" not in instr, instr
    assert "target image" not in instr, instr


def test_garment_details_empty_when_vision_fails(tmp_path):
    p = tmp_path / "g.jpg"; p.write_bytes(b"G" * 10)
    T._GARMENT_CACHE.clear()
    def boom(**kw):
        raise RuntimeError("vision down")
    down = types.SimpleNamespace(vision_model=boom)
    assert T._garment_details(down, str(p)) == ""
    assert T._garment_details(down, str(tmp_path / "missing.jpg")) == ""


def test_unmasked_fallback_that_adds_a_person_is_not_delivered(tmp_path):
    """The garment's mask failed and the whole-frame pass stood the reference's wearer
    (the girl in red armour) in front of the man it was to dress."""
    target = tmp_path / "man.jpg"; target.write_bytes(b"P" * 50)
    garment = tmp_path / "armor.jpg"; garment.write_bytes(b"A" * 50)
    pasted = tmp_path / "out.png"; pasted.write_bytes(b"O" * 50)
    T._GARMENT_CACHE.clear()
    for adds, want in ((True, None), (False, str(pasted))):
        ctx = types.SimpleNamespace(deps=types.SimpleNamespace(
            _garment_details=lambda ctx, p: "",
            edit_region_contained_via_firered=lambda *a, **k: None,
            transfer_with_references=lambda *a, **k: str(pasted),
            _no_new_faces=lambda src, out, _a=adds: not _a))
        ref = T.ReferenceImage(str(garment), T.ROLE_CLOTHING)
        ref.extracted_asset_path = str(garment)
        assert T.transfer_reference_contained(ctx, str(target), ref, "dress him in it") == want

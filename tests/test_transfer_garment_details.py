"""A clothing transfer names the garment's details (logos, patches) to FireRed in
its own "image 2" wording; the agent's "reference image" phrasing is rewritten."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "agent", "imaging"):
    sys.path.insert(0, os.path.join(ROOT, d))
os.environ.setdefault("F5_TEST_RUN", "1")

import image_transfer as T  # noqa: E402


def test_clothing_instruction_carries_garment_details(tmp_path, monkeypatch):
    """Live 10-06: a navy quarter-zip with a red monogram and a flag patch came out
    a plain navy jumper -- FireRed was never told about the monogram or the flag."""
    target = tmp_path / "person.jpg"; target.write_bytes(b"P" * 50)
    garment = tmp_path / "hoodie.jpg"; garment.write_bytes(b"H" * 50)
    T._GARMENT_CACHE.clear()
    monkeypatch.setattr(T, "_garment_details",
                        lambda ctx, p: "navy quarter-zip, red JCE monogram on left chest, "
                                       "US flag patch on left sleeve")
    seen = {}
    def fake_edit(ctx, tp, region, instr, **kw):
        seen["instr"] = instr
        return None
    monkeypatch.setattr(T, "edit_region_contained_via_firered", fake_edit)
    monkeypatch.setattr(T._image, "transfer_with_references", lambda *a, **k: None)
    ref = T.ReferenceImage(str(garment), T.ROLE_CLOTHING)
    ref.extracted_asset_path = str(garment)
    T.transfer_reference_contained(
        None, str(target), ref,
        "Dress the person in the target image in the clothing shown in the reference image.")
    instr = seen.get("instr", "")
    assert "red JCE monogram" in instr and "flag patch" in instr, instr
    assert "image 2" in instr and "reference image" not in instr, instr
    assert "target image" not in instr, instr


def test_garment_details_empty_when_vision_fails(tmp_path, monkeypatch):
    import llm
    p = tmp_path / "g.jpg"; p.write_bytes(b"G" * 10)
    T._GARMENT_CACHE.clear()
    def boom(**kw):
        raise RuntimeError("vision down")
    monkeypatch.setattr(llm, "analyze_image_with_llm", boom)
    assert T._garment_details(None, str(p)) == ""
    assert T._garment_details(None, str(tmp_path / "missing.jpg")) == ""

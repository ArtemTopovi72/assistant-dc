"""Object removal: ObjectClear only (user's pick 2026-09-25); FireRed only draws a fill hint.

Run: venv/Scripts/python.exe -m pytest tests/test_remove_object_objectclear.py -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["F5_TEST_RUN"] = "1"
from PIL import Image
import image_objects as O


def _img(tmp_path):
    p = tmp_path / "src.png"
    Image.new("RGB", (64, 64), (90, 90, 90)).save(p)
    return str(p)


def test_objectclear_result_is_used(tmp_path, monkeypatch):
    monkeypatch.setattr(O, "_remove_via_objectclear", lambda *a, **k: "oc.png")
    monkeypatch.setattr(O._image, "edit_region_contained_cropped",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("FireRed ran")))
    assert O.remove_object_with_comfy(None, _img(tmp_path), "vase") == "oc.png"


def test_no_firered_fallback(tmp_path, monkeypatch):
    # owner 10-03: the FireRed fallback left a cat-shaped halo -- removed
    monkeypatch.setattr(O, "_remove_via_objectclear", lambda *a, **k: None)
    monkeypatch.setattr(O._image, "edit_region_contained_cropped", lambda *a, **k: "fr.png")
    assert O.remove_object_with_comfy(None, _img(tmp_path), "vase") is None


def test_fill_hint_goes_straight_to_firered(tmp_path, monkeypatch):
    monkeypatch.setattr(O, "_remove_via_objectclear",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("ObjectClear ran")))
    monkeypatch.setattr(O._image, "edit_region_contained_cropped", lambda *a, **k: "fr.png")
    assert O.remove_object_with_comfy(None, _img(tmp_path), "vase", fill_hint="a lamp") == "fr.png"


def test_objectclear_mask_skips_vlm_cutout_judge(tmp_path, monkeypatch):
    # live 10-03: the VLM cutout judge rejected a clean cat mask twice, the removal
    # fell to FireRed and left a halo; the ObjectClear fill is judged by the smear check
    import image_contained_firered as CF
    seen = {}
    monkeypatch.setattr(CF, "_contained_region_mask", lambda *a, **k: seen.update(k) or None)
    O._remove_via_objectclear(None, _img(tmp_path), "cat", seed=1, timeout=5)
    assert seen.get("stage1_qa") is False


def test_firered_instruction_has_one_article(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(O._image, "edit_region_contained_cropped",
                        lambda ctx, p, phrase, instr, **k: seen.append(instr) or "fr.png")
    O.remove_object_with_comfy(None, _img(tmp_path), "the cat", fill_hint="a lamp")
    assert "the the" not in seen[0] and "the cat" in seen[0], seen


def test_mask_takes_the_shadow_below():
    # live 10-03: the cat's shadow on the table was outside its mask -> a dark stain
    m = Image.new("L", (100, 200), 0)
    m.paste(255, (40, 20, 60, 120))                      # object 100 px tall
    box = O._with_contact_shadow(m).getbbox()
    assert box[3] >= 165 and box[1] == 20 and box[0] == 40, box


def test_mask_over_the_cap_is_not_filled(tmp_path, monkeypatch):
    # live 10-08: a leftover-logo mask covered 53% of the picture; ObjectClear rewrote
    # the whole wall and the person
    import image_contained_firered as CF
    import image_lettering_remove as LR
    src = _img(tmp_path)
    big = Image.new("L", (64, 64), 0)
    big.paste(255, (0, 0, 64, 34))                       # 53% of the picture
    small = Image.new("L", (64, 64), 0)
    small.paste(255, (0, 0, 16, 16))                     # 6%
    filled = []
    monkeypatch.setattr(LR, "_fill_objectclear", lambda *a, **k: filled.append(1) or "oc.png")
    monkeypatch.setattr(CF, "_contained_region_mask", lambda *a, **k: (None, big, (0, 0, 64, 34)))
    assert O._remove_via_objectclear(None, src, "logo", seed=1, timeout=5, max_cover=0.2) is None
    assert not filled
    monkeypatch.setattr(CF, "_contained_region_mask", lambda *a, **k: (None, small, (0, 0, 16, 16)))
    assert O._remove_via_objectclear(None, src, "logo", seed=1, timeout=5, max_cover=0.2) == "oc.png"

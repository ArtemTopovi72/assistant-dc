"""Object removal: ObjectClear only ; FireRed only draws a fill hint.

Run: venv/Scripts/python.exe -m pytest tests/test_remove_object_objectclear.py -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["F5_TEST_RUN"] = "1"
from PIL import Image
import types
import image_objects as O


def _ctx(**deps):
    return types.SimpleNamespace(deps=types.SimpleNamespace(**deps))


def _boom(what):
    def f(*a, **k):
        raise AssertionError(what)
    return f


def _img(tmp_path):
    p = tmp_path / "src.png"
    Image.new("RGB", (64, 64), (90, 90, 90)).save(p)
    return str(p)


def test_objectclear_result_is_used(tmp_path):
    ctx = _ctx(_remove_via_objectclear=lambda *a, **k: "oc.png",
               edit_region_contained_cropped=_boom("FireRed ran"))
    assert O.remove_object_with_comfy(ctx, _img(tmp_path), "vase") == "oc.png"


def test_no_firered_fallback(tmp_path):
    # no FireRed fallback: it leaves a cat-shaped halo
    ctx = _ctx(_remove_via_objectclear=lambda *a, **k: None,
               edit_region_contained_cropped=lambda *a, **k: "fr.png")
    assert O.remove_object_with_comfy(ctx, _img(tmp_path), "vase") is None


def test_fill_hint_goes_straight_to_firered(tmp_path):
    ctx = _ctx(_remove_via_objectclear=_boom("ObjectClear ran"),
               edit_region_contained_cropped=lambda *a, **k: "fr.png")
    assert O.remove_object_with_comfy(ctx, _img(tmp_path), "vase", fill_hint="a lamp") == "fr.png"


def test_objectclear_mask_skips_vlm_cutout_judge(tmp_path):
    # the VLM cutout judge must not reject a clean mask (the removal would fall to
    # FireRed and leave a halo); the ObjectClear fill is judged by the smear check
    seen = {}
    ctx = _ctx(_contained_region_mask=lambda *a, **k: seen.update(k) or None)
    O._remove_via_objectclear(ctx, _img(tmp_path), "cat", seed=1, timeout=5)
    assert seen.get("stage1_qa") is False


def test_firered_instruction_has_one_article(tmp_path):
    seen = []
    ctx = _ctx(edit_region_contained_cropped=lambda ctx, p, phrase, instr, **k: seen.append(instr) or "fr.png")
    O.remove_object_with_comfy(ctx, _img(tmp_path), "the cat", fill_hint="a lamp")
    assert "the the" not in seen[0] and "the cat" in seen[0], seen


def test_mask_takes_the_shadow_below():
    # the cat's shadow on the table must be inside the mask, or it leaves a dark stain
    m = Image.new("L", (100, 200), 0)
    m.paste(255, (40, 20, 60, 120))                      # object 100 px tall
    box = O._with_contact_shadow(m).getbbox()
    assert box[3] >= 165 and box[1] == 20 and box[0] == 40, box


def test_mask_over_the_cap_is_not_filled(tmp_path):
    # a leftover-logo mask covering 53% of the picture must not be filled (ObjectClear
    # would rewrite the whole wall and the person)
    src = _img(tmp_path)
    big = Image.new("L", (64, 64), 0)
    big.paste(255, (0, 0, 64, 34))                       # 53% of the picture
    small = Image.new("L", (64, 64), 0)
    small.paste(255, (0, 0, 16, 16))                     # 6%
    filled = []
    ctx = _ctx(_fill_objectclear=lambda *a, **k: filled.append(1) or "oc.png",
               _contained_region_mask=lambda *a, **k: (None, big, (0, 0, 64, 34)))
    assert O._remove_via_objectclear(ctx, src, "logo", seed=1, timeout=5, max_cover=0.2) is None
    assert not filled
    ctx.deps._contained_region_mask = lambda *a, **k: (None, small, (0, 0, 16, 16))
    assert O._remove_via_objectclear(ctx, src, "logo", seed=1, timeout=5, max_cover=0.2) == "oc.png"


def test_a_thing_in_the_sky_gets_no_shadow_below(tmp_path):
    # the planet Earth's smear ran down over the cat in front of it and the cat was erased
    from fakes import reads
    tall = Image.new("L", (64, 64), 0)
    tall.paste(255, (20, 0, 44, 30))
    for target, rests in (("planet Earth", False), ("cat", True)):
        got = []
        ctx = _ctx(_fill_objectclear=lambda ctx, src, mp, g: got.append(Image.open(mp).getbbox()) or "oc.png",
                   _contained_region_mask=lambda *a, **k: (None, tall, (20, 0, 44, 30)))
        for k, v in reads(yes=lambda q, t: "ground" in q and t == "cat").items():
            setattr(ctx, k, v)
        O._remove_via_objectclear(ctx, _img(tmp_path), target, seed=1, timeout=5)
        assert (got[0][3] > 30) is rests, (target, got)

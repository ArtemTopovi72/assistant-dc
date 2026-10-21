"""edit_verdict: the model states facts, code grades them; a no-op edit is measured.

The graded verdict and the whole judge are scored on bench/edit_judge_eval.py (59 labelled
pairs); this suite pins the two deterministic parts.
Run: venv/Scripts/python.exe -m pytest tests/test_edit_verdict.py -q
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image, ImageDraw
import edit_verdict as E


def test_facts_are_graded_by_code():
    assert E._verdict({"asked_change_visible": True, "something_else_instead": False, "defect": ""}) == "done"
    assert E._verdict({"asked_change_visible": True, "something_else_instead": False, "defect": "a seam"}) == "partial"
    assert E._verdict({"asked_change_visible": True, "something_else_instead": True, "defect": ""}) == "not"
    assert E._verdict({"asked_change_visible": False, "something_else_instead": False, "defect": ""}) == "not"


def test_a_small_local_edit_is_a_change_and_a_copy_is_not(tmp_path):
    """A nose edit moved the whole-frame mean by 0.16 and was called «the picture did not change»
    (live 2026-10-22): the measure looks for a changed area."""
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    im = Image.new("RGB", (1000, 1000), (120, 100, 90))
    im.save(a)
    ImageDraw.Draw(im).ellipse((470, 470, 530, 540), fill=(200, 150, 130))     # ~0.3% of the frame
    im.save(b)
    assert E._unchanged(str(a), str(a)) is True
    assert E._unchanged(str(a), str(b)) is False
    assert E._unchanged(str(a), str(tmp_path / "missing.png")) is False


def test_a_small_edit_is_shown_close_up_and_a_whole_new_picture_is_not(tmp_path):
    """A bigger nose on a whole portrait read as «nothing» changed (live 2026-10-22): the judge
    also gets BEFORE/AFTER crops of the changed area, where it is plain to see."""
    a, b, c = tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"
    im = Image.new("RGB", (1000, 1000), (120, 100, 90))
    im.save(a)
    ImageDraw.Draw(im).ellipse((470, 470, 530, 540), fill=(200, 150, 130))
    im.save(b)
    Image.new("RGB", (1000, 1000), (20, 200, 40)).save(c)
    crops = E._close_ups(str(a), str(b))
    assert [label.split(",")[0] for label, _ in crops] == ["BEFORE", "AFTER"]
    w, h = Image.open(crops[1][1]).size
    assert w < 500 and h < 500 and w > 60
    assert E._close_ups(str(a), str(c)) == []

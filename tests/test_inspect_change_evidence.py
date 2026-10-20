"""The BEFORE/AFTER inspection gets the measured pixel change. Shown two whole frames, the
vision model called a picture with a round clock cleanly removed «identical» and the removal
was reported as failed; with the measured change it named all 15 clean removals and all 12
controls (thing still there) right (bench/removal_verdict_live.py, 2026-10-21)."""
import os, tempfile, types
import numpy as np
from PIL import Image
import tool_image_handlers as H


def _scene(path, square=None, frame=None, jpeg=False):
    rng = np.random.default_rng(7)
    a = (rng.integers(90, 110, (512, 512, 3))).astype(np.uint8)       # textured wall
    if square:
        x, y, s = square
        a[y:y + s, x:x + s] = (200, 40, 40)
    if frame:                                                          # glasses: a 2 px outline
        x, y, w, h = frame
        a[y:y + 2, x:x + w] = a[y + h - 2:y + h, x:x + w] = 10
        a[y:y + h, x:x + 2] = a[y:y + h, x + w - 2:x + w] = 10
    Image.fromarray(a).save(path, quality=88) if jpeg else Image.fromarray(a).save(path)


def _asked(before_kw, after_kw):
    d = tempfile.mkdtemp()
    before, after = os.path.join(d, "before.png"), os.path.join(d, "after" + (".jpg" if after_kw.pop("jpeg", 0) else ".png"))
    _scene(before, **before_kw)
    _scene(after, jpeg=after.endswith(".jpg"), **after_kw)
    seen = {}

    def vision(ctx=None, **kw):
        seen.update(kw)
        return "The clock was removed.\n\nIs the clock still there? MISSING"
    ctx = types.SimpleNamespace(set_stage=lambda *a, **k: None, vision_model=vision)
    H._handle_inspect_image(ctx, {"image_path": after, "edit_before": before}, {"check": "Is the clock still there?"})
    return seen


def test_a_small_removal_is_named_with_where_it_is():
    seen = _asked({"square": (30, 30, 40)}, {})
    assert "NOT identical" in seen["user_text"] and "top left" in seen["user_text"]
    assert len(seen["image_paths"]) == 2      # close-ups on top made it judge the BEFORE crop


def test_a_thin_outline_removed_is_never_called_no_change():
    # the speck filter opens a 2 px outline away; «changed nothing» would then be a lie
    seen = _asked({"frame": (200, 240, 110, 40)}, {})
    assert "changed nothing" not in seen["user_text"]


def test_a_re_encode_is_not_called_a_change():
    seen = _asked({"square": (30, 30, 40)}, {"square": (30, 30, 40), "jpeg": True})
    assert "NOT identical" not in seen["user_text"]


def test_an_unchanged_picture_is_said_to_be_unchanged():
    seen = _asked({"square": (30, 30, 40)}, {"square": (30, 30, 40)})
    assert "changed nothing" in seen["user_text"]

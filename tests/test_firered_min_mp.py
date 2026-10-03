"""FireRed's working resolution never drops below 1.0 MP.

A/B 2026-09-24: at 0.52 MP (a 960x544 source) the edit came back zoomed ~1.4x,
legs and sand cropped; at 1.0 and 2.0 MP the framing was identical to the source.
TextEncodeQwenImageEditPlus scales its reference to ~1 MP by itself.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image
import image as _image

_seen = {}


def _run(tmp_path, size, **kw):
    p = tmp_path / "src.png"
    Image.new("RGB", size, "gray").save(p)
    saved = (_image._submit_and_poll, _image._upload_image_to_comfy)
    _image._upload_image_to_comfy = lambda *a, **k: "src.png"
    _image._submit_and_poll = lambda ctx, wf, **k: _seen.update(mp=wf["191"]["inputs"]["megapixels"]) or None
    try:
        _image.edit_image_with_firered(None, str(p), "make it pink", seed=1, **kw)
    finally:
        _image._submit_and_poll, _image._upload_image_to_comfy = saved
    return _seen["mp"]


def test_small_source_is_worked_at_one_megapixel(tmp_path):
    assert _run(tmp_path, (960, 544)) == 1.0


def test_small_explicit_tile_is_raised_too(tmp_path):
    assert _run(tmp_path, (400, 300), work_megapixels=0.12) == 1.0


def test_large_source_keeps_its_detail(tmp_path):
    assert abs(_run(tmp_path, (1600, 900)) - 1.44) < 1e-6

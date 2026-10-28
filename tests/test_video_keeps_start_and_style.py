"""Animating a photo keeps its opening shot and its look: «где
был реализм там только гипер реализм, где мультяшно там мультяшно… первый
снимок это и есть стартовый ракурс», a crop or zoom only when asked."""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("F5_TEST_RUN", "1")

import types  # noqa: E402

import video as V  # noqa: E402
from tests.fakes import Over  # noqa: E402


def _run(images, description="she smiles and waves", camera=False, ir=None, restyle=False):
    got = {}

    def build(desc, **kw):
        got["prompt"] = desc
        raise V.VideoUnavailable("stop here")
    engine = {"engine_available": lambda ctx: (True, ""), "build_workflow": build,
              "_image_size": lambda p: (768, 768)}
    if ir is not None:
        engine.update(_context_ir_on=lambda: True, to_context_ir=lambda ctx, d, **kw: ir)
    ctx = types.SimpleNamespace(
        video_engine=Over(V, **engine),
        yes_no=lambda q, t, default=False, wait=True: restyle if "visual style" in q else camera)
    V.generate_video(ctx, description, images=images)
    return got["prompt"]


def _img():
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    f.write(b"x"); f.close()
    return f.name


def test_a_photo_to_animate_opens_on_itself_in_its_own_style():
    p = _run([_img()])
    assert V._START_LOCK_SUFFIX in p and V._FRAMING_LOCK_SUFFIX in p and V._STYLE_LOCK_SUFFIX in p
    assert "exact first frame" in p and "visual style unchanged" in p
    assert "real skin" not in V._STYLE_LOCK_SUFFIX and "faces" not in V._STYLE_LOCK_SUFFIX


def test_an_asked_camera_move_still_opens_on_the_photo():
    p = _run([_img()], "slow zoom in on her face", camera=True)
    assert V._FRAMING_LOCK_SUFFIX not in p            # the move is not fought
    assert V._START_LOCK_SUFFIX in p                  # but it starts from the photo


def test_the_lock_survives_the_rewrite():
    ir = "integrated_multimodal_description: she waves\n\noverall_soundscape: wind"
    p = _run([_img()], ir=ir)
    assert p.index(V._START_LOCK_SUFFIX) < p.index("overall_soundscape:")


def test_text_only_or_a_set_of_references_gets_no_start_lock():
    assert V._START_LOCK_SUFFIX not in _run([])
    assert V._START_LOCK_SUFFIX not in _run([_img(), _img(), _img()])


def test_an_asked_new_style_is_not_fought():
    p = _run([_img()], "оживи её в стиле аниме", restyle=True)
    assert V._START_LOCK_SUFFIX in p and V._STYLE_LOCK_SUFFIX not in p

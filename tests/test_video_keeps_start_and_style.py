"""Animating a photo keeps its opening shot and its look (owner 10-03): «где
был реализм там только гипер реализм, где мультяшно там мультяшно… первый
снимок это и есть стартовый ракурс», a crop or zoom only when asked."""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("F5_TEST_RUN", "1")

import video as V  # noqa: E402


def _run(monkeypatch, images, description="she smiles and waves", camera=False, ir=None):
    got = {}

    def build(desc, **kw):
        got["prompt"] = desc
        raise V.VideoUnavailable("stop here")
    monkeypatch.setattr(V, "engine_available", lambda ctx: (True, ""))
    monkeypatch.setattr(V, "build_workflow", build)
    monkeypatch.setattr(V, "_image_size", lambda p: (768, 768))
    import intent
    monkeypatch.setattr(intent, "ask_yes", lambda q, t: camera)
    if ir is not None:
        monkeypatch.setattr(V, "_context_ir_on", lambda: True)
        monkeypatch.setattr(V, "to_context_ir", lambda ctx, d, **kw: ir)
    V.generate_video(None, description, images=images)
    return got["prompt"]


def _img():
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    f.write(b"x"); f.close()
    return f.name


def test_a_photo_to_animate_opens_on_itself_in_its_own_style(monkeypatch):
    p = _run(monkeypatch, [_img()])
    assert V._START_LOCK_SUFFIX in p and V._FRAMING_LOCK_SUFFIX in p
    assert "exact first frame" in p and "a photograph stays a photograph" in p
    assert "a cartoon, anime" in p and "never switch between realistic and drawn" in p


def test_an_asked_camera_move_still_opens_on_the_photo(monkeypatch):
    p = _run(monkeypatch, [_img()], "slow zoom in on her face", camera=True)
    assert V._FRAMING_LOCK_SUFFIX not in p            # the move is not fought
    assert V._START_LOCK_SUFFIX in p                  # but it starts from the photo


def test_the_lock_survives_the_rewrite(monkeypatch):
    ir = "integrated_multimodal_description: she waves\n\noverall_soundscape: wind"
    p = _run(monkeypatch, [_img()], ir=ir)
    assert p.index(V._START_LOCK_SUFFIX) < p.index("overall_soundscape:")


def test_text_only_or_a_set_of_references_gets_no_start_lock(monkeypatch):
    assert V._START_LOCK_SUFFIX not in _run(monkeypatch, [])
    assert V._START_LOCK_SUFFIX not in _run(monkeypatch, [_img(), _img(), _img()])

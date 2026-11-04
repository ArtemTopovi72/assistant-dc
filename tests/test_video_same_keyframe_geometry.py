import os
import sys
from types import SimpleNamespace
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for directory in ('', 'agent', 'imaging', 'media', 'core'):
    sys.path.insert(0, os.path.join(ROOT, directory))

import video


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv('F5_TEST_RUN', '1')


def test_same_keyframe_uses_one_prepared_image(monkeypatch):
    monkeypatch.delenv('VIDEO_STATIC_GUIDES', raising=False)
    monkeypatch.setattr(video, '_upload', lambda path, kind='image': path)
    monkeypatch.setattr(video, '_server_has_node', lambda *_: False)
    ctx = SimpleNamespace()
    wf = video.build_workflow('still camera', mode='flf2va', width=1280,
                              height=576, frames=124, seed=1,
                              images=['frame.png', 'frame.png'], ctx=ctx,
                              two_stage=False)
    cond = wf[video.N_COND]['inputs']
    assert cond['first_frame'] == cond['last_frame']
    scale = wf[cond['first_frame'][0]]
    assert scale['class_type'] == 'ImageScale'
    assert (scale['inputs']['width'], scale['inputs']['height']) == (1280, 576)
    assert scale['inputs']['crop'] == 'disabled'
    assert not any(v['class_type'] == 'MiniMaxH3AddGuide' for v in wf.values())


def test_continuation_uses_source_video_geometry(tmp_path):
    from tests.fakes import Over
    image = tmp_path / 'face.png'
    image.write_bytes(b'image')
    seen = {}
    def build(*args, **kwargs):
        seen.update(kwargs)
        raise video.VideoUnavailable('stop after checking geometry')
    engine = Over(video, engine_available=lambda ctx: (True, ''),
                  _image_size=lambda _: (426, 480),
                  probe=lambda _: {'width': 1280, 'height': 576},
                  build_chain_part=build)
    ctx = SimpleNamespace(video_engine=engine)
    video.generate_video(ctx, 'continue', images=[str(image)], size_from='original.mp4', chain=('chain', 2),
                         prepared=True, seconds=5, seed=1)
    assert (seen['width'], seen['height']) == (1280, 576)


def test_experimental_static_guides_use_same_frame(monkeypatch):
    monkeypatch.setattr(video, '_upload', lambda path, kind='image': path)
    monkeypatch.setattr(video, '_server_has_node', lambda node: node == 'MiniMaxH3AddGuide')
    monkeypatch.setenv('VIDEO_STATIC_GUIDES', '1')
    ctx = SimpleNamespace(video_static_guides=True)
    wf = video.build_workflow('still camera', mode='flf2va', width=1280, height=576,
                              frames=243, seed=1, images=['frame.png', 'frame.png'],
                              ctx=ctx, two_stage=False)
    guides = [v for v in wf.values() if v['class_type'] == 'MiniMaxH3AddGuide']
    assert [g['inputs']['frame_idx'] for g in guides] == [81, 161]
    assert all(g['inputs']['image'] == wf[video.N_COND]['inputs']['first_frame'] for g in guides)
    assert wf[video.N_SAMPLER]['inputs']['positive'][0] in {
        node for node, value in wf.items() if value is guides[-1]}

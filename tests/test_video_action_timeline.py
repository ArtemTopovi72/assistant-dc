from types import SimpleNamespace

from tool_image_handlers import _check_actions, _check_static_background


def test_action_check_prioritizes_video_timeline_over_source_image():
    seen = {}
    def vision(ctx, **kwargs):
        seen.update(kwargs)
        return 'NONE'
    look = SimpleNamespace(duration_seconds=lambda _: 3, sample_times=lambda *a, **k: [0, 2],
                           extract_frames=lambda *a: [(0, 'start.jpg'), (2, 'end.jpg')])
    ctx = SimpleNamespace(vision_model=vision, modules={'video_look': look})
    _check_actions(ctx, 'video.mp4', ['The balloon moves right'], references=['original.jpg'])
    assert [path for _, path in seen['image_paths']] == ['start.jpg', 'end.jpg', 'original.jpg']
    assert 'compare' in seen['user_text'].lower()
    assert 'unrequested' in seen['user_text'].lower()


def test_static_background_drift_warns_only_without_camera_move():
    engine = SimpleNamespace(overlay_drift=lambda path: 64, OVERLAY_DRIFT_MAX=35)
    assert _check_static_background(SimpleNamespace(yes_no=lambda *a, **k: False), engine, 'x.mp4', 'still shot')
    assert not _check_static_background(SimpleNamespace(yes_no=lambda *a, **k: True), engine, 'x.mp4', 'pan right')
    engine.overlay_drift = lambda path: 10
    assert not _check_static_background(SimpleNamespace(), engine, 'x.mp4', 'still shot')

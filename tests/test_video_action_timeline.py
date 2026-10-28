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


def test_action_review_receives_measured_positions(tmp_path):
    seen = {}
    def vision(ctx, **kwargs):
        seen.update(kwargs)
        return 'NONE'
    look = SimpleNamespace(duration_seconds=lambda _: 3, sample_times=lambda *a, **k: [0, 2],
                           extract_frames=lambda *a: [(0, 'start.jpg'), (2, 'end.jpg')])
    engine = SimpleNamespace(track_simple_motion=lambda path: [(0, 0.3, 0.5), (2, 0.7, 0.5)])
    ctx = SimpleNamespace(vision_model=vision, video_engine=engine, modules={'video_look': look},
                          yes_no=lambda *a, **kw: False)
    _check_actions(ctx, 'video.mp4', ['The balloon moves right'], references=['original.jpg'])
    assert '30%' in seen['user_text'] and '70%' in seen['user_text']


def test_incorrect_motion_claim_does_not_override_tracker():
    look = SimpleNamespace(duration_seconds=lambda _: 3, sample_times=lambda *a, **k: [0, 2],
                           extract_frames=lambda *a: [(0, 'start.jpg'), (2, 'end.jpg')])
    engine = SimpleNamespace(track_simple_motion=lambda path: [(0, 0.3, 0.5), (2, 0.7, 0.5)])
    ctx = SimpleNamespace(vision_model=lambda *a, **kw: 'The balloon did not move right; smoke appears',
                          video_engine=engine, modules={'video_look': look},
                          yes_no=lambda q, *a, **kw: 'other than' in q)
    result = _check_actions(ctx, 'video.mp4', ['The balloon moves right'], references=['original.jpg'])
    assert '30%' in result and '70%' in result
    assert 'did not move' not in result


def test_false_motion_failure_alone_does_not_warn():
    look = SimpleNamespace(duration_seconds=lambda _: 2, sample_times=lambda *a, **k: [0, 2],
                           extract_frames=lambda *a: [(0, 'start.jpg'), (2, 'end.jpg')])
    engine = SimpleNamespace(track_simple_motion=lambda path: [(0, 0.3, 0.5), (1, 0.5, 0.5), (2, 0.7, 0.5)])
    ctx = SimpleNamespace(vision_model=lambda *a, **kw: 'The balloon did not move right',
                          video_engine=engine, modules={'video_look': look}, yes_no=lambda *a, **kw: False)
    assert not _check_actions(ctx, 'video.mp4', ['The balloon moves right'], references=['original.jpg'])


def test_tracking_flags_opposite_direction_even_when_vision_says_none():
    look = SimpleNamespace(duration_seconds=lambda _: 2, sample_times=lambda *a, **k: [0, 2],
                           extract_frames=lambda *a: [(0, 'start.jpg'), (2, 'end.jpg')])
    engine = SimpleNamespace(track_simple_motion=lambda path: [(0, 0.7, 0.5), (1, 0.5, 0.5), (2, 0.3, 0.5)])
    ctx = SimpleNamespace(vision_model=lambda *a, **kw: 'NONE', video_engine=engine,
                          modules={'video_look': look}, yes_no=lambda *a, **kw: False)
    result = _check_actions(ctx, 'video.mp4', ['The balloon moves right'], references=['original.jpg'])
    assert 'opposite' in result.lower()
    assert not _check_actions(ctx, 'video.mp4', ['The balloon does not move right'], references=['original.jpg'])
    ctx.yes_no = lambda *a, **kw: True
    assert not _check_actions(ctx, 'video.mp4', ['The camera pans right'], references=['original.jpg'])


def test_tracks_wrong_direction_before_subject_leaves_frame():
    look = SimpleNamespace(duration_seconds=lambda _: 5, sample_times=lambda *a, **k: [0, 2],
                           extract_frames=lambda *a: [(0, 'start.jpg'), (2, 'end.jpg')])
    engine = SimpleNamespace(track_simple_motion=lambda path: [(0, 0.5, 0.5), (1.25, 0.6, 0.5), (2.5, 0.8, 0.5)])
    ctx = SimpleNamespace(vision_model=lambda *a, **kw: 'NONE', video_engine=engine,
                          modules={'video_look': look}, yes_no=lambda *a, **kw: False)
    assert 'opposite' in _check_actions(ctx, 'video.mp4', ['The balloon moves left'], references=['original.jpg']).lower()

from types import SimpleNamespace
import shutil
import subprocess
import pytest
import video


def test_silence_classifier_does_not_confuse_no_speech():
    seen = []
    ctx = SimpleNamespace(yes_no=lambda q, text, **kw: seen.append((q, text)) or len(seen) > 1)
    assert not video.explicit_silence(ctx, 'Без речи, но дождь должен быть слышен')
    assert 'speech' in seen[0][0].lower() and 'background' in seen[0][0].lower()


def test_silence_requires_independent_confirmation_that_no_background_is_allowed():
    answers = iter([True, True])
    ctx = SimpleNamespace(yes_no=lambda *a, **kw: next(answers))
    assert video.explicit_silence(ctx, 'Оживи фото без речи') is None
    answers = iter([True, False])
    assert video.explicit_silence(ctx, 'Убери вообще весь звук')
    answers = iter([False, False])
    assert video.explicit_silence(ctx, 'Убери вообще весь звук') is None
    answers = iter([False, True])
    assert video.explicit_silence(ctx, 'Без речи') is False
    answers = iter([True, None])
    assert video.explicit_silence(ctx, 'Убери вообще весь звук') is None
    ctx.yes_no = lambda *a, **kw: kw['default']
    assert video.explicit_silence(ctx, 'Убери вообще весь звук') is None


def test_ambiguous_silence_is_not_delivered_as_confirmed_audio(tmp_path):
    from tests.test_video_chain import _Rig
    import tool_image_handlers as handlers
    rig = _Rig(tmp_path, explicit_silence=lambda *a: None)
    ctx = rig.ctx(last_video_path='previous.mp4')
    result = handlers._handle_generate_video(ctx,
        {'user_input_original': 'Сделай ролик без звука'}, {'description': 'A quiet scene.', 'seconds': 5})
    assert '[TOOL ERROR]' in result and 'audio setting' in result.lower()
    assert not rig.renders() and ctx.last_video_path == 'previous.mp4'


def test_ambiguous_silence_keeps_the_approved_plan(tmp_path):
    from tests.test_video_chain import _Rig
    import tool_image_handlers as handlers
    rig = _Rig(tmp_path, explicit_silence=lambda *a: None)
    ctx = rig.ctx(video_plan_parts=['A calm scene.'], video_plan_seconds=[5.2])
    result = handlers._handle_generate_video(ctx,
        {'user_input_original': 'Сделай ролик без звука'}, {'description': 'A calm scene.', 'seconds': 5})
    assert '[TOOL ERROR]' in result and not rig.renders()
    assert ctx.video_plan_parts == ['A calm scene.'] and ctx.video_plan_seconds == [5.2]


def test_long_request_is_not_truncated_before_audio_decision():
    seen = []
    ctx = SimpleNamespace(yes_no=lambda question, text, **kw: seen.append((question, text)) or None)
    request = ('Длинный сценарий. ' * 120) + ' Утверждённый план: полная тишина.'
    assert video.explicit_silence(ctx, request) is None
    assert len(seen) == 2 and all('полная тишина' in question for question, _ in seen)


def test_prechecked_audio_decision_is_bound_to_one_request(tmp_path):
    from tests.test_video_chain import _Rig
    import tool_image_handlers as handlers
    reads = []
    rig = _Rig(tmp_path, explicit_silence=lambda ctx, request: reads.append(request) or None)
    ctx = rig.ctx(video_silence_prechecked=('Создай видео', False))
    handlers._handle_generate_video(ctx, {'user_input_original': 'Создай видео'},
                                    {'description': 'A quiet scene.', 'seconds': 5})
    assert rig.renders() and not reads
    rig.calls.clear()
    result = handlers._handle_generate_video(ctx, {'user_input_original': 'Теперь без звука'},
                                              {'description': 'A quiet scene.', 'seconds': 5})
    assert '[TOOL ERROR]' in result and reads == ['Теперь без звука'] and not rig.renders()


def test_mute_video_removes_audio_without_touching_source(tmp_path):
    ff = shutil.which('ffmpeg')
    if not ff:
        pytest.skip('ffmpeg not installed')
    src = str(tmp_path / 'source.mp4')
    subprocess.run([ff, '-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=c=red:s=64x64:r=24:d=1',
                    '-f', 'lavfi', '-i', 'sine=frequency=440:duration=1', '-c:v', 'libx264',
                    '-c:a', 'aac', '-shortest', src], check=True)
    out = video.mute_video(src)
    assert out and out != src and video.probe(src)['has_audio']
    assert not video.probe(out)['has_audio']
    assert video.probe(out)['seconds'] > .9


def test_silent_video_tool_result_does_not_claim_generated_audio(tmp_path):
    from tests.test_video_chain import _Rig
    import tool_image_handlers as handlers
    silent = tmp_path / 'silent.mp4'
    silent.write_bytes(b'video')
    rig = _Rig(tmp_path, explicit_silence=lambda *a: True, mute_video=lambda _: str(silent))
    ctx = rig.ctx()
    state = {'user_input_original': 'Создай короткое видео в полной тишине'}
    result = handlers._handle_generate_video(ctx, state, {'description': 'A silent square.', 'seconds': 5})
    assert state['video_path'] == str(silent)
    assert 'without audio' in result
    assert 'with generated audio' not in result


def test_generated_audio_is_not_treated_as_verified_content(tmp_path):
    from tests.test_video_chain import _Rig
    import tool_image_handlers as handlers
    rig = _Rig(tmp_path)
    state = {'user_input_original': 'Сделай видео со звуком дождя'}
    result = handlers._handle_generate_video(rig.ctx(), state,
                                             {'description': 'A scene with rain sounds.', 'seconds': 5})
    assert 'do not claim' in result.lower() and 'audio content' in result.lower()


def test_explicitly_silent_finished_continuation_is_muted_after_join(tmp_path):
    from tests.test_video_chain import _continued
    import tool_image_handlers as handlers
    joined = tmp_path / 'joined.mp4'
    joined.write_bytes(b'joined')
    silent = tmp_path / 'silent.mp4'
    silent.write_bytes(b'silent')
    seen = []
    rig, ctx, src = _continued(tmp_path, rig_kw={
        'join_continuation': lambda *a: str(joined),
        '_adopt_output': lambda path, *a: path,
        'explicit_silence': lambda ctx, text: True,
        'mute_video': lambda path: seen.append(path) or str(silent),
    })
    state = {'user_input_original': 'Продолжи видео и убери звук из всего итогового ролика'}
    result = handlers._handle_generate_video(ctx, state, {'description': 'Continue silently.', 'seconds': 5})
    assert seen == [str(joined)]
    assert state['video_path'] == str(silent)
    assert src.exists()
    assert 'without audio' in result

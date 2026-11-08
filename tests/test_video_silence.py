from types import SimpleNamespace
import shutil
import subprocess
import pytest
import video


def test_silence_classifier_does_not_confuse_no_speech():
    seen = []
    ctx = SimpleNamespace(yes_no=lambda q, text, **kw: seen.append((q, text)) or False)
    assert not video.explicit_silence(ctx, 'Без речи, но дождь должен быть слышен')
    assert 'speech' in seen[0][0].lower() and 'all sound' in seen[0][0].lower()


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

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

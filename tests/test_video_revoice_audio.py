"""A failed voice conversion must never erase a talking head's original voice."""
import shutil
import subprocess
import sys
import types

import numpy as np
import pytest
import soundfile as sf
import torch

import video
import remix


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
@pytest.mark.parametrize('separation_fails', [False, True])
def test_revoice_uses_isolated_reference_speech(tmp_path, monkeypatch, separation_fails):
    clip, voice = tmp_path / 'clip.mp4', tmp_path / 'voice.wav'
    sf.write(voice, np.ones(22050, dtype=np.float32) * 0.2, 22050)
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i', 'color=s=64x64:d=1',
                    '-f', 'lavfi', '-i', 'sine=frequency=440:duration=1', '-c:v', 'mpeg4',
                    '-c:a', 'aac', str(clip)], check=True)
    class Separator:
        def __init__(self, **_): pass
        def separate_audio_file(self, path):
            if str(path) == str(voice):
                if separation_fails:
                    raise RuntimeError('reference separation unavailable')
                return None, {'vocals': torch.full((2, 44100), 0.05)}
            audio, _ = sf.read(path, always_2d=True)
            return None, {'vocals': torch.from_numpy(audio.T.astype(np.float32))}
    monkeypatch.setitem(sys.modules, 'demucs.api', types.SimpleNamespace(Separator=Separator))
    monkeypatch.setattr(video, '_revoice_ref', lambda path, dst: path)
    seen = []
    def converted(_ctx, _src, ref, dst):
        audio, sr = sf.read(ref)
        seen.append((float(np.mean(audio)), sr))
        sf.write(dst, np.ones(22050, dtype=np.float32) * 0.1, 22050)
        return True
    monkeypatch.setattr(remix, '_seedvc', converted)
    assert video.revoice(None, str(clip), str(voice))
    assert seen[0][1] == 22050
    assert seen[0][0] == pytest.approx(0.2 if separation_fails else 0.05, abs=0.002)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_silent_conversion_is_not_delivered_as_a_voiceless_video(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    voice = tmp_path / "voice.wav"
    sf.write(voice, np.ones(22050, dtype=np.float32) * 0.1, 22050)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=s=64x64:d=1",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:v", "mpeg4",
                    "-c:a", "aac", str(clip)], check=True)

    class Separator:
        def __init__(self, **_):
            pass

        def separate_audio_file(self, path):
            audio, _ = sf.read(path, always_2d=True)
            return None, {"vocals": torch.from_numpy(audio.T.astype(np.float32))}

    monkeypatch.setitem(sys.modules, "demucs.api", types.SimpleNamespace(Separator=Separator))
    monkeypatch.setattr(video, "_revoice_ref", lambda path, dst: path)

    def silent(_ctx, _src, _ref, dst):
        sf.write(dst, np.zeros(22050, dtype=np.float32), 22050)
        return True

    monkeypatch.setattr(remix, "_seedvc", silent)
    assert video.revoice(None, str(clip), str(voice)) is None


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_revoice_keeps_video_frames_when_source_audio_ends_early(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    voice = tmp_path / "voice.wav"
    sf.write(voice, np.ones(22050, dtype=np.float32) * 0.1, 22050)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=s=64x64:d=2",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:v", "mpeg4",
                    "-c:a", "aac", str(clip)], check=True)

    class Separator:
        def __init__(self, **_):
            pass

        def separate_audio_file(self, path):
            audio, _ = sf.read(path, always_2d=True)
            return None, {"vocals": torch.from_numpy(audio.T.astype(np.float32))}

    monkeypatch.setitem(sys.modules, "demucs.api", types.SimpleNamespace(Separator=Separator))
    monkeypatch.setattr(video, "_revoice_ref", lambda path, dst: path)

    def converted(_ctx, _src, _ref, dst):
        sf.write(dst, np.ones(22050, dtype=np.float32) * 0.1, 22050)
        return True

    monkeypatch.setattr(remix, "_seedvc", converted)
    result = video.revoice(None, str(clip), str(voice))
    assert result is not None
    assert video.probe(result)["video_seconds"] >= 1.9

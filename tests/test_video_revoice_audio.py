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

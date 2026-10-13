"""The app starts without the private voice files; voice output is simply off.

A missing F5 checkpoint/vocoder leaves tts_model None and synth_single_segment
returns None (no audio) without touching F5, so a fresh clone can still open
the app to chat.
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("F5_TEST_RUN", "1")


def _deps(**kw):
    return SimpleNamespace(deps=SimpleNamespace(**kw))


def test_models_load_without_voice_files(tmp_path):
    import models
    called = []
    m = models.Models.load(whisper=False, ctx=_deps(
        WEIGHTS_PATH=tmp_path / "missing.safetensors", VOCOS_DIR=tmp_path / "no_vocos",
        load_model=lambda *a, **k: called.append("model"),
        load_vocoder=lambda *a, **k: called.append("vocoder")))
    assert m.tts_model is None and m.vocoder is None
    assert called == [], "tried to load voice files that are not there"


def test_models_load_uses_absolute_vocos_dir(tmp_path):
    import models
    w = tmp_path / "w.safetensors"; w.write_bytes(b"x")
    voc = tmp_path / "voc"; voc.mkdir(); (voc / "config.yaml").write_text("x")
    seen = {}
    m = models.Models.load(whisper=False, ctx=_deps(
        WEIGHTS_PATH=w, VOCOS_DIR=voc,
        load_vocoder=lambda **k: seen.update(k) or "VOC", load_model=lambda *a, **k: "TTS"))
    assert (m.tts_model, m.vocoder) == ("TTS", "VOC")
    assert Path(seen["local_path"]) == voc


def test_synthesis_is_a_quiet_none_when_voice_is_off():
    import audio
    ctx = SimpleNamespace(models=SimpleNamespace(tts_model=None, vocoder=None))
    assert audio.synth_single_segment(ctx, 0, "DC", "Привет") is None


def test_models_load_survives_whisper_download_failure(tmp_path):
    """First start offline (or with Hugging Face blocked): faster-whisper raised
    from its download and the whole app died before the window opened."""
    import models

    def offline(*a, **k):
        raise OSError("Tunnel connection failed: 403 Forbidden")
    m = models.Models.load(whisper=True, ctx=_deps(
        WEIGHTS_PATH=tmp_path / "missing.safetensors", VOCOS_DIR=tmp_path / "no_vocos",
        WhisperModel=offline))
    assert m.whisper is None


def test_transcription_is_empty_when_whisper_is_off(tmp_path):
    import threading
    import audio
    ctx = SimpleNamespace(models=SimpleNamespace(whisper=None), asr_lock=threading.Lock(),
                          transcription_cache={}, save_cache=lambda: None,
                          deps=SimpleNamespace(_use_gigaam=lambda: False))
    import numpy as np
    assert audio.transcribe_audio_array(ctx, np.zeros(16000, dtype=np.float32)) == ""
    clip = tmp_path / "a.wav"; clip.write_bytes(b"RIFF")
    assert audio.transcribe_audio_file(ctx, str(clip)) == ""
    assert ctx.transcription_cache == {}, "an 'ASR off' result must not be cached as the text"


def test_mismatched_checkpoint_turns_voice_off(tmp_path):
    import models
    w = tmp_path / "w.safetensors"; w.write_bytes(b"x")
    voc = tmp_path / "voc"; voc.mkdir(); (voc / "config.yaml").write_text("x")

    def mismatch(*a, **k):
        raise RuntimeError("size mismatch for transformer.text_embed.text_embed.weight")
    m = models.Models.load(whisper=False, ctx=_deps(
        WEIGHTS_PATH=w, VOCOS_DIR=voc, load_vocoder=lambda **k: "VOC", load_model=mismatch))
    assert m.tts_model is None and m.vocoder is None

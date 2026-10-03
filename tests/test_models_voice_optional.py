"""The app starts without the private voice files; voice output is simply off.

Models.load used to load the F5 checkpoint and vocoder unconditionally, and
assistant.main refused to start without them, so a fresh clone could not open
the app to chat at all. Now a missing checkpoint/vocoder leaves tts_model None
and synth_single_segment returns None (no audio) without touching F5.
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("F5_TEST_RUN", "1")


def test_models_load_without_voice_files(monkeypatch, tmp_path):
    import config
    import models
    monkeypatch.setattr(config, "WEIGHTS_PATH", tmp_path / "missing.safetensors")
    monkeypatch.setattr(config, "VOCOS_DIR", tmp_path / "no_vocos")
    import f5_tts.infer.utils_infer as ui
    called = []
    monkeypatch.setattr(ui, "load_model", lambda *a, **k: called.append("model"))
    monkeypatch.setattr(ui, "load_vocoder", lambda *a, **k: called.append("vocoder"))
    m = models.Models.load(whisper=False)
    assert m.tts_model is None and m.vocoder is None
    assert called == [], "tried to load voice files that are not there"


def test_models_load_uses_absolute_vocos_dir(monkeypatch, tmp_path):
    import config
    import models
    w = tmp_path / "w.safetensors"; w.write_bytes(b"x")
    voc = tmp_path / "voc"; voc.mkdir(); (voc / "config.yaml").write_text("x")
    monkeypatch.setattr(config, "WEIGHTS_PATH", w)
    monkeypatch.setattr(config, "VOCOS_DIR", voc)
    import f5_tts.infer.utils_infer as ui
    seen = {}
    monkeypatch.setattr(ui, "load_vocoder", lambda **k: seen.update(k) or "VOC")
    monkeypatch.setattr(ui, "load_model", lambda *a, **k: "TTS")
    m = models.Models.load(whisper=False)
    assert (m.tts_model, m.vocoder) == ("TTS", "VOC")
    assert Path(seen["local_path"]) == voc


def test_synthesis_is_a_quiet_none_when_voice_is_off():
    import audio
    ctx = SimpleNamespace(models=SimpleNamespace(tts_model=None, vocoder=None))
    assert audio.synth_single_segment(ctx, 0, "DC", "Привет") is None

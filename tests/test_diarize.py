"""Diarized transcript: labels in order of appearance, one speaker = plain path."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, soundfile as sf
import diarize as D
import audio


def test_turns_merge_and_drop_blips():
    segs = [{"speaker": 3, "start": 0.0, "end": 2.0}, {"speaker": 3, "start": 2.5, "end": 4.0},
            {"speaker": 1, "start": 4.1, "end": 4.3}, {"speaker": 1, "start": 5.0, "end": 7.0}]
    t = D.turns(segs)
    assert [(x["speaker"], x["start"], x["end"]) for x in t] == [(3, 0.0, 4.0), (1, 5.0, 7.0)]
    assert D.label_order(t) == {3: 1, 1: 2}


def _wav(tmp_path, seconds=12):
    p = tmp_path / "a.wav"
    sf.write(p, np.random.RandomState(0).randn(16000 * seconds).astype("float32") * 0.1, 16000)
    return str(p)


def _on(monkeypatch, segs):
    monkeypatch.setattr(D, "enabled", lambda: True)
    monkeypatch.setattr(D, "segments", lambda wav, timeout=300: segs)


def test_two_speakers_labelled(monkeypatch, tmp_path):
    _on(monkeypatch, [{"speaker": 5, "start": 0.0, "end": 4.0}, {"speaker": 2, "start": 4.5, "end": 8.0},
                      {"speaker": 5, "start": 8.5, "end": 11.0}])
    said = iter(["Привет, как дела?", "Нормально.", "Ну и отлично."])
    monkeypatch.setattr(audio, "transcribe_audio_file", lambda ctx, p, **k: next(said))
    out = D.speaker_transcript(None, _wav(tmp_path))
    assert out == "Спикер 1: Привет, как дела?\nСпикер 2: Нормально.\nСпикер 1: Ну и отлично."


def test_one_speaker_short_clip_or_off_is_none(monkeypatch, tmp_path):
    _on(monkeypatch, [{"speaker": 0, "start": 0.0, "end": 11.0}])
    assert D.speaker_transcript(None, _wav(tmp_path)) is None
    _on(monkeypatch, [{"speaker": 0, "start": 0, "end": 2}, {"speaker": 1, "start": 2, "end": 4}])
    assert D.speaker_transcript(None, _wav(tmp_path, seconds=4)) is None
    monkeypatch.setattr(D, "enabled", lambda: False)
    assert D.speaker_transcript(None, _wav(tmp_path)) is None


def test_worker_failure_is_none(monkeypatch, tmp_path):
    _on(monkeypatch, None)
    assert D.speaker_transcript(None, _wav(tmp_path)) is None

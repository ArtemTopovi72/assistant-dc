"""Diarized transcript: labels in order of appearance, one speaker = plain path."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, soundfile as sf
from types import SimpleNamespace as NS
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


def _ctx(segs=None, on=True, said=None, answer=None):
    deps = NS(enabled=lambda: on, segments=lambda wav, timeout=300: segs)
    if said is not None:
        deps.transcribe_audio_file = lambda ctx, p, **k: next(said)
    c = NS(deps=deps)
    if answer is not None:
        c.text_model = answer if callable(answer) else (lambda *a, **k: answer)
    return c


def test_two_speakers_labelled(tmp_path):
    segs = [{"speaker": 5, "start": 0.0, "end": 4.0}, {"speaker": 2, "start": 4.5, "end": 8.0},
                      {"speaker": 5, "start": 8.5, "end": 11.0}]
    said = iter(["Привет, как дела?", "Нормально.", "Ну и отлично."])
    out = D.speaker_transcript(_ctx(segs, said=said), _wav(tmp_path))
    assert out == "Спикер 1: Привет, как дела?\nСпикер 2: Нормально.\nСпикер 1: Ну и отлично."


def test_one_speaker_short_clip_or_off_is_none(tmp_path):
    one = _ctx([{"speaker": 0, "start": 0.0, "end": 11.0}])
    assert D.speaker_transcript(one, _wav(tmp_path)) is None
    short = _ctx([{"speaker": 0, "start": 0, "end": 2}, {"speaker": 1, "start": 2, "end": 4}])
    assert D.speaker_transcript(short, _wav(tmp_path, seconds=4)) is None
    assert D.speaker_transcript(_ctx(on=False), _wav(tmp_path)) is None


def test_worker_failure_is_none(tmp_path):
    assert D.speaker_transcript(_ctx(None), _wav(tmp_path)) is None


def test_name_used_only_when_spoken():
    t = "Спикер 1: Иван, подойди сюда.\nСпикер 2: Иду.\nСпикер 1: Быстрее."
    c = _ctx(answer='{"1": "", "2": "Иван"}')
    assert D.name_speakers(c, t).startswith("Спикер 1: Иван, подойди")
    assert "Иван (Спикер 2): Иду." in D.name_speakers(c, t)
    never = _ctx(answer='{"1": "Пётр", "2": ""}')          # never said in the talk -> dropped
    assert D.name_speakers(never, t) == t


def test_name_pass_is_harmless_on_failure():
    t = "Спикер 1: Привет.\nСпикер 2: Здравствуй."
    assert D.name_speakers(_ctx(answer="not json at all"), t) == t
    assert D.name_speakers(_ctx(answer=lambda *a, **k: 1 / 0), t) == t

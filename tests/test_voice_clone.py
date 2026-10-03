"""Voice clone core: the cut F5 hears, a punctuation pass that cannot change words,
and a reference that never leaks into another chat's ctx."""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from pydub import AudioSegment
from pydub.generators import Sine
import voice_clone as V
import llm


def _speech(ms):
    return Sine(220).to_audio_segment(duration=ms).apply_gain(-12)


def _clip(parts):
    out = AudioSegment.silent(duration=0, frame_rate=24000)
    for kind, ms in parts:
        out += _speech(ms) if kind == "s" else AudioSegment.silent(duration=ms, frame_rate=24000)
    return out


@pytest.fixture(autouse=True)
def _loudness_spans(monkeypatch, request):
    # sine tones are not a voice to Silero: the window tests run on the
    # loudness fallback; test_voice_spans_beat_loud_music sets its own spans
    if "voice_spans" not in request.node.name:
        monkeypatch.setattr(V, "_voice_spans", lambda seg: None)


def test_voice_spans_beat_loud_music(monkeypatch):
    """Live 10-03: a YouTube clip's music was the loudest part and the sample
    had no words. Where the VAD hears a voice wins over where it is loud."""
    seg = _clip([("s", 12000), ("q", 1000), ("s", 8000)])     # all loud
    monkeypatch.setattr(V, "_voice_spans", lambda seg: [(13000, 15000), (15500, 21000)])
    cut = V.pick_speech(seg)
    assert 13000 - 150 - 100 <= 21000 - len(cut) + 250 + 100  # the cut sits on 13-21 s
    assert 7000 <= len(cut) <= 9000


def test_pick_speech_skips_long_silences_and_stays_under_cap():
    seg = _clip([("q", 3000), ("s", 2500), ("q", 500), ("s", 3000), ("q", 600),
                 ("s", 2500), ("q", 4000), ("s", 2000), ("q", 2000)])
    cut = V.pick_speech(seg)
    assert 6000 <= len(cut) <= 12000
    assert cut.dBFS > -30                           # mostly speech, not the 3 s lead-in


def test_one_long_monologue_is_capped():
    cut = V.pick_speech(_clip([("s", 30000)]))
    assert len(cut) <= 11500


def test_too_little_speech_refused():
    with pytest.raises(V.CloneError) as e:
        V.pick_speech(_clip([("q", 2000), ("s", 1500), ("q", 3000)]))
    assert str(e.value) == "too_little_speech"


def _llm_says(monkeypatch, text):
    monkeypatch.setattr(llm, "send_to_lm_studio", lambda *a, **k: {"content": text})


def test_polish_adds_punctuation(monkeypatch):
    _llm_says(monkeypatch, "Привет, как дела? Подключи USB.")
    assert V.polish(object(), "привет как дела подключи USB") == "Привет, как дела? Подключи USB."


def test_polish_may_not_change_words(monkeypatch):
    _llm_says(monkeypatch, "Здравствуйте, как ваши дела?")
    assert V.polish(object(), "привет как дела") == "привет как дела"


def test_polish_survives_llm_failure(monkeypatch):
    monkeypatch.setattr(llm, "send_to_lm_studio", lambda *a, **k: (_ for _ in ()).throw(RuntimeError))
    assert V.polish(object(), "раз два") == "раз два"


def test_clone_ctx_is_local():
    shared = types.SimpleNamespace(custom_ref_wav=None, tts_lock="L")
    c = V._CloneCtx(shared, "me.wav", "привет")
    assert c.custom_ref_wav == "me.wav" and c.tts_lock == "L"
    assert shared.custom_ref_wav is None             # other chats never see it


def test_speak_routes_through_the_house_tts_path(monkeypatch, tmp_path):
    import audio
    seen = {}
    def fake(ctx, idx, actor, text, out_stem=None, **k):
        seen.update(ref=ctx.custom_ref_wav, ref_text=ctx.custom_ref_text, text=text, stem=out_stem)
        return out_stem
    monkeypatch.setattr(audio, "synth_single_segment", fake)
    out = V.speak(types.SimpleNamespace(), "me.wav", "привет мир", "Скажи USB.", str(tmp_path))
    assert seen["ref"] == "me.wav" and seen["ref_text"] == "привет мир" and seen["text"] == "Скажи USB."
    assert out.endswith(".wav")


def test_padding_and_acronyms_come_from_the_shared_preprocessor():
    import audio
    out = audio.preprocess_text_for_synthesis(None, "Подключи USB", use_censoring=False, apply_stress=False)
    assert out.endswith("." * 15) and "USB" not in out

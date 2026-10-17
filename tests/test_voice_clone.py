"""Voice clone core: the cut F5 hears, a punctuation pass that cannot change words,
and a reference that never leaks into another chat's ctx."""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from pydub import AudioSegment
from pydub.generators import Sine
import voice_clone as V


def _speech(ms):
    return Sine(220).to_audio_segment(duration=ms).apply_gain(-12)


def _clip(parts):
    out = AudioSegment.silent(duration=0, frame_rate=24000)
    for kind, ms in parts:
        out += _speech(ms) if kind == "s" else AudioSegment.silent(duration=ms, frame_rate=24000)
    return out


def _pick(seg):
    # sine tones are not a voice to Silero: the window tests run on the
    # loudness fallback; test_voice_spans_beat_loud_music sets its own spans
    return V.pick_speech(seg, voice_spans=lambda seg: None)


def test_voice_spans_beat_loud_music():
    """A YouTube clip's music was the loudest part and the sample
    had no words. Where the VAD hears a voice wins over where it is loud."""
    seg = _clip([("s", 12000), ("q", 1000), ("s", 8000)])     # all loud
    cut = V.pick_speech(seg, voice_spans=lambda seg: [(13000, 15000), (15500, 21000)])
    assert 13000 - 150 - 100 <= 21000 - len(cut) + 250 + 100  # the cut sits on 13-21 s
    assert 7000 <= len(cut) <= 9000


def test_pick_speech_skips_long_silences_and_stays_under_cap():
    seg = _clip([("q", 3000), ("s", 2500), ("q", 500), ("s", 3000), ("q", 600),
                 ("s", 2500), ("q", 4000), ("s", 2000), ("q", 2000)])
    cut = _pick(seg)
    assert 6000 <= len(cut) <= 12000
    assert cut.dBFS > -30                           # mostly speech, not the 3 s lead-in


def test_one_long_monologue_is_capped():
    cut = _pick(_clip([("s", 30000)]))
    assert len(cut) <= 11500


def test_too_little_speech_refused():
    with pytest.raises(V.CloneError) as e:
        _pick(_clip([("q", 2000), ("s", 1500), ("q", 3000)]))
    assert str(e.value) == "too_little_speech"


def _llm_says(text):
    return types.SimpleNamespace(chat_model=lambda *a, **k: {"content": text})


def test_polish_adds_punctuation():
    ctx = _llm_says("Привет, как дела? Подключи USB.")
    assert V.polish(ctx, "привет как дела подключи USB") == "Привет, как дела? Подключи USB."


def test_polish_may_not_change_words():
    assert V.polish(_llm_says("Здравствуйте, как ваши дела?"), "привет как дела") == "привет как дела"


def test_polish_survives_llm_failure():
    ctx = types.SimpleNamespace(chat_model=lambda *a, **k: (_ for _ in ()).throw(RuntimeError))
    assert V.polish(ctx, "раз два") == "раз два"


def test_clone_ctx_is_local():
    shared = types.SimpleNamespace(custom_ref_wav=None, tts_lock="L")
    c = V._CloneCtx(shared, "me.wav", "привет")
    assert c.custom_ref_wav == "me.wav" and c.tts_lock == "L"
    assert shared.custom_ref_wav is None             # other chats never see it


def test_speak_routes_through_the_house_tts_path(tmp_path):
    seen = {}
    def fake(ctx, idx, actor, text, out_stem=None, **k):
        seen.update(ref=ctx.custom_ref_wav, ref_text=ctx.custom_ref_text, text=text, stem=out_stem)
        return out_stem
    out = V.speak(types.SimpleNamespace(deps=types.SimpleNamespace(synth_single_segment=fake)), "me.wav", "привет мир", "Скажи USB.", str(tmp_path))
    assert seen["ref"] == "me.wav" and seen["ref_text"] == "привет мир" and seen["text"] == "Скажи USB."
    assert out.endswith(".wav")


def test_padding_and_acronyms_come_from_the_shared_preprocessor():
    import audio
    out = audio.preprocess_text_for_synthesis(None, "Подключи USB", use_censoring=False, apply_stress=False)
    assert out.endswith("." * 15) and "USB" not in out


def test_a_word_list_with_no_long_pause_is_cut_between_words():
    """The DC sample, a list of words read with short pauses, was one voice span past the
    cap and was cut blind inside «возможно»: the transcript had the whole word, the audio
    «воз», and F5 said «можно» at the start of every audiobook piece (2026-10-17)."""
    parts = []
    for i in range(30):
        parts += [("s", 400), ("q", 60), ("s", 310), ("q", 230)]   # a 60 ms dip inside each word
    seg = _clip(parts)
    cut = V.pick_speech(seg, voice_spans=lambda s: [(0, len(s))])
    assert 6000 <= len(cut) <= 11500, len(cut)
    assert cut[-60:].dBFS < -60, cut[-60:].dBFS                  # it ends in a pause
    # and in a long one, not in a word's own short dip: the next 400 ms of the source is a word
    end = len(cut)
    assert seg[end:end + 400].dBFS > -30

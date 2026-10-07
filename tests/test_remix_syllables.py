"""New words are sung syllable for syllable where the original's were (10-07: the cover of
«3 сентября» spread the lines by syllable share and every word landed off the melody)."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "core", "agent", "media", "voice", "bot"):
    sys.path.insert(0, os.path.join(ROOT, d))
os.environ.setdefault("F5_TEST_RUN", "1")

import remix as R  # noqa: E402
import mashup_auto as M  # noqa: E402


def _w(w, s, e):
    return {"w": w, "s": s, "e": e}


def test_phrases_split_at_breaths():
    words = [_w("я", 0.0, 0.3), _w("календарь", 0.35, 1.0), _w("переверну", 1.8, 2.5), _w("и", 2.55, 2.7)]
    assert [[x["w"] for x in p] for p in R._phrases(words)] == [["я", "календарь"], ["переверну", "и"]]


def test_syllable_marks_one_per_vowel_plus_the_end():
    m = R._syl_marks([_w("сно", 1.0, 1.2), _w("ва", 1.2, 1.4), _w("тре", 2.0, 2.2), _w("тье", 2.2, 2.6)])
    assert m == [1.0, 1.2, 2.0, 2.2, 2.6]


def _ph(t, text):
    """A phrase starting at t: one word per space, 0.3 s each."""
    return [_w(x, t + 0.3 * k, t + 0.3 * k + 0.25) for k, x in enumerate(text.split())]


def test_a_line_takes_the_run_of_phrases_its_syllables_need():
    # 10-07: one line per Whisper phrase put 19-syllable lines on 3-syllable phrases
    ph = [_ph(0, "все не то"), _ph(2, "все не так ты мой друг"), _ph(5, "я твой враг"),
          _ph(7, "как же так")]
    line9 = "ехал по дороге говновоз"                          # 9 syllables
    plan = R._plan(ph, [line9, "да"])
    assert [len(R._syl_marks(w)) - 1 for w, _ in plan][:1] == [9]
    assert [l for _, l in plan] == [line9, "да", line9]       # "да" takes one phrase, then round again


def test_the_lyric_goes_round_and_vowelless_phrases_carry_nothing():
    ph = [_ph(i * 2, "ла") for i in range(5)]
    ph.insert(1, [_w("мм", 0.6, 0.8)])
    assert [l for _, l in R._plan(ph, ["ой", "ай"])] == ["ой", "ай", "ой", "ай", "ой"]


def test_whisper_hallucinations_on_the_backing_take_no_line():
    ph = [_ph(0, "Thank you."), _ph(2, "все не то"), _ph(4, "все не так")]
    plan = R._plan(ph, ["ой-ёй-ёй", "ай-яй"])
    assert all("Thank" not in " ".join(x["w"] for x in w) for w, _ in plan)


def test_a_line_is_fitted_to_the_phrase_syllables(monkeypatch):
    import llm
    asked = []

    def fake(ctx, sp, um, **kw):
        asked.append(um)
        return "ехал по дороге говновоз" if len(asked) > 1 else "ехал говновоз"
    monkeypatch.setattr(llm, "call_llm_simple", fake)
    assert R._syl("ехал по дороге говновоз") == 9
    assert R._fit_line(None, "едет говновоз", 9) == "ехал по дороге говновоз"
    assert "add 4" in asked[0] and len(asked) == 2      # told what it has and what to change
    assert R._fit_line(None, "ехал по дороге говновоз", 10) == "ехал по дороге говновоз"   # ±1 is fine


def test_a_held_syllable_stops_where_the_singer_stopped():
    e = R._syl_ends([_w("сно", 1.0, 1.2), _w("ва", 1.2, 1.4), _w("тре", 2.0, 2.2)])
    assert e == [1.2, 1.4, 2.2]          # "ва" ends at 1.4, not at "тре" (2.0): the breath stays


@pytest.mark.skipif(not os.path.exists(M.RUBBERBAND), reason="rubberband not installed")
def test_spoken_syllables_land_on_the_sung_ones(tmp_path):
    sr = R.SR
    src = [0.10, 0.30, 0.55, 0.70, 0.95]               # syllables as spoken (quick, uneven)
    dst = [0.00, 0.60, 0.90, 1.70, 2.10]               # where the singer had them
    y = np.zeros(int(1.2 * sr), np.float32)
    rng = np.random.default_rng(0)
    for t in src[:-1]:
        i = int(t * sr)
        y[i:i + 600] = rng.standard_normal(600).astype(np.float32) * 0.8
    out = R._onto_marks(y, src, dst, int(2.2 * sr), str(tmp_path), "t")
    env = np.abs(out)
    on = []
    for t in dst[:-1]:                                  # first burst near each target mark
        lo = int(max(0, t - 0.15) * sr)
        k = lo + int(np.argmax(env[lo:lo + int(0.3 * sr)] > 0.2))
        on.append(k / sr)
    assert np.max(np.abs(np.array(on) - np.array(dst[:-1]))) < 0.03, on

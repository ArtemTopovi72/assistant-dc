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


def test_lines_take_phrases_in_order_and_go_round():
    ph = [[_w("а", i, i + 0.5)] for i in range(5)]
    assert [l for _, l in R._plan(ph, ["x", "y"])] == ["x", "y", "x", "y", "x"]
    ph.insert(1, [_w("мм", 0.6, 0.8)])                  # no vowel: skipped, and no line lost to it
    assert [l for _, l in R._plan(ph, ["x", "y"])] == ["x", "y", "x", "y", "x"]


def test_a_line_is_fitted_to_the_phrase_syllables(monkeypatch):
    import llm
    asked = []

    def fake(ctx, sp, um, **kw):
        asked.append(um)
        return "ехал по дороге говновоз" if len(asked) > 1 else "ехал говновоз"
    monkeypatch.setattr(llm, "call_llm_simple", fake)
    assert R._syl("ехал по дороге говновоз") == 9
    assert R._fit_line(None, "едет говновоз", 9) == "ехал по дороге говновоз"
    assert "9" in asked[0] and len(asked) == 2      # a wrong count is asked again


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

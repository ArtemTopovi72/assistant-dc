"""H3 reads Russian stress its own way and ignores marks: the speech is re-said by F5
and laid word by word onto H3's timing (media/video_dub.py)."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "core", "media", "voice"):
    sys.path.insert(0, os.path.join(ROOT, d))
os.environ.setdefault("F5_TEST_RUN", "1")

import video_dub as VD  # noqa: E402


def test_anchors_pair_the_same_words_and_skip_misheard_ones():
    old = [("да", 0.2, 0.4), ("пошли", 0.4, 0.8), ("они", 0.8, 1.0), ("сдая", 1.0, 1.4), ("анализы", 1.4, 2.0)]
    new = [("да", 0.1, 0.3), ("пошли", 0.3, 0.6), ("они", 0.6, 0.8), ("сдай", 0.8, 1.0), ("анализы", 1.0, 1.5)]
    assert VD.anchors(old, new) == [(0.1, 0.2), (0.3, 0.4), (0.6, 0.8), (1.0, 1.4)]


def test_warp_puts_each_word_where_h3_said_it(tmp_path):
    # Plain resampling stands in for rubberband: only the placement is tested here.
    def stretch(y, target, work, k):
        return np.interp(np.linspace(0, len(y) - 1, target), np.arange(len(y)), y).astype(np.float32)
    sr = VD.SR
    new = np.zeros(2 * sr, dtype=np.float32)
    new[int(0.5 * sr):int(0.6 * sr)] = 1.0           # a burst at the new take's second word
    pairs = [(0.0, 0.0), (0.5, 1.5), (1.0, 2.5)]     # the new 0.5 s lands at the old 1.5 s
    out = VD.warp(new, pairs, 4 * sr, str(tmp_path), stretch)
    hot = np.flatnonzero(out > 0.5) / sr
    assert 1.45 < hot[0] < 1.55, hot[:3]
    assert len(out) == 4 * sr


def test_redub_gives_up_without_speech_models():
    class Ctx:
        models = None
    assert VD.redub(Ctx(), __file__, ["Привет."]) == ""


def test_the_bridge_sentence_shares_no_words_with_the_line():
    line = "Сегодня на улице тихо, вали в свой каменный замок."
    b = VD._bridge_sentence(line)
    said = {w.lower() for w in VD._WORD.findall(line)}
    assert not said & {w.lower() for w in VD._WORD.findall(b)}, b


def test_the_reference_is_the_longest_run_that_fits_f5():
    """From the first word to the last a 55 s clip went in; F5 cut it to 12 s and kept the
    whole transcript, so voice and text no longer matched (2026-10-20)."""
    import video_dub
    words = [("слово", float(i), float(i) + 0.8) for i in range(55)]
    i0, i1 = video_dub._ref_span(words)
    assert words[i1][2] - words[i0][1] <= video_dub.REF_MAX_S
    assert i1 - i0 >= 8


def test_a_monologue_is_said_a_few_whole_sentences_at_a_time():
    import video_dub
    text = " ".join(f"Это предложение номер {i} из нашего длинного монолога." for i in range(12))
    parts = video_dub._chunks(text)
    assert len(parts) > 1 and " ".join(parts) == text
    assert all(len(p.split()) <= video_dub.CHUNK_WORDS for p in parts)


def test_the_old_voices_residue_is_ducked_under_its_words():
    """The speech gate's leftover still carries H3's own voice: kept at full level it was
    heard as an echo of the new one (2026-10-20). Between the words the hum stays."""
    import video_dub
    g = video_dub._duck(video_dub.SR * 3, [("а", 1.0, 2.0)])
    assert abs(g[int(1.5 * video_dub.SR)] - video_dub.HUM_UNDER_WORDS) < 1e-3
    assert g[int(0.3 * video_dub.SR)] > 0.99 and g[int(2.7 * video_dub.SR)] > 0.99


def test_a_piece_is_cloned_from_speech_that_does_not_say_its_words():
    """A sample saying the very words carries H3's stress and F5 copies it: «рюкза́чком»
    though the text said рюкзачк+ом; from a sample of other words the line came out right."""
    import video_dub
    words = [(w, i * 0.6, i * 0.6 + 0.5) for i, w in enumerate(
        "нам нужен твой морозильник я отметил на карте улицу слушай будь с рюкзачком хорошо".split())]
    i0, i1 = video_dub._ref_span(words, avoid=frozenset({"слушай", "будь", "с", "рюкзачком", "хорошо"}))
    taken = {w for w, *_ in words[i0:i1 + 1]}
    assert "рюкзачком" not in taken and len(taken) >= video_dub.REF_MIN_WORDS


def test_a_sample_whose_words_were_heard_as_digits_is_not_taken():
    """«четыре-пять» came back as «4 5»: that text does not match the voice F5 is given."""
    import video_dub
    words = [(w, i * 0.6, i * 0.6 + 0.5) for i, w in enumerate("у нас есть 4 5 недель прежде чем опять прилетят мухи".split())]
    i0, i1 = video_dub._ref_span(words)
    assert not any(c.isdigit() for w, *_ in words[i0:i1 + 1] for c in w)

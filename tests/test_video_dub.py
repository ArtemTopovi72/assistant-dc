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

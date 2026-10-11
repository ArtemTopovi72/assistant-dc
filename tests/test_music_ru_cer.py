import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
import pytest
C = pytest.importorskip(
    "music_ru_cer",
    reason="bench/music_ru_cer.py is not in the repository (it exists only on the machine that wrote this test)")


def test_perfect_hearing_is_zero():
    assert C.cer(C.LYRICS, C.LYRICS.replace("[chorus]", "")) == 0.0


def test_tags_punctuation_and_yo_ignored():
    assert C.norm("[verse] Ёлка, ЛЕТИ!") == "елка лети"


def test_silence_is_total_loss_and_noise_is_capped():
    assert C.cer(C.LYRICS, "") == 1.0
    assert C.cer(C.LYRICS, "бла " * 500) == 1.0


def test_partial():
    assert 0 < C.cer("лети лети", "лети") < 1

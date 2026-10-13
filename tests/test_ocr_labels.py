"""OCR words are rejoined into the labels they sit in, never into one row across the picture.

A meme read as one joined row stitched a speech bubble («Всё равно…») to a cape label
two metres away («Opus») into «Всё равно Opus Claude», and the reply explained that
phrase as the joke (live 2026-10-13)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ocr_reader  # noqa: E402


def test_neighbouring_words_make_one_label():
    assert ocr_reader._labels([(100, 50, 160, 80, "whois"), (170, 50, 200, 80, "→")]) == ["whois →"]


def test_words_far_apart_on_one_row_stay_separate():
    got = ocr_reader._labels([(700, 500, 760, 530, "равно"), (300, 505, 360, 540, "Opus")])
    assert got == ["Opus", "равно"]


def test_a_much_bigger_word_is_another_label():
    assert ocr_reader._labels([(100, 100, 400, 220, "Claude"), (410, 180, 450, 200, "x")]) == ["Claude", "x"]

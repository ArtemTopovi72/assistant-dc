"""🎤 Cover: the lyric/style split and the model's lyric format."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["F5_TEST_RUN"] = "1"
import cover


def test_style_line_is_split_off():
    assert cover.split_style("Стиль: рок, мужской вокал\nраз\nдва") == ("раз\nдва", "рок, мужской вокал")
    assert cover.split_style("раз\nдва") == ("раз\nдва", "")


def test_untagged_lyric_becomes_one_verse():
    assert cover.cover_lyrics("раз\nдва") == "[Verse]\nраз\nдва\n"
    assert cover.cover_lyrics("[verse 2]\nраз\n[chorus]\nлети") == "[Verse]\nраз\n\n[Chorus]\nлети\n"
    assert cover.cover_lyrics("  ") == ""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
import pytest
G = pytest.importorskip(
    "mulacover_gen",
    reason="bench/mulacover_gen.py is not in the repository (it exists only on the machine that wrote this test)")


def test_sections_capitalised_and_separated():
    out = G.to_mulacover_lyrics("[verse]\nраз\nдва\n[chorus]\nлети")
    assert out == "[Verse]\nраз\nдва\n\n[Chorus]\nлети\n"

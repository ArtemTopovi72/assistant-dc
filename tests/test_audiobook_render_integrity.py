"""A delivered audiobook chapter must contain every requested piece."""
import os
import sys

import pytest
from pydub.generators import Sine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import audiobook


@pytest.mark.parametrize("failures", [0, 1, 2], ids=["complete", "retry_recovers", "missing_piece"])
def test_missing_middle_piece_is_retried_but_never_silently_skipped(tmp_path, failures):
    paragraphs = [
        "Утром герой вышел из дома и направился по дороге к далёкому лесу.",
        "В середине пути он встретил друга и рассказал ему о своём путешествии.",
        "К вечеру они добрались до города и нашли уютный дом для ночлега.",
    ]

    failed = 0

    def synth(text):
        nonlocal failed
        if "В середине" in text and failed < failures:
            failed += 1
            return None
        path = tmp_path / "piece.wav"
        Sine(220).to_audio_segment(duration=3000).export(path, format="wav")
        return str(path)

    output = tmp_path / "chapter.wav"
    complete = failures < 2
    assert audiobook.render_chapter(synth, "Глава 1", "\n".join(paragraphs), 1, str(output)) is complete
    assert output.exists() is complete

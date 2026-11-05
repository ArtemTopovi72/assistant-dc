from bench.vision_shootout import score


def test_directions_accept_correct_russian_and_english_without_contradictions():
    assert score('where', 'слева', 'left') == 1.0
    assert score('where', 'справа', 'справа') == 1.0
    assert score('where', 'справа', 'right') == 1.0
    assert score('where', 'слева', 'right') == 0.0
    assert score('where', 'слева', 'left or справа') == 0.0

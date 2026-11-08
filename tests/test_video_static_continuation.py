from types import SimpleNamespace

import video


def test_moving_object_overrides_mistaken_static_classification():
    video._STATIC_CONT_CACHE.clear()
    questions = []
    def yes(q, text, **kwargs):
        questions.append(q)
        return True
    ctx = SimpleNamespace(yes_no=yes)
    assert not video.static_continuation(ctx, 'A blue square moves right; the camera stays still.')
    assert len(questions) == 2


def test_speaking_or_waving_in_place_keeps_static_mode():
    video._STATIC_CONT_CACHE.clear()
    answers = iter((True, False))
    ctx = SimpleNamespace(yes_no=lambda *args, **kwargs: next(answers))
    assert video.static_continuation(ctx, 'The man waves in place; the camera stays still.')

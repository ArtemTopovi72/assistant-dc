from types import SimpleNamespace

import video


def test_moving_object_overrides_mistaken_static_classification(monkeypatch):
    monkeypatch.setattr(video, '_STATIC_CONT_CACHE', {})
    questions = []
    def yes(q, text, **kwargs):
        questions.append(q)
        return True
    ctx = SimpleNamespace(yes_no=yes)
    assert not video.static_continuation(ctx, 'A blue square moves right; the camera stays still.')
    assert len(questions) == 2


def test_speaking_or_waving_in_place_keeps_static_mode(monkeypatch):
    monkeypatch.setattr(video, '_STATIC_CONT_CACHE', {})
    answers = iter((True, False))
    ctx = SimpleNamespace(yes_no=lambda *args, **kwargs: next(answers))
    assert video.static_continuation(ctx, 'The man waves in place; the camera stays still.')


def test_static_cache_uses_full_text_sent_to_classifier(monkeypatch):
    monkeypatch.setattr(video, '_STATIC_CONT_CACHE', {})
    def yes(q, text, **kwargs):
        return 'DIFFERENT LOCATION' not in q or 'moves right' in text
    ctx = SimpleNamespace(yes_no=yes)
    common = 'Keep the background unchanged. ' * 20
    assert video.static_continuation(ctx, common + 'The man speaks in place.')
    assert not video.static_continuation(ctx, common + 'A square moves right.')


def test_long_request_does_not_select_static_when_end_would_be_truncated(monkeypatch):
    monkeypatch.setattr(video, '_STATIC_CONT_CACHE', {})
    ctx = SimpleNamespace(yes_no=lambda q, *args, **kwargs: 'DIFFERENT LOCATION' not in q)
    request = 'Keep the background unchanged. ' * 55 + 'The square moves right.'
    assert len(request) > 1500
    assert not video.static_continuation(ctx, request)


from types import SimpleNamespace

from graph_personality import _reverses_request


def _misreading_model(question, text, *args, **kwargs):
    return 'opposite' in question


def test_requested_quoted_lettering_is_not_blocked_as_opposite():
    ctx = SimpleNamespace(yes_no=_misreading_model)
    request = 'На фартуке напиши «GLM 5.3», а на бочке напиши «Кал всем и каждому».'
    edit = {'region': 'apron', 'instructions': 'the text "GLM 5.3" written on the apron'}
    assert not _reverses_request(ctx, request, edit)


def test_removal_request_still_goes_through_opposite_guard():
    ctx = SimpleNamespace(yes_no=_misreading_model)
    assert _reverses_request(ctx, 'Убери надпись «GLM 5.3»',
                             {'region': 'apron', 'instructions': 'write "GLM 5.3"'})


def test_explicitly_forbidden_word_does_not_bypass_guard():
    ctx = SimpleNamespace(yes_no=_misreading_model)
    assert _reverses_request(ctx, 'Напиши «А», но не пиши «Б»',
                             {'region': 'sign', 'instructions': 'write "Б"'})

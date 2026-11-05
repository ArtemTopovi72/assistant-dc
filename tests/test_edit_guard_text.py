from types import SimpleNamespace

from graph_personality import _reverses_request


def _misreading_model(question, text, *args, **kwargs):
    return 'opposite' in question


def test_requested_quoted_lettering_is_not_blocked_as_opposite():
    ctx = SimpleNamespace(yes_no=_misreading_model,
                          text_model=lambda *a, **k: 'VERDICT: OK')
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


def test_spatial_contradiction_is_blocked_when_generic_guard_misses_it():
    ctx = SimpleNamespace(yes_no=lambda *a, **k: False,
                          text_model=lambda *a, **k: 'Ends compared.\nVERDICT: CONTRADICTION')
    assert _reverses_request(ctx, 'Пики острым концом в землю, а не древком',
                             {'instructions': 'pikes with their sharp tips pointing upwards'})


def test_spatial_guard_does_not_block_uncertain_edits():
    ctx = SimpleNamespace(yes_no=lambda *a, **k: False,
                          text_model=lambda *a, **k: 'VERDICT: OK')
    assert not _reverses_request(ctx, 'Положи книгу слева от чашки',
                                 {'instructions': 'book left of cup'})


def test_quoted_lettering_does_not_hide_opposite_spatial_edit():
    ctx = SimpleNamespace(yes_no=lambda *a, **k: False,
                          text_model=lambda *a, **k: 'VERDICT: CONTRADICTION')
    assert _reverses_request(ctx, 'Напиши «GLM 5.3» и поставь стрелку влево',
                             {'instructions': 'write GLM 5.3, arrow points right'})

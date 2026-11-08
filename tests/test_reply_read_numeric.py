import reply_read


def test_plain_number_has_no_action_claim_and_needs_no_model(monkeypatch):
    monkeypatch.setenv('INTENT_LIVE', '1')
    monkeypatch.setattr(reply_read, 'text_model', lambda ctx: (_ for _ in ()).throw(AssertionError('model called')))
    result = reply_read.read(None, '43', 'Сколько будет 17 плюс 26?')
    assert result['ok'] and result['computed']
    assert not result['did'] and not result['promises'] and not result['delivers']


def test_number_with_text_still_goes_through_semantic_reader(monkeypatch):
    monkeypatch.setenv('INTENT_LIVE', '1')
    calls = []
    monkeypatch.setattr(reply_read, 'text_model', lambda ctx: lambda *a, **k: calls.append(True) or '{}')
    reply_read.read(None, '43 файла созданы', 'Сколько файлов?')
    assert calls

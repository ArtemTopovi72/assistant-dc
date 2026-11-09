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


def test_empty_model_output_is_not_cached_as_a_successful_read(monkeypatch):
    monkeypatch.setenv('INTENT_LIVE', '1')
    reply_read._CACHE.clear()
    answers = iter(['', '{"admits_failure": true}'])
    monkeypatch.setattr(reply_read, 'text_model', lambda ctx: lambda *a, **k: next(answers))
    assert not reply_read.read(None, 'Не получилось создать видео.', 'Создай видео')['ok']
    result = reply_read.read(None, 'Не получилось создать видео.', 'Создай видео')
    assert result['ok'] and result['admits_failure']


def test_invalid_model_fields_are_not_a_successful_read(monkeypatch):
    monkeypatch.setenv('INTENT_LIVE', '1')
    reply_read._CACHE.clear()
    monkeypatch.setattr(reply_read, 'text_model', lambda ctx: lambda *a, **k: '{"admits_failure": "invalid"}')
    assert not reply_read.read(None, 'Не получилось создать видео.', 'Создай видео')['ok']

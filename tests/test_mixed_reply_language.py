from types import SimpleNamespace

import graph_language as language


def test_mostly_english_reply_is_translated_but_russian_with_names_is_not(monkeypatch):
    calls = []
    monkeypatch.setattr(language, '_read', lambda *a, **k: {'reply_language': '', 'translate': False, 'language_mode': ''})
    monkeypatch.setattr(language, 'text_model', lambda ctx: lambda *a, **k: calls.append(a) or 'Это законченный ответ по-русски.')
    ctx = SimpleNamespace(is_cancelled=lambda: False, reply_lang='ru')
    mixed = 'Старый тракторист сидел за рулём. ' + ('He was driving the tractor and he was looking at the road. ' * 3)
    assert language._match_reply_language(ctx, mixed, 'Напиши историю по-русски') == 'Это законченный ответ по-русски.'
    assert len(calls) == 1
    russian = 'Нестор Махно — Nestor Makhno, подробности: https://example.com/makhno'
    assert language._match_reply_language(ctx, russian, 'Кто это?') == russian
    assert len(calls) == 1

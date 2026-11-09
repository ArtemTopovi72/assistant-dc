from types import SimpleNamespace

import graph_language as language


def test_back_translation_must_not_invent_gender():
    assert 'Do not infer gender' in language._TRANSLATE_OUT_PROMPT
    assert 'performer' in language._TRANSLATE_OUT_PROMPT


def test_short_english_answer_is_translated_without_touching_names(monkeypatch):
    calls = []
    monkeypatch.setattr(language, '_read', lambda *a, **k: {'reply_language': '', 'translate': False, 'language_mode': ''})
    monkeypatch.setattr(language, 'text_model', lambda ctx: lambda *a, **k: calls.append(a) or 'Четыре.')
    ctx = SimpleNamespace(is_cancelled=lambda: False, reply_lang='ru')
    assert language._match_reply_language(ctx, 'Four.', 'Сколько будет два плюс два?') == 'Четыре.'
    assert len(calls) == 1
    assert language._match_reply_language(ctx, 'Python', 'Как называется язык?') == 'Python'
    assert language._match_reply_language(ctx, '`Four`', 'Что вывести в коде?') == '`Four`'
    assert len(calls) == 1
    monkeypatch.setattr(language, '_read', lambda *a, **k: {'reply_language': 'en', 'translate': False, 'language_mode': ''})
    monkeypatch.setattr(language, 'names_english', lambda *a, **k: True)
    assert language._match_reply_language(ctx, 'Four.', 'Ответь по-английски') == 'Four.'
    assert len(calls) == 1


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


def test_entry_translation_rejects_story_generated_instead_of_request(monkeypatch):
    import graph_compose
    monkeypatch.setattr(language, '_read', lambda *a, **k: {'translate': False})
    monkeypatch.setattr(language, '_protect_literals', lambda text, **kw: (text, {}))
    monkeypatch.setattr(graph_compose, '_word_count_note', lambda *a, **k: '')
    monkeypatch.setattr(language, 'text_model', lambda ctx: lambda *a, **k: 'The old man drives. ' * 40)
    request = 'Напиши смешную историю про тракториста по-русски.'
    assert language._translate_to_english(SimpleNamespace(), request) == request


def test_entry_translation_preserves_long_dialogue_and_rejects_lost_literals(monkeypatch):
    import graph_compose
    monkeypatch.setattr(language, '_read', lambda *a, **k: {'translate': False, 'wants': []})
    monkeypatch.setattr(graph_compose, '_word_count_note', lambda *a, **k: '')
    line = 'Сегодня мы спокойно обсуждаем наши планы на вечер и вспоминаем как гуляли по городу когда светило солнце и было очень тепло и уютно.'
    request = f'Оживи фото: слева говорят «{line}», справа отвечают «Да, я помню».'
    monkeypatch.setattr(language, 'text_model', lambda ctx: lambda *a, **k: 'Animate photo: left says ⟦KEEP0⟧ and right says ⟦KEEP1⟧.')
    assert language._translate_to_english(SimpleNamespace(), request) == f'Animate photo: left says «{line}» and right says «Да, я помню».'
    for broken in ('Animate photo: left says ⟦KEEP0⟧.', 'Animate photo: ⟦KEEP0⟧ ⟦KEEP0⟧ ⟦KEEP1⟧.',
                   'Animate photo: left says ⟦KEEP1⟧ and right says ⟦KEEP0⟧.'):
        monkeypatch.setattr(language, 'text_model', lambda ctx: lambda *a, **k: broken)
        assert language._translate_to_english(SimpleNamespace(), request) == request


def test_literal_order_follows_source_not_placeholder_creation(monkeypatch):
    import graph_compose
    monkeypatch.setattr(language, '_read', lambda *a, **k: {'translate': False, 'wants': []})
    monkeypatch.setattr(graph_compose, '_word_count_note', lambda *a, **k: '')
    monkeypatch.setattr(language, '_protect_literals', lambda *a, **k: (
        'Надпись ⟦KEEP1⟧, потом фраза ⟦KEEP0⟧', {'⟦KEEP0⟧': '«Привет»', '⟦KEEP1⟧': 'Мир'}))
    monkeypatch.setattr(language, 'text_model', lambda ctx: lambda *a, **k: 'Show the sign ⟦KEEP1⟧ first, and then the person clearly says ⟦KEEP0⟧.')
    assert language._translate_to_english(SimpleNamespace(), 'Надпись Мир, потом фраза «Привет»') == 'Show the sign Мир first, and then the person clearly says «Привет».'


def test_russian_product_list_with_long_urls_is_not_translated(monkeypatch):
    monkeypatch.setattr(language, '_read', lambda *a, **k: {'reply_language': '', 'translate': False, 'language_mode': ''})
    monkeypatch.setattr(language, 'text_model', lambda ctx: lambda *a, **k: 'Обрезанный ответ')
    ctx = SimpleNamespace(is_cancelled=lambda: False, reply_lang='ru')
    answer = 'Нашёл:\n' + '\n'.join(
        f'• [Peak Design The Everyday {n}](https://www.ozon.ru/product/backpack-peak-design-the-everyday-{n}-3417731870/) — 20 000 ₽.'
        for n in range(5))
    assert language._match_reply_language(ctx, answer, 'Найди рюкзак на Озоне') == answer


def test_translation_does_not_drop_original_links(monkeypatch):
    monkeypatch.setattr(language, '_read', lambda *a, **k: {'reply_language': '', 'translate': False, 'language_mode': ''})
    monkeypatch.setattr(language, 'text_model', lambda ctx: lambda *a, **k: 'Вот начало перевода.')
    ctx = SimpleNamespace(is_cancelled=lambda: False, reply_lang='ru')
    answer = 'Here is the product and the details: https://example.com/product-123'
    assert language._match_reply_language(ctx, answer, 'Где купить?') == answer


def test_translation_does_not_replace_long_reply_with_a_fragment(monkeypatch):
    monkeypatch.setattr(language, '_read', lambda *a, **k: {'reply_language': '', 'translate': False, 'language_mode': ''})
    monkeypatch.setattr(language, 'text_model', lambda ctx: lambda *a, **k: 'Вот начало перевода.')
    ctx = SimpleNamespace(is_cancelled=lambda: False, reply_lang='ru')
    answer = 'Here are the details of the product and the available options. ' * 12
    assert language._match_reply_language(ctx, answer, 'Что известно?') == answer

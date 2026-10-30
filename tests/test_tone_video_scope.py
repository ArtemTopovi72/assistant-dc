from types import SimpleNamespace

import graph
from tests.fakes import reads


def test_unrelated_text_question_does_not_reopen_video(tmp_path):
    folder = tmp_path / 'video_123'
    folder.mkdir()
    sheet = folder / 'sheet.jpg'
    sheet.touch()
    ctx = SimpleNamespace(last_image_path=str(sheet), image_origin='user',
                          **reads(read=lambda _: {'is_question': True, 'about_picture': False}))
    assert not graph.needs_relook(ctx, {'user_input': 'А как же точки?'})


def test_tone_instruction_uses_original_and_allows_uncertainty():
    from graph_compose import _tone_note
    original = 'Проанализируй тон: «Спасибо. ТЫ МОЖЕШЬ ВСЁ»'
    note = _tone_note(original, [])
    assert original in note
    assert 'ironic' in note and 'sincere' in note
    assert not _tone_note('Найди рецепт супа', [])
    assert not _tone_note('Как настроен роутер?', [])
    assert not _tone_note('Я разочарован покупкой, как вернуть деньги?', [])
    assert not _tone_note('Сегодня плохое настроение, посоветуй фильм', [])
    followup = _tone_note('А как же точки?', [{'role': 'assistant', 'content': 'Тон сообщения может быть ироничным.'}])
    assert 'text being discussed in the conversation' in followup


def test_tone_guidance_distinguishes_punctuation_from_speculation():
    from graph_compose import _tone_note
    note = _tone_note('Есть ли сарказм в «Ну спасибо.»?', [])
    assert 'ordinarily sentence separators' in note
    assert 'do not assign' in note
    assert 'when supported by the context' in note

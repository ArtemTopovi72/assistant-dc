from types import SimpleNamespace

import pytest
import tg_bot as T


@pytest.fixture
def bot(tmp_path):
    T.redirect_data_dir(tmp_path)
    b = T.TelegramBot('123:TEST', lambda: None, lambda: None, lambda: {}, silent_mode=True)
    b.sent = []; b.jobs = []
    b._send_text = lambda cid, text, **kw: b.sent.append((text, kw.get('keyboard'))) or 1
    b._backend = SimpleNamespace(push=b.jobs.append, depth=lambda: 1, position_of=lambda tid: 1)
    b._api_post = lambda *a, **kw: {'ok': True, 'result': {'message_id': 1}}
    b._get_session(17).lang = 'ru'
    return b


def press(bot, key):
    bot._resolve_and_push(17, [{'type': 'text', 'text': T._b(key, 'ru')}])


def test_language_and_mode_persist_and_back_leaves_translator(bot):
    press(bot, 'translator')
    assert bot._get_session(17).menu == 'translator'
    press(bot, 'tr_lang_fr'); press(bot, 'tr_image')
    sess = bot._get_session(17)
    reloaded = T._Session(17, sess.to_dict())
    assert reloaded.translation_target == 'fr' and reloaded.translation_mode == 'image'
    press(bot, 'back')
    assert sess.menu == '' and not bot.jobs


def test_translation_uses_submitted_language_not_later_setting(bot):
    calls = []
    bot.modules = {'translator': SimpleNamespace(translate=lambda ctx, text, target: calls.append((text,target)) or 'Bonjour')}
    press(bot, 'translator'); press(bot, 'tr_lang_fr')
    bot._resolve_and_push(17, [{'type': 'text', 'text': 'Good morning'}])
    press(bot, 'tr_lang_de')
    assert len(bot.jobs) == 1
    from tg_translator import execute
    execute(bot, SimpleNamespace(is_cancelled=lambda: False), bot.jobs[0])
    assert calls == [('Good morning', 'fr')]
    assert any(text == 'Bonjour' for text, _ in bot.sent)


def test_clear_prevents_a_late_translation_response(bot):
    bot.modules = {'translator': SimpleNamespace(translate=lambda *a: 'Bonjour')}
    press(bot, 'translator')
    bot._resolve_and_push(17, [{'type': 'text', 'text': 'Hello'}])
    bot._get_session(17).context_generation += 1
    from tg_translator import execute
    execute(bot, SimpleNamespace(is_cancelled=lambda: False), bot.jobs[0])
    assert not any(text == 'Bonjour' for text, _ in bot.sent)


def test_mode_button_and_navigation_never_become_translation_input(bot):
    press(bot, 'translator'); press(bot, 'tr_voice'); press(bot, 'tr_languages')
    assert not bot.jobs
    assert 'Français' in str(bot.sent[-1][1])


def test_translation_enters_the_durable_task_queue(bot):
    queued = []
    bot._backend = SimpleNamespace(push=queued.append, depth=lambda: 1, position_of=lambda tid: 1)
    press(bot, 'translator')
    bot._resolve_and_push(17, [{'type': 'text', 'text': 'Hello'}])
    assert len(queued) == 1
    restored = T._Task.from_dict(queued[0].to_dict())
    assert restored.translation['items'][0]['text'] == 'Hello'
    assert restored.task_id in bot._pending_journal

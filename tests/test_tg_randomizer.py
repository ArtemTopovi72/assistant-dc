from types import SimpleNamespace

import pytest
import tg_bot as T


@pytest.fixture
def bot(tmp_path):
    T.redirect_data_dir(tmp_path)
    b = T.TelegramBot('123:TEST', lambda: None, lambda: None, lambda: {}, silent_mode=True)
    b.sent = []
    b._send_text = lambda cid, text, **kw: b.sent.append((text, kw.get('keyboard'))) or 1
    b._backend = SimpleNamespace(push=lambda *a: pytest.fail('randomizer must not queue an LLM task'), depth=lambda: 0)
    s = b._get_session(17); s.lang = 'ru'; b._store.put(s)
    return b


def press(bot, key):
    bot._resolve_and_push(17, [{'type': 'text', 'text': T._b(key, 'ru')}])


def test_menu_and_range_back_navigation(bot):
    press(bot, 'randomizer')
    assert bot._get_session(17).menu == 'randomizer'
    assert '🎰' in str(bot.sent[-1][1])
    press(bot, 'rnd_number')
    assert bot._get_session(17).menu == 'randomizer_range'
    bot._resolve_and_push(17, [{'type': 'text', 'text': '7 7'}])
    assert '7' in bot.sent[-1][0] and bot.sent[-1][1]
    press(bot, 'back')
    assert bot._get_session(17).menu == 'randomizer'
    press(bot, 'back')
    assert bot._get_session(17).menu == ''


def test_slot_jackpot_and_menu_survive_the_result(bot):
    bot.modules = {'randomizer': SimpleNamespace(spin=lambda: ('💎',) * 3, jackpot=lambda x: True)}
    press(bot, 'rnd_slots')
    text, kb = bot.sent[-1]
    assert text.count('💎') == 3 and 'Выигрыш' in text
    assert kb and bot._get_session(17).menu == 'randomizer'


def test_invalid_bounds_stay_in_range_mode_and_do_not_reach_the_agent(bot):
    press(bot, 'rnd_number')
    bot._resolve_and_push(17, [{'type': 'text', 'text': '100 1'}])
    assert 'Нижняя граница' in bot.sent[-1][0]
    assert bot._get_session(17).menu == 'randomizer_range'


def test_every_quick_draw_returns_a_result_and_the_keyboard(bot):
    for key in ('rnd_coin', 'rnd_yesno', 'rnd_die'):
        press(bot, key)
        assert bot.sent[-1][0] and bot.sent[-1][1]
        assert bot._get_session(17).menu == 'randomizer'


@pytest.mark.parametrize('period', ['day', 'week', 'month'])
def test_prediction_period_and_back_navigation(bot, period):
    calls = []
    bot.modules = {'randomizer': SimpleNamespace(prediction=lambda ctx, p, lang: calls.append((p, lang)) or 'Прогноз готов')}
    bot._run_cancellable = lambda cid, note, fn, **kw: fn(SimpleNamespace(is_cancelled=lambda: False))
    press(bot, 'rnd_prediction')
    assert bot._get_session(17).menu == 'randomizer_prediction'
    press(bot, 'rnd_pred_' + period)
    assert calls == [(period, 'ru')] and bot.sent[-1][0] == 'Прогноз готов'
    assert 'На месяц' in str(bot.sent[-1][1])
    press(bot, 'back')
    assert bot._get_session(17).menu == 'randomizer'

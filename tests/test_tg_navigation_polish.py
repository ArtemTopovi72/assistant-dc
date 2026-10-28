from types import SimpleNamespace

import pytest
import tg_bot as T
import tg_music as M


@pytest.fixture
def bot(tmp_path):
    T.redirect_data_dir(tmp_path)
    b = T.TelegramBot('123:TEST', lambda: None, lambda: None, lambda: {}, silent_mode=True)
    b.sent, b.edits = [], []
    b._send_text = lambda cid, text, **kw: b.sent.append((text, kw.get('keyboard'))) or 1
    b._edit_text = lambda cid, mid, text, **kw: b.edits.append((mid, text, kw.get('keyboard'))) or True
    b._api_post = lambda *a, **kw: {'ok': True}
    b.reads = SimpleNamespace(intent_read=lambda *a, **kw: {'song': ''})
    b._backend = SimpleNamespace(push=lambda *a: pytest.fail('navigation must not queue work'), depth=lambda: 0)
    b._user_store.put(T._User(chat_id=17, name='Test', status='approved'))
    b._get_session(17).lang = 'ru'
    return b


def tap(bot, key):
    bot._resolve_and_push(17, [{'type': 'text', 'text': T._b(key, 'ru')}])


@pytest.mark.parametrize('key', [k for k in T._PROMPT_KB if k.startswith('ozon_')])
def test_ozon_prompt_back_returns_one_level_then_main(bot, key):
    tap(bot, 'ozon')
    tap(bot, key)
    tap(bot, 'back')
    sess = bot._get_session(17)
    assert sess.menu == 'ozon'
    assert sess.pending_prefix == T._PROMPT_KB['ozon_find']
    assert T._b('ozon_find', 'ru') in str(bot.sent[-1][1])
    tap(bot, 'back')
    assert sess.menu == '' and not sess.pending_prefix


@pytest.mark.parametrize('picker', ['menu', 'genre', 'duration', 'quality'])
def test_song_back_edits_the_card_without_a_back_message(bot, picker):
    sess = bot._get_session(17)
    sess.menu = 'cr_music'
    kb = M._music_menu_kb(sess, 'ru') if picker == 'menu' else M._field_kb(sess, picker, 'ru')
    bot._dispatch({'callback_query': {
        'id': 'test', 'data': 'nav:back', 'from': {'id': 17},
        'message': {'chat': {'id': 17, 'type': 'private'}, 'message_id': 55, 'reply_markup': kb}}})
    assert not bot.sent
    assert bot.edits and bot.edits[-1][0] == 55
    assert sess.menu == 'cr_music'
    if picker == 'menu':
        assert bot.edits[-1][1] == T._t('menu_cr_music_title', 'ru')
        assert bot.edits[-1][2] == {'inline_keyboard': []}
    else:
        assert 'music:open:genre' in str(bot.edits[-1][2])

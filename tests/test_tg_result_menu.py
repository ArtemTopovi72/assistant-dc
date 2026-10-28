"""Delivered results restore the active reply menu without losing inline actions."""
import json
from types import SimpleNamespace

import pytest
import tg_bot as T
import tg_feedback
from tg_transport import TransportMixin
import tg_transport


class Bot(TransportMixin):
    _PREFIX_MENU = T.TelegramBot._PREFIX_MENU
    _state_kb = T.TelegramBot._state_kb

    def __init__(self, tmp_path, menu='creativity'):
        tg_feedback.redirect_store(str(tmp_path / 'feedback.db'))
        self.sess = SimpleNamespace(menu=menu, pending_prefix='', voice_on=False,
                                    is_admin=False, lang='ru', last_task_text='request')
        self.messages = []
        self.edits = []
        self.text_model = lambda *a, **k: None
        self.modules = {'comfy_client': SimpleNamespace(card_is_exclusive=lambda: False)}

    def _get_session(self, chat_id):
        return self.sess

    def _lang(self, sess):
        return sess.lang

    def _send_text(self, chat_id, text, **kwargs):
        self.messages.append((chat_id, text, kwargs.get('keyboard')))

    def _api_post(self, method, data):
        self.edits.append((method, data))
        return {'ok': True, 'result': {'message_id': 42}}


def accepted():
    return SimpleNamespace(json=lambda: {'ok': True, 'result': {'message_id': 42}})


@pytest.mark.parametrize('kind', ['music', 'image', 'video', 'document', 'voice', ''])
def test_result_restores_creativity_and_keeps_feedback(tmp_path, kind):
    bot = Bot(tmp_path)
    own = {'inline_keyboard': [[{'text': 'Again', 'callback_data': 'again:1'}]]}
    bot._attach_feedback(7, accepted(), kind, 'result', own)
    assert len(bot.messages) == 1
    assert bot.messages[0][2] == bot._state_kb(bot.sess)
    assert bot.sess.menu == 'creativity'
    if kind:
        inline = json.loads(bot.edits[-1][1]['reply_markup'])['inline_keyboard']
        assert inline[0] == own['inline_keyboard'][0]
        assert inline[1][0]['callback_data'].startswith('fb:')
    else:
        assert not bot.edits


def test_result_preserves_current_section_and_armed_mode(tmp_path):
    bot = Bot(tmp_path, 'draw')
    bot.sess.pending_prefix = 'generate an image of: '
    bot._attach_feedback(7, accepted(), '', 'image')
    assert bot.messages[-1][2] == bot._state_kb(bot.sess)
    assert bot.sess.pending_prefix == 'generate an image of: '
    assert bot.sess.menu == 'draw'


def test_existing_reply_keyboard_is_not_sent_twice(tmp_path):
    bot = Bot(tmp_path)
    bot._attach_feedback(7, accepted(), '', keyboard=bot._state_kb(bot.sess))
    assert not bot.messages


def test_song_menu_is_written_from_request_not_filename(tmp_path):
    bot = Bot(tmp_path)
    bot.sess.last_task_text = 'Спой про сапог, который вырвался из рук Степана'
    calls = []
    def model(ctx, system, user, **kwargs):
        calls.append(user)
        return 'Вот твоя песня про сбежавший сапог 🎵'
    bot.text_model = model
    with tg_transport.result_menu(bot, 7):
        bot._attach_feedback(7, accepted(), 'music', 'broken_name_123.mp3')
    assert bot.messages[-1][1] == 'Вот твоя песня про сбежавший сапог 🎵'
    assert len(calls) == 1
    assert bot.sess.last_task_text in calls[0]
    assert 'broken_name' not in calls[0]
    assert bot.messages[-1][2] == bot._state_kb(bot.sess)


@pytest.mark.parametrize('kind,expected', [('image', 'Вот твоя картинка 🖼'),
                                         ('video', 'Вот твоё видео 🎬'),
                                         ('document', 'Файл готов 📎'),
                                         ('voice', 'Голосовое готово 🎙'),
                                         ('', 'Готово ✅')])
def test_other_results_have_a_short_completion_line(tmp_path, kind, expected):
    bot = Bot(tmp_path)
    bot._attach_feedback(7, accepted(), kind, 'internal_render_123.png')
    assert bot.messages[-1][1] == expected


def test_slow_completion_does_not_delay_menu_and_is_cancelled(tmp_path):
    import threading
    bot = Bot(tmp_path)
    release, seen = threading.Event(), []
    def slow(ctx, *args, **kw):
        seen.append(ctx)
        release.wait(1)
        return 'late'
    bot.text_model = slow
    try:
        assert tg_transport._result_menu_text(bot, 'music', 'спой про сапог', 'ru', wait_s=0.02) == 'Вот твоя песня 🎵'
        assert seen[0].is_cancelled()
    finally:
        release.set()


def test_busy_gpu_does_not_start_optional_model_call(tmp_path):
    bot = Bot(tmp_path)
    bot.modules['comfy_client'].card_is_exclusive = lambda: True
    def forbidden(*a, **kw):
        raise AssertionError('must not wait behind a render')
    bot.text_model = forbidden
    assert tg_transport._result_menu_text(bot, 'video', 'сними кота', 'ru') == 'Вот твоё видео 🎬'


def test_unaccepted_result_does_not_restore_menu(tmp_path):
    bot = Bot(tmp_path)
    bot._attach_feedback(7, SimpleNamespace(json=lambda: {'ok': False}), 'music')
    assert not bot.messages


def test_failed_audio_does_not_send_result_menu(tmp_path):
    bot = Bot(tmp_path)
    assert not bot._send_audio(7, str(tmp_path / 'missing.wav'))
    assert not bot.messages


def test_batch_restores_menu_once_after_last_file_using_latest_section(tmp_path):
    bot = Bot(tmp_path)
    with tg_transport.result_menu(bot, 7):
        bot._attach_feedback(7, accepted(), '', 'first image')
        bot._attach_feedback(7, accepted(), '', 'second image')
        assert not bot.messages
        bot.sess.menu = 'cr_music'
    assert len(bot.messages) == 1
    assert bot.messages[-1][2] == bot._state_kb(bot.sess)


def test_intermediate_busy_work_without_result_does_not_open_menu(tmp_path):
    bot = Bot(tmp_path)
    with tg_transport.result_menu(bot, 7):
        pass
    assert not bot.messages


def test_nested_result_batch_does_not_duplicate_menu(tmp_path):
    bot = Bot(tmp_path)
    with tg_transport.result_menu(bot, 7):
        with tg_transport.result_menu(bot, 7):
            bot._attach_feedback(7, accepted(), '', 'audio')
        assert not bot.messages
    assert len(bot.messages) == 1


def test_menu_flushed_before_slow_history_cleanup_is_not_duplicated(tmp_path):
    bot = Bot(tmp_path)
    with tg_transport.result_menu(bot, 7):
        bot._attach_feedback(7, accepted(), '', 'song')
        tg_transport.flush_result_menu(bot, 7)
        assert len(bot.messages) == 1
    assert len(bot.messages) == 1


def test_synthesized_voice_restores_menu(tmp_path):
    import wave
    bot = Bot(tmp_path)
    wav = tmp_path / 'voice.wav'
    with wave.open(str(wav), 'wb') as fh:
        fh.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        fh.writeframes(b'\0\0' * 2400)
    bot.speech = SimpleNamespace(synth_single_segment=lambda *a: str(wav))
    bot.http = SimpleNamespace(post=lambda *a, **k: SimpleNamespace(status_code=200))
    bot._api = 'http://unused.invalid'
    ctx = SimpleNamespace(deps=SimpleNamespace(
        speakable=lambda *a: 'hello', post_process_answer=lambda text: text))
    assert bot._send_voice(ctx, 7, 'hello')
    assert bot.messages[-1][2] == bot._state_kb(bot.sess)


@pytest.mark.parametrize('flush', [False, True])
def test_completion_uses_this_results_delivered_limitations(tmp_path, flush):
    bot = Bot(tmp_path)
    bot.sess.last_task_text = 'Сними видео с плавным наездом камеры'
    actual = 'Видео готово. Плавное приближение камеры не проявилось.'
    stale = 'Предыдущий ролик: камера двигалась идеально.'
    calls = []
    def model(ctx, system, user, **kw):
        data = json.loads(user)
        calls.append(data)
        if actual in data.get('response', ''):
            return 'Вот ваше видео; приближение камеры не получилось.'
        return 'Вот ваше видео с плавным наездом камеры.'
    bot.text_model = model
    # Prior turns and other chats must never become evidence for this result.
    with tg_transport.result_menu(bot, 7):
        TransportMixin._send_text(bot, 7, stale)
    with tg_transport.result_menu(bot, 7):
        TransportMixin._send_text(bot, 8, stale)
        TransportMixin._send_text(bot, 7, actual)
        bot._attach_feedback(7, accepted(), 'video', 'internal_render_123.mp4')
        bot.sess.menu = 'draw'
        if flush:
            tg_transport.flush_result_menu(bot, 7)
    assert bot.messages[-1][1] == 'Вот ваше видео; приближение камеры не получилось.'
    assert bot.messages[-1][2] == bot._state_kb(bot.sess)
    assert len(calls) == 1
    assert stale not in calls[0].get('response', '')
    assert 'internal_render_123' not in json.dumps(calls)

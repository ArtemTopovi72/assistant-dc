"""Offline regressions found by the independent teacher review."""
import threading
import wave
from pathlib import Path
from types import MethodType, SimpleNamespace as NS

import tg_bot
import tg_teacher as teacher
from tg_queue import QueueMixin
from tg_transport import TransportMixin
from test_tg_teacher import bot as _fixture_bot, queue
import pytest

bot = _fixture_bot


@pytest.mark.parametrize('text, expected', [
    ('本当？！そうです。', ['本当？！', 'そうです。']),
    ('「元気ですか？」', ['「元気ですか？」']),
    ('你好。。。再见！', ['你好。。。', '再见！']),
])
def test_voice_chunks_do_not_turn_cjk_closing_punctuation_into_speech_jobs(text, expected):
    from test_translation_voice_batch import W
    assert W.split_text(text) == expected


def test_rule_explanation_mode_cannot_succeed_without_an_explanation():
    import language_teacher
    from test_language_teacher import context
    ctx, _ = context(lesson_intent='explain', input_language='ru', explanation='')
    with pytest.raises(ValueError, match='invalid lesson reply'):
        language_teacher.respond(ctx, {}, [], 'Объясни английские артикли', 'en', 'ru')


def test_third_language_reply_requires_the_requested_mismatch_feedback():
    import language_teacher
    from test_language_teacher import context
    ctx, _ = context(say='Wie geht es dir?', input_language='en', used_target=False,
                     correction='', explanation='')
    with pytest.raises(ValueError, match='invalid lesson reply'):
        language_teacher.respond(ctx, {}, [], 'I am tired today', 'de', 'ru')


def test_teacher_never_gets_session_under_session_lock(bot):
    sess = bot._get_session(1)
    original = bot._get_session
    nested = []

    def get_session(cid):
        acquired = sess._lock.acquire(blocking=False)
        nested.append(not acquired)
        if acquired:
            sess._lock.release()
        return original(cid)

    task = queue(bot)
    bot._get_session = get_session
    teacher.execute(bot, NS(is_cancelled=lambda: False), task)
    # _get_session takes _mgmt_lock; enqueue takes it before serializing sess.
    assert not any(nested), 'Session -> management lock reverses enqueue lock order'


def test_stop_cancels_teacher_while_lesson_audio_is_still_working(bot):
    task = queue(bot)
    sess = bot._get_session(1)
    sess.FLOW_STATES = ()
    token = threading.Event()
    for attr in ('_stop_lock', '_mgmt_lock', '_album_lock', '_task_lock'):
        setattr(bot, attr, threading.Lock())
    bot._stop_requests, bot._albums, bot._queues, bot._pending_journal = {}, {}, {}, {}
    bot._incoming_enqueued = set()
    bot._incoming = NS(pending=lambda: [], stop_chat=lambda *args: None)
    bot._backend = NS(drop_chat=lambda cid: 0)
    bot._running_task = {1: [task]}
    bot._task_cancels = {task.task_id: token}
    cancelled_during_speech = []

    def speech(ctx, *args, **kwargs):
        QueueMixin._request_stop(bot, 1)
        cancelled_during_speech.append(ctx.is_cancelled())
        return dict(paths=['target.wav', 'native.wav'], combined_path='both.wav')

    bot.modules['translator'].lesson_speech = speech
    teacher.execute(bot, NS(is_cancelled=token.is_set), task)
    assert cancelled_during_speech == [True]
    assert not bot.voices
    assert not task.delivered
    assert not any('Hallo?' in text for _,text in bot.sent)
    assert not sess.teacher_history


def test_rejected_telegram_text_is_not_a_delivered_lesson(bot):
    task = queue(bot)
    bot._with_wait_cancel = lambda cid, keyboard: keyboard
    bot._post_message = lambda *args: {'ok': False, 'description': 'sendMessage failed'}
    bot._send_text = MethodType(TransportMixin._send_text, bot)
    bot._send_get_id = MethodType(TransportMixin._send_get_id, bot)
    teacher.execute(bot, NS(is_cancelled=lambda: False), task)
    assert not task.delivered
    assert not bot._get_session(1).teacher_history


def test_russian_ui_keeps_the_actual_english_lesson_text(bot):
    bot._get_session(1).teacher_target = 'en'
    task = queue(bot)
    bot._lang = lambda sess: 'ru'
    original = bot.modules['language_teacher'].respond

    def respond(*args):
        return dict(original(*args), say='Drawing a picture?', translation='Рисуешь картину?',
                    correction='', explanation='')

    bot.modules['language_teacher'].respond = respond
    bot._with_wait_cancel = lambda cid, keyboard: keyboard
    requests = []
    bot._api_post = lambda method, payload: requests.append(dict(payload)) or {
        'ok': True, 'result': {'message_id': len(requests)}}
    bot._post_message = MethodType(TransportMixin._post_message, bot)
    bot._send_text = MethodType(TransportMixin._send_text, bot)
    bot._send_get_id = MethodType(TransportMixin._send_get_id, bot)
    teacher.execute(bot, NS(is_cancelled=lambda: False), task)
    assert any('Drawing a picture?' in p.get('text', '') for p in requests)


def test_core_lesson_speech_keeps_target_then_native_and_reference_snapshot(tmp_path):
    import translator
    reference = tmp_path / 'selected.wav'
    reference.write_bytes(b'reference')
    python = tmp_path / 'python.exe'
    python.touch()
    model = tmp_path / 'models_ext' / 'qwen3_tts_base'
    model.mkdir(parents=True)
    (model / 'model.safetensors').touch()
    jobs = []

    def synth(ctx, python, script, job, label, timeout, shared=False):
        assert shared is True
        jobs.extend(job['segments'])
        for segment in jobs:
            with wave.open(segment['output'], 'wb') as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(24000)
                output.writeframes(b'\x00\x00' * 12000)
        return True, ''

    ctx = NS(custom_ref_wav='later.wav', is_cancelled=lambda: False,
             modules={'config': NS(BASE_DIR=tmp_path, venv_python=lambda p: str(python))},
             deps=NS(run_gpu_worker=synth))
    result = translator.lesson_speech(ctx, 'Hallo?', 'de', 'Hello?', 'en',
                                     str(tmp_path / 'out'), reference=str(reference))
    assert [(job['text'], job['language'], job['source']) for job in jobs] == [
        ('Hallo?', 'German', str(reference)), ('Hello?', 'English', str(reference))]
    assert len(result['paths']) == 2 and Path(result['combined_path']).is_file()
    with wave.open(result['combined_path'], 'rb') as combined:
        assert abs(combined.getnframes() / combined.getframerate() - 1.8) < .01

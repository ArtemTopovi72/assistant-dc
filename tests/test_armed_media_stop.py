"""Stop must win over queued intake and a download already in progress."""
import threading
from types import SimpleNamespace

import pytest
import tg_bot as T

from test_tg_incoming_durability import make_bot, hold_worker


@pytest.mark.parametrize('kind', ['armed_media', 'reply_photo'])
def test_stop_discards_a_held_batch_before_any_attachment_side_effect(tmp_path, kind):
    bot = make_bot(tmp_path); hold_worker(bot, 101)
    sess = bot._get_session(101); sess.clone_state = 'want_audio'
    effects = []
    if kind == 'armed_media':
        item = {'type': kind, 'message': {
            'voice': {'file_id': 'sample'}, '_flow_name': 'clone_state',
            '_flow_revision': sess.flow_revision('clone_state'),
            '_context_generation': sess.context_generation}}
        bot._clone_take_media = lambda *a: effects.append('consume') or True
    else:
        item = {'type': 'text', 'text': 'Describe this photo',
                'reply_photo': {'file_id': 'original', 'message_id': 17}}
        bot._save_incoming_photo = lambda fid: effects.append('download') or ''
    bot._enqueue_item(101, item)
    held = bot._queues[101].get_nowait()
    bot._request_stop(101)
    bot._resolve_and_push(101, [held])
    assert effects == []
    assert bot._backend.depth() == 0


def test_stop_during_the_real_media_download_invalidates_only_its_chat_flow(tmp_path):
    bot = make_bot(tmp_path); hold_worker(bot, 101)
    sess = bot._get_session(101); sess.clone_state = 'want_audio'
    other = bot._get_session(202); other.clone_state = 'want_audio'
    revision = sess.flow_revision('clone_state')
    other_revision = other.flow_revision('clone_state')
    generation = sess.context_generation
    bot._enqueue_item(101, {'type': 'armed_media', 'message': {
        'voice': {'file_id': 'sample'}, '_worker_media': True,
        '_flow_name': 'clone_state', '_flow_revision': revision,
        '_context_generation': generation}})
    held = bot._queues[101].get_nowait()
    entered, release = threading.Event(), threading.Event()
    effects, errors = [], []

    def download(fid):
        entered.set()
        assert release.wait(3)
        return b'media'

    bot._dl_bytes = download
    bot._clone_dir = lambda cid: effects.append('file') or str(tmp_path)
    bot._clone_prepare = lambda *a: effects.append('prepare')

    def run():
        try:
            bot._resolve_and_push(101, [held])
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert entered.wait(3)
        bot._request_stop(101)
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive() and not errors
    assert effects == []
    assert sess.clone_state == 'want_audio' and sess.context_generation == generation
    assert sess.flow_revision('clone_state') > revision
    assert other.flow_revision('clone_state') == other_revision
    assert bot._store.get(101).flow_revision('clone_state') == sess.flow_revision('clone_state')


def test_stop_after_cover_disarms_suppresses_the_actual_render_delivery(tmp_path):
    bot = make_bot(tmp_path)
    sess = bot._get_session(101)
    sess.cover_state = 'want_text'; sess.cover_state = ''
    expected = (sess.context_generation, '', sess.flow_revision('cover_state'))
    entered, release = threading.Event(), threading.Event()
    delivered, errors = [], []

    def render(*a, **kw):
        entered.set()
        assert release.wait(3)
        return 'fake.wav'

    bot.modules = {'remix': SimpleNamespace(resing_available=lambda: True, resing=render),
                   'mashup_stems': SimpleNamespace(release_separator=lambda: None)}
    bot._send_audio = lambda *a, **kw: delivered.append('audio') or True

    def run():
        try:
            bot._cover_render(101, 'ru', 'fake-source.wav', 'lyrics', '', expected)
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert entered.wait(3)
        bot._request_stop(101)
    finally:
        release.set(); worker.join(3)
    assert not worker.is_alive() and not errors
    assert delivered == []
    assert sess.cover_state == ''


def test_stop_cancels_a_real_cancellable_job_without_cancelling_another_chat(tmp_path):
    bot = make_bot(tmp_path)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    cancelled = []

    def work(ctx):
        try:
            entered.set()
            assert release.wait(3)
            cancelled.append(ctx.is_cancelled())
        finally:
            finished.set()

    bot._run_cancellable(101, 'working', work)
    other = threading.Event()
    bot._busy_jobs['other'] = (202, other)
    try:
        assert entered.wait(3)
        bot._request_stop(101)
    finally:
        release.set()
        assert finished.wait(3)
    assert cancelled == [True]
    assert not other.is_set()


@pytest.mark.parametrize('kind', ['chat_busy', 'busy_job'])
def test_stop_reports_standalone_work_as_running(tmp_path, kind):
    bot = make_bot(tmp_path)
    if kind == 'chat_busy':
        bot._chat_busy[101] = 1
    else:
        bot._busy_jobs = {'work': (101, threading.Event())}
    sent = []
    bot._send_text = lambda cid, text, **kw: sent.append(text)
    bot._stop_and_report(101, bot._get_session(101), 'ru')
    assert T._t('stop_idle', 'ru') not in sent
    assert sent == [T._t('stopping', 'ru', extra='')]

from types import SimpleNamespace

import tg_bot as T
from fakes import reads
from test_tg_incoming_durability import make_bot, hold_worker


def message(text='', **extra):
    return {'message': {'message_id': 40, 'chat': {'id': 101, 'type': 'private'},
                        'from': {'id': 101}, 'text': text, **extra}}


def forbid_model(*args, **kw):
    raise AssertionError('Telegram ingress must not call the model')


def test_commands_and_short_messages_never_wait_for_a_model_in_the_poll_thread(tmp_path):
    bot = make_bot(tmp_path); hold_worker(bot, 101)
    bot.reads = SimpleNamespace(**reads(yes=forbid_model))
    seen = []
    bot._handle_command = lambda cid, text: seen.append(text)
    for text in ('/clear', '/stop', '/cancel'):
        bot._dispatch(message(text))
    assert seen == ['/clear', '/stop', '/cancel']
    bot._dispatch(message('hello'))
    assert bot._queues[101].get_nowait()['text'] == 'hello'


def test_unknown_reply_photo_download_is_deferred_and_keeps_the_original_target(tmp_path):
    bot = make_bot(tmp_path); hold_worker(bot, 101)
    bot.reads = SimpleNamespace(**reads(read=lambda t: None, yes=lambda q, t: False))
    downloaded = []
    path = tmp_path / 'reply.png'; path.write_bytes(b'photo')
    bot._save_incoming_photo = lambda fid: downloaded.append(fid) or str(path)
    bot._dispatch(message('Describe this photo please', reply_to_message={
        'message_id': 17, 'photo': [{'file_id': 'original-photo'}]}))
    assert downloaded == []
    item = bot._queues[101].get_nowait()
    bot._resolve_and_push(101, [item])
    assert downloaded == ['original-photo']
    task = bot._backend.pop(timeout=0)
    assert T._image_by_id(bot._get_session(101), task.image_id)['path'] == str(path)


def test_armed_media_is_processed_by_the_chat_worker_instead_of_polling(tmp_path):
    bot = make_bot(tmp_path); hold_worker(bot, 101)
    sess = bot._get_session(101); sess.clone_state = 'want_sample'
    bot.reads = SimpleNamespace(**reads(read=lambda t: None, yes=lambda q, t: False))
    consumed = []
    bot._clone_take_media = lambda *a: consumed.append('sample') or True
    bot._dispatch(message(voice={'file_id': 'voice', 'duration': 2}))
    assert not consumed
    item = bot._queues[101].get_nowait()
    assert item['type'] == 'armed_media'
    bot._resolve_and_push(101, [item])
    assert consumed == ['sample']
    assert bot._backend.depth() == 0


def test_back_navigation_does_not_wait_behind_an_attachment(tmp_path):
    bot = make_bot(tmp_path); hold_worker(bot, 101)
    sess = bot._get_session(101); sess.menu = 'randomizer'
    bot.reads = SimpleNamespace(**reads(yes=forbid_model))
    sent = []; bot._send_text = lambda cid, text, **kw: sent.append(text)
    bot._dispatch(message(T._b('back', 'ru')))
    assert sent and sess.menu != 'randomizer'
    assert 101 not in bot._queues


def test_new_prompt_leaves_an_old_media_capture_without_classifying_the_menu_label(tmp_path):
    bot = make_bot(tmp_path); hold_worker(bot, 101)
    sess = bot._get_session(101); sess.clone_state = 'want_sample'
    bot.reads = SimpleNamespace(**reads(read=forbid_model, yes=forbid_model))
    bot._dispatch(message(T._b('web_search', 'ru')))
    assert not sess.clone_state
    assert sess.pending_prefix == T._PROMPT_KB['web_search']


def test_a_new_prompt_cannot_capture_an_older_queued_request(tmp_path):
    bot = make_bot(tmp_path); hold_worker(bot, 101)
    bot.reads = SimpleNamespace(**reads(read=lambda t: None, yes=lambda q, t: False))
    bot._dispatch(message('Find the latest news'))
    old = bot._queues[101].get_nowait()
    bot._dispatch(message(T._b('gen_image', 'ru')))
    prefix = bot._get_session(101).pending_prefix
    bot._dispatch(message('A blue elephant'))
    fresh = bot._queues[101].get_nowait()
    bot._resolve_and_push(101, [old, fresh])
    first, second = (bot._backend.pop(timeout=0) for _ in range(2))
    assert first.user_text == 'Find the latest news'
    assert second.user_text == prefix + 'A blue elephant'
    assert not bot._get_session(101).pending_prefix


def test_older_queued_content_does_not_clear_a_new_photo_prompt(tmp_path):
    bot = make_bot(tmp_path); hold_worker(bot, 101)
    bot.reads = SimpleNamespace(**reads(read=lambda t: None, yes=lambda q, t: False))
    bot._dispatch(message('Explain gravity'))
    old = bot._queues[101].get_nowait()
    sess = bot._get_session(101); sess.pending_photo = 'Analyze the next photo'
    bot._resolve_and_push(101, [old])
    assert sess.pending_photo == 'Analyze the next photo'


def test_album_keeps_its_arrival_mode_until_flush_and_recovery(tmp_path):
    bot = make_bot(tmp_path); hold_worker(bot, 101)
    sess = bot._get_session(101)
    bot.album_collect_s = 60
    bot._buffer_album(101, 'album-one', 'photo-one', 'Explain these shapes')
    with bot._album_lock:
        bot._albums['album-one']['timer'].cancel()
    saved = bot._incoming.pending()[0]['item']
    sess.pending_prefix = T._PROMPT_KB['gen_image']
    bot._store.put(sess)
    bot._flush_album('album-one')
    item = bot._queues[101].get_nowait()
    assert item['_input_context']['prefix'] == ''
    assert item['_arrival_ts'] == saved['_arrival_ts']
    recovered = make_bot(tmp_path); hold_worker(recovered, 101)
    recovered._buffer_album(101, 'album-one', 'photo-one', 'Explain these shapes', saved=saved)
    with recovered._album_lock:
        recovered._albums['album-one']['timer'].cancel()
    recovered._flush_album('album-one')
    assert recovered._queues[101].get_nowait()['_input_context']['prefix'] == ''


def test_stop_during_reply_photo_download_cannot_publish_a_late_image(tmp_path):
    import threading
    bot = make_bot(tmp_path); hold_worker(bot, 101)
    path = tmp_path / 'reply.png'; path.write_bytes(b'photo')
    entered, release = threading.Event(), threading.Event()
    def download(fid):
        entered.set()
        assert release.wait(3)
        return str(path)
    bot._save_incoming_photo = download
    bot._dispatch(message('Describe this photo', reply_to_message={
        'message_id': 17, 'photo': [{'file_id': 'original-photo'}]}))
    item = bot._queues[101].get_nowait()
    worker = threading.Thread(target=bot._resolve_and_push, args=(101, [item]))
    worker.start()
    try:
        assert entered.wait(3)
        bot._request_stop(101)
    finally:
        release.set(); worker.join(3)
    assert not worker.is_alive()
    assert not bot._get_session(101).image_log and bot._backend.depth() == 0


def test_stop_during_as_file_classification_cannot_send_the_file(tmp_path):
    import threading
    bot = make_bot(tmp_path); hold_worker(bot, 101)
    path = tmp_path / 'picture.png'; path.write_bytes(b'photo')
    T._log_image(bot._get_session(101), str(path), msg_id=17)
    entered, release = threading.Event(), threading.Event()
    def yes(question, text):
        if 'as a FILE' not in question:
            return False
        entered.set()
        assert release.wait(3)
        return True
    bot.reads = SimpleNamespace(**reads(read=lambda t: None, yes=yes))
    sent = []; bot._send_document = lambda *a, **kw: sent.append(a) or True
    bot._dispatch(message('пришли файлом', reply_to_message={'message_id': 17}))
    item = bot._queues[101].get_nowait()
    worker = threading.Thread(target=bot._resolve_and_push, args=(101, [item]))
    worker.start()
    try:
        assert entered.wait(3)
        bot._request_stop(101)
    finally:
        release.set(); worker.join(3)
    assert not worker.is_alive() and not sent and bot._backend.depth() == 0


def test_sending_as_file_is_resolved_in_the_chat_worker(tmp_path):
    bot = make_bot(tmp_path); hold_worker(bot, 101)
    path = tmp_path / 'picture.png'; path.write_bytes(b'photo')
    T._log_image(bot._get_session(101), str(path), msg_id=17)
    asked = []; sent = []
    bot.reads = SimpleNamespace(**reads(read=lambda t: None,
        yes=lambda q, t: asked.append(t) or ('as a FILE' in q)))
    bot._send_document = lambda cid, file, **kw: sent.append(file) or True
    bot._dispatch(message('пришли файлом', reply_to_message={'message_id': 17}))
    assert not asked and not sent
    bot._resolve_and_push(101, [bot._queues[101].get_nowait()])
    assert sent == [str(path)] and bot._backend.depth() == 0


def test_queued_media_flow_identity_survives_process_recovery(tmp_path):
    bot = make_bot(tmp_path); hold_worker(bot, 101)
    sess = bot._get_session(101)
    sess.clone_state = 'want_sample'; sess.clone_state = ''; sess.clone_state = 'want_sample'
    bot._store.put(sess)
    bot._dispatch(message(voice={'file_id': 'saved-sample'}))
    recovered = make_bot(tmp_path); hold_worker(recovered, 101)
    consumed = []
    recovered._clone_take_media = lambda *a: consumed.append('sample') or True
    recovered._restore_incoming([])
    recovered._resolve_and_push(101, [recovered._queues[101].get_nowait()])
    assert consumed == ['sample']


def test_a_queued_sample_does_not_enter_a_reopened_flow(tmp_path):
    bot = make_bot(tmp_path); hold_worker(bot, 101)
    sess = bot._get_session(101); sess.clone_state = 'want_sample'
    consumed = []
    bot._clone_take_media = lambda *a: consumed.append('sample') or True
    bot._dispatch(message(voice={'file_id': 'obsolete-sample'}))
    item = bot._queues[101].get_nowait()
    sess.clone_state = ''; sess.clone_state = 'want_sample'
    bot._resolve_and_push(101, [item])
    assert not consumed


def test_two_queued_voice_replies_keep_their_own_quotes(tmp_path):
    bot = make_bot(tmp_path); hold_worker(bot, 101)
    bot.reads = SimpleNamespace(**reads(read=lambda t: None, yes=lambda q, t: False))
    bot._get_ctx = lambda: SimpleNamespace()
    bot._transcribe = lambda *a, **kw: 'What does it mean?'
    for quote in ('Unique quote A', 'Unique quote B'):
        bot._dispatch(message(voice={'file_id': quote}, reply_to_message={'message_id': 17, 'text': quote}))
    a, b = (bot._queues[101].get_nowait() for _ in range(2))
    bot._resolve_and_push(101, [a]); bot._resolve_and_push(101, [b])
    first = bot._backend.pop(timeout=0); second = bot._backend.pop(timeout=0)
    assert 'Unique quote A' in first.user_text and 'Unique quote B' not in first.user_text
    assert 'Unique quote B' in second.user_text and 'Unique quote A' not in second.user_text
    assert not bot._get_session(101).quoted_text


def test_request_read_survives_durable_task_handoff_without_claiming_another_chat(tmp_path):
    import json
    import threading
    import intent
    from models import Context
    bot = make_bot(tmp_path)
    calls = []
    def model(ctx, system, user, **kw):
        calls.append(user)
        return json.dumps({'needs_tool': False})
    shared = Context(models=SimpleNamespace(), transcription_cache={}, cache_file=tmp_path / 'cache.json',
                     asr_lock=threading.Lock(), tts_lock=threading.Lock())
    shared.text_model = model; shared.deps = SimpleNamespace(live=True)
    with intent.turn_scope():
        intent.bind_turn(['again'], 'private A', 'a photo', entry=True)
        intent.read(shared, 'again')
        task = T._Task(task_id='owned-a', chat_id=101, user_text='again')
        bot._persist_task(task)
    restored = T._Task.from_dict(json.loads(json.dumps(task.to_dict())))
    assert 'lock' not in restored.intent_snapshot
    with intent.turn_scope():
        intent.bind_turn(['again'], 'private B', 'a video', entry=True)
        intent.read(shared, 'again')
    bot._get_ctx = lambda: shared; bot._get_graph = lambda: object()
    seen = []
    def consume(task, sess, user, uname, lang, ctx, graph):
        seen.append(intent._bound('again', '', ''))
        intent.read(ctx, 'again')
    bot._run_task_inner = consume
    before = intent.current_turn()
    bot._execute_task(restored)
    assert seen == [('again', 'private A', 'a photo')]
    assert len(calls) == 2 and intent.current_turn() is before
    assert T._Task.from_dict({'task_id': 'legacy', 'chat_id': 101, 'user_text': 'old'}).intent_snapshot == {}

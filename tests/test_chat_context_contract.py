from types import SimpleNamespace
from collections import deque
from pathlib import Path

import graph_fastpath as F
import tg_bot as T
from fakes import reads
from test_tg_incoming_durability import make_bot, hold_worker
from tg_resolve import _batch_reply_id


def test_a_photo_question_is_never_intake_even_when_the_classifier_also_says_presentation():
    ctx = SimpleNamespace(**reads(read=lambda t: {
        'presents_photo': True, 'is_question': True, 'about_picture': True, 'needs_tool': False}))
    state = {'image_data': b'photo', 'vision_summary': 'A person in a photo.', 'picture_question': True}
    assert not F._is_photo_intake(ctx, state, 'Is she beautiful?', 'Она красивая?')


def test_scoped_context_does_not_inherit_desktop_attachment_lists():
    base = SimpleNamespace(session_memory=deque(), reference_images=['desktop-secret.png'],
                           recent_image_paths=['desktop-other.png'], image_undo=['desktop-history.png'])
    scoped = T._scoped_ctx(base)
    assert scoped.reference_images == []
    assert scoped.recent_image_paths == [] and scoped.image_undo == []
    scoped.reference_images.append('telegram.png')
    assert base.reference_images == ['desktop-secret.png']


def test_incoming_message_identity_survives_later_navigation_and_recovery(tmp_path):
    bot = make_bot(tmp_path)
    hold_worker(bot, 101)
    bot._msg_id = {101: 700}
    bot._enqueue_item(101, {'type': 'photo', 'file_id': 'photo', 'caption': 'question'})
    bot._msg_id[101] = 999
    restored = make_bot(tmp_path)
    hold_worker(restored, 101)
    restored._msg_id = {101: 999}
    restored._restore_incoming([])
    assert restored._queues[101].get_nowait()['_message_id'] == 700


def test_reply_anchor_uses_the_actual_question_not_a_forward_or_navigation():
    question = {'type': 'text', 'text': 'Она красивая?', '_message_id': 10}
    photo = {'type': 'photo', 'file_id': 'x', '_message_id': 11, 'forwarded': True}
    quote = {'type': 'text', 'text': 'Но я не оч', 'forwarded': True, '_message_id': 12}
    back = {'type': 'text', 'text': T._b('back', 'ru'), '_message_id': 99}
    assert _batch_reply_id([question, photo, quote]) == 10
    assert _batch_reply_id([photo, back]) == 11


def test_graph_keeps_the_picture_question_flag_between_nodes():
    from langgraph.graph import StateGraph, END
    from models import AgentState
    graph = StateGraph(AgentState)
    graph.add_node('look', lambda s: {'picture_question': True})
    graph.set_entry_point('look'); graph.add_edge('look', END)
    assert graph.compile().invoke({'user_input': 'question'})['picture_question'] is True


def test_slow_turn_adds_memory_without_overwriting_the_newer_attachment():
    sess = T._Session(101)
    old = sess.begin_context_turn()
    new = sess.begin_context_turn()
    a = {'kind': 'user', 'text': 'older question', 'ts': 1}
    b = {'kind': 'user', 'text': 'newer question', 'ts': 2}
    assert sess.commit_context_turn(new, [], [b], [], [], {'last_image_path': 'new.jpg'})
    assert sess.commit_context_turn(old, [], [a], [], [], {'last_image_path': 'old.jpg'})
    assert sess.last_image_path == 'new.jpg'
    assert sess.get_tg_memory() == [a, b]


def test_clear_invalidates_old_commits_and_active_attachment_references():
    sess = T._Session(101)
    token = sess.begin_context_turn()
    sess.turn_image, sess.turn_album = 'old-image', ['old.jpg']
    sess.clear_context()
    assert not sess.commit_context_turn(token, [], [{'kind': 'vision', 'text': 'old'}], [], [],
                                        {'last_image_path': 'old.jpg'})
    assert not sess.turn_image and not sess.turn_album and not sess.get_tg_memory()
    assert sess.begin_context_turn(token[0]) is None


def test_current_attachment_survives_session_persistence_without_sharing_other_image_prompts(tmp_path):
    make_bot(tmp_path)
    sess = T._Session(101)
    art = T._log_image(sess, str(tmp_path / 'art.png'), prompt='anime illustration')
    photo = T._log_image(sess, str(tmp_path / 'photo.png'), src='user')
    sess.turn_image = photo
    restored = T._Session(101, sess.to_dict())
    assert restored.turn_image == photo
    assert T._image_by_id(restored, art)['prompt'] == 'anime illustration'
    assert not T._image_by_id(restored, photo)['prompt']


def test_late_forwarded_media_cannot_restore_cleared_history(tmp_path):
    bot = make_bot(tmp_path)
    sess = bot._get_session(101); generation = sess.context_generation
    sess.clear_context()
    bot._commit_media_turn(101, 'old transcript', 'old retelling', True, generation)
    assert not sess.get_history()


def test_clear_during_forwarded_batch_transcription_drops_the_actual_late_result(tmp_path):
    bot = make_bot(tmp_path); sess = bot._get_session(101)
    bot._get_ctx = lambda: SimpleNamespace()
    def transcribe(*a, **kw):
        sess.clear_context()
        return 'Old private transcript'
    bot._transcribe = transcribe
    assert bot._fwd_batch(101, [{'type': 'fwd_voice', 'media': 'voice', 'file_id': 'old'}]) == ''
    assert not sess.fwd_transcript and not sess.get_history() and not sess.turn_image


def test_clear_during_single_forward_transcription_drops_the_actual_late_result(tmp_path):
    bot = make_bot(tmp_path); sess = bot._get_session(101)
    bot._get_ctx = lambda: SimpleNamespace()
    bot.reads = SimpleNamespace(**reads())
    bot.fwdv_intent = lambda text: ''
    def transcribe(*a, **kw):
        sess.clear_context()
        return 'Old private transcript'
    bot._transcribe = transcribe
    bot._resolve_and_push(101, [{'type': 'fwd_voice', 'media': 'voice', 'file_id': 'old'}])
    assert not sess.fwd_transcript and not sess.get_history() and not sess.turn_image


def test_late_forwarded_photo_does_not_inherit_the_waiting_questions_older_picture(tmp_path):
    bot = make_bot(tmp_path)
    previous = T._Task(task_id='question', chat_id=101, user_text='What is here?',
                       image_id='older-current-photo', image_selection='context')
    bot._backend = SimpleNamespace(service_order=lambda: [previous], drop_task=lambda tid: True)
    incoming = T._Task(task_id='photo', chat_id=101, user_text='[forwarded photo]',
                       image_path=str(tmp_path / 'new.png'), image_is_photo=True, image_selection='upload')
    assert bot._join_waiting(101, incoming)
    assert not incoming.image_id and incoming.image_path.endswith('new.png')


def test_forwarded_media_rewrites_keep_the_own_questions_reply_anchor(tmp_path):
    for count in (1, 2):
        bot = make_bot(tmp_path / str(count))
        bot._get_ctx = lambda: SimpleNamespace()
        bot.reads = SimpleNamespace(**reads())
        bot.fwdv_intent = lambda text: ''
        bot._transcribe = lambda *a, **kw: 'Forwarded speaker said hello'
        question = {'type': 'text', 'text': 'What does the speaker mean?', '_message_id': 10}
        media = [{'type': 'fwd_voice', 'media': 'voice', 'file_id': str(i),
                  'forwarded': True, '_message_id': 20+i} for i in range(count)]
        bot._resolve_and_push(101, [question, *media])
        task = bot._backend.pop(timeout=0)
        assert task.reply_to == 10
        assert 'What does the speaker mean?' in task.user_text


def test_explicit_picture_reference_survives_an_unrelated_newer_file(tmp_path):
    import os
    import graph as G
    picture = tmp_path / 'tg_old.png'; picture.write_bytes(b'photo')
    os.utime(picture, (1, 1))
    (tmp_path / 'new-notes.txt').write_text('unrelated')
    ctx = SimpleNamespace(last_image_path=str(picture), image_pointed_at=True,
                          sandbox=SimpleNamespace(root=tmp_path),
                          **reads(read=lambda t: {'about_picture': True, 'is_question': True}))
    assert G.needs_relook(ctx, {'user_input': 'What is in this picture?'})


def test_registered_user_photo_can_be_reexamined_without_a_working_copy_filename(tmp_path):
    import graph as G
    picture = tmp_path / 'tg_photo.png'; picture.write_bytes(b'photo')
    ctx = SimpleNamespace(last_image_path=str(picture), image_origin='user',
                          **reads(read=lambda t: {'about_picture': False, 'is_question': True}))
    assert G.needs_relook(ctx, {'user_input': 'Which item costs more?'})


def test_visual_memory_identifies_its_picture_and_does_not_claim_old_art_is_current():
    import threading
    from models import Context
    ctx = SimpleNamespace(memory_lock=threading.Lock(), session_memory=deque(),
                          current_image_id='art-1', last_image_path='/art.png', turn_id='turn-1')
    Context.remember(ctx, 'vision', 'An anime girl.', {'has_image': True})
    ctx.current_image_id = 'photo-2'; ctx.last_image_path = '/photo.png'
    text = Context.memory_text(ctx)
    assert ctx.session_memory[0]['meta']['attachment_id'] == 'art-1'
    assert 'art-1' in text and 'historical' in text and 'current' not in text


def test_target_is_captured_when_text_arrives_not_when_the_task_starts(tmp_path):
    bot = make_bot(tmp_path); hold_worker(bot, 101)
    sess = bot._get_session(101); sess.target_image = 'picture-a'
    bot._enqueue_item(101, {'type': 'text', 'text': 'Look closely'})
    sess.target_image = 'picture-b'
    item = bot._queues[101].get_nowait()
    assert item['image_id'] == 'picture-a' and sess.target_image == 'picture-b'


def test_duplicate_incoming_does_not_consume_a_new_image_gesture(tmp_path):
    bot = make_bot(tmp_path); hold_worker(bot, 101)
    sess = bot._get_session(101); sess.target_image = 'picture-a'
    bot._enqueue_item(101, {'type': 'text', 'text': 'question', '_inbox_id': 'same'})
    sess.target_image = 'picture-b'
    bot._enqueue_item(101, {'type': 'text', 'text': 'question', '_inbox_id': 'same'})
    assert sess.target_image == 'picture-b'
    assert bot._queues[101].qsize() == 1


def test_fresh_photo_consumes_an_old_gesture_without_selecting_it(tmp_path):
    bot = make_bot(tmp_path); hold_worker(bot, 101)
    sess = bot._get_session(101); sess.target_image = 'older-photo'
    bot._enqueue_item(101, {'type': 'photo', 'file_id': 'new-upload'})
    item = bot._queues[101].get_nowait()
    assert not sess.target_image and not item.get('image_id')


def test_recovered_task_does_not_reselect_a_newer_current_image(tmp_path):
    bot = make_bot(tmp_path)
    sess = bot._get_session(101)
    token = sess.begin_context_turn()
    task = T._Task(task_id='old', chat_id=101, user_text='hello',
                   context_generation=token[0], context_revision=token[1])
    sess.turn_image = 'newer-photo'
    restored = T._Task.from_dict(task.to_dict())
    bot._persist_task(restored)
    assert restored.image_id == ''


def test_compaction_does_not_discard_a_concurrently_completed_turn():
    sess = T._Session(101)
    before = [{'role': 'user', 'content': 'first'}]
    interject = {'role': 'assistant', 'content': 'newer answer'}
    own = [{'role': 'assistant', 'content': 'slow answer'}]
    sess.set_history(before + [interject])
    token = sess.begin_context_turn()
    sess.commit_history_turn(token, before, own, [{'role': 'system', 'content': 'summary'}])
    assert sess.get_history() == before + [interject] + own
    sess.clear_context()
    assert not sess.commit_history_turn(token, before, own)
    assert sess.get_history() == []


def test_missing_explicit_attachment_fails_without_invoking_the_graph(tmp_path):
    bot = make_bot(tmp_path)
    bot._get_ctx = lambda: SimpleNamespace()
    bot._get_graph = lambda: SimpleNamespace(invoke=lambda *a: (_ for _ in ()).throw(AssertionError('wrong image')))
    sent = []
    bot._send_text = lambda cid, text, **kw: sent.append(text)
    bot._execute_task(T._Task(task_id='missing', chat_id=101, user_text='what is here?',
                              image_id='deleted-picture', reply_to=700))
    assert sent == [T._t('img_gone', bot._lang(bot._get_session(101)))]


def test_album_reply_identity_comes_from_its_caption_not_later_navigation(tmp_path):
    bot = make_bot(tmp_path); hold_worker(bot, 101)
    bot.album_collect_s = 60
    bot._msg_id = {101: 17}
    bot._buffer_album(101, 'album', 'a', 'Compare these pictures')
    bot._msg_id[101] = 18
    bot._buffer_album(101, 'album', 'b', '')
    bot._albums['album']['timer'].cancel()
    bot._msg_id[101] = 99
    bot._flush_album('album')
    item = bot._queues[101].get_nowait()
    assert item['_message_id'] == 17 and item['file_ids'] == ['a', 'b']


def test_disk_memory_cannot_resurrect_facts_forgotten_by_a_newer_turn(tmp_path):
    import json
    import threading
    from models import Context
    bot = make_bot(tmp_path)
    shared = Context(models=SimpleNamespace(), transcription_cache={}, cache_file=tmp_path / 'cache.json',
                     asr_lock=threading.Lock(), tts_lock=threading.Lock())
    bot.reads = SimpleNamespace(**reads())
    bot._get_ctx = lambda: shared
    bot._get_graph = lambda: object()
    captured = []
    archived = []
    def capture(task, sess, user, uname, lang, ctx, graph):
        import context_v2 as C
        captured.append(ctx)
        archived.append((C._archive_dir(), C.archive_put('isolated conversation')))
    bot._run_task_inner = capture
    sess = bot._get_session(101)
    sess.tg_facts = [{'ts': 1, 'text': 'Old address'}]
    directory = T._MEMORY_DIR / 'chat_101'
    shared.pinned_facts = list(sess.tg_facts); shared.save_memory(directory)
    bot._execute_task(T._Task(task_id='a', chat_id=101, user_text='hello'))
    bot._execute_task(T._Task(task_id='b', chat_id=101, user_text='hello again'))
    old, new = captured
    assert Path(archived[0][0]) == directory / 'archive' / '0'
    new.pinned_facts = []; new.save_memory(directory)
    old.pin_fact('New preference'); old.save_memory(directory)
    assert [f['text'] for f in json.loads((directory / 'facts.json').read_text('utf8'))] == ['New preference']
    sess.clear_context(); T._delete_chat_facts_file(101)
    old.save_memory(directory)
    assert not (directory / 'facts.json').exists()


def test_clear_removes_all_reloadable_memory_files(tmp_path):
    make_bot(tmp_path)
    directory = T._MEMORY_DIR / 'chat_101'; directory.mkdir(parents=True)
    for name in ('facts.json', 'session_memory.json', 'summary.json'):
        (directory / name).write_text('{}')
    T._delete_chat_facts_file(101)
    assert list(directory.iterdir()) == []


def test_context_archives_are_isolated_by_chat_and_clear_generation(tmp_path):
    import context_v2 as C
    with C.archive_scope(tmp_path / 'chat-a' / 'generation-0'):
        aid = C.archive_put('Private older conversation')
        assert 'Private older conversation' in C.recall(aid)
    with C.archive_scope(tmp_path / 'chat-b' / 'generation-0'):
        assert C.archive_get(aid) is None
    with C.archive_scope(tmp_path / 'desktop'):
        assert C.archive_get(aid) is None
    with C.archive_scope(tmp_path / 'chat-a' / 'generation-1'):
        assert C.archive_get(aid) is None
    with C.archive_scope(tmp_path / 'chat-a' / 'generation-0'):
        late = C.archive_put('A late task after clear')
    with C.archive_scope(tmp_path / 'chat-a' / 'generation-1'):
        assert C.archive_get(late) is None

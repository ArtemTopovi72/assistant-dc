"""An album sent with a video request reaches the agent whole: the cat on the second picture was
lost because only the first one rode ctx.recent_image_paths (duo scenario, 2026-10-19)."""
import os
import sys
import tempfile
import threading
import time
import pytest

os.environ.setdefault("F5_TEST_RUN", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA = tempfile.mkdtemp(prefix="albumvideo_")
import logging  # noqa: E402
logging.basicConfig(level=logging.CRITICAL)
import tg_bot as T  # noqa: E402
T.redirect_data_dir(_DATA)
import graph as graph_mod  # noqa: E402
from PIL import Image  # noqa: E402

SEEN = []


class FakeCtx:
    def __init__(self):
        self.session_memory = []
        self.pinned_facts = []
        self.cancel_event = threading.Event()
        self.last_image_path = ""
        self.last_image_prompt = ""
        self.stage_callback = None
        self.total_user_turns = 0
        self.tts_disabled = True
        self.memory_lock = threading.Lock()

    def load_memory(self, *a, **k):
        pass

    def memory_text(self):
        return ""

    def set_stage(self, s):
        pass


class StubGraph:
    def __init__(self, ctx):
        self.ctx = ctx

    def invoke(self, base):
        SEEN.append((base.get("user_input", ""), getattr(self.ctx, "last_image_path", None),
                     list(getattr(self.ctx, "recent_image_paths", None) or []),
                     list(getattr(self.ctx, "reference_images", None) or [])))
        return {"final_answer": "ok", "messages": base.get("messages", [])}


def test_every_picture_of_an_album_reaches_the_agent():
    import uuid
    bot = T.TelegramBot("123:TEST", lambda: FakeCtx(), lambda: object(), lambda: {"messages": []},
                        silent_mode=True)
    bot.graph_builder = lambda ctx: StubGraph(ctx)
    bot._backend = T.InMemoryBackend()
    bot._send_text = lambda *a, **kw: 1
    bot._send_get_id = lambda *a, **kw: 1
    bot._edit_text = lambda *a, **kw: None
    bot._delete = lambda *a, **kw: None
    bot._api_post = lambda *a, **k: {}
    bot._api_get = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    bot._running = True
    threading.Thread(target=bot._consumer_loop, daemon=True).start()

    cid = 9_300_517
    bot._user_store.put(T._User(chat_id=cid, name="A", status="approved", is_admin=False))
    sess = bot._get_session(cid)
    sess.clear_context()
    sess.lang, sess.lang_chosen, sess.reg_state = "ru", True, ""
    bot._store.put(sess)
    d = tempfile.mkdtemp(prefix="album_")
    paths = []
    for color in ("red", "blue"):
        p = os.path.join(d, color + ".png")
        Image.new("RGB", (64, 64), color).save(p)
        paths.append(p)
    task = T._Task(task_id=str(uuid.uuid4()), chat_id=cid, user_text="animate them together",
                   image_path=paths[0], album_paths=paths[1:], enqueue_ts=time.time())
    with bot._task_lock:
        bot._pending_journal[task.task_id] = task
    bot._backend.push(task)
    end = time.monotonic() + 15
    while time.monotonic() < end and not SEEN:
        time.sleep(0.1)
    assert SEEN, "the request never reached the agent"
    assert SEEN[-1][2] == paths, SEEN[-1]


@pytest.mark.parametrize('rerun', ['voice', 'plan'])
@pytest.mark.parametrize('legacy', [False, True])
def test_the_album_survives_the_voice_question_rerun(rerun, legacy):
    """The request is re-run as bare text after «стандартные голоса»: both heroes still ride."""
    import uuid
    SEEN.clear()
    bot = T.TelegramBot("123:TEST", lambda: FakeCtx(), lambda: object(), lambda: {"messages": []},
                        silent_mode=True)
    bot.graph_builder = lambda ctx: StubGraph(ctx)
    bot._backend = T.InMemoryBackend()
    for name in ("_send_text", "_send_get_id"):
        setattr(bot, name, lambda *a, **kw: 1)
    bot._edit_text = bot._delete = lambda *a, **kw: None
    bot._api_post = bot._api_get = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    bot._running = True
    threading.Thread(target=bot._consumer_loop, daemon=True).start()
    cid = 9_300_518
    bot._user_store.put(T._User(chat_id=cid, name="A", status="approved", is_admin=False))
    sess = bot._get_session(cid)
    sess.clear_context()
    sess.lang, sess.lang_chosen, sess.reg_state = "ru", True, ""
    d = tempfile.mkdtemp(prefix="album2_")
    paths = []
    for color in ("red", "blue"):
        p = os.path.join(d, color + ".png")
        Image.new("RGB", (64, 64), color).save(p)
        paths.append(p)
    first_id = T._log_image(sess, paths[0], label="first", src="user")
    bot._store.put(sess)

    def push(text="video of the two", **kw):
        t = T._Task(task_id=str(uuid.uuid4()), chat_id=cid, user_text=text,
                    enqueue_ts=time.time(), **kw)
        with bot._task_lock:
            bot._pending_journal[t.task_id] = t
        n = len(SEEN)
        bot._backend.push(t)
        end = time.monotonic() + 15
        while time.monotonic() < end and len(SEEN) == n:
            time.sleep(0.1)

    push(image_path=paths[0], album_paths=paths[1:])
    time.sleep(1.5)                                                # the turn saves its session after the agent
    sess = bot._get_session(cid)
    if rerun == 'voice':
        sess.voice_pending = "video of the two"
        sess.voice_pending_images = [] if legacy else list(paths)
        sess.voice_choice = "default"                            # «стандартные» pressed
    else:
        sess.video_plan_request = "video of the two"
        sess.video_plan_ok = [{'prompt': 'the two together'}]      # approved plan consumed before image setup
        sess.video_plan_images = [] if legacy else list(paths)
        sess.video_plan_image = first_id
        sess.voice_choice = "default"
    if legacy:
        sess.turn_album = paths[1:]                               # older versions stored only the extra images
    else:
        # An unrelated message cannot spend the pending video identity or inherit its album.
        push(text="What is 2 plus 2?")
        assert SEEN[-1][1:4] == (None, [], []), SEEN[-1]
    sess.target_image = first_id
    bot._store.put(sess)
    push()                                                         # the bare-text rerun
    assert SEEN[-1][1] == paths[0] and SEEN[-1][2] == paths, SEEN[-1]
    assert SEEN[-1][3] == paths
    assert sess.turn_album == paths

    # A fresh attachment has its own identity, even if a voice choice remains armed.
    fresh = os.path.join(d, 'green.png')
    Image.new('RGB', (64, 64), 'green').save(fresh)
    sess.voice_choice = 'default'
    push(image_path=fresh, image_is_photo=True)
    assert SEEN[-1][1:4] == (fresh, [fresh], []), SEEN[-1]


@pytest.mark.parametrize('offer', ['voice', 'plan'])
def test_pending_video_offer_persists_its_own_images_and_enqueues_its_primary(tmp_path, offer):
    from test_tg_incoming_durability import make_bot
    bot = make_bot(tmp_path)
    sess = bot._get_session(101)
    paths = []
    for color in ('red', 'blue'):
        path = tmp_path / f'{color}.png'
        Image.new('RGB', (64, 64), color).save(path)
        paths.append(str(path))
    first = T._log_image(sess, paths[0], src='user')
    sess.turn_image = T._log_image(sess, paths[1], src='user')
    queued = []
    bot._enqueue_item = lambda cid, item: queued.append(item)
    bot._busy_note = lambda *a: None
    bot._voice_from_request_link = lambda *a: False
    bot.text_model = lambda *a, **kw: '{"lines": ["The two together"]}'
    if offer == 'voice':
        bot._offer_voices(101, sess, 'ru', 'video of the two', 2, images=paths)
    else:
        bot._offer_video_plan(101, sess, 'ru', 'video of the two',
                              [{'text': 'The two together', 'sec': 1}], images=paths)
    paths.append('a later unrelated path')
    restored = bot._store.get(101)
    saved = restored.voice_pending_images if offer == 'voice' else restored.video_plan_images
    assert saved == paths[:2]
    if offer == 'voice':
        bot._voices_go(101, sess, 'ru')
    else:
        bot._cb_video_plan(101, 'vp:go')
    assert queued[0]['image_id'] == first
    sess.clear_context()
    assert sess.voice_pending_images == sess.video_plan_images == []


@pytest.mark.parametrize('voice_choice', ['', 'own'])
@pytest.mark.parametrize('has_seed', [False, True])
def test_approved_video_plan_restores_its_own_original_clip(tmp_path, voice_choice, has_seed):
    import uuid
    from test_tg_incoming_durability import make_bot
    bot = make_bot(tmp_path)
    seen = []
    class Graph:
        def __init__(self, ctx): self.ctx = ctx
        def invoke(self, base):
            seen.append((self.ctx.continue_tail, self.ctx.continue_src, self.ctx.voice_choice,
                         self.ctx.last_image_path))
            return {'final_answer': 'ok', 'messages': base.get('messages', [])}
    bot._get_ctx = lambda: FakeCtx()
    bot._get_graph = lambda: object()
    bot._get_base_state = lambda: {'messages': []}
    bot.graph_builder = lambda ctx: Graph(ctx)
    bot.text_model = lambda *a, **kw: '{"lines": ["Next"]}'
    bot._backend = T.InMemoryBackend()
    bot._send_get_id = lambda *a, **k: 1
    bot._edit_text = bot._delete = lambda *a, **k: None
    bot._running = True
    threading.Thread(target=bot._consumer_loop, daemon=True).start()
    tail, source = tmp_path / 'tail.mp4', tmp_path / 'original.mp4'
    tail.write_bytes(b'tail'); source.write_bytes(b'original')
    seed, stale = tmp_path / 'seed.png', tmp_path / 'stale.png'
    Image.new('RGB', (64, 64), 'red').save(seed)
    Image.new('RGB', (64, 64), 'blue').save(stale)
    sess = bot._get_session(101)
    sess.video_plan_request = 'Продолжи этот ролик'
    sess.video_plan_ok = ['говорит одну фразу']
    sess.video_plan_continuation = {'tail': str(tail), 'src': str(source), 'people': []}
    sess.video_plan_images = [str(seed)] if has_seed else []
    sess.target_image = T._log_image(sess, str(stale), src='user')
    sess.voice_choice = voice_choice
    bot._store.put(sess)
    task = T._Task(task_id=str(uuid.uuid4()), chat_id=101, user_text=sess.video_plan_request,
                   image_id=sess.target_image, image_path=str(stale), enqueue_ts=time.time())
    with bot._task_lock: bot._pending_journal[task.task_id] = task
    bot._backend.push(task)
    end = time.monotonic() + 15
    while time.monotonic() < end and not seen: time.sleep(0.1)
    assert seen == [(str(tail), str(source), voice_choice or 'default', str(seed) if has_seed else None)]


def test_video_plan_persists_and_clears_its_continuation(tmp_path):
    from test_tg_incoming_durability import make_bot
    bot = make_bot(tmp_path)
    sess = bot._get_session(101)
    bot.text_model = lambda *a, **kw: '{"lines": ["Next"]}'
    continuation = {'tail': 'tail.mp4', 'src': 'source.mp4', 'people': ['person.png']}
    bot._offer_video_plan(101, sess, 'ru', 'Продолжи ролик',
                          [{'text': 'Next', 'sec': 5}], continuation=continuation)
    assert bot._store.get(101).video_plan_continuation == continuation
    bot._cb_video_plan(101, 'vp:x')
    assert bot._store.get(101).video_plan_continuation == {}


def test_video_plan_stores_the_reference_used_by_the_tool(tmp_path):
    import uuid
    from test_tg_incoming_durability import make_bot
    bot = make_bot(tmp_path)
    bot.text_model = lambda *a, **kw: '{"lines": ["Next"]}'
    source, tail, seed = (tmp_path / name for name in ('source.mp4', 'tail.mp4', 'seed.png'))
    source.write_bytes(b'source'); tail.write_bytes(b'tail')
    Image.new('RGB', (64, 64), 'red').save(seed)
    class Graph:
        def __init__(self, ctx): self.ctx = ctx
        def invoke(self, base):
            self.ctx.continue_tail, self.ctx.continue_src = str(tail), str(source)
            return {'final_answer': 'План ниже', 'messages': base.get('messages', []),
                    'video_plan': [{'text': 'Next', 'sec': 5}], 'video_plan_images': [str(seed)]}
    bot._get_ctx = lambda: FakeCtx()
    bot._get_graph = lambda: object()
    bot._get_base_state = lambda: {'messages': []}
    bot.graph_builder = lambda ctx: Graph(ctx)
    bot._backend = T.InMemoryBackend()
    bot._send_get_id = lambda *a, **k: 1
    bot._edit_text = bot._delete = lambda *a, **k: None
    bot._running = True
    threading.Thread(target=bot._consumer_loop, daemon=True).start()
    task = T._Task(task_id=str(uuid.uuid4()), chat_id=101, user_text='Продолжи ролик', enqueue_ts=time.time())
    with bot._task_lock: bot._pending_journal[task.task_id] = task
    bot._backend.push(task)
    end = time.monotonic() + 15
    while time.monotonic() < end and not bot._get_session(101).video_plan: time.sleep(0.1)
    sess = bot._get_session(101)
    assert sess.video_plan_images == [str(seed)]
    assert sess.video_plan_continuation['src'] == str(source)


@pytest.mark.parametrize('missing', ['source', 'seed'])
def test_plan_with_missing_source_does_not_generate_from_another_file(tmp_path, missing):
    import uuid
    from test_tg_incoming_durability import make_bot
    bot = make_bot(tmp_path)
    bot._get_ctx = lambda: FakeCtx()
    bot._get_graph = lambda: object()
    bot._get_base_state = lambda: {'messages': []}
    seen, sent = [], []
    class Graph:
        def invoke(self, base): seen.append(base)
    bot.graph_builder = lambda ctx: Graph()
    bot._backend = T.InMemoryBackend()
    bot._send_text = lambda cid, text, **kw: sent.append(text)
    bot._running = True
    threading.Thread(target=bot._consumer_loop, daemon=True).start()
    tail = tmp_path / 'tail.mp4'
    tail.write_bytes(b'tail')
    source = tmp_path / 'source.mp4'
    if missing != 'source': source.write_bytes(b'source')
    sess = bot._get_session(101)
    sess.video_plan_request = 'Продолжи ролик'
    sess.video_plan_ok = ['Продолжение']
    sess.video_plan_continuation = {'tail': str(tail), 'src': str(source)}
    sess.video_plan_images = [str(tmp_path / 'missing.png')] if missing == 'seed' else []
    bot._store.put(sess)
    task = T._Task(task_id=str(uuid.uuid4()), chat_id=101, user_text=sess.video_plan_request,
                   enqueue_ts=time.time())
    with bot._task_lock: bot._pending_journal[task.task_id] = task
    bot._backend.push(task)
    end = time.monotonic() + 15
    while time.monotonic() < end and not any('больше недоступ' in text for text in sent):
        time.sleep(0.1)
    assert any('больше недоступ' in text for text in sent)
    assert not seen

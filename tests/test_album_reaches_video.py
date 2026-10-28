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

"""An album sent with a video request reaches the agent whole: the cat on the second picture was
lost because only the first one rode ctx.recent_image_paths (duo scenario, 2026-10-19)."""
import os
import sys
import tempfile
import threading
import time

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
                     list(getattr(self.ctx, "recent_image_paths", None) or [])))
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


def test_the_album_survives_the_voice_question_rerun():
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

    def push(**kw):
        t = T._Task(task_id=str(uuid.uuid4()), chat_id=cid, user_text="video of the two",
                    enqueue_ts=time.time(), **kw)
        with bot._task_lock:
            bot._pending_journal[t.task_id] = t
        bot._backend.push(t)
        end = time.monotonic() + 15
        n = len(SEEN)
        while time.monotonic() < end and len(SEEN) == n:
            time.sleep(0.1)

    push(image_path=paths[0], album_paths=paths[1:])
    time.sleep(1.5)                                                # the turn saves its session after the agent
    sess = bot._get_session(cid)
    sess.voice_choice, sess.target_image = "default", first_id      # «стандартные» pressed
    bot._store.put(sess)
    push()                                                         # the bare-text rerun
    assert sorted(SEEN[-1][2]) == sorted(paths), SEEN[-1]

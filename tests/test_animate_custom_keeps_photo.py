"""✏️ Custom motion under 🎬 animates the photo chosen there.

Live 10-10: «Сделай всё как тут: поджигает, кипит…» typed after ✏️ reached
generate_video with 0 images -- the one-turn target slot was empty by the time
the words came, and the clip was invented from nothing. The picture chosen
under 🎬 (pending_animate_target) now rides the request.
"""
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
_DATA = tempfile.mkdtemp(prefix="animcustom_")
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


def test_custom_motion_uses_the_photo_chosen_under_animate():
    graph_mod.build_graph = lambda ctx: StubGraph(ctx)
    bot = T.TelegramBot("123:TEST", lambda: FakeCtx(), lambda: object(), lambda: {"messages": []},
                        silent_mode=True)
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

    cid = 9_300_515
    bot._user_store.put(T._User(chat_id=cid, name="A", status="approved", is_admin=False))
    sess = bot._get_session(cid)
    sess.clear_context()
    sess.lang, sess.lang_chosen, sess.reg_state = "ru", True, ""
    T._IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    photo = str(T._IMAGE_DIR / "kettle.jpg")
    other = str(T._IMAGE_DIR / "other.jpg")
    Image.new("RGB", (64, 64), "red").save(photo)
    Image.new("RGB", (64, 64), "blue").save(other)
    pid = T._log_image(sess, photo, label="to animate", src="user")
    oid = T._log_image(sess, other, label="later", src="bot")
    sess.pending_animate_target = pid
    sess.pending_prefix = "animate this photo: "
    sess.target_image, sess.turn_image = "", oid      # the slot emptied, another picture in play
    bot._store.put(sess)

    bot._dispatch({"update_id": 1, "message": {
        "message_id": 7, "chat": {"id": cid, "type": "private"}, "from": {"id": cid},
        "date": int(time.time()), "text": "поджигает, кипит, диктор открывает крышку"}})
    end = time.monotonic() + 15
    while time.monotonic() < end and not SEEN:
        time.sleep(0.1)
    assert SEEN, "the request never reached the agent"
    text, last, recent = SEEN[-1]
    assert text.startswith("animate this photo:")
    assert last == photo and recent == [photo], (last, recent)
    assert bot._get_session(cid).pending_animate_target == ""


if __name__ == "__main__":
    test_custom_motion_uses_the_photo_chosen_under_animate()
    print("ok")

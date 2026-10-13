"""🎬 on a bot result keeps the file: the result lives where it was made
(runtime/lettering_clean_*.png), the queue took it for a temp upload and deleted it,
and the second 🎬 under it said «Этой картинки больше нет».
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


def test_animating_a_bot_result_does_not_delete_it():
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

    cid = 9_300_516
    bot._user_store.put(T._User(chat_id=cid, name="A", status="approved", is_admin=False))
    sess = bot._get_session(cid)
    sess.clear_context()
    sess.lang, sess.lang_chosen, sess.reg_state = "ru", True, ""
    result = os.path.join(tempfile.mkdtemp(prefix="runtime_"), "lettering_clean_1.png")
    Image.new("RGB", (64, 64), "red").save(result)
    sess.pending_animate_target = T._log_image(sess, result, label="clean", src="bot")
    bot._store.put(sess)

    bot._cb_animate_preset(cid, "animate_preset:wave")
    end = time.monotonic() + 15
    while time.monotonic() < end and not SEEN:
        time.sleep(0.1)
    assert SEEN, "the request never reached the agent"
    time.sleep(1.0)
    assert os.path.exists(result), "the animated bot result was deleted"


if __name__ == "__main__":
    test_animating_a_bot_result_does_not_delete_it()
    print("ok")

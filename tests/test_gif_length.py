"""A GIF arrives as a video/mp4 document; its length is on `animation`, not unknown
(live 10-03: a 3 s GIF was offered as a «16667 мин» long video)."""
import os, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _sweep_stub  # noqa: F401
os.environ["F5_TEST_RUN"] = "1"
import tg_bot as T
T.redirect_data_dir(tempfile.mkdtemp(prefix="tg_gif_"))
bot = T.TelegramBot("1:TEST", lambda: None, lambda: None, lambda: {}, silent_mode=True)
bot._send_text = lambda *a, **k: None
bot._api_post = lambda m, p=None, **k: {"ok": True, "result": {"message_id": 1}}
bot._user_store.put(T._User(chat_id=7002, name="G", status="approved"))
items = []
bot._enqueue_item = lambda cid, item: items.append(item)
doc = {"file_id": "G1", "mime_type": "video/mp4"}
bot._dispatch({"message": {"chat": {"id": 7002, "type": "private"}, "from": {"id": 7002},
                           "date": int(time.time()), "document": doc,
                           "animation": dict(doc, duration=3)}})
assert items and items[0]["seconds"] == 3, items
print("PASS")

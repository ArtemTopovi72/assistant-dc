"""A keyboard label is never «send it as a file» (live 10-03: «📚 Документы» was
read as that request and the last picture came back as a document instead of
the Documents menu)."""
import os, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _sweep_stub  # noqa: F401
os.environ["F5_TEST_RUN"] = "1"
import tg_bot as T
import tg_dispatch as D
from tg_strings import _b
T.redirect_data_dir(tempfile.mkdtemp(prefix="tg_btnfile_"))
bot = T.TelegramBot("1:TEST", lambda: None, lambda: None, lambda: {}, silent_mode=True)
bot._send_text = lambda *a, **k: 1
bot._api_post = lambda m, p=None, **k: {"ok": True, "result": {"message_id": 1}}
bot._user_store.put(T._User(chat_id=7201, name="B", status="approved"))
asked = []
D._asks_as_file = lambda text: asked.append(text) or True     # the model says «yes»
bot._enqueue_item = lambda cid, item: None
bot._dispatch({"message": {"chat": {"id": 7201, "type": "private"}, "from": {"id": 7201},
                           "date": int(time.time()), "text": _b("library", "ru")}})
assert asked == [], asked
print("PASS")

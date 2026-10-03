"""A message typed while the bot was off carries its age, so «ты спишь?» from yesterday is not answered as now."""
import os, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _sweep_stub  # noqa: F401  the model's narrow reads, stubbed
os.environ["F5_TEST_RUN"] = "1"
import tg_bot as T
T.redirect_data_dir(tempfile.mkdtemp(prefix="tg_age_"))
bot = T.TelegramBot("1:TEST", lambda: None, lambda: None, lambda: {}, silent_mode=True)
bot._send_text = lambda *a, **k: None
bot._user_store.put(T._User(chat_id=7001, name="F", status="approved"))
class _B:
    def __init__(self): self.pushed = []
    def push(self, t): self.pushed.append(t)
    def chat_depth(self, c): return 0
    def depth(self): return len(self.pushed)
bot._api_post = lambda m, p=None, **k: {"ok": True, "result": {"message_id": 1}}
got = []
for age in (86400 * 1.2, 5):
    bot._backend = _B()
    bot._enqueue_item = lambda cid, item: None
    bot._dispatch_logged({"message": {"chat": {"id": 7001, "type": "private"}, "from": {"id": 7001},
                                      "date": int(time.time() - age), "text": "а ты спишь"}})
    bot._resolve_and_push(7001, [{"type": "text", "text": "а ты спишь"}])
    got.append(bot._backend.pushed[0].user_text if bot._backend.pushed else "")
print(got)
assert "28 h ago" in got[0] and "ago" not in got[1], got
print("PASS")

# «пришли эту картинку файлом» as a reply to the user's own photo
docs = []
bot._send_document = lambda cid, p, *a, **k: docs.append(p) or True
bot._dl_bytes = lambda fid: b"JPG" if fid == "F2" else None
bot._dispatch({"message": {"chat": {"id": 7001, "type": "private"}, "from": {"id": 7001}, "date": int(time.time()),
                           "text": "пришли эту картинку файлом",
                           "reply_to_message": {"message_id": 99, "photo": [{"file_id": "F1"}, {"file_id": "F2"}]}}})
assert docs and open(docs[0], "rb").read() == b"JPG", docs
print("PASS own photo as file")

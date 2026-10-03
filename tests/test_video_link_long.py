"""A long video by link is offered as the portioned job and asked about first
(live 10-03: an 81-min film got «посмотрю первые 10 минут, пришли файл»;
owner: «ты ток спроси надо это или нет»)."""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _sweep_stub  # noqa: F401
os.environ["F5_TEST_RUN"] = "1"
import tg_bot as T
import tg_links as L
T.redirect_data_dir(tempfile.mkdtemp(prefix="tg_lvlink_"))
bot = T.TelegramBot("1:TEST", lambda: None, lambda: None, lambda: {}, silent_mode=True)
sent = []
bot._send_text = lambda cid, text, **k: sent.append((text, k.get("keyboard"))) or 1
bot._activity.log = lambda *a, **k: None
CID = 7101
bot._user_store.put(T._User(chat_id=CID, name="L", status="approved"))
URL = "https://youtu.be/film"
_real = L.fetch_video
calls = []
L.fetch_video = lambda u, max_seconds=L.VIDEO_MAX_SECONDS: (
    calls.append(max_seconds) or {"too_long": True, "seconds": 81 * 60, "title": "Üç Kağıtçı"})
try:
    sess = bot._get_session(CID)
    handled = bot._read_links(object(), sess, CID, None, "глянь " + URL, [])
    assert handled is True
    text, kb = sent[-1]
    assert "81" in text, text
    datas = [b["callback_data"] for row in kb["inline_keyboard"] for b in row]
    assert any(d.startswith("lv:go:") for d in datas) and any(d.startswith("lv:skip:") for d in datas), datas
    lv = bot._get_session(CID).long_video
    assert lv and lv["url"] == URL, lv
finally:
    L.fetch_video = _real
print("PASS")

"""A long video by link is offered as the portioned job and asked about first
(an 81-min film must not get «посмотрю первые 10 минут, пришли файл»:
«ты ток спроси надо это или нет»)."""
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
calls = []
from seams import Over
bot.links = Over(L, fetch_video=lambda u, max_seconds=L.VIDEO_MAX_SECONDS: (
    calls.append(max_seconds) or {"too_long": True, "seconds": 81 * 60, "title": "Üç Kağıtçı"}))
sess = bot._get_session(CID)
handled = bot._read_links(object(), sess, CID, None, "глянь " + URL, [])
assert handled is True
text, kb = sent[-1]
assert "81" in text, text
datas = [b["callback_data"] for row in kb["inline_keyboard"] for b in row]
assert any(d.startswith("lv:go:") for d in datas) and any(d.startswith("lv:skip:") for d in datas), datas
lv = bot._get_session(CID).long_video
assert lv and lv["url"] == URL, lv
print("PASS")

# The site refused the download («Sign in to confirm you're not a bot»): the page shell was
# read instead and the agent retold web-search junk about «video <id>» (2026-10-17).
bot.links = Over(L, fetch_video=lambda u, max_seconds=L.VIDEO_MAX_SECONDS: {},
                 link_context=lambda t: (_ for _ in ()).throw(AssertionError("page read")))
scaffold = []
assert not bot._read_links(object(), bot._get_session(CID), CID, None, "перескажи " + URL, scaffold)
assert scaffold and "could not be downloaded" in scaffold[0] and "NOT watched" in scaffold[0], scaffold
print("PASS refused download")

# The download, ASR and look run before the task's status line: a forwarded link sat
# 49 s with no word (2026-10-17). The user is told at once.
sent.clear()
order = []
bot.links = Over(L, fetch_video=lambda u, max_seconds=L.VIDEO_MAX_SECONDS: order.append("fetch") or {})
bot._send_text = lambda cid, text, **k: order.append(text) or 1
bot._read_links(object(), bot._get_session(CID), CID, None, "смотри https://youtu.be/new", [])
assert order and order[0] == T._t("fwd_video_work", "ru") and order[1] == "fetch", order
print("PASS told before the download")

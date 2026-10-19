"""👥 Video with several characters: a card of slots, each a picture, an action and a voice."""
import io
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tg_bot as T
T.redirect_data_dir(tempfile.mkdtemp(prefix="tgtest_cast_"))
from PIL import Image

CID = 999101


def _png(color):
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, "PNG")
    return buf.getvalue()


def _bot():
    bot = T.TelegramBot("123:TEST", lambda: object(), lambda: object(), lambda: {}, silent_mode=True)
    bot.sent, bot.pushed = [], []
    bot._send_text = lambda cid, text, **kw: bot.sent.append((text, kw.get("keyboard")))
    bot._api_post = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    bot._dl_bytes = lambda fid: _png({"a": "red", "b": "blue"}.get(fid, "green"))
    bot._try_steer = lambda cid, task: False
    bot._write_inflight = lambda: None
    bot._queue_note = lambda *a, **k: None
    bot._backend.push = bot.pushed.append
    s = bot._get_session(CID); s.clear_context(); s.lang = "ru"; bot._store.put(s)
    return bot


def _cbs(kb):
    return [b["callback_data"] for row in (kb or {}).get("inline_keyboard", []) for b in row]


def test_a_slot_holds_a_picture_an_action_and_a_voice_and_go_sends_one_request():
    bot = _bot()
    s = bot._get_session(CID)
    bot._cast_start(CID, s, "ru")
    assert {"cast:pic:0", "cast:act:1", "cast:voice:2", "cast:go"} <= set(_cbs(bot.sent[-1][1]))

    bot._cb_cast(CID, "cast:pic:0")                      # an album fills the slots from 1 on
    s = bot._get_session(CID)
    bot._cast_take_photos(CID, s, "ru", {"type": "album", "file_ids": ["a", "b"]})
    s = bot._get_session(CID)
    assert all(s.cast_pics[:2]) and not s.cast_pics[2] and s.cast_state == "card"

    bot._cb_cast(CID, "cast:act:1")
    assert bot._cast_take_text(CID, bot._get_session(CID), "ru", "мяукает")
    s = bot._get_session(CID)
    voice = os.path.join(tempfile.mkdtemp(), "v.wav"); open(voice, "wb").write(b"x")
    s.anim_voices = ["", voice]
    bot._store.put(s)

    bot._cast_go(CID, bot._get_session(CID), "ru")
    task = bot.pushed[-1]
    assert task.user_text.startswith("video with several characters:")
    assert "Picture 2: мяукает." in task.user_text
    assert task.image_path == s.cast_pics[0] and task.album_paths == [s.cast_pics[1]]
    s = bot._get_session(CID)
    assert s.anim_voices == ["", voice] and s.voice_choice == "own" and not s.cast_state


def test_one_picture_is_not_a_cast():
    bot = _bot()
    bot._cast_start(CID, bot._get_session(CID), "ru")
    bot._cb_cast(CID, "cast:pic:0")
    bot._cast_take_photos(CID, bot._get_session(CID), "ru", {"type": "photo", "file_id": "a"})
    bot._cast_go(CID, bot._get_session(CID), "ru")
    assert not bot.pushed


def test_a_voice_sent_for_a_slot_returns_to_the_cast_card():
    bot = _bot()
    bot._cast_start(CID, bot._get_session(CID), "ru")
    bot._cb_cast(CID, "cast:voice:1")
    assert bot._get_session(CID).anim_voice_state == "collect"
    bot._anim_voice_added(CID, bot._get_session(CID), "ru")
    assert "cast:go" in _cbs(bot.sent[-1][1])

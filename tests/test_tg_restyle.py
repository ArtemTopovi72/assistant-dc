"""🎨 Restyle video in Telegram, on the REAL bot with the network and the render stubbed."""
import os, sys, tempfile, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tg_bot as T
T.redirect_data_dir(tempfile.mkdtemp(prefix="tgtest_restyle_"))
import tg_strings as S
import video_control as VC
import comfy_client, video

CID = 999088


def _bot():
    bot = T.TelegramBot("123:TEST", lambda: object(), lambda: object(), lambda: {}, silent_mode=True)
    bot.sent, bot.queued, bot.videos = [], [], []
    bot._send_text = lambda cid, text, **kw: bot.sent.append(text)
    bot._send_get_id = lambda cid, text, **kw: (bot.sent.append(text), 1)[1]
    bot._api_post = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    bot._dl_bytes = lambda fid: b"\x00\x00\x00\x18ftypmp42" + b"0" * 1000
    bot._enqueue_item = lambda cid, item: bot.queued.append(item)
    bot._send_video = lambda cid, path, caption="", **k: bot.videos.append((path, caption)) or True
    s = bot._get_session(CID); s.clear_context(); s.lang = "ru"; bot._store.put(s)
    return bot


def test_button_video_text_queues_a_restyle_task():
    bot = _bot()
    bot._resolve_and_push(CID, [{"type": "text", "text": S._BTN["restyle_btn"]["ru"]}])
    assert bot._get_session(CID).restyle_state == "want_video"
    assert bot._restyle_take_media(CID, bot._get_session(CID), "ru", {"video": {"file_id": "v1"}})
    s = bot._get_session(CID)
    assert s.restyle_state == "want_text" and os.path.exists(s.restyle_src)
    bot._resolve_and_push(CID, [{"type": "text", "text": "аниме"}])
    assert bot.queued == [{"type": "text", "text": "[restyle] аниме"}]
    assert bot._get_session(CID).restyle_state == ""


def test_menu_button_disarms():
    bot = _bot()
    bot._resolve_and_push(CID, [{"type": "text", "text": S._BTN["restyle_btn"]["ru"]}])
    bot._resolve_and_push(CID, [{"type": "text", "text": S._BTN["cover_btn"]["ru"]}])
    assert bot._get_session(CID).restyle_state == ""


def test_run_restyle_delivers(monkeypatch):
    bot = _bot()
    s = bot._get_session(CID)
    s.restyle_src = os.path.join(bot._restyle_dir(CID), "source.mp4")
    open(s.restyle_src, "wb").write(b"x"); bot._store.put(s)
    seen = []
    monkeypatch.setattr(VC, "restyle", lambda ctx, src, prompt: seen.append(prompt) or "out.mp4")
    ctx = types.SimpleNamespace(is_cancelled=lambda: False)
    assert bot._run_restyle(CID, "ru", "anime", ctx)
    assert seen == ["anime"] and bot.videos == [("out.mp4", "🎨 Перерисовал")]


def test_control_server_swaps_and_restores_url(monkeypatch):
    monkeypatch.setattr(VC, "_up", lambda url: True)      # already running: nothing started
    old = comfy_client.COMFY_URL
    with VC.on_control_server():
        assert comfy_client.COMFY_URL == video.COMFY_URL == VC.QI21_URL
    assert comfy_client.COMFY_URL == old and video.COMFY_URL == old

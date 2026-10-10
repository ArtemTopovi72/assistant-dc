"""🎙 Voices for 🎬 Animate + «видео моим голосом», on the REAL bot / tool handler."""
import os, sys, tempfile, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tg_bot as T
T.redirect_data_dir(tempfile.mkdtemp(prefix="tgtest_anv_"))
import tool_image_handlers as H
import video as V

CID = 999099


def _bot():
    bot = T.TelegramBot("123:TEST", lambda: object(), lambda: object(), lambda: {}, silent_mode=True)
    bot.sent = []
    bot._send_text = lambda cid, text, **kw: bot.sent.append((text, kw.get("keyboard")))
    bot._api_post = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    bot._dl_bytes = lambda fid: b"OggS" + b"0" * 500
    s = bot._get_session(CID); s.clear_context(); s.lang = "ru"; bot._store.put(s)
    return bot


def _cbs(kb):
    return [b["callback_data"] for row in (kb or {}).get("inline_keyboard", []) for b in row]


def test_voices_are_slots_left_to_right():
    """owner 10-10: «голоса в видео хочется видеть как конфигурируемые слоты»."""
    bot = _bot()
    bot._animate_ask_voices(CID, bot._get_session(CID), "ru")
    assert "animate_preset:wave" in _cbs(bot.sent[-1][1]) and "anv:yes" in _cbs(bot.sent[-1][1])
    bot._cb_anim_voices(CID, "anv:yes")
    text, kb = bot.sent[-1]
    assert "1. ▫️ стандартный" in text and {"anv:slot:0", "anv:slot:2", "anv:done"} <= set(_cbs(kb))
    bot._cb_anim_voices(CID, "anv:slot:1")                       # the middle person first
    assert "человека 2" in bot.sent[-1][0]
    assert bot._anim_voice_take_media(CID, bot._get_session(CID), "ru", {"voice": {"file_id": "a"}})
    s = bot._get_session(CID)
    assert not s.anim_voices[0] and s.anim_voices[1] and not s.anim_voices[2]
    assert "anv:clr:1" in _cbs(bot.sent[-1][1]) and s.anim_voice_state == ""
    assert not bot._anim_voice_take_media(CID, s, "ru", {"voice": {"file_id": "x"}})   # not armed
    old = s.anim_voices[1]
    bot._cb_anim_voices(CID, "anv:slot:1")                       # re-record replaces that slot
    bot._anim_voice_take_media(CID, bot._get_session(CID), "ru", {"voice": {"file_id": "b"}})
    s2 = bot._get_session(CID)
    assert s2.anim_voices[1] != old and not s2.anim_voices[0]
    bot._cb_anim_voices(CID, "anv:clr:1")
    assert not any(bot._get_session(CID).anim_voices)
    bot._cb_anim_voices(CID, "anv:done")
    assert any(c.startswith("animate_preset:") for c in _cbs(bot.sent[-1][1]))


def test_no_skips_to_presets():
    bot = _bot()
    bot._animate_ask_voices(CID, bot._get_session(CID), "ru")
    bot._cb_anim_voices(CID, "anv:no")
    assert any(c.startswith("animate_preset:") for c in _cbs(bot.sent[-1][1]))
    assert not bot._anim_voice_take_media(CID, bot._get_session(CID), "ru", {"voice": {"file_id": "x"}})


def _run_tool(monkeypatch, ctx, args):
    seen = {}
    monkeypatch.setattr(V, "engine_available", lambda ctx: (True, ""))
    monkeypatch.setattr(V, "generate_video",
                        lambda ctx, d, **kw: seen.update(desc=d, **kw) or {"path": None, "status": "fail"})
    import tools
    monkeypatch.setattr(tools, "_render_budget_exhausted", lambda *a, **k: None)
    H._handle_generate_video(ctx, {}, args)
    return seen


def test_tool_uses_collected_voices_once(monkeypatch, tmp_path):
    a, b = tmp_path / "v1.ogg", tmp_path / "v2.ogg"
    a.write_bytes(b"x"); b.write_bytes(b"x")
    ctx = types.SimpleNamespace(anim_voices=[str(a), str(b)], voice_ref="", set_stage=lambda s: None)
    seen = _run_tool(monkeypatch, ctx, {"description": "two people talk", "use_current_images": False})
    assert seen["audios"] == [str(a), str(b)] and "<Audio 2>" in seen["desc"]
    assert ctx.anim_voices == []


def test_empty_slot_keeps_its_person_default(monkeypatch, tmp_path):
    b = tmp_path / "v2.ogg"
    b.write_bytes(b"x")
    ctx = types.SimpleNamespace(anim_voices=["", str(b)], voice_ref="", set_stage=lambda s: None)
    seen = _run_tool(monkeypatch, ctx, {"description": "two people talk", "use_current_images": False})
    assert seen["audios"] == [str(b)] and "second from the left person speaks with the voice of <Audio 1>" in seen["desc"]


def test_my_voice_needs_a_clone(monkeypatch, tmp_path):
    ctx = types.SimpleNamespace(anim_voices=[], voice_ref="", set_stage=lambda s: None)
    monkeypatch.setattr(V, "engine_available", lambda ctx: (True, ""))
    import tools
    monkeypatch.setattr(tools, "_render_budget_exhausted", lambda *a, **k: None)
    out = H._handle_generate_video(ctx, {}, {"description": "я говорю привет", "use_my_voice": True,
                                             "use_current_images": False})
    assert "Клон голоса" in out
    ref = tmp_path / "me.wav"; ref.write_bytes(b"x")
    ctx.voice_ref = str(ref)
    seen = _run_tool(monkeypatch, ctx, {"description": "я говорю привет", "use_my_voice": True,
                                        "use_current_images": False})
    assert seen["audios"] == [str(ref)] and "<Audio 1>" in seen["desc"]


def test_menu_button_ends_collection():
    import tg_strings as S
    bot = _bot()
    bot._animate_ask_voices(CID, bot._get_session(CID), "ru")
    bot._cb_anim_voices(CID, "anv:slot:0")
    bot._resolve_and_push(CID, [{"type": "text", "text": S._BTN["cover_btn"]["ru"]}])
    assert bot._get_session(CID).anim_voice_state == ""


def test_video_link_is_a_voice_sample(monkeypatch):
    import tg_links
    bot = _bot()
    bot._animate_ask_voices(CID, bot._get_session(CID), "ru")
    bot._cb_anim_voices(CID, "anv:yes")
    bot._cb_anim_voices(CID, "anv:slot:0")
    bot._run_busy = lambda chat_id, fn, *a, **k: fn(*a)           # the job inline
    monkeypatch.setattr(tg_links, "fetch_video", lambda url, *a, **k: {"data": b"\x00" * 600})
    s = bot._get_session(CID)
    assert bot._anim_voice_take_link(CID, s, "ru", "https://www.youtube.com/watch?v=UhPKSA2UhnU")
    assert bot._get_session(CID).anim_voices[0]
    s = bot._get_session(CID)
    s.anim_voice_state = "collect"
    assert not bot._anim_voice_take_link(CID, s, "ru", "просто текст")       # plain text is not ours
    bot._cb_anim_voices(CID, "anv:done")
    assert not bot._anim_voice_take_link(CID, bot._get_session(CID), "ru", "https://youtu.be/x")   # not collecting

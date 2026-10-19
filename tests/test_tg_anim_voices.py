"""🎙 Voices for 🎬 Animate + «видео моим голосом», on the REAL bot / tool handler."""
import os, sys, tempfile, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tg_bot as T
T.redirect_data_dir(tempfile.mkdtemp(prefix="tgtest_anv_"))
import tool_image_handlers as H
import video as V
from tests.fakes import Over

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
    """Voices in the video are configurable slots, filled left to right."""
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


def _engine(seen):
    return Over(V, engine_available=lambda ctx: (True, ""),
                generate_video=lambda ctx, d, **kw: seen.update(desc=d, **kw) or {"path": None, "status": "fail"})


def _run_tool(ctx, args):
    seen = {}
    ctx.video_engine = _engine(seen)
    H._handle_generate_video(ctx, {}, args)
    return seen


def test_tool_uses_collected_voices_once(tmp_path):
    a, b = tmp_path / "v1.ogg", tmp_path / "v2.ogg"
    a.write_bytes(b"x"); b.write_bytes(b"x")
    ctx = types.SimpleNamespace(anim_voices=[str(a), str(b)], voice_ref="", set_stage=lambda s: None)
    seen = _run_tool(ctx, {"description": "two people talk", "use_current_images": False})
    assert seen["audios"] == [str(a), str(b)] and "<Audio 2>" in seen["desc"]
    assert ctx.anim_voices == []


def test_empty_slot_keeps_its_person_default(tmp_path):
    b = tmp_path / "v2.ogg"
    b.write_bytes(b"x")
    ctx = types.SimpleNamespace(anim_voices=["", str(b)], voice_ref="", set_stage=lambda s: None)
    seen = _run_tool(ctx, {"description": "two people talk", "use_current_images": False})
    assert seen["audios"] == [str(b)] and "second from the left person speaks with the voice of <Audio 1>" in seen["desc"]


def test_my_voice_needs_a_clone(tmp_path):
    ctx = types.SimpleNamespace(anim_voices=[], voice_ref="", set_stage=lambda s: None)
    ctx.video_engine = _engine({})
    out = H._handle_generate_video(ctx, {}, {"description": "я говорю привет", "use_my_voice": True,
                                             "use_current_images": False})
    assert "Клон голоса" in out
    ref = tmp_path / "me.wav"; ref.write_bytes(b"x")
    ctx.voice_ref = str(ref)
    seen = _run_tool(ctx, {"description": "я говорю привет", "use_my_voice": True,
                                        "use_current_images": False})
    assert seen["audios"] == [str(ref)] and "<Audio 1>" in seen["desc"]


def test_menu_button_ends_collection():
    import tg_strings as S
    bot = _bot()
    bot._animate_ask_voices(CID, bot._get_session(CID), "ru")
    bot._cb_anim_voices(CID, "anv:slot:0")
    bot._resolve_and_push(CID, [{"type": "text", "text": S._BTN["cover_btn"]["ru"]}])
    assert bot._get_session(CID).anim_voice_state == ""


def test_video_link_is_a_voice_sample():
    import tg_links
    bot = _bot()
    bot._animate_ask_voices(CID, bot._get_session(CID), "ru")
    bot._cb_anim_voices(CID, "anv:yes")
    bot._cb_anim_voices(CID, "anv:slot:0")
    bot._run_busy = lambda chat_id, fn, *a, **k: fn(*a)           # the job inline
    bot.links = Over(tg_links, fetch_video=lambda url, *a, **k: {"data": b"\x00" * 600})
    s = bot._get_session(CID)
    assert bot._anim_voice_take_link(CID, s, "ru", "https://www.youtube.com/watch?v=UhPKSA2UhnU")
    assert bot._get_session(CID).anim_voices[0]
    s = bot._get_session(CID)
    s.anim_voice_state = "collect"
    assert not bot._anim_voice_take_link(CID, s, "ru", "просто текст")       # plain text is not ours
    bot._cb_anim_voices(CID, "anv:done")
    assert not bot._anim_voice_take_link(CID, bot._get_session(CID), "ru", "https://youtu.be/x")   # not collecting


def _link_bot(is_voice):
    import tg_links
    bot = _bot()
    bot._run_busy = lambda chat_id, fn, *a, **k: fn(*a)
    bot.links = Over(tg_links, fetch_video=lambda url, *a, **k: {"data": b"\x00" * 600})
    bot.reads = types.SimpleNamespace(yes_no=lambda prompt, text: is_voice)
    bot.queued = []
    bot._enqueue_item = lambda cid, item: bot.queued.append(item)
    return bot


def test_link_named_as_voice_skips_the_card():
    """«Голос возьми отсюда: <ссылка>» in the request itself: the link's sound is slot 1
    and the clip reruns, no «send voices» card."""
    req = "Анимируй. Голос возьми отсюда: https://youtu.be/JCjqy3R4yQw"
    bot = _link_bot(True)
    bot._offer_voices(CID, bot._get_session(CID), "ru", req, 1)
    s = bot._get_session(CID)
    assert s.anim_voices[0] and s.voice_choice == "own"
    assert bot.queued == [{"type": "text", "text": req}]
    assert not any("anv:yes" in _cbs(kb) for _, kb in bot.sent)


def test_link_not_about_voice_still_asks():
    bot = _link_bot(False)
    bot._offer_voices(CID, bot._get_session(CID), "ru", "Анимируй, видео как здесь https://youtu.be/x", 1)
    assert "anv:yes" in _cbs(bot.sent[-1][1]) and not bot.queued

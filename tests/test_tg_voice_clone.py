"""🎙 Clone voice in Telegram, on the REAL bot object with the network stubbed."""
import os, sys, tempfile, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tg_bot as T
T.redirect_data_dir(tempfile.mkdtemp(prefix="tgtest_clone_"))
import tg_voice_clone as TV
import voice_clone as V

CID = 999077


class _Sync:
    def __init__(self, target, args=(), daemon=None, name=None): self.t, self.a = target, args
    def start(self): self.t(*self.a)


def _bot(monkeypatch):
    monkeypatch.setattr(TV.threading, "Thread", _Sync)
    bot = T.TelegramBot("123:TEST", lambda: object(), lambda: object(), lambda: {}, silent_mode=True)
    bot.sent, bot.pushed, bot.voices = [], [], []
    bot._send_text = lambda cid, text, **kw: bot.sent.append(text) or 1
    bot._send_get_id = lambda cid, text, **kw: (bot.sent.append(text), 1)[1]
    bot._api_post = lambda *a, **k: {}
    bot._api_get = lambda *a, **k: {}
    bot._backend = types.SimpleNamespace(push=lambda t: bot.pushed.append(t), depth=lambda: 1,
                                         name=lambda: "stub")
    bot._activity.log = lambda *a, **k: None
    bot._dl_bytes = lambda fid: b"OggS" + b"0" * 2000
    bot._send_voice_from_wav = lambda cid, wav, keyboard=None: bot.voices.append(wav) or True
    s = bot._get_session(CID); s.clear_context(); s.lang = "ru"; bot._store.put(s)
    return bot


def _press_clone(bot):
    bot._resolve_and_push(CID, [{"type": "text", "text": __import__("tg_strings")._BTN["clone_btn"]["ru"]}])


def test_full_flow_media_then_every_text_is_voiced(monkeypatch, tmp_path):
    bot = _bot(monkeypatch)
    ref = tmp_path / "ref.wav"; ref.write_bytes(b"RIFF")
    monkeypatch.setattr(V, "prepare_reference", lambda ctx, src, d, lang="ru": (str(ref), "привет это я"))
    said = []
    monkeypatch.setattr(V, "polish", lambda ctx, t: t.capitalize() + ".")
    def fake_speak(ctx, r, rt, text, d):
        said.append((r, rt, text)); p = os.path.join(d, "o.wav"); open(p, "wb").write(b"x"); return p
    monkeypatch.setattr(V, "speak", fake_speak)

    _press_clone(bot)
    assert bot._get_session(CID).clone_state == "want_audio"
    assert bot._clone_take_media(CID, bot._get_session(CID), "ru",
                                 {"video_note": {"file_id": "k1"}})
    s = bot._get_session(CID)
    assert s.clone_state == "want_text" and s.clone_ref == str(ref)
    bot._resolve_and_push(CID, [{"type": "text", "text": "подключи usb к ноутбуку"}])
    bot._resolve_and_push(CID, [{"type": "text", "text": "и ещё раз"}])
    assert [t for _, _, t in said] == ["Подключи usb к ноутбуку.", "И ещё раз."]
    assert said[0][1] == "привет это я" and len(bot.voices) == 2
    assert not bot.pushed                                  # never reached the agent


def test_any_button_leaves_the_flow(monkeypatch):
    bot = _bot(monkeypatch)
    _press_clone(bot)
    bot._resolve_and_push(CID, [{"type": "text", "text": __import__("tg_strings")._BTN["back"]["ru"]}])
    assert bot._get_session(CID).clone_state == ""


def test_text_while_waiting_for_audio_goes_to_the_agent(monkeypatch):
    bot = _bot(monkeypatch)
    _press_clone(bot)
    monkeypatch.setattr(V, "speak", lambda *a, **k: (_ for _ in ()).throw(AssertionError("voiced")))
    bot._resolve_and_push(CID, [{"type": "text", "text": "напиши стих про осень"}])
    assert not bot.voices and bot._get_session(CID).clone_state == "want_audio"


def test_bad_sample_explains_and_stays_armed(monkeypatch):
    bot = _bot(monkeypatch)
    monkeypatch.setattr(V, "prepare_reference",
                        lambda *a, **k: (_ for _ in ()).throw(V.CloneError("no_words")))
    _press_clone(bot)
    bot._clone_take_media(CID, bot._get_session(CID), "ru", {"voice": {"file_id": "v"}})
    assert T._t("clone_fail_no_words", "ru") in bot.sent
    assert bot._get_session(CID).clone_state == "want_audio"


def test_media_kinds():
    m = TV.VoiceCloneMixin._clone_media_of
    assert m({"audio": {"file_id": "a"}}) == ("a", ".mp3")
    assert m({"document": {"file_id": "d", "mime_type": "video/mp4", "file_name": "x.mov"}}) == ("d", ".mov")
    assert m({"document": {"file_id": "d", "mime_type": "application/pdf"}}) is None
    assert m({"photo": [{"file_id": "p"}]}) is None


def test_assistant_voice_takes_its_own_sample(monkeypatch, tmp_path):
    """🗣 is its own setting: its clip becomes the assistant's voice; 🎙 clone stays separate."""
    import audio
    bot = _bot(monkeypatch)
    press = lambda: bot._resolve_and_push(CID, [{"type": "text", "text": __import__("tg_strings")._BTN["my_voice"]["ru"]}])
    ref = tmp_path / "asst.wav"; ref.write_bytes(b"RIFF")
    monkeypatch.setattr(V, "prepare_reference", lambda ctx, src, d, lang="ru": (str(ref), "это я"))
    s = bot._get_session(CID); s.clone_ref, s.assistant_ref, s.voice_on = "other.wav", "", False; bot._store.put(s)
    press()                                                # arms: asks for a sample
    assert bot._get_session(CID).clone_state == "want_assistant" and "Пришли" in bot.sent[-1]
    assert bot._clone_take_media(CID, bot._get_session(CID), "ru", {"voice": {"file_id": "a"}})
    s = bot._get_session(CID)
    assert s.assistant_ref == str(ref) and s.clone_ref == "other.wav" and not s.clone_state
    assert not s.voice_on and "выключены" in bot.sent[-1]    # not switched on for the user: told + offered
    bot._cb_my_voice(CID, "myv:on")
    assert bot._get_session(CID).voice_on
    seen = []
    monkeypatch.setattr(audio, "synth_single_segment",
                        lambda ctx, i, actor, text, **k: seen.append((actor, ctx.custom_ref_wav, ctx.custom_ref_text)))
    bot._send_voice(object(), CID, "привет")
    assert seen == [("clone", str(ref), "это я")]
    bot._cb_my_voice(CID, "myv:reset")
    assert not bot._get_session(CID).assistant_ref
    s = bot._get_session(CID); s.assistant_ref = str(ref); bot._store.put(s)
    press()                                                  # second press: asks for a new one, keeps the old
    assert bot._get_session(CID).assistant_ref == str(ref) and bot._get_session(CID).clone_state == "want_assistant"


def test_video_link_is_the_sample(monkeypatch, tmp_path):
    """Owner 10-03: «сделать с голоса ютуба» -- a YouTube link while armed."""
    import tg_links
    bot = _bot(monkeypatch)
    ref = tmp_path / "ref.wav"; ref.write_bytes(b"RIFF")
    got = []
    monkeypatch.setattr(V, "prepare_reference",
                        lambda ctx, src, d, lang="ru": got.append(open(src, "rb").read()) or (str(ref), "привет это я"))
    monkeypatch.setattr(tg_links, "fetch_video",
                        lambda url: {"data": b"MP4DATA", "seconds": 30, "title": "t"})
    _press_clone(bot)
    bot._resolve_and_push(CID, [{"type": "text", "text": "https://youtube.com/shorts/5uI-Dae7-qs"}])
    s = bot._get_session(CID)
    assert got == [b"MP4DATA"] and s.clone_state == "want_text" and s.clone_ref == str(ref)
    assert not bot.pushed

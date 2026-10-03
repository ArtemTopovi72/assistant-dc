"""A clip where people talk asks once: own voice samples or the default voices
(user 2026-10-01: «не хватает вопроса про подкладывание голосов … и кнопки
поехать на дефолтных»). The answer reruns the same request."""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("F5_TEST_RUN", "1")
import tool_image_handlers as H
import video as V
import tools as T

V.engine_available = lambda ctx: (True, "")
T._render_budget_exhausted = lambda *a, **k: None
made = []
V.generate_video = lambda ctx, d, **k: made.append((d, k.get("audios"))) or {"status": "cancelled"}


def ctx(choice, voices=()):
    return types.SimpleNamespace(voice_choice=choice, anim_voices=list(voices), set_stage=lambda s: None,
                                 last_image_path=None, last_video_path=None)


def test_asks_before_rendering_people_talking():
    st = {}
    out = H._handle_generate_video(ctx(""), st, {"description": "two friends chat", "speakers": 2,
                                                 "use_current_images": False})
    assert "NOT MADE YET" in out and st["ask_voices"] == 2 and not made


def test_quoted_lines_ask_even_when_the_model_forgot_speakers():
    # live 10-02: two quoted lines, speakers=0 -> no question, default voices
    st = {}
    out = H._handle_generate_video(ctx(""), st, {"description": 'He yells "Иди ты!", the other "Сам иди!"',
                                                 "use_current_images": False})
    assert "NOT MADE YET" in out and st["ask_voices"] == 2


def test_default_choice_renders_without_asking():
    st = {}
    H._handle_generate_video(ctx("default"), st, {"description": "two friends chat", "speakers": 2,
                                                  "use_current_images": False})
    assert "ask_voices" not in st and made and not made[-1][1]


def test_own_voices_map_left_to_right(tmp_path):
    a, b = tmp_path / "v1.wav", tmp_path / "v2.wav"
    a.write_bytes(b"x"); b.write_bytes(b"x")
    H._handle_generate_video(ctx("own", [str(a), str(b)]), {}, {"description": "two friends chat",
                                                                "speakers": 2, "use_current_images": False})
    d, audios = made[-1]
    assert audios == [str(a), str(b)] and "left to right" in d


def test_scenery_and_other_hosts_never_ask():
    for c, sp in ((ctx(""), 0), (types.SimpleNamespace(set_stage=lambda s: None, anim_voices=[],
                                                        last_image_path=None, last_video_path=None), 2)):
        st = {}
        H._handle_generate_video(c, st, {"description": "rain", "speakers": sp, "use_current_images": False})
        assert "ask_voices" not in st


def test_buttons_rerun_the_request():
    import tempfile, tg_bot
    tg_bot.redirect_data_dir(tempfile.mkdtemp(prefix="tgtest_vq_"))
    bot = tg_bot.TelegramBot("123:TEST", lambda: None, lambda: None, lambda: {}, silent_mode=True)
    sent, queued = [], []
    bot._send_text = lambda cid, text, **kw: sent.append((text, kw)) or 1
    bot._enqueue_item = lambda cid, item: queued.append(item)
    sess = bot._get_session(5)
    bot._offer_voices(5, sess, "ru", "сделай видео где мы болтаем", 2)
    kb = sent[-1][1]["keyboard"]["inline_keyboard"][0]
    assert [b["callback_data"] for b in kb] == ["anv:yes", "anv:default"]
    bot._cb_anim_voices(5, "anv:default")
    assert queued[-1]["text"] == "сделай видео где мы болтаем"
    assert bot._get_session(5).voice_choice == "default"

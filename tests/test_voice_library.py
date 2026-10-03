"""📚 One voice library per chat: the last 2 unnamed voices are remembered,
named ones are kept, and every voice flow offers them as buttons."""
import os, sys, tempfile, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("F5_TEST_RUN", "1")
import tg_bot
import tg_voice_library as L


def _bot():
    tg_bot.redirect_data_dir(tempfile.mkdtemp(prefix="tgtest_vl_"))
    b = tg_bot.TelegramBot("123:TEST", lambda: None, lambda: None, lambda: {}, silent_mode=True)
    b.sent = []
    b._send_text = lambda cid, text, **kw: b.sent.append((text, kw)) or 1
    b._run_busy = lambda cid, fn, *a: fn(*a)
    return b


def _wav(d, n):
    p = os.path.join(d, f"{n}.wav"); open(p, "wb").write(b"x"); return p


def test_recent_two_and_named_kept(tmp_path):
    s = types.SimpleNamespace(voices=[])
    a, b, c = (_wav(tmp_path, n) for n in "abc")
    va = L.vl_add(s, a)
    L.vl_add(s, a, name="Степан")                     # named: kept for good
    L.vl_add(s, b); L.vl_add(s, c); L.vl_add(s, _wav(tmp_path, "d"))
    names = [v.get("name") for v in L.vl_list(s)]
    assert names[0] == "Степан" and len(names) == 3   # named + the last 2 recent
    assert L.vl_get(s, va["id"])["name"] == "Степан"


def test_name_flow_and_clip_buttons(tmp_path):
    bot = _bot()
    sess = bot._get_session(9)
    v = L.vl_add(sess, _wav(tmp_path, "s"))
    bot._store.put(sess)
    bot._cb_voice_library(9, f"vl:name:{v['id']}")
    sess = bot._get_session(9)
    assert bot._vl_take_name(9, sess, "ru", "Бабушка")
    assert L.vl_get(bot._get_session(9), v["id"])["name"] == "Бабушка"
    bot._enqueue_item = lambda *a: None
    bot._offer_voices(9, bot._get_session(9), "ru", "видео", 1)
    rows = bot.sent[-1][1]["keyboard"]["inline_keyboard"]
    assert any(b["callback_data"] == f"anv:lib:{v['id']}" for r in rows for b in r)
    bot._cb_anim_voices(9, f"anv:lib:{v['id']}")
    assert bot._get_session(9).anim_voices == [v["ref"]]


def test_library_voice_becomes_assistant(tmp_path, monkeypatch):
    bot = _bot()
    sess = bot._get_session(10)
    v = L.vl_add(sess, _wav(tmp_path, "t"), text="привет")
    bot._store.put(sess)
    bot._cb_voice_library(10, f"vl:asst:{v['id']}")
    assert bot._get_session(10).assistant_ref == v["ref"]

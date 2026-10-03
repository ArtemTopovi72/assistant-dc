"""📚 One voice library per chat: the last 5 unnamed voices are remembered,
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


def test_recent_five_and_named_kept(tmp_path):
    s = types.SimpleNamespace(voices=[])
    va = L.vl_add(s, _wav(tmp_path, "a"))
    L.vl_add(s, va["ref"], name="Степан")             # named: kept for good
    for n in "bcdefgh":
        L.vl_add(s, _wav(tmp_path, n))
    names = [v.get("name") for v in L.vl_list(s)]
    assert names[0] == "Степан" and len(names) == 6   # named + the last 5 recent
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


def test_unnamed_voices_are_told_apart_by_when_they_came(tmp_path):
    s = types.SimpleNamespace(voices=[])
    v = L.vl_add(s, _wav(tmp_path, "u"))
    v["ts"] = 1760000000
    import time
    assert L.vl_label(v, "ru") == "голос от " + time.strftime("%d.%m %H:%M", time.localtime(1760000000))


def test_clip_voice_offers_save_then_manage_rename_and_delete(tmp_path, monkeypatch):
    """Owner 10-03: a voice sent for a clip can be kept with a name, renamed,
    deleted, and picked next time without sending it again."""
    bot = _bot()
    bot._dl_bytes = lambda fid: b"RIFFfake"
    import cover
    monkeypatch.setattr(cover, "to_wav", lambda src, dst: src)
    sess = bot._get_session(11)
    sess.anim_voice_state = "collect"; bot._store.put(sess)
    assert bot._anim_voice_take_media(11, sess, "ru", {"voice": {"file_id": "F1"}})
    sess = bot._get_session(11)
    first = sess.anim_voices[0]
    rows = bot.sent[-1][1]["keyboard"]["inline_keyboard"]
    save = [b["callback_data"] for r in rows for b in r if b["callback_data"].startswith("vl:name:")]
    assert save and any(b["callback_data"] == "vl:manage" for r in rows for b in r)
    # a second voice does not overwrite the first file
    assert bot._anim_voice_take_media(11, bot._get_session(11), "ru", {"voice": {"file_id": "F2"}})
    assert len(set(bot._get_session(11).anim_voices)) == 2 and os.path.exists(first)
    # name it
    bot._cb_voice_library(11, save[0])
    assert bot._vl_take_name(11, bot._get_session(11), "ru", "Степан")
    vid = save[0].split(":")[2]
    assert L.vl_get(bot._get_session(11), vid)["name"] == "Степан"
    # rename it
    bot._cb_voice_library(11, f"vl:name:{vid}")
    assert bot._vl_take_name(11, bot._get_session(11), "ru", "Стёпа")
    assert L.vl_get(bot._get_session(11), vid)["name"] == "Стёпа"
    # the manage screen lists it with listen / rename / delete
    bot._cb_voice_library(11, "vl:manage")
    rows = bot.sent[-1][1]["keyboard"]["inline_keyboard"]
    assert [b["callback_data"] for b in rows[0]] == [f"vl:play:{vid}", f"vl:name:{vid}", f"vl:drop:{vid}"]
    assert rows[0][0]["text"] == "▶️ Стёпа"
    # delete it: off the list, and the file goes once nothing uses it
    s = bot._get_session(11); s.anim_voices = []; bot._store.put(s)
    ref = L.vl_get(bot._get_session(11), vid)["ref"]
    bot._cb_voice_library(11, f"vl:drop:{vid}")
    assert L.vl_get(bot._get_session(11), vid) is None and not os.path.exists(ref)

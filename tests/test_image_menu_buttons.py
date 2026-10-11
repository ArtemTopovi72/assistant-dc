"""🎭/🎬 preset menus name their picture in every button.

The menu used to send bare `style_preset:anime`, and the picture rode a one-shot
session slot that the first press spent: the second press in the same menu said
«Этой картинки больше нет». A picture must also never stick to later messages
(it once followed the chat ten messages on), so the slot is still spent per press.
"""
import json, os, sys, tempfile, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("F5_TEST_RUN", "1")
import tg_bot as T
T.redirect_data_dir(tempfile.mkdtemp(prefix="tgtest_imgmenu_"))
from PIL import Image

CID = 999_191


def _bot():
    bot = T.TelegramBot("123:TEST", lambda: object(), lambda: object(), lambda: {}, silent_mode=True)
    bot.sent, bot.kbs, bot.tasks = [], [], []
    bot._send_text = lambda cid, text, **kw: (bot.sent.append(text), bot.kbs.append(kw.get("keyboard")), 1)[2]
    bot._api_post = lambda *a, **k: {}
    bot._api_get = lambda *a, **k: {}
    bot._enqueue_task = lambda t: bot.tasks.append(t)
    bot._try_steer = lambda cid, t: False
    bot._backend = types.SimpleNamespace(push=lambda t: bot.tasks.append(t), depth=lambda: 1)
    bot._user_store.put(T._User(chat_id=CID, name="A", status="approved", is_admin=False))
    sess = bot._get_session(CID)
    sess.clear_context()
    sess.lang, sess.lang_chosen, sess.reg_state = "ru", True, ""
    bot._store.put(sess)
    return bot


def _pic(bot, name="p.png"):
    path = os.path.join(tempfile.mkdtemp(), name)
    Image.new("RGB", (32, 32), "red").save(path)
    sess = bot._get_session(CID)
    pid = T._log_image(sess, path, label="x", src="bot")
    bot._store.put(sess)
    return pid, path


def _press(bot, data, mid=50):
    bot._dispatch_callback({"id": "1", "data": data,
                            "message": {"message_id": mid, "chat": {"id": CID, "type": "private"}}})


def _buttons(kb):
    return [b["callback_data"] for row in (kb or {}).get("inline_keyboard", []) for b in row]


def _queued_paths(bot):
    return [getattr(t, "image_path", None) or (t.get("image_path") if isinstance(t, dict) else None)
            for t in bot.tasks]


def test_style_menu_survives_a_second_press():
    bot = _bot()
    pid, path = _pic(bot)
    _press(bot, f"style:{pid}")
    menu = [b for b in _buttons(bot.kbs[-1]) if b.startswith("style_")]
    assert menu and all(b.endswith(":" + pid) for b in menu), menu
    preset = next(b for b in menu if b.startswith("style_preset:"))
    _press(bot, preset)
    _press(bot, preset)
    assert _queued_paths(bot) == [path, path], (bot.sent, bot.tasks)
    sess = bot._get_session(CID)
    assert sess.pending_style_target == "" and sess.target_image in ("", pid)


def test_animate_menu_survives_a_second_press():
    bot = _bot()
    pid, path = _pic(bot)
    _press(bot, f"animate:{pid}")
    menu = [b for b in _buttons(bot.kbs[-1]) if b.startswith("animate_")]
    assert menu and all(b.endswith(":" + pid) for b in menu), menu
    preset = next(b for b in menu if b.startswith("animate_preset:"))
    _press(bot, preset)
    _press(bot, preset)
    assert _queued_paths(bot) == [path, path], (bot.sent, bot.tasks)
    assert bot._get_session(CID).pending_animate_target == ""


def test_each_menu_acts_on_its_own_picture():
    bot = _bot()
    a, pa = _pic(bot, "a.png")
    b, pb = _pic(bot, "b.png")
    _press(bot, f"style_preset:anime:{a}")
    _press(bot, f"style_preset:anime:{b}")
    _press(bot, f"style_preset:anime:{a}")
    assert _queued_paths(bot) == [pa, pb, pa]


def test_old_bare_keyboard_still_works_once():
    bot = _bot()
    pid, path = _pic(bot)
    sess = bot._get_session(CID)
    sess.pending_style_target = pid
    bot._store.put(sess)
    _press(bot, "style_preset:anime")
    assert _queued_paths(bot) == [path]


def test_a_deleted_picture_is_said_not_guessed():
    bot = _bot()
    pid, path = _pic(bot)
    os.remove(path)
    _press(bot, f"animate_preset:wave:{pid}")
    assert not bot.tasks and any("больше нет" in s for s in bot.sent), bot.sent

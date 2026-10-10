"""A button pressed in a group chat does nothing.

Messages from groups were already refused (tg_dispatch: the bot would map a
whole group to one account); buttons were not, so whoever sat in such a group
could press the owner's ▶ Render. Found by running the harness-review
authorization check against _dispatch_callback, 2026-10-10.
"""
import os
import sys
import tempfile

os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tg_bot as T  # noqa: E402

T.redirect_data_dir(tempfile.mkdtemp(prefix="cbpriv_"))


def _press(bot, cid, chat_type):
    bot._dispatch_callback({"id": "1", "data": "vp:go", "from": {"id": 77},
                            "message": {"chat": {"id": cid, "type": chat_type}, "message_id": 5}})


def test_group_button_is_ignored_private_one_works():
    bot = T.TelegramBot("123:TEST", lambda: object(), lambda: object(), lambda: {}, silent_mode=True)
    queued = []
    bot._send_text = lambda *a, **k: 1
    bot._api_post = lambda *a, **k: {}
    bot._enqueue_item = lambda cid, item: queued.append(cid)
    bot._activity.log = lambda *a, **k: None
    for cid in (-500, 500):
        bot._user_store.put(T._User(chat_id=cid, name="u", status="approved", is_admin=False))
        s = bot._get_session(cid)
        s.video_plan, s.video_plan_request = [{"text": "a", "sec": 5}], "сними видео"
        bot._store.put(s)
    _press(bot, -500, "supergroup")
    assert queued == []
    _press(bot, 500, "private")
    assert queued == [500]


if __name__ == "__main__":
    test_group_button_is_ignored_private_one_works()
    print("ok")

"""A turn that runs out of time hands over the clip it had joined.

Five of six parts of an approved talking head were made over an hour, the turn hit its
ceiling, and the user got «⏱ … отменён. Попробуй ещё раз.» -- the joined parts were on disk
(live 2026-10-20).
"""
import os
import sys
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "core", "agent", "media", "bot"):
    sys.path.insert(0, os.path.join(ROOT, d))
os.environ.setdefault("F5_TEST_RUN", "1")

import tg_tasks  # noqa: E402


def _bot():
    sent = []
    bot = types.SimpleNamespace(
        _chat_videos={}, _api_post=lambda *a, **k: None,
        _send_video=lambda chat, path, ctx=None: sent.append(("video", path)) or True,
        _send_text=lambda chat, text, **k: sent.append(("text", text)),
        _activity=types.SimpleNamespace(log=lambda *a, **k: None))
    return bot, sent


def test_the_joined_parts_reach_the_user_with_a_note(tmp_path):
    clip = tmp_path / "vidjoin.mp4"
    clip.write_bytes(b"x")
    bot, sent = _bot()
    ok = tg_tasks.TaskRunnerMixin._deliver_salvaged_video(
        bot, types.SimpleNamespace(video_made=str(clip)), 7, "ru", "u", since=time.time() - 60)
    assert ok and sent[0] == ("video", str(clip))
    assert sent[1][0] == "text" and "успело" in sent[1][1]


def test_a_clip_older_than_the_turn_is_never_sent(tmp_path):
    clip = tmp_path / "old.mp4"
    clip.write_bytes(b"x")
    os.utime(clip, (time.time() - 3600, time.time() - 3600))
    bot, sent = _bot()
    assert not tg_tasks.TaskRunnerMixin._deliver_salvaged_video(
        bot, types.SimpleNamespace(video_made=str(clip)), 7, "ru", "u", since=time.time() - 60)
    assert not tg_tasks.TaskRunnerMixin._deliver_salvaged_video(
        bot, types.SimpleNamespace(), 7, "ru", "u", since=0)
    assert not sent


def test_a_task_never_inherits_the_desktops_clip():
    import threading
    import tg_bot
    base = types.SimpleNamespace(video_made="desktop_render.mp4")
    assert tg_bot._scoped_ctx(base, cancel_event=threading.Event()).video_made == ""

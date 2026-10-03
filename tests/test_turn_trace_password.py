"""A typed password never reaches the turn trace (runtime/turns/*.jsonl).

Bug: _dispatch_logged opened the trace with the message text BEFORE _dispatch
ran; registration flags a password message `_secret` only inside _dispatch.
chatlog honoured the flag, but the trace kept the plaintext and was written
to disk whenever that turn logged a warning (a failed deleteMessage, a
backup error...).
"""
import json
import logging
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("F5_TEST_RUN", "1")


def test_password_turn_is_written_without_the_password(monkeypatch, tmp_path):
    import tg_bot as T
    import turn_trace
    T.redirect_data_dir(tempfile.mkdtemp())
    monkeypatch.setenv("TURNS_DIR", str(tmp_path))
    monkeypatch.setattr(turn_trace, "TURNS_DIR", tmp_path)
    bot = T.TelegramBot("1:T", lambda: object(), lambda: object(), lambda: {}, silent_mode=True)

    def dispatch(upd):
        msg = upd["message"]
        msg["_secret"] = True                     # what _scrub_secret does
        logging.getLogger("assistant.tg").warning("deleteMessage failed")
    bot._dispatch = dispatch
    bot._dispatch_logged({"message": {"chat": {"id": 6}, "text": "hunter2-secret",
                                      "message_id": 3}})
    written = "".join(p.read_text(encoding="utf-8") for p in tmp_path.glob("*.jsonl"))
    assert written, "the turn logged a warning, so it should have been written"
    assert "hunter2-secret" not in written
    assert json.loads(written.splitlines()[-1])["meta"]["text"] == "[secret]"

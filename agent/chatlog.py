"""Full per-chat transcript: everything in, everything out, and every log line
the bot writes while working on that chat (tool calls, steer notes, songs).

One JSONL file per chat under <data>/chat_logs/. Unlike the activity log
nothing is truncated: the point is to read a bad session back in full.
Passwords never land here -- a message _scrub_secret deletes is marked
secret and written as a placeholder.
"""
from __future__ import annotations
import contextlib
import contextvars
import json
import logging
import os
import threading
import time

# A context variable, not a thread-local: work the bot hands to another thread
# (turn_trace.spawn) keeps writing to the chat it was for. A thread-local left
# a forwarded video's retelling, a song, a render out of the transcript.
_cur: contextvars.ContextVar = contextvars.ContextVar("chatlog_chat", default=None)
_lock = threading.Lock()
_SKIP_METHODS = {"sendChatAction", "answerCallbackQuery", "getUpdates", "getMe", "getFile"}


def _dir() -> str:
    import tg_bot   # read at call time: redirect_data_dir() rebinds it in tests
    d = os.path.join(os.path.dirname(str(tg_bot._MASHUP_DIR)), "chat_logs")
    os.makedirs(d, exist_ok=True)
    return d


def write(chat_id, kind: str, data) -> None:
    if not chat_id:
        return
    entry = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, "data": data}
    try:
        line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"
        with _lock, open(os.path.join(_dir(), f"{chat_id}.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(line)
    except Exception:
        pass            # a transcript must never break the chat


@contextlib.contextmanager
def bind(chat_id):
    """Log lines written on this thread meanwhile go to this chat's file."""
    token = _cur.set(chat_id)
    try:
        yield
    finally:
        _cur.reset(token)


def outgoing(method: str, payload) -> None:
    if method in _SKIP_METHODS or not isinstance(payload, dict):
        return
    write(payload.get("chat_id") or _cur.get(), "out",
          dict(payload, method=method))


class _Handler(logging.Handler):
    def emit(self, record):
        chat = _cur.get()
        if chat and record.levelno >= logging.INFO:
            try:
                write(chat, "log", f"{record.name} {record.levelname}: {record.getMessage()}")
            except Exception:
                pass


_installed = False


def install() -> None:
    global _installed
    if not _installed:
        _installed = True
        h = _Handler()
        try:
            import log_redact           # requests errors quote the Bot API URL, token and all
            log_redact.install(h)
        except Exception:
            pass
        logging.getLogger().addHandler(h)

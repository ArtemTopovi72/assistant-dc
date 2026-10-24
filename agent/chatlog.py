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


_PROGRESS_END = ("…", "...")


def _line(entry) -> str:
    """One transcript entry as the line a reader of the chat would see, "" for log lines,
    progress stages and button chrome."""
    import re
    d = entry.get("data")
    if entry.get("kind") == "seen" and isinstance(d, dict) and d.get("text"):
        return "bot: [sees in it: " + " ".join(str(d["text"]).split())[:300] + "]"
    if not isinstance(d, dict) or entry.get("kind") not in ("in", "out"):
        return ""
    if entry["kind"] == "in":
        tags = [t for k, t in (("photo", "a picture"), ("video", "a video"), ("video_note", "a video note"),
                               ("voice", "a voice message"), ("audio", "an audio file"), ("document", "a file"))
                if d.get(k)]
        fwd = bool(d.get("forward_origin") or d.get("forward_date") or d.get("forward_from"))
        words = (d.get("text") or d.get("caption") or "").strip()
        head = ("forwards " if fwd else "sends ") + ", ".join(tags) if tags else ("forwards a message:" if fwd else "")
        said = " ".join(p for p in (f"[{head}]" if head else "", words) if p)
        return "user: " + said if said else ""
    method = d.get("method") or ""
    if method in ("sendPhoto", "sendMediaGroup"):
        return "bot: [sends a picture]"
    if method in ("sendVideo", "sendAnimation"):
        return "bot: [sends a video]"
    if method in ("sendVoice", "sendAudio"):
        return "bot: [sends audio]"
    import html
    text = html.unescape(re.sub(r"<[^>]+>", "", d.get("text") or "")).strip()
    # «⚙️ Пишу ответ… ⏳ 0:24» ticks took three of the eight lines a read gets
    if not text or text in ("✅", "👌") or (len(text) < 70 and (text.endswith(_PROGRESS_END) or "⏳" in text)):
        return ""
    return "bot: " + " ".join(text.split())[:300]


def seen(text: str) -> None:
    """What the bot saw in a forwarded or sent video, as one chat line for later reads: a bare
    «[forwards a video note]» left «Сколько стоит чай» under a receipt video «unclear»."""
    if (text or "").strip():
        write(_cur.get(), "seen", {"text": text.strip()})


def recent(chat_id, n: int = 8) -> list:
    """The last `n` lines of this chat as its people saw them -- what was forwarded, what the
    bot asked, which pictures went by -- for reads that need the talk, not one message."""
    try:
        with open(os.path.join(_dir(), f"{chat_id}.jsonl"), "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 400_000))
            raw = fh.read().decode("utf-8", "replace").splitlines()[1:]
    except OSError:
        return []
    entries = []
    for r in raw:
        try:
            entries.append(json.loads(r))
        except ValueError:
            pass
    return lines_of(entries, n)


def lines_of(entries: list, n: int = 8) -> list:
    out = [s for s in (_line(e) for e in entries) if s]
    return out[-n:]

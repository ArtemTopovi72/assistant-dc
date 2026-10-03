"""Keep the Telegram bot token out of the logs.

Every Bot API URL carries the token (https://api.telegram.org/bot<token>/...),
and a requests error message quotes the URL: "Max retries exceeded with url:
/bot123456:AAH.../sendVideo". Those errors are logged at WARNING into
assistant_app.log -- the file users paste into chats when something breaks --
so the token went with it.

RedactFilter is attached to the handlers, not the loggers: a logger-level
filter does not see records propagated from child loggers, a handler-level one
sees every record it would write.
"""
from __future__ import annotations

import logging
import re

# <bot id>:<35-ish char secret>; also the /file/bot<token>/ download URLs.
_TOKEN = re.compile(r"(?<![A-Za-z0-9])(bot)?(\d{6,12}):[A-Za-z0-9_-]{30,}")


def redact(text: str) -> str:
    if not text or ":" not in text:
        return text
    return _TOKEN.sub(lambda m: f"{m.group(1) or ''}{m.group(2)}:<redacted>", text)


class RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            clean = redact(msg)
            if clean != msg:
                record.msg, record.args = clean, None
            if record.exc_info and not record.exc_text:
                record.exc_text = logging.Formatter().formatException(record.exc_info)
            if record.exc_text:
                record.exc_text = redact(record.exc_text)
        except Exception:
            pass                # a log line is never lost to its own redaction
        return True


_FILTER = RedactFilter()


def install(*handlers: logging.Handler) -> None:
    """Attach the filter to these handlers, or to every handler of the root
    and "assistant" loggers when none are given. Idempotent."""
    if not handlers:
        handlers = tuple(logging.getLogger().handlers) + tuple(
            logging.getLogger("assistant").handlers)
    for h in handlers:
        if _FILTER not in h.filters:
            h.addFilter(_FILTER)

"""The bot token never reaches a log file (core/log_redact.py)."""
import io
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("F5_TEST_RUN", "1")

import log_redact  # noqa: E402

TOKEN = "123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw"


def _logger():
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(logging.Formatter("%(message)s"))
    log_redact.install(h)
    lg = logging.getLogger("assistant.test_redact")
    lg.handlers[:] = [h]
    lg.propagate = False
    lg.setLevel(logging.INFO)
    return lg, buf


def test_requests_error_text_is_redacted():
    lg, buf = _logger()
    err = ("HTTPSConnectionPool(host='api.telegram.org', port=443): Max retries exceeded "
           f"with url: /bot{TOKEN}/sendVideo (Caused by ...)")
    lg.warning("sendVideo attempt %d error: %s", 1, err)
    out = buf.getvalue()
    assert TOKEN.split(":")[1] not in out
    assert "/bot123456789:<redacted>/sendVideo" in out


def test_tracebacks_are_redacted():
    lg, buf = _logger()
    try:
        raise ConnectionError(f"https://api.telegram.org/file/bot{TOKEN}/photos/x.jpg")
    except ConnectionError:
        lg.exception("download failed")
    assert TOKEN.split(":")[1] not in buf.getvalue()
    assert "download failed" in buf.getvalue()


def test_ordinary_text_is_untouched():
    for s in ("12:30 meeting", "ratio 16:9", "id 42", "", "x" * 100):
        assert log_redact.redact(s) == s


def test_the_app_log_handlers_carry_it():
    src = open(os.path.join(ROOT, "core", "assistant.py"), encoding="utf-8").read()
    assert "log_redact.install()" in src
    assert "log_redact.install(self.log_handler)" in open(
        os.path.join(ROOT, "gui", "gui.py"), encoding="utf-8").read()

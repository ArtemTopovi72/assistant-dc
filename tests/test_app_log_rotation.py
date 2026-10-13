"""The app log keeps growing while another process holds it open.

Windows cannot rename a file another handle holds: RotatingFileHandler's rename-based
rollover failed on every record and assistant_app.log froze at 2 MB (2026-10-13)."""
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
os.environ.setdefault("F5_TEST_RUN", "1")
import assistant  # noqa: E402


def test_rollover_with_the_file_held_open_keeps_logging(tmp_path):
    path = tmp_path / "app.log"
    h = assistant._CopyTruncateHandler(str(path), maxBytes=200, backupCount=2, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(message)s"))
    other = open(path, "a", encoding="utf-8")          # another process's handle
    try:
        for i in range(20):
            h.emit(logging.LogRecord("t", logging.INFO, __file__, 1, f"record {i:02d} " + "x" * 40, None, None))
        h.flush()
        assert "record 19" in path.read_text(encoding="utf-8")
        assert (tmp_path / "app.log.1").exists()
        assert path.stat().st_size <= 400
    finally:
        other.close(); h.close()

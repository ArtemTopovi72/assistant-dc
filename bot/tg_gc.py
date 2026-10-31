"""Bound Telegram chat files without deleting media still used by a conversation."""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import nullcontext
from pathlib import Path

logger = logging.getLogger("assistant.tg_gc")
RETENTION_DAYS = 30
LOG_BYTES = 2 * 1024 * 1024


def _paths(value):
    if isinstance(value, str) and os.path.isabs(value):
        yield os.path.normcase(os.path.abspath(value))
    elif isinstance(value, dict):
        for item in value.values():
            yield from _paths(item)
    elif isinstance(value, list):
        for item in value:
            yield from _paths(item)


def collect(root: Path, *, now: float | None = None, artifact_store=None) -> dict[str, int]:
    """Remove old unreferenced Telegram media and bound per-chat transcripts."""
    root = Path(root).resolve()
    cutoff = (time.time() if now is None else now) - RETENTION_DAYS * 86400
    counts = {"artifacts": 0, "files": 0, "logs": 0}
    live = set()
    sessions = root / "tg_sessions.json"
    if sessions.exists():
        data = json.loads(sessions.read_text(encoding="utf-8"))
        live.update(_paths(data))

    db = root / "tg_artifacts.db"
    store = None
    if db.exists():
        import tg_artifacts_store
        store = artifact_store or tg_artifacts_store._store()
        if Path(store._path).resolve() != db.resolve():
            raise ValueError("artifact store does not belong to this data directory")

    with (store._lock if store is not None else nullcontext()):
        if store is not None:
            with store._conn() as conn:
                counts["artifacts"] = conn.execute(
                    "SELECT COUNT(*) FROM artifacts WHERE ts < ?", (cutoff,)).fetchone()[0]
                conn.execute("DELETE FROM artifacts WHERE ts < ?", (cutoff,))
                live.update(os.path.normcase(os.path.abspath(row[0])) for row in
                            conn.execute("SELECT path FROM artifacts"))

        for name in ("tg_images", "runtime/generated"):
            candidate = root / name
            directory = candidate.resolve()
            if candidate.is_symlink() or not directory.is_dir() or not directory.is_relative_to(root):
                continue
            for path in directory.rglob("*"):
                try:
                    if (path.is_file() and not path.is_symlink() and
                            path.stat().st_mtime < cutoff and
                            os.path.normcase(os.path.abspath(path)) not in live):
                        path.unlink()
                        counts["files"] += 1
                except OSError:
                    logger.warning("could not remove old Telegram file %s", path, exc_info=True)

    log_candidate = root / "chat_logs"
    logs = log_candidate.resolve()
    if logs.is_dir() and not log_candidate.is_symlink() and logs.is_relative_to(root):
        import chatlog
        with chatlog._lock:
            for path in logs.glob("*.jsonl"):
                try:
                    size = path.stat().st_size
                    if path.stat().st_mtime < cutoff:
                        path.unlink()
                        counts["logs"] += 1
                    elif size > LOG_BYTES:
                        with path.open("rb") as source:
                            source.seek(size - LOG_BYTES)
                            source.readline()
                            tail = source.read()
                        temp = path.with_suffix(".jsonl.tmp")
                        temp.write_bytes(tail)
                        temp.replace(path)
                        counts["logs"] += 1
                except OSError:
                    logger.warning("could not trim Telegram transcript %s", path, exc_info=True)
    logger.info("Telegram GC: %s", counts)
    return counts

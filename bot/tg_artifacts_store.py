"""Durable, restart-safe registry of every file this bot has delivered.

Companion to tg_userstore.py's users/usage tables, same conventions (SQLite,
WAL, one lock around a short-lived connection per call) but its own DB file
and its own module, since write volume here (one row per photo/video/song/
document actually SENT) is much higher than the user table's and should not
multiply tg_userstore's per-write backup-and-prune cost.

WHY THIS EXISTS (live, 2026-09-19): `sess.image_log` already survives a
restart (it rides inside tg_sessions.json), but nothing else does — a video
just delivered is remembered only in an in-memory `dict` on the bot object
(`self._chat_videos`), and a song or presentation is not remembered at all
past the turn that made it. Restarting the app, or the file underneath a
still-remembered path being deleted by hand, both surface today as either a
silent failure downstream or (for photos, the one kind that already checks)
the honest "🖼 Этой картинки больше нет" refusal. This module generalises
that pattern to every artifact kind, in one durable place, so:
  - a photo/video/song/document reference can survive a restart;
  - a row whose file went missing (deleted by hand, or wiped by an app
    restart's own cleanup) is pruned automatically rather than dangling; and
  - "the picture/video/song is gone, please resend" becomes something any
    caller can produce uniformly instead of re-deriving its own os.path.exists
    check per artifact kind, as img_gone alone used to.

WHY THIS DOES NOT REPEAT THE DCIM.ZIP INCIDENT (journey 33, 2026-09-14): the
only writers of this table are the four real Telegram delivery points —
`_log_image`, and `_send_video`/`_send_document`/`_send_audio` in
tg_transport.py — each firing once per file actually handed to one real chat.
Nothing here ever walks a directory or a sandbox extraction tree, and every
read is capped at `_MAX_ROWS` regardless of how it is called, so there is no
path from "a zip full of 173 photos landed somewhere on disk" to this store
ever holding, or handing back, anything but what the bot genuinely sent.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

_ARTIFACTS_DB = Path(__file__).resolve().parents[1].joinpath("tg_artifacts.db")

# No query this module answers, however it is called, ever hands back more
# than this many rows. That is the belt-and-suspenders half of the guard
# described above: even a caller that got its filtering wrong cannot turn
# this store into a way to enumerate "everything on disk".
_MAX_ROWS = 50

_KINDS = ("photo", "video", "music", "document")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
    id           TEXT PRIMARY KEY,
    chat_id      INTEGER NOT NULL,
    kind         TEXT NOT NULL,
    path         TEXT NOT NULL,
    label        TEXT DEFAULT '',
    src          TEXT DEFAULT 'bot',
    ts           REAL NOT NULL,
    content_hash TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_artifacts_chat_kind_ts ON artifacts(chat_id, kind, ts);
CREATE INDEX IF NOT EXISTS idx_artifacts_content ON artifacts(chat_id, content_hash);
"""


def _content_key(path: str) -> str:
    """Cheap content identity: first 1MB's hash plus the exact file size.

    Not a full-file hash (large videos would make every delivery pay for a
    multi-second read) but enough to catch the "working copy" duplicate
    pattern documented in tg_bot._image_by_content -- a byte-identical copy of
    an already-registered file under a different name, which is what made a
    single styled photo look like two different reference images to
    generate_video (live, 2026-09-19).
    """
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            head = fh.read(1 << 20)
        return f"{size}:{hashlib.sha1(head).hexdigest()}"
    except OSError:
        return ""


class _ArtifactStore:
    def __init__(self, path):
        self._path = str(path)
        self._lock = threading.Lock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._conn() as conn:
            conn.executescript(_SCHEMA)

    def add(self, chat_id: int, kind: str, path: str, *,
            label: str = "", src: str = "bot") -> str:
        """Register one delivered file. Returns its id (empty if `path` is falsy).

        Re-registering the same path for the same chat refreshes it in place
        instead of piling up duplicate rows, mirroring _log_image's own
        dedupe-by-path behaviour.
        """
        if not path:
            return ""
        chash = _content_key(path)
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT id FROM artifacts WHERE chat_id=? AND path=?",
                (int(chat_id), str(path))).fetchone()
            if row:
                conn.execute(
                    "UPDATE artifacts SET label=?, ts=?, content_hash=? WHERE id=?",
                    (label[:120], time.time(), chash, row["id"]))
                return row["id"]
            aid = uuid.uuid4().hex[:12]
            conn.execute(
                "INSERT INTO artifacts (id, chat_id, kind, path, label, src, ts, "
                "content_hash) VALUES (?,?,?,?,?,?,?,?)",
                (aid, int(chat_id), kind, str(path), label[:120], src, time.time(),
                 chash))
            return aid

    def recent(self, chat_id: int, kind: Optional[str] = None,
               limit: int = 10) -> list[dict]:
        """Newest-first artifacts still actually on disk; stale rows are pruned
        as they're found rather than returned."""
        limit = max(1, min(int(limit), _MAX_ROWS))
        q = "SELECT * FROM artifacts WHERE chat_id=?"
        args: list = [int(chat_id)]
        if kind:
            q += " AND kind=?"
            args.append(kind)
        q += " ORDER BY ts DESC LIMIT ?"
        args.append(min(limit * 3, _MAX_ROWS))   # over-fetch a bit for pruned rows
        with self._lock, self._conn() as conn:
            rows = [dict(r) for r in conn.execute(q, args).fetchall()]
        out, stale = [], []
        for r in rows:
            if os.path.exists(r["path"]):
                out.append(r)
                if len(out) >= limit:
                    break
            else:
                stale.append(r["id"])
        if stale:
            self._delete_ids(stale)
        return out

    def get(self, artifact_id: str) -> Optional[dict]:
        if not artifact_id:
            return None
        with self._lock, self._conn() as conn:
            row = conn.execute("SELECT * FROM artifacts WHERE id=?",
                               (artifact_id,)).fetchone()
        if not row:
            return None
        row = dict(row)
        if not os.path.exists(row["path"]):
            self._delete_ids([row["id"]])
            return None
        return row

    def by_content(self, chat_id: int, path: str) -> Optional[dict]:
        """An already-registered row with the SAME bytes as `path`, if any."""
        chash = _content_key(path)
        if not chash:
            return None
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM artifacts WHERE chat_id=? AND content_hash=? "
                "ORDER BY ts DESC LIMIT 1", (int(chat_id), chash)).fetchone()
        if not row:
            return None
        row = dict(row)
        if not os.path.exists(row["path"]):
            self._delete_ids([row["id"]])
            return None
        return row

    def _delete_ids(self, ids: list[str]) -> None:
        if not ids:
            return
        with self._lock, self._conn() as conn:
            conn.executemany("DELETE FROM artifacts WHERE id=?",
                             [(i,) for i in ids])

    def sweep_missing(self, chat_id: Optional[int] = None) -> int:
        """Garbage-collect rows whose file is gone from disk (deleted by hand,
        or an app restart's own scratch cleanup). Returns how many were pruned.

        Called once at bot startup so a restart also catches whatever went
        missing while the process was down; every read above additionally
        prunes lazily, so this is a top-up, not the only line of defence
        (no polling loop needed to keep the table honest between restarts).
        """
        with self._lock, self._conn() as conn:
            if chat_id is None:
                rows = conn.execute("SELECT id, path FROM artifacts").fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, path FROM artifacts WHERE chat_id=?",
                    (int(chat_id),)).fetchall()
        dead = [r["id"] for r in rows if not os.path.exists(r["path"])]
        self._delete_ids(dead)
        return len(dead)


_instance_lock = threading.Lock()
_instance: Optional[_ArtifactStore] = None


def _store() -> _ArtifactStore:
    """The process-wide store, rebuilt if `_ARTIFACTS_DB` has been redirected
    (tests call redirect_data_dir() before constructing a bot; see tg_bot.py)."""
    global _instance
    with _instance_lock:
        if _instance is None or _instance._path != str(_ARTIFACTS_DB):
            _instance = _ArtifactStore(_ARTIFACTS_DB)
    return _instance


def log_artifact(chat_id: int, kind: str, path: str, *,
                 label: str = "", src: str = "bot") -> str:
    assert kind in _KINDS, f"unknown artifact kind {kind!r}"
    return _store().add(chat_id, kind, path, label=label, src=src)


def recent_artifacts(chat_id: int, kind: Optional[str] = None,
                     limit: int = 10) -> list[dict]:
    return _store().recent(chat_id, kind, limit)


def get_artifact(artifact_id: str) -> Optional[dict]:
    return _store().get(artifact_id)


def artifact_by_content(chat_id: int, path: str) -> Optional[dict]:
    return _store().by_content(chat_id, path)


def sweep_missing(chat_id: Optional[int] = None) -> int:
    return _store().sweep_missing(chat_id)

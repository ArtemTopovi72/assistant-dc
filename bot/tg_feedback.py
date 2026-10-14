"""👍/👎 under every text answer, and the numbers behind the Feedback tab.

One row per answer sent (`replies`), one row per person's vote on it (`votes`, a
second press of the same button takes the vote back). SQLite beside the other bot
stores, same conventions as tg_artifacts_store: WAL, one lock, a short-lived
connection per call, the connect function handed in.

The Feedback tab (gui_feedback_tab) reads the aggregates below; nothing here knows Qt.
"""
from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

_FEEDBACK_DB = Path(__file__).resolve().parents[1].joinpath("tg_feedback.db")

_SNIP = 600          # chars of the question and the answer kept for the "what went wrong" list

_SCHEMA = """
CREATE TABLE IF NOT EXISTS replies (
    id         TEXT PRIMARY KEY,
    ts         REAL NOT NULL,
    chat_id    INTEGER NOT NULL,
    kind       TEXT DEFAULT 'chat',
    model      TEXT DEFAULT '',
    secs       REAL DEFAULT 0,
    user_text  TEXT DEFAULT '',
    reply_text TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_replies_ts ON replies(ts);
CREATE TABLE IF NOT EXISTS votes (
    reply_id TEXT NOT NULL,
    voter    INTEGER NOT NULL,
    vote     INTEGER NOT NULL,
    ts       REAL NOT NULL,
    PRIMARY KEY (reply_id, voter)
);
CREATE INDEX IF NOT EXISTS idx_votes_ts ON votes(ts);
"""


class _FeedbackStore:
    def __init__(self, path, connect=None):
        self._path = str(path)
        self._connect = connect or sqlite3.connect
        self._lock = threading.Lock()
        with self._lock, self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        conn = self._connect(self._path, check_same_thread=False, timeout=10)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            with conn:
                yield conn
        finally:
            conn.close()

    def record_reply(self, chat_id: int, user_text: str, reply_text: str, *,
                     kind: str = "chat", model: str = "", secs: float = 0.0) -> str:
        rid = uuid.uuid4().hex[:10]
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO replies (id, ts, chat_id, kind, model, secs, user_text, reply_text) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (rid, time.time(), int(chat_id), kind or "chat", model or "", float(secs or 0),
                 (user_text or "")[:_SNIP], (reply_text or "")[:_SNIP]))
        return rid

    def vote(self, reply_id: str, voter: int, value: int) -> dict:
        """Set `voter`'s vote on a reply (+1 / -1); the same vote again clears it.
        -> {"up", "down", "mine"} after the change, or {} for a reply that is not known."""
        value = 1 if value > 0 else -1
        with self._lock, self._conn() as conn:
            if not conn.execute("SELECT 1 FROM replies WHERE id=?", (reply_id,)).fetchone():
                return {}
            row = conn.execute("SELECT vote FROM votes WHERE reply_id=? AND voter=?",
                               (reply_id, int(voter))).fetchone()
            if row and row["vote"] == value:
                conn.execute("DELETE FROM votes WHERE reply_id=? AND voter=?", (reply_id, int(voter)))
                mine = 0
            else:
                conn.execute("INSERT OR REPLACE INTO votes (reply_id, voter, vote, ts) VALUES (?,?,?,?)",
                             (reply_id, int(voter), value, time.time()))
                mine = value
            return {**self._counts(conn, reply_id), "mine": mine}

    @staticmethod
    def _counts(conn, reply_id: str) -> dict:
        r = conn.execute("SELECT COALESCE(SUM(vote>0),0) up, COALESCE(SUM(vote<0),0) down "
                         "FROM votes WHERE reply_id=?", (reply_id,)).fetchone()
        return {"up": int(r["up"]), "down": int(r["down"])}

    def counts(self, reply_id: str, voter: int = 0) -> dict:
        with self._lock, self._conn() as conn:
            out = self._counts(conn, reply_id)
            row = conn.execute("SELECT vote FROM votes WHERE reply_id=? AND voter=?",
                               (reply_id, int(voter))).fetchone()
            return {**out, "mine": row["vote"] if row else 0}

    # ---- what the Feedback tab shows --------------------------------------------------
    def summary(self, days: int = 30) -> dict:
        since = time.time() - days * 86400
        with self._lock, self._conn() as conn:
            replies = conn.execute("SELECT COUNT(*) n FROM replies WHERE ts>=?", (since,)).fetchone()["n"]
            v = conn.execute("SELECT COALESCE(SUM(vote>0),0) up, COALESCE(SUM(vote<0),0) down "
                             "FROM votes v JOIN replies r ON r.id=v.reply_id WHERE r.ts>=?",
                             (since,)).fetchone()
        up, down = int(v["up"]), int(v["down"])
        return {"replies": int(replies), "up": up, "down": down,
                "rate": (up / (up + down)) if up + down else None}

    def daily(self, days: int = 30) -> list:
        """[{"day": "YYYY-MM-DD", "replies", "up", "down"}] for each of the last `days` days,
        zero-filled, oldest first. A vote counts on the day its answer was sent."""
        since = time.time() - days * 86400
        with self._lock, self._conn() as conn:
            rep = {r["d"]: r["n"] for r in conn.execute(
                "SELECT date(ts,'unixepoch','localtime') d, COUNT(*) n FROM replies WHERE ts>=? GROUP BY d",
                (since,))}
            vot = {r["d"]: (r["up"], r["down"]) for r in conn.execute(
                "SELECT date(r.ts,'unixepoch','localtime') d, SUM(v.vote>0) up, SUM(v.vote<0) down "
                "FROM votes v JOIN replies r ON r.id=v.reply_id WHERE r.ts>=? GROUP BY d", (since,))}
        out = []
        for i in range(days - 1, -1, -1):
            d = time.strftime("%Y-%m-%d", time.localtime(time.time() - i * 86400))
            up, down = vot.get(d, (0, 0))
            out.append({"day": d, "replies": int(rep.get(d, 0)), "up": int(up or 0), "down": int(down or 0)})
        return out

    def by_kind(self, days: int = 30) -> list:
        """[{"kind", "up", "down", "replies"}], the most-answered kinds first."""
        since = time.time() - days * 86400
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT r.kind kind, COUNT(DISTINCT r.id) replies, COALESCE(SUM(v.vote>0),0) up, "
                "COALESCE(SUM(v.vote<0),0) down FROM replies r LEFT JOIN votes v ON v.reply_id=r.id "
                "WHERE r.ts>=? GROUP BY r.kind ORDER BY replies DESC", (since,)).fetchall()
        return [{"kind": r["kind"], "replies": int(r["replies"]), "up": int(r["up"]), "down": int(r["down"])}
                for r in rows]

    def recent_down(self, limit: int = 30) -> list:
        """Answers somebody disliked, newest vote first: what to look at next."""
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT r.id, r.kind, r.user_text, r.reply_text, MAX(v.ts) vts FROM votes v "
                "JOIN replies r ON r.id=v.reply_id WHERE v.vote<0 GROUP BY r.id "
                "ORDER BY vts DESC LIMIT ?", (int(limit),)).fetchall()
        return [{"id": r["id"], "kind": r["kind"], "user": r["user_text"], "reply": r["reply_text"],
                 "ts": r["vts"]} for r in rows]


def keyboard(reply_id: str, counts: dict | None = None) -> dict:
    """The 👍/👎 row. The chosen side is ticked; the totals show once anybody voted."""
    c = counts or {}
    mine = c.get("mine", 0)
    up, down = c.get("up", 0), c.get("down", 0)
    return {"inline_keyboard": [[
        {"text": ("👍" + (f" {up}" if up else "")) + ("✓" if mine > 0 else ""),
         "callback_data": f"fb:{reply_id}:u"},
        {"text": ("👎" + (f" {down}" if down else "")) + ("✓" if mine < 0 else ""),
         "callback_data": f"fb:{reply_id}:d"},
    ]]}


def with_feedback(kb: dict | None, reply_id: str, counts: dict | None = None) -> dict:
    """`kb` (an inline keyboard or nothing) with the feedback row under it."""
    rows = list((kb or {}).get("inline_keyboard") or [])
    return {**(kb or {}), "inline_keyboard": rows + keyboard(reply_id, counts)["inline_keyboard"]}


# ---- the process-wide store (what the bot and the tab share) -----------------------------
_instance: _FeedbackStore | None = None
_instance_lock = threading.Lock()


def store() -> _FeedbackStore:
    global _instance
    with _instance_lock:
        if _instance is None or _instance._path != str(_FEEDBACK_DB):
            _instance = _FeedbackStore(_FEEDBACK_DB)
        return _instance


def redirect_store(path) -> None:
    """Point the process-wide store at another file (a suite's temp dir)."""
    global _FEEDBACK_DB, _instance
    with _instance_lock:
        _FEEDBACK_DB, _instance = Path(path), None

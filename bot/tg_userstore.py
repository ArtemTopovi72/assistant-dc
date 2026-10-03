"""Registered users: the record, the SQLite store, and password hashing.

Lifted out of tg_bot.py. Nothing here knows about Telegram updates, sessions
or the queue — it answers who exists, who is approved, and what they have
spent today.

ONE THING MATTERS MORE THAN THE LINE COUNT. The JSON migration source and the
rolling backup directory are defined HERE and nowhere else, and tg_bot's
redirect_data_dir() rebinds them ON THIS MODULE. Two copies of a path that a
test reassigns is how a suite quietly writes its fixtures into the live
tg_users.db — which has happened in this repo before.
"""
import hashlib
import hmac
import json
import os
import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("assistant.tg_bot")

# Legacy JSON store, read once to migrate, and the rolling backup directory.
# tg_bot.redirect_data_dir() reassigns both of these attributes on this module;
# never import them by value anywhere else.
_USERS_FILE_JSON    = Path(__file__).resolve().parents[1].joinpath("tg_users.json")
_USERS_BACKUP_DIR   = Path(__file__).resolve().parents[1].joinpath("tg_users_backups")
_USERS_BACKUP_KEEP  = 10   # keep last N backup snapshots

# ── password hashing ──────────────────────────────────────────────────────────
# Stored form: "v2$<salt_hex>$<scrypt_hex>".
#
# The original scheme was a single sha256(password + chat_id): one fast pass, so
# anyone who obtained tg_users.db could brute-force a 4-character password at
# billions of guesses per second. chat_id acted as a salt (no rainbow tables) but
# did nothing about speed. scrypt is memory-hard, which is the property that makes
# offline cracking expensive. Legacy hashes still verify and are transparently
# re-hashed on the next successful login, so nobody has to reset anything.
_LEGACY_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _scrypt_params() -> tuple:
    try:
        import config as _cfg_mod
        n_log2 = int(getattr(_cfg_mod, "TG_SCRYPT_N_LOG2", 14))
    except Exception:
        n_log2 = 14
    n = 2 ** max(12, min(20, n_log2))
    return n, 8, 1


def _hash_password(password: str, chat_id: int) -> str:
    n, r, p = _scrypt_params()
    salt = os.urandom(16)
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p,
                        dklen=32, maxmem=256 * 1024 * 1024)
    return f"v2${salt.hex()}${dk.hex()}"


def _verify_password(password: str, chat_id: int, stored: str) -> tuple:
    """Return (ok, upgraded_hash). upgraded_hash is non-empty when the stored hash
    used the old scheme and should be replaced with the returned scrypt hash."""
    stored = (stored or "").strip()
    if not stored:
        return False, ""
    if stored.startswith("v2$"):
        try:
            _, salt_hex, want_hex = stored.split("$", 2)
            n, r, p = _scrypt_params()
            dk = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt_hex),
                                n=n, r=r, p=p, dklen=32, maxmem=256 * 1024 * 1024)
            return hmac.compare_digest(dk.hex(), want_hex), ""
        except Exception as exc:
            logger.warning("password verify error: %s", exc)
            return False, ""
    if _LEGACY_HASH_RE.match(stored):
        legacy = hashlib.sha256((password + str(chat_id)).encode()).hexdigest()
        if hmac.compare_digest(legacy, stored):
            return True, _hash_password(password, chat_id)
        return False, ""
    return False, ""


def _utc_day() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())

# ═══════════════════════════════════════════════════════════════════════════════
# USER STORE
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class _User:
    chat_id:       int
    name:          str  = ""
    tg_username:   str  = ""
    password_hash: str  = ""
    status:        str  = "pending"   # "pending" | "approved" | "banned"
    registered_at: float = field(default_factory=time.time)
    subscriptions: list  = field(default_factory=list)   # e.g. ["startup"]
    is_admin:      bool  = False
    # Free-form per-user customization (badge emoji, signature sticker/photo
    # file_id, anything added later) -- a single generic column so a new
    # personalization option never needs its own schema migration. Nothing
    # in this dict is interpreted here; callers own their own keys.
    prefs:         dict  = field(default_factory=dict)

    def __post_init__(self):
        # `subscriptions` is stored as JSON, so a legacy or hand-edited row can
        # come back as null (json.loads("null") -> None) — and every reader does
        # `"startup" in user.subscriptions`. Coerce once here rather than guarding
        # at each of the eight call sites.
        if not isinstance(self.subscriptions, list):
            self.subscriptions = [] if self.subscriptions is None \
                                 else list(self.subscriptions)
        if not isinstance(self.prefs, dict):
            self.prefs = {} if self.prefs is None else dict(self.prefs)

    def to_dict(self) -> dict:
        return {
            "chat_id":       self.chat_id,
            "name":          self.name,
            "tg_username":   self.tg_username,
            "password_hash": self.password_hash,
            "status":        self.status,
            "registered_at": self.registered_at,
            "subscriptions": self.subscriptions,
            "is_admin":      self.is_admin,
            "prefs":         self.prefs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "_User":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class _UserStore:
    """SQLite-backed user store.

    Durability guarantees
    ─────────────────────
    • WAL journal mode — readers never block writers; crash recovery replays
      the WAL log automatically on next open.
    • PRAGMA synchronous=NORMAL — each transaction is flushed to the OS before
      returning; survives OS crashes (not power loss at the storage layer, but
      SQLite's WAL + NORMAL is the recommended production trade-off for embedded
      databases — Full sync adds ~10-100 ms per write for no practical gain here).
    • PRAGMA wal_checkpoint(TRUNCATE) — WAL file is trimmed after each write so
      it never grows unboundedly.
    • Rolling backup — after every `put()` a point-in-time copy is written to
      tg_users_backups/tg_users_TIMESTAMP.db and old backups are pruned to the
      last N snapshots.  This is the practical equivalent of "replication" for a
      single-machine deployment: if the primary DB file is ever corrupted you can
      restore from the most recent backup.
    • Migration — if tg_users.json exists from a previous version it is
      automatically imported and the JSON file is renamed to .json.migrated.
    """

    _CREATE = """
        CREATE TABLE IF NOT EXISTS users (
            chat_id       INTEGER PRIMARY KEY,
            name          TEXT    NOT NULL DEFAULT '',
            tg_username   TEXT    NOT NULL DEFAULT '',
            password_hash TEXT    NOT NULL DEFAULT '',
            status        TEXT    NOT NULL DEFAULT 'pending',
            registered_at REAL    NOT NULL DEFAULT 0,
            subscriptions TEXT    NOT NULL DEFAULT '[]',
            is_admin      INTEGER NOT NULL DEFAULT 0,
            prefs         TEXT    NOT NULL DEFAULT '{}'
        )
    """
    # Daily usage counters. Keyed by UTC day so the reset point is unambiguous
    # regardless of where the users are.
    _CREATE_USAGE = """
        CREATE TABLE IF NOT EXISTS usage (
            chat_id INTEGER NOT NULL,
            day     TEXT    NOT NULL,
            kind    TEXT    NOT NULL,
            count   INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (chat_id, day, kind)
        )
    """
    _UPSERT = """
        INSERT INTO users
            (chat_id, name, tg_username, password_hash, status,
             registered_at, subscriptions, is_admin, prefs)
        VALUES (?,?,?,?,?,?,?,?,?)
        ON CONFLICT(chat_id) DO UPDATE SET
            name          = excluded.name,
            tg_username   = excluded.tg_username,
            password_hash = excluded.password_hash,
            status        = excluded.status,
            registered_at = excluded.registered_at,
            subscriptions = excluded.subscriptions,
            is_admin      = excluded.is_admin,
            prefs         = excluded.prefs
    """

    def __init__(self, db_path: Path,
                 backup_dir: Path | None = None,
                 legacy_json: Path | None = None):
        # Resolved at call time, not bound as a default at def time, so that
        # redirect_data_dir() can point a test run away from the live files.
        self._db   = db_path
        self._bdir = backup_dir if backup_dir is not None else _USERS_BACKUP_DIR
        legacy_json = legacy_json if legacy_json is not None else _USERS_FILE_JSON
        self._lock = threading.Lock()
        self._init_db()
        self._migrate_json(legacy_json)

    # ── connection factory ────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db), check_same_thread=False,
                               timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ── init + migration ──────────────────────────────────────────────────────

    def _init_db(self):
        try:
            with self._conn() as conn:
                conn.execute(self._CREATE)
                conn.execute(self._CREATE_USAGE)
                # A DB created before `prefs` existed has no such column, and
                # CREATE TABLE IF NOT EXISTS above is a no-op against an
                # already-existing table -- so every existing deployment needs
                # this ALTER once. Cheap to check every startup; sqlite has no
                # "ADD COLUMN IF NOT EXISTS", hence the table_info probe.
                cols = {r[1] for r in conn.execute("PRAGMA table_info(users)")}
                if "prefs" not in cols:
                    conn.execute("ALTER TABLE users ADD COLUMN prefs TEXT NOT NULL DEFAULT '{}'")
        except Exception as exc:
            logger.error("UserStore init failed: %s", exc)
            raise

    # ── usage / quota accounting ──────────────────────────────────────────────

    def usage_today(self, chat_id: int) -> dict:
        """{kind: count} for the current UTC day."""
        day = _utc_day()
        with self._lock:
            try:
                with self._conn() as conn:
                    rows = conn.execute(
                        "SELECT kind, count FROM usage WHERE chat_id=? AND day=?",
                        (chat_id, day)).fetchall()
                    return {r["kind"]: int(r["count"]) for r in rows}
            except Exception as exc:
                logger.error("usage_today(%s): %s", chat_id, exc)
                return {}

    def bump_usage(self, chat_id: int, kind: str, n: int = 1) -> None:
        day = _utc_day()
        with self._lock:
            try:
                with self._conn() as conn:
                    conn.execute(
                        "INSERT INTO usage (chat_id, day, kind, count) VALUES (?,?,?,?) "
                        "ON CONFLICT(chat_id, day, kind) DO UPDATE SET "
                        "count = count + excluded.count",
                        (chat_id, day, kind, n))
                    # Keep the table from growing forever; a fortnight is plenty
                    # of history for "why did I hit my limit yesterday?".
                    conn.execute("DELETE FROM usage WHERE day < ?",
                                 (time.strftime("%Y-%m-%d",
                                                time.gmtime(time.time() - 14 * 86400)),))
            except Exception as exc:
                logger.error("bump_usage(%s,%s): %s", chat_id, kind, exc)

    def usage_totals(self, days: int = 7) -> list:
        """[(chat_id, kind, count)] over the last N UTC days — admin stats."""
        since = time.strftime("%Y-%m-%d", time.gmtime(time.time() - days * 86400))
        with self._lock:
            try:
                with self._conn() as conn:
                    rows = conn.execute(
                        "SELECT chat_id, kind, SUM(count) c FROM usage "
                        "WHERE day >= ? GROUP BY chat_id, kind ORDER BY c DESC",
                        (since,)).fetchall()
                    return [(int(r["chat_id"]), r["kind"], int(r["c"])) for r in rows]
            except Exception as exc:
                logger.error("usage_totals: %s", exc)
                return []

    def _migrate_json(self, json_path: Path):
        if not json_path.exists():
            return
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            count = 0
            for d in data.values():
                try:
                    self.put(_User.from_dict(d))
                    count += 1
                except Exception as exc:
                    logger.warning("migration: skip record %s: %s",
                                   d.get("chat_id"), exc)
            logger.info("UserStore: migrated %d users from JSON", count)
            json_path.rename(json_path.with_suffix(".json.migrated"))
        except Exception as exc:
            logger.error("UserStore JSON migration error: %s", exc)

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def get(self, chat_id: int) -> Optional[_User]:
        with self._lock:
            try:
                with self._conn() as conn:
                    row = conn.execute(
                        "SELECT * FROM users WHERE chat_id=?", (chat_id,)
                    ).fetchone()
                    return self._row(row) if row else None
            except Exception as exc:
                logger.error("UserStore.get(%s): %s", chat_id, exc)
                return None

    def put(self, user: _User):
        with self._lock:
            try:
                with self._conn() as conn:
                    conn.execute(self._UPSERT, (
                        user.chat_id, user.name, user.tg_username,
                        user.password_hash, user.status, user.registered_at,
                        json.dumps(user.subscriptions), int(user.is_admin),
                        json.dumps(user.prefs),
                    ))
                # checkpoint runs AFTER commit (WAL does it automatically on conn.close)
                self._make_backup()
            except Exception as exc:
                logger.error("UserStore.put(%s): %s", user.chat_id, exc)

    def delete(self, chat_id: int) -> bool:
        """Remove a user row (profile replacement). Returns True if a row was
        deleted. A backup is taken first — this is the only destructive user op."""
        with self._lock:
            try:
                # Its own file: a same-second put() used to overwrite the
                # per-second snapshot with the post-delete state, and routine
                # puts rotated it out within minutes.
                self._make_backup(name=f"deleted_{chat_id}_{time.strftime('%Y%m%d_%H%M%S')}.db")
                with self._conn() as conn:
                    cur = conn.execute("DELETE FROM users WHERE chat_id=?", (chat_id,))
                    removed = cur.rowcount > 0
                logger.info("UserStore.delete(%s): removed=%s", chat_id, removed)
                return removed
            except Exception as exc:
                logger.error("UserStore.delete(%s): %s", chat_id, exc)
                return False

    def all(self) -> list[_User]:
        with self._lock:
            try:
                with self._conn() as conn:
                    rows = conn.execute(
                        "SELECT * FROM users ORDER BY registered_at DESC"
                    ).fetchall()
                    return [self._row(r) for r in rows]
            except Exception as exc:
                logger.error("UserStore.all: %s", exc)
                return []

    def pending(self) -> list[_User]:
        with self._lock:
            try:
                with self._conn() as conn:
                    rows = conn.execute(
                        "SELECT * FROM users WHERE status='pending'"
                    ).fetchall()
                    return [self._row(r) for r in rows]
            except Exception as exc:
                logger.error("UserStore.pending: %s", exc)
                return []

    def approved(self) -> list[_User]:
        with self._lock:
            try:
                with self._conn() as conn:
                    rows = conn.execute(
                        "SELECT * FROM users WHERE status='approved'"
                    ).fetchall()
                    return [self._row(r) for r in rows]
            except Exception as exc:
                logger.error("UserStore.approved: %s", exc)
                return []

    # ── backup ────────────────────────────────────────────────────────────────

    def _make_backup(self, name: str = ""):
        """Copy the live SQLite DB to a timestamped snapshot file (or `name`,
        which the rotation below never prunes)."""
        try:
            self._bdir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            dst   = self._bdir / (name or f"tg_users_{stamp}.db")
            # sqlite3.connect backup API is the correct way to copy a live WAL db
            src_conn = sqlite3.connect(str(self._db))
            dst_conn = sqlite3.connect(str(dst))
            src_conn.backup(dst_conn, pages=64)
            dst_conn.close()
            src_conn.close()
            self._prune_backups()
        except Exception as exc:
            logger.warning("UserStore backup failed: %s", exc)

    def _prune_backups(self):
        """Keep only the last _USERS_BACKUP_KEEP snapshot files."""
        try:
            snaps = sorted(self._bdir.glob("tg_users_*.db"))
            for old in snaps[:-_USERS_BACKUP_KEEP]:
                old.unlink(missing_ok=True)
        except Exception: pass

    def restore_from_backup(self, backup_path: Path | str) -> bool:
        """Replace the live DB with a backup snapshot.  Returns True on success."""
        backup_path = Path(backup_path)
        if not backup_path.exists():
            logger.error("restore: backup not found: %s", backup_path)
            return False
        with self._lock:
            try:
                src = sqlite3.connect(str(backup_path))
                dst = sqlite3.connect(str(self._db))
                src.backup(dst, pages=64)
                dst.close(); src.close()
                logger.info("UserStore restored from %s", backup_path)
                return True
            except Exception as exc:
                logger.error("restore failed: %s", exc)
                return False

    def list_backups(self) -> list[Path]:
        try:
            return sorted(self._bdir.glob("tg_users_*.db"), reverse=True)
        except Exception:
            return []

    # ── helper ────────────────────────────────────────────────────────────────

    @staticmethod
    def _row(row: sqlite3.Row) -> _User:
        try:
            subs = json.loads(row["subscriptions"])
        except Exception:
            subs = []
        try:
            prefs = json.loads(row["prefs"]) if "prefs" in row.keys() else {}
        except Exception:
            prefs = {}
        return _User(
            chat_id=row["chat_id"],
            name=row["name"],
            tg_username=row["tg_username"],
            password_hash=row["password_hash"],
            status=row["status"],
            registered_at=row["registered_at"],
            subscriptions=subs,
            is_admin=bool(row["is_admin"]),
            prefs=prefs if isinstance(prefs, dict) else {},
        )

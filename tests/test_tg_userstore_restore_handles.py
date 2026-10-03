"""UserStore backup/restore close their SQLite handles even when the copy fails.

Bug: restore_from_backup() of a file that is not a database raised inside
src.backup() and skipped both close() calls; on Windows the open handle kept
the live tg_users.db (and the bad file) locked until garbage collection.
"""
import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("F5_TEST_RUN", "1")

import tg_userstore as U  # noqa: E402


def _tracking(monkeypatch):
    opened = []
    real = sqlite3.connect

    def connect(*a, **k):
        c = real(*a, **k)
        opened.append(c)
        return c
    monkeypatch.setattr(U.sqlite3, "connect", connect)
    return opened


def _closed(c) -> bool:
    try:
        c.execute("SELECT 1")
        return False
    except sqlite3.ProgrammingError:          # "Cannot operate on a closed database"
        return True


def test_failed_restore_closes_both_handles(tmp_path, monkeypatch):
    s = U._UserStore(tmp_path / "u.db", backup_dir=tmp_path / "bk", legacy_json=tmp_path / "none.json")
    bad = tmp_path / "tg_users_bad.db"
    bad.write_bytes(b"this is not a sqlite database" * 100)
    opened = _tracking(monkeypatch)
    assert s.restore_from_backup(bad) is False
    assert opened and all(_closed(c) for c in opened)


def test_backup_and_restore_round_trip(tmp_path, monkeypatch):
    s = U._UserStore(tmp_path / "u.db", backup_dir=tmp_path / "bk", legacy_json=tmp_path / "none.json")
    s.put(U._User(chat_id=5, name="Kept"))
    s._make_backup(name="tg_users_manual.db")
    snap = tmp_path / "bk" / "tg_users_manual.db"
    s.delete(5)
    opened = _tracking(monkeypatch)
    assert s.restore_from_backup(snap) is True
    assert all(_closed(c) for c in opened)
    assert s.get(5) is not None and s.get(5).name == "Kept"

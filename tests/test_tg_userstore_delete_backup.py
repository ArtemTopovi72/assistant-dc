"""A deleted profile survives in its own backup: a same-second put() used to
overwrite the per-second snapshot with the post-delete state.

Run: venv/Scripts/python.exe tests/test_tg_userstore_delete_backup.py
"""
import os, sqlite3, sys, tempfile
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tg_userstore as U

d = Path(tempfile.mkdtemp())
s = U._UserStore(d / "u.db", backup_dir=d / "bk", legacy_json=d / "none.json")
s.put(U._User(chat_id=7, name="Old"))
s.delete(7)
s.put(U._User(chat_id=7, name="New"))          # same second, as in a profile replacement
for _ in range(30):
    s.put(U._User(chat_id=8, name="x"))        # routine rotation
snap = [p for p in (d / "bk").iterdir() if p.name.startswith("deleted_7_")]
names = [r[0] for p in snap for r in sqlite3.connect(p).execute("SELECT name FROM users WHERE chat_id=7")]
ok = snap and names == ["Old"]
print(("PASS" if ok else "FAIL") + "  the pre-delete profile is kept", snap, names)
sys.exit(0 if ok else 1)

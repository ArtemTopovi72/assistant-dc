"""redirect_data_dir() must move EVERY persistent path, including the ones the
user store owns.

Why this suite exists: the store's JSON migration source and its rolling backup
directory live in tg_userstore, not tg_bot, so redirect_data_dir has to set
them on that module. If it ever goes back to assigning local copies, the store
keeps writing beside the LIVE tg_users.db while every other suite still passes
— which is exactly how test chat ids once ended up registered and approved in
production. Nothing else asserts this, so it is asserted here.

Run: venv/Scripts/python.exe tests/test_tg_data_redirect.py
"""
import hashlib
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import tg_bot
import tg_userstore

RESULTS = []
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))


_REPO = Path(tg_bot.__file__).resolve().parents[1]
_live = _REPO / "tg_users.db"
_before = hashlib.sha1(_live.read_bytes()).hexdigest() if _live.exists() else "absent"

TMP = tempfile.mkdtemp(prefix="tgredir_")
tg_bot.redirect_data_dir(TMP)

# Every path a bot writes to, wherever it is defined.
for label, value in (
    ("users db",     tg_bot._USERS_DB),
    ("sessions",     tg_bot._SESSION_FILE),
    ("activity log", tg_bot._ACTIVITY_LOG_FILE),
    ("offset",       tg_bot._OFFSET_FILE),
    ("feedback",     tg_bot._FEEDBACK_FILE),
    ("inflight",     tg_bot._INFLIGHT_FILE),
    ("libraries",    tg_bot._LIBRARY_DIR),
    ("images",       tg_bot._IMAGE_DIR),
    ("memory",       tg_bot._MEMORY_DIR),
    # These two are OWNED by tg_userstore. Reading them off tg_bot would prove
    # nothing — the store resolves them on its own module at call time.
    ("users json (tg_userstore)",    tg_userstore._USERS_FILE_JSON),
    ("users backups (tg_userstore)", tg_userstore._USERS_BACKUP_DIR),
):
    check(f"redirected: {label}", str(value).startswith(TMP), f"{value}")

# Drive the real store hard enough to touch the DB and the backup directory.
store = tg_userstore._UserStore(tg_bot._USERS_DB)
user = tg_userstore._User(chat_id=999778, name="redirect probe",
                          password_hash=tg_userstore._hash_password("pw", 999778))
store.put(user)
store.bump_usage(999778, "image")
store._make_backup()

check("round-trip through the redirected store", store.get(999778) is not None)
check("password verifies", tg_userstore._verify_password("pw", 999778,
                                                         store.get(999778).password_hash)[0])
check("wrong password rejected", not tg_userstore._verify_password(
    "nope", 999778, store.get(999778).password_hash)[0])
check("backup landed under the temp dir",
      all(str(p).startswith(TMP) for p in store.list_backups()))

# The session store's READ path. A _Store.get that ignores the file entirely and
# returns a blank _Session survived every suite tried (tg_hardening,
# library_fixes, tg_localization) — they all put and read back through one live
# _Store instance, which a broken get can satisfy from memory. Reading through a
# SECOND _Store over the same file is what makes the load real.
import tg_sessions

_sfile = tg_bot._SESSION_FILE
_store_a = tg_sessions._Store(_sfile)
_s = _store_a.get(4242)
_s.set_tg_facts(["persisted fact"])
_s.lang = "en"
_store_a.put(_s)

_store_b = tg_sessions._Store(_sfile)          # fresh instance, same file
_reloaded = _store_b.get(4242)
check("session state survives a new _Store over the same file",
      _reloaded.get_tg_facts() == ["persisted fact"], f"{_reloaded.get_tg_facts()}")
check("scalar session field survives the round trip", _reloaded.lang == "en",
      f"{_reloaded.lang}")
check("session file was written under the temp dir", str(_sfile).startswith(TMP))
_store_b.delete(4242)
check("delete removes it from a fresh reader",
      tg_sessions._Store(_sfile).get(4242).get_tg_facts() == [])

_after = hashlib.sha1(_live.read_bytes()).hexdigest() if _live.exists() else "absent"
check("live tg_users.db untouched", _before == _after)
check("no stray tg_users.json in the repo", not (_REPO / "tg_users.json").exists())

# ── the QUEUE is shared state too ─────────────────────────────────────────────
# The operator's REDIS_URL lives in the environment, so a suite that builds a
# bot without this pushes its fixtures into the live ring: the running bot
# would execute them, and leftovers made unrelated suites fail at random.
import tg_queue_backends as _qb
check("redirect_data_dir isolates the queue backend too", _qb._FORCE_INMEMORY)
_bot_q = tg_bot.TelegramBot("123:TEST", lambda: None, lambda: None,
                            lambda: {}, silent_mode=True)
check("and the bot it builds really is on the in-memory queue",
      isinstance(_bot_q._backend, _qb.InMemoryBackend),
      type(_bot_q._backend).__name__)

_bad = [r[0] for r in RESULTS if not r[1]]
print(f"\n{len(RESULTS) - len(_bad)}/{len(RESULTS)} checks passed")
if _bad:
    print("FAILED CHECKS: " + ", ".join(_bad))
sys.exit(1 if _bad else 0)

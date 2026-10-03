"""Remove test-fixture users that leaked into the LIVE store.

The tg_bot test suites construct the real TelegramBot, whose data paths were
module-level constants — so before `tg_bot.redirect_data_dir()` existed, every
run registered its fixture chat ids in the production files. Chat ids 999101
("Boss", admin) and 999102 ("Member") were found approved in tg_users.db.

This removes ONLY those synthetic ids. Real users are untouched. A timestamped
backup of both files is taken first.

  python cleanup_test_fixtures.py --dry-run     # show what would change
  python cleanup_test_fixtures.py               # apply
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

# Fixture ids used by tests/test_tg_hardening.py, test_tg_live_integration.py
# and test_tg_button_scenarios.py. Deliberately an explicit allow-list rather
# than a range, so a real account can never be caught by it.
FIXTURE_IDS = {999101, 999102, 999201}

USERS_DB = BASE / "tg_users.db"
SESSIONS = BASE / "tg_sessions.json"
ACTIVITY = BASE / "tg_activity.jsonl"
BACKUPS = BASE / "tg_users_backups"
LIBRARIES = BASE / "tg_libraries"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")

    if not args.dry_run:
        BACKUPS.mkdir(parents=True, exist_ok=True)
        for src in (USERS_DB, SESSIONS, ACTIVITY):
            if src.exists():
                dst = BACKUPS / f"{src.stem}_precleanup_{stamp}{src.suffix}"
                shutil.copy2(src, dst)
                print(f"backup  {dst.name}")

    # ── users.db ────────────────────────────────────────────────────────────
    if USERS_DB.exists():
        conn = sqlite3.connect(str(USERS_DB))
        rows = conn.execute(
            "SELECT chat_id, name, is_admin FROM users WHERE chat_id IN "
            f"({','.join('?' * len(FIXTURE_IDS))})", tuple(FIXTURE_IDS)).fetchall()
        for cid, name, admin in rows:
            print(f"users   remove {cid} {name!r}{' [ADMIN]' if admin else ''}")
        if rows and not args.dry_run:
            q = f"DELETE FROM {{}} WHERE chat_id IN ({','.join('?' * len(FIXTURE_IDS))})"
            conn.execute(q.format("users"), tuple(FIXTURE_IDS))
            for table in ("usage",):        # optional tables
                try:
                    conn.execute(q.format(table), tuple(FIXTURE_IDS))
                except sqlite3.OperationalError:
                    pass
            conn.commit()
        remaining = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        conn.close()
        print(f"users   {len(rows)} fixture row(s); {remaining} row(s) remain")

    # ── sessions ────────────────────────────────────────────────────────────
    if SESSIONS.exists():
        try:
            data = json.loads(SESSIONS.read_text(encoding="utf-8") or "[]")
        except Exception as exc:
            print(f"sessions unreadable ({exc}) — skipped")
            data = None
        if isinstance(data, list):
            keep = [s for s in data
                    if int(s.get("chat_id", 0) or 0) not in FIXTURE_IDS]
        elif isinstance(data, dict):
            keep = {k: v for k, v in data.items()
                    if str(k).lstrip("-").isdigit() and int(k) not in FIXTURE_IDS}
        else:
            keep = None
        if keep is not None:
            print(f"session {len(data)} -> {len(keep)}")
            if not args.dry_run and len(keep) != len(data):
                SESSIONS.write_text(json.dumps(keep, ensure_ascii=False, indent=2),
                                    encoding="utf-8")

    # ── activity log ────────────────────────────────────────────────────────
    if ACTIVITY.exists():
        kept, dropped = [], 0
        for line in ACTIVITY.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                if int(json.loads(line).get("chat_id", 0) or 0) in FIXTURE_IDS:
                    dropped += 1
                    continue
            except Exception:
                pass
            kept.append(line)
        print(f"activity drop {dropped} line(s)")
        if dropped and not args.dry_run:
            ACTIVITY.write_text("\n".join(kept) + ("\n" if kept else ""),
                                encoding="utf-8")

    # ── per-user library indexes ────────────────────────────────────────────
    for cid in sorted(FIXTURE_IDS):
        db = LIBRARIES / f"lib_{cid}.db"
        if db.exists():
            print(f"library remove {db.name}")
            if not args.dry_run:
                db.unlink()

    print("DRY RUN — nothing written" if args.dry_run else "done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

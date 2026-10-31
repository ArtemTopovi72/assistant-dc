"""Old chat files leave the disk, but current references remain usable."""
import json
import os
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "agent"), str(ROOT / "bot")]
import tg_gc
import tg_artifacts_store


def test_collect_keeps_live_media_and_discards_old_context_files(tmp_path):
    now = 40 * 86400
    old = now - 35 * 86400
    media = tmp_path / "tg_images"
    media.mkdir()
    active = media / "active.jpg"
    orphan = media / "orphan.jpg"
    recent = media / "recent.jpg"
    for path in (active, orphan, recent):
        path.write_bytes(b"image")
        os.utime(path, (old, old))
    (tmp_path / "tg_sessions.json").write_text(
        json.dumps({"42": {"image_log": [{"path": str(active)}]}}), encoding="utf-8")
    store = tg_artifacts_store._ArtifactStore(tmp_path / "tg_artifacts.db")
    with store._conn() as conn:
        conn.executemany(
            "INSERT INTO artifacts (id,chat_id,kind,path,ts) VALUES (?,?,?,?,?)",
            [("old", 42, "photo", str(orphan), old),
             ("new", 42, "photo", str(recent), now)],
        )
    logs = tmp_path / "chat_logs"
    logs.mkdir()
    log = logs / "42.jsonl"
    log.write_bytes((b'{"line":"old"}\n' * 170000) + b'{"line":"last"}\n')

    result = tg_gc.collect(tmp_path, now=now, artifact_store=store)

    assert active.exists() and recent.exists()
    assert not orphan.exists()
    assert result["artifacts"] == result["files"] == result["logs"] == 1
    assert log.stat().st_size <= tg_gc.LOG_BYTES
    assert log.read_bytes().endswith(b'{"line":"last"}\n')
    with store._conn() as conn:
        assert conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 1

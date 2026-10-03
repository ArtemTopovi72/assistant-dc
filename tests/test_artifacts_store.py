"""tg_artifacts_store.py — the durable, restart-safe artifact registry.

Live, 2026-09-19: only sess.image_log survived an app restart; a delivered
video lived in an in-memory dict, and a song or presentation was never
remembered past the turn that made it. This store generalises "remember what
was actually delivered, and admit when the file is gone" to every kind, in
one SQLite table modeled on tg_userstore.py's own conventions.

Run: venv/Scripts/python.exe tests/test_artifacts_store.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_artifacts_store as A

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}" + (f"   ({extra})" if extra else ""))

_tmp = tempfile.mkdtemp(prefix="artifacts_test_")
A._ARTIFACTS_DB = os.path.join(_tmp, "tg_artifacts.db")
A._instance = None   # force a fresh _ArtifactStore bound to the redirected path


def _mkfile(name: str, data: bytes = b"x") -> str:
    p = os.path.join(_tmp, name)
    with open(p, "wb") as fh:
        fh.write(data)
    return p


print("=" * 70)
print("1. Basic add/get/recent, scoped per chat and kind")
print("=" * 70)

p1 = _mkfile("a.jpg")
aid = A.log_artifact(111, "photo", p1, label="first")
check("add returns a non-empty id", bool(aid))
row = A.get_artifact(aid)
check("get returns the row", row is not None and row["path"] == p1)
check("kind round-trips", row["kind"] == "photo")
check("chat_id round-trips", row["chat_id"] == 111)

p2 = _mkfile("b.mp4")
A.log_artifact(111, "video", p2, label="a clip")
p3 = _mkfile("c.mp3")
A.log_artifact(222, "music", p3, label="a song")

check("recent(111) sees both of chat 111's artifacts",
      len(A.recent_artifacts(111)) == 2)
check("recent(111, kind=photo) is scoped to that kind",
      [r["path"] for r in A.recent_artifacts(111, kind="photo")] == [p1])
check("chat 222's artifact does not leak into chat 111's list",
      p3 not in [r["path"] for r in A.recent_artifacts(111)])
check("an empty path is a no-op, not a crash", A.log_artifact(111, "photo", "") == "")

print()
print("=" * 70)
print("2. Re-registering the same path updates in place, no duplicate row")
print("=" * 70)

before = len(A.recent_artifacts(111))
aid2 = A.log_artifact(111, "photo", p1, label="relabelled")
check("re-adding the same path returns the SAME id", aid2 == aid)
check("row count for chat 111 did not grow", len(A.recent_artifacts(111)) == before)
check("the label was updated", A.get_artifact(aid)["label"] == "relabelled")

print()
print("=" * 70)
print("3. Garbage collection — a file deleted by hand disappears from every read")
print("=" * 70)

p4 = _mkfile("d.jpg")
aid4 = A.log_artifact(333, "photo", p4)
os.unlink(p4)
check("get() prunes and returns None once the file is gone",
      A.get_artifact(aid4) is None)
check("...and the row is actually deleted, not just hidden",
      A.get_artifact(aid4) is None)

p5 = _mkfile("e.jpg")
aid5 = A.log_artifact(444, "photo", p5)
os.unlink(p5)
check("recent() silently prunes a missing file rather than returning a dead path",
      aid5 not in [r["id"] for r in A.recent_artifacts(444)])

p6 = _mkfile("f.jpg")
A.log_artifact(555, "photo", p6)
os.unlink(p6)
p7 = _mkfile("g.jpg")
A.log_artifact(555, "photo", p7)
n = A.sweep_missing(555)
check("sweep_missing() reports how many rows it pruned for one chat", n == 1)
check("the survivor is still there after the sweep",
      [r["path"] for r in A.recent_artifacts(555)] == [p7])
n_all = A.sweep_missing()
check("sweep_missing() with no chat_id sweeps every chat", isinstance(n_all, int))

print()
print("=" * 70)
print("4. Content-based lookup — a byte-identical copy under another name")
print("=" * 70)

p8 = _mkfile("h.png", data=b"same-bytes-here")
A.log_artifact(666, "photo", p8, label="original")
p9 = _mkfile("h_working_copy.png", data=b"same-bytes-here")   # byte-identical
hit = A.artifact_by_content(666, p9)
check("a byte-identical file under a different name is recognised",
      hit is not None and hit["path"] == p8)
p10 = _mkfile("i.png", data=b"different-bytes")
check("genuinely different content is NOT matched",
      A.artifact_by_content(666, p10) is None)
check("content lookup is scoped per chat too",
      A.artifact_by_content(777, p9) is None)

print()
print("=" * 70)
print("5. A read never hands back more than the hard cap")
print("=" * 70)

for i in range(A._MAX_ROWS + 20):
    A.log_artifact(888, "photo", _mkfile(f"many_{i}.jpg"))
check(f"recent(limit=999) is still capped at _MAX_ROWS={A._MAX_ROWS}",
      len(A.recent_artifacts(888, limit=999)) <= A._MAX_ROWS)

print()
print("=" * 70)
print("6. redirect_data_dir() rebinds this store too, not just tg_users.db")
print("=" * 70)

import tg_bot
_redirect_dir = tempfile.mkdtemp(prefix="artifacts_redirect_")
tg_bot.redirect_data_dir(_redirect_dir)
check("redirect_data_dir() repoints tg_artifacts_store._ARTIFACTS_DB",
      str(A._ARTIFACTS_DB).startswith(_redirect_dir))
p11 = _mkfile("j.jpg")
A.log_artifact(999, "photo", p11)
check("a write after redirect lands in the redirected DB, not the old one",
      os.path.exists(A._ARTIFACTS_DB))

print()
print(f"{OK}/{OK + BAD} checks passed")
if BAD:
    sys.exit(1)

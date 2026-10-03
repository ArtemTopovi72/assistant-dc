"""A recycled pid must not keep the card locked forever.

gpu_lock treated "pid_exists" as "the holder is alive". The lock is held for a
whole training day; a trainer that is killed leaves the file behind, and
Windows recycles pids freely, so any unrelated process landing on that number
kept the card blocked with no way out but deleting the file by hand.

That was survivable while only bench scripts read the lock. It is not any more:
the app and the bot now REFUSE to draw while the lock is held, so a phantom
holder locks the user out of their own renderer.

The lock therefore records the holder's process START TIME, and a pid whose
process started after the lock was written is a different process wearing the
same number.

Nothing here starts a process or touches the GPU: psutil is stubbed, and the
lock file is redirected to a temp path.

Run: venv/Scripts/python.exe tests/test_gpu_lock_pid_reuse.py
"""
import os
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name
          + ("" if cond or not detail else "  -- " + str(detail)[:300]))
    if not cond and os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))


import gpu_lock

gpu_lock.LOCK_PATH = Path(tempfile.mkdtemp(prefix="gpulock_")) / "gpu.lock"


class _FakePsutil:
    """Stand-in for psutil with a scripted process table."""
    def __init__(self, table):
        self.table = table            # {pid: create_time}

    def pid_exists(self, pid):
        return pid in self.table

    def Process(self, pid):
        if pid not in self.table:
            raise RuntimeError("no such process")
        return types.SimpleNamespace(create_time=lambda: self.table[pid])


class _Psutil:
    def __init__(self, table):
        self.fake = _FakePsutil(table)
        self.orig = None

    def __enter__(self):
        self.orig = sys.modules.get("psutil")
        sys.modules["psutil"] = self.fake
        return self

    def __exit__(self, *a):
        if self.orig is None:
            sys.modules.pop("psutil", None)
        else:
            sys.modules["psutil"] = self.orig


OTHER = 424242          # never this process


def _write(text):
    gpu_lock.LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    gpu_lock.LOCK_PATH.write_text(text, encoding="utf-8")


def test_a_live_holder_is_respected():
    _write("%d @1000 Ideogram 4 training" % OTHER)
    with _Psutil({OTHER: 1000.0}):
        held = gpu_lock.owner()
    check("the card reads as held", held is not None, held)
    check("with the right pid and label",
          held == (OTHER, "Ideogram 4 training"), held)


def test_a_recycled_pid_is_seen_through():
    """Same pid, different process: it started long after the lock was written."""
    _write("%d @1000 Ideogram 4 training" % OTHER)
    with _Psutil({OTHER: 9_000_000.0}):
        held = gpu_lock.owner()
    check("the lock is treated as stale", held is None, held)


def test_a_dead_pid_is_still_stale():
    _write("%d @1000 Ideogram 4 training" % OTHER)
    with _Psutil({}):
        check("nothing alive means nothing held", gpu_lock.owner() is None)


def test_a_lock_from_an_older_build_still_reads():
    """No @token: fall back to pid_exists alone rather than refusing to parse.
    A lock file written by the previous version is still a real holder."""
    _write("%d Ideogram 4 training" % OTHER)
    with _Psutil({OTHER: 1000.0}):
        held = gpu_lock.owner()
    check("it parses", held is not None, held)
    check("and the label is not polluted by the missing token",
          held and held[1] == "Ideogram 4 training", held)


def test_a_label_containing_an_at_sign_is_not_mistaken_for_the_token():
    _write("%d @1000 run @ 2x" % OTHER)
    with _Psutil({OTHER: 1000.0}):
        held = gpu_lock.owner()
    check("only the FIRST token is the timestamp", held == (OTHER, "run @ 2x"), held)


def test_acquire_records_the_start_time():
    gpu_lock.LOCK_PATH.unlink(missing_ok=True)
    with _Psutil({os.getpid(): 4242.0}):
        got = gpu_lock.acquire("a test", wait=0)
    check("the lock was taken", got is True)
    raw = gpu_lock.LOCK_PATH.read_text(encoding="utf-8")
    check("the file carries pid, start time and label",
          raw.startswith("%d @4242 a test" % os.getpid()), raw)
    gpu_lock.release()
    check("and release removes it", not gpu_lock.LOCK_PATH.exists())


def test_our_own_lock_never_blocks_us():
    _write("%d @1 whatever" % os.getpid())
    with _Psutil({os.getpid(): 1.0}):
        check("a job does not wait for itself", gpu_lock.owner() is None)
    gpu_lock.LOCK_PATH.unlink(missing_ok=True)


def _main():
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        try:
            fn()
        except Exception as exc:
            check(fn.__name__ + " raised", False, exc)
    bad = [n for n, ok in RESULTS if not ok]
    print("\n%d/%d passed" % (len(RESULTS) - len(bad), len(RESULTS)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(_main())

"""One writer at a time on the GPU.

This exists because it already went wrong. A second trainer was launched on top
of a run that was still going: both processes fought for the card, the step time
went 5.93 -> 8.98 s, and both wrote the same log and the same checkpoint names.
Every job here is a whole-card job -- training, a batch of renders, a captioning
pass -- so "wait your turn" is the only correct policy, and it has to be
enforced by something both jobs can see, not by whoever remembers to check.

A lock is a file holding the owner's PID and a label. It is released on exit,
and a lock whose PID is gone is treated as stale and taken over, so a crashed or
killed job cannot block the card forever.

    with gpu_lock.hold("ideogram A/B", wait=3600):
        ...
"""
import contextlib
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

LOCK_PATH = Path(__file__).resolve().parents[1] / "runtime" / "gpu.lock"


def _create_time(pid: int):
    """When that pid's process started, or None if we cannot tell."""
    try:
        import psutil
        return float(psutil.Process(pid).create_time())
    except Exception:
        return None


def _alive(pid: int, born: float = 0.0) -> bool:
    """Is the process that took the lock still the process wearing that pid?

    pid_exists alone is not enough. The lock is held for a whole training day;
    a trainer that is killed leaves the file behind, and Windows recycles pids
    freely, so any unrelated process landing on that number would keep the card
    blocked with no way out but deleting the file by hand -- and the app now
    REFUSES to draw while the lock is held, so a phantom holder locks the user
    out of their own renderer.

    `born` is the holder's recorded process start time. A pid whose process
    started AFTER the lock was written is a different process wearing the same
    number, however alive it is.
    """
    try:
        import psutil
        if not psutil.pid_exists(pid):
            return False
        if born:
            now = _create_time(pid)
            # A second of slack: the recorded value and psutil's own reading of
            # the same process can differ in the last decimals.
            if now is not None and abs(now - born) > 1.0:
                logger.info("gpu.lock: pid %d was recycled (started %.0f, lock "
                            "recorded %.0f) -- treating the lock as stale",
                            pid, now, born)
                return False
        return True
    except Exception:
        return True          # cannot tell: assume held, never steal on a guess


def owner():
    """(pid, label) of the live holder, or None if the card is free."""
    try:
        raw = LOCK_PATH.read_text(encoding="utf-8").strip()
    except Exception:
        return None
    pid_s, _, rest = raw.partition(" ")
    try:
        pid = int(pid_s)
    except ValueError:
        return None
    # "<pid> @<create_time> <label>". The @token is optional so a lock written
    # by an older build still reads, it just falls back to pid_exists alone.
    born = 0.0
    if rest.startswith("@"):
        token, _, rest = rest.partition(" ")
        try:
            born = float(token[1:])
        except ValueError:
            born = 0.0
    label = rest
    if pid == os.getpid() or not _alive(pid, born):
        return None
    return pid, (label or "?")


def acquire(label: str, wait: float = 0.0, poll: float = 20.0) -> bool:
    """Take the lock, waiting up to `wait` seconds. False if someone kept it."""
    deadline = time.time() + max(0.0, wait)
    warned = None
    while True:
        held = owner()
        if held is None:
            LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
            LOCK_PATH.write_text(
                "%d @%.0f %s" % (os.getpid(), _create_time(os.getpid()) or 0, label),
                encoding="utf-8")
            # Re-read: two starters a moment apart can both see a free lock and
            # both write. Last writer wins, and the other must find that out.
            time.sleep(0.4)
            again = owner()
            if again is None:
                return True
            held = again
        if held[:1] != warned:
            logger.info("GPU held by pid %d (%s) -- waiting", held[0], held[1])
            print("GPU held by pid %d (%s) -- waiting" % held)
            warned = held[:1]
        if time.time() >= deadline:
            return False
        time.sleep(poll)


def release() -> None:
    try:
        raw = LOCK_PATH.read_text(encoding="utf-8").strip()
    except Exception:
        return
    if raw.split(" ")[0] == str(os.getpid()):
        with contextlib.suppress(Exception):
            LOCK_PATH.unlink()


@contextlib.contextmanager
def hold(label: str, wait: float = 0.0, poll: float = 20.0):
    """Raise RuntimeError rather than run a whole-card job beside another one."""
    if not acquire(label, wait=wait, poll=poll):
        held = owner()
        raise RuntimeError(
            "the GPU is busy%s -- refusing to start %r beside it"
            % (" with %r (pid %d)" % (held[1], held[0]) if held else "", label))
    try:
        yield
    finally:
        release()

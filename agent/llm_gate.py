"""Our own queue in front of LM Studio.

Before this, every process (the desktop app, the Telegram bot's workers, the
benches) posted straight to LM Studio, and LM Studio queued whatever did not
fit its parallel slots ("PROCESSING PROMPT (+3 QUEUED)"). Its queue is FIFO
and blind: a bench could sit ahead of a live user, nobody could see their
place, and a cancelled turn still waited its turn to run.

Here every model call first takes one of N slots, N = the loaded model's
PARALLEL (lms ps) or LLM_SLOTS. Slots are OS file locks in runtime/llm_gate,
so the gate works ACROSS processes and a crashed holder frees its slot the
moment it dies. Waiters register a ticket named by priority and arrival:

    0  a person waiting for an answer (default in the app and the bot)
    1  background work (set ctx.llm_priority = 1)
    2  benches and test runs (default when the main script lives in bench/)

A waiter may take a free slot only when fewer than N live tickets are ahead of
it. While waiting it reports its place through ctx.set_stage ("Waiting for
the model (2 ahead)") and gives up at once when the turn is cancelled.

LLM_GATE=0 turns the gate off.
"""
from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

GATE_DIR = Path(os.getenv("LLM_GATE_DIR",
                          Path(__file__).resolve().parents[1] / "runtime" / "llm_gate"))
POLL = 0.05
_SLOTS_TTL = 60.0
_slots_cache = [0, 0.0]
_seq = [0]
_seq_lock = threading.Lock()


class Cancelled(Exception):
    """The turn was cancelled while waiting for a slot."""


def enabled() -> bool:
    return os.getenv("LLM_GATE", "1").strip() not in ("0", "false", "no")


def slots() -> int:
    env = os.getenv("LLM_SLOTS", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    now = time.perf_counter()
    if _slots_cache[0] and now - _slots_cache[1] < _SLOTS_TTL:
        return _slots_cache[0]
    n = 1
    try:
        import lmstudio
        n = max(1, int(lmstudio.loaded_parallel()))
    except Exception:
        pass
    _slots_cache[0], _slots_cache[1] = n, now
    return n


def _process_priority() -> int:
    env = os.getenv("LLM_PRIORITY", "").strip()
    if env.isdigit():
        return int(env)
    main = str(getattr(sys.modules.get("__main__"), "__file__", "") or "").replace("\\", "/")
    return 2 if ("/bench/" in main or "/tests/" in main) else 0


def priority(ctx) -> int:
    p = getattr(ctx, "llm_priority", None)
    return int(p) if isinstance(p, int) else _process_priority()


# -- OS locks ------------------------------------------------------------------
if os.name == "nt":
    import msvcrt

    def _try_lock(fh) -> bool:
        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def _unlock(fh) -> None:
        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
else:
    import fcntl

    def _try_lock(fh) -> bool:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def _unlock(fh) -> None:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass


def _alive(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:
        return True


def _tickets_ahead(mine: Path) -> int:
    """Live tickets that sort before `mine`; dead owners' tickets are removed."""
    ahead = 0
    for t in GATE_DIR.glob("t_*"):
        if t.name >= mine.name:
            continue
        try:
            pid = int(t.name.split("_")[3])
        except (IndexError, ValueError):
            continue
        if pid != os.getpid() and not _alive(pid):
            with contextlib.suppress(OSError):
                t.unlink()
            continue
        ahead += 1
    return ahead


@contextlib.contextmanager
def slot(ctx=None):
    """Hold one model slot for the duration of the block."""
    if not enabled():
        yield
        return
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    n = slots()
    prio = min(9, max(0, priority(ctx)))
    with _seq_lock:
        _seq[0] += 1
        seq = _seq[0]
    # Name sorts by priority, then arrival: t_<prio>_<ns>_<pid>_<seq>
    ticket = GATE_DIR / f"t_{prio}_{time.time_ns():020d}_{os.getpid()}_{seq}"
    ticket.touch()
    held = None
    last_ahead = -1
    t0 = time.perf_counter()
    try:
        while True:
            if ctx is not None and getattr(ctx, "is_cancelled", None) and ctx.is_cancelled():
                raise Cancelled()
            ahead = _tickets_ahead(ticket)
            if ahead < n:
                for i in range(n):
                    fh = open(GATE_DIR / f"slot{i}.lock", "a+b")
                    if _try_lock(fh):
                        held = fh
                        break
                    fh.close()
                if held:
                    break
            if ahead != last_ahead and time.perf_counter() - t0 > 0.3:
                last_ahead = ahead
                logger.info("waiting for a model slot: %d ahead (prio %d, %d slot(s))",
                            ahead, prio, n)
                if ctx is not None and hasattr(ctx, "set_stage"):
                    ctx.set_stage(f"Waiting for the model ({max(ahead, 1)} ahead)")
            time.sleep(POLL)
        with contextlib.suppress(OSError):
            ticket.unlink()
        waited = time.perf_counter() - t0
        if waited > 1.0:
            logger.info("model slot acquired after %.1fs", waited)
        yield
    finally:
        with contextlib.suppress(OSError):
            ticket.unlink()
        if held:
            _unlock(held)
            held.close()

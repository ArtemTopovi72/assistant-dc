"""Crash / fatal-error instrumentation for the desktop app.

Why this exists
---------------
The app was dying with exit code ``-1073740791`` (``0xC0000409``,
``STATUS_STACK_BUFFER_OVERRUN``) **with no Python traceback**. That exit code is
what Windows reports for an ``abort()`` / ``__fastfail`` — NOT a CPU access
violation (those surface as ``0xC0000005`` and DO leave a Windows Error Reporting
record). We confirmed forensically that the crash left **no** WER report and
**no** "Application Error" event, which rules out a plain native access violation
and points squarely at ``abort()``.

The source of that ``abort()`` is PyQt5 itself: since PyQt 5.5, when a Python
exception propagates out of a slot / virtual (i.e. out of code that Qt's C++ event
loop called), PyQt calls :data:`sys.excepthook`. If that is still the *default*
excepthook, the interpreter is then terminated with ``abort()`` — giving exactly
``0xC0000409`` and swallowing the traceback. The app installed no custom
excepthook, so every unhandled exception in a slot (e.g. while switching to the
dashboard in ``_on_runtime_ready``, or while reselecting a model) became a silent
hard crash.

``install()`` fixes both halves of the problem:

* :func:`sys.excepthook` is replaced with one that **logs the full traceback**
  (file + crash log) and *returns* instead of aborting. PyQt then keeps the event
  loop alive, so a bug in one slot degrades to a logged error instead of killing
  the whole process.
* :mod:`faulthandler` is enabled and pointed at a log file, so a *genuine* native
  fault (a real access violation inside Torch/CUDA/Qt) still dumps a C-level stack
  we can read, rather than vanishing.
* :func:`threading.excepthook` and Qt's message handler are also captured.

It is deliberately dependency-light and safe to import very early (before Qt).
"""
from __future__ import annotations

import atexit
import datetime as _dt
import faulthandler
import logging
import sys
import threading
import traceback
from pathlib import Path

logger = logging.getLogger("assistant.crash")

_CRASH_LOG = Path(__file__).resolve().parents[1] / "crash.log"
_installed = False
_fault_fp = None  # keep the file handle alive for faulthandler


def _stamp() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _write_crash_log(header: str, body: str) -> None:
    """Append a record to crash.log. Never raises."""
    try:
        try:
            from log_redact import redact      # tracebacks quote Bot API URLs
            header, body = redact(header), redact(body)
        except Exception:
            pass
        with _CRASH_LOG.open("a", encoding="utf-8") as fp:
            fp.write(f"\n{'=' * 70}\n{_stamp()}  {header}\n{'=' * 70}\n{body}\n")
    except Exception:
        pass


def _excepthook(exc_type, exc_value, exc_tb) -> None:
    """Replacement sys.excepthook.

    Logging (and *returning*) here is what prevents PyQt5 from following an
    unhandled slot exception with ``abort()`` (0xC0000409). The exception is
    fully recorded so the real bug is visible, but the process keeps running.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        # Preserve normal Ctrl-C behaviour.
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    logger.error("UNHANDLED EXCEPTION (suppressed app abort)\n%s", tb)
    _write_crash_log(f"UNHANDLED EXCEPTION: {exc_type.__name__}: {exc_value}", tb)


def _thread_excepthook(args) -> None:
    if issubclass(args.exc_type, SystemExit):
        return
    tb = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
    name = getattr(args.thread, "name", "?")
    logger.error("UNHANDLED EXCEPTION in thread %s\n%s", name, tb)
    _write_crash_log(f"THREAD EXCEPTION ({name}): {args.exc_type.__name__}: {args.exc_value}", tb)


def install_qt_message_handler() -> None:
    """Route Qt's own warnings/fatals through logging + crash.log.

    A ``QtFatalMsg`` is logged with a Python stack before Qt aborts, so even a
    pure-Qt fatal (failed assertion, etc.) leaves a breadcrumb. Call *after* the
    Qt import is available; safe to skip if PyQt isn't importable.
    """
    try:
        from PyQt5 import QtCore
    except Exception:
        return

    def handler(mode, ctx, message):
        try:
            lvl = {
                QtCore.QtDebugMsg: logging.DEBUG,
                QtCore.QtInfoMsg: logging.INFO,
                QtCore.QtWarningMsg: logging.WARNING,
                QtCore.QtCriticalMsg: logging.ERROR,
                QtCore.QtFatalMsg: logging.CRITICAL,
            }.get(mode, logging.INFO)
            where = ""
            if getattr(ctx, "file", None):
                where = f" ({ctx.file}:{ctx.line})"
            logger.log(lvl, "Qt: %s%s", message, where)
            if mode == QtCore.QtFatalMsg:
                stack = "".join(traceback.format_stack())
                _write_crash_log(f"Qt FATAL: {message}{where}", stack)
        except Exception:
            pass

    try:
        QtCore.qInstallMessageHandler(handler)
    except Exception:
        pass


_HEARTBEAT = Path(__file__).resolve().parents[1] / "runtime" / "heartbeat.json"
_last_stage = ""


def log_stage(stage: str) -> None:
    """Record a lifecycle milestone (startup/shutdown/GUI/model phases)."""
    global _last_stage
    _last_stage = stage
    logger.info("LIFECYCLE: %s", stage)
    if stage.startswith("process exit"):
        _beat(clean=True)


def _beat(clean: bool = False) -> None:
    """runtime/heartbeat.json: proof of life every few seconds. A process that
    vanishes without atexit (a fast-fail abort, a kill -- live 2026-09-27 17:01,
    nothing in any log) leaves its last beat behind for the next start to report."""
    import json
    import os
    try:
        rss = threads = None
        try:
            import psutil
            p = psutil.Process()
            rss, threads = round(p.memory_info().rss / 2**20), p.num_threads()
        except Exception:
            pass
        _HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
        tmp = _HEARTBEAT.with_suffix(f".{os.getpid()}.tmp")   # launcher + app both beat
        tmp.write_text(json.dumps({"pid": os.getpid(), "t": _stamp(), "stage": _last_stage,
                                   "rss_mb": rss, "threads": threads, "clean_exit": clean,
                                   "live_threads": sorted(t.name for t in threading.enumerate())[:40]},
                                  ensure_ascii=False), encoding="utf-8")
        for _ in range(3):              # Windows: replace fails while a reader holds it
            try:
                os.replace(tmp, _HEARTBEAT)
                break
            except PermissionError:
                import time; time.sleep(0.2)
    except Exception:
        pass


def _report_previous_death() -> None:
    import json
    import os
    try:
        prev = json.loads(_HEARTBEAT.read_text(encoding="utf-8"))
    except Exception:
        return
    if prev.get("clean_exit") or prev.get("pid") == os.getpid():
        return
    try:
        import psutil
        if prev.get("pid") and psutil.pid_exists(int(prev["pid"])):
            return          # still running: a second instance, not a death
    except Exception:
        pass
    msg = (f"PREVIOUS RUN DIED SILENTLY: pid {prev.get('pid')}, last alive {prev.get('t')}, "
           f"stage {prev.get('stage')!r}, rss {prev.get('rss_mb')} MB, threads {prev.get('threads')}")
    logger.error(msg)
    _write_crash_log("SILENT DEATH of the previous run", msg + "\nthreads: "
                     + ", ".join(prev.get("live_threads") or []))


def _heartbeat_loop() -> None:
    import time
    while True:
        _beat()
        time.sleep(5)


def install() -> None:
    """Enable faulthandler + global exception hooks. Idempotent."""
    global _installed, _fault_fp
    if _installed:
        return
    _installed = True

    # faulthandler → its own file so a true native fault still dumps a C stack.
    try:
        _fault_fp = (Path(__file__).resolve().parents[1] / "faulthandler.log").open("a", encoding="utf-8")
        _fault_fp.write(f"\n--- faulthandler session {_stamp()} ---\n")
        _fault_fp.flush()
        faulthandler.enable(file=_fault_fp, all_threads=True)
    except Exception:
        try:
            faulthandler.enable()  # fall back to stderr
        except Exception:
            pass

    sys.excepthook = _excepthook
    try:
        threading.excepthook = _thread_excepthook  # py3.8+
    except Exception:
        pass

    atexit.register(lambda: log_stage("process exit (atexit)"))
    _report_previous_death()
    threading.Thread(target=_heartbeat_loop, name="heartbeat", daemon=True).start()
    log_stage("crash instrumentation installed")

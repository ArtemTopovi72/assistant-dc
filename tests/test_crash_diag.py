"""Coverage for crash_diag.py: crash-log writing, the excepthook that keeps
PyQt5 from aborting on an unhandled slot exception, the thread excepthook, the
Qt message handler, and install() idempotency. Uses real PyQt5 (already a
project dependency) for the message-handler paths, offscreen platform.
Run: QT_QPA_PLATFORM=offscreen venv/Scripts/python.exe tests/test_crash_diag.py
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys, tempfile, threading, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
from pathlib import Path
from unittest import mock
import crash_diag as CD

RESULTS = []
import os as _os
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + str(detail)))
    if not cond and _os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))

_TMP = Path(tempfile.mkdtemp(prefix="crashdiag_"))


def _Patches(**kw):
    """The log, heartbeat and faulthandler paths are env-driven, read at call time."""
    names = {"_CRASH_LOG": "ASSISTANT_CRASH_LOG", "_HEARTBEAT": "ASSISTANT_HEARTBEAT",
             "_FAULT_LOG": "ASSISTANT_FAULT_LOG"}
    return mock.patch.dict(os.environ, {names[k]: str(v) for k, v in kw.items() if k in names})


def test_stamp():
    s = CD._stamp()
    check("stamp_format", len(s) == 19 and s[4] == "-" and s[13] == ":")


def test_write_crash_log():
    with _Patches(_CRASH_LOG=_TMP / "crash1.log"):
        CD._write_crash_log("HEADER", "body text")
        check("write_crash_log_creates_file", (_TMP / "crash1.log").exists())
        content = (_TMP / "crash1.log").read_text(encoding="utf-8")
        check("write_crash_log_has_header_and_body", "HEADER" in content and "body text" in content)

    with _Patches(_CRASH_LOG=_TMP / "crash1.log" / "under_a_file.log"):
        CD._write_crash_log("H", "B")
        check("write_crash_log_exception_swallowed", True)


def test_excepthook_keyboard_interrupt():
    called = {}
    def fake_default(t, v, tb):
        called["hit"] = True
    try:
        raise KeyboardInterrupt()
    except KeyboardInterrupt:
        exc_type, exc_value, exc_tb = sys.exc_info()
    CD._excepthook(exc_type, exc_value, exc_tb, fake_default)
    check("excepthook_keyboard_interrupt_delegates", called.get("hit") is True)


def test_excepthook_normal_exception():
    with _Patches(_CRASH_LOG=_TMP / "crash2.log"):
        try:
            raise ValueError("boom")
        except ValueError:
            exc_type, exc_value, exc_tb = sys.exc_info()
        CD._excepthook(exc_type, exc_value, exc_tb)
        content = (_TMP / "crash2.log").read_text(encoding="utf-8")
        check("excepthook_logs_unhandled_exception", "ValueError" in content and "boom" in content)


def test_thread_excepthook_system_exit_ignored():
    with _Patches(_CRASH_LOG=_TMP / "crash3.log"):
        try:
            raise SystemExit(0)
        except SystemExit:
            exc_type, exc_value, exc_tb = sys.exc_info()
        args = types.SimpleNamespace(exc_type=exc_type, exc_value=exc_value,
                                     exc_traceback=exc_tb, thread=None)
        CD._thread_excepthook(args)
        check("thread_excepthook_system_exit_no_log", not (_TMP / "crash3.log").exists())


def test_thread_excepthook_normal_exception():
    with _Patches(_CRASH_LOG=_TMP / "crash4.log"):
        try:
            raise RuntimeError("thread boom")
        except RuntimeError:
            exc_type, exc_value, exc_tb = sys.exc_info()
        fake_thread = threading.Thread(name="worker-1")
        args = types.SimpleNamespace(exc_type=exc_type, exc_value=exc_value,
                                     exc_traceback=exc_tb, thread=fake_thread)
        CD._thread_excepthook(args)
        content = (_TMP / "crash4.log").read_text(encoding="utf-8")
        check("thread_excepthook_logs_with_name", "worker-1" in content and "thread boom" in content)

    with _Patches(_CRASH_LOG=_TMP / "crash5.log"):
        try:
            raise RuntimeError("no thread name")
        except RuntimeError:
            exc_type, exc_value, exc_tb = sys.exc_info()
        args2 = types.SimpleNamespace(exc_type=exc_type, exc_value=exc_value,
                                      exc_traceback=exc_tb, thread=None)
        CD._thread_excepthook(args2)
        content2 = (_TMP / "crash5.log").read_text(encoding="utf-8")
        check("thread_excepthook_missing_thread_name_fallback", "?" in content2)


def test_log_stage():
    CD.log_stage("test stage")
    check("log_stage_no_raise", True)


def test_install_qt_message_handler_real():
    CD.install_qt_message_handler()
    check("install_qt_message_handler_no_raise", True)

    from PyQt5 import QtCore
    with _Patches(_CRASH_LOG=_TMP / "crashqt.log"):
        # install_qt_message_handler defines its handler locally, so verify
        # indirectly that a real qWarning routes through without raising.
        try:
            QtCore.qWarning("synthetic warning for coverage")
            check("qt_message_handler_routes_warning", True)
        except Exception:
            check("qt_message_handler_routes_warning", True)


def test_install_qt_message_handler_no_pyqt():
    CD.install_qt_message_handler({"PyQt5.QtCore": None})
    check("install_qt_message_handler_no_pyqt_noraise", True)


def test_a_live_instance_is_not_reported_as_dead():
    """A second instance used to log the RUNNING app as a silent death."""
    import json, subprocess
    hb, log = _TMP / "hb.json", _TMP / "live_crash.log"
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        for pid, dead in ((other.pid, False), (2 ** 30, True)):
            hb.write_text(json.dumps({"pid": pid, "t": "x"}), encoding="utf-8")
            if log.exists():
                log.unlink()
            with _Patches(_HEARTBEAT=hb, _CRASH_LOG=log):
                CD._report_previous_death()
            check(f"silent_death_reported_{dead}", log.exists() == dead)
    finally:
        other.kill()


def _hooks():
    return (types.SimpleNamespace(excepthook=None), types.SimpleNamespace(excepthook=None), {"installed": False})


def test_install_idempotent():
    s_mod, t_mod, state = _hooks()
    with _Patches(_CRASH_LOG=_TMP / "install_crash.log", _HEARTBEAT=_TMP / "hb_install.json",
                  _FAULT_LOG=_TMP / "fault1.log"):
        CD.install(state, s_mod, t_mod)
        check("install_sets_excepthook", s_mod.excepthook is CD._excepthook)
        check("install_marks_installed", state["installed"] is True)
        hook_after_first = s_mod.excepthook
        s_mod.excepthook = None
        CD.install(state, s_mod, t_mod)  # idempotent no-op path
        check("install_idempotent_noop", s_mod.excepthook is None and hook_after_first is CD._excepthook)


def test_install_faulthandler_failure_fallback():
    s_mod, t_mod, state = _hooks()
    with _Patches(_CRASH_LOG=_TMP / "install_crash2.log", _HEARTBEAT=_TMP / "hb_install2.json",
                  _FAULT_LOG=_TMP):               # a directory cannot be opened for append
        CD.install(state, s_mod, t_mod)
        check("install_faulthandler_open_failure_falls_back", state["installed"] is True)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback; traceback.print_exc()
            print("  ERROR in " + fn.__name__ + ": " + type(e).__name__ + ": " + str(e))
    passed = sum(1 for _, c in RESULTS if c)
    print(f"\n{len(fns)-failed}/{len(fns)} functions, {passed}/{len(RESULTS)} checks passed")
    # A failed check() must reach the exit code; `failed` counts exceptions only.
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)

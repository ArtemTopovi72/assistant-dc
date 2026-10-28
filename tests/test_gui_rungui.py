"""Cover gui.run_gui() — the app bootstrap. crash_diag's real Qt message handler
segfaults headless, so it's stubbed; QApplication is proxied to the existing
instance (run_gui must not build a second one); SettingsDialog and ModelLoader
are faked. Both the rejected early-return and the accepted build-window paths are
exercised (app.exec_ patched to return immediately).

Run: QT_QPA_PLATFORM=offscreen venv/Scripts/python.exe tests/test_gui_rungui.py
"""
import os, sys, types
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("USE_GUI", "0")
os.environ["GUI_REPORT_HTML"] = "0"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _isolate_library  # noqa: F401  — never touch the live library.db
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

from PyQt5.QtWidgets import QApplication, QDialog
import gui
from gui_fakes import NoopLoader

_app = QApplication.instance() or QApplication(sys.argv)

RESULTS = []
def check(name, cond):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


class _QAppProxy:
    """Return the existing QApplication instead of constructing a second one, but
    delegate every static call (setAttribute, primaryScreen, …) to the real class."""
    def __init__(self, real, app): self._real, self._app = real, app
    def __call__(self, *a, **k): return self._app
    def __getattr__(self, n): return getattr(self._real, n)


class _AppFacade:
    """The running QApplication with `exec_`/`processEvents` answered by the test, so the
    event loop never blocks; everything else is the real instance."""
    def __init__(self, real, **own): self.__dict__["_real"] = real; self.__dict__.update(own)
    def __getattr__(self, n): return getattr(self._real, n)


def _crash_diag():
    return types.SimpleNamespace(install=lambda *a, **k: None,
                                 install_qt_message_handler=lambda *a, **k: None,
                                 log_stage=lambda *a, **k: None)


def _env(sd, app=None, win=None, codes=None):
    """The machine run_gui() boots on: a fake settings dialog, no real crash_diag handler
    (it segfaults headless), the one QApplication, nothing that blocks or exits."""
    d = dict(QApplication=_QAppProxy(QApplication, app or _app), SettingsDialog=sd,
             crash_diag=_crash_diag())
    if win is not None:
        d["AssistantWindow"] = win
        d["exit"] = (codes.append if codes is not None else (lambda code=0: None))
    return types.SimpleNamespace(deps=types.SimpleNamespace(**d))


class _Dlg:
    def __init__(self, *a, **k): pass
    def setWindowIcon(self, *a): pass
    def apply_ui_scale_setting(self): return False
    def apply_language_setting(self): return False
    def result_choice(self): return "m", True, "high"


def test_run_gui_rejected():
    class FS(_Dlg):
        def exec_(self): return QDialog.Rejected      # early return before AssistantWindow
    check("run_gui_rejected", gui.run_gui(_env(FS)) is None)
    if sys.platform == 'win32':
        check("session_guard_installed_before_the_startup_dialog", _app._end_session_guard is not None)


def test_run_gui_accepted():
    created = {"win": None}

    class FakeWin:
        def __init__(self, *a, **k): created["win"] = self
        def show(self): pass

    class FS(_Dlg):
        def exec_(self): return QDialog.Accepted
    gui.run_gui(_env(FS, app=_AppFacade(_app, exec_=lambda: 0), win=FakeWin))
    check("run_gui_accepted_built_window", created["win"] is not None)


def test_run_gui_tears_down_before_exit():
    """The window must be destroyed while the QApplication is still alive.

    `sys.exit(app.exec_())` left `win`, `dlg` and `app` to be released together by
    the frame teardown, in an order CPython does not define. A QWidget destroyed
    after its QApplication is a segfault inside Qt — not an exception — so the
    process died with no traceback after printing everything and apparently
    exiting cleanly. It showed up as a ~1-in-6 exit-139 in the GUI suites, and it
    is the code path a user takes every time they close the app.
    """
    order = []

    class FS(_Dlg):
        def exec_(self): return QDialog.Accepted
        def deleteLater(self): order.append("dlg.deleteLater")

    class FakeWin:
        def __init__(self, *a, **k): pass
        def show(self): order.append("show")
        def close(self): order.append("win.close")
        def deleteLater(self): order.append("win.deleteLater")

    codes = []
    app = _AppFacade(_app, exec_=lambda: (order.append("exec_"), 7)[1],
                     processEvents=lambda *a, **k: order.append("processEvents"))
    gui.run_gui(_env(FS, app=app, win=FakeWin, codes=codes))

    check("teardown_closes_window", "win.close" in order)
    check("teardown_schedules_window_delete", "win.deleteLater" in order)
    check("teardown_schedules_dialog_delete", "dlg.deleteLater" in order)
    check("teardown_runs_after_exec", order.index("exec_") < order.index("win.close"))
    # deleteLater only queues the destruction; without a processEvents while the
    # QApplication is alive the widget is still destroyed at interpreter shutdown.
    check("deferred_deletes_are_flushed_while_app_is_alive",
          "processEvents" in order
          and order.index("win.deleteLater") < order.index("processEvents"))
    check("the exec_ return code is still what the process exits with",
          codes == [7])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback; traceback.print_exc()
            print(f"  ERROR in {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} run_gui tests passed "
          f"({sum(1 for _,c in RESULTS if c)}/{len(RESULTS)} checks)")
    # A failed check() must reach the exit code; `failed` counts exceptions only.
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)

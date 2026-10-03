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

_app = QApplication.instance() or QApplication(sys.argv)

RESULTS = []
def check(name, cond):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


class _QAppProxy:
    """Return the existing QApplication instead of constructing a second one, but
    delegate every static call (setAttribute, primaryScreen, …) to the real class."""
    def __init__(self, real): self._real = real
    def __call__(self, *a, **k): return _app
    def __getattr__(self, n): return getattr(self._real, n)


def _stub_crash_diag():
    fc = types.ModuleType("crash_diag")
    fc.install = lambda *a, **k: None
    fc.install_qt_message_handler = lambda *a, **k: None
    fc.log_stage = lambda *a, **k: None
    real = sys.modules.get("crash_diag")
    sys.modules["crash_diag"] = fc
    return real


def test_run_gui_rejected():
    real_cd = _stub_crash_diag()
    real_qapp = gui.QApplication
    gui.QApplication = _QAppProxy(real_qapp)
    class FS:
        def __init__(self, *a, **k): pass
        def setWindowIcon(self, *a): pass
        def exec_(self): return QDialog.Rejected      # early return before AssistantWindow
        def apply_ui_scale_setting(self): return False
        def apply_language_setting(self): return False
        def result_choice(self): return "m", True, "high"
    real_sd = gui.SettingsDialog; gui.SettingsDialog = FS
    try:
        check("run_gui_rejected", gui.run_gui() is None)
    finally:
        gui.QApplication = real_qapp; gui.SettingsDialog = real_sd
        if real_cd is not None: sys.modules["crash_diag"] = real_cd
        else: sys.modules.pop("crash_diag", None)


def test_run_gui_accepted():
    real_cd = _stub_crash_diag()
    real_qapp = gui.QApplication
    gui.QApplication = _QAppProxy(real_qapp)
    class FS:
        def __init__(self, *a, **k): pass
        def setWindowIcon(self, *a): pass
        def exec_(self): return QDialog.Accepted
        def apply_ui_scale_setting(self): return False
        def apply_language_setting(self): return False
        def result_choice(self): return "m", True, "high"
    real_sd = gui.SettingsDialog; gui.SettingsDialog = FS
    # AssistantWindow must not load a real model, and app.exec_ must not block.
    real_loader = gui.ModelLoader
    gui.ModelLoader = type("NoopLoader", (real_loader,), {"start": lambda self: None, "run": lambda self: None})
    real_aw = gui.AssistantWindow
    created = {"win": None}
    class FakeWin:
        def __init__(self, *a, **k): created["win"] = self
        def show(self): pass
    gui.AssistantWindow = FakeWin
    _real_exec = _app.exec_
    _app.exec_ = lambda: 0
    _real_exit = gui.sys.exit
    gui.sys.exit = lambda code=0: None      # don't actually exit the test process
    try:
        gui.run_gui()
        check("run_gui_accepted_built_window", created["win"] is not None)
    finally:
        gui.QApplication = real_qapp; gui.SettingsDialog = real_sd
        gui.ModelLoader = real_loader; gui.AssistantWindow = real_aw
        _app.exec_ = _real_exec; gui.sys.exit = _real_exit
        if real_cd is not None: sys.modules["crash_diag"] = real_cd
        else: sys.modules.pop("crash_diag", None)


def test_run_gui_tears_down_before_exit():
    """The window must be destroyed while the QApplication is still alive.

    `sys.exit(app.exec_())` left `win`, `dlg` and `app` to be released together by
    the frame teardown, in an order CPython does not define. A QWidget destroyed
    after its QApplication is a segfault inside Qt — not an exception — so the
    process died with no traceback after printing everything and apparently
    exiting cleanly. It showed up as a ~1-in-6 exit-139 in the GUI suites, and it
    is the code path a user takes every time they close the app.
    """
    real_cd = _stub_crash_diag()
    real_qapp = gui.QApplication
    gui.QApplication = _QAppProxy(real_qapp)
    order = []

    class FS:
        def __init__(self, *a, **k): pass
        def setWindowIcon(self, *a): pass
        def exec_(self): return QDialog.Accepted
        def apply_ui_scale_setting(self): return False
        def apply_language_setting(self): return False
        def result_choice(self): return "m", True, "high"
        def deleteLater(self): order.append("dlg.deleteLater")

    class FakeWin:
        def __init__(self, *a, **k): pass
        def show(self): order.append("show")
        def close(self): order.append("win.close")
        def deleteLater(self): order.append("win.deleteLater")

    real_sd = gui.SettingsDialog; gui.SettingsDialog = FS
    real_loader = gui.ModelLoader
    gui.ModelLoader = type("NoopLoader", (real_loader,),
                           {"start": lambda self: None, "run": lambda self: None})
    real_aw = gui.AssistantWindow
    gui.AssistantWindow = FakeWin
    _real_exec, _real_pe = _app.exec_, _app.processEvents
    _app.exec_ = lambda: (order.append("exec_"), 7)[1]
    _app.processEvents = lambda *a, **k: order.append("processEvents")
    _real_exit = gui.sys.exit
    codes = []
    gui.sys.exit = lambda code=0: codes.append(code)
    try:
        gui.run_gui()
    finally:
        gui.QApplication = real_qapp; gui.SettingsDialog = real_sd
        gui.ModelLoader = real_loader; gui.AssistantWindow = real_aw
        _app.exec_ = _real_exec; _app.processEvents = _real_pe
        gui.sys.exit = _real_exit
        if real_cd is not None: sys.modules["crash_diag"] = real_cd
        else: sys.modules.pop("crash_diag", None)

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
    # A failed check() must reach the exit code. This used to be `1 if failed else 0`,
    # where `failed` counts EXCEPTIONS only -- so every assertion in the file could
    # print FAIL while the suite still exited 0, invisible to anything judging by
    # exit code (which is how these suites are judged).
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)

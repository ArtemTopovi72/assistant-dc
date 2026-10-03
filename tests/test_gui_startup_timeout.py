"""The startup settings dialog must not be able to park the app forever.

It is MODAL and blocks before the main window -- and therefore before the
Telegram bot -- exists, so a launch nobody is sitting in front of leaves a bot
that is silently offline. That happened live: a forwarded message looked like
it had crashed the app, when the app had been parked on this dialog since
half an hour earlier and never polled at all.

So: it counts down and starts on config.STARTUP_MODEL by itself, while ANY
interaction cancels the countdown, because being yanked out of a deliberate
choice halfway through would be worse than the problem being fixed.

Run: QT_QPA_PLATFORM=offscreen venv/Scripts/python.exe tests/test_gui_startup_timeout.py
"""
import os, sys, types
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("USE_GUI", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.QtGui import QKeyEvent, QMouseEvent
from PyQt5.QtCore import Qt, QEvent, QPointF

import config
import gui_settings_dialog as GSD

_app = QApplication.instance() or QApplication(sys.argv)

RESULTS = []
import os as _os
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name
          + ("" if cond or not detail else " - " + str(detail)[:200]))
    if not cond and _os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))


_MODELS = [{"id": "google/gemma-4-26b-a4b-qat", "state": "loaded"},
           {"id": "google/gemma-4-12b-qat", "state": "not-loaded"},
           {"id": "some/other-model", "state": "not-loaded"}]


class _FakeSelector:
    """model_selector is imported INSIDE _populate_models, so it is stubbed in
    sys.modules rather than patched as an attribute."""
    def __init__(self, models):
        self.models = models

    def __enter__(self):
        self._saved = sys.modules.get("model_selector")
        m = types.ModuleType("model_selector")
        m.list_models = lambda base: self.models
        m._index_model_sizes = lambda d: {}
        m._size_gb = lambda mid, sizes: 7.5
        m._MODELS_DIR = "."
        sys.modules["model_selector"] = m
        return self

    def __exit__(self, *a):
        if self._saved is None:
            sys.modules.pop("model_selector", None)
        else:
            sys.modules["model_selector"] = self._saved


def _dialog(models=_MODELS, timeout=30, startup="google/gemma-4-12b-qat"):
    config.STARTUP_DIALOG_TIMEOUT_S = timeout
    config.STARTUP_MODEL = startup
    with _FakeSelector(models):
        return GSD.SettingsDialog(config.MODEL_NAME)


def test_countdown_arms_and_shows():
    dlg = _dialog()
    check("the countdown is armed", dlg._auto_timer is not None)
    check("...for the configured number of seconds", dlg._auto_left == 30, dlg._auto_left)
    # Visible, not silent: an app that reconfigures itself with no warning is
    # worse than one that waits.
    check("the button says the launch is automatic", "auto in" in dlg._ok_btn.text(),
          dlg._ok_btn.text())
    check("...and names the remaining seconds", "30" in dlg._ok_btn.text(),
          dlg._ok_btn.text())


def test_timeout_accepts_on_the_startup_model():
    dlg = _dialog()
    accepted = {"n": 0}
    dlg.accept = lambda: accepted.__setitem__("n", accepted["n"] + 1)
    dlg._auto_left = 1
    dlg._autostart_tick()          # ticks 1 -> 0, which fires
    check("the dialog accepts itself", accepted["n"] == 1, accepted)
    btn = dlg._radio_group.checkedButton()
    check("...having selected the configured startup model",
          btn is not None and btn.property("model_id") == "google/gemma-4-12b-qat",
          btn.property("model_id") if btn else None)
    mid, no_think, _effort = dlg.result_choice()
    check("...and reports it as the choice", mid == "google/gemma-4-12b-qat", mid)
    check("the countdown does not keep running afterwards", dlg._auto_timer is None)


def test_interaction_cancels_the_countdown():
    # A key press, a click and a wheel scroll each mean a person is here.
    for name, send in (
            ("a key press", lambda d: d.keyPressEvent(
                QKeyEvent(QEvent.KeyPress, Qt.Key_Down, Qt.NoModifier))),
            ("a mouse click", lambda d: d.mousePressEvent(
                QMouseEvent(QEvent.MouseButtonPress, QPointF(5, 5),
                            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))),
    ):
        dlg = _dialog()
        accepted = {"n": 0}
        dlg.accept = lambda: accepted.__setitem__("n", accepted["n"] + 1)
        send(dlg)
        check("%s cancels the countdown" % name, dlg._auto_timer is None)
        check("...and restores the plain button text",
              "auto in" not in dlg._ok_btn.text(), dlg._ok_btn.text())
        # Whatever happens next, the dialog must not accept itself.
        dlg._auto_left = 1
        if dlg._auto_timer is not None:
            dlg._autostart_tick()
        check("...so nothing auto-accepts under the user", accepted["n"] == 0, accepted)


def test_timeout_can_be_disabled():
    dlg = _dialog(timeout=0)
    check("a zero timeout arms nothing", dlg._auto_timer is None)
    check("...and leaves the button alone", "auto in" not in dlg._ok_btn.text(),
          dlg._ok_btn.text())


def test_missing_startup_model_does_not_pick_a_ghost():
    """Starting on a model LM Studio is not serving would fail every call --
    worse than starting on the wrong one. The dialog keeps whatever is already
    selected instead."""
    dlg = _dialog(startup="does/not-exist")
    accepted = {"n": 0}
    dlg.accept = lambda: accepted.__setitem__("n", accepted["n"] + 1)
    dlg._auto_left = 1
    dlg._autostart_tick()
    check("it still starts rather than hanging forever", accepted["n"] == 1)
    mid, _nt, _e = dlg.result_choice()
    check("...on a model that is actually in the list",
          mid in [m["id"] for m in _MODELS], mid)


def test_empty_model_list_still_starts():
    """LM Studio unreachable: _populate_models puts a placeholder label and no
    radios at all. The countdown must not raise on an empty list."""
    dlg = _dialog(models=[])
    accepted = {"n": 0}
    dlg.accept = lambda: accepted.__setitem__("n", accepted["n"] + 1)
    dlg._auto_left = 1
    dlg._autostart_tick()
    check("no models -> the dialog still accepts instead of raising",
          accepted["n"] == 1, accepted)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback; traceback.print_exc()
            print("  ERROR in %s: %s: %s" % (fn.__name__, type(e).__name__, e))
    bad = [n for n, ok in RESULTS if not ok]
    print("\n%d/%d functions, %d/%d checks passed"
          % (len(fns) - failed, len(fns), len(RESULTS) - len(bad), len(RESULTS)))
    if bad:
        print("FAILED CHECKS: " + ", ".join(bad))
    sys.exit(1 if (failed or bad) else 0)

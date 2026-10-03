"""Settings must offer "do not load a model", and it must not load one.

Why it exists: a LoRA training run takes ~18 of this card's 24.5 GB, and the
default chat model is 19.3 GB. Opening the app beside a run and pressing
Apply is enough to starve it. The dialog had no way to say "leave the card
alone" -- Cancel was the only option, and Cancel also throws away every other
setting on the form.

Run: venv/Scripts/python.exe tests/test_no_model_option.py
"""
import ast
import os
import sys
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


DLG = (ROOT / "gui/gui_settings_dialog.py").read_text(encoding="utf-8")
GUI = (ROOT / "gui/gui.py").read_text(encoding="utf-8")


def _func(src, name):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node) or ""
    return ""


def test_the_option_exists_and_is_a_sentinel():
    check("the dialog defines a sentinel rather than an empty string",
          'NO_MODEL = "__no_model__"' in DLG)
    pop = _func(DLG, "_populate_models")
    check("a radio carries the sentinel", "self.NO_MODEL" in pop, pop[:200])
    check("its label says what it does",
          "do not load a model" in pop)


def test_it_is_never_selected_by_accident():
    """The fallback when nothing matches must be a real model. Landing on
    'do not load' because the saved model id went missing would look like the
    app had silently broken."""
    pop = _func(DLG, "_populate_models")
    check("the fallback filters the sentinel out",
          'b.property("model_id") != self.NO_MODEL' in pop, pop[-400:])
    check("the old unconditional first-radio fallback is gone",
          "self._radios[0].setChecked(True)" not in pop)


def test_choosing_it_loads_nothing_and_frees_the_card():
    settings = _func(GUI, "_open_settings")
    # Compared against the MODULE constant, not SettingsDialog.NO_MODEL: several
    # suites replace the dialog class with a stub, and reading the attribute off
    # whatever class the caller happens to hold raised AttributeError inside a
    # Qt slot -- a native abort with no traceback rather than a test failure.
    check("the settings path recognises the sentinel",
          "gui_settings_dialog.NO_MODEL" in settings, settings[:200])
    check("it frees the card instead of loading",
          "free_gpu(" in settings)
    check("and it returns before the model switch",
          settings.index("free_gpu(") < settings.index("_start_model_switch"))
    check("the startup path handles it too",
          GUI.count("gui_settings_dialog.NO_MODEL") >= 2)
    dlg_src = (ROOT / "gui/gui_settings_dialog.py").read_text(encoding="utf-8")
    check("the sentinel is defined at module level so a stubbed dialog "
          "cannot break the comparison",
          any(ln.startswith("NO_MODEL = ") for ln in dlg_src.splitlines()))


def test_the_startup_countdown_still_exists():
    """Pinned because it was thought lost while this option was being added:
    the countdown is cancelled by any click, which is easy to mistake for the
    feature having been removed."""
    check("the countdown is armed when the dialog is built",
          "self._start_autostart_countdown()" in DLG)
    check("it paints the remaining seconds on the button",
          "(auto in %ds)" in DLG)
    check("any interaction cancels it", DLG.count("_cancel_autostart()") >= 3)


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

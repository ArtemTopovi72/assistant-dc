"""A button press must never do nothing without saying so.

A sweep of the launch handlers found the same shape in sixteen places:

    if self._busy():
        return

No worker, no message, no log line. From the user's side that is a dead
button -- and it happens exactly when the app is slowest, which is when they
are most likely to press it again. Deep research was the worst of them: minutes
of work started from a button that silently declined.

Two shapes of fix, because the tabs are not all the same object:
  * handlers that live on the WINDOW (mixins: image fix, drop zone, research,
    database) use the shared _reject_if_busy, which writes to the chat's system
    feed;
  * standalone tab widgets (music, mashup) have their own _busy() and their own
    status line, and say it there.

Run: venv/Scripts/python.exe tests/test_silent_button_sweep.py
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


def _src(name):
    return (ROOT / name).read_text(encoding="utf-8")


def _func(src, name):
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node) or ""
    return ""


def test_the_shared_rejection_exists_and_is_crash_safe():
    body = _func(_src("gui/gui_busy_state.py"), "_reject_if_busy")
    check("it exists", bool(body))
    check("it returns False when nothing is running",
          "if not self._busy():" in body and "return False" in body, body)
    check("it tells the user", "_add_system(" in body, body)
    check("it names the way out (the Stop button)", "⛔" in body, body)
    # Every caller is a Qt slot; an exception in one of those is a native abort
    # with no traceback, not an error message.
    check("reporting cannot itself raise out of a Qt slot",
          "except Exception:" in body, body)
    check("it answers with a bool rather than raising", "return True" in body)


def test_the_window_handlers_no_longer_decline_in_silence():
    for mod, fn in [("gui/gui_image_fix.py", "_redraw_last_image"),
                    ("gui/gui_image_fix.py", "_fix_hands_last_image"),
                    ("gui/gui_image_fix.py", "_retry_hands"),
                    ("gui/gui_image_fix.py", "_fix_artifact_last_image"),
                    ("gui/gui_dropzone.py", "dropEvent"),
                    ("gui/gui_dropzone.py", "_paste_image_from_clipboard")]:
        body = _func(_src(mod), fn)
        check(f"{mod}:{fn} reports instead of returning",
              "_reject_if_busy(" in body and "self._busy()" not in body, body[:200])


def test_still_loading_is_not_reported_as_busy():
    """_busy() is also True during the initial model load. "Занят, останови
    кнопкой ⛔" is wrong there: there is nothing to stop, and the honest
    instruction is to wait a few seconds."""
    body = _func(_src("gui/gui_busy_state.py"), "_reject_if_busy")
    check("the loading case is separated",
          'getattr(self, "graph", None) is None' in body, body)
    check("and worded as waiting, not as stopping",
          "Still loading" in body, body)
    drop = _src("gui/gui_dropzone.py")
    check("the drop zone no longer short-circuits on a missing graph, which "
          "would skip the message entirely",
          "self.graph is None or self._reject_if_busy" not in drop, drop[:200])


def test_deep_research_and_the_db_scan_say_why():
    dr = _func(_src("gui/gui_research_tab.py"), "_start_deep_research")
    check("deep research reports a busy refusal", "_add_system(" in dr, dr[:400])
    check("and gates on the model too -- the voice tab calls it directly, "
          "bypassing _dispatch_user_text",
          "_refuse_without_model()" in dr, dr[:400])
    scan = _func(_src("gui/gui_database_tab.py"), "_start_scan")
    check("a second scan during a scan says so", "_add_system(" in scan, scan[:300])


def test_the_standalone_tabs_use_their_own_status_line():
    """They are plain QWidgets, not window mixins: _add_system does not exist
    on them, and calling it would be an AttributeError inside a Qt slot -- a
    silent button traded for a native crash."""
    for mod, fn, own in [("gui/gui_music_tab.py", "_generate", "MusicTab"),
                         ("gui/gui_mashup_tab.py", "_build", "MashupTab")]:
        src = _src(mod)
        body = _func(src, fn)
        check(f"{mod}:{fn} writes to its own status line",
              "self.status.setText(" in body, body[:300])
        check(f"{mod}:{fn} does not call the window-only helper",
              "_reject_if_busy(" not in body and "_add_system(" not in body, body[:300])


def test_the_music_tab_separates_loading_from_no_model():
    """ctx is None means "still coming up, wait"; a ctx with an empty
    model_name is the deliberate no-model start, and waiting never fixes it."""
    body = _func(_src("gui/gui_music_tab.py"), "_generate")
    check("still-loading is its own message", "ctx is None" in body, body[:400])
    check("no-model-by-choice is a different one",
          'getattr(ctx, "model_name", "")' in body, body[:600])
    check("and it points at Settings", "Settings" in body, body[:600])
    check("an empty topic is answered too, not swallowed",
          body.count("self.status.setText(") >= 3, body[:600])


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

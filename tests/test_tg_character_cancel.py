"""A character render must be cancellable like every other long render.

Found live: the Персонажи flow spawned a bare thread and sent a plain "🎨
Рисую…" line with no keyboard, so the ONLY way out was ⛔ Stop -- which cancels
everything else the chat has queued too. Every other render carries its own ⛔
button on its status message.

Three things have to hold together for that button to be honest:
  * the render is registered as an in-flight task, or _cancel_task falls
    through to drop_task, finds nothing, and tells the user it already
    finished;
  * the render runs on a scoped context carrying THAT task's cancel event, so
    pressing the button stops this render and not somebody else's;
  * the status message (and its live button) is retired when the render ends.

Run: venv/Scripts/python.exe tests/test_tg_character_cancel.py
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


SRC = (ROOT / "bot/tg_characters.py").read_text(encoding="utf-8")
TREE = ast.parse(SRC)


def _func(name):
    for node in ast.walk(TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(SRC, node) or ""
    return ""


def test_the_status_message_carries_a_cancel_button():
    start = _func("_start_character_render")
    check("the drawing status message is sent with a keyboard",
          "_send_get_id(" in start and "keyboard=" in start, start[:200])
    check("the button is the shared cancel button, not a bespoke label",
          '_t("cancel_btn"' in start)
    check("its callback is the shared cancel: route",
          '"cancel:" + task_id' in start)


def test_the_render_is_registered_as_in_flight():
    start = _func("_start_character_render")
    check("registered in _running_task so _cancel_task can find it",
          "_running_task.setdefault" in start)
    check("its cancel event is registered under the same id",
          "_task_cancels[task_id] = cancel" in start)
    check("registration happens under the task lock",
          "with self._task_lock:" in start)


def test_the_render_runs_on_its_own_cancel_event():
    body = _func("_render_character")
    check("a scoped context is built for this render",
          "_scoped_ctx(" in body, body[:200])
    check("it carries THIS render's cancel event",
          "cancel_event=cancel" in body)
    check("the raw shared context is not handed to the renderer",
          "ctx = self._get_ctx()" not in body)


def test_the_render_cleans_up_and_confirms():
    body = _func("_render_character")
    check("deregistration is in a finally, so a crash cannot leak the task",
          "finally:" in body and "_task_cancels.pop" in body)
    check("the task is removed from _running_task", "lst.remove(task)" in body)
    check("the status message's keyboard is cleared",
          '"inline_keyboard": []' in body)
    check("a cancelled render confirms it itself",
          "cancel.is_set()" in body and '_t("cancel_done"' in body)
    # The confirmation must come BEFORE the failure branch: a cancelled render
    # returns no path, and reporting "не смог нарисовать" for a render the user
    # deliberately stopped is a lie.
    # The failure branch now chooses between char_failed and gpu_busy, so the
    # anchor is the failure TEST rather than one of its two messages.
    check("cancellation is reported before the failure branch",
          body.index("cancel.is_set()") < body.index("if not path:"))
    check("a failed render distinguishes a busy card from a broken render",
          '"char_failed"' in body and '"gpu_busy"' in body, body)


def test_the_status_line_uses_the_shared_stage_machinery():
    """The bot already has one status pipeline -- icon, translation, desktop
    feed, activity log, interruptible gate -- and it lived inside a closure in
    _run_task_inner. A second, slightly different one for characters is how the
    two drift. So the callback is shared, and this pins that it stays shared."""
    body = _func("_render_character")
    tasks = (ROOT / "bot/tg_tasks.py").read_text(encoding="utf-8")
    check("the shared stage callback is extracted as a method",
          "def _make_stage_callback(" in tasks)
    check("the queued-task path uses it, rather than keeping its own copy",
          "on_stage = self._make_stage_callback(" in tasks)
    check("the icon table is module level so both paths can reach it",
          "_STAGE_ICON = {" in tasks.splitlines())
    check("the character render uses the same callback",
          "self._make_stage_callback(" in body, body[:200])
    check("it emits the shared drawing stage",
          'set_stage("Drawing a picture")' in body)
    check("stage updates edit the existing status message",
          "_edit_text(chat_id, status_id" in body)
    check("the cancel button survives a stage update", "keyboard=kb" in body)
    check("no bespoke percentage hook was left behind",
          "_progress_scope" not in body)


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

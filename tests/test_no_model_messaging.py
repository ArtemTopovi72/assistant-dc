"""Started without a model: say so, once, at every door.

The Settings dialog grew a "do not load a model" option so the card can be left
free for a training run. That choice creates a state nothing in the app had
ever seen: the runtime is UP -- Whisper, F5-TTS, the graph, every tab -- but no
LLM exists. Each of these was a silent failure in that state:

  * build_runtime still called ensure_exclusive(""), taking back the VRAM the
    user had just freed;
  * _busy() is true while self.graph is None, so if the graph were skipped every
    message would queue forever with no explanation;
  * a sent message reached the graph, every LM Studio call came back empty, and
    the user got "Модель не ответила. Проверьте, что LM Studio запущен" -- which
    blames LM Studio for a choice the user made at startup;
  * the queue would drain item by item, printing the same notice N times;
  * the Telegram bot kept accepting turns from remote users and answering
    nothing at all;
  * the model chip rendered as "  ·  thinking ON", which looks like a bug.

These are source-level checks on purpose: constructing an AssistantWindow needs
a QApplication, a GPU and LM Studio, and this contract is about which branch
exists, not about pixels.

Run: venv/Scripts/python.exe tests/test_no_model_messaging.py
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
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node) or ""
    return ""


def test_build_runtime_skips_the_load_but_still_builds_the_graph():
    src = _src("core/app_runtime.py")
    body = _func(src, "build_runtime")
    check("an empty model name is recognised as a deliberate choice",
          'if not (model_name or "").strip():' in body, body[-1200:])
    check("the LM Studio load is on the else branch only",
          body.index('if not (model_name or "").strip():')
          < body.index("_load_llm(model_name"), body[-800:])
    check("ensure_exclusive is no longer called inline from build_runtime",
          "ensure_exclusive" not in body)
    # The graph MUST still be built: _busy() returns True while graph is None,
    # so skipping it would make every message queue forever.
    check("the graph is still assembled", "build_graph(ctx)" in body)
    check("and still returned", "return ctx, base_state, assistant_graph" in body)


def test_the_gui_refuses_instead_of_queueing():
    src = _src("gui/gui.py")
    check("there is one place that answers 'is a model loaded'",
          "def _model_missing(self)" in src)
    check("and one place that reports it", "def _refuse_without_model(self)" in src)
    send = _func(src, "_send_text")
    check("the send path refuses BEFORE the busy check, so nothing is queued",
          send.index("_refuse_without_model()") < send.index("self._busy()"), send)
    dispatch = _func(src, "_dispatch_user_text")
    check("the dispatch path (queue drain, deep research, db scan) is gated too",
          "_refuse_without_model()" in dispatch, dispatch)
    worker = _func(src, "_start_worker")
    check("the worker path (voice, drop zone) is gated too",
          "_refuse_without_model()" in worker, worker)
    check("the notice names the way out (Settings)",
          "Settings" in src[src.index("NO_MODEL_NOTICE"):src.index("NO_MODEL_NOTICE") + 400])


def test_the_greeting_and_the_chip_tell_the_truth():
    src = _src("gui/gui.py")
    ready = _func(src, "_on_runtime_ready")
    check("the greeting branches on whether a model is loaded",
          "self._model_missing()" in ready, ready[-900:])
    check("a healthy start says nothing -- only problems are announced",
          "Ask me anything" not in ready, ready)
    chip = _func(src, "_refresh_chip")
    check("the chip has a name to show without a model",
          "no model" in chip, chip)


def test_startup_does_not_report_stopped_training():
    """The user asked for startup to carry only what is important; a stopped
    training run is on the Characters tab."""
    ready = _func(_src("gui/gui.py"), "_on_runtime_ready")
    check("no stopped-run notice at startup", "RUN_STOPPED" not in ready, ready[-1200:])


def test_the_queue_holds_rather_than_burning_through():
    q = _func(_src("gui/gui_queue.py"), "_maybe_drain_queue")
    check("the drain checks for a missing model", "_model_missing()" in q, q[-900:])
    check("and pauses instead of dispatching", "_queue_paused = True" in q)
    check("the hold is reported once, with the shared notice",
          "NO_MODEL_NOTICE" in q)


def test_unloading_from_settings_records_the_state():
    """Freeing the VRAM is only half of it.

    Picking "do not load a model" from inside a running app used to evict
    everything and leave ctx.model_name naming the model that was just thrown
    out -- so every gate above still believed a model was loaded and kept
    sending turns to an LM Studio with nothing resident."""
    body = _func(_src("gui/gui.py"), "_open_settings")
    i = body.index("NO_MODEL")
    branch = body[i:i + 1400]
    check("the context is told the model is gone",
          'self.ctx.model_name = ""' in branch, branch)
    check("the active-model id is cleared, so a reselect really reloads",
          '_active_model_id = ""' in branch, branch)
    check("the chip is refreshed", "_refresh_chip()" in branch, branch)
    check("the VRAM is still actually freed", "free_gpu" in branch)


def test_a_loaded_model_lifts_the_queue_hold():
    body = _func(_src("gui/gui.py"), "_on_model_switch_done")
    check("the hold is lifted on success", "_queue_paused = False" in body, body)
    check("only when something is actually waiting",
          'it.get("status") == "pending"' in body, body)
    check("and the queue is kicked", "_maybe_drain_queue()" in body, body)
    check("the rollback on FAILURE is untouched",
          "self.ctx.model_name = self._active_model_id" in body)


def test_telegram_users_are_told_too():
    strings = _src("bot/tg_strings.py")
    check("a localized string exists", '"no_model_loaded"' in strings)
    check("it is translated, not English-only",
          '"ru":' in strings[strings.index('"no_model_loaded"'):
                             strings.index('"no_model_loaded"') + 600])
    inner = _func(_src("bot/tg_tasks.py"), "_run_task_inner")
    check("a queued turn refuses instead of running an answerless graph",
          "no_model_loaded" in inner, inner[:900])
    check("it refuses before opening the status message",
          inner.index("no_model_loaded") < inner.index("_send_get_id"), inner[:900])
    chars = _func(_src("bot/tg_characters.py"), "_start_character_render")
    check("a character render refuses too -- its caption is planned by the LLM",
          "no_model_loaded" in chars, chars)


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

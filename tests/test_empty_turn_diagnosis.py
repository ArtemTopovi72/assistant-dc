"""A turn that produced nothing must say WHICH nothing it was.

One line covered four different situations and prescribed the same cure for
all of them -- "проверьте, что LM Studio запущен и модель загружена":

  * LM Studio genuinely not answering (the only case that message fits);
  * LM Studio up, but the chosen model evicted from under us -- restarting LM
    Studio does not fix it, reselecting the model does;
  * the model loaded with a context too small for the tool-bearing system
    prompt, which rejects every call with "n_keep >= n_ctx" and returns
    nothing at all -- restarting changes nothing, reloading with a bigger
    context does;
  * the model answering with an empty string, where retrying is the answer.

The method is pulled off the class and run against a stub, so no QApplication,
GPU or LM Studio is needed -- and nothing here touches the network: fetch_models
and loaded_context_length are replaced for the duration.

Run: venv/Scripts/python.exe tests/test_empty_turn_diagnosis.py
"""
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name
          + ("" if cond or not detail else "  -- " + str(detail)[:300]))
    if not cond and os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))


# Importing gui.py pulls in PyQt5 and the whole app; the method under test only
# uses self.ctx and module-level LM_STUDIO_BASE, so lift its source instead.
import ast
SRC = (ROOT / "gui/gui.py").read_text(encoding="utf-8")
_tree = ast.parse(SRC)
_seg = next(ast.get_source_segment(SRC, n) for n in ast.walk(_tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_empty_turn_diagnosis")

import lmstudio


def _diagnose(models, ctx_len, model_name="house-model"):
    """Run the real method body against a stubbed LM Studio."""
    ns = {"LM_STUDIO_BASE": "http://127.0.0.1:1234/v1",
          "logger": logging.getLogger("t")}
    exec("import textwrap\n" + _seg.replace("    def ", "def ", 1)
         .replace("\n    ", "\n"), ns)
    fn = ns["_empty_turn_diagnosis"]
    orig_fm, orig_cl = lmstudio.fetch_models, lmstudio.loaded_context_length
    lmstudio.fetch_models = lambda base: models
    lmstudio.loaded_context_length = lambda base, mid: ctx_len
    try:
        return fn(types.SimpleNamespace(
            ctx=types.SimpleNamespace(model_name=model_name)))
    finally:
        lmstudio.fetch_models, lmstudio.loaded_context_length = orig_fm, orig_cl


LOADED = [{"id": "house-model", "state": "loaded"}]


def test_lm_studio_down_is_named_as_such():
    msg = _diagnose([], 0)
    check("it says LM Studio is not answering", "not answering" in msg, msg)
    check("and names the address, so a wrong port is visible",
          "127.0.0.1:1234" in msg, msg)


def test_an_evicted_model_sends_the_user_to_settings_not_to_a_restart():
    msg = _diagnose([{"id": "something-else", "state": "loaded"}], 0)
    check("it names the model that is missing", "house-model" in msg, msg)
    check("it lists what IS loaded, which is usually the whole story",
          "something-else" in msg, msg)
    check("and points at Settings, not at restarting LM Studio",
          "Settings" in msg and "start it" not in msg.lower(), msg)


def test_nothing_loaded_at_all_reads_correctly():
    msg = _diagnose([{"id": "house-model", "state": "not-loaded"}], 0)
    check("an empty loaded list does not render as an empty parenthesis",
          "nothing is loaded" in msg, msg)


def test_a_starved_context_is_named_instead_of_blamed_on_lm_studio():
    msg = _diagnose(LOADED, 8192)
    check("the actual number is shown", "8192" in msg, msg)
    check("the cure is a bigger context, not a restart",
          "16384" in msg, msg)


def test_a_healthy_setup_says_the_model_simply_answered_nothing():
    msg = _diagnose(LOADED, 32768)
    check("no false alarm about LM Studio",
          "not answering" not in msg and "not loaded" not in msg, msg)
    check("it suggests retrying or clearing the context",
          "try again" in msg, msg)


def test_the_four_messages_are_actually_different():
    msgs = [_diagnose([], 0),
            _diagnose([{"id": "other", "state": "loaded"}], 0),
            _diagnose(LOADED, 8192),
            _diagnose(LOADED, 32768)]
    check("four situations, four messages", len(set(msgs)) == 4, msgs)


def test_a_broken_probe_falls_back_instead_of_raising():
    """This runs inside a Qt slot: an exception here is a native crash with no
    traceback, not an error message."""
    orig = lmstudio.fetch_models
    lmstudio.fetch_models = lambda base: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        msg = _diagnose(LOADED, 32768)
    finally:
        lmstudio.fetch_models = orig
    check("it returned a string rather than raising", isinstance(msg, str) and msg, msg)


def test_the_diagnosis_is_only_for_a_turn_that_produced_NOTHING():
    """A video- or document-only turn legitimately has no text: the artefact IS
    the answer. Diagnosing "модель не ответила" over it would be a lie about a
    turn that worked. The list used to name only image_status and
    research_report, so a delivered clip or file was reported as silence."""
    src = (ROOT / "gui/gui.py").read_text(encoding="utf-8")
    i = src.index("_empty_turn_diagnosis()")
    guard = src[max(0, i - 900):i]
    for key in ("image_status", "image_path", "research_report",
                "video_path", "document_path"):
        check("a turn that produced %s is not called silent" % key,
              key in guard, guard[-400:])


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

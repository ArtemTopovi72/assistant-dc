"""Regression: execute_tool must ALWAYS return a str, and the graph tool loop must
never crash on a rogue None/non-str tool result.

BUG (found by the randomized fault-injection campaign): a tool handler returning
None (e.g. a missing `return` on some code path) propagated up through
execute_tool and crashed the personality_node loop at `tool_result.lstrip()` with
`AttributeError: 'NoneType' object has no attribute 'lstrip'`, aborting the entire
graph.invoke — a hard worker crash in the GUI. execute_tool's docstring already
PROMISED it converts everything to a [TOOL ERROR] string "rather than
propagating"; the contract was simply not enforced for the return value.

Run: venv/Scripts/python.exe tests/test_tool_result_contract.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Hermetic: this suite is not a live-model test, but it was reaching
# localhost:1234 (see tests/offline_guard.py). Nothing here depends on
# the answers — the calls only made it slow and machine-dependent.
import offline_guard; offline_guard.offline_llm()
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)


def test_execute_tool_coerces_none_to_error():
    import tools as tools_mod
    from tools import ToolSpec
    # register a deliberately broken handler that returns None
    name = "_broken_none_tool"
    tools_mod._BY_NAME[name] = ToolSpec(
        name=name, schema={}, args_model=None,
        handler=lambda c, s, a: None)
    try:
        out = tools_mod.execute_tool(None, {}, name, {})
        assert isinstance(out, str), f"execute_tool returned {type(out)}, not str"
        assert out.startswith("[TOOL ERROR]"), f"None not coerced to tool error: {out!r}"
        # and a non-str, non-None return is stringified (never crashes callers)
        tools_mod._BY_NAME[name] = ToolSpec(
            name=name, schema={}, args_model=None,
            handler=lambda c, s, a: {"weird": 1})
        out2 = tools_mod.execute_tool(None, {}, name, {})
        assert isinstance(out2, str) and out2, "non-str return not stringified"
    finally:
        tools_mod._BY_NAME.pop(name, None)
    print("PASS execute_tool coerces None/non-str handler results to str")


def test_graph_loop_survives_none_tool_result():
    """Even if execute_tool were bypassed and yielded a non-str, the graph loop's
    own guard must keep .lstrip() from crashing. We simulate by monkeypatching the
    module-level execute_tool the graph calls to return None once."""
    import threading, copy
    from pathlib import Path
    import graph as graph_mod, tools as tools_mod
    from models import Context
    from prompts import SYSTEM_PROMPT_PERSONALITY
    from config import MODEL_NAME
    # stub image ops so no ComfyUI
    tools_mod.generate_image_with_refinement = lambda *a, **k: {
        "path": None, "score": 0, "attempts": 1, "status": "fail"}

    calls = {"n": 0}
    real = graph_mod.execute_tool
    def flaky(ctx, state, name, args):
        calls["n"] += 1
        if calls["n"] == 1:
            return None  # rogue result that used to crash .lstrip()
        return real(ctx, state, name, args)

    # Force at least one tool call by scripting the LLM response.
    import llm as llm_mod
    seq = [
        {"content": "", "tool_calls": [{"id": "x1", "type": "function",
            "function": {"name": "calculate", "arguments": '{"expression": "2+2"}'}}]},
        {"content": "Готово: 4.", "tool_calls": []},
    ]
    it = iter(seq)
    orig_send = graph_mod.send_to_lm_studio
    graph_mod.send_to_lm_studio = lambda *a, **k: next(it, {"content": "ok", "tool_calls": []})
    graph_mod.execute_tool = flaky
    try:
        c = Context(models=None, transcription_cache={}, cache_file=Path("tests/_c.json"),
                    asr_lock=threading.Lock(), tts_lock=threading.Lock(),
                    model_name=MODEL_NAME, no_think=True)
        c.tts_disabled = True; c.gui_mode = True
        c.active_memory_dir = Path(__file__).resolve().parent / "_cmem"
        c.active_memory_dir.mkdir(parents=True, exist_ok=True)
        g = graph_mod.build_graph(c)
        st = {"messages": [{"role": "system", "content": SYSTEM_PROMPT_PERSONALITY}],
              "user_input": "посчитай 2+2", "image_data": None, "final_answer": "",
              "session_memory_text": "", "vision_summary": "", "image_path": "",
              "image_score": 0, "image_attempt": 0, "image_status": ""}
        final = g.invoke(st)  # must NOT raise
        assert isinstance(final.get("final_answer", ""), str)
        print("PASS graph tool loop survives a None tool result (no AttributeError)")
    finally:
        graph_mod.execute_tool = real
        graph_mod.send_to_lm_studio = orig_send


if __name__ == "__main__":
    test_execute_tool_coerces_none_to_error()
    test_graph_loop_survives_none_tool_result()
    print("\ndone")

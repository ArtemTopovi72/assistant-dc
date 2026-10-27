"""Arithmetic the model must not be *asked* to delegate, but made to.

The calculate description already states the rule ("three or more digits, or a
percentage -> you MUST call this tool"). On the routing bench the model obeyed
it about half the time and computed `1234 * 5678` in its head instead, getting
it wrong. intent.must_call states the same rule where the model has no vote.

The interesting half of this suite is the negatives: forcing a tool call on a
turn that does not want one is worse than the defect being fixed.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import graph_personality as G

# Which messages must be calculated is the model's read (intent.must_call);
# the phrases, negatives included, run live in bench/intent_force_live.py.
_CALC = {"needs_tool": True, "wants": ["calculate"], "must_call": "calculate"}


def _read_of(text):
    return _CALC if "1234" in text else None


# --- the loop actually uses it ---------------------------------------------
# Not a source grep: this drives the real loop and reads the tool_choice that
# reached the LLM boundary.

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "_tgf", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "test_graph_full.py"))
_TGF = _ilu.module_from_spec(_spec)
try:
    _spec.loader.exec_module(_TGF)
except SystemExit:
    pass  # that module runs its own checks at import and exits 0


def _rounds_for(text, tool_result="7006652"):
    """Run one real turn and return (tool_choice, [tool names]) per LLM round."""
    llm = _TGF.LLM([
        _TGF._asst("", [_TGF._tc("calculate", {"expression": "1234*5678"}, "c1")]),
        _TGF._asst("Получилось 7006652."),
    ])
    tools = _TGF.Tools({"calculate": tool_result})
    ctx = _TGF._ctx()
    # through _run's own `read`: set on the context it was overwritten by _run's, and the suite
    # passed only when the live model translated the message (2026-10-20)
    _TGF._run(ctx, llm, tools, {"user_input": text, "messages": []}, read=_read_of(text),
              fastpath=False)
    return [(c["kw"].get("tool_choice"),
             [t.get("function", {}).get("name") for t in (c["kw"].get("tools") or [])])
            for c in llm.calls]


def test_the_loop_forces_calculate_on_hard_arithmetic():
    rounds = _rounds_for("сколько будет 1234 * 5678?")
    assert rounds, "the loop made no LLM call"
    choice, names = rounds[0]
    assert choice == "auto", rounds          # one schema + no-think prefill first
    assert names == ["calculate"], names


def test_forced_call_answered_in_text_is_retried_with_required():
    llm = _TGF.LLM([
        _TGF._asst("Сейчас посчитаю."),                                   # text, no call
        _TGF._asst("", [_TGF._tc("calculate", {"expression": "1234*5678"}, "c1")]),
        _TGF._asst("Получилось 7006652."),
    ])
    tools = _TGF.Tools({"calculate": "7006652"})
    ctx = _TGF._ctx()
    _TGF._run(ctx, llm, tools, {"user_input": "сколько будет 1234 * 5678?", "messages": []},
              read=_CALC, fastpath=False)
    choices = [c["kw"].get("tool_choice") for c in llm.calls]
    # The retry is "auto with an explicit order" first and `required` only if
    # that is ignored too (graph_personality). What matters: the text answer was
    # not accepted, and the calculator ran.
    assert len(choices) >= 2, choices                      # the text was not accepted
    assert [n for n, _ in tools.calls] == ["calculate"], tools.calls


def test_the_loop_releases_the_force_after_the_tool_ran():
    """Round two must be free, or the model can never state the result."""
    rounds = _rounds_for("сколько будет 1234 * 5678?")
    assert len(rounds) > 1 and rounds[1][0] == "auto", rounds
    assert len(rounds[1][1]) > 1, "the full tool set must come back"


def test_the_loop_does_not_force_on_an_ordinary_message():
    assert _rounds_for("расскажи анекдот")[0][0] == "auto"


def test_required_code_execution_cannot_finish_as_a_code_block(tmp_path):
    from types import SimpleNamespace
    from code_sandbox import Sandbox
    draft = "```python\nprint('AUDIT_HOLD_DONE')\n```"
    llm = _TGF.LLM([_TGF._asst(draft)] * 3)
    tools = _TGF.Tools()
    ctx = _TGF._ctx(sandbox=Sandbox(tmp_path),
                    sandbox_user=SimpleNamespace(prefs={"sandbox": "code"}))
    final = _TGF._run(ctx, llm, tools,
                      {"user_input": "Выполни Python-код в песочнице: print('AUDIT_HOLD_DONE')",
                       "messages": []},
                      read={"needs_tool": True, "wants": ["run_code"], "must_call": "run_code"},
                      fastpath=False)
    assert not tools.calls
    assert len(llm.calls) == 3
    assert "```" not in final["final_answer"]
    assert "не выполнен" in final["final_answer"].lower()

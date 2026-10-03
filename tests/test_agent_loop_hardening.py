"""Regression tests for personality_node loop hardening (found via live adversarial
fuzzing of the real graph with a scripted malicious LLM).

Covered:
  - tool-call ids emitted null/duplicate by the model are normalized to present+unique
    ON THE ASSISTANT MESSAGE, and every id is answered by exactly one tool response
    (else a strict OpenAI-compatible backend rejects the re-sent turn).
  - a single assistant message with a huge tool-call batch executes at most
    MAX_TOOL_CALLS_PER_ROUND tools but still answers every tool_call_id.
  - a model that NEVER emits plain text (always a tool call) still terminates,
    bounded by the round budget.

No LM Studio required: llm.send_to_lm_studio / graph.send_to_lm_studio are patched
with a scripted responder, so the REAL loop, dispatch, validation and guards run.
Run: venv/Scripts/python.exe tests/test_agent_loop_hardening.py
"""
import os, sys, json, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

from pathlib import Path
from models import Context
from prompts import SYSTEM_PROMPT_PERSONALITY
import graph as graph_mod
import llm as llm_mod
from config import MAX_TOOL_ROUNDS, RECOVERY_EXTRA_ROUNDS, MODEL_NAME
from graph import MAX_TOOL_CALLS_PER_ROUND
from graph_personality import RUNAWAY_TOOL_CALLS

MAX_ROUNDS = MAX_TOOL_ROUNDS + RECOVERY_EXTRA_ROUNDS
_script = {"fn": None, "n": 0}


def _send(ctx, messages, tools=None, **k):
    _script["n"] += 1
    return _script["fn"](_script["n"])


graph_mod.send_to_lm_studio = _send
llm_mod.send_to_lm_studio = _send


def _msg(content="", tcs=None):
    m = {"role": "assistant", "content": content}
    if tcs is not None:
        m["tool_calls"] = tcs
    return m


def _tc(name, args, cid):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def _ctx():
    c = Context(models=None, transcription_cache={}, cache_file=Path("tests/_h.json"),
                asr_lock=threading.Lock(), tts_lock=threading.Lock(),
                model_name=MODEL_NAME, no_think=True)
    c.tts_disabled = True
    c.gui_mode = True
    c.web_search_enabled = False
    c.active_memory_dir = Path(__file__).resolve().parent / "_hmem"
    c.active_memory_dir.mkdir(parents=True, exist_ok=True)
    return c


def _base_state():
    return {"messages": [{"role": "system", "content": SYSTEM_PROMPT_PERSONALITY}],
            "user_input": "", "image_data": None, "final_answer": "",
            "session_memory_text": "", "vision_summary": "",
            "image_path": "", "image_score": 0, "image_attempt": 0, "image_status": ""}


def _ids_match_responses(messages):
    """Every assistant tool_calls turn has present+unique ids, each answered by exactly
    one tool response in order."""
    for i, m in enumerate(messages):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            aids = [tc.get("id") for tc in m["tool_calls"]]
            if any(not a for a in aids):
                return f"null/empty id in assistant msg: {aids}"
            if len(set(aids)) != len(aids):
                return f"duplicate ids in assistant msg: {aids}"
            resp = [x for x in messages[i + 1:i + 1 + len(aids)] if x.get("role") == "tool"]
            tids = [x.get("tool_call_id") for x in resp]
            if tids != aids:
                return f"response ids {tids} != assistant ids {aids}"
    return None


import re as _re
_MATCH_EVERYTHING = _re.compile("")


def _run(fn, user="go", state_extra=None):
    """Drive the REAL loop with a scripted LLM.

    The short-message fast path answers in one lite call BEFORE the tool round
    loop, so it silently ate scripted response #1 and every assertion below slid
    by one — `test_null_and_duplicate_ids_normalized` was passing vacuously
    because with no tool-call turn in the transcript there is nothing to check.
    Neutralising _TOOL_TRIGGER_RE is the same switch production uses: anything
    tool-ish skips the fast path. These tests are ABOUT the loop, so they say so.
    """
    _script["fn"] = fn
    _script["n"] = 0
    c = _ctx()
    g = graph_mod.build_graph(c)
    st = _base_state()
    st["user_input"] = user
    if state_extra:
        st.update(state_extra)
    pass
    pass
    try:
        final = g.invoke(st)
    finally:
        pass
    return c, final


def _assert_tool_turn_present(messages, what):
    """Guard against a vacuous pass: the id checks only mean something once an
    assistant tool-call turn actually exists."""
    assert any(m.get("role") == "assistant" and m.get("tool_calls") for m in messages), \
        f"{what}: no assistant tool_calls turn in the transcript — nothing was tested"


def test_null_and_duplicate_ids_normalized():
    # model emits 4 tool calls ALL with id=None
    def fn(i):
        if i == 1:
            return _msg(tcs=[{"id": None, "type": "function",
                              "function": {"name": "calculate", "arguments": '{"expression":"1+1"}'}}
                             for _ in range(4)])
        return _msg("done")
    _, final = _run(fn)
    _assert_tool_turn_present(final["messages"], "null/duplicate ids")
    err = _ids_match_responses(final["messages"])
    assert err is None, f"id normalization failed: {err}"
    print("PASS null/duplicate ids normalized and matched")


def test_batch_cap_bounds_execution():
    calls = {"calc": 0}
    import tools as tools_mod
    spec = tools_mod._BY_NAME["calculate"]
    orig = spec.handler
    def counting(c, s, a):
        calls["calc"] += 1
        return orig(c, s, a)
    object.__setattr__(spec, "handler", counting)
    try:
        def fn(i):
            if i == 1:
                return _msg(tcs=[_tc("calculate", {"expression": f"{k}+{k}"}, f"s{k}")
                                 for k in range(RUNAWAY_TOOL_CALLS)])
            return _msg("done")
        _, final = _run(fn)
    finally:
        object.__setattr__(spec, "handler", orig)
    assert calls["calc"] <= MAX_TOOL_CALLS_PER_ROUND, \
        f"executed {calls['calc']} > cap {MAX_TOOL_CALLS_PER_ROUND}"
    # but ALL 40 ids must still be answered (API validity)
    _assert_tool_turn_present(final["messages"], "batch cap")
    first_assistant = next(m for m in final["messages"]
                           if m.get("role") == "assistant" and m.get("tool_calls"))
    n = len(first_assistant["tool_calls"])
    idx = final["messages"].index(first_assistant)
    answered = [m for m in final["messages"][idx + 1:idx + 1 + n] if m.get("role") == "tool"]
    assert len(answered) == n, f"only {len(answered)}/{n} tool ids answered"
    err = _ids_match_responses(final["messages"])
    assert err is None, err
    print(f"PASS batch cap: executed {calls['calc']}<= {MAX_TOOL_CALLS_PER_ROUND}, all {n} answered")


def test_runaway_round_runs_nothing():
    """Live 2026-09-18 09:02: an outpaint's [done] was answered with 83
    redraw_image calls; the first was RUN (a second expansion on top of the
    first) and the refusals behind it counted as consecutive failures. A round
    that big is a runaway: none of it runs, the model answers with what it has."""
    calls = {"calc": 0}
    import tools as tools_mod
    spec = tools_mod._BY_NAME["calculate"]
    orig = spec.handler
    def counting(c, s, a):
        calls["calc"] += 1
        return orig(c, s, a)
    object.__setattr__(spec, "handler", counting)
    try:
        def fn(i):
            if i == 1:
                return _msg(tcs=[_tc("calculate", {"expression": "2+2"}, "ok")])
            if i == 2:
                return _msg(tcs=[_tc("calculate", {"expression": f"{k}+{k}"}, f"r{k}")
                                 for k in range(RUNAWAY_TOOL_CALLS + 1)])
            return _msg("Готово: 4.")
        _, final = _run(fn)
    finally:
        object.__setattr__(spec, "handler", orig)
    assert calls["calc"] == 1, f"the runaway round executed {calls['calc'] - 1} call(s)"
    storms = [m for m in final["messages"]
              if m.get("role") == "assistant" and len(m.get("tool_calls") or []) > RUNAWAY_TOOL_CALLS]
    assert not storms, "the runaway round stayed in history"
    assert "4" in (final.get("final_answer") or ""), final.get("final_answer")
    err = _ids_match_responses(final["messages"])
    assert err is None, err
    print("PASS runaway round: nothing run, retracted, answered")


def test_never_terminates_still_terminates():
    def fn(i):
        return _msg(tcs=[_tc("calculate", {"expression": f"{i}+{i}"}, f"n{i}")])
    _, final = _run(fn)
    _assert_tool_turn_present(final["messages"], "never-terminating loop")
    assert _script["n"] <= MAX_ROUNDS + 2, f"unbounded: {_script['n']} llm calls"
    assert isinstance(final.get("final_answer", ""), str)
    err = _ids_match_responses(final["messages"])
    assert err is None, err
    print(f"PASS never-terminating tool loop bounded at {_script['n']} llm calls")


def _ensure_probe() -> str:
    """The looks need a real picture on disk; runtime/ is not in git, so a
    fresh checkout (or a cleaned runtime) made both look tests fail with 0 calls."""
    p = Path(__file__).resolve().parent.parent / "runtime" / "probe.jpg"
    if not p.exists():
        from PIL import Image
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), (128, 128, 128)).save(p)
    return str(p)


def test_repeated_identical_look_is_not_re_executed():
    """Live 2026-09-18 00:46-00:48 (chat 100000001): the model asked for the
    SAME inspect_image check on the SAME unchanged picture 9+ times per round
    across several rounds -- inspect_image always "succeeds" (a critique is
    never a [TOOL ERROR]), so the identical-failure guard never saw it, and
    each repeat spent a real vision round trip on an answer already given.
    The second and later identical calls must be answered from cache, not
    re-executed."""
    calls = {"look": 0}
    import tools as tools_mod
    spec = tools_mod._BY_NAME["inspect_image"]
    orig = spec.handler
    def counting(c, s, a):
        calls["look"] += 1
        return "Cylinder: PARTIAL — a capsule, not a cylinder."
    object.__setattr__(spec, "handler", counting)
    _probe = _ensure_probe()
    try:
        def fn(i):
            args = {"check": "Is it a clean cylinder?", "image_path": "input_file_0.png"}
            if i == 1:
                return _msg(tcs=[_tc("inspect_image", args, "a"),
                                 _tc("inspect_image", args, "b"),
                                 _tc("inspect_image", args, "c")])
            if i == 2:
                return _msg(tcs=[_tc("inspect_image", args, "d")])
            return _msg("Готово.")
        _, final = _run(fn, state_extra={"image_path": _probe})
    finally:
        object.__setattr__(spec, "handler", orig)
    assert calls["look"] == 1, f"executed {calls['look']} real looks, wanted 1"
    err = _ids_match_responses(final["messages"])
    assert err is None, err
    print("PASS repeated identical look: executed once, cached the rest")


def test_readonly_cache_clears_after_an_edit():
    """The cache must not survive a real edit: a look before and after a
    redraw_image is a look at two DIFFERENT pictures and both must run."""
    calls = {"look": 0}
    import tools as tools_mod
    look_spec = tools_mod._BY_NAME["inspect_image"]
    redraw_spec = tools_mod._BY_NAME["redraw_image"]
    orig_look, orig_redraw = look_spec.handler, redraw_spec.handler
    def counting(c, s, a):
        calls["look"] += 1
        return "Looks fine."
    object.__setattr__(look_spec, "handler", counting)
    object.__setattr__(redraw_spec, "handler", lambda c, s, a: "Image redrawn and saved: x.png")
    _probe = _ensure_probe()
    try:
        args = {"check": "Is it fixed?", "image_path": "input_file_0.png"}
        def fn(i):
            if i == 1:
                return _msg(tcs=[_tc("inspect_image", args, "a")])
            if i == 2:
                return _msg(tcs=[_tc("redraw_image", {"mode": "redraw"}, "b")])
            if i == 3:
                return _msg(tcs=[_tc("inspect_image", args, "c")])
            return _msg("Готово.")
        _run(fn, state_extra={"image_path": _probe})
    finally:
        object.__setattr__(look_spec, "handler", orig_look)
        object.__setattr__(redraw_spec, "handler", orig_redraw)
    assert calls["look"] == 2, f"executed {calls['look']} looks across the edit, wanted 2"
    print("PASS readonly cache invalidated by a real edit")


if __name__ == "__main__":
    test_null_and_duplicate_ids_normalized()
    test_batch_cap_bounds_execution()
    test_runaway_round_runs_nothing()
    test_never_terminates_still_terminates()
    test_repeated_identical_look_is_not_re_executed()
    test_readonly_cache_clears_after_an_edit()
    print("\n6/6 passed")

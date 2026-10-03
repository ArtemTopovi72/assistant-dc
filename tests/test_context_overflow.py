"""Recovering from a prompt that does not fit the model's context.

Observed in the chaos bench log: a long tool chain hit "Context size has been
exceeded.", the round came back None, and the turn was lost. Three separate
defects lined up behind it:

  * the rejection was recognised only in its llama.cpp phrasing
    ("n_keep: 9001 >= n_ctx: 8192"); the newer, numberless "Context size has
    been exceeded." matched nothing, so the shortfall parsed as 0, the
    self-heal declined to act and the hint logged nothing at all;
  * the caller could not tell a context rejection from any other empty result;
  * so it re-sent the SAME oversized payload three times and gave up.
"""
import sys, os, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm


# --- recognising the rejection ---------------------------------------------

def test_both_phrasings_are_recognised():
    assert llm._looks_like_context_overflow("Context size has been exceeded.")
    assert llm._looks_like_context_overflow("n_keep: 9001 >= n_ctx: 8192")
    assert llm._looks_like_context_overflow("The prompt is too long")

def test_unrelated_errors_are_not_mistaken_for_it():
    """Shrinking the prompt in response to an unrelated failure would silently
    throw away the conversation for no reason."""
    for body in ["model overloaded", "rate limit exceeded", "connection reset",
                 "500 Internal Server Error", ""]:
        assert not llm._looks_like_context_overflow(body), body


# --- making the prompt fit --------------------------------------------------

def _payload(msgs):
    return {"model": "m", "messages": copy.deepcopy(msgs)}

BIG = "x" * 5000

def _chain():
    """A realistic long turn: an older exchange, then a three-call tool chain.

    Sized deliberately -- a two-call fixture cannot exercise this at all, since
    the most recent result is always spared and the other one is a single step.
    """
    return [
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "первый вопрос"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]},
        {"role": "tool", "tool_call_id": "a", "content": BIG},
        {"role": "assistant", "content": "ответ на первый"},
        {"role": "user", "content": "второй вопрос"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "b"}]},
        {"role": "tool", "tool_call_id": "b", "content": BIG},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c"}]},
        {"role": "tool", "tool_call_id": "c", "content": BIG},
    ]


def test_first_pass_truncates_the_oldest_tool_result():
    p = _payload(_chain())
    assert llm._shrink_payload(p) is True
    m = p["messages"]
    assert len(m) == 10, "pass one must not drop anything"
    assert len(m[3]["content"]) < len(BIG), "the oldest result goes first"
    assert m[3]["content"].endswith(llm._TRUNC_NOTE)

def test_it_shrinks_one_step_at_a_time():
    """Each retry should give up as little as possible."""
    p = _payload(_chain())
    llm._shrink_payload(p)
    rest = [m for m in p["messages"] if m.get("role") == "tool"][1:]
    assert all(len(m["content"]) == len(BIG) for m in rest)

def test_the_recent_messages_are_never_touched():
    p = _payload(_chain())
    for _ in range(6):
        if not llm._shrink_payload(p):
            break
    m = p["messages"]
    assert m[0]["role"] == "system", "the persona must survive"
    assert m[-1]["content"].endswith(llm._TRUNC_NOTE) or m[-1]["content"] == BIG,         "the newest tool result must survive"
    assert any(x.get("content") == "второй вопрос" for x in m),         "the live question must survive"

def test_a_dropped_assistant_turn_takes_its_tool_replies_with_it():
    """A tool result with nothing it answers is a 400, not a repair."""
    p = _payload(_chain())
    for _ in range(8):
        if not llm._shrink_payload(p):
            break
        ids = {m.get("tool_calls", [{}])[0].get("id")
               for m in p["messages"] if m.get("tool_calls")}
        for m in p["messages"]:
            if m.get("role") == "tool":
                assert m["tool_call_id"] in ids, \
                    f"orphaned tool result {m['tool_call_id']}"

def test_it_reports_when_there_is_nothing_left_to_give():
    """The caller must be able to stop instead of looping forever."""
    p = _payload([{"role": "system", "content": "p"},
                  {"role": "user", "content": "q"}])
    assert llm._shrink_payload(p) is False

def test_shrinking_terminates():
    p = _payload(_chain())
    for _ in range(50):
        if not llm._shrink_payload(p):
            return
    raise AssertionError("_shrink_payload never reported exhaustion")


# --- the retry loop actually uses it ---------------------------------------

def test_the_retry_loop_shrinks_instead_of_resending(monkeypatch):
    """The measured failure: three identical rejections, then a lost turn."""
    seen = []

    def fake_stream(ctx, payload):
        seen.append(sum(len(str(m.get("content") or ""))
                        for m in payload["messages"]))
        if len(seen) < 3:
            return llm._CTX_OVERFLOW
        return {"role": "assistant", "content": "готово"}

    monkeypatch.setattr(llm, "_stream_chat", fake_stream)
    monkeypatch.setattr(llm, "throttle_external_calls", lambda ctx: None)

    class Ctx:
        def is_cancelled(self): return False
        def set_stage(self, *a, **k): pass

    out = llm.send_to_lm_studio(Ctx(), _chain(), tools=None)
    assert out and out.get("content") == "готово", "the turn must not be lost"
    assert len(seen) == 3
    assert seen[1] < seen[0], f"payload was re-sent unshrunk: {seen}"
    assert seen[2] < seen[1], f"payload stopped shrinking: {seen}"


def test_an_unshrinkable_prompt_gives_up_instead_of_looping(monkeypatch):
    monkeypatch.setattr(llm, "_stream_chat",
                        lambda ctx, payload: llm._CTX_OVERFLOW)
    monkeypatch.setattr(llm, "throttle_external_calls", lambda ctx: None)

    class Ctx:
        def is_cancelled(self): return False
        def set_stage(self, *a, **k): pass

    assert llm.send_to_lm_studio(
        Ctx(), [{"role": "system", "content": "p"},
                {"role": "user", "content": "q"}], tools=None) is None


# --- _stream_chat must RAISE the signal, not just log it -------------------
# Missing these let a mutation that returns plain None from the HTTP-error path
# pass the whole suite: the retry-loop tests above patch _stream_chat wholesale
# and never exercise its own return value.

class _Resp:
    def __init__(self, status=400, body="", lines=()):
        self.status_code = status; self.text = body; self._lines = lines
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def iter_lines(self, decode_unicode=False):
        for l in self._lines: yield l
    def close(self): pass


def test_an_http_context_rejection_returns_the_overflow_signal(monkeypatch):
    monkeypatch.setattr(llm.requests, "post",
                        lambda *a, **k: _Resp(400, "Context size has been exceeded."))
    assert llm._stream_chat(None, {"messages": [], "model": "m"}) is llm._CTX_OVERFLOW


def test_an_unrelated_http_error_still_returns_none(monkeypatch):
    """Only a context rejection may trigger trimming. A bare 5xx is the
    _SERVER_ERROR sentinel (not None, not the context-overflow signal)."""
    monkeypatch.setattr(llm.requests, "post",
                        lambda *a, **k: _Resp(500, "Internal Server Error"))
    out = llm._stream_chat(None, {"messages": [], "model": "m"})
    assert out is llm._SERVER_ERROR and out is not llm._CTX_OVERFLOW


def test_an_in_stream_context_error_returns_the_overflow_signal(monkeypatch):
    body = b'data: {"error": {"message": "Context size has been exceeded."}}'
    monkeypatch.setattr(llm.requests, "post",
                        lambda *a, **k: _Resp(200, "", [body]))
    assert llm._stream_chat(None, {"messages": [], "model": "m"}) is llm._CTX_OVERFLOW


def test_the_numberless_rejection_still_names_a_size(caplog):
    """The hint exists to tell the user which number to raise. On the phrasing
    that carries no numbers it logged nothing at all."""
    import logging
    payload = {"model": "gemma", "messages": [{"role": "user", "content": "x" * 3000}],
               "tools": [{"function": {"name": "a", "description": "d" * 900}}] * 20}
    with caplog.at_level(logging.ERROR):
        llm._log_context_hint("Context size has been exceeded.", payload)
    assert caplog.text, "the numberless rejection logged nothing"
    assert "tool schema" in caplog.text and "gemma" in caplog.text


def test_an_unrelated_error_still_logs_no_hint(caplog):
    import logging
    with caplog.at_level(logging.ERROR):
        llm._log_context_hint("rate limit exceeded", {"model": "m", "messages": []})
    assert not caplog.text

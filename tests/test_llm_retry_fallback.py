"""Regression + fault-injection for the LLM transport layer (llm.py).

The retry/fallback logic and the streaming assembler are reachable with NO live
LM Studio: we monkeypatch `_stream_chat` (for the retry loop) and `requests.post`
(for the SSE assembler) with fakes that reproduce every failure mode:

  * transient None message -> retry -> exhaustion returns None
  * exception on every attempt -> exhaustion returns None
  * cancellation sentinel mid-stream -> returns None immediately
  * cancellation flag between attempts -> stops retrying, returns None
  * empty content recovered from reasoning_content (Qwen) but NOT for gpt-oss
  * <think></think> prefill appended as an assistant turn (Qwen only)
  * gpt-oss path sets reasoning_effort and skips prefill/no_think
  * HTTP non-200 -> None
  * [DONE] terminates; tool_calls accumulate across delta chunks by index
  * wall-clock cap and char cap abort but KEEP partial output
  * mid-stream Timeout with partial output is KEPT; with nothing is re-raised

Run: venv/Scripts/python.exe tests/test_llm_retry_fallback.py
"""
import os, sys, json, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
import requests
import llm as L

_REAL_STREAM = L._stream_chat   # capture before the retry tests monkeypatch it
_REAL_SEND = L.send_to_lm_studio
import requests as _rq
_REAL_POST = _rq.post


class Ctx:
    def __init__(self, model_name="qwen3.5-9b", no_think=False, cancel=False):
        self.model_name = model_name
        self.no_think = no_think
        self.reasoning_effort = "high"
        self._cancel = cancel
    def is_cancelled(self):
        return self._cancel


def _no_sleep(monkey=True):
    L.time.sleep = lambda *_a, **_k: None
    L.throttle_external_calls = lambda *_a, **_k: None


# ---------------------------------------------------------------- retry loop

def test_none_message_exhausts_to_none():
    _no_sleep()
    calls = {"n": 0}
    def fake_stream(ctx, payload):
        calls["n"] += 1
        return None
    L._stream_chat = fake_stream
    out = L.send_to_lm_studio(Ctx(), [{"role": "user", "content": "hi"}])
    assert out is None, f"expected None, got {out!r}"
    assert calls["n"] == L.LLM_MAX_RETRIES, f"retry count {calls['n']} != {L.LLM_MAX_RETRIES}"
    print("PASS transient None message retries then returns None")


def test_exception_every_attempt_returns_none():
    _no_sleep()
    calls = {"n": 0}
    def boom(ctx, payload):
        calls["n"] += 1
        raise RuntimeError("connection reset")
    L._stream_chat = boom
    out = L.send_to_lm_studio(Ctx(), [{"role": "user", "content": "hi"}])
    assert out is None and calls["n"] == L.LLM_MAX_RETRIES
    print("PASS exception on every attempt returns None after full retries")


def test_cancel_sentinel_returns_none_no_retry():
    _no_sleep()
    calls = {"n": 0}
    def cancelled(ctx, payload):
        calls["n"] += 1
        return L._CANCELLED
    L._stream_chat = cancelled
    out = L.send_to_lm_studio(Ctx(), [{"role": "user", "content": "hi"}])
    assert out is None and calls["n"] == 1, f"cancel should not retry (n={calls['n']})"
    print("PASS _CANCELLED sentinel returns None without retrying")


def test_cancel_flag_between_attempts_stops_retry():
    _no_sleep()
    calls = {"n": 0}
    ctx = Ctx()
    def fail_then_cancel(c, payload):
        calls["n"] += 1
        ctx._cancel = True          # user cancels after the 1st failed attempt
        raise RuntimeError("net down")
    L._stream_chat = fail_then_cancel
    out = L.send_to_lm_studio(ctx, [{"role": "user", "content": "hi"}])
    assert out is None and calls["n"] == 1, f"cancel flag must stop retry (n={calls['n']})"
    print("PASS cancellation flag between attempts stops the retry loop")


def test_empty_content_recovered_from_reasoning_qwen():
    _no_sleep()
    L._stream_chat = lambda c, p: {"role": "assistant", "content": "",
                                   "reasoning_content": "the answer is 42"}
    out = L.send_to_lm_studio(Ctx(model_name="qwen3.5-9b"), [{"role": "user", "content": "x"}])
    assert out and out["content"] == "the answer is 42", f"reasoning recovery failed: {out!r}"
    print("PASS empty content recovered from reasoning_content (Qwen)")


def test_tool_markup_in_reasoning_is_NOT_executed():
    """A thought about calling a tool is not a call.

    The reasoning-recovery branch above hands `reasoning_content` on as the
    answer; the textual tool-call parser used to run over that text too. So a
    model deliberating "I should call:search{...}" — inside the channel it is
    supposed to think in — had that thought executed as a real search. Qwen's
    own parser refuses to read tool tags before </think> for this exact reason.
    """
    _no_sleep()
    L._stream_chat = lambda c, p: {
        "role": "assistant", "content": "",
        "reasoning_content": 'Maybe I should call:search{query:<|"|>weather<|"|>} '
                             'but the user already told me, so no.'}
    out = L.send_to_lm_studio(Ctx(model_name="qwen3.5-9b"), [{"role": "user", "content": "x"}])
    assert out is not None, "message dropped entirely"
    assert not out.get("tool_calls"), \
        f"a tool call was fabricated from the reasoning channel: {out.get('tool_calls')!r}"
    print("PASS tool markup inside reasoning_content is not executed")


def test_tool_markup_in_real_content_still_executes():
    """The guard must not disarm the fallback on its actual job: the same markup
    arriving in `content` is a real call and still has to run."""
    _no_sleep()
    L._stream_chat = lambda c, p: {
        "role": "assistant",
        "content": 'call:search{query:<|"|>weather<|"|>}',
        "reasoning_content": "some thinking"}
    out = L.send_to_lm_studio(Ctx(model_name="qwen3.5-9b"), [{"role": "user", "content": "x"}])
    assert out and out.get("tool_calls"), f"real textual tool call was lost: {out!r}"
    assert out["tool_calls"][0]["function"]["name"] == "search", out["tool_calls"]
    print("PASS textual tool call in content still executes")


def test_empty_content_NOT_recovered_for_gpt_oss():
    _no_sleep()
    L._stream_chat = lambda c, p: {"role": "assistant", "content": "",
                                   "reasoning_content": "internal harmony analysis"}
    out = L.send_to_lm_studio(Ctx(model_name="gpt-oss-20b"), [{"role": "user", "content": "x"}])
    assert out is not None and out["content"] == "", \
        f"gpt-oss must NOT leak reasoning_content as answer: {out!r}"
    print("PASS gpt-oss does NOT leak reasoning_content as the answer")


def test_prefill_appended_for_qwen_not_oss():
    _no_sleep()
    seen = {}
    def cap(ctx, payload):
        seen["msgs"] = payload["messages"]
        seen["payload"] = payload
        return {"role": "assistant", "content": "ok"}
    L._stream_chat = cap
    L.send_to_lm_studio(Ctx(model_name="qwen3.5-9b"), [{"role": "user", "content": "x"}],
                        prefill="<think></think>")
    assert seen["msgs"][-1] == {"role": "assistant", "content": "<think></think>"}, \
        "prefill not appended for Qwen"
    assert seen["payload"].get("reasoning") == "off"
    # gpt-oss must ignore prefill and set reasoning_effort
    seen.clear()
    L.send_to_lm_studio(Ctx(model_name="gpt-oss-20b"), [{"role": "user", "content": "x"}],
                        prefill="<think></think>", reasoning_effort="low")
    assert seen["msgs"][-1]["role"] == "user", "gpt-oss must not append <think> prefill"
    assert seen["payload"]["reasoning_effort"] == "low"
    print("PASS prefill appended for Qwen, ignored for gpt-oss; reasoning_effort set")


def test_no_think_marker_applied():
    _no_sleep()
    seen = {}
    L._stream_chat = lambda c, p: (seen.setdefault("msgs", p["messages"]),
                                   {"role": "assistant", "content": "ok"})[1]
    L.send_to_lm_studio(Ctx(model_name="qwen3.5-9b", no_think=True),
                        [{"role": "system", "content": "You are X."},
                         {"role": "user", "content": "hi"}])
    sysmsg = seen["msgs"][0]["content"]
    assert "/no_think" in sysmsg, f"no_think marker not applied: {sysmsg!r}"
    print("PASS /no_think marker injected into system turn")


# ---------------------------------------------------------------- SSE assembler

class FakeResp:
    def __init__(self, lines, status=200):
        self._lines = lines
        self.status_code = status
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def iter_lines(self, decode_unicode=False):
        for ln in self._lines:
            yield ln
    def close(self): pass


def _post_factory(lines, status=200, exc_after=None, exc=None):
    state = {"i": 0}
    def fake_post(url, **kw):
        return FakeResp(lines, status)
    return fake_post


def _sse(obj):
    return ("data: " + json.dumps(obj)).encode("utf-8")


def test_stream_http_error_returns_none():
    requests.post = lambda url, **kw: FakeResp([], status=503)
    msg = _REAL_STREAM(Ctx(), {"messages": []})
    # 2026-09-18: a 5xx is the _SERVER_ERROR sentinel so the retry loop can
    # mend the payload before trying again; a 4xx stays a plain None.
    assert msg is L._SERVER_ERROR, f"HTTP 503 must return _SERVER_ERROR, got {msg!r}"
    requests.post = lambda url, **kw: FakeResp([], status=400)
    assert _REAL_STREAM(Ctx(), {"messages": []}) is None, "HTTP 400 must return None"
    print("PASS _stream_chat returns None on non-200")


def test_stream_assembles_content_and_tool_calls():
    lines = [
        _sse({"choices": [{"delta": {"content": "Hel"}}]}),
        _sse({"choices": [{"delta": {"content": "lo"}}]}),
        _sse({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "function": {"name": "calc", "arguments": "{\"a\":"}}]}}]}),
        _sse({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "1}"}}]}}]}),
        b": keep-alive",
        b"data: [DONE]",
    ]
    requests.post = lambda url, **kw: FakeResp(lines)
    msg = _REAL_STREAM(Ctx(), {"messages": []})
    assert msg["content"] == "Hello", f"content assembly wrong: {msg!r}"
    assert msg["tool_calls"][0]["id"] == "call_1"
    assert msg["tool_calls"][0]["function"]["name"] == "calc"
    assert msg["tool_calls"][0]["function"]["arguments"] == '{"a":1}', "args not concatenated"
    print("PASS SSE content + multi-chunk tool_calls assembled correctly")


def test_stream_char_cap_keeps_partial():
    old = L.LLM_STREAM_MAX_CHARS
    L.LLM_STREAM_MAX_CHARS = 5
    try:
        lines = [_sse({"choices": [{"delta": {"content": "abcdefghij"}}]}),
                 _sse({"choices": [{"delta": {"content": "MORE"}}]})]
        requests.post = lambda url, **kw: FakeResp(lines)
        msg = _REAL_STREAM(Ctx(), {"messages": []})
        assert msg["content"] == "abcdefghij", f"char-cap must keep first chunk: {msg!r}"
    finally:
        L.LLM_STREAM_MAX_CHARS = old
    print("PASS char cap aborts but keeps partial output")


def test_stream_cancel_mid_stream_returns_sentinel():
    ctx = Ctx()
    lines = [_sse({"choices": [{"delta": {"content": "x"}}]}) for _ in range(5)]
    ctx._cancel = True
    requests.post = lambda url, **kw: FakeResp(lines)
    msg = _REAL_STREAM(ctx, {"messages": []})
    assert msg is L._CANCELLED, f"cancel mid-stream must return sentinel, got {msg!r}"
    print("PASS cancellation mid-stream returns _CANCELLED")


def test_stream_timeout_with_partial_is_kept():
    class TimeoutResp(FakeResp):
        def iter_lines(self, decode_unicode=False):
            yield _sse({"choices": [{"delta": {"content": "partial"}}]})
            raise requests.exceptions.Timeout("read timed out")
    requests.post = lambda url, **kw: TimeoutResp([])
    msg = _REAL_STREAM(Ctx(), {"messages": []})
    assert msg is not None and msg["content"] == "partial", \
        f"partial output must be kept on mid-stream timeout: {msg!r}"
    print("PASS mid-stream timeout with partial output keeps the partial")


def test_stream_timeout_with_nothing_reraises():
    class DeadResp(FakeResp):
        def iter_lines(self, decode_unicode=False):
            raise requests.exceptions.ConnectionError("dropped")
            yield  # pragma: no cover
    requests.post = lambda url, **kw: DeadResp([])
    raised = False
    try:
        _REAL_STREAM(Ctx(), {"messages": []})
    except requests.exceptions.ConnectionError:
        raised = True
    assert raised, "a drop with NO partial output must re-raise for the retry loop"
    print("PASS mid-stream drop with nothing buffered re-raises to the retry loop")


def test_payload_tools_and_force_think():
    _no_sleep()
    seen = {}
    def cap(ctx, payload):
        seen["p"] = payload
        return {"role": "assistant", "content": "ok"}
    L._stream_chat = cap
    tools = [{"type": "function", "function": {"name": "calc", "parameters": {}}}]
    L.send_to_lm_studio(Ctx(model_name="qwen3.5-9b"), [{"role": "user", "content": "x"}],
                        tools=tools, tool_choice="required")
    assert seen["p"]["tools"] == tools and seen["p"]["tool_choice"] == "required"
    # reasoning is off centrally: force_think does not turn it back on
    seen.clear()
    L.send_to_lm_studio(Ctx(model_name="qwen3.5-9b", no_think=True),
                        [{"role": "user", "content": "x"}], force_think=True)
    assert seen["p"].get("reasoning") == "off", seen["p"]
    assert seen["p"].get("chat_template_kwargs") != {"enable_thinking": True}
    print("PASS tools/tool_choice attached; force_think cannot re-enable reasoning")


def test_stream_wall_clock_cap_keeps_partial(monkeypatch=None):
    # Force the wall-clock deadline to trip immediately after the first chunk.
    real_mono = L.time.monotonic
    seq = iter([0.0, 0.0, 10_000.0, 10_000.0, 10_000.0])
    L.time.monotonic = lambda: next(seq, 10_000.0)
    try:
        lines = [_sse({"choices": [{"delta": {"content": "kept"}}]}),
                 _sse({"choices": [{"delta": {"content": "dropped"}}]})]
        requests.post = lambda url, **kw: FakeResp(lines)
        msg = _REAL_STREAM(Ctx(), {"messages": []})
    finally:
        L.time.monotonic = real_mono
    assert msg["content"] == "kept", f"wall-clock cap must keep the pre-deadline chunk: {msg!r}"
    print("PASS wall-clock cap aborts but keeps partial output")


def test_call_llm_simple_none_and_content():
    _no_sleep()
    L.send_to_lm_studio = lambda *a, **k: None
    assert L.call_llm_simple(Ctx(), "sys", "hi") is None
    L.send_to_lm_studio = lambda *a, **k: {"role": "assistant", "content": "answer"}
    assert L.call_llm_simple(Ctx(), "sys", "hi", history=[{"role": "user", "content": "q"}]) == "answer"
    # empty content coerces to None (falsy -> None)
    L.send_to_lm_studio = lambda *a, **k: {"role": "assistant", "content": ""}
    assert L.call_llm_simple(Ctx(), "sys", "hi") is None
    print("PASS call_llm_simple maps None/empty->None and returns content")


def test_analyze_image_retries_once_on_empty():
    _no_sleep()
    # RESTORE it afterwards. Leaving the stub in place meant the very next test
    # -- the regression guard for BUG #12, "a missing image must not reach the
    # model" -- was handed a data URL for a file that does not exist, so the
    # guard it exists to protect was never exercised and the test failed.
    _orig_durl = L.file_to_data_url
    L.file_to_data_url = lambda p: "data:image/png;base64,AAAA"
    calls = {"n": 0}
    def flaky(ctx, messages, **k):
        calls["n"] += 1
        return {"content": ""} if calls["n"] == 1 else {"content": "a dog"}
    L.send_to_lm_studio = flaky
    try:
        out = L.analyze_image_with_llm(Ctx(), image_path="x.png", user_text="what?")
        assert out == "a dog" and calls["n"] == 2, f"empty-retry failed: {out!r} n={calls['n']}"
        # no image at all -> None, no call
        assert L.analyze_image_with_llm(Ctx()) is None
    finally:
        L.file_to_data_url = _orig_durl
    print("PASS analyze_image_with_llm retries once on empty, None with no image")


def test_analyze_image_missing_path_returns_none_no_crash():
    """BUG #12 (found by live integration): a missing/unreadable image path made
    analyze_image_with_llm raise FileNotFoundError from file_to_data_url instead of
    degrading. Two hot call sites (tools.py inspect-image QA, image.py evaluate_image)
    do NOT wrap the call, so a stale/deleted temp render crashed the whole turn."""
    _no_sleep()
    called = {"n": 0}
    def must_not_reach(*a, **k):
        called["n"] += 1
        return {"content": "should never get here"}
    L.send_to_lm_studio = must_not_reach
    out = L.analyze_image_with_llm(Ctx(), image_path="C:/no/such/file_zzz.png", user_text="x")
    assert out is None, f"missing path must return None, got {out!r}"
    assert called["n"] == 0, "must not even reach the LLM with an unreadable image"
    # unreadable bytes path (a directory-as-bytes can't happen, but a broken encoder can);
    # simulate by making the data-url builder raise
    import utils
    orig = utils.file_to_data_url
    utils.file_to_data_url = lambda p: (_ for _ in ()).throw(OSError("permission denied"))
    L.file_to_data_url = utils.file_to_data_url
    try:
        out2 = L.analyze_image_with_llm(Ctx(), image_path="x.png", user_text="y")
        assert out2 is None, f"OSError on read must return None, got {out2!r}"
    finally:
        utils.file_to_data_url = orig
        L.file_to_data_url = orig
    print("PASS analyze_image_with_llm returns None (no crash) on missing/unreadable image (BUG #12)")


def test_analyze_image_persistent_empty_returns_none():
    _no_sleep()
    L.image_bytes_to_data_url = lambda b, mime_type="image/jpeg": "data:image/jpeg;base64,AAAA"
    L.send_to_lm_studio = lambda *a, **k: {"content": ""}
    out = L.analyze_image_with_llm(Ctx(), image_bytes=b"\x89PNG", user_text="")
    # It returns "" (not None) here — unlike call_llm_simple which coerces to None.
    # VERIFIED non-defect: all 6 call sites (graph.py:236, tools.py:502, image.py
    # 1449/1656/3631/5013) test the result with truthiness (`if resp`, `or ""`,
    # `if not resp`), never an `is None` identity check, so "" and None are
    # behaviorally identical. Assert falsy, the actual contract callers rely on.
    assert not out, f"persistent empty must be falsy, got {out!r}"
    print("PASS analyze_image_with_llm persistent empty is falsy (callers use truthiness)")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        # restore module globals so tests can't leak monkeypatches into each other
        L._stream_chat = _REAL_STREAM
        L.send_to_lm_studio = _REAL_SEND
        requests.post = _REAL_POST
        fn()
    print(f"\ndone ({len(fns)} tests)")

"""Branch-closure tests for llm.py — targets the exact lines/branches the main
suite (test_llm_retry_fallback.py) leaves uncovered so llm.py reaches 100%.

Missing before this file (coverage branch=True):
  42->41, 44->46, 47-48   _apply_no_think: non-system msg / already-marked / no-system
  182-183                 _is_cancelled: ctx.is_cancelled() raises -> False
  189, 192                _decode_line: None -> "" ; str passthrough
  228->313                _stream_chat: empty stream exits loop normally
  244                     blank raw bytes -> continue
  248                     whitespace-only line -> continue
  254->257, 263-264       non-data / non-JSON line -> parse error -> continue
  273-274, 318            reasoning_content delta accumulated + attached
  300->285                two tool_calls in one delta -> loop back

Run: venv/Scripts/python.exe tests/test_llm_branch_closure.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
import requests
import llm as L

_REAL_POST = requests.post


class Ctx:
    def __init__(self, cancel=False, raise_cancel=False):
        self.model_name = "qwen3.5-9b"
        self.no_think = False
        self.reasoning_effort = "high"
        self._cancel = cancel
        self._raise = raise_cancel
    def is_cancelled(self):
        if self._raise:
            raise RuntimeError("cancel probe blew up")
        return self._cancel


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


def _sse(obj):
    return ("data: " + json.dumps(obj)).encode("utf-8")


# ---------------------------------------------------------- _apply_no_think

def test_apply_no_think_no_system_message_prepends():
    # 47-48: no system turn present -> a minimal system message is prepended.
    out = L._apply_no_think([{"role": "user", "content": "hi"}])
    assert out[0] == {"role": "system", "content": L._NO_THINK_MARKER}
    assert out[1]["role"] == "user"
    print("PASS _apply_no_think prepends system marker when none exists")


def test_apply_no_think_skips_non_system_then_marks():
    # 42->41: iterate past a non-system (user) message before finding system.
    out = L._apply_no_think([{"role": "user", "content": "q"},
                             {"role": "system", "content": "You are X."}])
    assert out[0]["role"] == "user"
    assert out[1]["content"].endswith(L._NO_THINK_MARKER)
    print("PASS _apply_no_think skips non-system messages then marks the system turn")


def test_apply_no_think_already_marked_and_nonstr_content_noop():
    # 44->46: system content already has the marker OR is non-str -> left untouched.
    already = [{"role": "system", "content": "sys " + L._NO_THINK_MARKER}]
    out = L._apply_no_think(already)
    assert out[0]["content"].count(L._NO_THINK_MARKER) == 1, "marker must not double-append"
    nonstr = [{"role": "system", "content": [{"type": "text", "text": "hi"}]}]
    out2 = L._apply_no_think(nonstr)
    assert out2[0]["content"] == [{"type": "text", "text": "hi"}], "non-str content untouched"
    print("PASS _apply_no_think no-ops on already-marked / non-str system content")


# ---------------------------------------------------------- _is_cancelled

def test_is_cancelled_swallows_exception():
    # 182-183: a ctx.is_cancelled that raises must be treated as "not cancelled".
    assert L._is_cancelled(Ctx(raise_cancel=True)) is False
    # also the no-callback case
    class Bare: pass
    assert L._is_cancelled(Bare()) is False
    print("PASS _is_cancelled returns False when the probe raises or is absent")


# ---------------------------------------------------------- _decode_line

def test_decode_line_none_and_str_and_bytes():
    assert L._decode_line(None) == ""            # 189
    assert L._decode_line("already text") == "already text"   # 192
    assert L._decode_line(b"\xe2\x9c\x93") == "✓"        # bytes UTF-8
    assert L._decode_line(b"\xff\xfe") == "��"      # invalid -> replaced
    print("PASS _decode_line handles None / str / valid+invalid bytes")


# ---------------------------------------------------------- _stream_chat body

def test_stream_empty_stream_exits_loop_normally():
    # 228->313: iterator yields nothing -> loop body never runs, falls to message build.
    requests.post = lambda url, **kw: FakeResp([])
    msg = L._stream_chat(Ctx(), {"messages": []})
    # Anchored to the CONTRACT, not the exact key set: the message also carries
    # finish_reason (and may gain other diagnostics), which says nothing about
    # whether an empty stream produced an empty assistant message.
    assert msg["role"] == "assistant" and msg["content"] == "" \
        and not msg.get("tool_calls"), f"empty stream -> empty msg: {msg!r}"
    print("PASS empty stream exits the read loop and returns an empty assistant message")


def test_stream_skips_blank_whitespace_noise_lines():
    # 244 (falsy raw), 248 (whitespace-only), 254->257 + 263-264 (non-data / bad json).
    lines = [
        b"",                                   # 244: falsy raw -> continue
        b"   ",                                # 248: strips to "" -> continue
        b": keep-alive",                       # SSE comment -> continue
        b"event: ping",                        # 254->257 non-data, 257 not [DONE],
                                               #   263-264 json.loads fails -> continue
        b"data: not-json-at-all",              # data: prefix but bad JSON -> continue
        _sse({"choices": [{"delta": {"content": "ok"}}]}),
        b"data: [DONE]",
    ]
    requests.post = lambda url, **kw: FakeResp(lines)
    msg = L._stream_chat(Ctx(), {"messages": []})
    assert msg["content"] == "ok", f"noise must be skipped, content kept: {msg!r}"
    assert "tool_calls" not in msg
    print("PASS blank / whitespace / comment / non-data / bad-JSON lines are all skipped")


def test_stream_reasoning_content_accumulated_and_attached():
    # 273-274 (accumulate reasoning delta) + 318 (attach reasoning_content to message).
    lines = [
        _sse({"choices": [{"delta": {"reasoning_content": "think "}}]}),
        _sse({"choices": [{"delta": {"reasoning_content": "harder"}}]}),
        _sse({"choices": [{"delta": {"content": "answer"}}]}),
        b"data: [DONE]",
    ]
    requests.post = lambda url, **kw: FakeResp(lines)
    msg = L._stream_chat(Ctx(), {"messages": []})
    assert msg["content"] == "answer"
    assert msg["reasoning_content"] == "think harder", f"reasoning not assembled: {msg!r}"
    print("PASS reasoning_content deltas accumulate and attach to the message")


def test_stream_two_tool_calls_in_one_delta_loop_back():
    # 300->285: a single delta carrying TWO tool_calls forces the for-loop to iterate.
    lines = [
        _sse({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c0", "function": {"name": "a", "arguments": "{}"}},
            {"index": 1, "id": "c1", "function": {"name": "b", "arguments": "{}"}},
        ]}}]}),
        b"data: [DONE]",
    ]
    requests.post = lambda url, **kw: FakeResp(lines)
    msg = L._stream_chat(Ctx(), {"messages": []})
    names = [tc["function"]["name"] for tc in msg["tool_calls"]]
    assert names == ["a", "b"], f"both tool_calls in one delta must be captured: {msg!r}"
    print("PASS two tool_calls in a single delta iterate the accumulation loop")


def test_stream_tool_call_without_name_or_args_slot_created_only():
    # tcd.get('id') falsy AND fn empty -> slot created, name/args stay empty (298/300 false).
    lines = [
        _sse({"choices": [{"delta": {"tool_calls": [{"index": 0}]}}]}),
        b"data: [DONE]",
    ]
    requests.post = lambda url, **kw: FakeResp(lines)
    msg = L._stream_chat(Ctx(), {"messages": []})
    assert msg["tool_calls"][0]["function"] == {"name": "", "arguments": ""}
    assert msg["tool_calls"][0]["id"] is None
    print("PASS tool_call delta with no id/name/args creates an empty slot")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        requests.post = _REAL_POST
        fn()
    print(f"\ndone ({len(fns)} tests)")

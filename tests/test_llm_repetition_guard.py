"""Regression: the streaming repetition guard must watch BOTH channels.

BUG: llm._stream_chat counts content AND reasoning into `streamed_chars` (so the
periodic checkpoint fires on either), but built the sampled tail from
`content_parts` alone. A model looping inside its REASONING channel therefore hit
the checkpoint over and over against an EMPTY tail, `_looks_degenerate` returned
"not looping" every time, and the stream ran on to the last-resort caps
(LLM_STREAM_MAX_CHARS = 200k chars / LLM_STREAM_MAX_SECONDS = 600s).

That is the house model's channel. Gemma 4 spends most of its budget thinking —
it is why send_to_lm_studio floors max_tokens to GEMMA_MIN_TOKENS — so the guard
whose stated purpose is "bail as soon as the output is provably looping, which
costs seconds instead of minutes" was blind exactly where it was needed.

Measured with the identical 47-char loop before the fix:
    content channel    -> aborted after   50 chunks /   2,400 chars
    reasoning channel  -> ran to the end, 3000 chunks / 144,000 chars

Offline and deterministic: requests.post is replaced with a fake SSE response.
No LM Studio, no GPU, no network.
Run: venv/Scripts/python.exe tests/test_llm_repetition_guard.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import llm

RESULTS = []
import os as _os
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + detail))
    if not cond and _os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))


N_CHUNKS = 3000
LOOP = "I should consider the user's request carefully. "


class _FakeResp:
    """A 200 OK SSE stream of `n` deltas on `channel`, counting what was read."""
    status_code = 200

    def __init__(self, channel, chunks, counter):
        self._channel = channel
        self._chunks = chunks
        self._counter = counter

    def __enter__(self): return self
    def __exit__(self, *a): return False
    def close(self): pass

    def iter_lines(self, **kw):
        for text in self._chunks:
            self._counter["n"] += 1
            yield ("data: " + json.dumps(
                {"choices": [{"delta": {self._channel: text}}]})).encode()
        yield b"data: [DONE]"


def _stream(channel, chunks):
    """Run _stream_chat against a fake stream; return (chunks_read, message)."""
    counter = {"n": 0}
    orig = llm.requests.post
    llm.requests.post = lambda *a, **k: _FakeResp(channel, chunks, counter)
    try:
        msg = llm._stream_chat(None, {"messages": []})
    finally:
        llm.requests.post = orig
    return counter["n"], msg


def _kept(msg):
    return len(msg.get("content", "") or "") + len(msg.get("reasoning_content", "") or "")


def test_content_channel_loop_is_cut():
    read, msg = _stream("content", [LOOP] * N_CHUNKS)
    check("content_loop_aborts_early", read < N_CHUNKS, f"read {read}/{N_CHUNKS}")
    check("content_loop_keeps_partial_output", _kept(msg) > 0, str(_kept(msg)))


def test_reasoning_channel_loop_is_cut():
    # THE REGRESSION. Before the fix this read all 3000 chunks.
    read, msg = _stream("reasoning_content", [LOOP] * N_CHUNKS)
    check("reasoning_loop_aborts_early", read < N_CHUNKS, f"read {read}/{N_CHUNKS}")
    check("reasoning_loop_keeps_partial_output",
          (msg.get("reasoning_content") or "") != "", "reasoning was discarded")


def test_both_channels_are_cut_at_the_same_point():
    # The reasoning path must be no more permissive than the content path — the
    # whole bug was that one channel got a 60x longer leash than the other.
    read_c, _ = _stream("content", [LOOP] * N_CHUNKS)
    read_r, _ = _stream("reasoning_content", [LOOP] * N_CHUNKS)
    check("reasoning_leash_matches_content_leash", read_r == read_c,
          f"content={read_c} reasoning={read_r}")


# Healthy-stream cases are kept well under LLM_STREAM_MAX_CHARS (200k) so that
# what they measure is the REPETITION guard, not the unrelated total-character
# cap — at N_CHUNKS of ~100-char prose the char cap fires first and the test
# would "fail" on behaviour that has nothing to do with repetition.
N_HEALTHY = 600


def test_healthy_reasoning_is_not_aborted():
    # False-positive guard: long, VARIED reasoning prose must stream to the end.
    # A guard that trips on ordinary thinking would truncate every Gemma turn.
    varied = ["Considering step %d, the relevant factor is %s which suggests a "
              "different approach than step %d did. " % (i, "abcdefghij"[i % 10], i - 1)
              for i in range(N_HEALTHY)]
    read, msg = _stream("reasoning_content", varied)
    check("varied_reasoning_streams_to_completion", read == N_HEALTHY,
          f"read {read}/{N_HEALTHY}")
    check("varied_reasoning_fully_preserved",
          len(msg.get("reasoning_content", "")) == sum(len(v) for v in varied))


def test_healthy_content_is_not_aborted():
    varied = ["Paragraph %d explains %s in its own terms, without repeating the "
              "previous one. " % (i, "abcdefghij"[i % 10]) for i in range(N_HEALTHY)]
    read, msg = _stream("content", varied)
    check("varied_content_streams_to_completion", read == N_HEALTHY,
          f"read {read}/{N_HEALTHY}")


def test_clean_content_does_not_reset_a_reasoning_loop():
    # Per-channel signatures: the two-strike rule must be tracked per channel, or
    # a stream that emits a little clean content alongside looping reasoning
    # would keep clearing the strike and never abort.
    orig = llm.requests.post
    counter = {"n": 0}

    class _Mixed(_FakeResp):
        def iter_lines(self, **kw):
            for i in range(N_CHUNKS):
                counter["n"] += 1
                delta = {"reasoning_content": LOOP}
                if i % 5 == 0:
                    delta["content"] = "Distinct sentence number %d here. " % i
                yield ("data: " + json.dumps({"choices": [{"delta": delta}]})).encode()
            yield b"data: [DONE]"

    llm.requests.post = lambda *a, **k: _Mixed("content", [], counter)
    try:
        llm._stream_chat(None, {"messages": []})
    finally:
        llm.requests.post = orig
    check("mixed_stream_still_aborts_on_reasoning_loop", counter["n"] < N_CHUNKS,
          f"read {counter['n']}/{N_CHUNKS}")


def test_guard_can_be_disabled():
    # LLM_REPEAT_CHECK_EVERY=0 documents "turns the guard off entirely".
    orig = llm.LLM_REPEAT_CHECK_EVERY
    llm.LLM_REPEAT_CHECK_EVERY = 0
    try:
        read, _ = _stream("reasoning_content", [LOOP] * 300)
    finally:
        llm.LLM_REPEAT_CHECK_EVERY = orig
    check("check_every_zero_disables_the_guard", read == 300, f"read {read}/300")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback; traceback.print_exc()
            print("  ERROR in " + fn.__name__ + ": " + type(e).__name__ + ": " + str(e))
    passed = sum(1 for _, c in RESULTS if c)
    print(f"\n{len(fns)-failed}/{len(fns)} functions, {passed}/{len(RESULTS)} checks passed")
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)

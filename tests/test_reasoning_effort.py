"""Offline unit tests for gpt-oss reasoning-effort wiring (no LM Studio needed).

gpt-oss models reason via a `reasoning_effort` LEVEL (low/medium/high), NOT the
Qwen `/no_think` + `reasoning:"off"` + enable_thinking switch. These tests pin the
payload that send_to_lm_studio builds for each model family by stubbing the network
layer, so a refactor can't silently send the wrong reasoning control.
"""
import sys
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import threading
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import llm  # noqa: E402


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_is_gpt_oss_detection():
    for name in ("openai/gpt-oss-20b", "gpt-oss-120b", "lmstudio-community/gpt-oss-20b-MXFP4",
                 "GPT-120", "gptoss-20b"):
        _check(llm._is_gpt_oss(name), f"should detect gpt-oss: {name}")
    for name in ("HauhauCS/Qwen9B-NO-THINK", "qwen3-9b", "llama-3.1-8b", "", None):
        _check(not llm._is_gpt_oss(name), f"should NOT detect gpt-oss: {name}")
    print("PASS: _is_gpt_oss detection (gpt-oss variants vs others)")
    return True


def test_reasoning_effort_coercion():
    _check(llm._reasoning_effort(SimpleNamespace(reasoning_effort="low")) == "low", "low lost")
    _check(llm._reasoning_effort(SimpleNamespace(reasoning_effort="HIGH")) == "high", "case not normalized")
    _check(llm._reasoning_effort(SimpleNamespace(reasoning_effort="ultra")) == "high", "invalid not defaulted")
    _check(llm._reasoning_effort(SimpleNamespace()) == "high", "missing attr not defaulted")
    print("PASS: _reasoning_effort coercion (valid/case/invalid/missing → high default)")
    return True


def _capture_payload(model_name, no_think, effort):
    cap = {}

    def fake_stream(ctx, payload):
        cap.update(payload)
        return {"content": "ok"}

    saved_stream, saved_throttle = llm._stream_chat, llm.throttle_external_calls
    llm._stream_chat = fake_stream
    llm.throttle_external_calls = lambda ctx: None
    try:
        ctx = SimpleNamespace(model_name=model_name, no_think=no_think,
                              reasoning_effort=effort, is_cancelled=lambda: False,
                              cancel_event=None, api_lock=threading.Lock(),
                              last_api_call_time=0.0, api_min_interval=0.0)
        llm.send_to_lm_studio(ctx, [{"role": "user", "content": "hi"}])
    finally:
        llm._stream_chat, llm.throttle_external_calls = saved_stream, saved_throttle
    return cap


def test_gpt_oss_payload_sends_reasoning_effort():
    # gpt-oss cannot switch reasoning off: every call gets the lowest level.
    p = _capture_payload("openai/gpt-oss-20b", no_think=True, effort="high")
    _check(p.get("reasoning_effort") == "low", f"reasoning_effort not low: {p.get('reasoning_effort')}")
    # The Qwen-specific controls must NOT be sent for gpt-oss (they fight harmony).
    _check("reasoning" not in p, "stale reasoning:'off' sent to gpt-oss")
    _check("chat_template_kwargs" not in p, "enable_thinking sent to gpt-oss")
    # effort flows through from ctx
    p_low = _capture_payload("gpt-oss-120b", no_think=False, effort="low")
    _check(p_low.get("reasoning_effort") == "low", "effort=low not honored")
    print("PASS: gpt-oss payload carries reasoning_effort, drops Qwen reasoning controls")
    return True


def _run_with_message(model_name, msg, prefill=None, reasoning_effort=None):
    """Drive send_to_lm_studio with a stubbed stream returning `msg`; capture both
    the outbound payload and the returned (post-processed) message."""
    cap = {}

    def fake_stream(ctx, payload):
        cap["payload"] = payload
        return dict(msg)

    saved_stream, saved_throttle = llm._stream_chat, llm.throttle_external_calls
    llm._stream_chat = fake_stream
    llm.throttle_external_calls = lambda ctx: None
    try:
        ctx = SimpleNamespace(model_name=model_name, no_think=False,
                              reasoning_effort="high", is_cancelled=lambda: False,
                              cancel_event=None, api_lock=threading.Lock(),
                              last_api_call_time=0.0, api_min_interval=0.0)
        out = llm.send_to_lm_studio(ctx, [{"role": "user", "content": "hi"}],
                                    prefill=prefill, reasoning_effort=reasoning_effort)
    finally:
        llm._stream_chat, llm.throttle_external_calls = saved_stream, saved_throttle
    return out, cap.get("payload", {})


def test_gpt_oss_never_dumps_reasoning_as_content():
    """THE awful-report bug: gpt-oss burned its budget on the analysis channel and
    returned empty content; the old fallback dumped reasoning_content (raw chain of
    thought) as the answer. For gpt-oss, empty content must STAY empty; only non-oss
    (Qwen finetunes that inline the answer into reasoning_content) may recover it."""
    leaked = {"role": "assistant", "content": "",
              "reasoning_content": "Thus we should produce four blocks... (raw CoT)"}
    out_oss, _ = _run_with_message("openai/gpt-oss-120b", leaked)
    _check(out_oss["content"] == "", f"gpt-oss leaked CoT as content: {out_oss['content']!r}")
    out_qwen, _ = _run_with_message("HauhauCS/Qwen9B-NO-THINK", leaked)
    _check(out_qwen["content"].startswith("Thus we should"),
           "Qwen empty-content recovery from reasoning_content regressed")
    print("PASS: gpt-oss never emits reasoning_content as the answer (Qwen recovery intact)")
    return True


def test_gpt_oss_skips_think_prefill_and_honors_effort_override():
    """For gpt-oss the <think></think> prefill is a Qwen-ism (junk in harmony) and must
    NOT be appended as an assistant turn; and a per-call reasoning_effort override
    (e.g. 'low' for synthesis) must win over ctx's default 'high'."""
    msg = {"role": "assistant", "content": "ok"}
    _, payload = _run_with_message("openai/gpt-oss-120b", msg,
                                   prefill="<think></think>", reasoning_effort="low")
    _check(payload["messages"][-1]["content"] != "<think></think>",
           "gpt-oss got a literal <think></think> assistant prefill")
    _check(payload.get("reasoning_effort") == "low", "per-call effort override not honored")
    # Qwen still gets the prefill turn.
    _, payload_q = _run_with_message("HauhauCS/Qwen9B-NO-THINK", msg, prefill="<think></think>")
    _check(payload_q["messages"][-1]["content"] == "<think></think>",
           "Qwen lost its <think></think> prefill lever")
    print("PASS: gpt-oss skips <think> prefill + honors per-call effort; Qwen prefill intact")
    return True


def test_non_gpt_oss_payload_unchanged():
    p = _capture_payload("HauhauCS/Qwen9B-NO-THINK", no_think=True, effort="high")
    _check(p.get("reasoning") == "off", "Qwen no longer gets reasoning:'off'")
    _check(p.get("chat_template_kwargs") == {"enable_thinking": False}, "Qwen lost enable_thinking")
    _check("reasoning_effort" not in p, "reasoning_effort leaked to a non-gpt-oss model")
    print("PASS: non-gpt-oss payload still uses reasoning:'off' + enable_thinking, no effort leak")
    return True


if __name__ == "__main__":
    tests = [
        test_is_gpt_oss_detection,
        test_reasoning_effort_coercion,
        test_gpt_oss_payload_sends_reasoning_effort,
        test_gpt_oss_never_dumps_reasoning_as_content,
        test_gpt_oss_skips_think_prefill_and_honors_effort_override,
        test_non_gpt_oss_payload_unchanged,
    ]
    results = []
    for t in tests:
        try:
            results.append(bool(t()))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"FAIL: {t.__name__}: {e}")
            results.append(False)
    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)

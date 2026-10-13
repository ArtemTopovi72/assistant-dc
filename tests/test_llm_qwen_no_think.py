"""Qwen fine-tunes that ignore /no_think get their closed think block as a prefill
(Qwen3.6-35B-A3B HauhauCS spent the intent read's 300 tokens thinking)."""
import os
import sys
import threading
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "core", "agent"):
    sys.path.insert(0, os.path.join(ROOT, d))
os.environ.setdefault("F5_TEST_RUN", "1")

import llm  # noqa: E402


class Ctx:
    def __init__(self, model, stream=None):
        self.model_name, self.no_think = model, True
        self.api_lock, self.last_api_call_time, self.api_min_interval = threading.Lock(), 0.0, 0.0
        self.deps = types.SimpleNamespace(_stream_chat=stream, _wait_for_free_card=lambda ctx: None)


def _payload(model, **kw):
    seen = {}

    def fake(ctx, payload):
        seen.update(payload)
        return {"role": "assistant", "content": "ok"}
    llm.send_to_lm_studio(Ctx(model, fake), [{"role": "user", "content": "привет"}], **kw)
    return seen


def test_qwen_gets_its_closed_think_block():
    p = _payload("qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive")
    assert p["messages"][-1] == {"role": "assistant", "content": llm.QWEN_NO_THINK_PREFILL}


def test_forced_tool_call_on_qwen_has_no_prefill():
    tools = [{"type": "function", "function": {"name": "calculate", "parameters": {"type": "object"}}}]
    p = _payload("qwen3.6-35b-a3b", tools=tools, tool_choice="required")
    assert p["messages"][-1]["role"] == "user"


def test_gemma_keeps_its_own_prefill():
    p = _payload("gemma4-26b-a4b-uncensored-hauhaucs-balanced")
    assert p["messages"][-1]["content"] == llm.GEMMA_NO_THINK_PREFILL


def test_qwen_gets_no_system_turn_after_the_first():
    seen = {}
    msgs = [{"role": "system", "content": "you are"}, {"role": "user", "content": "hi"},
            {"role": "system", "content": "the tool failed"}]
    llm.send_to_lm_studio(Ctx("qwen3.6-35b-a3b", lambda ctx, p: seen.update(p) or {"role": "assistant", "content": "ok"}), msgs)
    roles = [m["role"] for m in seen["messages"]]
    assert roles[0] == "system" and "system" not in roles[1:]
    assert "the tool failed" in seen["messages"][2]["content"]

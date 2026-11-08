"""Native reasoning controls and grammar-safe requests for Qwen and Gemma."""
import os
import sys
import threading
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "core", "agent"):
    sys.path.insert(0, os.path.join(ROOT, d))
os.environ.setdefault("F5_TEST_RUN", "1")

import llm  # noqa: E402


def test_cancelled_turn_does_not_wait_for_render_or_call_model(monkeypatch):
    import comfy_client
    cancelled = threading.Event()
    waits = []
    monkeypatch.setattr(comfy_client, 'this_thread_holds_card', lambda: False)
    monkeypatch.setattr(comfy_client, 'card_is_exclusive', lambda: 'video')
    def wait(timeout):
        waits.append(timeout)
        cancelled.set()
        return False
    monkeypatch.setattr(comfy_client, 'wait_for_card', wait)
    ctx = Ctx('qwen3.8', lambda *args: (_ for _ in ()).throw(AssertionError('model called')))
    ctx.deps._wait_for_free_card = llm._wait_for_free_card
    ctx.is_cancelled = cancelled.is_set
    assert llm.send_to_lm_studio(ctx, [{'role': 'user', 'content': 'test'}]) is None
    assert waits and waits[0] <= 1


def test_batch_classifier_uses_its_turn_cancellation(monkeypatch):
    import intent
    monkeypatch.setattr(llm, '_wait_for_free_card', lambda ctx: None)
    monkeypatch.setattr(llm, '_stream_chat', lambda *args: {'role': 'assistant', 'content': 'unexpected'})
    with intent.turn_scope() as turn:
        turn.cancel_context = types.SimpleNamespace(is_cancelled=lambda: True)
        assert llm.send_to_lm_studio(None, [{'role': 'user', 'content': 'test'}]) is None


class Ctx:
    def __init__(self, model, stream=None):
        self.model_name, self.no_think = model, False
        self.api_lock, self.last_api_call_time, self.api_min_interval = threading.Lock(), 0.0, 0.0
        self.deps = types.SimpleNamespace(_stream_chat=stream, _wait_for_free_card=lambda ctx: None)


def _payload(model, **kw):
    seen = {}

    def fake(ctx, payload):
        seen.update(payload)
        result = {"role": "assistant", "content": "ok"}
        if payload.get('tool_choice') == 'required':
            result['tool_calls'] = [{'function': {'name': 'calculate', 'arguments': '{}'}}]
        return result
    llm.send_to_lm_studio(Ctx(model, fake), [{"role": "user", "content": "привет"}], **kw)
    return seen


def test_qwen_native_reasoning_is_enabled():
    p = _payload("qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive")
    assert p["messages"][-1]["role"] == "user"
    assert p.get('chat_template_kwargs', {}).get('enable_thinking') is True
    assert p.get('reasoning') != 'off'


def test_qwen_thinking_off_prefills_only_unconstrained_text():
    seen = []
    ctx = Ctx('qwen-aggressive', lambda ctx, payload: seen.append(payload) or
              {'role': 'assistant', 'content': 'done'})
    ctx.no_think = True
    llm.send_to_lm_studio(ctx, [{'role': 'user', 'content': 'translate'}], max_tokens=200)
    assert seen[-1]['messages'][-1] == {'role': 'assistant', 'content': llm.QWEN_NO_THINK_PREFILL}
    schema = {'type': 'object', 'properties': {'answer': {'type': 'string'}}}
    llm.send_to_lm_studio(ctx, [{'role': 'user', 'content': 'classify'}], json_schema=schema)
    assert seen[-1]['messages'][-1]['role'] == 'user'


def test_qwen_alias_uses_reported_architecture(monkeypatch):
    monkeypatch.setattr(llm, '_model_architecture', lambda name: 'qwen35', raising=False)
    p = _payload('hauhaucs-aggressive@?')
    assert p['messages'][-1]['role'] == 'user'
    assert p.get('chat_template_kwargs', {}).get('enable_thinking') is True


def test_reasoning_prefill_is_never_combined_with_grammar():
    schema = {'type': 'object', 'properties': {'answer': {'type': 'integer'}}}
    p = _payload('qwen3.8', prefill='<think></think>', json_schema=schema)
    assert p['messages'][-1]['role'] == 'user'
    assert p['response_format']['type'] == 'json_schema'


def test_native_reasoning_has_room_and_is_not_returned_as_answer():
    p = _payload('qwen3.8', max_tokens=80)
    assert p['max_tokens'] >= 1104
    ctx = Ctx('qwen3.8', lambda ctx, payload: {
        'role': 'assistant', 'content': '', 'reasoning_content': 'private deliberation'})
    result = llm.send_to_lm_studio(ctx, [{'role': 'user', 'content': 'test'}])
    assert result['content'] == ''
    assert result['reasoning_only'] is True


def test_gemma_tool_grammar_gets_reasoning_headroom_when_switch_off():
    seen = {}
    def fake(ctx, payload):
        seen.update(payload)
        return {'role': 'assistant', 'content': '',
                'tool_calls': [{'function': {'name': 'calculate', 'arguments': '{}'}}]}
    ctx = Ctx('gemma4-26b', fake)
    ctx.no_think = True
    tool = {'type': 'function', 'function': {'name': 'calculate', 'parameters': {'type': 'object'}}}
    llm.send_to_lm_studio(ctx, [{'role': 'user', 'content': 'test'}],
                          max_tokens=80, tools=[tool], tool_choice='required')
    assert seen['messages'][-1]['role'] == 'user'
    assert seen['max_tokens'] >= 1104


def test_thought_followed_by_tool_call_is_not_marked_incomplete():
    ctx = Ctx('qwen3.8', lambda ctx, payload: {
        'role': 'assistant', 'content': '<think>' + 'x' * 250 + '</think>',
        'tool_calls': [{'function': {'name': 'calculate', 'arguments': '{}'}}]})
    result = llm.send_to_lm_studio(ctx, [{'role': 'user', 'content': 'test'}])
    assert result['tool_calls'] and not result.get('reasoning_only')


def test_required_tool_never_returns_unexecuted_reasoning_as_answer():
    ctx = Ctx('qwen3.8', lambda ctx, payload: {
        'role': 'assistant', 'content': 'Repeated thought</think>not a tool', 'finish_reason': 'length'})
    tools = [{'type': 'function', 'function': {'name': 'calculate', 'parameters': {'type': 'object'}}}]
    result = llm.send_to_lm_studio(ctx, [{'role': 'user', 'content': 'test'}], tools=tools, tool_choice='required')
    assert result is None


def test_architecture_lookup_uses_loaded_alias_and_caches(monkeypatch):
    calls = []
    response = types.SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
        'models': [{'key': 'source-model', 'architecture': 'qwen35',
                    'loaded_instances': [{'id': 'alias'}]}]})
    monkeypatch.setattr(llm.requests, 'get', lambda *a, **k: calls.append(a) or response)
    monkeypatch.setattr(llm, '_MODEL_ARCH_CACHE', {})
    assert llm._is_qwen('alias') and llm._is_qwen('alias')
    assert len(calls) == 1


def test_architecture_lookup_failure_does_not_guess(monkeypatch):
    def offline(*args, **kwargs):
        raise llm.requests.ConnectionError('offline')
    monkeypatch.setattr(llm.requests, 'get', offline)
    monkeypatch.setattr(llm, '_MODEL_ARCH_CACHE', {})
    assert not llm._is_qwen('unknown-alias')


def test_forced_tool_call_on_qwen_has_no_prefill():
    tools = [{"type": "function", "function": {"name": "calculate", "parameters": {"type": "object"}}}]
    p = _payload("qwen3.6-35b-a3b", tools=tools, tool_choice="required")
    assert p["messages"][-1]["role"] == "user"


def test_gemma_native_reasoning_has_no_prefill():
    p = _payload("gemma4-26b-a4b-uncensored-hauhaucs-balanced")
    assert p["messages"][-1]["role"] == "user"
    assert 'reasoning' not in p and 'chat_template_kwargs' not in p


def test_qwen_gets_no_system_turn_after_the_first():
    seen = {}
    msgs = [{"role": "system", "content": "you are"}, {"role": "user", "content": "hi"},
            {"role": "system", "content": "the tool failed"}]
    llm.send_to_lm_studio(Ctx("qwen3.6-35b-a3b", lambda ctx, p: seen.update(p) or {"role": "assistant", "content": "ok"}), msgs)
    roles = [m["role"] for m in seen["messages"]]
    assert roles[0] == "system" and "system" not in roles[1:]
    assert "the tool failed" in seen["messages"][2]["content"]

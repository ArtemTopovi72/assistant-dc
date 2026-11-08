from types import SimpleNamespace
import llm


def payload(no_think=True, tool_choice='auto', force_think=False, json_schema=None, multiple=False):
    captured = {}
    ctx = SimpleNamespace(model_name='gemma4-26b-a4b-uncensored-hauhaucs-balanced@q5_k_m',
                          no_think=no_think, is_cancelled=lambda: False,
                          deps=SimpleNamespace(_stream_chat=lambda ctx, body: captured.update(body) or {'content': 'ok'},
                                               _wait_for_free_card=lambda ctx: None,
                                               throttle_external_calls=lambda ctx: None))
    tools = [{'type': 'function', 'function': {'name': 'generate_video', 'parameters': {}}}]
    if multiple:
        tools.append({'type': 'function', 'function': {'name': 'run_code', 'parameters': {}}})
    llm.send_to_lm_studio(ctx, [{'role': 'user', 'content': 'Make a video'}], tools=tools,
                          tool_choice=tool_choice, prefill='<think></think>',
                          force_think=force_think, json_schema=json_schema)
    return captured


def test_gemma_no_think_auto_tool_gets_native_prefill():
    p = payload()
    assert p['messages'][-1]['content'] == llm.GEMMA_NO_THINK_PREFILL


def test_constrained_or_reasoning_calls_keep_native_reasoning():
    for kw in ({'tool_choice': 'required'}, {'no_think': False}, {'force_think': True}, {'multiple': True},
               {'json_schema': {'type': 'object'}}):
        assert payload(**kw)['messages'][-1]['role'] != 'assistant'

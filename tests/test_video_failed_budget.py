from types import SimpleNamespace

import graph_personality as graph
from tool_image_handlers import _render_budget_exhausted


def test_video_failure_does_not_instruct_delivery_of_nonexistent_clip():
    state = {'_renders_used': 3, '_video_failure_reason': 'not enough free VRAM'}
    result = _render_budget_exhausted(state, 'generate_video', cost=3)
    assert 'No video was produced' in result
    assert 'not enough free VRAM' in result
    assert 'Deliver the best clip' not in result


def test_failed_video_is_not_forced_again(monkeypatch):
    monkeypatch.setattr(graph, '_turn_intent', lambda *a: {'must_call': 'generate_video'})
    tools = [{'function': {'name': 'generate_video'}}]
    assert graph._forced_intent(SimpleNamespace(), {}, 'continue the video', True, tools, (),
                                {('generate_video', '{}'): 1}) == ''

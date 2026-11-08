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


def test_unavailable_speech_check_marks_the_video_unverified(monkeypatch):
    import subprocess
    import tool_image_handlers as handlers
    def fail(*args, **kwargs):
        raise subprocess.TimeoutExpired('ffmpeg', 1)
    monkeypatch.setattr(subprocess, 'run', fail)
    ctx = SimpleNamespace(video_speech_unverified=False)
    result = handlers._speech_ok(ctx, 'missing.mp4', ['hello'])
    assert result is handlers._SPEECH_CHECK_UNAVAILABLE
    assert ctx.video_speech_unverified


def test_different_failed_edit_calls_stop_forcing_same_tool(monkeypatch):
    monkeypatch.setattr(graph, '_turn_intent', lambda *a: {'must_call': 'inpaint_image'})
    tools = [{'function': {'name': 'inpaint_image'}}]
    failures = {('inpaint_image', '{"region":"barrel"}'): 1,
                ('inpaint_image', '{"region":"apron"}'): 1}
    assert graph._forced_intent(SimpleNamespace(), {}, 'change the inscriptions', True,
                                tools, (), failures) == ''

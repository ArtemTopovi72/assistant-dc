import contextlib
import types
import comfy_client as CC


def test_nested_video_retry_evicts_idle_comfy_models_before_vram_check(monkeypatch):
    available = iter([1300, 22000])
    calls = []
    fake_lt = types.SimpleNamespace(free_gpu=lambda **kw: calls.append('free'))
    response = types.SimpleNamespace(raise_for_status=lambda: None,
                                     json=lambda: {'queue_running': [], 'queue_pending': []})
    ctx = types.SimpleNamespace(deps=types.SimpleNamespace(
        server_healthy=lambda **kw: True, gpu_holder=lambda: None,
        _gpu_slot=lambda **kw: contextlib.nullcontext(),
        gpu_free_mb=lambda: next(available), lora_training=fake_lt,
        requests=types.SimpleNamespace(get=lambda *a, **kw: response)))
    monkeypatch.setattr(CC, 'this_thread_holds_card', lambda: True)
    monkeypatch.setattr(CC, '_submit_and_poll_locked', lambda *a, **kw: 'result.mp4')
    assert CC._submit_and_poll(ctx, {}, label='H3 i2va', exclusive=True, min_free_mb=20000) == 'result.mp4'
    assert calls == ['free']


def test_nested_retry_does_not_evict_an_active_comfy_job(monkeypatch):
    calls = []
    response = types.SimpleNamespace(raise_for_status=lambda: None,
                                     json=lambda: {'queue_running': [['other']], 'queue_pending': []})
    ctx = types.SimpleNamespace(deps=types.SimpleNamespace(
        server_healthy=lambda **kw: True, gpu_holder=lambda: None,
        _gpu_slot=lambda **kw: contextlib.nullcontext(), gpu_free_mb=lambda: 1300,
        lora_training=types.SimpleNamespace(free_gpu=lambda **kw: calls.append('free')),
        requests=types.SimpleNamespace(get=lambda *a, **kw: response)))
    monkeypatch.setattr(CC, 'this_thread_holds_card', lambda: True)
    assert CC._submit_and_poll(ctx, {}, label='H3 i2va', exclusive=True, min_free_mb=20000) is None
    assert calls == []

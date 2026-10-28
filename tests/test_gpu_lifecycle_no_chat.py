from types import SimpleNamespace
import comfy_client as cc


def test_exclusive_job_parks_models_even_without_chat_model():
    calls = []
    lt = SimpleNamespace(loaded_llm_id=lambda: '', free_gpu=lambda **kw: calls.append('park'),
                         FREE_GPU_WAIT_S=0, wait_vram_free=lambda *a, **kw: 25000,
                         reload_llm=lambda *a, **kw: calls.append('chat'))
    ctx = SimpleNamespace(deps=SimpleNamespace(lora_training=lt, _free_comfy_models=lambda: calls.append('free_comfy')))
    cc.reset_card_state()
    try:
        with cc._gpu_slot(ctx=ctx, exclusive=True, label='image'):
            assert calls == ['park']
            assert cc.card_is_exclusive() == 'image'
        assert calls == ['park', 'free_comfy']
        assert cc.wait_for_card(0)
    finally:
        cc.reset_card_state()
def test_embedding_and_voice_are_restored_even_when_render_fails():
    calls = []
    def restore():
        assert cc.card_is_exclusive() == 'Restoring models'
        calls.append('restore')
    lifecycle = SimpleNamespace(park=lambda: calls.append('park'), restore=restore)
    lt = SimpleNamespace(loaded_llm_id=lambda: '', free_gpu=lambda **kw: None,
                         FREE_GPU_WAIT_S=0, wait_vram_free=lambda *a, **kw: calls.append('drain') or 25000,
                         reload_llm=lambda *a, **kw: None)
    ctx = SimpleNamespace(deps=SimpleNamespace(lora_training=lt, gpu_models=lifecycle,
                                               _free_comfy_models=lambda: calls.append('free_comfy')))
    cc.reset_card_state()
    try:
        try:
            with cc._gpu_slot(ctx=ctx, exclusive=True, label='image'):
                assert calls == ['park']
                raise RuntimeError('render failed')
        except RuntimeError:
            pass
        assert calls == ['park', 'free_comfy', 'drain', 'restore']
        assert cc.wait_for_card(0)
    finally:
        cc.reset_card_state()

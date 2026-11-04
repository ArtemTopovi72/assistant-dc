from types import SimpleNamespace
import pytest
import comfy_client as cc


def test_interrupted_claim_restores_model_and_reopens_speech(monkeypatch):
    events = []
    class Lifecycle:
        FREE_GPU_WAIT_S = 0
        @staticmethod
        def loaded_llm_id(): return 'test-model'
        @staticmethod
        def free_gpu(**kwargs):
            events.append('unloaded')
            raise KeyboardInterrupt()
        @staticmethod
        def wait_vram_free(wanted, timeout): return wanted
        @staticmethod
        def reload_llm(model, **kwargs): events.append('restored:' + model)
    monkeypatch.setenv('F5_TEST_RUN', '1')
    monkeypatch.setattr(cc, '_free_comfy_models', lambda: None)
    cc.reset_card_state()
    ctx = SimpleNamespace(deps=SimpleNamespace(lora_training=Lifecycle))
    try:
        with pytest.raises(KeyboardInterrupt):
            cc._claim_card('test', ctx=ctx)
        assert events == ['unloaded', 'restored:test-model']
        assert cc._excl_users == 0 and cc.wait_for_card(.1) and cc._SPEECH_FREE.is_set()
    finally:
        cc.reset_card_state()

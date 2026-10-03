"""A failed render reports ComfyUI's real reason, not a blank "no file".

With only "the render produced no file" the chat model invented a cause for
the user ("a conflict in the animation algorithms", live 2026-10-03) while the
actual error sat in the log.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("F5_TEST_RUN", "1")
import comfy_client as CC


class _Resp:
    def __init__(self, code, data):
        self.status_code, self._d = code, data

    def json(self):
        return self._d


def _no_server_checks(monkeypatch):
    monkeypatch.setattr(CC, "server_healthy", lambda: True)
    monkeypatch.setattr(CC, "gpu_holder", lambda: "")
    monkeypatch.setattr(CC, "_apply_model_fallbacks", lambda wf: wf)
    monkeypatch.setattr(CC, "throttle_external_calls", lambda ctx: None)


def test_validation_error_is_the_reason(monkeypatch):
    _no_server_checks(monkeypatch)
    monkeypatch.setattr(CC.requests, "post", lambda *a, **k: _Resp(400, {"node_errors": {
        "7": {"class_type": "UNETLoader", "errors": [
            {"message": "Value not in list", "details": "unet_name: 'h3.safetensors' not in []"}]}}}))
    assert CC._submit_and_poll(None, {"7": {}}, timeout=5, label="H3 i2v") is None
    why = CC.last_failure()
    assert "rejected" in why and "unet_name" in why and "UNETLoader" in why, why


def test_execution_error_is_the_reason(monkeypatch):
    _no_server_checks(monkeypatch)
    monkeypatch.setattr(CC.requests, "post", lambda *a, **k: _Resp(200, {"prompt_id": "p1"}))
    hist = {"p1": {"outputs": {}, "status": {"status_str": "error", "completed": False, "messages": [
        ["execution_error", {"node_type": "SamplerCustomAdvanced", "exception_type":
                             "torch.OutOfMemoryError",
                             "exception_message": "CUDA out of memory. Tried to allocate 2 GiB\nmore"}]]}}}
    monkeypatch.setattr(CC.requests, "get", lambda *a, **k: _Resp(200, hist))
    monkeypatch.setattr(CC.time, "sleep", lambda s: None)
    assert CC._submit_and_poll(None, {}, timeout=5, label="H3 i2v") is None
    why = CC.last_failure()
    assert "SamplerCustomAdvanced" in why and "out of memory" in why and "more" not in why, why


def test_server_down_is_the_reason(monkeypatch):
    monkeypatch.setattr(CC, "server_healthy", lambda: False)
    assert CC._submit_and_poll(None, {}, timeout=5) is None
    assert "not running" in CC.last_failure()


def test_a_new_submit_clears_the_old_reason(monkeypatch):
    CC._fail("old")
    monkeypatch.setattr(CC, "server_healthy", lambda: False)
    CC._submit_and_poll(None, {}, timeout=5)
    assert "old" not in CC.last_failure()

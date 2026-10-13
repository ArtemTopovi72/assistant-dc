"""A failed render reports ComfyUI's real reason, not a blank "no file".

With only "the render produced no file" the chat model invented a cause for
the user ("a conflict in the animation algorithms") while the actual error sat
in the log.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("F5_TEST_RUN", "1")
import time
import types

import pytest

import comfy_client as CC
from fakes import Over


@pytest.fixture
def fakes():
    """What a test hands the client on its context: {name: v}, {"requests.x": v}, {"time.x": v}."""
    return {}


def _W(fakes):
    flat = {k: v for k, v in fakes.items() if "." not in k}
    sub = {m: {k.split(".", 1)[1]: v for k, v in fakes.items() if k.startswith(m + ".")}
           for m in ("requests", "time")}
    return types.SimpleNamespace(deps=types.SimpleNamespace(
        **flat, requests=Over(CC.requests, **sub["requests"]), time=Over(time, **sub["time"])))


class _Resp:
    def __init__(self, code, data):
        self.status_code, self._d = code, data

    def json(self):
        return self._d


def _no_server_checks(fakes):
    fakes["server_healthy"] = (lambda **k: True)
    fakes["gpu_holder"] = (lambda: "")
    fakes["_apply_model_fallbacks"] = (lambda wf: wf)
    fakes["throttle_external_calls"] = (lambda ctx: None)


def test_validation_error_is_the_reason(fakes):
    _no_server_checks(fakes)
    fakes["requests.post"] = (lambda *a, **k: _Resp(400, {"node_errors": {
        "7": {"class_type": "UNETLoader", "errors": [
            {"message": "Value not in list", "details": "unet_name: 'h3.safetensors' not in []"}]}}}))
    assert CC._submit_and_poll(_W(fakes), {"7": {}}, timeout=5, label="H3 i2v") is None
    why = CC.last_failure()
    assert "rejected" in why and "unet_name" in why and "UNETLoader" in why, why


def test_execution_error_is_the_reason(fakes):
    _no_server_checks(fakes)
    fakes["requests.post"] = (lambda *a, **k: _Resp(200, {"prompt_id": "p1"}))
    hist = {"p1": {"outputs": {}, "status": {"status_str": "error", "completed": False, "messages": [
        ["execution_error", {"node_type": "SamplerCustomAdvanced", "exception_type":
                             "torch.OutOfMemoryError",
                             "exception_message": "CUDA out of memory. Tried to allocate 2 GiB\nmore"}]]}}}
    fakes["requests.get"] = (lambda *a, **k: _Resp(200, hist))
    fakes["time.sleep"] = (lambda s: None)
    assert CC._submit_and_poll(_W(fakes), {}, timeout=5, label="H3 i2v") is None
    why = CC.last_failure()
    assert "SamplerCustomAdvanced" in why and "out of memory" in why and "more" not in why, why


def test_server_down_is_the_reason(fakes):
    fakes["server_healthy"] = (lambda **k: False)
    assert CC._submit_and_poll(_W(fakes), {}, timeout=5) is None
    assert "not running" in CC.last_failure()


def test_a_new_submit_clears_the_old_reason(fakes):
    CC._fail("old")
    fakes["server_healthy"] = (lambda **k: False)
    CC._submit_and_poll(_W(fakes), {}, timeout=5)
    assert "old" not in CC.last_failure()


def test_submit_exception_is_the_reason(fakes):
    _no_server_checks(fakes)

    def boom(*a, **k):
        raise CC.requests.exceptions.ConnectionError("refused")
    fakes["requests.post"] = (boom)
    assert CC._submit_and_poll(_W(fakes), {}, timeout=5) is None
    why = CC.last_failure()
    assert "could not submit" in why and "ConnectionError" in why, why


def test_a_timed_out_job_is_stopped_on_the_server(fakes):
    _no_server_checks(fakes)
    fakes["requests.post"] = (lambda *a, **k: _Resp(200, {"prompt_id": "p9"}))
    fakes["requests.get"] = (lambda *a, **k: _Resp(200, {}))
    fakes["_job_still_known"] = (lambda pid: True)
    cancelled = []
    fakes["_cancel_job"] = (cancelled.append)
    clock = iter(range(0, 10_000, 7))
    fakes["time.time"] = (lambda: next(clock))
    fakes["time.sleep"] = (lambda s: None)
    assert CC._submit_and_poll(_W(fakes), {}, timeout=30) is None
    assert cancelled == ["p9"]
    assert "did not finish" in CC.last_failure()

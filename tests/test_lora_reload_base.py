"""The chat model's give-back after a render must poll LM Studio's REAL model list.

lora_training.reload_llm defaulted to ".../v1", lmstudio appended /api/v0/models,
and every render's restore polled a 404-ish endpoint and reported failure.
Run: venv/Scripts/python.exe -m pytest tests/test_lora_reload_base.py -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["F5_TEST_RUN"] = "1"
import config
import lmstudio
import lora_training as LT


def test_base_has_no_v1(monkeypatch):
    for b in ("http://127.0.0.1:1234", "http://127.0.0.1:1234/", "http://127.0.0.1:1234/v1"):
        monkeypatch.setattr(config, "LM_STUDIO_BASE", b)
        assert LT._lms_base() == "http://127.0.0.1:1234"


def test_reload_llm_passes_the_root(monkeypatch):
    seen = {}
    monkeypatch.setattr(lmstudio, "ensure_exclusive", lambda base, m: (seen.update(base=base), (True, ""))[1])
    monkeypatch.setattr(config, "LM_STUDIO_BASE", "http://127.0.0.1:1234")
    assert LT.reload_llm("m") is True
    assert not seen["base"].endswith("/v1")

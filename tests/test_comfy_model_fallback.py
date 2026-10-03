"""A graph naming an unpublished local model file is sent with its public stand-in.

The Ideogram graph names int8 files that were quantized locally and exist
nowhere to download; a fresh install (which fetches the official fp8 files)
failed every drawing with "value not in list".
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("F5_TEST_RUN", "1")
import comfy_client as CC


def _wf():
    return {"1": {"class_type": "UNETLoader",
                  "inputs": {"unet_name": "ig4-int8mixedrow_simple.safetensors"}},
            "2": {"class_type": "UNETLoader",
                  "inputs": {"unet_name": "ig4_uncond-int8mixedrow_simple.safetensors"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}}}


def _serve(monkeypatch, names):
    CC._AVAILABLE.clear()
    monkeypatch.setattr(CC, "_available_models", lambda c, f: set(names))


def test_missing_int8_uses_fp8(monkeypatch):
    _serve(monkeypatch, ["ideogram4_fp8_scaled.safetensors",
                         "ideogram4_unconditional_fp8_scaled.safetensors"])
    wf = CC._apply_model_fallbacks(_wf())
    assert wf["1"]["inputs"]["unet_name"] == "ideogram4_fp8_scaled.safetensors"
    assert wf["2"]["inputs"]["unet_name"] == "ideogram4_unconditional_fp8_scaled.safetensors"
    assert wf["3"]["inputs"]["vae_name"] == "flux2-vae.safetensors"


def test_present_int8_is_kept(monkeypatch):
    _serve(monkeypatch, ["ig4-int8mixedrow_simple.safetensors", "ideogram4_fp8_scaled.safetensors",
                         "ig4_uncond-int8mixedrow_simple.safetensors"])
    wf = CC._apply_model_fallbacks(_wf())
    assert wf["1"]["inputs"]["unet_name"] == "ig4-int8mixedrow_simple.safetensors"


def test_unknown_server_list_changes_nothing(monkeypatch):
    CC._AVAILABLE.clear()
    monkeypatch.setattr(CC, "_available_models", lambda c, f: None)
    assert CC._apply_model_fallbacks(_wf()) == _wf()


def test_object_info_parsing(monkeypatch):
    CC._AVAILABLE.clear()

    class R:
        def json(self):
            return {"UNETLoader": {"input": {"required": {
                "unet_name": [["a.safetensors", "b.safetensors"], {}]}}}}
    monkeypatch.setattr(CC.requests, "get", lambda *a, **k: R())
    assert CC._available_models("UNETLoader", "unet_name") == {"a.safetensors", "b.safetensors"}

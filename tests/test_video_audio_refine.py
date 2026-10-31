"""H3 may spend extra steps on sound while returning the original video frames."""
from types import SimpleNamespace

import psutil
import video


def test_audio_refine_cache_is_ram_only_when_enabled(monkeypatch):
    monkeypatch.delenv("VIDEO_AUDIO_REFINE_CACHE", raising=False)
    monkeypatch.setattr(video, "_server_has_node", lambda _: True)
    monkeypatch.setattr(video, "_audio_cache_fits", lambda _: True)
    wf = video.build_workflow("A motor running", mode="t2va", width=576,
                              height=576, frames=124, seed=7, steps=5,
                              two_stage=False)
    refine = next(v for v in wf.values() if v["class_type"] == "H3AudioRefineSampler")
    cache_id = refine["inputs"]["model"][0]
    cache = wf[cache_id]
    assert cache["class_type"] == "H3FrozenVideoCache"
    assert cache["inputs"]["backend"] == "ram"
    assert cache["inputs"]["precision"] == "int4"
    assert cache["inputs"]["verbose"] is False
    assert cache["inputs"]["vram_margin_gb"] == 1.0
    assert cache["inputs"]["allow_disk"] is False
    assert cache["inputs"]["free_after_pass"] is True


def test_audio_refine_runs_without_optional_cache_node(monkeypatch):
    monkeypatch.delenv("VIDEO_AUDIO_REFINE_CACHE", raising=False)
    monkeypatch.setattr(video, "_server_has_node", lambda name: name == "H3AudioRefineSampler")
    wf = video.build_workflow("A motor running", mode="t2va", width=576,
                              height=576, frames=124, seed=7, steps=5,
                              two_stage=False)
    refine = next(v for v in wf.values() if v["class_type"] == "H3AudioRefineSampler")
    assert wf[refine["inputs"]["model"][0]]["class_type"] == "MiniMaxH3SigmaShift"
    assert not any(v["class_type"] == "H3FrozenVideoCache" for v in wf.values())


def test_audio_cache_skips_long_clip_when_ram_is_tight(monkeypatch):
    monkeypatch.setattr(video, "_server_has_node", lambda _: True)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=30 << 30))
    wf = video.build_workflow("A motor running", mode="t2va", width=1344,
                              height=768, frames=362, seed=7, steps=5,
                              two_stage=False)
    assert not any(v["class_type"] == "H3FrozenVideoCache" for v in wf.values())
    assert any(v["class_type"] == "H3AudioRefineSampler" for v in wf.values())


def test_audio_cache_skips_remote_comfy_without_remote_memory_check(monkeypatch):
    monkeypatch.setattr(video, "COMFY_URL", "http://other-host:8188")
    monkeypatch.setattr(video, "_server_has_node", lambda _: True)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=100 << 30))
    wf = video.build_workflow("A motor running", mode="t2va", width=576,
                              height=576, frames=124, seed=7, steps=5,
                              two_stage=False)
    assert not any(v["class_type"] == "H3FrozenVideoCache" for v in wf.values())


def test_audio_refine_defaults_to_five_steps_without_changing_video(monkeypatch):
    monkeypatch.delenv("VIDEO_AUDIO_REFINE_STEPS", raising=False)
    monkeypatch.setenv("VIDEO_AUDIO_REFINE_CACHE", "0")
    monkeypatch.setattr(video, "_server_has_node", lambda _: True)
    wf = video.build_workflow("A motor running", mode="t2va", width=576,
                              height=576, frames=124, seed=7, steps=5,
                              two_stage=False)
    refine = next(k for k, v in wf.items() if v["class_type"] == "H3AudioRefineSampler")
    assert wf[refine]["inputs"]["steps"] == 5
    assert wf[refine]["inputs"]["video_denoise"] == 0.0
    plain_model = wf[refine]["inputs"]["model"][0]
    assert plain_model != wf["8"]["inputs"]["model"][0]
    assert wf[plain_model]["inputs"]["model"] == [video.N_UNET, 0]
    assert wf["9"]["inputs"]["samples"] == ["8", 0]
    assert wf["10"]["inputs"]["samples"] == [refine, 0]


def test_audio_refine_freezes_video_and_keeps_first_pass_step_count(monkeypatch):
    monkeypatch.setenv("VIDEO_AUDIO_REFINE_STEPS", "2")
    monkeypatch.setenv("VIDEO_AUDIO_REFINE_CACHE", "0")
    monkeypatch.setattr(video, "_server_has_node", lambda _: True)
    for mode in ("t2va", "ref2va"):
        wf = video.build_workflow("A motor and screams", mode=mode, width=576,
                                  height=576, frames=141, seed=7, steps=8,
                                  two_stage=False)
        refine = next(k for k, v in wf.items() if v["class_type"] == "H3AudioRefineSampler")
        assert wf["8"]["inputs"]["steps"] == 8
        assert wf[refine]["inputs"]["steps"] == 2
        assert wf[refine]["inputs"]["video_denoise"] == 0.0
        plain_model = wf[refine]["inputs"]["model"][0]
        assert wf[plain_model]["inputs"]["model"] == [video.N_UNET, 0]
        assert wf["9"]["inputs"]["samples"] == ["8", 0]
        assert wf["10"]["inputs"]["samples"] == [refine, 0]


def test_audio_refine_falls_back_when_node_is_not_loaded(monkeypatch):
    monkeypatch.setenv("VIDEO_AUDIO_REFINE_STEPS", "2")
    monkeypatch.setattr(video, "_server_has_node", lambda _: False)
    wf = video.build_workflow("A motor", mode="ref2va", width=576,
                              height=576, frames=141, seed=7, two_stage=False)
    assert not any(v["class_type"] == "H3AudioRefineSampler" for v in wf.values())
    assert wf["10"]["inputs"]["samples"] == ["8", 0]


def test_two_stage_refines_stage_one_audio_and_keeps_upscaled_video(monkeypatch):
    monkeypatch.setenv("VIDEO_AUDIO_REFINE_STEPS", "2")
    monkeypatch.setenv("VIDEO_AUDIO_REFINE_CACHE", "0")
    monkeypatch.setattr(video, "_server_has_node", lambda _: True)
    wf = video.build_workflow("Motorcycle", mode="ref2va", width=576,
                              height=576, frames=141, seed=7, steps=8,
                              two_stage=True)
    refine = next(k for k, v in wf.items() if v["class_type"] == "H3AudioRefineSampler")
    assert wf[refine]["inputs"]["latent"] == ["8", 0]
    plain_model = wf[refine]["inputs"]["model"][0]
    assert wf[plain_model]["inputs"]["model"] == [video.N_UNET, 0]
    assert wf["9"]["inputs"]["samples"] != ["8", 0]
    assert wf["10"]["inputs"]["samples"] == [refine, 0]

"""H3 may spend extra steps on sound while returning the original video frames."""
import video


def test_audio_refine_freezes_video_and_keeps_first_pass_step_count(monkeypatch):
    monkeypatch.setenv("VIDEO_AUDIO_REFINE_STEPS", "2")
    monkeypatch.setattr(video, "_server_has_node", lambda _: True)
    for mode in ("t2va", "ref2va"):
        wf = video.build_workflow("A motor and screams", mode=mode, width=576,
                                  height=576, frames=141, seed=7, steps=8,
                                  two_stage=False)
        refine = next(k for k, v in wf.items() if v["class_type"] == "H3AudioRefineSampler")
        assert wf["8"]["inputs"]["steps"] == 8
        assert wf[refine]["inputs"]["steps"] == 2
        assert wf[refine]["inputs"]["video_denoise"] == 0.0
        assert wf[refine]["inputs"]["model"] == wf["8"]["inputs"]["model"]
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
    monkeypatch.setattr(video, "_server_has_node", lambda _: True)
    wf = video.build_workflow("Motorcycle", mode="ref2va", width=576,
                              height=576, frames=141, seed=7, steps=8,
                              two_stage=True)
    refine = next(k for k, v in wf.items() if v["class_type"] == "H3AudioRefineSampler")
    assert wf[refine]["inputs"]["latent"] == ["8", 0]
    assert wf[refine]["inputs"]["model"] == wf["8"]["inputs"]["model"]
    assert wf["9"]["inputs"]["samples"] != ["8", 0]
    assert wf["10"]["inputs"]["samples"] == [refine, 0]

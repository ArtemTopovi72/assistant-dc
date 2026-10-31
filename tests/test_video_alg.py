"""Early low-pass conditioning must return to the original image before decode."""

import video


def test_single_image_uses_blurred_condition_only_in_first_step(monkeypatch):
    monkeypatch.setenv("VIDEO_ALG", "1")
    monkeypatch.setattr(video, "_upload", lambda *args, **kwargs: "photo.png")
    wf = video.build_workflow("A toy drone lifts off", mode="t2va", width=576,
                              height=576, frames=124, seed=7, images=["photo.png"],
                              steps=3, two_stage=False)

    first = next(k for k, v in wf.items()
                 if v["class_type"] == "KSamplerAdvanced" and v["inputs"]["end_at_step"] == 1)
    assert wf[first]["inputs"]["start_at_step"] == 0
    assert wf[first]["inputs"]["return_with_leftover_noise"] == "enable"
    assert wf["8"]["inputs"]["latent_image"] == [first, 0]
    assert wf["8"]["inputs"]["start_at_step"] == 1
    assert wf["8"]["inputs"]["end_at_step"] == 3
    assert wf["8"]["inputs"]["positive"] == ["6", 0]
    assert wf["9"]["inputs"]["samples"] == ["8", 0]
    blurred = wf[first]["inputs"]["positive"][0]
    assert wf[blurred]["inputs"]["first_frame"] != wf["6"]["inputs"]["first_frame"]
    assert sum(v["class_type"] == "ImageScale" for v in wf.values()) == 2


def test_lowpass_does_not_change_text_or_reference_modes(monkeypatch):
    monkeypatch.setenv("VIDEO_ALG", "1")
    for mode in ("t2va", "ref2va"):
        wf = video.build_workflow("A toy drone lifts off", mode=mode, width=576,
                                  height=576, frames=124, seed=7, steps=3,
                                  two_stage=False)
        assert wf["8"]["class_type"] == "KSampler"

"""Default sampling changes only for an ordinary single-photo FL2VA clip."""

from types import SimpleNamespace

from PIL import Image

import video


def test_video_step_routes(monkeypatch, tmp_path):
    photo = tmp_path / "photo.png"
    Image.new("RGB", (576, 576), "purple").save(photo)
    monkeypatch.setattr(video, "engine_available", lambda ctx: (True, ""))
    monkeypatch.setattr(video, "_normalize_save_node", lambda wf: wf)
    monkeypatch.setattr(video, "_adopt_output", lambda path, *args: path)
    monkeypatch.setattr(video, "comfy_client", SimpleNamespace(
        _submit_and_poll=lambda *args, **kwargs: "fake.mp4"))
    captured = []
    monkeypatch.setattr(video, "build_workflow", lambda *args, **kwargs: captured.append(kwargs) or {})

    cases = [
        ({"images": [str(photo)]}, video.VIDEO_STEPS_FL2VA),
        ({}, video.VIDEO_STEPS),
        ({"images": [str(photo), str(photo)]}, video.VIDEO_STEPS_REF2VA),
        ({"images": [str(photo)], "keyframe": True}, video.VIDEO_STEPS),
        ({"images": [str(photo)], "steps": 7}, 7),
    ]
    for kwargs, expected in cases:
        result = video.generate_video(None, "A car drives forward", prepared=True,
                                      seconds=5.9, **kwargs)
        assert result["status"] == "success"
        assert captured[-1]["steps"] == expected


def test_five_step_photo_keeps_taomate(monkeypatch):
    monkeypatch.setenv("VIDEO_ALG", "0")
    monkeypatch.setattr(video, "_upload", lambda *args, **kwargs: "photo.png")
    photo = video.build_workflow("A car drives", mode="t2va", width=576,
                                 height=576, frames=124, seed=7,
                                 images=["photo.png"], steps=5, two_stage=False)
    text = video.build_workflow("A car drives", mode="t2va", width=576,
                                height=576, frames=124, seed=7,
                                steps=5, two_stage=False)
    anchored = video.build_workflow("A person talks", mode="t2va", width=576,
                                    height=576, frames=124, seed=7,
                                    images=["photo.png", "photo.png"],
                                    steps=5, two_stage=False)
    assert photo[video.N_LORA]["inputs"]["lora_name"] == video.VIDEO_FL2VA_LORA_3STEP
    assert photo[video.N_LORA]["inputs"]["strength_model"] == 1.0
    assert text[video.N_LORA]["inputs"]["lora_name"] != video.VIDEO_FL2VA_LORA_3STEP
    assert anchored[video.N_LORA]["inputs"]["lora_name"] != video.VIDEO_FL2VA_LORA_3STEP

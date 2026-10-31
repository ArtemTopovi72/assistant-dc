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

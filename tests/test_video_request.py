"""The user's current request outranks a tool draft copied from older chat history."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import tool_image_handlers as handlers
import video
from tests.test_video_chain import _Rig


REQUEST = "Animate the red balloon with a slow camera push-in. Keep the blue box and white background."
DRAFT = "The green balloon sways while the camera pushes in."


@pytest.mark.parametrize("approved", [[], ["The red balloon sways."]])
def test_finished_video_is_checked_against_request_or_approved_plan(tmp_path, approved):
    still = tmp_path / "source.png"
    still.write_bytes(b"source")
    seen = {}
    look = SimpleNamespace(
        duration_seconds=lambda path: 5.17,
        sample_times=lambda *a, **kw: [0.0, 3.0],
        extract_frames=lambda path, times, dest: [(t, Path(dest) / f"{t}.png") for t in times],
    )

    def vision(ctx, **kw):
        seen.update(kw)
        return "The balloon is not green" if "green balloon" in kw["user_text"] else "NONE"

    rig = _Rig(tmp_path)
    remembered = []
    ctx = rig.ctx(
        text_model=lambda *a, **kw: "", vision_model=vision,
        modules={"video_look": look}, video_plan_parts=approved,
        deps=SimpleNamespace(asks_for_video=lambda *a: True),
        remember=lambda kind, text, meta: remembered.append(text),
    )
    result = handlers._handle_generate_video(
        ctx, {"image_path": str(still), "user_input_original": REQUEST},
        {"description": DRAFT, "seconds": 5.17, "use_current_images": True})
    expected = approved[0] if approved else REQUEST
    assert expected in seen["user_text"]
    assert "green balloon" not in seen["user_text"]
    assert ctx.video_action_note == ""
    assert "actions are NOT visible" not in result
    assert any(path == str(still) for _, path in seen["image_paths"])
    assert ctx.last_video_prompt == expected
    assert remembered == [f"Video for: {expected}"]


def test_context_ir_sees_authoritative_request_separately_from_tool_draft():
    seen = {}
    good = ("integrated_multimodal_description: [Shot 1] A red balloon sways.\n\n"
            "overall_soundscape: room tone.\n\nnon_diegetic_music: N/A")
    ctx = SimpleNamespace(
        video_request=REQUEST,
        vision_model=lambda ctx, **kw: seen.update(kw) or good,
    )
    assert video.to_context_ir(ctx, DRAFT, mode="i2va", seconds=5.17, images=["source.png"])
    assert REQUEST in seen["user_text"]
    assert DRAFT in seen["user_text"]


def test_camera_instruction_comes_from_current_request_rather_than_old_draft():
    asked = []
    ctx = SimpleNamespace(
        video_request="Animate this photo with a static camera.",
        yes_no=lambda question, text, **kw: asked.append(text) or "zoom" in text,
    )
    result = video._lock_framing_unless_requested("The camera zooms in.", ctx)
    assert asked == [ctx.video_request]
    assert video._FRAMING_LOCK_SUFFIX in result


@pytest.mark.parametrize("spoken_output", ["", '<d>[English] Buy an Ozon cable now</d>'])
def test_silent_user_request_never_restores_old_draft_speech(spoken_output):
    calls = []
    prompt = (f"integrated_multimodal_description: [Shot 1] A red cube. {spoken_output}\n\n"
              "overall_soundscape: room tone.\n\nnon_diegetic_music: N/A")
    ctx = SimpleNamespace(
        video_request="Animate the red cube. No people. No speech.",
        text_model=lambda *a, **kw: calls.append(True) or prompt,
        yes_no=lambda *a, **kw: False,
    )
    result = video.to_context_ir(ctx, 'A man says "Buy an Ozon cable now".', mode="t2va", seconds=5.17)
    assert "Buy an Ozon cable now" not in result
    assert len(calls) == 1


def test_only_current_part_authorized_speech_is_restored():
    prompt = ("integrated_multimodal_description: [Shot 1] A woman stands.\n\n"
              "overall_soundscape: room tone.\n\nnon_diegetic_music: N/A")
    ctx = SimpleNamespace(
        video_request='First she says "The first sentence". Later she says "The second sentence".',
        text_model=lambda *a, **kw: prompt,
        yes_no=lambda *a, **kw: False,
    )
    result = video.to_context_ir(ctx, 'She says "The first sentence".', mode="t2va", seconds=5.17)
    assert "The first sentence" in result
    assert "The second sentence" not in result


def test_words_in_a_silent_request_do_not_authorize_a_spoken_line():
    prompt = ("integrated_multimodal_description: [Shot 1] A red cube.\n\n"
              "overall_soundscape: room tone.\n\nnon_diegetic_music: N/A")
    ctx = SimpleNamespace(
        video_request="Animate the red cube. No speech.",
        text_model=lambda *a, **kw: prompt,
        yes_no=lambda *a, **kw: False,
    )
    result = video.to_context_ir(ctx, 'A man says "the red cube".', mode="t2va", seconds=5.17)
    assert "He says:" not in result


def test_approved_plan_cannot_reuse_speech_removed_from_original_request(tmp_path):
    line = "Buy an Ozon cable now"
    draft = f'A man says "{line}".'
    rig = _Rig(tmp_path, dictated_speech=lambda *a: line)
    look = SimpleNamespace(
        duration_seconds=lambda path: 0, sample_times=lambda *a, **kw: [],
        extract_frames=lambda *a: [],
    )
    ctx = rig.ctx(
        text_model=lambda *a, **kw: "", voice_choice="default",
        video_plan_parts=["A red cube. No people. No speech."],
        modules={"video_look": look}, deps=SimpleNamespace(asks_for_video=lambda *a: True),
    )
    handlers._handle_generate_video(ctx, {"user_input_original": draft},
                                   {"description": draft, "seconds": 5.17, "use_current_images": False})
    prompt = ("integrated_multimodal_description: [Shot 1] A red cube.\n\n"
              "overall_soundscape: room tone.\n\nnon_diegetic_music: N/A")
    ctx.text_model = lambda *a, **kw: prompt
    result = video.to_context_ir(ctx, draft, mode="t2va", seconds=5.17)
    assert line not in result

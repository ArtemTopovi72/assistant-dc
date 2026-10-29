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


def test_frame_review_is_reported_as_uncertain(tmp_path):
    still = tmp_path / 'source.png'
    still.write_bytes(b'source')
    look = SimpleNamespace(duration_seconds=lambda path: 5.17,
                           sample_times=lambda *a, **kw: [0.0, 3.0],
                           extract_frames=lambda path, times, dest: [(t, Path(dest) / f'{t}.png') for t in times])
    rig = _Rig(tmp_path)
    ctx = rig.ctx(text_model=lambda *a, **kw: '', vision_model=lambda *a, **kw: 'ball did not move',
                  modules={'video_look': look}, deps=SimpleNamespace(asks_for_video=lambda *a: True))
    result = handlers._handle_generate_video(ctx, {'image_path': str(still), 'user_input_original': REQUEST},
                                             {'description': DRAFT, 'seconds': 5.17, 'use_current_images': True})
    assert 'possible issues' in result and 'can misread motion' in result
    assert 'actions are NOT visible' not in result


def test_large_background_change_is_reported_when_vision_says_none(tmp_path):
    still = tmp_path / 'source.png'
    still.write_bytes(b'source')
    look = SimpleNamespace(duration_seconds=lambda path: 5.17,
                           sample_times=lambda *a, **kw: [0.0, 3.0],
                           extract_frames=lambda path, times, dest: [(t, Path(dest) / f'{t}.png') for t in times])
    rig = _Rig(tmp_path, overlay_drift=lambda path: 64, OVERLAY_DRIFT_MAX=35)
    ctx = rig.ctx(text_model=lambda *a, **kw: '', vision_model=lambda *a, **kw: 'NONE',
                  yes_no=lambda *a, **kw: False,
                  modules={'video_look': look}, deps=SimpleNamespace(asks_for_video=lambda *a: True))
    result = handlers._handle_generate_video(ctx, {'image_path': str(still), 'user_input_original': REQUEST},
                                             {'description': DRAFT, 'seconds': 5.17, 'use_current_images': True})
    assert ctx.video_action_note == 'The background may have changed unexpectedly.'
    assert 'possible issues' in result


def test_motion_and_background_problems_are_both_reported(tmp_path, monkeypatch):
    still = tmp_path / 'source.png'
    still.write_bytes(b'source')
    rig = _Rig(tmp_path)
    ctx = rig.ctx(text_model=lambda *a, **kw: '', vision_model=lambda *a, **kw: 'NONE',
                  deps=SimpleNamespace(asks_for_video=lambda *a: True))
    monkeypatch.setattr(handlers, '_check_actions', lambda *a, **kw: 'Wrong direction.')
    monkeypatch.setattr(handlers, '_check_static_background', lambda *a, **kw: 'Background changed.')
    handlers._handle_generate_video(ctx, {'image_path': str(still), 'user_input_original': REQUEST},
                                     {'description': DRAFT, 'seconds': 5.17, 'use_current_images': True})
    assert 'Wrong direction' in ctx.video_action_note and 'Background changed' in ctx.video_action_note


def test_final_failed_object_check_is_not_lost(tmp_path, monkeypatch):
    still = tmp_path / 'source.png'
    still.write_bytes(b'source')
    rig = _Rig(tmp_path, drawn_over=lambda *a: 0.9, DRAWN_OVER_MAX=0.5,
               overlay_drift=lambda path: 5, OVERLAY_DRIFT_MAX=35)
    ctx = rig.ctx(text_model=lambda *a, **kw: '', vision_model=lambda *a, **kw: 'NONE',
                  yes_no=lambda *a, **kw: False, deps=SimpleNamespace(asks_for_video=lambda *a: True))
    monkeypatch.setattr(handlers, '_check_actions', lambda *a, **kw: '')
    monkeypatch.setattr(handlers, '_check_static_background', lambda *a, **kw: '')
    result = handlers._handle_generate_video(ctx, {'image_path': str(still), 'user_input_original': 'Animate the balloon with a stationary camera.'},
                                             {'description': 'The balloon moves.', 'seconds': 5, 'use_current_images': True})
    assert 'possible issues' in result and 'objects' in result.lower()


def test_static_single_image_retries_obvious_background_corruption(tmp_path):
    still = tmp_path / 'source.png'
    still.write_bytes(b'source')
    scores = iter([64.0, 5.0])
    rig = _Rig(tmp_path, drawn_over=lambda *a: 0.0,
               overlay_drift=lambda path: next(scores), OVERLAY_DRIFT_MAX=35)
    ctx = rig.ctx(text_model=lambda *a, **kw: '', vision_model=lambda *a, **kw: 'NONE',
                  yes_no=lambda *a, **kw: False,
                  deps=SimpleNamespace(asks_for_video=lambda *a: True))
    handlers._handle_generate_video(ctx, {'image_path': str(still), 'user_input_original': 'Animate the balloon with a stationary camera.'},
                                     {'description': 'The red balloon moves right; the camera is stationary.', 'seconds': 5.17, 'use_current_images': True})
    assert len([call for call in rig.calls if call[0] != 'PREP']) == 2


def test_camera_motion_does_not_trigger_background_retry(tmp_path):
    still = tmp_path / 'source.png'
    still.write_bytes(b'source')
    rig = _Rig(tmp_path, drawn_over=lambda *a: 0.9,
               overlay_drift=lambda path: 64, OVERLAY_DRIFT_MAX=35)
    ctx = rig.ctx(text_model=lambda *a, **kw: '', vision_model=lambda *a, **kw: 'NONE',
                  yes_no=lambda *a, **kw: True,
                  deps=SimpleNamespace(asks_for_video=lambda *a: True))
    handlers._handle_generate_video(ctx, {'image_path': str(still), 'user_input_original': 'Animate the balloon and pan the camera right.'},
                                     {'description': 'Pan right past the balloon.', 'seconds': 5.17, 'use_current_images': True})
    assert len([call for call in rig.calls if call[0] != 'PREP']) == 1


def test_chain_does_not_continue_from_corrupted_first_part(tmp_path):
    still = tmp_path / 'source.png'
    still.write_bytes(b'source')
    rig = _Rig(tmp_path, drawn_over=lambda *a: 0.0,
               overlay_drift=lambda path: 64, OVERLAY_DRIFT_MAX=35)
    ctx = rig.ctx(text_model=lambda *a, **kw: '', vision_model=lambda *a, **kw: 'NONE',
                  yes_no=lambda *a, **kw: False, video_plan_parts=['The red balloon moves left.', 'The balloon moves back.'],
                  deps=SimpleNamespace(asks_for_video=lambda *a: True))
    result = handlers._handle_generate_video(ctx, {'image_path': str(still), 'user_input_original': 'Animate the balloon with a stationary camera.'},
                                             {'description': 'The balloon moves left.', 'seconds': 12, 'use_current_images': True})
    assert '[TOOL ERROR]' in result
    assert len([call for call in rig.calls if call[0] != 'PREP']) == 3
    assert not rig.stitched


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


def test_context_ir_ask_carries_the_style_anchor_and_one_action_arc():
    """The grandpa clip (2026-10-29): the rewrite said «Live-action» over a cartoon
    first frame and stacked five actions -- H3 hedged into ambient smoke. The ask
    must carry the style-anchor and one-arc rules to the rewriter."""
    seen = {}
    good = ("integrated_multimodal_description: [Shot 1] A red balloon sways.\n\n"
            "overall_soundscape: room tone.\n\nnon_diegetic_music: N/A")
    ctx = SimpleNamespace(
        video_request="Дедушка какает в свой кулачок левой руки",
        vision_model=lambda ctx, **kw: seen.update(kw) or good,
    )
    assert video.to_context_ir(ctx, DRAFT, mode="i2va", seconds=5.17, images=["source.png"])
    assert "STYLE AND ACTION RULES" in seen["user_text"]
    assert "never write 'Live-action'" in seen["user_text"]
    assert "ONE action arc" in seen["user_text"]

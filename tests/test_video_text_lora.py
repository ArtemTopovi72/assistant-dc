"""The typography route must keep each H3 adapter at its trained step count."""
import os
import sys
from types import SimpleNamespace

os.environ["F5_TEST_RUN"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for directory in ("", "agent", "imaging", "media", "core"):
    sys.path.insert(0, os.path.join(ROOT, directory))

import video

original_upload = video._upload
video._upload = lambda path, kind="image", post=None: os.path.basename(path)
try:
    text = SimpleNamespace(video_text_source=True)
    ordinary = SimpleNamespace(video_text_source=False)
    args = dict(prompt="SOL and TERRA on shirts", mode="t2va", width=768,
                height=448, frames=124, seed=42, images=["source.png"],
                steps=5, two_stage=False)
    selected = video.build_workflow(**args, ctx=text)
    assert selected[video.N_LORA]["inputs"]["lora_name"] == video.VIDEO_TEXT_FL2VA_LORA
    assert selected[video.N_LORA]["inputs"]["strength_model"] == 1.0
    assert selected[video.N_SAMPLER]["inputs"]["steps"] == 4

    normal = video.build_workflow(**args, ctx=ordinary)
    assert normal[video.N_SAMPLER]["inputs"]["steps"] == 5
    assert normal[video.N_LORA]["inputs"]["lora_name"] != video.VIDEO_TEXT_FL2VA_LORA

    chain = video.build_chain_part("SOL and TERRA", chain="test", idx=1,
                                   width=768, height=448, seconds=5.2, seed=42,
                                   ctx=text)
    assert chain[video.N_LORA]["inputs"]["lora_name"] == video.VIDEO_TEXT_FL2VA_LORA
    assert chain[video.N_SAMPLER]["inputs"]["steps"] == 4

    # A real tilted source frame was read as S0Z/ERRA. Those OCR guesses must
    # not become instructions that replace the true lettering in the video.
    original_ir, original_framing = video._context_ir_on, video._lock_framing_unless_requested
    original_style, original_stress = video._asks_new_style, video.VIDEO_SPEECH_STRESS
    try:
        video._context_ir_on = lambda: False
        video._lock_framing_unless_requested = lambda description, ctx: description
        video._asks_new_style = lambda request, ctx: False
        video.VIDEO_SPEECH_STRESS = False
        ocr_ctx = SimpleNamespace(video_untrusted_ocr=("S0Z", "ERRA"), video_request="оживи")
        prompt, _ = video.prepare_prompt(
            ocr_ctx, "The shirt text S0Z and ERRA stays visible. He says 'hello'.",
            images=["source.png"], seconds=5.2, mode="t2va")
        assert "S0Z" not in prompt and "ERRA" not in prompt
        assert "exact original lettering" in prompt and "hello" in prompt
        prompt, _ = video.prepare_prompt(
            ocr_ctx, "Keep 'SOZ' and 'ERRA' on the shirts. He says 'hello'.",
            images=['source.png'], seconds=5.2, mode='t2va')
        assert "'SOZ'" not in prompt and "'ERRA'" not in prompt and "hello" in prompt
        assert '"the exact original lettering' not in prompt and "'the exact original lettering" not in prompt
        prompt, _ = video.prepare_prompt(
            ocr_ctx, 'Keep "SOL" and "ERRA" on the shirts. He says "hello".',
            images=['source.png'], seconds=5.2, mode='t2va')
        assert '"SOL"' not in prompt and '"ERRA"' not in prompt and '"hello"' in prompt
        ocr_ctx.video_request = 'Сохрани надпись ERRA и пусть он скажет «hello».'
        prompt, _ = video.prepare_prompt(
            ocr_ctx, "Keep SOZ and ERRA on shirts. He says 'hello'.",
            images=['source.png'], seconds=5.2, mode='t2va')
        assert 'SOZ' not in prompt and 'ERRA' in prompt and 'hello' in prompt
        ocr_ctx.video_request = 'оживи'
        ocr_ctx.video_untrusted_ocr = ("THE",)
        prompt, _ = video.prepare_prompt(
            ocr_ctx, "The man reads THE on the shirt; the camera stays still.",
            images=["source.png"], seconds=5.2, mode="t2va")
        assert "The man" in prompt and "the camera" in prompt
        assert "reads THE" not in prompt
    finally:
        video._context_ir_on, video._lock_framing_unless_requested = original_ir, original_framing
        video._asks_new_style, video.VIDEO_SPEECH_STRESS = original_style, original_stress
finally:
    video._upload = original_upload
print("video typography adapter and OCR guard: 8 checks passed")

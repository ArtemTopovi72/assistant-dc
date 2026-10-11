"""Whole-image restyle with FireRed: the one place both style entry points render.

redraw_image(mode="redraw") with a style and the router's style_transfer category
both land here. Three things make a style land (bench/style_ab.py, 2026-10-11):

  * the style spelled out -- a preset's own text, or the user's free words turned
    into concrete traits (medium, line, shading, faces, palette) by the chat model,
    the way Qwen-Image-Edit's own demo rewrites edit instructions;
  * real guidance: the template's Lightning 8 steps at cfg 1 barely moved a photo
    (anime == comic, «oil» a warm filter); 12 steps at cfg 3 with a negative
    against the look being left behind gives each style its own face;
  * towards a photo, Anything2Real (a Qwen-Image-Edit-2511 LoRA) and who is in the
    picture: without the description an anime schoolgirl came back a young man and
    a Pixar baker lost his mustache; without the LoRA he stayed a 3D doll.
"""
import logging
import os
from typing import Optional

import config as _cfg
import style_presets

logger = logging.getLogger("assistant.image")

STYLE_STEPS = _cfg.env_int("FIRERED_STYLE_STEPS", 12)
STYLE_CFG = _cfg.env_float("FIRERED_STYLE_CFG", 3.0)
PHOTO_LORA = os.getenv("FIRERED_PHOTO_LORA", "style\\anything2real_2601_A.safetensors")
PHOTO_LORA_STRENGTH = _cfg.env_float("FIRERED_PHOTO_LORA_STRENGTH", 0.9)

NEG_DRAWN = ("photo, photograph, photorealistic, realistic skin texture, skin pores, 3d render, "
             "blurry, low quality, watermark, text, signature")
NEG_PHOTO = ("cartoon, anime, illustration, drawing, painting, 3d render, cgi, pixar, plastic "
             "skin, doll, big eyes, cel shading, lineart, blurry, low quality, watermark")
NEG_OTHER = "blurry, low quality, watermark, signature"

# The entire image, and everyone at the same size and the same position: a collage
# restyle once came back cropped with a person cut out (test_style_transfer_whole_image).
# The vision layout description that used to ride along is gone: it diluted the
# style (an oil painting came back a photo with smeared dirt), and these words kept
# the framing of every preset on their own (bench/style_ab.py).
_KEEP = (" Restyle the entire image including the background. Keep the same composition, "
         "framing and crop, every person at the same size, the same position and the same "
         "pose, {clothes}and the setting; do not crop, zoom or move anyone. Any signs, screens "
         "or lettering become abstract marks -- no new legible words.")

_EXPAND_SYS = (
    "You turn a request to restyle a picture into an instruction for an image-editing model. "
    "Reply with JSON only: {\"target\": \"drawn\" | \"photo\" | \"other\", \"style\": \"...\"}. "
    "target: drawn = any drawn, painted, animated or rendered art style; photo = make it look "
    "like a real photograph; other = a change that is neither (lighting, era, filter, season). "
    "style: ONE English paragraph starting with \"Turn picture 1 into\" that spells out what "
    "makes this exact style recognisable and different from its neighbours: the medium, the "
    "line work, the shading, how faces and eyes are drawn, the palette and the texture. "
    "Concrete visual traits only, no artist names you are unsure of, no composition changes.")

_WHO_PROMPT = (
    "For each person or character in this picture, left to right, give one short phrase: "
    "gender, approximate age, hair colour and style, facial hair, glasses, and clothing. "
    "Join them with '; '. Nothing else. If there are no people, describe the main subject "
    "in one short phrase.")


def _expand(ctx, words: str) -> tuple:
    """(target, style paragraph) for free words; falls back to the words themselves."""
    import llm
    from utils import safe_json_from_llm
    try:
        raw = llm.call_llm_simple(ctx, _EXPAND_SYS, words, temperature=0.2, max_tokens=400) or ""
        d = safe_json_from_llm(raw, ["target", "style"]) or {}
    except Exception:
        logger.exception("style: expanding %r failed", words[:80])
        d = {}
    target = d.get("target") if d.get("target") in ("drawn", "photo", "other") else ""
    if not target:          # no answer: the edit classifier's word on whether it is a style
        from image_router import classify_edit_intent
        target = "drawn" if classify_edit_intent(words) == "style_transfer" else "other"
    style = " ".join(str(d.get("style") or "").split())
    return target, (style or words)


def _who(ctx, image_path: str) -> str:
    from llm import analyze_image_with_llm
    try:
        return " ".join((analyze_image_with_llm(
            ctx=ctx, image_path=image_path, user_text=_WHO_PROMPT,
            system_prompt="You are a precise, literal visual describer.") or "").split())[:500]
    except Exception:
        logger.exception("style: describing who is in %s failed", image_path)
        return ""


def build(ctx, instructions: str, image_path: Optional[str] = None) -> tuple:
    """(instruction, target) for a restyle request."""
    import image_grounding
    key = style_presets.preset_for(instructions)
    if key:
        p = style_presets.STYLE_PRESETS[key]
        target, style, keeps_clothes = p["target"], p["prompt"], p.get("keeps_clothes", True)
    else:
        words = image_grounding._english_instructions(ctx, instructions) or instructions
        target, style = _expand(ctx, words)
        keeps_clothes = True
    keep = _KEEP.format(clothes="the same clothing " if keeps_clothes else "")
    if target == "photo":
        who = _who(ctx, image_path) if image_path else ""
        lead = f"change the picture 1 to realistic photograph, {who}. " if who else ""
        tail = (" Each person keeps the same gender, age, hair and facial hair, at real-life "
                "human body proportions (a normal-sized head on a normal adult body).") if who else ""
        return lead + style + tail + keep, target
    return style + keep, target


def restyle(ctx, image_path: str, instructions: str, *, seed=None, timeout=None):
    """Render the restyle; returns the saved path or None. A request that is not a
    style at all (target "other": relight, season, a filter) is edited as written."""
    import image as image_mod
    instr, target = build(ctx, instructions, image_path)
    kw = {k: v for k, v in (("seed", seed), ("timeout", timeout)) if v is not None}
    if target == "other" and not style_presets.preset_for(instructions):
        return image_mod.edit_image_with_firered(ctx, image_path, instructions, **kw)
    neg = {"drawn": NEG_DRAWN, "photo": NEG_PHOTO}.get(target, NEG_OTHER)
    logger.info("style: target=%s instr=%r", target, instr[:160])
    with image_mod.firered_sampler(STYLE_STEPS, STYLE_CFG, lightning=False, negative=neg), \
            image_mod.firered_extra_lora(PHOTO_LORA if target == "photo" else None,
                                         PHOTO_LORA_STRENGTH):
        return image_mod.edit_image_with_firered(ctx, image_path, instr, **kw)

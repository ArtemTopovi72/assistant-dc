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
from seams import vision_model, text_model, dep, yes_no
import logging
import os
import re
from typing import Optional

import config as _cfg
import style_presets

logger = logging.getLogger("assistant.image")

STYLE_STEPS = _cfg.env_int("FIRERED_STYLE_STEPS", 12)
STYLE_CFG = _cfg.env_float("FIRERED_STYLE_CFG", 3.0)
PHOTO_LORA = os.getenv("FIRERED_PHOTO_LORA", "style\\anything2real_2601_A.safetensors")
# A flat drawing keeps its face and armour details at 0.75 (at 0.9 an armoured midriff
# became one latex suit); a 3D cartoon render stays a doll below 0.9.
PHOTO_LORA_STRENGTH = _cfg.env_float("FIRERED_PHOTO_LORA_STRENGTH", 0.75)
PHOTO_LORA_STRENGTH_3D = _cfg.env_float("FIRERED_PHOTO_LORA_STRENGTH_3D", 0.9)

NEG_DRAWN = ("photo, photograph, photorealistic, realistic skin texture, skin pores, 3d render, "
             "blurry, low quality, watermark, text, signature")
NEG_PHOTO = ("cartoon, anime, illustration, drawing, painting, 3d render, cgi, pixar, plastic "
             "skin, doll, big eyes, cel shading, lineart, body tattoo, chest tattoo, body paint, "
             "blurry, low quality, watermark")
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
    "like a real photograph; other = a change that is neither (lighting, era, filter, season), "
    "and any change of WHAT the picture shows -- a new place, an action, a fight, other clothes, "
    "other people -- even when it also names a look («a cinematic, highly detailed scene of him "
    "in armor fighting demons in lava» is other: a new scene, not a style of this one). "
    "style: ONE English paragraph starting with \"Turn picture 1 into\" that spells out what "
    "makes this exact style recognisable and different from its neighbours: the medium, the "
    "line work, the shading, how faces and eyes are drawn, the palette and the texture. "
    "Concrete visual traits only, no artist names you are unsure of, no composition changes.")

# Clothing alone lost a character's scythes and boots and made her armour a cloth top:
# what each thing is MADE of, what is held or attached, and the background are named.
_WHO_PROMPT = (
    "First line: 'MEDIUM: 3D' if this is a 3D render or CGI cartoon, else 'MEDIUM: 2D'. Then "
    "describe this picture for a photographer who must recreate it exactly. For each person or "
    "character, left to right: gender, approximate age, body build exactly as drawn, "
    "judged against the drawing rather than an average person: bust size (small, medium, large, "
    "very large), waist (narrow, average, wide), hips, shoulders; hair colour, its texture "
    "(sleek and straight, wavy, spiky) and the length of each part (the fringe, the side locks "
    "-- to the cheek, the chin, the shoulders -- and the back); everything worn "
    "on the head (horns, ear-like or wing-like pieces, clips, headbands) with its exact colour "
    "and shape (a thick round horn or a flat thin fin or blade) -- look closely, such pieces are often nearly the colour of the hair; {face}"
    "every garment and costume "
    "piece SEPARATELY with its exact colour, its material (rigid metal armor plates, leather, "
    "cloth, latex), its exact shape and where on the body it sits -- long panels or tails "
    "hanging from the hips, and what covers the groin (armored briefs vs cloth "
    "panties), are armor if they have the same plating, edges or spikes as the armor, not "
    "cloth; never merge separate pieces into one "
    "'bodysuit' or 'suit' (armour cups on the chest, a hip piece and gauntlets are three pieces "
    "with skin between them); a skin-tight suit can be the colour of skin: look at the neck and "
    "wrists -- a drawn line round the neck (a collar edge) where the skin of the face stops, or a "
    "body drawn smooth with no nipples or navel, is a skin-tight bodysuit; name it as one, "
    "never as bare skin; then which body areas are bare skin (e.g. bare midriff from "
    "under the chest pieces down to the hip piece); the footwear with its height (ankle, knee or thigh-high), "
    "heel (flat, block, stiletto) and material; and everything "
    "held or attached (weapons, props, wings, tails) with its shape and size. If the same "
    "character is shown twice "
    "(front and back views), say so once. Then the background in a few words (e.g. 'plain "
    "white background'). Join with '; '. Nothing else.")
# The face is named for a flat drawing (eye colour, face shape: the owner asked for the
# face kept closer) but not for a 3D cartoon, whose «round nose, fleshy face» kept the
# render a doll on every wording tried: a 3D source is described a second time without.
_FACE_2D = ("facial hair, glasses, eye colour, the expression and character the face shows "
            "(e.g. cold half-closed eyes, a confident smirk, a raised brow), how mature and hard or "
            "soft the face is (sharp cheekbones, narrow eyes, thin lips), whether the face has a "
            "marking, tattoo or face paint and what it looks like (never say on which side or "
            "where: picture 1 shows that), and the face as it "
            "would look on a real person (no cartoon sizes like 'large eyes'); ")
_FACE_3D = "facial hair and glasses (nothing else about the face); "


def _expand(ctx, words: str) -> tuple:
    """(target, style paragraph) for free words; falls back to the words themselves."""
    import llm
    from utils import safe_json_from_llm
    try:
        raw = text_model(ctx)(ctx, _EXPAND_SYS, words, temperature=0.2, max_tokens=400) or ""
        d = safe_json_from_llm(raw, ["target", "style"]) or {}
    except Exception:
        logger.exception("style: expanding %r failed", words[:80])
        d = {}
    target = d.get("target") if d.get("target") in ("drawn", "photo", "other") else ""
    if not target:          # no answer: the edit classifier's word on whether it is a style
        from image_router import classify_edit_intent
        target = "drawn" if classify_edit_intent(words, ctx=ctx) == "style_transfer" else "other"
    style = " ".join(str(d.get("style") or "").split())
    # Its own question: inside the JSON, «an epic fantasy painting of the cat riding a dragon» stayed
    # a style, got «keep the same composition», and the picture came back as itself repainted
    # (bench/restyle_target_live.py).
    if target in ("drawn", "photo") and yes_no(ctx)(
            "{text}\n\nThis asks for a change to a picture. Does it change WHAT the picture shows -- "
            "a new place, an action or fight, other clothes, other people or animals added -- and "
            "not only HOW the same picture is drawn or photographed? A style name alone is HOW, "
            "even one with its own look (cyberpunk neon, steampunk, noir, a painter's manner).", words):
        target = "other"
    return target, (style or words)


def _sharp_enough(image_path: str) -> str:
    """A small sheet's face is ~40 px: the describer read a cold, marked face as «soft,
    rounded, no markings» and fins as round horns. It reads an enlarged copy."""
    import tempfile
    from PIL import Image
    try:
        im = Image.open(image_path)
        k = 1600 / max(im.size)
        if k <= 1.2:
            return image_path
        big = im.convert("RGB").resize((int(im.width * k), int(im.height * k)), Image.LANCZOS)
        out = os.path.join(tempfile.gettempdir(), "style_who_" + os.path.basename(image_path))
        big.save(out)
        return out
    except Exception:
        return image_path


def _describe(ctx, image_path: str, face: str) -> str:
    from llm import analyze_image_with_llm
    return " ".join((vision_model(ctx)(
        ctx=ctx, image_path=_sharp_enough(image_path), user_text=_WHO_PROMPT.format(face=face),
        system_prompt="You are a precise, literal visual describer.") or "").split())


def _who(ctx, image_path: str) -> tuple:
    """(description, is_3d) of what is in the picture, for the photo instruction."""
    try:
        text = _describe(ctx, image_path, _FACE_2D)
        m = re.match(r"\s*MEDIUM:\s*(2D|3D)\W*", text, re.I)
        is_3d = bool(m and m.group(1).upper() == "3D")
        if is_3d:
            text = _describe(ctx, image_path, _FACE_3D)
            m = re.match(r"\s*MEDIUM:\s*(2D|3D)\W*", text, re.I)
    except Exception:
        logger.exception("style: describing who is in %s failed", image_path)
        return "", False
    text = text[m.end():] if m else text
    # The describer writes «large brown eyes» even when told not to; the words alone
    # pull a face back to the drawing.
    text = re.sub(r"\b(?:very\s+)?(?:large|big|huge|oversized|enormous)\s+((?:\w+\s+)?eyes)\b",
                  r"\1", text, flags=re.I)
    return text[:1200], is_3d


def build(ctx, instructions: str, image_path: Optional[str] = None) -> tuple:
    """(instruction, target, realism LoRA strength) for a restyle request."""
    import image_grounding
    key = style_presets.preset_for(instructions)
    if key:
        p = style_presets.STYLE_PRESETS[key]
        target, style, keeps_clothes = p["target"], p["prompt"], p.get("keeps_clothes", True)
    else:
        words = dep(ctx, "_english_instructions", image_grounding._english_instructions)(ctx, instructions) or instructions
        target, style = dep(ctx, "_expand", _expand)(ctx, words)
        keeps_clothes = True
    keep = _KEEP.format(clothes="the same clothing " if keeps_clothes else "")
    if target == "photo":
        who, is_3d = dep(ctx, "_who", _who)(ctx, image_path) if image_path else ("", False)
        lead = f"change the picture 1 to realistic photograph, {who}. " if who else ""
        tail = (" Each person keeps the same gender, age, hair and facial hair, at real-life "
                "human head-to-body proportions (a normal-sized head) with the same body build "
                "as described (bust, waist, hips, limbs as drawn), and every "
                "garment, armor piece, boot, weapon and prop named above stays, in the same place, "
                "the same size and the SAME COLOUR (painted dark armor never turns bare silver "
                "or chrome), as a real costume or prop; anything worn or held that is not named "
                "above keeps exactly its colour in picture 1 too (a white glove stays white).Each face keeps the expression and "
                "character described above, never younger, softer or cuter than drawn; any marking "
                "or line on the face is copied from picture 1 at exactly its spot and side, and "
                "nothing is drawn on the skin anywhere else. Nobody is undressed or dressed: a line "
                "round the neck where the face's skin stops, or a body drawn smooth with no nipples "
                "or navel, is a skin-tight suit of exactly the skin tone drawn there, covering all "
                "that picture 1 covers (torso and arms too), with its collar seam at the neck, and "
                "stays one; skin is bare only where picture 1 draws it bare.") if who else ""
        return (lead + style + tail + keep, target,
                PHOTO_LORA_STRENGTH_3D if is_3d else PHOTO_LORA_STRENGTH)
    return style + keep, target, 0.0


def _new_scene(ctx, instructions: str) -> bool:
    """The people are put into a new scene, pose or action, rather than kept as they stand while
    something is changed or added. Kept, the source pins the pose: «cyber-Lenin punches a demon» came
    back Lenin in his portrait pose with demons hanging on him (live 2026-10-21)."""
    return not yes_no(ctx)(
        "{text}\n\nThis asks to change a picture of people. Do the people ALREADY in the picture keep "
        "the pose they have in it -- only something about them or around them changes (a hat, other "
        "clothes, another background, night, another man added beside them, however he stands) -- "
        "rather than being shown doing something new (fighting, dancing, surfing, riding)?",
        instructions, True)


def _heads(ctx, image_path: str) -> list:
    """Each face of the picture cropped with its hair, largest first (up to three), as files."""
    import tempfile
    from PIL import Image
    import identity_metrics
    try:
        boxes = dep(ctx, "face_boxes", identity_metrics.face_boxes)(image_path, pad=0.6)
        if not boxes:
            return []
        # a face far smaller than the main one is background: the advert's woman on a screenshot
        # of Lenin's page rode along as a second person
        area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
        boxes = [b for b in boxes if area(b) >= 0.25 * area(boxes[0])][:3]
        im = Image.open(image_path).convert("RGB")
        out = []
        for i, box in enumerate(boxes):
            p = os.path.join(tempfile.mkdtemp(prefix="scene_head_"), f"head{i}.png")
            im.crop(tuple(int(v) for v in box)).save(p)
            out.append(p)
        return out
    except Exception:
        logger.warning("style: cropping the faces of %s failed", image_path, exc_info=True)
        return []


def _scene_from_faces(ctx, image_mod, image_path: str, instructions: str, heads: list, **kw):
    """A new scene drawn with the faces as references: the picture no longer pins pose and frame
    (with references Klein runs without the Consistency LoRA), and a face may turn to any angle.
    4 of 4 takes fought where the plain edit kept the portrait pose (tmp face_ref_ab, 2026-10-21)."""
    refs = ", ".join(f"picture {i}" for i in range(2, len(heads) + 2))
    prompt = (f"Draw a completely new full picture, nothing of picture 1 kept but the people's faces: "
              f"{instructions.strip()} The people keep the faces shown in {refs}, seen at whatever angle "
              "and with whatever expression the scene needs.")
    logger.info("style: new scene with %d face reference(s): %r", len(heads), prompt[:160])
    once = dep(ctx, "_edit_image_with_firered_once", image_mod._edit_image_with_firered_once)
    return once(ctx, image_path, prompt, reference_paths=heads, **kw)


def restyle(ctx, image_path: str, instructions: str, *, seed=None, timeout=None):
    """Render the restyle; returns the saved path or None. A request that is not a
    style at all (target "other": relight, season, a filter) is edited as written."""
    import image as image_mod
    image_mod = dep(ctx, "image", image_mod)
    instr, target, strength = build(ctx, instructions, image_path)
    kw = {k: v for k, v in (("seed", seed), ("timeout", timeout)) if v is not None}
    if target == "other" and not style_presets.preset_for(instructions):
        heads = _heads(ctx, image_path) if _new_scene(ctx, instructions) else []
        if heads:
            return _scene_from_faces(ctx, image_mod, image_path, instructions, heads, **kw)
        return image_mod.edit_image_with_firered(ctx, image_path, instructions, **kw)
    neg = {"drawn": NEG_DRAWN, "photo": NEG_PHOTO}.get(target, NEG_OTHER)
    logger.info("style: target=%s instr=%r", target, instr[:160])
    with image_mod.firered_sampler(STYLE_STEPS, STYLE_CFG, lightning=False, negative=neg,
                                  upsize=True), \
            image_mod.firered_extra_lora(PHOTO_LORA if target == "photo" else None,
                                         strength):
        out = image_mod.edit_image_with_firered(ctx, image_path, instr, **kw)
        if out and target == "photo" and strength >= PHOTO_LORA_STRENGTH_3D:
            # A 3D cartoon comes out of one pass half-real (a doll's face on a real
            # body); a second pass over that result finishes it, one pass did not
            # on any wording tried (bench/style_ab.py, tmp pixar runs 2026-10-12).
            out = image_mod.edit_image_with_firered(ctx, out, instr, **kw) or out
        return out

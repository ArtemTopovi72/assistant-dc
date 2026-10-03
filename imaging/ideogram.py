"""Ideogram 4 drawing mode.

Ideogram 4 does NOT take a plain prose prompt. Its text encoder expects a
*structured JSON caption* that deconstructs the picture into a background plus a
list of elements, each with an optional bounding box saying WHERE that element
goes. Feeding it a bare sentence makes the model emit its grey "Image blocked by
safety filter" card instead of a picture (verified 2026-07-22 — the same request
as a structured caption rendered correctly).

So this module has two halves:

  1. `build_caption` / `bbox` — the exact caption format the model wants,
     mirroring KJNodes' Ideogram4PromptBuilderKJ (0-1000 grid, [ymin, xmin,
     ymax, xmax]).
  2. `plan_caption` — the DRAWING part: the assistant's own LLM reads the user's
     request and lays the scene out, choosing what goes where. That layout step
     is the whole point of this engine: the model composes rather than guesses.

Public entry point: `generate(ctx, prompt, ...) -> saved image path or None`.
"""
import json
import logging
import random
import re
from pathlib import Path
from typing import Optional

from config import (IDEOGRAM_CFG, IDEOGRAM_POLISH_CFG,
                    IDEOGRAM_POLISH_STEPS, IDEOGRAM_SHIFT, IDEOGRAM_STEPS,
                    IDEOGRAM_TURBO_LORA, IDEOGRAM_PHOTO_LORA, IDEOGRAM_PHOTO_LORA_STRENGTH, IDEOGRAM_STEPS_TURBO, IDEOGRAM_CFG_TURBO,
                    IDEOGRAM_TURBO)
# The ComfyUI transport used to be reached with a function-local
# `from image import _submit_and_poll`, placed there only to dodge the
# image <-> ideogram import cycle. It now lives in its own module, so this is a
# plain top-level import and the cycle is gone.
import comfy_client
# Lettering geometry. This used to be a function-local `from draw_agent import
# text_geometry, MIN_TEXT_H`, deferred only because draw_agent imports this
# module. The rule is arithmetic over a string and a box, so it now lives in a
# module neither of us owns and the cycle is gone.
from text_layout import text_geometry, MIN_TEXT_H as MIN_LETTER_H

logger = logging.getLogger("assistant.ideogram")


class ContentRefused(RuntimeError):
    """Ideogram 4 declined the prompt and drew its refusal card instead.

    The model itself renders a flat grey 'Image blocked by safety filter' image —
    it is not a ComfyUI node and there is no switch for it. The job still reports
    success, so without this the assistant would post the grey rectangle as a
    finished picture. Callers should report the refusal, NOT retry the same prompt
    on a different model.
    """


# The refusal card is a flat GREY frame. A frame darker than this is not the
# card, it is a render that came out black -- see is_black_frame.
REFUSAL_MEAN_FLOOR = 40.0
# Below this mean luminance a render is a black frame: the model collapsed
# (measured 2026-09-14: mean 1-7, std 2-17) rather than drew a dark scene
# (a real night scene measured mean 25-50 with std 50+ on the same day).
BLACK_MEAN_CEILING = 12.0


def _frame_stats(path: str) -> tuple:
    """(mean, std, unique colours) of the picture."""
    import numpy as np
    from PIL import Image
    arr = np.asarray(Image.open(path).convert("RGB")).astype("float32")
    return float(arr.mean()), float(arr.std()), len(np.unique(arr.reshape(-1, 3), axis=0))


def is_refusal_card(path: str) -> bool:
    """Detect the refusal card: a near-flat grey frame with a line of text.

    Measured (2026-07-22): refusal cards land at std≈9-11 with ~5k unique colours;
    real renders sit at std≈50-70 with 150k+. The gap is wide enough that a plain
    threshold is reliable, and a genuinely flat legitimate image (a solid colour
    study) is not something this engine is asked for.

    Live 2026-09-14: seven renders in a row were reported as refusals and the
    user saw «Не получилось нарисовать» for each. None was the card -- every one
    was a BLACK frame (mean 1-7). The card is grey, so the mean is checked too;
    a black frame is a different failure and is handled by is_black_frame.
    """
    try:
        mean, std, colours = _frame_stats(path)
        if std < 20.0 and colours < 20000 and mean >= REFUSAL_MEAN_FLOOR:
            logger.warning("Ideogram refusal card detected (mean=%.1f, std=%.1f, colours=%d): %s",
                           mean, std, colours, path)
            return True
        return False
    except Exception:
        logger.exception("refusal-card check failed — treating the image as real")
        return False


def is_black_frame(path: str) -> bool:
    """True when the render collapsed to a (near-)black frame.

    Not a refusal and not a dark scene: the model produced nothing. Seen live
    with captions carrying ten narrow vertical lettering strips. The caller
    re-rolls the seed once rather than delivering a black rectangle or
    reporting a refusal that never happened.
    """
    try:
        mean, std, colours = _frame_stats(path)
        if mean < BLACK_MEAN_CEILING and std < 20.0:
            logger.warning("Ideogram black frame (mean=%.1f, std=%.1f, colours=%d): %s",
                           mean, std, colours, path)
            return True
        return False
    except Exception:
        return False


# Layout / style / caption model, extracted to ideogram_layout.py.
# Re-exported so every existing ideogram.<name> caller and suite keeps working.
# That layer is stdlib-only and calls nothing here, so there is no cycle and no
# proxy seam to maintain.
from ideogram_layout import (
    _looks_like_layout, _repair_json, _extract_json, _iter_json_objects, named_style,
    _DEFAULT_PHOTO_STYLE, wants_realism, apply_style_floor, bbox, element, blank_layout,
    normalize_layout, layout_to_caption, caption_to_layout, hex_palette, build_caption,
    merged_layout, filled_layout,
)
WORKFLOW_IDEOGRAM_PATH = Path(__file__).resolve().parents[1] / "workflows" / "image" / "workflow_ideogram4.json"

# The last composition this module actually submitted, so the Storyboard tab can
# pick up what the AGENT drew and let the user move the boxes and repaint it.
# `LAST_SERIAL` ticks on every submit — that is what the tab polls, since a caption
# can legitimately be submitted twice unchanged (a reroll).
LAST_CAPTION = None
LAST_PROMPT = ""
LAST_IMAGE = None
LAST_SERIAL = 0

# Node ids inside workflow_ideogram4.json (kept in one place so a graph edit is
# a one-line change here rather than a hunt through string literals).
NODE_PROMPT = "167"          # CLIPTextEncode  <- the structured caption
NODE_NOISE = "165"           # RandomNoise     <- seed
NODE_LATENT = "160"          # EmptyFlux2LatentImage <- width/height
NODE_SCHEDULER = "190"       # BasicScheduler  <- steps
NODE_GUIDER = "182"          # DualModelGuider <- cfg
NODE_SHIFT = "198"           # ModelSamplingAuraFlow <- shift
NODE_UNET_COND = "166"       # UNETLoader, CONDITIONAL branch
NODE_SAMPLER = "161"         # SamplerCustomAdvanced (main pass)
NODE_SAMPLER_SELECT = "163"  # KSamplerSelect
NODE_DECODE = "162"          # VAEDecode <- reads the LAST sampler in the chain


def attach_lora(workflow: dict, lora_name: str, strength: float = 1.0) -> None:
    """Apply a character LoRA to the CONDITIONAL branch only.

    The Ideogram graph carries TWO UNETLoaders: 166 holds the conditional model
    and 181 the unconditional one, which DualModelGuider consumes as
    `model_negative`. The generic image_lora.inject_lora attaches to every model
    loader it finds, which here would put the character into the negative
    prediction as well -- and under CFG that prediction is SUBTRACTED, so the
    adapter would be working against itself. Hence a targeted injection rather
    than the shared helper.
    """
    if not lora_name:
        return
    if NODE_UNET_COND not in workflow:
        raise ValueError("Ideogram workflow has no conditional UNET loader")
    new_id = "lora_" + NODE_UNET_COND
    # Rewire the consumers BEFORE adding the node, so it does not point at itself.
    for node in workflow.values():
        for key, val in list((node.get("inputs") or {}).items()):
            if isinstance(val, list) and len(val) == 2 and str(val[0]) == NODE_UNET_COND:
                node["inputs"][key] = [new_id, 0]
    workflow[new_id] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {"lora_name": lora_name,
                   "strength_model": float(strength),
                   "model": [NODE_UNET_COND, 0]},
    }


def attach_turbo_lora(workflow: dict, lora_name: str, strength: float = 1.0) -> None:
    """Apply the speed-turbo LoRA to the CONDITIONAL branch, same targeting as
    attach_lora above (never the negative/unconditional branch).

    Uses a DIFFERENT node id ("turbo_" + NODE_UNET_COND) so a character LoRA
    from attach_lora can still be layered on top: call this one FIRST -- its
    generic id-string rewire (matching on NODE_UNET_COND) means a later
    attach_lora() call correctly re-splices itself between the base model and
    this node, chain ending up base -> character LoRA -> turbo LoRA -> guider.
    """
    if not lora_name:
        return
    if NODE_UNET_COND not in workflow:
        raise ValueError("Ideogram workflow has no conditional UNET loader")
    new_id = "turbo_" + NODE_UNET_COND
    for node in workflow.values():
        for key, val in list((node.get("inputs") or {}).items()):
            if isinstance(val, list) and len(val) == 2 and str(val[0]) == NODE_UNET_COND:
                node["inputs"][key] = [new_id, 0]
    workflow[new_id] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {"lora_name": lora_name,
                   "strength_model": float(strength),
                   "model": [NODE_UNET_COND, 0]},
    }


def ensure_trigger(caption: dict, trigger: str) -> dict:
    """Guarantee the adapter's trigger word actually reaches the caption.

    A character LoRA only fires when its trigger is in the text the encoder
    sees. On the old model path the trigger is glued onto the prompt string and
    that string IS the conditioning, so it always arrives. Here the prompt is
    first rewritten into JSON by the planner, and the planner is free to drop a
    word it reads as a typo -- leaving the adapter attached but never invoked,
    which renders a stranger under that person's name and reports success. So
    the word is put back explicitly rather than hoped for.
    """
    if not trigger or not caption:
        return caption
    comp = caption.get("compositional_deconstruction") or {}
    elements = comp.get("elements") or []
    low = trigger.lower()

    # NEVER inside a lettering element. Ideogram renders `text` as legible
    # letters on the picture, so a trigger that lands there is drawn ON the
    # character instead of conditioning them -- the word appearing across a
    # shirt, which is what it looked like live. Strip it wherever it got in,
    # whoever put it there, before deciding whether it is already present.
    _trig = re.compile(r"\b%s\b" % re.escape(trigger), re.IGNORECASE)
    for el in elements:
        t = str(el.get("text") or "")
        if t and low in t.lower():
            cleaned = re.sub(r"\s{2,}", " ", _trig.sub("", t))
            el["text"] = re.sub(r"\s*,\s*,", ",", cleaned).strip(" ,")

    # In TRAINING the trigger sits in high_level_description AND in the subject
    # element's desc. Rendering with it in only one of the two is a different
    # conditioning from the one the adapter was fitted on.
    hl = str(caption.get("high_level_description") or "")
    if hl and low not in hl.lower():
        caption["high_level_description"] = "%s, %s" % (trigger, hl.strip())

    if any(low in str(el.get("desc") or "").lower() for el in elements):
        return caption
    if not elements:                      # no subject box: put it on the scene
        comp["background"] = "%s, %s" % (trigger, comp.get("background") or "")
        return caption
    subj = elements[_subject_index(elements)]
    subj["desc"] = "%s, %s" % (trigger, str(subj.get("desc") or "").strip())
    return caption


# Nouns that make an element a PERSON. The trigger conditions a face, so it has
# to land on the person; the planner decides the element ORDER, and in a scene
# like "the man robs a bank" it happily puts the building first. Gluing the
# trigger to elements[0] then trains the caption on a bank, and the render comes
# back with a stranger -- reported as success, because the adapter was attached.
_PERSON = ("man", "woman", "person", "boy", "girl", "guy", "lady", "male",
           "female", "figure", "character", "cowboy", "soldier", "robber",
           "bandit", "outlaw", "мужчина", "женщина", "человек", "парень")


def _subject_index(elements: list) -> int:
    """Which element is the character? Person words first, then the biggest box.

    Size is the fallback rather than the rule: a bank facade is usually the
    largest thing in frame, so it only decides when nothing names a person at
    all -- and then the largest element is the best guess available.
    """
    for i, el in enumerate(elements):
        low = str(el.get("desc") or "").lower()
        if any(re.search(r"\b%s\b" % w, low) for w in _PERSON):
            return i
    best, best_area = 0, -1.0
    for i, el in enumerate(elements):
        box = el.get("bbox") or []
        if len(box) == 4:
            try:
                area = (float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1]))
            except (TypeError, ValueError):
                continue
            if area > best_area:
                best, best_area = i, area
    return best


def add_polish_pass(workflow: dict, steps: int, polish: int, cfg: float) -> bool:
    """Split the schedule into a high-guidance pass and a low-guidance polish.

    Ideogram 4's own presets never sample at a constant guidance weight: every
    one of them runs the bulk of the steps at gw=7 and finishes with a handful
    at gw=3. Reproducing that in ComfyUI means cutting the sigma list in two and
    running the tail through a second guider, because DualModelGuider takes one
    cfg for the whole run.

    Returns True if the workflow was rewired, False if the request was a no-op
    (which keeps the single-pass graph byte-identical to what it always was).
    """
    if polish <= 0 or polish >= steps:
        return False
    split = steps - polish
    workflow["900"] = {"class_type": "SplitSigmas",
                       "inputs": {"sigmas": [NODE_SCHEDULER, 0], "step": split}}
    workflow["901"] = {"class_type": "DisableNoise", "inputs": {}}
    # A copy of the main guider with the polish cfg, so both passes see the same
    # models and the same conditioning -- only the guidance weight differs.
    main = workflow[NODE_GUIDER]
    workflow["902"] = {"class_type": main["class_type"],
                       "inputs": dict(main["inputs"], cfg=cfg)}
    workflow["903"] = {"class_type": "SamplerCustomAdvanced",
                       "inputs": {"noise": ["901", 0], "guider": ["902", 0],
                                  "sampler": [NODE_SAMPLER_SELECT, 0],
                                  "sigmas": ["900", 1],
                                  "latent_image": [NODE_SAMPLER, 0]}}
    # The first pass now stops at the split, and the decoder reads the polish.
    workflow[NODE_SAMPLER]["inputs"]["sigmas"] = ["900", 0]
    workflow[NODE_DECODE]["inputs"]["samples"] = ["903", 0]
    return True


NODE_SAVE = "200"            # SaveImage


# ---------------------------------------------------------------------------
# The drawing step: lay the scene out with the LLM
# ---------------------------------------------------------------------------
_PLANNER_PROMPT = """You are a scene compositor for an image model that draws from a \
structured layout. Turn the user's request into a JSON layout.

Answer with ONLY a JSON object, no prose, no markdown fence, in this exact shape:
{
  "high_level_description": "one sentence describing the whole picture",
  "aesthetics": "mood/quality words",
  "lighting": "the light in the scene",
  "photo": "camera/lens words if photographic, else empty string",
  "medium": "photography | oil painting | 3d render | illustration | ...",
  "background": "the setting behind everything, always filled in",
  "elements": [
    {"desc": "what this thing is, described concretely", "x": 0.1, "y": 0.4, "w": 0.5, "h": 0.5},
    {"desc": "a sign", "text": "OPEN", "x": 0.6, "y": 0.1, "w": 0.3, "h": 0.15}
  ]
}

Rules for the boxes:
- x, y, w, h are fractions of the frame, 0-1, with x=0,y=0 at the TOP-LEFT.
- x+w and y+h must stay <= 1.0.
- Place 2-6 elements. The main subject gets the largest, most central box.
- Boxes may overlap when things really overlap (a hand on a table).
- Respect any placement the user asked for ("on the left", "in the sky").
- Describe each element on its own, as if the others did not exist.
- Describe what IS there, never what is absent: "no whiskers" or "without a hat" names the
  thing and the model draws it. "кот без усов" is "a smooth-faced cat"; leave absent things out.
- A surface the request puts things ON or IN ("a vase on the table", "books on a
  shelf") is its own element, boxed under what it carries -- never dropped.

Rules for lettering — this is where the model most often produces nonsense:
- Any word that must appear IN the picture gets its own element with a "text"
  field holding the exact string, nothing else. Lettering mentioned only in a
  "desc" comes out as scribble.
- Write the string exactly as it should be printed, including capitalisation.
  Keep it SHORT — one word or a few; long sentences render as gibberish.
- Words the user dictates to be written count whether quoted or not: a menu
  "Эспрессо 150, Капучино 220" is one text element PER LINE ("Эспрессо 150"),
  and the board they are on is the background. "the menu items" with no text
  came out as scribble.
- The "desc" of a text element describes the SURFACE and the lettering style —
  "a sign board with bold white block capitals", "the side of a police car with
  large white lettering" — never a rewording of the text itself.
- Size the box for the string: a text box must be at least 0.08 of the frame
  high, and roughly 0.6 x (number of characters) times as wide as it is high.
  "POLICE" (6 characters) in a box 0.10 high wants about 0.36 wide. A box that
  is too small or the wrong shape is exactly what squeezes the letters into an
  unreadable smear.
- One string per element. Two words in different places are two elements.
- Do not invent lettering the user did not ask for.

Rules for the style — this decides what KIND of picture comes out:
- The user's own words win. "realistic", "photo", "photorealistic", "реалистично",
  "фото" mean medium="photography": fill "photo" with camera/lens words, put
  "photorealistic" in "aesthetics", and leave "art_style" EMPTY.
- Only set "art_style" when the user actually asked for a drawn or painted look
  (cartoon, anime, watercolour, oil painting, 3d render, sketch). Setting it
  discards the "photo" field, so never set both.
- Never leave every style field empty. An empty style is not neutral — the model
  falls back to its own stylised look, which is wrong for any realism request.
- When the user says nothing about style, default to photography.
- A logo, icon, emblem, sticker or badge is graphic design, not a photograph:
  medium="flat vector graphic design", "photo" empty.

Write every field in English, whatever language the request is in — EXCEPT
"text". "text" is not a description of the picture, it is the lettering that
will be PAINTED INTO it: copy it exactly as the user wrote it, in their own
alphabet, and never translate or transliterate it. A request for a poster
reading «Добро пожаловать» must carry text "Добро пожаловать", not "Welcome".
Answer with ONE JSON object and nothing after it — no second copy, no commentary."""


# The last-rung prompt: the smallest ask that still produces a usable layout.
# Deliberately shows the shape instead of describing it — a small model copies a
# worked example far more reliably than it follows a specification.
_PLANNER_MINIMAL = """Break the scene into elements and place each one in the frame.

Reply with ONLY this JSON. No prose, no code fence, no explanation.

{"high_level_description": "one sentence describing the whole picture",
 "background": "what fills the frame behind everything",
 "elements": [
   {"desc": "a red tractor with black wheels", "x": 0.05, "y": 0.40, "w": 0.40, "h": 0.50},
   {"desc": "a wooden sign", "text": "FARM", "x": 0.55, "y": 0.05, "w": 0.30, "h": 0.12}
 ]}

x and y are the TOP-LEFT corner as a fraction of the frame; w and h are the size.
x+w and y+h must be at most 1. Use 2 to 6 elements. Add "text" ONLY for lettering
that must be painted in the picture. Copy the structure above exactly."""

_FRAC = {"type": "number", "minimum": 0, "maximum": 1}
# Decoded under this grammar the reply is the JSON object, never prose or a fence.
LAYOUT_SCHEMA = {"type": "object", "required": ["high_level_description", "background", "elements"],
                 "properties": {**{k: {"type": "string"} for k in (
                     "high_level_description", "aesthetics", "lighting", "photo", "medium", "background")},
                     "elements": {"type": "array", "minItems": 1, "maxItems": 8, "items": {
                         "type": "object", "required": ["desc", "x", "y", "w", "h"],
                         "properties": {"desc": {"type": "string"}, "text": {"type": "string"},
                                        "x": _FRAC, "y": _FRAC, "w": _FRAC, "h": _FRAC}}}}}


def _ask_planner(ctx, prompt: str, attempts: int = 2) -> Optional[dict]:
    """Run the planner, retrying once if the reply is not layout-shaped JSON.

    One malformed reply used to decide the whole picture: the fallback carries no
    style, and a styleless caption is what turns a realism request into a cartoon.
    A retry is far cheaper than silently drawing the wrong kind of image.
    """
    import llm
    last = None
    # Measured: google/gemma-4-26b-a4b-qat plans this correctly first time, while
    # the app's DEFAULT model (a 9B) returned no layout-shaped JSON on both
    # attempts and every storyboard silently fell back to a flat, box-less layout —
    # which is why "it edited without using the boxes". So the ladder now varies the
    # ASK, not just the temperature: a smaller model needs a shorter prompt, a
    # worked example, and more room, not the same 900-token squeeze twice.
    rungs = list(range(max(2, attempts, 3)))
    # Gemma 4 thinks on EVERY turn and cannot be told not to (the <think></think>
    # prefill is a Qwen lever and is dropped for Gemma), so the narrow rungs are
    # not a cheap first try on the house model — they are a guaranteed miss.
    # Measured on gemma-4-26b-a4b-qat: rung 0 spends its whole budget reasoning
    # (8.7k characters), returns nothing, and the ladder then jumps to the wide
    # rung anyway — about 45 s of GPU burnt before the real attempt starts, on
    # every single drawing. Start where this model can actually answer.
    #
    # The narrow rungs stay for everything else: a smaller non-Gemma model does
    # succeed at 900 tokens, and starting it at 6000 would be slower, not faster.
    try:
        import llm as _llm
        # Only while it thinks: with the default no-think prefill the full
        # prompt answers in 3-5 s and keeps the table under "на столе ваза"
        # and the style fields the minimal one drops (measured 2026-09-29).
        if (_llm._is_gemma4(getattr(ctx, "model_name", "") or "")
                and not getattr(ctx, "no_think", True)):
            logger.info("Ideogram planner: %s thinks unconditionally — starting at "
                        "the wide rung instead of burning the narrow ones",
                        getattr(ctx, "model_name", ""))
            rungs = [rungs[-1]]
    except Exception:
        pass          # unknown model: keep the full ladder, it is only slower
    i = -1
    while rungs:
        i = rungs.pop(0)
        system = _PLANNER_PROMPT
        user = (prompt or "").strip()
        if getattr(ctx, "reply_lang", "") == "ru" and not re.search("[а-яё]", user, re.I):
            # The request reached us translated: «вывеска для пекарни» was lettered BAKERY.
            user += "\n\n(The user writes Russian: lettering you make up yourself is in Russian.)"
        kw = {"temperature": 0.4, "max_tokens": 900, "prefill": "<think></think>"}
        if i == 1:
            kw = {"temperature": 0.1, "max_tokens": 1600, "prefill": "<think></think>"}
        elif i >= 2:
            # Last rung: the minimum viable ask. No style vocabulary, no prefill
            # (it is unreliable outside Qwen), a literal example of the shape, and
            # a budget that cannot truncate the JSON mid-object.
            #
            # 6000, not 2000: this model thinks on every turn and cannot be
            # stopped (the <think></think> prefill is deleted for Gemma), and it
            # was measured burning 7.4k-8.4k CHARACTERS of reasoning here. A
            # 2000-token ceiling is spent before the JSON is even started, so the
            # "last rung" could never have succeeded on the house model. Same
            # floor that fixed the research briefs.
            system = _PLANNER_MINIMAL
            kw = {"temperature": 0.0, "max_tokens": 6000, "prefill": None}
        res = llm.send_to_lm_studio(
            ctx,
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            tools=[], tool_choice="none", json_schema=LAYOUT_SCHEMA, **kw)
        data = _extract_json((res or {}).get("content") or "")
        if _looks_like_layout(data):
            if i:
                logger.info("Ideogram planner recovered on attempt %d", i + 1)
            return data
        last = data
        logger.warning("Ideogram planner attempt %d produced no layout-shaped JSON",
                       i + 1)
        # The model spent the entire budget thinking and never wrote the JSON.
        # Measured live on gemma-4-26b: 7.4k then 8.4k characters of pure
        # reasoning on rungs 1 and 2, ~90 s for nothing, because the retry was
        # the SAME squeeze. A budget below the model's thinking floor cannot be
        # fixed by repeating it — go straight to the widest rung.
        if (res or {}).get("reasoning_only") and rungs:
            logger.warning("Ideogram planner: %d chars of reasoning and no answer "
                           "— the budget is below this model's thinking floor; "
                           "skipping to the widest rung",
                           (res or {}).get("reasoning_chars", 0))
            rungs = [rungs[-1]]
    logger.error("Ideogram planner failed on every attempt — the picture will be "
                 "drawn from a FLAT layout with no editable boxes")
    return last


# Lettering the USER asked for, in either language. The planner is supposed to put
# these in a text element and often does not — a request for a neon sign reading
# "CAFE ROSA" came back as six elements with no `text` anywhere, so the whole
# lettering path (sizing, read-back, repair) was never entered.
_QUOTED_RE = re.compile(r'"([^"\n]{1,32})"|“([^”\n]{1,32})”|«([^»\n]{1,32})»'
                        r"|'([^'\n]{1,32})'|„([^“\n]{1,32})“")


# More quoted phrases than this and the text is a QUOTED CONVERSATION, not a
# lettering request. Live 2026-09-14: a forwarded chat with ten «...» quotes was
# handed to the character painter; every quote became a lettering strip, the
# caption carried thirteen elements, and Ideogram rendered a black frame.
MAX_REQUESTED_STRINGS = 4


def requested_strings(prompt: str) -> list:
    """The exact strings the user asked to see IN the picture, in order."""
    p = (prompt or "")
    out = []
    for m in _QUOTED_RE.finditer(p):
        s = next((g for g in m.groups() if g), "").strip()
        if s and any(ch.isalnum() for ch in s) and s not in out:
            out.append(s)
    # A quote only means LETTERING when the sentence says so: "draw a 'cat'"
    # must not have the word cat painted across the picture.
    if out:
        import intent
        if not intent.ask_yes("A user asked for this picture: {text}. Does the user ask for "
                              "written words (a sign, a label, a caption, an inscription) to "
                              "appear IN the picture?", p):
            return []
    if len(out) > MAX_REQUESTED_STRINGS:
        logger.info("Ideogram: %d quoted phrases in the prompt — treating them as "
                    "prose, not lettering", len(out))
        return []
    return out


# Surfaces lettering is normally painted on — used to attach a requested string to
# the element the user meant, rather than floating it over the middle of the scene.
_SURFACE_WORDS = ("sign", "board", "banner", "poster", "door", "window", "awning",
                  "wall", "plate", "placard", "screen", "label", "marquee", "shop",
                  "storefront", "facade", "front", "car", "truck", "van", "box",
                  "вывеск", "таблич", "двер", "окн", "стен", "плакат", "щит")


def ensure_text_elements(layout: dict, prompt: str) -> dict:
    """Make sure every string the user asked for is carried by a text element.

    The planner is instructed to do this and mostly does not: measured live, a
    request for a neon sign reading a specific name produced six elements and no
    `text` field at all, so nothing downstream knew the picture had lettering in
    it. Attaching the string to the surface it belongs on — or adding a properly
    sized element for it — is what turns the whole lettering path on.
    """
    layout = normalize_layout(layout, prompt)
    strings = requested_strings(prompt)
    if not strings:
        return layout
    els = layout["elements"]
    for el in els:      # the user's own casing: 'Зерно' came back as 'ЗЕРНО'
        el_text = str(el.get("text") or "").strip()
        el["text"] = next((s for s in strings if s.lower() == el_text.lower()), el.get("text"))
    # The planner often names the string in an element's desc («DLL files labeled
    # 'MSVCP140.dll'») without a `text` field: that element IS its carrier. Adding
    # a second one drew the label twice (live 10-03).
    for s in strings:
        if any(str(el.get("text") or "").strip().lower() == s.lower() for el in els):
            continue
        owner = next((el for el in els if not el.get("text")
                      and s.lower() in str(el.get("desc") or "").lower()), None)
        if owner is not None:
            owner["text"] = s
    have = [str(el.get("text") or "").strip().lower() for el in els]
    used = set()
    for s in strings:
        if s.lower() in have:
            continue
        target = None
        for el in els:
            if el.get("text") or id(el) in used:
                continue
            desc = str(el.get("desc") or "").lower()
            if any(word in desc for word in _SURFACE_WORDS):
                target = el
                break
        # Always a SEPARATE element: reshaping the surface itself to the string's
        # aspect turned a storefront into a 0.94x0.17 letterbox and destroyed the
        # composition. The lettering sits ON the surface instead.
        if target is not None:
            used.add(id(target))    # one string per surface: two on «fragments» doubled up
            w, h = text_geometry(s, min(0.14, max(target["h"] * 0.3, MIN_LETTER_H)))
            cx = target["x"] + target["w"] / 2
            y = target["y"] + target["h"] * 0.12
            desc = f'lettering on {str(target.get("desc") or "the sign").strip()}'
            logger.info("Ideogram: the planner dropped the requested lettering %r — "
                        "placing it on %r", s, str(target.get("desc"))[:40])
        else:
            w, h = text_geometry(s, 0.12)
            cx, y = 0.5, 0.12
            desc = f"a sign carrying the lettering “{s}”"
            logger.info("Ideogram: the planner dropped the requested lettering %r — "
                        "added an element for it", s)
        w, h = min(w, 1.0), min(h, 1.0)
        els.append({"desc": desc, "text": s,
                    "x": max(0.0, min(cx - w / 2, 1.0 - w)),
                    "y": max(0.0, min(y, 1.0 - h)), "w": w, "h": h})
        have.append(s.lower())
    return normalize_layout(layout, prompt)


def plan_layout(ctx, prompt: str) -> dict:
    """Ask the model to lay the scene out, as an EDITABLE layout (fractional boxes).

    Kept separate from `plan_caption` so the storyboard editor can show, move and
    reword the boxes before anything is rendered.
    """
    fallback = lambda: ensure_text_elements(
        apply_style_floor(blank_layout(prompt), prompt), prompt)
    if ctx is None:
        return fallback()
    try:
        data = _ask_planner(ctx, prompt)
        if not isinstance(data, dict):
            logger.warning("Ideogram planner returned no JSON — using the flat layout")
            return fallback()
        layout = normalize_layout(data, prompt)
        if not layout["elements"]:
            logger.warning("Ideogram planner produced no usable elements — flat layout")
            return fallback()
        # "three tall palm trees" in one box render as one merged mass, as in an edit.
        from draw_agent import split_counted_boxes, merge_text_column   # draw_agent imports this module
        layout, _ = split_counted_boxes(merge_text_column(layout), [])
        return ensure_text_elements(apply_style_floor(layout, prompt), prompt)
    except Exception:
        logger.exception("Ideogram planning failed — using the flat layout")
        return fallback()


def plan_caption(ctx, prompt: str) -> dict:
    """Ask the model to lay the scene out. Falls back to a single full-frame
    element built from the prompt itself, so a planner failure still draws
    something sensible rather than nothing."""
    # Built through the layout so the style floor applies to the fallback too — a
    # bare build_caption() here carried no style at all. Built LAZILY: eagerly
    # meant the floor's "no style, applying the default" warning was logged on
    # every realism prompt, including the ones where the planner did fine.
    def fallback():
        return layout_to_caption(apply_style_floor(blank_layout(prompt), prompt))
    if ctx is None:
        return fallback()
    try:
        data = _ask_planner(ctx, prompt)
        if not isinstance(data, dict):
            logger.warning("Ideogram planner returned no JSON — using the flat caption")
            return fallback()
        # Routed through the SAME layout pipeline plan_layout uses (normalize_layout
        # -> apply_style_floor -> ensure_text_elements -> layout_to_caption) rather
        # than building caption elements by hand. The hand-built version used to
        # skip ensure_text_elements entirely, so a requested lettering string the
        # planner dropped (e.g. a neon sign's exact wording) silently never made it
        # into the picture on THIS path, even though the storyboard/plan_layout path
        # already had a backstop for exactly that — found during a refactor pass
        # that noticed plan_layout and plan_caption had quietly diverged.
        layout = normalize_layout(data, prompt)
        if not layout["elements"]:
            logger.warning("Ideogram planner produced no usable elements — flat caption")
            return fallback()
        layout = ensure_text_elements(apply_style_floor(layout, prompt), prompt)
        return layout_to_caption(layout)
    except Exception:
        logger.exception("Ideogram planning failed — using the flat caption")
        return fallback()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
IDEOGRAM_MIN_SIDE = 256
IDEOGRAM_MAX_SIDE = 2048
IDEOGRAM_MAX_RATIO = 6.0


def _snap16(n: int) -> int:
    """Ideogram 4 wants multiples of 16, 256-2048 per side (published range).

    The old 64-4096 window was a guess: below 256 and above 2048 the model is
    outside what it was trained on — the high end is where the doubled horizons
    and second heads come from.
    """
    return max(IDEOGRAM_MIN_SIDE,
               min(IDEOGRAM_MAX_SIDE, int(round(n / 16)) * 16))


def ideogram_size(width: int, height: int) -> tuple:
    """Put a requested size inside Ideogram's supported envelope.

    Beyond 6:1 the aspect is unsupported, so the LONG side is pulled in rather
    than stretching the short one — that keeps the requested framing instead of
    silently turning a banner into a landscape.
    """
    w, h = _snap16(width), _snap16(height)
    ratio = max(w, h) / float(min(w, h))
    if ratio > IDEOGRAM_MAX_RATIO:
        if w > h:
            w = _snap16(int(h * IDEOGRAM_MAX_RATIO))
        else:
            h = _snap16(int(w * IDEOGRAM_MAX_RATIO))
        logger.warning("Ideogram: aspect %.1f:1 exceeds the supported %.0f:1 — "
                       "clamped to %dx%d", ratio, IDEOGRAM_MAX_RATIO, w, h)
    return w, h


def is_photo_caption(caption: dict) -> bool:
    """A photograph: the planner filled style.photo (art_style and photo are exclusive)."""
    st = (caption or {}).get("style_description") or {}
    if not isinstance(st, dict):         # a planner may hand back a plain string
        return False
    return bool(str(st.get("photo") or "").strip()) and not str(st.get("art_style") or "").strip()


def generate(ctx, prompt: str, *, width: int = 1024, height: int = 1024,
             seed: Optional[int] = None, steps: Optional[int] = None,
             cfg: Optional[float] = None, caption: Optional[dict] = None,
             lora_name: Optional[str] = None, lora_strength: float = 1.0,
             trigger: Optional[str] = None,
             timeout: int = 1900, on_progress=None) -> Optional[str]:
    """Draw `prompt` with Ideogram 4. Returns the saved image path, or None.

    Pass `caption` to skip the planner and supply the layout yourself.
    """
    if not (prompt or "").strip() and not caption:
        logger.error("ideogram.generate: empty prompt — refusing to submit")
        return None
    if not WORKFLOW_IDEOGRAM_PATH.exists():
        logger.error("Ideogram workflow missing: %s", WORKFLOW_IDEOGRAM_PATH)
        return None
    try:
        with open(WORKFLOW_IDEOGRAM_PATH, "r", encoding="utf-8") as fh:
            workflow = json.load(fh)
    except Exception as exc:
        logger.error("Failed to load the Ideogram workflow: %s", exc)
        return None

    caption = caption or plan_caption(ctx, prompt)
    caption = ensure_trigger(caption, trigger)
    global LAST_CAPTION, LAST_PROMPT, LAST_SERIAL, LAST_IMAGE
    LAST_CAPTION, LAST_PROMPT, LAST_SERIAL = caption, (prompt or "").strip(), LAST_SERIAL + 1
    # Compact separators + ensure_ascii=False are what the published guide asks
    # for: the training captions are serialised this way, and \uXXXX escapes are
    # a documented verifier warning. indent=4 fed the encoder whitespace the
    # model never saw in training.
    caption_json = json.dumps(caption, separators=(",", ":"), ensure_ascii=False)
    n_elements = len(caption.get("compositional_deconstruction", {}).get("elements", []))
    _w, _h = ideogram_size(width, height)
    logger.info("Ideogram caption: %d element(s), %dx%d", n_elements, _w, _h)
    logger.debug("Ideogram caption JSON:\n%s", caption_json)

    workflow[NODE_PROMPT]["inputs"]["text"] = caption_json
    workflow[NODE_NOISE]["inputs"]["noise_seed"] = (
        seed if seed and seed > 0 else random.randint(1, 999_999_999))
    workflow[NODE_LATENT]["inputs"]["width"] = _w
    workflow[NODE_LATENT]["inputs"]["height"] = _h
    # An explicit steps/cfg means the caller wants the plain, non-distilled
    # model at that exact setting -- ostris/ideogram_4_turbotime_lora is
    # trained for ~2-4 step, near-CFG-free sampling, and only applies as the
    # default when nobody asked for a specific step/cfg (2026-09-20: 35-40%
    # faster across 3 varied scenes, no quality loss on lettering or a crowd
    # scene; see IDEOGRAM_TURBO_LORA in config.py for the portrait-glitch fix).
    turbo = IDEOGRAM_TURBO and steps is None and cfg is None
    n_steps = IDEOGRAM_STEPS_TURBO if turbo else max(1, min(int(steps or IDEOGRAM_STEPS), 100))
    workflow[NODE_SCHEDULER]["inputs"]["steps"] = n_steps
    workflow[NODE_GUIDER]["inputs"]["cfg"] = (
        IDEOGRAM_CFG_TURBO if turbo else max(0.0, min(float(cfg or IDEOGRAM_CFG), 30.0)))
    workflow[NODE_SHIFT]["inputs"]["shift"] = float(IDEOGRAM_SHIFT)
    if turbo:
        attach_turbo_lora(workflow, IDEOGRAM_TURBO_LORA, 1.0)
        if IDEOGRAM_CFG_TURBO == 1.0:
            # cfg 1 needs no unconditional branch: one-model guider, and the
            # 9 GB uncond model is never loaded. A/B 2026-10-02 (bench/
            # ideogram_int8_ab.py app vs turbo4): same pictures, 4-9 s vs 7-13 s.
            g = workflow[NODE_GUIDER]["inputs"]
            model = g["model"]
            if str(model[0]) == "184":      # CFGOverride: a late cfg bump, moot at cfg 1
                model = workflow["184"]["inputs"]["model"]
            workflow[NODE_GUIDER] = {"class_type": "CFGGuider", "inputs": {
                "cfg": 1.0, "model": model, "positive": g["positive"], "negative": g["positive"]}}
            for k in ("181", "184", "159"):
                workflow.pop(k, None)
        logger.info("Ideogram: turbo LoRA, %d steps @ cfg %.1f", n_steps, IDEOGRAM_CFG_TURBO)
    if not lora_name and is_photo_caption(caption) and IDEOGRAM_PHOTO_LORA:
        # Ideogram's known plastic skin: Lenovo UltraReal on photographs only
        # (A/B 10-02 runtime/ideogram_int8_ab/skin.jpg: pores and film light,
        # same face, same lettering, same 40 s). A character LoRA keeps its own.
        lora_name, lora_strength = IDEOGRAM_PHOTO_LORA, IDEOGRAM_PHOTO_LORA_STRENGTH
    if lora_name:
        attach_lora(workflow, lora_name, lora_strength)
        logger.info("Ideogram LoRA attached to the conditional branch: %s @ %.2f",
                    lora_name, lora_strength)
    # The turbo LoRA is trained for one constant low/no-cfg pass -- bolting on
    # the normal 2-step, cfg=3.0 polish tail fights that and is the traced
    # cause of a colour-block glitch seen on a face-heavy render, so it is
    # skipped whenever turbo is engaged.
    if not turbo and add_polish_pass(workflow, n_steps, int(IDEOGRAM_POLISH_STEPS),
                                     float(IDEOGRAM_POLISH_CFG)):
        logger.info("Ideogram: %d steps @ cfg %.1f + %d polish @ cfg %.1f",
                    n_steps - IDEOGRAM_POLISH_STEPS,
                    float(cfg or IDEOGRAM_CFG), int(IDEOGRAM_POLISH_STEPS),
                    float(IDEOGRAM_POLISH_CFG))

    # Called through the module rather than bound with `from ... import` so there
    # is exactly ONE place to intercept the render: patching comfy_client is seen
    # by every caller. A bound name would have to be patched per module.
    # exclusive: the caption is already planned by the time we get here, so this
    # job needs nothing but the GPU -- and Ideogram is the largest model set we
    # load, the one that suffers most from sharing the card with the chat model.
    result = comfy_client._submit_and_poll(ctx, workflow, timeout=timeout,
                                           label="ideogram4",
                                           on_progress=on_progress,
                                           exclusive=True)
    if result and is_black_frame(result):
        # One re-roll on a fresh seed. A second black frame is delivered as it
        # is: the judge / the user can see it, and a loop here would just burn
        # the card on a caption the model cannot draw.
        workflow[NODE_NOISE]["inputs"]["noise_seed"] = random.randint(1, 999_999_999)
        logger.warning("Ideogram: black frame — re-rolling once on seed %d",
                       workflow[NODE_NOISE]["inputs"]["noise_seed"])
        retry = comfy_client._submit_and_poll(ctx, workflow, timeout=timeout,
                                              label="ideogram4",
                                              on_progress=on_progress,
                                              exclusive=True)
        result = retry or result
    LAST_IMAGE = result
    if result and is_refusal_card(result):
        raise ContentRefused(
            "Ideogram 4 declined this prompt and returned its safety card instead "
            "of an image.")
    return result

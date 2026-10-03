"""Region grounding and edit-instruction language work.

Everything that turns a user's free-text edit request into something the mask
and edit pipelines can act on: the VLM region inventory, phrase-variant
grounding, removal detection, region presence checks, English normalisation of
regions/instructions, item attribute classification and the edit-prompt refiner.

Extracted from image.py. This layer talks to the LLM/VLM but never to ComfyUI,
so it sits below the mask and edit pipelines and can be imported by both.
"""
import logging
import os
import re
from typing import Optional

from prompts import EDIT_PROMPT_REFINER
from llm import call_llm_simple, analyze_image_with_llm
from utils import safe_json_from_llm

logger = logging.getLogger("assistant.image")


_EDIT_TARGET_PROMPT = (
    "You convert an image-edit request into a LOCALIZED inpaint spec so only one "
    "region of the photo changes and the rest (face, body, background, framing) is "
    "left untouched. Reply with ONLY JSON:\n"
    '{"region": "<short English noun naming the single area to change, e.g. hat, '
    'jacket, shirt, tie, glasses, hair, lips, face, background>", '
    '"result": "<short English description of how that area should look after the '
    'edit>", "scope": "<\'tweak\' for a small change, \'replace\' for a full swap>"}\n'
    "If the request changes the WHOLE image (overall style/filter/repaint), set "
    'region to "whole". '
    "If the message includes a list of regions ACTUALLY VISIBLE in the image, the "
    '"region" you return MUST be one of those exact nouns (pick the one the change '
    "applies to) — this is what the segmenter will look for, so an off-list noun it "
    'cannot find wastes a full render. If the thing to change is not in that list at '
    'all, set region to "absent". Reply with JSON only, no prose.'
)


# Vision-side inventory of separately-segmentable regions, used to GROUND edit-region
# selection in what SAM3/the segmenter can actually find — so we pick a maskable region
# BEFORE redrawing, instead of discovering an empty mask after a wasted inpaint cycle.
_REGION_INVENTORY_PROMPT = (
    "You are a segmentation scout. List the distinct, separately-editable things "
    "visible in the image: each person's parts (face, hair, hat, shirt, jacket, "
    "dress, tie, glasses), any held or worn objects, animals, and the "
    "background/setting. Reply with ONLY a short comma-separated list of singular "
    "English nouns, most prominent first — no counts, no adjectives, no sentences."
)


# Per-image inventory cache (keyed by path + mtime) so grounding the region costs
# ONE vision call per source image even when several edits target it in a row.
_INVENTORY_CACHE: dict = {}


def _image_region_inventory(ctx, image_path: str) -> list:
    """Short list of segmentable region nouns actually visible in ``image_path``.

    Used to ground edit-region selection (the SAM3 ask): the model picks the region
    from things that are really there, so the chosen phrase masks cleanly instead of
    failing and forcing the slow whole-frame fallback. Returns [] on any failure, so
    selection degrades gracefully to the ungrounded path. Cached per image."""
    if not image_path or not os.path.exists(image_path):
        return []
    try:
        key = (image_path, os.path.getmtime(image_path))
    except OSError:
        key = (image_path, 0)
    if key in _INVENTORY_CACHE:
        return _INVENTORY_CACHE[key]
    try:
        # NO prefill: a closed-<think> prefill makes these fine-tunes return EMPTY
        # for a VISION task (verified live — the same bug that broke evaluate_image
        # and inspect_image). An empty response here silently disables region
        # grounding and pushes edits onto the whole-frame fallback.
        resp = analyze_image_with_llm(
            ctx=ctx, image_path=image_path, user_text="List the editable regions.",
            system_prompt=_REGION_INVENTORY_PROMPT, max_tokens=80)
    except Exception as exc:
        logger.warning("region inventory failed: %s", exc)
        return []
    out = []
    for tok in re.split(r"[,\n;/]+", resp or ""):
        tok = tok.strip(" .-\t").lower()
        if tok and len(tok) <= 30 and tok not in out:
            out.append(tok)
    out = out[:15]
    _INVENTORY_CACHE[key] = out
    if len(_INVENTORY_CACHE) > 64:        # bound the cache
        _INVENTORY_CACHE.pop(next(iter(_INVENTORY_CACHE)))
    return out


def _phrase_overlaps(a: str, b: str) -> bool:
    """Word-level containment for region matching ('shirt' ⊂ 'blue shirt').
    Raw substring `in` matched across word boundaries ('hair' ⊂ 'chair',
    'cat' ⊂ 'scatter') and grounded edits onto the wrong region."""
    ta, tb = set(a.split()), set(b.split())
    return bool(ta and tb and (ta <= tb or tb <= ta))


# Canonical segmenter-friendly nouns for common region variants. SAM3/Florence
# ground short everyday nouns far more reliably than niche synonyms ("spectacles"
# often returns nothing while "glasses" masks cleanly). Applied when grounding a
# region and when building mask-retry variants.
_REGION_SYNONYMS = {
    "tshirt": "shirt", "t-shirt": "shirt", "tee": "shirt", "top": "shirt",
    "blouse": "shirt", "sweater": "shirt", "hoodie": "shirt", "jumper": "shirt",
    "spectacles": "glasses", "eyeglasses": "glasses", "sunglasses": "glasses",
    "shades": "glasses",
    "trousers": "pants", "jeans": "pants", "slacks": "pants",
    "sneakers": "shoes", "boots": "shoes", "trainers": "shoes", "footwear": "shoes",
    "cap": "hat", "beanie": "hat", "headwear": "hat",
    "backdrop": "background", "surroundings": "background", "scenery": "background",
    "puppy": "dog", "kitten": "cat",
    "vehicle": "car", "automobile": "car",
    "beard": "facial hair", "moustache": "facial hair", "mustache": "facial hair",
}


def _region_phrase_variants(phrase: str) -> list:
    """Deterministic fallback phrases when the segmenter finds no mask for
    ``phrase``. Ordered, deduped, EXCLUDING the original: last noun word (drops
    color/size adjectives — 'blue denim shirt' -> 'shirt'), a canonical synonym,
    and a naive singular/plural toggle. Bounded to 3 variants so a retry ladder
    costs at most a few quick segmentation jobs, never a spiral."""
    p = (phrase or "").strip().lower()
    if not p:
        return []
    out = []
    words = p.split()
    if len(words) > 1:
        out.append(words[-1])                       # head noun without adjectives
    for key in (p, words[-1] if words else p):
        syn = _REGION_SYNONYMS.get(key)
        if syn:
            out.append(syn)
    if p.endswith("s") and len(p) > 3:
        out.append(p[:-1])                          # naive singular
    elif not p.endswith("s"):
        out.append(p + "s")                         # naive plural
    seen = {p}
    uniq = []
    for v in out:
        v = v.strip()
        if v and v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq[:3]


def _ground_region_phrase(ctx, image_path: str, region: str) -> str:
    """Map an edit ``region`` noun onto something the segmenter can actually find in
    ``image_path``, using the visible-region inventory (the SAM3 ask: decide what to
    redraw BEFORE redrawing). Returns the grounded noun, the original noun if it is
    already present or grounding is unavailable, or "absent" when the target is
    genuinely not in the frame (so the caller skips a doomed whole-frame redraw)."""
    region = (region or "").strip().lower()
    if not region:
        return region
    inv = _image_region_inventory(ctx, image_path)
    if not inv:
        return region                      # no inventory -> don't second-guess
    # Already groundable: the requested noun matches a visible region either way.
    for item in inv:
        if region == item or _phrase_overlaps(region, item):
            return region
    # Synonym pass before spending an LLM call: 'tshirt' should ground onto a
    # visible 'shirt' deterministically.
    for variant in _region_phrase_variants(region):
        for item in inv:
            if variant == item or _phrase_overlaps(variant, item):
                logger.info("region grounding: %r -> %r (synonym/variant)", region, item)
                return item
    # Fuzzy pass: catches near-misses/typos ('backgroud' -> 'background').
    try:
        import difflib
        close = difflib.get_close_matches(region, inv, n=1, cutoff=0.8)
        if close:
            logger.info("region grounding: %r -> %r (fuzzy)", region, close[0])
            return close[0]
    except Exception:
        pass
    # Ask the small LLM to map the request onto exactly one visible region (or NONE).
    try:
        sys = ("Map the user's edit target to ONE region from the provided list of "
               "things visible in the image. Reply with the single best-matching "
               "item EXACTLY as written, or the word NONE if the target is not in "
               "the list at all. One word/phrase only, no punctuation, no prose.")
        ask = f"Edit target: {region}\nVisible regions: {', '.join(inv)}"
        resp = (call_llm_simple(ctx, sys, ask, max_tokens=16, temperature=0.0,
                                prefill="<think></think>") or "").strip().lower()
    except Exception as exc:
        logger.warning("region grounding map failed: %s", exc)
        return region
    pick = resp.strip(" .\"'\n\t")
    if pick in ("none", "") or pick.startswith("none"):
        logger.info("region grounding: %r not among visible regions %s -> absent", region, inv)
        return "absent"
    for item in inv:
        if pick == item or _phrase_overlaps(pick, item):
            if item != region:
                logger.info("region grounding: %r -> %r (from inventory)", region, item)
            return item
    # The map returned something off-list — salvage it with a fuzzy match before
    # keeping the original ungrounded phrase (LLMs often echo a slightly reworded
    # inventory item, e.g. 'the shirt' / 'shirts').
    try:
        import difflib
        close = difflib.get_close_matches(pick, inv, n=1, cutoff=0.75)
        if close:
            logger.info("region grounding: LLM pick %r -> %r (fuzzy)", pick, close[0])
            return close[0]
    except Exception:
        pass
    return region                          # map returned something off-list; keep original


def _extract_edit_target(ctx, instruction: str, image_path: Optional[str] = None):
    """Parse an edit request into (region_phrase, result_prompt, denoise) for the
    contained-edit pipeline. Returns (None, ...) when the edit is whole-image or
    parsing fails, so the caller can fall back to a whole-frame edit.

    When ``image_path`` is given, the region is grounded against a vision inventory
    of what's actually in the frame, so the chosen region is one the segmenter can
    mask — defining the right region up front is far cheaper than redrawing badly
    and starting over."""
    instr = (instruction or "").strip()
    if not instr:
        return None, "", 0.75
    inventory = _image_region_inventory(ctx, image_path) if image_path else []
    user = instr
    if inventory:
        user = (f"{instr}\n\nRegions actually visible in the image (choose \"region\" "
                f"from THIS list — the closest match to what should change): "
                f"{', '.join(inventory)}")
        logger.info("edit-target grounding: inventory=%s", inventory)
    try:
        resp = call_llm_simple(ctx, _EDIT_TARGET_PROMPT, user,
                               max_tokens=120, temperature=0.0, prefill="<think></think>")
        data = safe_json_from_llm(resp) if resp else None
    except Exception as exc:
        logger.warning("edit-target extraction failed: %s", exc)
        data = None
    if not isinstance(data, dict):
        # Deterministic fallback: the LLM's JSON flaked (empty/garbled). Common
        # edit phrasings are regular enough to parse directly — a parsed contained
        # edit beats silently degrading to a whole-frame re-render.
        m = re.match(
            r"^\s*(?:please\s+)?(?:change|turn|convert)\s+(?:the\s+|his\s+|her\s+)?"
            r"(?P<region>[\w\s-]{2,40}?)\s+(?:to|into)\s+(?P<result>.+)$",
            instr, re.IGNORECASE) or re.match(
            r"^\s*(?:please\s+)?make\s+(?:the\s+|his\s+|her\s+)?"
            r"(?P<region>[\w\s-]{2,40}?)\s+(?P<result>[\w\s,-]+)$",
            instr, re.IGNORECASE)
        if m:
            region = m.group("region").strip().lower()
            result = m.group("result").strip()
            if region and region not in ("whole", "image", "picture", "everything", "it"):
                logger.info("edit-target: deterministic fallback parsed region=%r result=%r",
                            region, result[:50])
                return region, f"{result} {region}".strip(), 0.75
        return None, "", 0.75
    region = (data.get("region") or "").strip().lower()
    result = (data.get("result") or "").strip()
    scope = (data.get("scope") or "tweak").strip().lower()
    if not region or region in ("whole", "image", "all", "everything", "none"):
        return None, "", 0.75
    if region == "absent":
        # The target isn't in the frame; redrawing would invent it whole-frame and
        # wreck the rest. Signal the caller to skip the contained path.
        logger.info("edit-target: requested region not present in image (%r)", instr[:60])
        return "absent", (result or instr), 0.75
    return region, (result or instr), (0.6 if scope == "tweak" else 0.9)


# CLIPSeg (Kijai/clipseg-rd64) is English-only AND wants a short concrete noun: a
# non-English region word yields a garbage/near-inverted mask (the inpaint regenerates
# everything AROUND the target), and a long descriptive phrase ("left side road edge
# background" — seen from the real agent) segments just as badly. A short English term
# passes straight through; anything else is reduced by the LLM to 1–2 English words
# (verified: платье→dress, очки→glasses, «синее платье»→dress).
_REGION_TRANSLATE_PROMPT = (
    "You name a region of an image for an image segmenter. Given a word or phrase in "
    "any language naming a part of an image, reply with ONLY a short English noun "
    "(one or two lowercase words) the segmenter can locate, e.g. 'face', 'shirt', "
    "'hair', 'dress', 'background', 'sky', 'left tank'. Nothing else."
)


def _english_region(ctx, region: str) -> str:
    """Return a short English CLIPSeg term for ``region``. Short ASCII input passes
    through; non-English or long phrases are reduced to 1–2 English words by the LLM."""
    r = (region or "").strip()
    if not r:
        return "face"
    if r.isascii() and len(r.split()) <= 2:
        return r.lower()
    try:
        resp = call_llm_simple(ctx, _REGION_TRANSLATE_PROMPT, r,
                               max_tokens=16, temperature=0.0, prefill="<think></think>")
        first = (resp or "").strip().splitlines()[0] if (resp or "").strip() else ""
        word = "".join(c for c in first if c.isascii() and (c.isalpha() or c == " ")).strip().lower()
        if word and len(word.split()) <= 3:
            logger.info("CLIPSeg region normalized: %r -> %r", r, word)
            return word
    except Exception as exc:
        logger.warning("Region normalization failed for %r: %s", r, exc)
    # Last resort: keep it short — CLIPSeg degrades fast beyond ~2 words.
    return " ".join(r.lower().split()[:2])


_INSTR_TRANSLATE_PROMPT = (
    "Translate the following image-editing instruction into a short English phrase "
    "describing the desired visual result for an image generator. Reply with ONLY "
    "the English phrase, nothing else."
)


def _english_instructions(ctx, text: str) -> str:
    """Translate non-English inpaint instructions to English for the positive prompt.
    The rest of the pipeline prompts the old model in English; keep the edit prompt
    consistent (the agent mostly sends English but falls back to Russian sometimes)."""
    t = (text or "").strip()
    if not t or t.isascii():
        return t
    try:
        resp = call_llm_simple(ctx, _INSTR_TRANSLATE_PROMPT, t,
                               max_tokens=60, temperature=0.0, prefill="<think></think>")
        first = (resp or "").strip().splitlines()[0].strip()
        if first and first.isascii():
            logger.info("Inpaint instructions translated: %r -> %r", t[:60], first[:60])
            return first
    except Exception as exc:
        logger.warning("Instruction translation failed for %r: %s", t[:60], exc)
    return t


# LLM-driven object understanding. Replaces the old hardcoded item-category
# regexes (_WORN_ITEM_RE / _SMALL_ITEM_RE): those only covered anticipated
# items (shoes, scarves, hats...) and silently mis-handled everything else.
# One cached call classifies ANY phrase the user mentions; each attribute is
# tri-state (True/False/None) and every consumer fails open on None, so an
# unreachable LLM degrades to the neutral behaviour instead of blocking.
_ITEM_ATTR_PROMPT = (
    "You classify an object mentioned in a photo-editing request. Reply with ONLY a "
    'JSON object, no other text: {"worn": true|false, "small": true|false, '
    '"large": true|false, "multi": true|false, "part": "face|hair_or_head|other", '
    '"former_place": true|false}.\n'
    "worn: the object is worn on or attached to a person's body (clothing, footwear, "
    "headwear, eyewear, jewelry, accessories, facial hair, tattoos, makeup, ...) so "
    "removing it exposes occluded body parts underneath. Free-standing objects, "
    "body parts themselves, and scenery are not worn.\n"
    "small: in a typical photo of a person the object plausibly covers well under a "
    "quarter of the frame (footwear, glasses, jewelry, a phone, a cup, ...). Large "
    "garments, hair, people, furniture and backgrounds are not small.\n"
    "large: the region legitimately spans most of the frame (background, sky, wall, "
    "floor, the whole scene).\n"
    "multi: the phrase names something that normally appears as SEVERAL separate "
    "instances in one photo — paired items (shoes, gloves, earrings, socks, eyes) or "
    "an explicit plural/group (the flowers, all the buttons). A single object is not multi.\n"
    "part: first, if the phrase covers a body or a whole person (even when it also "
    "names the face: \"the man's body and face\"), part is \"other\". Otherwise "
    "\"face\" when the phrase is the face or a part of it (eyes, lips, mouth, "
    "nose, cheeks, chin, skin, beard, moustache, makeup, expression, smile); "
    "\"hair_or_head\" for hair, a hairstyle, the head, ears; \"other\" otherwise.\n"
    "former_place: the phrase names the place where something USED to be, not a thing "
    "(\"the area where the vase was\", \"место, где стояла ваза\")."
)


_item_attr_cache: dict = {}


def _item_attributes(ctx, phrase: str) -> dict:
    """Physical attributes of an arbitrary object phrase, via one cached LLM call.
    Returns {"worn", "small", "large", "multi"} as bool|None and "part"
    ("face" | "hair_or_head" | "other") as str|None, plus "facial" (the face
    itself) and "head" (face, hair or head) derived from it
    — None means "could
    not classify" and consumers must treat it as unknown/neutral."""
    key = (phrase or "").strip().casefold()
    out = {"worn": None, "small": None, "large": None, "multi": None, "part": None,
           "former_place": None}
    if not key:
        return out
    if key in _item_attr_cache:
        return _derived(_item_attr_cache[key])
    try:
        resp = call_llm_simple(ctx, _ITEM_ATTR_PROMPT, phrase.strip(),
                               max_tokens=120, temperature=0.0, prefill="<think></think>")
        # Was a NON-greedy first-{...}: a leading "{}" in the reply, or a nested
        # object, won over the real answer and every attribute silently fell back
        # to its default — which then routes the edit down the wrong pipeline.
        data = safe_json_from_llm(resp or "", required_keys=tuple(out))
        if data:
            for k in out:
                if isinstance(data.get(k), bool):
                    out[k] = data[k]
            if data.get("part") in ("face", "hair_or_head", "other"):
                out["part"] = data["part"]
            logger.info("item attributes for %r: %s", phrase[:60], out)
    except Exception as exc:
        logger.warning("item attribute classification failed for %r: %s", phrase[:60], exc)
    # Only cache real classifications. An all-None result means the LLM was
    # unreachable AT THAT MOMENT (LM Studio evicted while ComfyUI holds VRAM is
    # routine here) — caching it would poison the whole session and every later
    # edit of that phrase would silently lose its size/worn priors.
    if any(v is not None for v in out.values()):
        _item_attr_cache[key] = out
    return _derived(out)


def _derived(attrs: dict) -> dict:
    part = attrs.get("part")
    return {**attrs, "facial": None if part is None else part == "face",
            "head": None if part is None else part in ("face", "hair_or_head")}


def _is_facial(ctx, phrase: str) -> bool:
    """The region is the face or a part of it (hair is not). The model's read
    (_item_attributes); it replaced a word list that missed «губы» and every
    other Russian word. Unknown -> False: face protection stays on."""
    return bool((phrase or "").strip()) and _item_attributes(ctx, phrase).get("facial") is True


def _is_removal_instruction(text: str) -> bool:
    """The instruction takes something out of the picture (the model's read,
    image_router.edit_plan: a removal kind, or a removal step). Was a verb regex: «сними» and «take off»
    were removals, «clean up» and «без» were not, each tuned by hand."""
    import image_router
    if not (text or "").strip():
        return False
    plan = image_router.edit_plan(text)
    # a removal kind with nothing named («make it cleaner») is a tidy-up
    reads = [plan] + [image_router.edit_plan(s) for s in plan["steps"]]
    return any(r["kind"] in ("object_remove", "person_remove") and r["target"] for r in reads)


def _is_removal_of(instructions: str, region: str = "", ctx=None) -> bool:
    """The edit leaves `region` gone -- removed, or described as absent ("bare
    neck, no scarf", "the empty table where the vase was"). Not when it removes
    something ELSE on the region: live 2026-09-11 'Remove the large black shadow
    covering the cat-shaped object' deleted the cat from the layout and the
    agent spent four renders getting it back. The model's read (edit_plan
    removes_region); it replaced a verb regex, an object parser, stem
    matching on both nouns and four phrase patterns."""
    import image_router
    if not (instructions or "").strip() or not (region or "").strip():
        return False
    return image_router.removes_region(instructions, region)


# First-word tokens the vision model uses to signal absence of a region. Covers
# the IMAGE_INSPECT_PROMPT labels (MISSING) plus alternate phrasings (ABSENT/NONE).
_ABSENCE_LABELS = frozenset({"MISSING", "ABSENT", "NONE", "ABSENT.", "MISSING.", "NONE."})


def _region_present(ctx, image_path: str, region: str) -> Optional[bool]:
    """Ask the vision model whether ``region`` is visible in the image.

    Returns True on presence, False on confirmed absence, None when uncertain
    (callers fail open — never block on a vague answer). Used by the face-detailer
    preflight (enhance_faces_with_comfy) to skip the heavy workflow on imageless
    scenes.
    """
    try:
        from prompts import IMAGE_INSPECT_PROMPT
        # NO prefill: a closed-<think> prefill returns EMPTY on vision tasks with
        # these fine-tunes (verified live), which turned every presence check into
        # an "unclear" fail-open and ran the heavy detailer blind.
        resp = analyze_image_with_llm(
            ctx=ctx, image_path=image_path,
            user_text=(
                f"Check one thing: is there a {region} visible in the image?\n"
                "Answer with PRESENT, PARTIAL, or MISSING as the first word, "
                "then a few words of evidence."),
            system_prompt=IMAGE_INSPECT_PROMPT,
            max_tokens=60,
        )
        ans = (resp or "").strip().upper()
        first = ans.split()[0].rstrip(".,;:") if ans.split() else ""
        if first in ("PRESENT", "PARTIAL"):
            return True
        if first in _ABSENCE_LABELS:
            return False
        if ans.startswith(("NOT ", "NO ", "ABSENT", "NONE")):
            return False
        # IMAGE_INSPECT_PROMPT has it describe the scene first, then the verdict
        # ("A woman stands ...\n\nMISSING no bag is visible") -- read the first
        # verdict word in capitals, wherever it is.
        m = re.search(r"\b(PRESENT|PARTIAL|MISSING|ABSENT)\b", resp or "")
        if m:
            return m.group(1) in ("PRESENT", "PARTIAL")
        logger.warning("Region presence check returned unclear answer for %r: %r", region, resp)
    except Exception as exc:
        logger.warning("Region presence check failed for %r: %s", region, exc)
    return None


def _firered_instruction(ctx, region: str, instructions: str, removal: bool) -> str:
    """Turn the mask-oriented (region, instructions) pair into one plain-English
    edit instruction for FireRed/Qwen-Image-Edit, which has no mask — it edits
    the whole image from the sentence, so we name the target and demand the rest
    stays put. Deterministic fallback for _refine_edit_prompts."""
    region_en = _english_region(ctx, region) if region else "it"
    # NOTE: no forced "Photorealistic." — the source may be an anime frame, a
    # painting or a render; demanding photorealism used to restyle such edits.
    keep = ("Keep everything else in the image exactly the same, "
            "matching the original style and lighting.")
    if removal:
        return f"Remove the {region_en} and its shadow and reflection from the image completely. {keep}"
    instr_en = _english_instructions(ctx, instructions) or instructions
    return f"Change the {region_en} to: {instr_en}. {keep}"


_STYLE_LAYOUT_PROMPT = (
    "In 2-4 short sentences, describe the EXACT layout of this image so someone "
    "could redraw it without changing the composition. State: how many people/"
    "subjects and their left-to-right order; each one's approximate size and "
    "position (e.g. 'left edge, cropped at the shoulder, small' / 'center, "
    "waist-up, medium' / 'right, extreme close-up filling a third of the frame'); "
    "any border or background band whose color differs from the main background; "
    "any text overlay and roughly where it sits. Be factual and concrete, no "
    "opinions, no description of style or mood."
)


def _firered_style_instruction(ctx, instructions: str, image_path: Optional[str] = None) -> str:
    """Whole-image restyle instruction for the style_transfer category.

    _firered_instruction names one region and demands everything ELSE stay
    put -- right for a targeted edit, self-contradictory for a style transfer
    with no region at all. Live bug (2026-09-20): calling it with region=""
    produced "Change the it to: X. Keep everything else the same", which read
    as "restyle a subject, preserve the frame" and came back as a cropped,
    partial redraw that dropped the title text and clipped a person out of
    frame instead of restyling the whole collage.

    A second live pass the same day showed a generic "keep everyone the same
    size/position" instruction is still too vague under a heavy stylization
    (comic/cartoon): the model recomposed toward comic-poster conventions
    (oversized dramatic foreground faces) and pushed a person past the frame
    edge. A concrete, vision-derived description of the ACTUAL layout ("4
    people left-to-right, leftmost cropped at the shoulder near the left
    edge...") gives the model specific anchors instead of a vague demand, so
    when `image_path` is available this asks the vision model to describe the
    source first and folds that description into the instruction. Falls back
    to the generic wording if the vision call fails or no path is given.
    """
    instr_en = _english_instructions(ctx, instructions) or instructions
    layout_note = ""
    if image_path:
        try:
            from llm import analyze_image_with_llm
            desc = (analyze_image_with_llm(
                ctx=ctx, image_path=image_path, user_text=_STYLE_LAYOUT_PROMPT,
                system_prompt="You are a precise, literal visual-layout describer.",
            ) or "").strip()
            if desc:
                layout_note = f" The source image's exact layout: {desc} Reproduce this layout precisely."
        except Exception as exc:
            logger.warning("style_transfer layout description failed: %s", exc)
    return (f"Redraw the ENTIRE image in this style: {instr_en}. "
            "Restyle every part of the image -- all subjects, all background, "
            "any text or logo -- nothing stays photorealistic or untouched. "
            "Keep the exact same composition, framing, crop and layout as the "
            "source image: every person must stay at the SAME size, the SAME "
            "position, and the SAME pose as in the source, fully inside the "
            "frame with margin from every edge -- do not enlarge, shrink, move, "
            "re-pose, crop, or zoom in on anyone or anything, even to follow "
            "the conventions of the requested style. "
            # This model cannot render legible lettering under a full restyle --
            # confirmed live 2026-09-20, a restyled poster's background neon
            # signs came out as illegible mush. Prompting it to "keep the text
            # crisp" does not work (it has no mechanism to render clean glyphs
            # here); the reliable fix is to stop it from ATTEMPTING new words at
            # all. Any text that was already legible in the source (e.g. a
            # title) may be restyled in its existing position, but nothing
            # invents fresh wording.
            "Any signs, screens, posters or other lettering-shaped objects in "
            "the scene must be restyled as abstract glowing shapes or blurred "
            "marks -- do not attempt to render new legible words or letters "
            "anywhere in the image." + layout_note)


# One-slot memo so the FireRed-contained, old-model-contained and whole-frame
# fallback stages of a single edit don't each re-pay the refinement LLM call.
_EDIT_PROMPT_CACHE = {"key": None, "value": None}


def _refine_edit_prompts(ctx, region: str, instructions: str, *,
                         removal: bool = False) -> tuple:
    """Rewrite the agent's raw (region, instructions) into two well-formed edit
    prompts via the LLM: an imperative INSTRUCTION for the instruction-following
    editor (FireRed/Qwen-Image-Edit) and a self-contained FILL_PROMPT for the
    masked-fill engines (which see only a crop, so 'make it red' means nothing).

    The agent's own tool args are often terse or command-shaped; this pass fixes
    verb choice, binds the change to the named target, and strips change-verbs
    from the fill prompt. Falls back to the deterministic templates whenever the
    LLM is unavailable or returns anything malformed — never raises, never
    blocks the edit. Returns (instruction, fill_prompt)."""
    fallback_instr = _firered_instruction(ctx, region, instructions, removal)
    fallback_fill = _english_instructions(ctx, instructions) or instructions
    key = (region, instructions, removal)
    if _EDIT_PROMPT_CACHE["key"] == key and _EDIT_PROMPT_CACHE["value"]:
        return _EDIT_PROMPT_CACHE["value"]
    try:
        user = f"REGION: {region or 'the image'} || CHANGE: {instructions}"
        resp = call_llm_simple(ctx, EDIT_PROMPT_REFINER, user,
                               max_tokens=220, temperature=0.0,
                               prefill="<think></think>")
        # Was first-{ to last-}: trailing prose containing a brace made the whole
        # slice unparseable, so the refiner's answer was discarded and the raw
        # fallback instruction went to the edit engine instead.
        data = safe_json_from_llm(resp or "",
                                  required_keys=("instruction", "fill_prompt")) or {}
        instr = (data.get("instruction") or "").strip()
        fill = (data.get("fill_prompt") or "").strip()
        # Sanity gates: English, sane length, and no change-verbs in the fill
        # prompt (a masked-fill engine would render the verb as noise).
        ok_instr = 10 <= len(instr) <= 400 and instr.isascii()
        ok_fill = (6 <= len(fill) <= 400 and fill.isascii() and not
                   re.match(r"(?i)\s*(make|change|replace|remove|add|turn)\b", fill))
        result = (instr if ok_instr else fallback_instr,
                  fill if ok_fill else fallback_fill)
        if ok_instr or ok_fill:
            logger.info("edit-prompt refiner: region=%r -> instr=%r | fill=%r",
                        region[:30], result[0][:80], result[1][:80])
        else:
            logger.info("edit-prompt refiner: output rejected, using templates")
        _EDIT_PROMPT_CACHE["key"], _EDIT_PROMPT_CACHE["value"] = key, result
        return result
    except Exception as exc:
        logger.warning("edit-prompt refiner failed (%s); using templates", exc)
        return fallback_instr, fallback_fill


def _strip_lead(text: str, pattern: str) -> str:
    """Remove a leading command phrase (regex) so the remainder is the payload."""
    return re.sub(pattern, "", text, count=1, flags=re.IGNORECASE).strip(" ,.:;-\t")

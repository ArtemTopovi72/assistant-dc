"""Edit-intent routing for the image pipeline.

Extracted from image.py. This is the 16-category intent router and the
dispatcher that turns a classified intent into a concrete ComfyUI pipeline
call, plus the two orchestration helpers it needs.

SEAM NOTE -- read before adding an import here.
Every pipeline this module dispatches to still lives in image.py, and the
suites stub those pipelines as `image.<name>`.  Before the split those were
plain module globals, so the stub was seen automatically.  Importing them by
value here would bind the same name on BOTH sides of the split: patching
`image.X` would move only half the behaviour, the stub would silently die, and
the REAL pipeline would run while the suite still printed PASS.  So they are
NOT imported -- they are reached through `_image.` below, which defers the
import to call time and always resolves the CURRENT binding on image.py.
The same applies to the grounding/sizing helpers (_item_attributes,
_region_present, _extract_edit_target, _firered_instruction,
_firered_style_instruction, _source_dims,
_strip_lead, _REMOVAL_RE), which image.py re-exports and the suites patch there.
"""
import logging
import os
import random
import re
import time
from typing import Literal, Optional

from pydantic import BaseModel, ValidationError

from config import OUTPUT_DIR, DEFAULT_WIDTH, DEFAULT_HEIGHT
import config as _config          # read IMAGE_ENGINE live so the GUI can switch it

logger = logging.getLogger("assistant.image")


class _ImageProxy:
    """Attribute proxy onto the still-monolithic image.py.

    Reading through it defers the import to call time (no import cycle) and
    always resolves the CURRENT binding, so a runtime patch of image.<name> is
    honoured here even though the caller has moved out of image.py.
    """

    def __getattr__(self, name):
        import image
        return getattr(image, name)


_image = _ImageProxy()



# Which kind of edit an instruction is, read by the model -- it replaced a
# table of 17 keyword regexes plus clause splitting on "and"/"then"/commas,
# each tuned against the last live miss ("prison-style tattoos" sent a forearm
# through whole-frame restyles, "rim lighting" in a cyberpunk preset stole it
# for relight, "restore the cat so its whole body is visible" became photo
# restoration). bench/edit_intent_live.py runs the phrases against the model.
EDIT_CATEGORIES = {
    "transform": "rotate, flip or mirror the whole picture, or crop it to the face",
    "text_add": "add NEW lettering whose exact words are given",
    "text_edit": "fix, change or rewrite lettering already in the picture (spelling, wording)",
    "background_remove": "remove the background: transparent, or one plain colour (white, black)",
    "background_replace": "change the background: a different one, another place, or the same "
                          "one blurred or darkened",
    "colour_convert": "black and white, grayscale, sepia, desaturate the whole picture",
    "style_transfer": "redraw the whole picture in another style (anime, oil painting, cyberpunk, comic...)",
    "restore": "repair an old or damaged photo (scratches, noise, blur, faded colours), "
               "colourize it, or restore/enhance the faces (also with upscaling: 'upscale and restore the face')",
    "relight": "change the lighting or time of day of the scene",
    "upscale": "higher resolution, bigger, sharper, crisper -- the content unchanged, no face restoring",
    "outpaint": "extend the canvas: zoom out, wider shot, make it square/wider",
    "person_remove": "remove a person or people",
    "object_remove": "remove, erase or clean away a thing (an object, a stain, a watermark)",
    "object_insert": "add a new thing or person to the picture ('нарисуй блондинку', 'добавь кота')",
    "face_edit": "change the face, expression, eyes, hair or hair colour, beard, skin, makeup",
    "clothing_edit": "change what a person wears",
    "product_edit": "change a product, its packaging or label for a product shot",
    "subject_edit": "any other change to what is in the picture",
}

_EDIT_SYSTEM = (
    "You sort a picture-editing instruction for an image editor. Kinds:\n{kinds}\n\n"
    "Read what the instruction MEANS (any language), not single words: "
    "'prison-style tattoos' is a kind of tattoo, not a restyle; a style preset that "
    "mentions 'rim lighting' is still a style; 'restore the cat so its whole body is "
    "visible' brings a thing back (subject_edit), not photo repair; 'remove the "
    "background, make it white' is ONE background_remove; transform is the WHOLE "
    "picture, so 'mirror the text on the sign' is a lettering edit; 'remove the cat "
    "and add a tree' is two steps (a removal, then an insert).\n"
    "steps: when the instruction asks for two or more edits of DIFFERENT kinds, each "
    "edit as its own short instruction in the order given; otherwise [].\n"
    "target: the thing the instruction takes out of the picture, as a short English noun "
    "phrase (\"the puddle\", \"the man on the left\"); \"\" when nothing is removed, and "
    "\"\" when the instruction NAMES no thing (\"make it cleaner\" is a clean-up: never "
    "invent what to remove).\n"
    "fill: for background_remove, \"white\" or \"black\" when that plain colour is "
    "asked for, else \"transparent\".\n"
    "resizes: the edit makes something bigger or smaller or fixes its proportions "
    "(\"the head is too big\", \"увеличь кота\").\n"
    "reformats: it asks for another orientation or aspect ratio of the whole picture "
    "(horizontal, vertical, 16:9, square) -- \"a snowy landscape\" is a scene, not a format.\n"
    "Answer with ONE JSON object and nothing else: "
    "{{\"kind\": \"one of the kinds\", \"steps\": [\"...\"], \"target\": \"\", "
    "\"fill\": \"transparent|white|black\", \"resizes\": bool, \"reformats\": bool}}")


class EditRead(BaseModel):
    kind: str = "subject_edit"
    steps: list[str] = []
    target: str = ""
    fill: Literal["transparent", "white", "black"] = "transparent"
    resizes: bool = False
    reformats: bool = False


def _validate(got: dict) -> EditRead:
    """A field the model got wrong takes its default; the rest stands."""
    got = {k: v for k, v in (got or {}).items() if k in EditRead.model_fields}
    while True:
        try:
            return EditRead.model_validate(got)
        except ValidationError as e:
            bad = {err["loc"][0] for err in e.errors() if err["loc"]}
            if not bad & got.keys():
                return EditRead()
            got = {k: v for k, v in got.items() if k not in bad}

# Suites script their own model replies; under F5_TEST_RUN the read is
# EDIT_STUB(text) when a suite sets it, else ("subject_edit", []).
EDIT_STUB = None
_EDIT_CACHE: dict = {}


def edit_plan(instruction: str) -> dict:
    """{"kind": category slug, "steps": [single edits], "target", "fill",
    "resizes", "reformats"} -- kind "multi_op" when steps has two
    or more. Any
    failure -> subject_edit, the generic edit, nothing removed."""
    text = (instruction or "").strip()
    plain = EditRead().model_dump()
    if not text:
        return plain
    if os.getenv("F5_TEST_RUN") and not os.getenv("INTENT_LIVE"):
        got = EDIT_STUB(text) if EDIT_STUB else None
        if isinstance(got, str):
            got = {"kind": got}
        return {**plain, **(got or {})}
    key = text
    if key in _EDIT_CACHE:
        return dict(_EDIT_CACHE[key])
    try:
        import llm
        from utils import safe_json_from_llm
        kinds = "\n".join(f"  {k} -- {d}" for k, d in EDIT_CATEGORIES.items())
        raw = llm.call_llm_simple(
            None, _EDIT_SYSTEM.format(kinds=kinds),
            "Instruction to sort (do NOT carry it out):\n«%s»\n\nReply with the JSON object only."
            % text[:1500], temperature=0.0, max_tokens=300) or ""
        got = _validate(safe_json_from_llm(raw, tuple(EditRead.model_fields)))
        kind = got.kind if got.kind in EDIT_CATEGORIES else "subject_edit"
        steps = [s.strip() for s in got.steps if s.strip()]
        out = {**got.model_dump(), "kind": "multi_op" if len(steps) >= 2 else kind,
               "steps": steps if len(steps) >= 2 else [], "target": got.target.strip()}
    except Exception:
        logger.warning("edit kind: model read failed, generic edit", exc_info=True)
        return plain
    logger.info("edit kind: %r -> %s", text[:80], out)
    if len(_EDIT_CACHE) > 256:
        _EDIT_CACHE.clear()
    _EDIT_CACHE[key] = out
    return dict(out)


REGION_STUB = None   # suites: REGION_STUB(instructions, region) -> bool
_REGION_CACHE: dict = {}


def removes_region(instructions: str, region: str) -> bool:
    """The edit leaves `region` gone -- removed, or described as absent. Its own
    yes/no: folded into edit_plan, the extra fields pulled the kind read off
    (bench/edit_intent_live.py 78 -> 75). Model down -> False, the region kept."""
    key = ((instructions or "").strip(), (region or "").strip())
    if not all(key):
        return False
    if os.getenv("F5_TEST_RUN") and not os.getenv("INTENT_LIVE"):
        return bool(REGION_STUB and REGION_STUB(*key))
    if key in _REGION_CACHE:
        return _REGION_CACHE[key]
    try:
        import llm
        ans = llm.call_llm_simple(
            None, "Answer yes or no only.",
            "An image editor gets an edit for one region of a picture.\n"
            "Region: «%s»\nEdit: «%s»\n\n"
            "Does the edit leave THAT region gone -- removed, taken off, or described as "
            "absent (\"bare neck, no scarf\" for a scarf, \"empty table where the vase was\" "
            "for a vase, \"barefoot\" for shoes)? No when it changes or replaces the region "
            "(\"a red vase with tulips\" for a blue vase), removes something else on or "
            "around it (\"remove the shadow covering the cat\" for the cat), or just does not "
            "mention it (\"table where the cat sat, now with a book\" for a vase)."
            % (key[1][:200], key[0][:800]), temperature=0.0, max_tokens=5) or ""
        got = ans.strip().lower().startswith(("yes", "да"))
    except Exception:
        logger.warning("removes_region: model read failed, keeping the region", exc_info=True)
        return False
    if len(_REGION_CACHE) > 256:
        _REGION_CACHE.clear()
    _REGION_CACHE[key] = got
    return got


def classify_edit_intent(instruction: str) -> str:
    """The edit category of a free-text instruction (EDIT_CATEGORIES, or
    "multi_op" for two or more edits of different kinds)."""
    return edit_plan(instruction)["kind"]


_NO_LAYOUT_NOTE = (
    "[TOOL ERROR] The lettering in this picture cannot be edited: it was not "
    "composed from a layout (an uploaded photo, or drawn by another engine), so "
    "there are no text boxes to correct. Re-drawing the whole frame does not fix "
    "spelling — it produces different wrong letters. Tell the user plainly that "
    "the lettering cannot be corrected on this image, and offer to draw a NEW "
    "picture with the correct wording instead. Do not call another edit tool.")


# Edits that a LAYOUT can express: they change what is in the picture, which is
# exactly what a box describes. Everything outside this set is a pixel operation
# the layout has no vocabulary for — upscaling, restoring a damaged photo,
# cutting a transparent background, extending the canvas — and those keep using
# the pipelines built for them even on an image we composed.
_LAYOUT_EDITABLE = frozenset({
    "text_edit", "text_add", "subject_edit", "face_edit", "clothing_edit", "product_edit",
    "object_insert", "object_remove", "person_remove", "background_replace",
    "style_transfer", "relight", "multi_op",
})


def edit_via_layout(ctx, image_path: str, instruction: str,
                    seed: Optional[int] = None,
                    fallback_on_problems: bool = False) -> Optional[str]:
    """Edit a picture WE composed by moving its boxes, then drawing it again.

    For an image with a stored layout this is the whole editing model: the agent
    may move a box, resize it, reword what it holds, add one or delete one, and
    the picture is re-rendered from the result. Nothing is repainted over the top,
    so the frame stays coherent — which is the point. The alternative, a
    whole-frame instruction edit, re-imagines the person: asked to make a man
    blond it blended a blonde woman's face over his.

    The SEED is reused, so a re-render is an edit and not a re-roll — the framing
    and everything untouched come back the way they were.

    Returns None when the picture has no layout (an uploaded photo). That is a
    refusal for lettering, where nothing else can do the job; for other edits the
    caller falls back to the pixel pipelines, which is correct for a photo.
    """
    rec = _image.load_layout_for(image_path)
    if not rec:
        return None
    import draw_agent
    from PIL import Image as _PILImage
    try:
        with _PILImage.open(image_path) as im:
            width, height = im.size
    except Exception:
        width = int(rec.get("width") or DEFAULT_WIDTH)
        height = int(rec.get("height") or DEFAULT_HEIGHT)
    layout, notes = draw_agent.edit_layout(ctx, rec["layout"], instruction)
    logger.info("layout edit %r: %s", instruction[:60],
                "; ".join(notes) if notes else "no layout change")
    if notes == ["the model proposed no change"]:
        # Nothing was understood. Re-rendering an unchanged layout would burn a
        # generation to hand back the same picture and call it an edit.
        logger.warning("layout edit produced no change — not re-rendering")
        return None
    use_seed = seed if seed is not None else rec.get("seed")
    rounds = max(1, int(getattr(_config, "IDEOGRAM_TEXT_ROUNDS", 1) or 1))
    res = draw_agent.run(ctx, rec.get("prompt") or instruction, layout=layout,
                         width=width, height=height, seed=use_seed, rounds=rounds)
    out = res.get("image")
    _image._LAYOUT_EDIT_PROBLEMS[:] = ([] if res.get("stopped") == "ok"
                                       else [str(p) for p in (res.get("problems") or [])][:3])
    if out:
        _image.save_layout_for(out, rec.get("prompt") or instruction,
                        res.get("layout") or layout, width=width, height=height,
                        seed=res.get("seed", use_seed))
    # The re-render was judged and the edit is NOT in it ("the elephant is
    # not pink as requested", twice, live 2026-09-18): the same seed keeps
    # the old picture's colours whatever the box now says. Hand the ORIGINAL
    # to the pixel pipeline, which paints the change in place, instead of
    # shipping the unchanged picture as an edit.
    if out and fallback_on_problems and res.get("problems"):
        logger.warning("layout edit %r re-rendered but the critic still sees %s -- "
                       "falling back to the pixel pipeline",
                       instruction[:50], res.get("problems")[:2])
        _image._LAYOUT_EDIT_PROBLEMS.clear()   # this render is not the one delivered
        return None
    return out


def convert_colour(image_path: str, instruction: str) -> Optional[str]:
    """Grayscale / sepia the WHOLE picture, deterministically.

    No model, no mask, no re-render: every pixel of this operation is defined, it
    is exact, it keeps the resolution, and it cannot invent a different subject —
    which is precisely what the generative path did when this request reached it.
    """
    try:
        from PIL import Image as _PIL, ImageOps as _Ops
        src = _PIL.open(image_path).convert("RGB")
        grey = _Ops.grayscale(src)
        if re.search(r"sepia|сепи\w*", instruction or "", re.IGNORECASE):
            out = _Ops.colorize(grey, black=(28, 16, 8), white=(255, 240, 210))
        else:
            out = grey.convert("RGB")
        ts = int(time.time() * 1000)
        final = OUTPUT_DIR / f"colour-convert_{ts}.png"
        final.parent.mkdir(parents=True, exist_ok=True)
        out.save(final)
        logger.info("Colour conversion (%s) -> %s", instruction[:60], final)
        return str(final)
    except Exception:
        logger.exception("colour conversion failed for %s", image_path)
        return None


def transform_image(image_path: str, instruction: str) -> Optional[str]:
    """Rotate / mirror / crop to the face on the pixels. FireRed redrew the whole
    picture for "rotate the image 90 degrees" and "crop to the face"."""
    from PIL import Image as _PIL, ImageOps as _Ops
    t = instruction or ""
    src = _PIL.open(image_path).convert("RGB")
    import intent
    op = intent.ask_choice(
        "A user asked to transform a picture: {text}. Which? crop = crop to the face; "
        "mirror = flip left-right; flip = turn upside down; rotate_left = rotate "
        "counter-clockwise; rotate_right = rotate clockwise.",
        t, ("crop", "mirror", "flip", "rotate_left", "rotate_right"), "rotate_right")
    if op == "crop":
        import identity_metrics
        boxes = identity_metrics.face_boxes(image_path, pad=0.8)
        if not boxes:
            return None
        out = src.crop(boxes[0])
    elif op in ("mirror", "flip"):
        out = _Ops.flip(src) if op == "flip" else _Ops.mirror(src)
    else:
        deg = int((re.search(r"(\d{2,3})", t) or [0, "90"])[1])
        out = src.rotate(deg if op == "rotate_left" else -deg, expand=True)
    final = OUTPUT_DIR / f"transform_{int(time.time() * 1000)}.png"
    final.parent.mkdir(parents=True, exist_ok=True)
    out.save(final)
    return str(final)


_QUOTED_RE = re.compile(r"[\"«“']([^\"»”']{1,80})[\"»”']")
_CAPS_TEXT_RE = re.compile(r"(?-i:([A-ZА-ЯЁ0-9][A-ZА-ЯЁ0-9!?&%.,\- ]*[A-ZА-ЯЁ0-9!?%]))")
TEXT_ADD_TRIES = int(os.getenv("TEXT_ADD_TRIES", "3"))

TEXT_ADD_FAILED_NOTE = (
    "[TOOL ERROR] The lettering was drawn {n} times and read back each time, and "
    "it never came out right (best reading: {best!r}, asked: {want!r}). No picture "
    "was delivered. Tell the user plainly that the model keeps misspelling this "
    "text; offer to try different wording or to draw a new picture with the text "
    "composed from a layout. Do not claim the text was added.")


def add_text_verified(ctx, image_path: str, instruction: str,
                      seed: Optional[int] = None) -> tuple:
    """Add NEW lettering with FireRed, then read it back and reseed until the
    words match. Returns (path | None, best_reading, wanted).

    FireRed paints good poster type but spells from priors: "УРОЖАЙ 2026"
    came back "2025" on four variants in a row. Delivering that is a false
    success; the OCR read-back is what turns it into a retry.
    """
    wanted = [w.strip() for w in _QUOTED_RE.findall(instruction or "") if w.strip()]
    if not wanted:
        # "add the text SALE on the window": the agent's args often drop the quotes.
        wanted = [m.group(1) for m in _CAPS_TEXT_RE.finditer(instruction or "")]
    if not wanted:
        return None, "", ""
    from draw_text import _similarity, TEXT_MATCH_OK
    import ocr_reader
    quoted = ", ".join(f'"{w}"' for w in wanted)
    instr = (f"{instruction.strip()}\n\nThe text must read exactly {quoted} -- every letter "
             "and every digit exactly as written, nothing added or changed. Keep "
             "everything else in the picture unchanged.")
    best, best_score = "", -1.0
    base = seed if seed and seed > 0 else random.randint(1, 2**31 - 1)
    for k in range(max(1, TEXT_ADD_TRIES)):
        out = _image.edit_image_with_firered(ctx, image_path, instr, seed=base + k * 7919)
        if not out:
            continue
        reads = ocr_reader.read(out)
        if reads is None:             # no OCR on this machine: cannot verify, deliver
            return out, "", " / ".join(wanted)
        score, reading = 2.0, ""
        for w in wanted:
            s, r = max(((_similarity(w, c), c) for c in reads), default=(0.0, ""))
            if s < score:
                score, reading = s, r
        logger.info("text_add try %d: read %r for %r (%.2f)", k + 1, reading, wanted, score)
        if score > best_score:
            best, best_score = reading, score
        if score >= TEXT_MATCH_OK:
            return out, reading, " / ".join(wanted)
    # FireRed could not spell it (a year prior: "2026" -> "2025" in 21/21
    # renders on the night bench). Lay the exact words onto the ORIGINAL --
    # the misspelt render is not a base to fix, its letters are already wrong.
    try:
        import text_overlay
        out = text_overlay.overlay(image_path, wanted, where=text_overlay.placement(instruction),
                                   out_dir=str(OUTPUT_DIR))
    except Exception:
        logger.warning("text overlay fallback failed", exc_info=True)
        out = None
    if out:
        logger.info("text_add: FireRed misspelt %r as %r %d times; drew the text instead",
                    wanted, best, max(1, TEXT_ADD_TRIES))
        return out, " / ".join(wanted), " / ".join(wanted)
    return None, best, " / ".join(wanted)


# A pixel-local removal (a stray dog head on a poster) leaves the lettering and
# the boxes exactly where they were. Without carrying the layout record to the
# new file, the NEXT "замени нижнюю надпись" found no layout and refused (live,
# journey 25). Other categories repaint too much for the old boxes to hold.
_LAYOUT_SURVIVES = frozenset({"object_remove", "person_remove"})


def route_edit_request(ctx, image_path: str, instruction: str, **kw) -> tuple:
    category, out = _route_edit_request(ctx, image_path, instruction, **kw)
    if out and category in _LAYOUT_SURVIVES and not _image.load_layout_for(out):
        rec = _image.load_layout_for(image_path)
        if rec:
            layout = rec["layout"]
            # The removal was done on the pixels; the record must lose the box
            # too, or the next box edit redraws the removed thing.
            try:
                import draw_agent
                layout, _notes = draw_agent.edit_layout(ctx, layout, instruction)
            except Exception:
                logger.warning("removal: could not drop the box from the layout", exc_info=True)
            _image.save_layout_for(out, rec.get("prompt") or "", layout,
                                   width=rec.get("width") or 0, height=rec.get("height") or 0,
                                   seed=rec.get("seed"))
    return category, out


def _route_edit_request(
        ctx, image_path: str, instruction: str,
        *, region: Optional[tuple] = None, seed: Optional[int] = None,
        timeout: int = 1900) -> tuple:
    """Single entry point for edit dispatch. Classifies `instruction`, extracts the
    per-category payload, and calls the matching specialized pipeline.

    Returns (category, output_path | None). `region` is forwarded to object
    insertion when the caller already has an explicit box.
    """
    category = classify_edit_intent(instruction)
    text = (instruction or "").strip()
    logger.info("Router: intent=%s for instruction=%r", category, text[:100])

    # A picture WE composed is edited through its boxes. Not a preference — a
    # whole-frame re-render re-imagines whatever it repaints, which is how "make
    # him blond" came back with a blonde woman's face over the man's. The layout
    # moves only what was asked for and redraws from the same seed.
    # Removals are the exception: the re-render reframed the scene, could draw
    # the removed thing again, and a second pixel pass then left floating petals
    # (journeys #39/#40, 2026-09-25). The user's verdict: the pixel removal on
    # the original (scene intact) is the good one. The box is dropped from the
    # record in route_edit_request.
    if (category in _LAYOUT_EDITABLE and category not in _LAYOUT_SURVIVES
            and _image.load_layout_for(image_path)):
        out = edit_via_layout(ctx, image_path, text, seed=seed,
                              fallback_on_problems=(category != "text_edit"))
        if out:
            return category, out
        # Lettering has no second option: a pixel pipeline cannot spell, and
        # falling through would reproduce the original false success.
        if category == "text_edit":
            return category, None
        logger.warning("layout edit failed for %s (%s) — falling back to the "
                       "pixel pipeline", category, image_path)
    if category == "text_add":
        out, _read, _want = add_text_verified(ctx, image_path, text, seed=seed)
        if not out and ctx is not None:
            try:
                ctx.last_text_add_failure = (_read, _want)
            except Exception:
                pass
        return category, out
    elif category == "text_edit":
        # No layout at all: nothing here can change what a sign says.
        logger.warning("text_edit on an image with no layout record (%s) — refusing "
                       "rather than re-rendering it blind", image_path)
        return category, None

    if category == "colour_convert":
        return category, convert_colour(image_path, text)
    if category == "transform":
        return category, transform_image(image_path, text)

    if category == "background_replace":
        # FireRed instruction edit. The old path synthesized a new background
        # with the old model and composited the matted subject over it; the old model was
        # removed from the product.
        bg = _image._strip_lead(text, r"^.*?\bbackground\b\s*(?:to|into|with|as|:)?")
        bg = bg or text
        instr = (f"Replace the background with {bg}. Keep every person and object in "
                 "the foreground exactly as they are: same face, pose, clothing, size "
                 "and position. Match the lighting of the new background.")
        if not re.search(r"\b(?:replace|change|swap|switch|put)\b|\bbackground\s+(?:to|with|into)\b",
                         text, re.I):
            # "make the background blurry" read as "Replace the background with blurry."
            instr = (f"{text.rstrip('. ')}. Keep every person and object in the foreground "
                     "exactly as they are: same face, pose, clothing, size and position.")
        return category, _image.edit_image_with_firered(ctx, image_path, instr,
                                                        seed=seed, timeout=timeout)

    if category == "background_remove":
        mode = edit_plan(text)["fill"]
        return category, _image.remove_background_with_comfy(ctx, image_path, mode=mode, timeout=timeout)

    if category in ("object_remove", "person_remove"):
        _image._REMOVAL_VERIFY["incomplete"] = False
        _image._REMOVAL_VERIFY["target"] = ""
        target = edit_plan(text)["target"] or ("the person" if category == "person_remove" else "the object")
        # FireRed only: the Big-LaMa fill was removed 2026-09-24 -- over fur,
        # fabric or hair it left a flat smudge that no check looked at.
        out = _image.remove_object_with_comfy(ctx, image_path, target, seed=seed, timeout=timeout)
        # Post-removal verification (fail-open: None/uncertain = accept). A recolor
        # or partial fill passes every geometric QA gate -- only an "is it still
        # there?" check catches it.
        if out and category == "object_remove":
            still = _image._region_present(ctx, out, target)
            if still is True:
                # Generative removal is STOCHASTIC: the same tile that fails with
                # one seed (shoes rendered translucent instead of removed) succeeds
                # with another (live A/B: seed 1234 failed, 777 clean). One retry
                # with a fresh seed before conceding.
                logger.warning("removal: %r still visible after generative removal — "
                               "retrying once with a fresh seed", target[:60])
                retry_seed = random.randint(1, 999_999_999)
                out3 = _image.remove_object_with_comfy(ctx, image_path, target,
                                                seed=retry_seed, timeout=timeout)
                if out3 and _image._region_present(ctx, out3, target) is not True:
                    logger.info("removal: fresh-seed retry succeeded for %r", target[:60])
                    out = out3
                else:
                    if out3:
                        out = out3   # newest attempt is still the better failure
                    logger.warning("removal: %r STILL visible after fresh-seed retry — "
                                   "flagging removal_incomplete so the agent can't claim "
                                   "success", target[:60])
                    _image._REMOVAL_VERIFY["incomplete"] = True
                    _image._REMOVAL_VERIFY["target"] = target
        return category, out

    if category == "object_insert":
        # FireRed adds the object by instruction and places it itself; the old
        # masked old-model insert needed a box and painted a patch into it.
        obj = _image._strip_lead(text, r"^(?:please\s+)?(?:add|insert|put|place|include|draw\s+in|добавь|вставь|помести|нарисуй)\b")
        obj = obj or text
        instr = (f"Add {obj}. It must look like a natural part of the scene, with "
                 "matching lighting, perspective and style. Keep everything else in "
                 "the image exactly the same.")
        return category, _image.edit_image_with_firered(ctx, image_path, instr,
                                                        seed=seed, timeout=timeout)

    if category in ("upscale", "restore", "outpaint"):
        # No engine left for these: upscale/restore ran on ESRGAN models, outpaint
        # on the old model, and FireRed (the only edit engine) can do neither. Refuse
        # with a reason the tool layer turns into an honest answer instead of
        # letting the request fall through to a whole-frame re-render.
        logger.info("Router: %s is not supported any more (FireRed-only product)", category)
        _image._INPAINT_FAILURE["reason"] = "unsupported"
        return category, None

    if category == "multi_op":
        return category, _orchestrate_multi_op(ctx, image_path, text, seed=seed, timeout=timeout)

    if category == "relight":
        prompt = _image._strip_lead(text, r"^.*?\b(?:relight|re-?light|light(?:ing)?|освещени)\b\s*(?:the\s+\w+\s*)?(?:to|with|as|like|:)?")
        prompt = prompt or text
        return category, _image.relight_image_with_comfy(ctx, image_path, prompt, seed=seed, timeout=timeout)

    # Localized generative edits (face/clothing/product/subject): prefer the
    # CONTAINED path (mask -> inpaint -> composite-back) so only the named region
    # changes and identity/face/background are preserved by construction. Style
    # transfer is inherently whole-image, so it stays on FireRed.
    if category in ("face_edit", "clothing_edit", "product_edit", "subject_edit"):
        # Ground the region in what's actually visible (SAM3/vision inventory) BEFORE
        # redrawing, so we mask a region the segmenter can find rather than burning a
        # full render on a phrase it can't locate.
        if edit_plan(text)["resizes"]:
            # A size/proportion fix changes the OUTLINE: the new head is bigger or
            # smaller than the mask of the old one, so the contained path's QA
            # rejects it as leaking every time (live 2026-09-27: "fix the head
            # proportions" failed twice, picture unchanged). Whole-frame FireRed.
            logger.info("Router: proportion/size change -> whole-frame FireRed")
            # Not _firered_instruction(region=""): that reads "Change the it to: X".
            instr = (f"Correct the proportions: {text}. Keep the same person, face, "
                     "clothing, pose, framing and background; change only the size "
                     "and proportions that were asked about.")
            return category, _image.edit_image_with_firered(ctx, image_path, instr,
                                                            seed=seed, timeout=timeout)
        region, result_prompt, denoise = _image._extract_edit_target(ctx, text, image_path=image_path)
        if region == "absent":
            # The thing to change isn't in the frame. Redrawing whole-frame would
            # invent it and wreck the rest; report instead of producing a bad image.
            logger.info("Router: edit target not present in image; skipping redraw")
            return category, None
        if region:
            # FireRed is the only edit engine: it instruction-edits the whole
            # tile (shape/texture aware), and the feathered composite-back limits
            # writes to the region. The masked-fill engines once chosen here
            # (the old model, BrushNet, FLUX.1-Fill) were removed from the product.
            engine = "firered"
            out = _contained_edit_validated(
                ctx, image_path, region, result_prompt, denoise=denoise,
                seed=seed, timeout=timeout, engine=engine)
            if out:
                return category, out
            if ctx is not None and getattr(ctx, "is_cancelled", None) and ctx.is_cancelled():
                logger.info("Router: cancelled after contained-edit stage")
                return category, None
            logger.warning("Contained edit failed/rejected for region=%r; "
                           "falling back to whole-frame FireRed", region)

    # Fallback / style_transfer: FireRed whole-image instruction edit.
    # style_transfer gets its own instruction builder: _firered_instruction's
    # region="" branch produces "Change the it to: X. Keep everything else the
    # same", which reads as "restyle one subject, preserve the frame" — a
    # contradiction for a whole-canvas restyle that came back live (2026-09-20)
    # as a cropped, partial cartoon redraw missing the title text and a person.
    if category == "style_transfer":
        instr = _image._firered_style_instruction(ctx, text, image_path=image_path)
    else:
        instr = _image._firered_instruction(ctx, "", text, removal=False)
    return category, _image.edit_image_with_firered(ctx, image_path, instr, seed=seed, timeout=timeout)


def _contained_edit_validated(ctx, image_path, region, result_prompt, *,
                              denoise=0.75, seed=None, timeout=1900, engine="firered"):
    """Run the contained edit and gate it on identity/containment.

    For a non-face region the face must remain the same person; if the mask escaped
    onto the face (identity cosine collapses) we retry once tighter, then give up so
    the caller can fall back. Gate is skipped when face-id tooling is unavailable or
    the edit target IS the face. Returns a validated path or None.
    """
    edits_face = _image._item_attributes(ctx, region).get("head") is True
    attempts = [(denoise, 12), (min(denoise, 0.55), 6)]   # retry tighter on failure
    for i, (dn, grow) in enumerate(attempts):
        if ctx is not None and getattr(ctx, "is_cancelled", None) and ctx.is_cancelled():
            return None
        out = _image.edit_region_contained_with_comfy(
            ctx, image_path, region, result_prompt, denoise=dn, grow=grow,
            seed=seed, timeout=timeout, engine=engine)
        if not out:
            return None
        if edits_face:
            return out          # identity change is the point; accept
        try:
            import identity_metrics as idm
            cos = idm.identity_cosine(image_path, out)
        except Exception as exc:
            logger.info("identity gate unavailable (%s); accepting contained edit", exc)
            return out
        if cos is None:
            return out          # no face to protect — containment already guaranteed
        if cos >= 0.90:
            logger.info("Contained edit PASSED identity gate: cosine=%.3f (region=%r)",
                        cos, region)
            return out
        logger.warning("Contained edit identity cosine=%.3f < 0.90 (attempt %d) — "
                       "mask may have escaped onto the face; %s", cos, i + 1,
                       "retrying tighter" if i + 1 < len(attempts) else "rejecting")
    return None


def _orchestrate_multi_op(ctx, image_path: str, text: str, *,
                          seed: Optional[int] = None, timeout: int = 1900) -> Optional[str]:
    """Decompose a multi-step request into ordered single-ops and apply them in
    sequence, threading each step's output into the next. Removals/background ops
    run before insertions/style so later steps see the cleaned scene.

    Returns the final image path, or the last successful intermediate, or None.
    """
    clauses = edit_plan(text)["steps"] or [text]

    # stable ordering: destructive/background first, then edits, then additive/finishing
    _ORDER = {"background_remove": 0, "person_remove": 1, "object_remove": 1,
              "background_replace": 2, "relight": 3, "face_edit": 4, "clothing_edit": 4,
              "product_edit": 4, "subject_edit": 4, "style_transfer": 5,
              "object_insert": 6, "outpaint": 7, "restore": 8, "upscale": 9}
    ordered = sorted(clauses, key=lambda c: _ORDER.get(classify_edit_intent(c), 4))

    current = image_path
    applied = 0
    for clause in ordered:
        cat = classify_edit_intent(clause)
        if cat == "multi_op":  # avoid infinite recursion on a mis-split clause
            cat = "subject_edit"
        logger.info("Multi-op step %d: %r -> %s", applied + 1, clause[:60], cat)
        try:
            _c, out = route_edit_request(ctx, current, clause, seed=seed, timeout=timeout)
        except Exception as exc:
            logger.warning("Multi-op step failed (%s): %s", clause[:40], exc)
            out = None
        if out and os.path.exists(out):
            current = out
            applied += 1
        else:
            logger.warning("Multi-op step produced no output, continuing: %r", clause[:40])
    logger.info("Multi-op done: %d/%d steps applied", applied, len(ordered))
    # Geometry verification: each single-op now restores its own input canvas, so
    # a chain should preserve the original dimensions end-to-end (no compounding
    # downscale — RC-5). Verify and, if a step still drifted, restore the canvas.
    if applied and current and current != image_path:
        src = _image._source_dims(image_path)
        out = _image._source_dims(current)
        if src and out and src != out:
            logger.warning("Multi-op geometry drift %s -> %s; restoring to source", src, out)
            try:
                from PIL import Image
                fixed = os.path.splitext(current)[0] + "_geofix.png"
                with Image.open(current) as im:
                    im.convert("RGB").resize(src, Image.LANCZOS).save(fixed)
                current = fixed
            except Exception as exc:
                logger.warning("Multi-op geometry restore failed: %s", exc)
        elif src and out:
            logger.info("Multi-op geometry preserved: %s == %s", src, out)
    return current if applied else None

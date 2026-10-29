"""Image-EDITING tool handlers: inspect, inpaint, transfer, hands, artifacts, find.

Lifted out of tools.py. Every handler takes (ctx, state, args) and returns the
string the model sees — this is the back half of the tool table, not a new
abstraction over it.

The render seam, and why three handlers read it through `tools`:
generate_image_with_refinement and
_render_budget_exhausted are replaced by eight suites at 18 sites, and every
one of them assigns onto the TOOLS module (tools.generate_image_with_refinement
= fake). Binding those names here at import time would copy them once and
ignore every later patch — the stub would be dropped and a real ComfyUI render
would run inside the test suite.

So _handle_generate_image, _handle_generate_video and _handle_redraw_image do
`import tools as _t` at CALL time and go through _t.<name>. tools owns the
seam; this module borrows it per call. That keeps all 18 patch sites working
untouched, and it is the same durable-seam rule used across this refactor:
reach a dependency through the module that owns it, never by value across a
split.

llm, prompts, identity_metrics and search stay CALL-time imports inside the
functions that need them, exactly as before: hoisting them here would drag the
model stack into every import of the tool table.
"""
import contextlib
import functools
import logging
import math
import os
import re
from typing import Optional

import config

import image as image_mod
import tool_graph
from tool_context import _edit_source, _remember, _set_current_image

from seams import choice, complaint_kinds, dep, module, intent_read, text_model, vision_model, yes_no

logger = logging.getLogger("assistant.tools")


def _image(ctx):
    """The image engine for this turn: `ctx.image_engine` when one was handed in (a suite's
    stand-in renderer), the `image` module otherwise."""
    return getattr(ctx, "image_engine", None) or image_mod


def _is_absence_check(check: str, ctx=None) -> bool:
    """The inspect_image check asks whether something is gone or still there
    (verifying a removal). The model's yes/no; it replaced a word list
    («still», «осталась», «исчез»...). Model down -> False, the plain verdict."""
    check = (check or "").strip()
    if not check:
        return False
    if os.getenv("F5_TEST_RUN") and not os.getenv("INTENT_LIVE"):
        own = dep(ctx, "absence_read", None)   # suites: absence_read(check) -> bool
        return bool(own and own(check))
    return _absence_read(check)


@functools.lru_cache(maxsize=256)
def _absence_read(check: str) -> bool:
    try:
        import llm
        ans = llm.call_llm_simple(
            None, "Answer yes or no only.",
            "Does this check on a picture verify that something was REMOVED or erased -- "
            "asking whether it is gone or still left over? Checking that things are kept, "
            "added, still recognisable, or about a colour, size or look is no.\n\nCheck: «%s»"
            % check[:500], temperature=0.0, max_tokens=5) or ""
        return ans.strip().lower().startswith(("yes", "да"))
    except Exception:
        logger.warning("absence check: model read failed", exc_info=True)
        return False


def _note_source(state, source) -> None:
    """Record which picture an editing tool starts from, so the delivery layer
    can register the result as a VERSION of it. A free-text edit ("надень на
    кота шляпу") would otherwise produce a picture with no parent, and the next
    edit would ask "Which picture?" between the two."""
    if source:
        state["image_derived_from"] = str(source)


# Shown BEFORE and AFTER, the inspector says what stayed «unchanged» -- the people and
# places the edit was not about; graph_personality reads the labels instead.
PAIR_INSPECTION = "Inspection of the current image (BEFORE vs AFTER):"


def _where(box) -> str:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    v = "top" if cy < 1 / 3 else "bottom" if cy > 2 / 3 else "middle"
    h = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "centre"
    return "centre" if (v, h) == ("middle", "centre") else f"{v} {h}"


def _change_evidence(before: str, after: str) -> str:
    """The measured pixel change, as a note for the inspector. Shown the two whole frames, it
    said «the two images are identical» about a round clock cleanly removed from a station
    roof (bench/removal_verdict_live.py). Close-ups of the changed part on top made it worse
    (4 pictures: it judged the BEFORE close-up and called removed things still there)."""
    import identity_metrics
    try:
        got = identity_metrics.changed_box(before, after)
    except Exception:
        logger.debug("change measure failed", exc_info=True)
        return ""
    if not got:
        # Opened away is not the same as unchanged: a 2 px outline goes too. Said unchanged
        # only when the raw difference is nil as well; a JPEG re-encode (0.13% raw) and real
        # removed glasses (0.15% raw) overlap, so between the two the note says nothing.
        raw = identity_metrics.changed_fraction(before, after, thresh=30).get("overall", 1.0)
        return ("\n\nMeasured: the two pictures are the same, pixel for pixel -- the edit "
                "changed nothing visible." if raw < 0.0002 else "")
    frac, box = got
    return (f"\n\nMeasured: the edit changed {frac * 100:.1f}% of the picture, in the "
            f"{_where(box)} part. The pictures are NOT identical there; look at that part.")


def _handle_inspect_image(ctx, state, args: dict) -> str:
    source = state.get("image_path") or getattr(ctx, "last_image_path", None)
    if not source or not os.path.exists(source):
        return (
            "[TOOL ERROR] No image to inspect — generate one with generate_image or load one "
            "first. Tell the user there is no picture to look at yet."
        )
    check = (args.get("check") or "").strip() or "Describe exactly what is in the image."
    ctx.set_stage("Looking at the picture")
    logger.info("Tool: inspect_image(check=%r, source=%s)", check[:80], source)
    # «что общего у двух фото»: the chat's earlier pictures, newest last, each
    # read on its own and labelled, so the answer can compare them.
    _n = int(args.get("earlier") or 0)
    if _n > 0:
        _before = [p for p in (getattr(ctx, "image_undo", None) or [])
                   if p and p != source and os.path.exists(p)][-_n:]
        if _before:
            _parts = []
            for _i, _p in enumerate(_before + [source], 1):
                _sub = _handle_inspect_image(ctx, {**state, "image_path": _p, "edit_before": None},
                                            {"check": check})
                _parts.append(f"Picture {_i} of {len(_before) + 1}"
                              f"{' (the latest)' if _p == source else ''}:{chr(10)}{_sub}")
            return "\n\n".join(_parts)

    analyze_image_with_llm = vision_model(ctx)
    from prompts import IMAGE_INSPECT_PROMPT
    # NO prefill. A closed-<think> prefill makes these fine-tunes return EMPTY for a
    # vision task (verified live — same bug that broke evaluate_image). Vision is also
    # flaky when a non-multimodal chat model holds the VRAM, so try twice.
    # The same measured-exposure evidence the outer judge gets: this vision
    # model calls a normally lit picture "extremely dark" and then labels every
    # element MISSING, which sent the agent into a second 70-second re-render of
    # a picture that was already right (the recolour edit).
    import exposure as _exposure
    _measured = _exposure.evidence(source)
    # After an edit, BEFORE and AFTER in one call: shown the result alone, the inspector
    # said «nothing changed, it stayed as it was» about a swap that had replaced a man
    # with a sponge (live via the debug user, 2026-10-13).
    _before = state.get("edit_before")
    _pair = ([("BEFORE the edit:", _before), ("AFTER the edit (judge this one):", source)]
             if _before and _before != source and os.path.exists(_before) else None)
    # ...and whether the people the edit was not about are still the same people: side
    # by side a changed face shows, rated alone it passes as «a man». No example object in
    # the duplicate question: «a second helmet» in it made the checker report «the second
    # helmet is MISSING» on a clean edit (2026-10-17).
    _ask = check + ("\n\nTwo pictures: BEFORE and AFTER the edit. Say first what changed "
                    "between them, then whether every person the edit was not about is "
                    "still the same person (face, features, count), then name anything AFTER "
                    "has twice or newly that the edit did not ask for, then judge the check "
                    "on AFTER." if _pair else "")
    if _pair:
        _ask += _change_evidence(_before, source)
    result = ""
    for _ in range(2):
        result = analyze_image_with_llm(
            ctx=ctx, image_path=None if _pair else source, image_paths=_pair,
            user_text=_ask + ("\n\n" + _measured if _measured else ""),
            system_prompt=IMAGE_INSPECT_PROMPT,
        ) or ""
        if result.strip():
            break
    logger.info("inspect_image result: %s", result.strip()[:400].replace("\n", " | "))
    if _measured and result.strip() and (
            complaint_kinds(ctx)(ctx, [result])[0] in ("dark", "blind")):
        logger.info("inspect_image: darkness complaint contradicted by the measured "
                    "exposure -- treated as a pass")
        result = ("The picture is normally exposed (measured). The inspector's only "
                  "complaint was that it looked dark, which the measurement "
                  "contradicts; nothing else was flagged.")
    if not result.strip():
        # Honest, non-fabricating fallback. Do NOT pretend we saw the image, and do
        # NOT push the agent into a fix/inspect loop — just report the check is
        # unavailable so it answers the user plainly instead of spiralling.
        return (
            "[TOOL ERROR] Could not inspect the image — the vision model returned "
            "nothing (no multimodal model available right now). Do NOT guess what is "
            "in the image and do NOT keep re-editing to 'fix' unverified flaws. Tell "
            "the user the picture is ready but you could not automatically verify the "
            "details."
        )

    # Route the model to the right next step. The inspector labels each element
    # PRESENT / PARTIAL / MISSING / DISTORTED (see IMAGE_INSPECT_PROMPT). For a
    # REMOVAL verification ("is the puddle still there?") the labels INVERT —
    # MISSING/ABSENT means the removal SUCCEEDED — so the generic "MISSING = flaw,
    # fix it" rule would wrongly trigger an extra pass that ruins a clean result
    # (the over-correction the user hit). Detect a removal/absence check and route
    # accordingly, leaving the actual judgement to the model.
    verdict = result.upper()
    if _is_absence_check(check, ctx=ctx):
        footer = (
            "\n\nNext step: this was a REMOVAL check. If the thing you removed is now "
            "gone — or only a faint trace/shadow remains — that is SUCCESS: STOP "
            "editing, make NO more tool calls, and answer the user (briefly note any "
            "faint leftover). Run another inpaint_image pass ONLY if the object is "
            "still CLEARLY and substantially present. Do not chase faint traces — "
            "extra passes regenerate the area and make the picture worse."
        )
    elif any(k in verdict for k in ("MISSING", "DISTORTED", "PARTIAL")):
        footer = (
            "\n\nNext step: a flaw was flagged. Fix it ONLY if it is part of what the "
            "user actually asked for, using inpaint_image (region = the flawed object "
            "as a simple noun, instructions = the desired correct look), then "
            "inspect_image once more. If the flaw is something the user never asked "
            "about — e.g. a defect that was already in the original photo — LEAVE IT "
            "ALONE and answer the user, briefly mentioning it; chasing such defects "
            "with more edits usually makes the picture worse. If the same fix has "
            "already failed twice, stop and tell the user honestly what is wrong."
        )
    else:
        footer = (
            "\n\nNext step: everything checks out. STOP editing now — no more tool "
            "calls, no extra 'improvement' edits (they risk ruining a good result). "
            "Answer the user."
        )
    head = PAIR_INSPECTION if _pair else "Inspection of the current image:"
    return f"{head}\n{result}{footer}"


def _edit_checkpoint(ctx, args: dict) -> str:
    """The model that renders an instruction edit. Always FireRed: the base
    Qwen-Image-Edit alternative (the GUI 🧬 toggle and the tools' 'engine'
    argument) was removed from the product."""
    return "firered"


def _undo(ctx, state) -> str:
    """«нет, верни как было» is the previous FILE, not a redraw of it."""
    said = f'{state.get("user_input_original") or ""} {state.get("user_input") or ""}'
    hist = getattr(ctx, "image_undo", None) or []
    prev = hist[-1] if hist and said.strip() and intent_read(ctx)(None, said.strip())["undo"] else ""
    if not (prev and os.path.exists(prev)):
        return ""
    ctx.image_undo = hist[:-1]
    _set_current_image(ctx, state, prev)
    return ("[done] The picture before the last change is back, exactly as it was -- "
            "nothing was redrawn. It is on screen. Briefly confirm, in the user's language.")


def _adds_object(ctx, region: str, instructions: str) -> bool:
    """The edit puts something NEW on the picture (glasses on a face, a caption, a hat) and
    the named part stays. Such an edit goes to the whole-frame add path: through the
    region's mask the added thing was cut at the mask's edge -- aviator glasses lost their
    temple and a caption its first letters on a face mask (2026-10-22). Asked directly: a
    keyword pre-gate read «wearing sunglasses» as a face edit and never asked.
    Scored on bench/additive_route_eval.py."""
    return choice(ctx)(
        "An image edit names a part of the picture and what to do there:\n{text}\n\n"
        "added = a new thing appears on, over or next to that part: glasses over the eyes, a "
        "hat on a head, a moustache on a face, a cup on a table, a balloon in the sky, a "
        "missing piece of an object, a caption. "
        "changed = the part itself becomes different: another colour, size, shape or "
        "material, other clothes, or something else in its place.",
        f"part: {region}\nwhat to do: {instructions}", ("added", "changed"), "changed") == "added"


def _reshapes_face_feature(ctx, region: str, instructions: str) -> bool:
    """The edit changes the shape or size of a feature of a face (nose, lips, eyes, ears, chin).
    Those stay on the mask path: the crop zooms in on the feature and the face around it is
    pasted back untouched; on the whole frame Klein barely grew a nose and aged the face."""
    return choice(ctx)(
        "An image edit names a part of the picture and what to do there:\n{text}\n\n"
        "feature = it changes the shape or size of a feature of a face: nose, lips, mouth, eyes, "
        "eyebrows, ears, chin, cheeks. other = anything else: hair, clothes, objects, hands, "
        "colours, the background.",
        f"part: {region}\nwhat to do: {instructions}", ("feature", "other"), "other") == "feature"


def _handle_inpaint_image(ctx, state, args: dict) -> str:
    source = _edit_source(ctx, state)
    _note_source(state, source)
    _undone = _undo(ctx, state)
    if _undone:
        return _undone
    if not source or not os.path.exists(source):
        return (
            "[TOOL ERROR] No image to edit — generate an image first with generate_image, "
            "then call inpaint_image. Tell the user there is no picture to work on yet."
        )

    instructions = (args.get("instructions") or "").strip()
    region = (args.get("region") or "").strip()
    if not instructions:
        return (
            "[TOOL ERROR] inpaint_image needs 'instructions' describing the desired result "
            "for the region (e.g. 'wearing black glasses'). Ask the user what to change."
        )
    if not region:
        # Never guess a default region — silently inpainting the face when the user
        # meant something else is far worse than asking for a retry.
        return (
            "[TOOL ERROR] inpaint_image needs 'region' — the thing in the image to change, "
            "as a simple noun (e.g. 'dress', 'hair', 'sky', 'background'). Re-call the tool "
            "with the region the user meant."
        )

    # Region-aware: also catches result-style phrasings ("bare neck, no scarf",
    # "без шарфа") and take-off/сними verbs the plain verb regex misses —
    # those get routed as "change the scarf to ..." and come back recoloured.
    removal = _image(ctx)._is_removal_of(instructions, region, ctx=ctx)
    # A region that names an ABSENCE ("the area where the vase was") is not a
    # thing to remove: the thing is already gone. The
    # model re-ran a removal on that phrase after a good one; grounding
    # picked a patch of wall and the delivered picture got a pasted light square.
    if removal and _image(ctx).names_former_place(ctx, region):
        return ("[TOOL ERROR] region names where something WAS, so it is already removed. "
                "Do not remove again: look at the current picture (inspect_image) and "
                "answer the user.")
    # Clear any stale still-visible verdict from a PREVIOUS removal so paths that
    # don't run the post-removal check (whole-frame mode, contained non-route edits)
    # can't inherit it and falsely report failure on a successful edit.
    if hasattr(image_mod, "_REMOVAL_VERIFY"):
        _image(ctx)._REMOVAL_VERIFY["incomplete"] = False
        _image(ctx)._REMOVAL_VERIFY["target"] = ""

    # Short-circuit for additive instructions: "add/insert/place X in/above/through
    # the Y" with an explicit region Y. Building synth = "change the sky to fried
    # chicken" → classify → subject_edit → inpaint_region → replaces the ENTIRE sky.
    # Instead, hand the router an explicit "add" request: it moves the boxes of
    # our own Ideogram picture, and has FireRed add the object to a user's photo.
    engine = _edit_checkpoint(ctx, args)
    manual_whole = ((getattr(ctx, "image_edit_engine", "auto") or "auto")
                    .strip().lower() == "firered_whole")
    # The instructions alone read as an insert when they name the result:
    # «сделай красный круг зелёным» came as region='red circle', instructions='a green
    # circle' and a second, green circle was added next to the red one.
    if not removal and not manual_whole and region and _adds_object(ctx, region, instructions):
        ctx.set_stage("Inserting object")
        logger.info("Tool: inpaint_image additive-shortcut -> route_edit_request "
                    "anchor=%r instructions=%r source=%s", region, instructions[:80], source)
        synth = (instructions if region.lower() in instructions.lower()
                 else f"{instructions} ({region})")
        category, new_path = _image(ctx).route_edit_request(ctx, source, synth)
        edit_path = f"route_edit_request:{category}:additive"
        # skip the rest of the routing block; fall through to delivery section
    else:
        # Synthesize one natural-language edit request from the (region, instructions)
        # pair so the intent router can pick the most appropriate SPECIALIZED pipeline
        # (subject-preserving bg-replace, localized object/person removal, ...) instead
        # of the old one-size-fits-all whole-image FireRed path. The router falls back
        # to FireRed for genuinely generative edits (face/clothing/subject/style).
        if removal:
            synth = f"remove the {region}"
        elif region.lower() in ("background", "bg", "backdrop", "фон"):
            synth = f"change the background to {instructions}"
        else:
            synth = f"change the {region} to {instructions}"
        category = _image(ctx).classify_edit_intent(synth, ctx=ctx)
        ctx.set_stage("Removing from the picture" if removal else "Editing the picture")
        logger.info("Tool: inpaint_image(region=%r, instructions=%r, removal=%s) -> intent=%s, source=%s",
                    region, instructions[:80], removal, category, source)

        _has_layout = bool(_image(ctx).load_layout_for(source))
        import image_lettering_remove as _lett
        _lettering_out = None
        if removal and not _has_layout and _lett.is_lettering_removal(region, instructions, ctx=ctx):
            # "Remove all the lettering": the mask is every piece the OCR reads,
            # not the one strip a segmenter finds for a phrase ("INSGATE"
            # survived on the left). A picture with a
            # layout still goes through its boxes below.
            ctx.set_stage("Removing the lettering")
            _lettering_out = _lett.remove_lettering_and_logos(ctx, source)
        if _lettering_out:
            new_path = _lettering_out
            edit_path = "remove_lettering:ocr-mask"
            category = "object_remove"
        elif category == "colour_convert":
            # Exact arithmetic over the WHOLE frame — there is no region to mask
            # and nothing to re-imagine. Handled before every generative branch
            # below, including the layout one: re-rendering a picture to grey
            # returned a different subject, in colour.
            new_path = _image(ctx).convert_colour(source, instructions or synth, ctx=ctx)
            edit_path = "convert_colour"
        elif _image(ctx).classify_edit_intent(instructions or "", ctx=ctx) == "transform":
            import image_router
            new_path = image_router.transform_image(source, instructions, ctx=ctx)
            edit_path = "transform"
        elif category in ("text_edit", "text_add") or (_has_layout and category in _image(ctx)._LAYOUT_EDITABLE):
            # A picture WE composed is edited through its boxes, and this branch
            # runs BEFORE the FireRed category list and before whole-frame mode.
            # Both of those repaint pixels, and repainting re-imagines whatever it
            # covers: asked to make a man blond, the whole-frame edit blended a
            # blonde woman's face over his. The layout moves only what was asked
            # for and redraws the scene from the same seed. The whole-frame toggle
            # is deliberately overridden — it chooses between renderers, and the
            # question here is not which renderer.
            _cat, new_path = _image(ctx).route_edit_request(ctx, source, synth)
            edit_path = f"route_edit_request:{_cat}"
        elif manual_whole:
            # GUI "Whole frame" switch ON: bypass the category router and every masked
            # pipeline; the whole image goes through the FireRed instruction edit.
            edit_path = "inpaint_region_with_comfy:whole-frame-mode"
            new_path = _image(ctx).inpaint_region_with_comfy(ctx, source, region, instructions,
                                                 removal=removal, engine=engine)
        elif category == "style_transfer":
            # A style change (anime/cartoon/oil painting/...) applies to the WHOLE
            # picture, never to just the region the agent happened to name (it called
            # inpaint_image with e.g. region="her"/"the girl", not "whole image").
            # The contained pipeline segmented that region, cropped
            # it, re-rendered ONLY the crop in the new style, and composited it back
            # over the untouched original -- so the result came back with a hard seam
            # cutting through the head, half photorealistic and half anime. Passing an
            # empty region here forces the whole-frame branch inside
            # inpaint_region_with_comfy (no segmentation, no crop, no composite).
            edit_path = "inpaint_region_with_comfy:style-whole-frame"
            new_path = _image(ctx).inpaint_region_with_comfy(ctx, source, "", instructions,
                                                 removal=removal, engine=engine)
        elif (category in ("subject_edit", "face_edit", "clothing_edit", "product_edit", "multi_op")
              and not removal and not _reshapes_face_feature(ctx, region, instructions)):
            # Klein on the whole frame, the annotator's box lock around the change: the mask path
            # pasted a halo round a blue shirt, a rectangle of brown hair at the face box and a
            # floating fist (bench/edit_path_ab.py, 9 real edits: whole frame better 4, mask 1).
            edit_path = "klein_whole_frame"
            new_path = _image(ctx)._edit_image_with_firered_once(
                ctx, source, f"change the {region} to {instructions}")
        elif category in ("subject_edit", "face_edit", "clothing_edit", "product_edit",
                          "multi_op"):
            edit_path = "inpaint_region_with_comfy"
            new_path = _image(ctx).inpaint_region_with_comfy(ctx, source, region, instructions,
                                                 removal=removal, engine=engine)
        else:
            _cat, new_path = _image(ctx).route_edit_request(ctx, source, synth)
            edit_path = f"route_edit_request:{_cat}"

    # AGENT/UI BOUNDARY: this is the exact object handed back to the agent and then
    # the GUI (state['image_path'] -> final['image_path'] -> images_panel). Log the
    # full delivery record and HARD-REJECT any intermediate scratch tile so a
    # cropped working artifact can never be surfaced as a final result.
    logger.info("Tool: inpaint_image DELIVERY [BUILD_ID=%s] edit_path=%s source=%s -> returned=%s",
                getattr(image_mod, "IMAGE_BUILD_ID", "?"), edit_path, source, new_path)
    _image(ctx).log_edit_decision(
        request=f"region={region!r} instructions={instructions!r}",
        classifier=category, tool=f"inpaint_image -> {edit_path}",
        workflow=("contained-firered / firered fallback"
                  if edit_path == "inpaint_region_with_comfy"
                  else "route_edit_request specialized pipeline"),
        returned_file=new_path, source=source,
        extra={"removal": removal})
    new_path = _image(ctx).assert_deliverable(new_path, where="tools._handle_inpaint_image",
                                            source_path=source)
    if not new_path or not os.path.exists(new_path):
        failure = _image(ctx)._INPAINT_FAILURE.get("reason", "server_error")
        if failure == "unsupported":
            detail = (
                f"'{category}' is not something this assistant can do any more: "
                "upscaling, photo restoration and canvas expansion (outpaint) had "
                "their engines removed, and the remaining editor cannot do them. "
                "Nothing was changed. Tell the user plainly that this is not "
                "supported; do NOT offer to try again or claim it worked."
            )
        elif failure == "region_absent":
            detail = (
                f"the vision model confirmed that '{region}' is not present in the image "
                "at all — nothing was changed. Ask the user which part they meant, or "
                "describe what they want to change more precisely."
            )
        elif failure == "not_found":
            detail = (
                f"the segmenter ran but found no '{region}' region (the output was "
                "identical to the source image). The region name may not match what is "
                "visible — try a more specific or different noun for the same object."
            )
        elif failure == "qa_rejected":
            detail = (
                f"the edit of '{region}' was rendered, but the visual check found it misplaced "
                "or leaking outside that area, so it was NOT applied (the picture is unchanged, "
                "and a whole-frame redraw was deliberately not tried). Retry once — the engine "
                "is stochastic — or ask the user to name the area more precisely."
            )
        elif failure == "bad_mask":
            detail = (
                f"'{region}' was detected but the segmentation mask covered too large an "
                "area of the frame (would damage the rest of the picture). The segmenter "
                "grabbed a large neighbouring object. Try a more specific region name that "
                "isolates just the target object, or use a narrower spatial description. If "
                "the request is a whole new scene rather than one thing in this picture, call "
                "redraw_image instead -- inpaint_image changes one area only."
            )
        elif failure == "noisy_mask":
            detail = (
                f"the segmenter could not cleanly locate '{region}' — it returned a "
                "fragmented, scattered mask (noise, not a real region), which was rejected "
                "rather than used to render garbage. Try a clearer, more concrete region "
                "name for something actually visible in the image, or have the user draw "
                "the area manually."
            )
        elif category == "text_add":
            import image_router as _ir
            _rd, _wt = getattr(ctx, "last_text_add_failure", None) or ("", "")
            return _ir.TEXT_ADD_FAILED_NOTE.format(n=_ir.TEXT_ADD_TRIES, best=_rd, want=_wt)
        elif category == "text_edit":
            # Not a server failure — a refusal. The picture has no layout, so the
            # words in it cannot be corrected by any tool we have. Say so; the
            # alternative the model would otherwise reach for (re-render the frame)
            # is precisely what produces new garbled letters and a false success.
            return _image(ctx)._NO_LAYOUT_NOTE
        else:
            detail = "the ComfyUI server failed to process the request"
        return (
            f"[TOOL ERROR] Could not edit the image — {detail}. "
            "Do NOT claim a new image was produced."
        )

    # Through _set_current_image: set by hand here, the edit kept no BEFORE, and the
    # inspector asked «is the rest as it was?» answered MISSING about a clean recolour
    # («a single image, not a comparison») and the edit was run a second time.
    _set_current_image(ctx, state, new_path)
    # Fold the edit into the running subject prompt so a later redraw_image (which
    # re-renders from ctx.last_image_prompt) keeps this change instead of reverting it.
    base_prompt = (getattr(ctx, "last_image_prompt", "") or "").strip()
    if base_prompt:
        ctx.last_image_prompt = (f"{base_prompt}, (no {region}, removed)" if removal
                                 else f"{base_prompt}, ({region}: {instructions})")
    dep(ctx, "_remember", _remember)(ctx, state, "inpaint", f"Edited {region}: {instructions}", {"path": new_path})
    # Post-edit efficacy: if the removal pipeline's own "is it still there?" check
    # confirmed the target is STILL visible, hand the agent an HONEST failure signal
    # rather than a success message — the model must NOT narrate a fabricated
    # partial-success ("I removed the left shoe") when nothing actually changed.
    if removal and getattr(image_mod, "_REMOVAL_VERIFY", {}).get("incomplete"):
        logger.info("Removal verification: %r still visible in %s", region, new_path)
        return (
            f"[TOOL ERROR] The removal did NOT succeed — after editing, the vision check "
            f"confirms the '{region}' is STILL visible in the image (nothing effective was "
            f"removed). Do NOT claim it was removed, and do NOT invent a partial result "
            f"(e.g. 'removed one of them'). Tell the user plainly that the '{region}' could "
            f"not be removed, or retry once with a more specific region naming exactly what "
            f"to erase.")
    verb = f"removed the {region}" if removal else f"{region} → {instructions}"
    _probs = list(getattr(image_mod, "_LAYOUT_EDIT_PROBLEMS", []) or [])
    if _probs:
        _image(ctx)._LAYOUT_EDIT_PROBLEMS.clear()     # reported once, for this edit
        return (
            f"Image edited ({verb}) and saved: {new_path}\n"
            "[LETTERING CHECK FAILED] The text reader still sees problems in the new "
            "picture: " + " | ".join(_probs) + "\nDo NOT say it is done or correct. Show "
            "it, tell the user plainly that the lettering came out wrong (say what is "
            "wrong) and offer to try again.")
    return (
        f"Image edited ({verb}) and saved: {new_path}"
    )


def _handle_transfer_image(ctx, state, args: dict) -> str:
    """Multi-image reference transfer: take a thing from one or more reference
    images and apply it to the target (the most recent image). Reference images
    come from ctx.reference_images (everything loaded/pasted this session); the
    newest is the target, the earlier ones are the references."""
    imgs = [p for p in (getattr(ctx, "reference_images", None) or []) if p and os.path.exists(p)]
    # de-dupe preserving order
    seen = set(); imgs = [p for p in imgs if not (p in seen or seen.add(p))]
    target = _edit_source(ctx, state)
    if target and os.path.exists(target) and target not in imgs:
        imgs.append(target)
    if len(imgs) < 2:
        return ("[TOOL ERROR] transfer_image needs at least TWO images loaded (a target "
                "plus one reference). The user must paste/load the reference image(s) and "
                "the target first. Only " + str(len(imgs)) + " image(s) are available.")
    target = target if (target and os.path.exists(target)) else imgs[-1]

    # Person-detection heuristic for the CHAT path (the GUI Transfer tab selects the
    # target explicitly): users naturally paste the person first and the thing to
    # transfer (hat/dress/tattoo) second, which makes the accessory the "newest
    # image" and thus the target — the transfer then runs backwards. If the default
    # target has NO face but exactly one other loaded image does, that person photo
    # is almost certainly the intended canvas: swap. Ambiguous cases (no faces
    # anywhere, several faces, detector unavailable) keep the newest-image default.
    try:
        import identity_metrics as _idm
        if _idm.face_box(target) is None:
            with_face = [p for p in imgs if p != target and _idm.face_box(p) is not None]
            if len(with_face) == 1:
                logger.info("transfer_image: newest image %s has no face; using person "
                            "photo %s as target instead",
                            os.path.basename(target), os.path.basename(with_face[0]))
                target = with_face[0]
    except Exception as exc:
        logger.info("transfer_image: face-based target heuristic skipped: %s", exc)

    # By CONTENT: the graph edits a byte-identical _working_input_* copy of the
    # target, so the target's own file came back as a "reference"
    # (👗 + a hoodie photo): refs=2, two passes, the first "dressed" her in her
    # own sweater -- twice the time, twice the damage.
    def _key(p):
        try:
            return os.path.getsize(p), open(p, "rb").read(65536)
        except OSError:
            return p
    seen = {_key(target)}
    references = []
    for p in imgs:
        k = _key(p)
        if p != target and k not in seen:
            seen.add(k)
            references.append(p)
    if not references:
        return ("[TOOL ERROR] transfer_image needs a reference image different from the "
                "target; only copies of the target are loaded.")

    instructions = (args.get("instructions") or "").strip()
    roles = args.get("roles") or []
    if not instructions:
        return ("[TOOL ERROR] transfer_image needs 'instructions' describing what to "
                "transfer, e.g. 'put the hat from the reference onto the person'.")

    shown = list(dict.fromkeys(_key(p) for p in imgs))
    if len(roles) == len(shown) == len(references) + 1 and _key(target) in shown:
        # roles counted every photo of the album, the target too: «the man from the
        # first photo» became identity_reference, and the armour photo's clothing_source
        # role fell off the end -- the man's face was pasted instead of the armour.
        roles = [r for i, r in enumerate(roles) if i != shown.index(_key(target))]
    refs = []
    for i, p in enumerate(references):
        role = None
        if i < len(roles) and roles[i] in _image(ctx)._ALL_ROLES:
            role = roles[i]
        refs.append(_image(ctx).ReferenceImage(p, role))

    ctx.set_stage("Transferring between images")
    logger.info("Tool: transfer_image(target=%s, refs=%d, roles=%r, instr=%r)",
                os.path.basename(target), len(refs), roles, instructions[:80])
    new_path = dep(ctx, "plan_and_execute_transfer", _image(ctx).plan_and_execute_transfer)(
        ctx, target, refs, instructions)
    new_path = _image(ctx).assert_deliverable(new_path, where="tools._handle_transfer_image",
                                            source_path=target)
    if not new_path or not os.path.exists(new_path):
        return ("[TOOL ERROR] Could not complete the transfer — the editor returned no "
                "result. Tell the user the transfer failed; do NOT claim it worked.")
    _set_current_image(ctx, state, new_path)
    ctx.reference_images = (ctx.reference_images or []) + [new_path]
    dep(ctx, "_remember", _remember)(ctx, state, "transfer", f"Reference transfer: {instructions}", {"path": new_path})
    missing = _transfer_missing(ctx, references[-1], new_path, instructions)
    return (
        f"Transfer complete and saved: {new_path}\n"
        + (f"Compared with the reference, still missing or wrong: {missing}\n" if missing else "")
    )


def _transfer_missing(ctx, ref: str, result: str, instructions: str) -> str:
    """What of the reference's item did not make it onto the result, the two pictures in
    ONE vision call: described from the garment alone, the logo and the flag patch of a
    quarter-zip were lost and nobody compared. "" when nothing is missing or no model."""
    try:
        analyze_image_with_llm = vision_model(ctx)
        ans = " ".join((analyze_image_with_llm(
            ctx, temperature=0.0, max_tokens=120,
            image_paths=[("REFERENCE:", ref), ("RESULT:", result)],
            user_text=f"The edit was: {instructions}. List, in a few words each, the details "
                      "of the transferred item in REFERENCE (logos, patches, prints, collar, "
                      "closures, colour, shape) that are missing or wrong in RESULT. Answer "
                      "NONE when everything came across.") or "").split())
    except Exception:
        return ""
    return "" if not ans or ans.upper().startswith("NONE") else ans[:400]


def _handle_swap_person(ctx, state, args: dict) -> str:
    """🔁 the person `person` (from the left) on the current photo replaced by who is on
    the other loaded picture -- the same person_swap.swap the 🔁 button runs: one pass
    on the whole photo, only that person's region pasted back."""
    person_swap = module(ctx, "person_swap")
    # A second try swaps on the ORIGINAL photo: on the first result the person is gone
    # and «person 4 of 3» does not exist (live via the debug user, 2026-10-13).
    target = (state.get("swap_source") or state.get("image_path")
              or getattr(ctx, "last_image_path", None))
    if not (target and os.path.exists(target)):
        return "[TOOL ERROR] swap_person needs the photo with the people loaded first."
    refs = [p for p in (getattr(ctx, "reference_images", None) or [])
            if p and os.path.exists(p) and os.path.abspath(p) != os.path.abspath(target)
            and p != state.get("swap_done")]
    if not refs:
        return ("[TOOL ERROR] swap_person needs a second picture: who goes in. Ask the "
                "user to send it.")
    boxes = person_swap.people(target)
    n = int(args.get("person") or 0)
    if not boxes:
        return "[TOOL ERROR] No people were found on the photo; nothing to swap."
    if not 1 <= n <= len(boxes):
        return (f"[TOOL ERROR] The photo has {len(boxes)} people (counted from the left); "
                f"person {n} does not exist. Ask the user which one.")
    ctx.set_stage("Swapping a person")
    logger.info("Tool: swap_person(target=%s, person=%d/%d, ref=%s, wish=%r)",
                os.path.basename(target), n, len(boxes), os.path.basename(refs[-1]),
                (args.get("wish") or "")[:80])
    new_path = person_swap.swap(ctx, target, boxes[n - 1], refs[-1], args.get("wish") or "")
    if not new_path or not os.path.exists(new_path):
        return ("[TOOL ERROR] The swap failed, nothing was changed. Tell the user so; do "
                "NOT claim it worked.")
    _set_current_image(ctx, state, new_path)
    state["swap_done"] = new_path
    state["swap_source"] = target
    ctx.reference_images = (ctx.reference_images or []) + [new_path]
    dep(ctx, "_remember", _remember)(ctx, state, "swap", f"Person {n} swapped", {"path": new_path})
    return (f"Swap complete and saved: {new_path}\n"
            "Call inspect_image to check it. Other editors are locked on it; if it is off, "
            "call swap_person again with a clearer wish, or tell the user what is off.")


def _handle_fix_hands(ctx, state, args: dict) -> str:
    """Repair deformed hands/fingers in the current image via the MeshGraphormer
    Hand Refiner (geometry-guided depth-ControlNet inpaint of only the hand)."""
    source = _edit_source(ctx, state)
    _note_source(state, source)
    if not source or not os.path.exists(source):
        return ("[TOOL ERROR] No image to fix — generate or load an image first, then "
                "call fix_hands. Tell the user there is no picture to work on yet.")

    engine = _edit_checkpoint(ctx, args)
    ctx.set_stage("Fixing the hands")
    logger.info("Tool: fix_hands(source=%s, engine=%s)", source, engine)
    try:
        new_path = _image(ctx).fix_hands(ctx, source, engine=engine)
    except Exception as exc:
        logger.exception("fix_hands raised")
        return (f"[TOOL ERROR] The hand-repair pipeline failed with an internal error: {exc}. "
                "Tell the user it could not complete; do NOT claim the hands were fixed.")

    logger.info("Tool: fix_hands DELIVERY [BUILD_ID=%s] source=%s -> returned=%s",
                getattr(image_mod, "IMAGE_BUILD_ID", "?"), source, new_path)
    _image(ctx).log_edit_decision(
        request="fix hands/fingers", classifier="hand_repair",
        tool="fix_hands -> MeshGraphormer Hand Refiner",
        workflow="meshgraphormer depth-controlnet masked inpaint",
        returned_file=new_path, source=source, extra={})
    new_path = _image(ctx).assert_deliverable(new_path, where="tools._handle_fix_hands",
                                            source_path=source)
    if not new_path or not os.path.exists(new_path):
        return ("[TOOL ERROR] Could not fix the hands — the refiner returned no result "
                "(ComfyUI may have failed, or no hand was detected in the image). Do NOT "
                "claim the hands were fixed; ask the user to confirm a hand is visible, "
                "or try inpaint_image on the specific hand region.")

    _set_current_image(ctx, state, new_path)
    dep(ctx, "_remember", _remember)(ctx, state, "fix_hands", "Repaired the hands/fingers", {"path": new_path})
    return (
        f"Hands repaired and saved: {new_path}\n"
        "Call inspect_image to verify the hand now has five natural fingers and no "
        "artifacts before telling the user it is done; if it is still malformed, you may "
        "call fix_hands once more (the refiner is stochastic) or inpaint_image the hand."
    )


def _handle_fix_artifact(ctx, state, args: dict) -> str:
    """Repair an awkward spot / artifact in the current image: locate a named region
    (Florence/SAM segmentation), regenerate ONLY that region with the contained
    FireRed engine to remove the flaw, and composite it back so everything else
    stays pixel-identical. The conversational counterpart of the GUI '🩹 Fix artifact'
    button (which uses a hand-drawn mask instead of a region phrase)."""
    source = _edit_source(ctx, state)
    _note_source(state, source)
    if not source or not os.path.exists(source):
        return ("[TOOL ERROR] No image to fix — generate or load an image first, then call "
                "fix_artifact. Tell the user there is no picture to work on yet.")
    region = (args.get("region") or "").strip()
    if not region:
        return ("[TOOL ERROR] fix_artifact needs 'region' — a short noun phrase naming WHERE "
                "the flaw is so it can be located, e.g. 'the seam on the left shoulder', "
                "'the smear above the table', 'the background near her elbow'. Ask the user "
                "which area looks wrong.")
    issue = (args.get("issue") or "").strip()
    engine = _edit_checkpoint(ctx, args)
    instruction = issue or (
        "Repair this area: fix any artifact, awkward transition, smear or distortion and "
        "make it look natural and seamless, matching the surrounding texture, colour and "
        "lighting.")
    ctx.set_stage("Fixing the flaw")
    logger.info("Tool: fix_artifact(region=%r, issue=%r, engine=%s) source=%s",
                region, issue[:80], engine, source)
    try:
        new_path = _image(ctx).edit_region_contained_via_firered(
            ctx, source, region, instruction, protect_face=True, engine=engine)
    except Exception as exc:
        logger.exception("fix_artifact raised")
        return (f"[TOOL ERROR] The artifact-repair pipeline failed with an internal error: {exc}. "
                "Tell the user it could not complete; do NOT claim it was fixed.")

    logger.info("Tool: fix_artifact DELIVERY [BUILD_ID=%s] source=%s -> returned=%s",
                getattr(image_mod, "IMAGE_BUILD_ID", "?"), source, new_path)
    _image(ctx).log_edit_decision(
        request=f"fix artifact region={region!r} issue={issue!r}",
        classifier="artifact_repair", tool=f"fix_artifact -> {engine}",
        workflow="contained masked inpaint (Florence/SAM region + FireRed)",
        returned_file=new_path, source=source, extra={"engine": engine})
    new_path = _image(ctx).assert_deliverable(new_path, where="tools._handle_fix_artifact",
                                            source_path=source)
    if not new_path or not os.path.exists(new_path):
        return ("[TOOL ERROR] Could not fix that area — the region may not have been located, "
                f"or the edit was rejected by quality checks. Do NOT claim it was fixed; ask the "
                f"user to name the spot differently (a clearer noun for where '{region}' is), or "
                "suggest the GUI '🩹 Fix artifact' button to draw the area by hand.")

    _set_current_image(ctx, state, new_path)
    # Avoid a double article ("the the smear") when the region phrase already leads with one.
    region_disp = region if region.lower().startswith(("the ", "a ", "an ")) else f"the {region}"
    dep(ctx, "_remember", _remember)(ctx, state, "fix_artifact", f"Repaired {region_disp}", {"path": new_path})
    return (
        f"Repaired {region_disp} and saved: {new_path}\n"
        "Call inspect_image to verify the flaw is gone and nothing else changed before "
        "telling the user it is done; if the area still looks wrong, retry fix_artifact "
        "(the engine is stochastic, a retry gives a different result)."
    )


def _handle_find_photo(ctx, state, args: dict) -> str:
    """Search the web for a real photo of someone/something, validate it actually
    shows a face, and POST it into the chat as the working image — so the user can
    then edit it with inpaint_image/redraw_image/transfer_image like any image they
    dropped in. This is the transparent alternative to the all-in-one reference-person
    mode: the agent fetches and shows the photo; the user drives the edit."""
    if not getattr(ctx, "web_search_enabled", True):
        return ("[TOOL ERROR] find_photo needs internet search, which is currently "
                "disabled. Ask the user to enable web search, or proceed without a photo.")
    query = (args.get("query") or "").strip()
    if not query:
        return ("[TOOL ERROR] find_photo needs a 'query' naming who/what to find a photo of, "
                "e.g. 'Sydney Sweeney portrait'.")
    want_face = bool(args.get("require_face", True))

    import search as _search
    ctx.set_stage("Finding a photo online")
    import time
    ts = int(time.time() * 1000)
    dest = os.path.join(str(_image(ctx).OUTPUT_DIR), f"webphoto_{ts}.jpg")
    try:
        _image(ctx).OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.debug("could not create the output folder", exc_info=True)
    photo = dep(ctx, "fetch_reference_photo", _search.fetch_reference_photo)(ctx, query, dest, require_face=want_face)
    if not photo or not os.path.exists(photo):
        return (f"[TOOL ERROR] Could not find a usable {'photo with a clear face' if want_face else 'photo'} "
                f"for '{query}'. Tell the user no suitable photo was found; do NOT claim one was posted.")

    # Make it the working image AND a reference (so a later transfer can use it too).
    ctx.last_image_path = photo
    ctx.reference_images = (ctx.reference_images or []) + [photo]
    state["image_path"] = photo
    state["image_status"] = "success"
    dep(ctx, "_remember", _remember)(ctx, state, "find_photo", f"Posted a web photo for: {query}", {"path": photo})
    logger.info("Tool: find_photo(%r) -> %s", query[:60], os.path.basename(photo))
    return (
        f"Found a photo for '{query}' and posted it to the chat as the current working image: {photo}\n"
        "It is now loaded — the user can ask for an edit (e.g. 'make the dress red') and it will be "
        "inpainted on THIS photo. Tell the user the photo is ready and ask what to change."
    )


def _render_budget_exhausted(state, what: str, cost: int = 1) -> Optional[str]:
    """Spend `cost` of this turn's full renders; returns a refusal when they run out.

    The agent is told to inspect its picture and fix what is wrong "until it is
    right" — good advice with no cost attached. One turn ran redraw →
    inspect → redraw → inspect → draw, each a full render, holding ComfyUI for
    minutes while other users queued. The self-correction loop is kept; it just
    gets a budget. On exhaustion the agent is told to deliver what it has and say
    what it could not fix, which is the honest outcome and the one the reply
    guards already expect.

    `cost` exists because a VIDEO is not one more still: a single H3 clip is a 33B
    transformer swapping through VRAM for many minutes. Charging it 1 would let one
    turn run three of them. It is charged the whole cap, so a clip is the last
    render of its turn and a second one is refused.
    """
    cap = max(1, int(getattr(config, "IMAGE_MAX_RENDERS_PER_TURN", 3) or 3))
    prior = int(state.get("_renders_used", 0))
    used = prior + max(1, int(cost))
    state["_renders_used"] = used
    if used > cap:
        logger.warning("%s refused: this turn already spent %d/%d full renders "
                       "(this one costs %d)", what, prior, cap, cost)
        noun = "clip" if "video" in what else "picture"
        return (
            f"[TOOL ERROR] This turn has already spent its rendering budget "
            f"({prior}/{cap}). Do NOT render again. Deliver the best {noun} you "
            f"already have and tell the user PLAINLY which part you could not get "
            f"right — an honest 'the lettering still came out wrong' is correct here. "
            f"If they want another attempt they will ask."
        )
    return None


_TRANSLIT = str.maketrans({"а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
                           "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
                           "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
                           "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
                           "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya"})


def _keep_quoted_names(description: str, state) -> str:
    """A name the user quoted in Cyrillic («Колосок») is lettering, verbatim.

    The model wrote 'Kolosok' into its description and the logos came out in
    Latin letters. A transliteration of the quoted name is
    swapped back; a quoted name missing altogether is stated at the end."""
    import graph_language as _gl
    own = str((state or {}).get("user_input_original") or (state or {}).get("user_input") or "")
    for q in _gl._QUOTED_LITERAL_RE.findall(own):
        lit = q[1:-1].strip()
        if not lit or not re.search("[а-яё]", lit, re.I) or lit in description:
            continue
        tr = lit.lower().translate(_TRANSLIT)
        pat = re.compile(re.escape(tr).replace("y", "[yi]").replace("kh", "(?:kh|h)"), re.I)
        if pat.search(description):
            description = pat.sub(f"«{lit}»", description)
        else:
            description += f' The lettering reads exactly «{lit}» in Cyrillic letters.'
    return description


def _user_look(ctx, state) -> str:
    """The look THIS message asks for: a drawn look, "photo", "none", or "" when
    there are no user words to read (a direct call). Only this message: «нарисуй
    кота в стиле аниме», then «нарисуй собаку» drew the dog in anime too
    (the style comes only from the current message). A redo («ещё раз») says
    no look of its own and repeats the request before it, look included."""
    from prompt_guard import user_words
    from ideogram_layout import _look
    st = state or {}
    text = user_words(str(st.get("user_input_original") or st.get("user_input") or "")).strip()
    if not text:
        return ""
    got = _look(text, ctx)
    if got != "none":
        return got
    try:
        redo = intent_read(ctx)(None, text)["redo"] in ("same", "changed")
    except Exception:
        redo = False
    if redo:
        for m in reversed(st.get("messages") or []):
            if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
                t = user_words(m["content"]).strip()
                if t and t != text:
                    return _look(t, ctx)
    return "none"


def _draws_the_sent_photos_people(ctx, state, source) -> bool:
    """A photo with a face came with this request, and the request draws that person."""
    sent = state.get("sent_photo")
    if not sent or sent != source or not dep(ctx, "exists", os.path.exists)(sent):
        return False
    import identity_metrics
    if not dep(ctx, "face_boxes", identity_metrics.face_boxes)(sent):
        return False
    from prompt_guard import user_words
    said = user_words(str(state.get("user_input_original") or state.get("user_input") or "")).strip()
    return bool(said) and yes_no(ctx)(
        "A user sent a photo of a person together with these words: {text}\n\nIs the picture "
        "they ask for of the person in that photo (named, or him/her/me/us/them) -- not a "
        "picture with nobody from the photo in it?", said, default=True)


def _handle_generate_image(ctx, state, args: dict) -> str:
    import tools as _t   # the render seam is owned by tools; read it at CALL time
    description = (args.get("description") or "").strip()
    if not description:
        return ("[TOOL ERROR] generate_image needs a 'description' of what to draw. "
                "Re-call it with a clear English description of the requested scene.")
    _undone = _undo(ctx, state)
    if _undone:
        return _undone

    # The user pressed ✏️ and described a CHANGE to the picture in front of them.
    # Drawing a new one throws their picture away and answers a question nobody
    # asked. Observed live: "add the name of a Polish philosopher to his shirt"
    # came back as a brand-new jar of honey. The system prompt already says this;
    # the prompt lost, so the rule is enforced here instead of hoped for.
    source = state.get("image_path") or getattr(ctx, "last_image_path", None)
    _note_source(state, source)
    if state.get("edit_intent") and source and os.path.exists(source):
        logger.warning("generate_image refused: this turn is an EDIT of %s "
                       "(routing to inpaint_image)", source)
        return (
            "[TOOL ERROR] This turn is an EDIT of the picture already loaded, not a "
            "request for a new one — the user pressed the edit button and described a "
            "change. Generating would discard their image. Call inpaint_image with the "
            "region to change and the requested change as instructions (or redraw_image "
            "if the WHOLE picture must be re-rendered). Do not call generate_image again "
            "for this turn."
        )

    # «Нарисуй кибер-Ленина … душит демона» sent WITH Lenin's photo was drawn from scratch:
    # a stranger's face and invented signs (live 2026-10-21). The people of a photo sent
    # with the request are drawn from their faces -- redraw_image's new-scene path.
    if _draws_the_sent_photos_people(ctx, state, source):
        logger.info("generate_image -> redraw_image: the photo sent with the request shows who to draw")
        return dep(ctx, "_handle_redraw_image", _handle_redraw_image)(ctx, state, {"instructions": description})

    _over = dep(ctx, "_render_budget_exhausted", _t._render_budget_exhausted)(state, "generate_image")
    if _over:
        return _over

    description = _keep_quoted_names(description, state)
    import object_refs
    import person_look
    draw_refs = dep(ctx, "object_refs", object_refs.find)(ctx, description)
    # A face is read off its photo, not off web text: the words for Lenin from the web said «often
    # wears glasses», and he was drawn in red sunglasses (debug chat 2026-10-26).
    looks = [dep(ctx, "face_words", object_refs.face_words)(ctx, who, photo)
             for who, photo, kind in draw_refs if kind == "person"]
    if any(looks):
        description += "".join("\n" + x for x in looks if x)
    else:
        description = person_look.with_look(description, ctx=ctx)     # «Ленин» -> bald, goatee, ...
    ctx.set_stage("Drawing a picture")
    steps, width, height, seed = args.get("steps"), args.get("width"), args.get("height"), args.get("seed")
    logger.info("Tool: generate_image(%s, steps=%s, %sx%s)", description, steps, width, height)

    # REFERENCE-PERSON MODE (opt-in, default off): for a real, named public figure,
    # fetch an actual web photo as the identity reference and render the requested
    # outfit/scene onto it (real likeness), instead of a from-scratch text2image
    # guess. Any failure falls through to the normal path below.
    if getattr(ctx, "reference_person_mode", False) and getattr(ctx, "web_search_enabled", True):
        try:
            ctx.set_stage("Finding a reference photo")
            ref_out, info = _image(ctx).generate_person_from_reference(ctx, description, seed=seed)
        except Exception as exc:
            logger.warning("reference-person mode errored (%s); using text2image", exc)
            ref_out, info = None, "exception"
        if ref_out and os.path.exists(ref_out):
            state["image_path"] = ref_out
            state["image_status"] = "success"
            state["image_score"] = 0
            ctx.last_image_path = ref_out
            ctx.last_render_path = ref_out          # rendered, not merely loaded
            ctx.last_render_status = "success"
            ctx.last_image_prompt = description
            dep(ctx, "_remember", _remember)(ctx, state, "generate", f"Reference-person image: {description}", {"path": ref_out})
            who = info.get("person") if isinstance(info, dict) else "the person"
            return (
                f"Image generated using a real reference photo of {who}: {ref_out}\n"
                "Mode: reference-person (web photo used as the identity reference).\n"
                "Next step: call inspect_image to confirm the likeness and the requested "
                "outfit/scene are right before answering."
            )
        logger.info("reference-person mode did not produce an image (%s); using text2image", info)

    # The kind of picture is the USER's call, not the agent's rewrite (see
    # ideogram_layout.enforce_user_look): read from their own words, this turn
    # first, then the last two requests («ещё раз» after «в стиле аниме»).
    _prev_look = getattr(ctx, "user_look", "")
    ctx.user_look = _user_look(ctx, state)
    ctx.draw_refs = draw_refs
    from prompt_guard import user_words
    ctx.draw_request = user_words(str(state.get("user_input_original") or state.get("user_input") or "")).strip()
    ctx.set_stage("Drawing a picture")
    try:
        result = dep(ctx, "generate_image_with_refinement", _t.generate_image_with_refinement)(
            ctx=ctx, description=description, steps=steps, width=width, height=height, seed=seed,
        )
    finally:
        ctx.user_look = _prev_look
        ctx.draw_refs = []
        ctx.draw_request = ""
    img_path = result.get("path")
    # A returned path must actually EXIST on disk before we treat it as a result. A
    # stale/bogus path from the renderer, or a temp file deleted between render and
    # hand-off, would otherwise be reported as "Image generated and saved: <path>"
    # (false success) — every other image handler already verifies existence; this one
    # did not. Fall through to the failure message if the file is missing.
    if img_path and not os.path.exists(img_path):
        logger.error("generate_image: renderer returned a non-existent path %r", img_path)
        img_path = None
        result = dict(result); result["status"] = "fail"  # keep status consistent w/ the missing file
    # Face enhance is NOT run automatically here — only on explicit user request.
    # (Auto-enhance was changing faces the user didn't ask to touch.)
    score = result.get("score", 0)
    attempts = result.get("attempts", 1)
    status = result.get("status", "fail")

    state["image_path"] = img_path or ""
    state["image_score"] = score
    state["image_attempt"] = attempts
    state["image_status"] = status

    if img_path:
        ctx.last_image_path = img_path
        # Recorded separately from last_image_path so a turn that runs out of
        # time can still hand over a picture it actually finished.
        ctx.last_render_path = img_path
        ctx.last_render_status = status
        ctx.last_image_prompt = result.get("prompt", description)
        dep(ctx, "_remember", _remember)(ctx, state, "generate", f"Image for task: {description}", {"path": img_path})
        quality_note = " (base quality — refinement unavailable)" if status == "partial" else ""
        # An ACCEPTED render is final for this turn. The renderer has already
        # judged it with the same vision model the agent would use; a second
        # look does not see more, it invents -- a 10/10
        # picture was once "fixed" for a shadow that was not there, lost its subject,
        # and was redrawn six more times. tool_graph refuses edits on it; the
        # user asks for changes in their own words, in their own turn.
        accepted = status == "success" and int(score or 0) >= tool_graph.ACCEPTED_SCORE
        state["fresh_render"] = {"path": img_path, "score": score,
                                 "status": status, "accepted": accepted}
        if accepted:
            return (
                f"Image generated and saved{quality_note}: {img_path}\n"
                f"Score: {score}/10, attempts: {attempts}, status: {status}\n"
                "The picture has already been checked against the request by the "
                "vision model and accepted. Do NOT inspect it again and do NOT edit "
                "it on your own: answer the user and deliver it as it is. If they "
                "want something changed, they will say so."
            )
        _why = str(result.get("reason") or "").strip()
        if status == "partial" and _why:
            # The checker rejected it and the user got "Вот три варианта" with a
            # garbled name on one of them: say what is wrong.
            return (
                f"Image generated and saved: {img_path}\n"
                f"Score: {score}/10, attempts: {attempts}, status: {status}\n"
                f"[QUALITY CHECK FAILED] The checker still sees: {_why[:400]}\n"
                "Deliver it, but tell the user plainly and briefly what came out wrong "
                "(say it in their language) and offer to redraw. Do not call it done "
                "or perfect, and do NOT redraw or edit it on your own this turn -- the "
                "renderer already tried twice; a third pass cropped the logo's name.")
        return (
            f"Image generated and saved{quality_note}: {img_path}\n"
            f"Score: {score}/10, attempts: {attempts}, status: {status}\n"
            "Next step: if the request named a specific person, action, or particular "
            "objects that must all be present, call inspect_image now to verify them "
            "before answering; for a simple generic scene, just answer the user."
        )
    gpu = _gpu_busy_error("Image generation")
    if gpu:
        return gpu
    reason = _image(ctx)._GENERATE_FAILURE.get("reason", "server_error")
    if reason == "refused":
        detail = ("the drawing model DECLINED this prompt — it returned its safety card "
                  "instead of a picture. Retrying the same words will be declined again. "
                  "Tell the user plainly that this description was refused and offer to "
                  "rephrase it")
    elif reason == "engine_failed":
        # Deliberate: the old model fallback is off so a fresh picture is always
        # composed from a layout. Without this the model reads a bare failure as
        # "the server is down" and tells the user to try later, when the real
        # answer is that this one render failed and asking again usually works.
        detail = ("the drawing engine failed on this request. A fresh picture is only "
                  "ever drawn by Ideogram, so nothing was substituted. Tell the user the "
                  "picture could not be drawn and offer to try again")
    else:
        detail = ("the ComfyUI image server encountered an error and could not complete "
                  "the request. Tell the user the image could not be generated")
    return (
        f"[TOOL ERROR] Image generation failed — no image was produced: {detail}. "
        "You MUST tell the user clearly that the image could not be generated; "
        "do NOT imply or claim that an image was created."
    )


def _gpu_busy_error(what: str):
    """The failure message for "we never even tried — the card is taken", or None.

    Asked ahead of whatever sub-reason the renderer recorded, and asked in one
    place rather than plumbed through every engine: a whole-card job holding the
    GPU is the true explanation and the only one the user can act on. Telling
    them "the server had an error, try again" while their own training run is
    what blocked the card sends them into a retry loop that cannot succeed for
    hours.

    Deliberately a RECENT REFUSAL rather than "is the card busy now": Ideogram
    can decline a prompt on content grounds during a training run, and blaming
    the trainer for that would be a confident wrong answer.
    """
    try:
        import comfy_client as _cc
        held = _cc.recent_gpu_refusal()
    except Exception:
        return None
    if not held:
        return None
    return (
        f"[TOOL ERROR] {what} was not attempted — the GPU is busy with another "
        f"whole-card job ({held}), typically a LoRA training run. Tell the user "
        "plainly that this is unavailable until that job finishes, name it, and "
        "do NOT offer to retry now."
    )


def _current_image_paths(ctx, state) -> list:
    """Every picture this turn could reasonably animate, newest last.

    There is no single "the images" slot: a chat may have one working image, or a
    handful the user just sent. Prefer an explicit multi-image list when a caller
    set one (the Telegram per-chat register does), else fall back to the single
    working image, and drop anything no longer on disk.

    Deduped by CONTENT, not just path. tg_bot._image_by_content's own docstring
    explains why a second path can exist for the same picture: the graph edits a
    byte-identical "working copy" (_working_input_<ts>.*) of whatever image it
    started from, under its own filename. Pressing Animate
    under a just-styled photo passed BOTH the styled file's own path and that
    working-copy snapshot of the SAME bytes into generate_video, which saw two
    distinct-looking images and picked first-frame/last-frame mode (morphing
    between two copies of one picture) instead of a clean single-image animate.
    """
    cands = []
    for src in (state.get("image_paths"), getattr(ctx, "recent_image_paths", None)):
        if isinstance(src, (list, tuple)):
            cands.extend(src)
    for one in (state.get("image_path"), getattr(ctx, "last_image_path", None)):
        if one:
            cands.append(one)
    seen_paths, seen_content, out = set(), set(), []
    for p in cands:
        if not p or p in seen_paths or not os.path.exists(p):
            continue
        seen_paths.add(p)
        try:
            key = (os.path.getsize(p), open(p, "rb").read(65536))
        except OSError:
            continue
        if key in seen_content:
            continue
        seen_content.add(key)
        out.append(p)
    return out


def _builds_on_chat_picture(ctx, state) -> bool:
    """A clip asked for with no photo in this message animates an earlier picture only
    when the words say so. A photo sent with the request, a ▶ plan or ▶️ Continue
    always does."""
    if state.get("image_data") or getattr(ctx, "video_plan_parts", None) or getattr(ctx, "continue_tail", ""):
        return True
    from prompt_guard import user_words
    said = user_words(str(state.get("user_input_original") or state.get("user_input") or "")).strip()
    return yes_no(ctx)(
        "A user asked for a video clip with these words: {text}\n\nDo they ask to animate "
        "or continue a picture already in the chat (this photo, the picture above, its "
        "people or scene) -- not a new scene described from scratch?", said, default=True)


def asks_for_video(ctx, state) -> bool:
    """The user's words this turn ask for a clip, or ask to redo the last one. A clip is
    minutes of GPU and the whole turn: «Почему он у тебя не лысый … ты нарисовал какого-то
    урода, а не Ленина» -- a complaint about a PICTURE -- started one."""
    from prompt_guard import user_words
    said = user_words(str((state or {}).get("user_input_original") or "")
                      + " " + str((state or {}).get("user_input") or "")).strip()
    if not said:     # no words to judge (a direct call): not ours to refuse
        return True
    # A button already said "video": ▶ under the plan card, or ▶️ Continue with its clip.
    # Live 2026-10-11 the ▶ rerun was re-asked here, the model said no, the turn failed.
    if getattr(ctx, "video_plan_parts", None) or getattr(ctx, "continue_tail", ""):
        return True
    from graph_fastpath import _IMAGE_ACTION_TOOLS
    try:
        r = intent_read(ctx)(None, said)
    except Exception:
        r = {}
    wants = set(r.get("wants") or ())
    # The router heard a picture: «нарисуй его за рулём…, едет, куры разбегаются» after a noodle
    # clip read as redo «changed», passed, and became a 9-minute clip (2026-10-24).
    if wants & _IMAGE_ACTION_TOOLS and "generate_video" not in wants:
        return False
    last = getattr(ctx, "last_video_path", None)
    if last and os.path.exists(last) and r.get("redo") in ("same", "changed"):
        return True
    # A shot script with no word «видео» is still a clip («Мужчина берёт банку … бросает
    # банку … опрокидывает стеллаж»); a complaint about a picture is not. The word list that
    # answered yes before this read also said yes to «что в этом видео?» and «какие
    # видеокарты лучше» (bench/asks_video_eval.py, 2026-10-25).
    return yes_no(ctx)(
        "A user wrote: {text}\n\nIs this a request to MAKE a video clip: animate a photo, a "
        "clip, a gif, a video note, a cartoon, or a scene that unfolds over time (people act "
        "one after another, move, speak lines)? Not a still picture, a complaint about a "
        "picture, a question about a video or a film, or a request to find or download one.",
        said, default=True)     # the agent already chose a clip: only a read «no» stops it


def _video(ctx):
    """The video engine for this turn: `ctx.video_engine` when one was handed in (a suite's
    fake renderer), the `video` module otherwise. Handlers never reach the module directly,
    so a fake never has to be written over the module for everyone else."""
    import video
    return getattr(ctx, "video_engine", None) or video


def _clips(ctx):
    """Frame and tail cutting of delivered clips: `ctx.clip_tools`, else `tg_continue`."""
    import tg_continue
    return getattr(ctx, "clip_tools", None) or tg_continue

def _bridged(ctx, clip: str, part: str) -> str:
    """`part` led by the move that brings in what it needs and the clip's last frame lacks."""
    import tempfile
    tg_continue = _clips(ctx)
    video_mod = _video(ctx)
    frame = os.path.join(tempfile.mkdtemp(prefix="vidbridge_"), "last.jpg")
    return video_mod.bridge_part(ctx, frame, part) if tg_continue.seed_frame(clip, frame) else part


def _speech_ok(ctx, clip: str, lines: list) -> bool:
    """Whisper hears `lines` in the clip's own sound. A broken check passes: it must never
    block a render."""
    import subprocess
    import tempfile
    try:
        from audio import transcribe_audio_file
        wav = os.path.join(tempfile.mkdtemp(prefix="vidsay_"), "a.wav")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", clip, "-vn", "-ac", "1", "-ar", "16000", wav],
                       capture_output=True, check=True, timeout=60)
        heard = transcribe_audio_file(ctx, wav, lang_hint="ru") or ""
        logger.info("video chain: heard %r for %s", heard[:160], lines)
        return _video(ctx).heard_lines(lines, heard)
    except Exception:
        logger.warning("video chain: the speech check failed", exc_info=True)
        return True


def _voice_sim(clip: str, voices) -> Optional[float]:
    """Similarity of the clip's voice to the reference it was given; None when there is no
    reference or it cannot be told."""
    if not voices:
        return None
    import voice_clone
    return voice_clone.speaker_similarity(clip, voices[0])


def _render_chain_parts(ctx, first: str, parts: list, args: dict, chain: str, start: int = 1) -> int:
    """Parts 2..n of a long script as the next links of `chain` (each continues the
    previous part's saved latent). `parts` are ALL the parts as (prompt, seconds) from video.prepare_prompt,
    made before the card was taken: nothing in here calls a model. `start` is the first part's link
    (a continued chain goes on from its own last link). Returns the last link made; a failure or a
    cancel stops there and the parts so far are still joined and delivered."""
    import tempfile
    tg_continue = _clips(ctx)
    video_mod = _video(ctx)
    last, prev = start, first
    work = tempfile.mkdtemp(prefix="chainjoins_")
    ctx.chain_joins = []           # frames on both sides of each join, looked at once the card is free
    for i, part in enumerate(parts[1:], start=2):
        if getattr(ctx, "is_cancelled", None) and ctx.is_cancelled():
            break
        ctx.video_part = f"{i}/{len(parts)}"
        ctx.set_stage(f"Generating a video ({i}/{len(parts)})")
        said = video_mod.quoted_lines(part[0])
        r, crashed = {}, False
        for attempt in (1, 2):
            try:
                r = video_mod.generate_video(ctx, part[0],
                                             seconds=part[1], prepared=True,
                                             aspect=args.get("aspect") or "", seed=args.get("seed"),
                                             chain=(chain, start + i - 1), size_from=prev)
            except Exception:
                logger.exception("video chain: part %d crashed", i)
                crashed = True
                break
            if attempt == 2 or not r.get("path"):
                break
            # The part keeps the camera unless it opens on a cut; drawn flowers that cover
            # the frame then show at its edges (live 2026-10-15: leaves came in at part 2).
            still = video_mod.INSERT_HEAD.strip() not in part[0] and video_mod.INSERT_BACK.strip() not in part[0]
            drawn = video_mod.drawn_over(r["path"])
            drift = video_mod.overlay_drift(r["path"]) if still and drawn is None else None
            if (drift is not None and drift > video_mod.OVERLAY_DRIFT_MAX) or \
                    (drawn is not None and drawn > video_mod.DRAWN_OVER_MAX):
                logger.warning("video chain: part %d is covered over (edge drift %s, drawn over %s); "
                               "once more with a new seed", i, drift, drawn)
                continue
            if not said or _speech_ok(ctx, r["path"], said):
                break
            logger.warning("video chain: part %d did not say %s; once more with a new seed", i, said)
        if crashed:
            break
        new = r.get("path")
        if not (new and os.path.exists(new)):
            logger.warning("video chain: part %d produced no clip (%s)", i, r.get("reason"))
            break
        # A link opens on its protected head (a remake of the previous link up to the handover)
        # and the stitch keeps it from the frame after. The raw last frame beside the raw first
        # one is ~2 s apart, and every smooth join read as a jump (live 2026-10-25).
        head = int(video_mod.HERRGOTT_CONTEXT)
        before, after = os.path.join(work, f"end_{i}.jpg"), os.path.join(work, f"start_{i}.jpg")
        if tg_continue.first_frame(new, before, head - 1) and tg_continue.first_frame(new, after, head):
            ctx.chain_joins.append((before, after))
        last, prev = start + i - 1, new
    return last


def _render_remaining_parts(ctx, path: str, parts: list, args: dict, people=(), voices=(), fresh_from="") -> str:
    """Parts 2..n of a long script, each continuing the clip so far from its real
    last frame (its tail as <Video 1>, that frame as <Picture 1>), joined on. A part
    that fails or a cancel stops here: the clip made so far is still delivered.
    `fresh_from`: a talking head on one still -- every part starts from that still instead of
    the clip so far: each continuation inherited the drift of the one before and the face
    turned into a horror mask by the end (owner, 2026-10-19). The parts are cut together."""
    import tempfile
    tg_continue = _clips(ctx)
    video_mod = _video(ctx)
    speech_ok = getattr(ctx, "speech_ok", None) or _speech_ok       # seams for the suites
    voice_sim = getattr(ctx, "voice_sim", None) or _voice_sim
    work = tempfile.mkdtemp(prefix="vidparts_")
    joins = []                     # (last frame of the clip so far, first frame of the new part)
    for i, part in enumerate(parts[1:], start=2):
        if getattr(ctx, "is_cancelled", None) and ctx.is_cancelled():
            break
        first_try, r = None, {}
        ctx.video_part = f"{i}/{len(parts)}"
        ctx.set_stage(f"Generating a video ({i}/{len(parts)})")
        frame = os.path.join(work, f"seed_{i}.jpg")
        tail = os.path.join(work, f"tail_{i}.mp4")
        if fresh_from:
            tg_continue.seed_frame(path, frame)         # only for the join check below
        elif not (tg_continue.seed_frame(path, frame) and tg_continue.cut_tail(path, tail)):
            logger.warning("video parts: could not take the end of part %d", i - 1)
            break
        # The new people of a continuation ride on EVERY part: given to part 1 only, the
        # guy from the user's кружок walked out in part 2 as a stranger (live 2026-10-12).
        # Pinned frames take no pictures, so with people the reference path is used.
        pinned = video_mod.motion_context_on() and not people and not fresh_from
        if pinned:
            part = video_mod.bridge_part(ctx, frame, part)
        said = video_mod.quoted_lines(part)
        try:
            if fresh_from:
                # Takes of the still, each checked for its lines and the still's eyes; the best
                # stays. Even ending on the still, one part in six had its eyes go light and wide,
                # and its retake pushed in to an extreme close-up -- unchecked (live 2026-10-20).
                takes = []
                for attempt in (1, 2, 3):
                    r = video_mod.generate_video(ctx, part, images=[fresh_from], keyframe=True,
                                                 aspect=args.get("aspect") or "",
                                                 seed=args.get("seed") if attempt == 1 else None)
                    if not r.get("path"):
                        break
                    said_ok = not said or speech_ok(ctx, r["path"], said)
                    eyes = video_mod.eye_drift(fresh_from, r["path"], light=False)
                    takes.append((not said_ok, eyes if eyes is not None else 0.0, attempt, r))
                    if said_ok and (eyes is None or eyes <= 0):
                        break
                    logger.warning("video parts: part %d take %d did not say %s or drifted off the "
                                   "still's eyes (%s)", i, attempt, said, eyes)
                if takes:
                    best = min(takes, key=lambda t: t[:3])
                    r = best[3]
                    if best[2] != takes[-1][2]:
                        logger.info("video parts: part %d keeps take %d of %d", i, best[2], len(takes))
                    r = {**r, "path": _pull_light(video_mod, fresh_from, r["path"], i)}
            for attempt in (1, 2) if not fresh_from else ():
                if pinned:
                    r = video_mod.generate_video(ctx, video_mod.CONTINUE_CTX_PREFIX
                                                 + video_mod.with_scene(part, " ".join(parts[:i - 1])),
                                                 context_video=tail,
                                                 aspect=args.get("aspect") or "", seed=args.get("seed"))
                else:
                    look = "".join(f" The character of <Picture {k}> looks exactly as in that picture."
                                   for k in range(2, len(people) + 2))
                    # The voice rode part 1 only and each part copied the sound of the one before:
                    # by the end it was another man's voice (owner, 2026-10-19).
                    if voices:
                        look += " The speaker's voice stays exactly the voice of <Audio 1>."
                    r = video_mod.generate_video(ctx, video_mod.CONTINUE_PREFIX + part + look,
                                                 images=[frame, *people], videos=[tail], audios=list(voices),
                                                 aspect=args.get("aspect") or "", seed=args.get("seed"))
                # no chain check here before: a part whose opening lines H3 skipped went into the clip
                if attempt == 2 or not r.get("path"):
                    break
                said_ok = not said or speech_ok(ctx, r["path"], said)
                sim = voice_sim(r["path"], voices)
                if said_ok and (sim is None or sim >= video_mod.VOICE_MIN_SIM):
                    break
                logger.warning("video parts: part %d did not say %s or drifted off the voice (%s); "
                               "once more with a new seed", i, said, sim)
                first_try = (r, sim) if said_ok else None
                args = {**args, "seed": None}
            if first_try and r.get("path") and first_try[1] is not None:
                again = voice_sim(r["path"], voices)
                if again is not None and again < first_try[1]:
                    r = first_try[0]        # the second take is further from the voice than the first
                    logger.info("video parts: part %d keeps its first take (voice %.2f > %.2f)", i, first_try[1], again)
        except Exception:
            logger.exception("video parts: part %d crashed", i)
            break
        new = r.get("path")
        if not (new and os.path.exists(new)):
            logger.warning("video parts: part %d produced no clip (%s)", i, r.get("reason"))
            break
        first = os.path.join(work, f"first_{i}.jpg")
        if tg_continue.first_frame(new, first) and not fresh_from:      # a cut back to the still is meant
            joins.append((frame, first))
        # pinned: the new part starts on the frame after the old one ends -- butt-join.
        # fresh_from: both sides of the cut are the still (each part ends on it too), so the
        # doubled frame goes and a one-frame blend is enough; a longer fade mixed two mouths.
        joined = (video_mod.join_pinned(path, new) if pinned
                  else video_mod.join_continuation(path, new, fade=1 / 24, drop_frames=1) if fresh_from
                  else video_mod.join_continuation(path, new))
        if not joined:
            logger.warning("video parts: joining part %d failed", i)
            break
        path = video_mod._adopt_output(joined, video_mod.OUTPUT_DIR)
        logger.info("video parts: %d/%d joined -> %s", i, len(parts), os.path.basename(path))
    ctx.video_join_note = _check_joins(ctx, joins)
    return path


def _pull_light(video_mod, still: str, path: str, i: int) -> str:
    """A talking-head part whose light drifted, pulled back to the still's -- on the still's framing
    only: run 18's push-in to the face came out with a blue beard and orange lips (2026-10-20)."""
    light = video_mod.look_drift(still, path)
    if light is None or light + video_mod.LOOK_MAX_SHIFT - video_mod.LOOK_PULL_SHIFT <= 0:
        return path
    if not video_mod.framing_kept(still, path):
        logger.warning("video parts: part %d light is off (%.2f) but its framing too; left as it is", i, light)
        return path
    pulled = video_mod.match_look(still, path)
    if not pulled:
        return path
    logger.info("video parts: part %d light pulled back to the still (%.2f -> %s)",
                i, light, video_mod.look_drift(still, pulled))
    return pulled


def _check_joins(ctx, joins: list) -> str:
    """Every join of a chained clip -- the frame before it and the frame after -- in ONE
    vision call, after all parts are made (per part, the vision model was loaded back
    beside H3 between every two parts, ~10 min each). Names the joins where the camera
    jumps or the person changes: part 2 re-framed and a ghost crossed the fade, and the
    reply called it seamless (2026-10-12). "" when all are smooth or no model."""
    if not joins:
        return ""
    try:
        analyze_image_with_llm = vision_model(ctx)
        pics = []
        for n, (a, b) in enumerate(joins, 1):
            pics += [(f"Join {n}, BEFORE:", a), (f"Join {n}, AFTER:", b)]
        ans = " ".join((analyze_image_with_llm(
            ctx, temperature=0.0, max_tokens=150, image_paths=pics,
            user_text="These are the frames on both sides of each join of one continuous "
                      "video. For each join where the camera framing jumps, the place changes "
                      "or a person becomes someone else, say which join and what jumps, in a "
                      "few words. Answer NONE when every join is continuous.") or "").split())
    except Exception:
        logger.warning("video joins: the check failed", exc_info=True)
        return ""
    logger.info("video joins: %d checked -> %s", len(joins), ans[:200] or "no answer")
    return "" if _says_none(ans) else ans[:400]


def _preserves_static_scene(ctx, request: str) -> bool:
    try:
        return not yes_no(ctx)("The user wrote: {text}. Do they request camera movement, a scene change, "
                               "or objects or people entering, leaving, appearing, disappearing or transforming?",
                               request, default=True)
    except Exception:
        return False


def _check_static_background(ctx, video_mod, path: str, request: str) -> str:
    try:
        drift = video_mod.overlay_drift(path)
        if drift is None or drift <= video_mod.OVERLAY_DRIFT_MAX:
            return ''
        if not _preserves_static_scene(ctx, request):
            return ''
        return 'The background may have changed unexpectedly.'
    except Exception:
        logger.exception('video background check failed')
        return ''


def _check_actions(ctx, path: str, parts: list, references=()) -> str:
    """The current request against the finished clip, in ONE vision call on frames spread over it.
    The reply said «Готово! швыряет шапку» over a clip where the hat stayed on (live 2026-10-18),
    and the user's own words were «не вижу чтобы он срал»: the writer of the reply saw the script,
    never the clip. Returns the actions the frames do not show, "" when all are there, nothing
    was asked or no model answered. Skipped for what cannot be judged: no script, no frames."""
    script = getattr(ctx, "video_request", "") or " ".join(str(p).strip() for p in parts if p)
    for rule in (video_mod_rules(ctx)):
        script = script.replace(rule, "")
    script = " ".join(script.split())[:900]
    if len(script) < 20:
        return ""
    import tempfile
    from pathlib import Path
    try:
        motion = None
        try:
            tracker = getattr(_video(ctx), 'track_simple_motion', None)
            motion = tracker(path) if references and callable(tracker) else None
            if motion and yes_no(ctx)("The user wrote: {text}. Did they request camera movement "
                                      "or a subject or scene transformation?", script, default=True):
                motion = None
        except Exception:
            logger.debug('motion tracking unavailable', exc_info=True)
            motion = None
        measured = ("A tracker measured the single foreground object's center in the frame: "
                    + ", ".join(f"{t:.1f}s x={x:.0%} y={y:.0%}" for t, x, y in motion)
                    + ". Positions are relative to the frame; compare them when judging movement. " if motion else "")
        look = module(ctx, "video_look")
        duration = look.duration_seconds(path)
        frames = look.extract_frames(path, look.sample_times(duration, every=1.5, max_frames=10),
                                     Path(tempfile.mkdtemp(prefix="actcheck_")))
        if len(frames) < 2:
            return ""
        ans = " ".join((vision_model(ctx)(
            ctx, temperature=0.0, max_tokens=200,
            image_paths=[(f"Video at {t:.1f}s:", str(f)) for t, f in frames]
                        + [(f"Original reference image {i} (not a video frame):", p)
                           for i, p in enumerate(references, 1)],
            user_text="These frames are in time order from ONE video. The user's request or approved plan:\n"
                      f"{script}\n"
                      "Compare subject position and appearance between the timestamped video frames to judge movement. "
                      "The original reference images are still pictures, not frames of the video; do not judge "
                      "video motion from them. Any changes the user requests take precedence. "
                      + measured +
                      ("Compare the reference with the video for major unrequested new subjects, effects or "
                         "background changes, while allowing the motion the user asked for. " if references else "") +
                      "Judge only this request, never requirements invented by a tool draft or earlier chat. "
                      "List, in a few words each, requested visible actions or appearance constraints the frames fail "
                      "(an object that never moves or leaves, a person who never does it). Speech and "
                      "sound cannot be seen: ignore them. Answer NONE when every action is visible.") or "").split())
    except Exception:
        logger.warning("video actions: the check failed", exc_info=True)
        return ""
    logger.info("video actions: checked against the request -> %s", ans[:200] or "no answer")
    matches_requested = False
    if motion and duration and motion[-1][0] >= duration * 0.3:
        right = bool(re.search(r'\b(?:right(?:wards)?|вправо|направо)\b', script, re.I))
        left = bool(re.search(r'\b(?:left(?:wards)?|влево|налево)\b', script, re.I))
        negated = re.search(r'\b(?:не|not|never|no)\b.{0,25}\b(?:right|left|вправо|влево|направо|налево)\b', script, re.I)
        dx = motion[-1][1] - motion[0][1]
        if not negated and right != left and ((right and dx < -0.05) or (left and dx > 0.05)):
            return measured + 'The tracked object moved opposite to the requested horizontal direction in the measured frames.'
        matches_requested = motion[-1][0] >= duration * 0.8 and not negated and right != left and ((right and dx > 0.05) or (left and dx < -0.05))
    if matches_requested and re.search(r"\b(?:does\s+not|doesn't|did\s+not|never|fails?\s+to|not)\b.{0,45}\b(?:move|moving|drift|shift)\w*\b|\b(?:remains?|stays?)\b.{0,25}\b(?:centered|stationary|still|central)\b", ans, re.I):
        try:
            other = yes_no(ctx)("A video review says: {text}. Does it identify any problem other than the position or motion of the main object, such as added smoke, extra people, a changed background, or a missing separate action?", ans, default=True)
        except Exception:
            other = True
        if not other:
            return ''
    if (motion and not _says_none(ans)
            and (max(p[1] for p in motion) - min(p[1] for p in motion) > 0.05
                 or max(p[2] for p in motion) - min(p[2] for p in motion) > 0.05)
            and re.search(r'\b(?:mov\w*|motion|stationary|static|still|center\w*|centr\w*|right|left|движ\w*|двига\w*|неподвиж\w*|вправо|влево)\b', ans, re.I)):
        return measured + 'The visual review suggests possible other differences; inspect the clip.'
    return "" if _says_none(ans) else ans[:400]


def _says_none(ans: str) -> bool:
    """A vision check's «nothing wrong»: NONE first or last, bare or **bold**. The model
    reasoned first and ended on «**NONE**.», the reply was told an action was missing and
    told the user to «pick a voice» (live 2026-10-20)."""
    words = [w.strip("*_.,:;!()\"'`").upper() for w in (ans or "").split()]
    words = [w for w in words if w]
    return not words or words[0] == "NONE" or words[-1] == "NONE"


def video_mod_rules(ctx) -> tuple:
    v = _video(ctx)
    return tuple(r for r in (getattr(v, "CAUSE_RULE", ""), getattr(v, "RU_SPEECH", "")) if r)


def _handle_generate_video(ctx, state, args: dict) -> str:
    import tools as _t   # the render seam is owned by tools; read it at CALL time
    video_mod = _video(ctx)
    ctx.video_part = ""             # a render that failed mid-chain left its «(6/6)» behind
    ctx.video_made = ""             # the finished clip of this call, for a turn that runs out of time
    from prompt_guard import user_words
    ctx.video_request = user_words(str(state.get("user_input_original") or state.get("user_input") or "")).strip()
    ctx.video_speech = ""

    description = (args.get("description") or "").strip()
    if not description:
        return ("[TOOL ERROR] generate_video needs a 'description' of the shot. "
                "Re-call it describing what should be on screen, what moves, and "
                "what it should sound like.")
    # A Russian chat with no line written out: the clip's own writer invented "Get out!"
    # in English (live 2026-10-11). Lines the user did write keep their language.
    if (re.search(r"[а-яё]", str((state or {}).get("user_input_original") or ""), re.I)
            and not re.search(r'[«"“]|<d>', description)):
        description += video_mod.RU_SPEECH
    description += video_mod.CAUSE_RULE
    if not dep(ctx, "asks_for_video", asks_for_video)(ctx, state):
        logger.warning("generate_video refused: the user did not ask for a video")
        return ("[TOOL ERROR] The user did not ask for a video in this message, so no "
                "clip is made. Answer what they actually said: a complaint about a "
                "picture is fixed on the picture (redraw_image / inpaint_image / "
                "generate_image), a question is answered. Do not call generate_video "
                "again this turn.")

    ok, why = video_mod.engine_available(ctx)
    if not ok:
        # Fail BEFORE promising anything. The expensive failure here is telling the
        # user a video is coming and only then discovering there is no engine.
        logger.error("generate_video unavailable: %s", why)
        return (
            f"[TOOL ERROR] Video generation is not available right now: {why}. "
            "Tell the user plainly that video is not set up yet and that no clip "
            "was made. Do NOT claim a video was created, and do not retry."
        )

    # A clip costs the whole turn: minutes of GPU, not seconds.
    _over = dep(ctx, "_render_budget_exhausted", _t._render_budget_exhausted)(
        state, "generate_video",
        cost=max(1, int(getattr(config, "IMAGE_MAX_RENDERS_PER_TURN", 3) or 3)))
    if _over:
        return _over

    images, videos = [], []
    if args.get("use_current_images", True):
        images = dep(ctx, "_current_image_paths", _current_image_paths)(ctx, state)
    if images and "use_current_images" not in args and not _builds_on_chat_picture(ctx, state):
        # «сделай видео: лиса бежит по лесу» opened on the man with the shovel from three
        # turns back and morphed into the forest (2026-10-17): the default took every
        # picture in the chat, whatever the words were about.
        # But a description that itself anchors on <Picture N> counts the chat's
        # picture in even when the yes/no read said "new scene": the model wrote
        # "close-up of the old man from <Picture 1>" and the drop left the reference
        # dangling -- H3 t2va invented its own cartoonish old man (2026-10-29 12:30).
        if re.search(r"<Picture \d+>", description):
            logger.info("generate_video: the description anchors on %s -- keeping the "
                        "chat's pictures", re.findall(r"<Picture \d+>", description))
        else:
            logger.info("generate_video: the words describe a new scene -- not animating %s",
                        [os.path.basename(p) for p in images])
            images = []
        # More than H3 can take: keep the most RECENT, which is what the user is
        # most likely talking about.
        if len(images) > video_mod.MAX_REF_IMAGES:
            images = images[-video_mod.MAX_REF_IMAGES:]
    if args.get("use_current_video"):
        lv = getattr(ctx, "last_video_path", None)
        if lv and os.path.exists(lv):
            videos = [lv]

    # A forwarded video is in play as its frame sheet, its clip kept beside it: a clip asked
    # for on it continues that clip from its real end. Animating the sheet itself made a new
    # scene from a grid of frames (live 2026-10-11, «сделай продолжение как имам…»).
    if not getattr(ctx, "continue_tail", ""):
        import video_look
        for p in images:
            clip = os.path.join(os.path.dirname(p), video_look.CONTINUE_CLIP)
            if os.path.basename(p).startswith("sheet") and os.path.exists(clip):
                tg_continue = _clips(ctx)
                seed, tail = clip[:-4] + "_seed.jpg", clip[:-4] + "_tail.mp4"
                if tg_continue.seed_frame(clip, seed) and tg_continue.cut_tail(clip, tail):
                    ctx.continue_tail, ctx.continue_src = tail, clip
                    images = [seed if q == p else q for q in images]
                break

    # «продолжи это видео» in words: made a new clip from nothing, another style even
    # (live 2026-10-13) -- only the ▶️ button set the clip up. The chat's last clip here.
    lv = getattr(ctx, "last_video_path", None)
    # A picture sent after the clip is what «оживи» is about: a fresh meme was read as
    # «continue the old clip» and its plan went on from an unrelated video (live 2026-10-13).
    newer_pic = lv and os.path.exists(lv) and any(
        os.path.exists(p) and os.path.getmtime(p) > os.path.getmtime(lv) for p in (images or []))
    if not getattr(ctx, "continue_tail", "") and lv and os.path.exists(lv) and not newer_pic:
        said = str((state or {}).get("user_input_original") or (state or {}).get("user_input") or "")
        if said and yes_no(ctx)(
                "The user wrote: {text}. Do they ask to continue or extend the video already in "
                "this chat -- what happens next in it -- rather than a new clip?", said, default=False):
            import tempfile
            tg_continue = _clips(ctx)
            d = tempfile.mkdtemp(prefix="contlast_")
            seed, tail = os.path.join(d, "seed.jpg"), os.path.join(d, "tail.mp4")
            if tg_continue.seed_frame(lv, seed) and tg_continue.cut_tail(lv, tail):
                logger.info("generate_video: continuing the chat's last clip %s", os.path.basename(lv))
                ctx.continue_tail, ctx.continue_src = tail, lv
                images = [seed]

    # ▶️ Continue video (tg_continue): the clip's tail is the reference -- its motion, faces and
    # its own sound -- and the new part is joined to the original afterwards.
    cont_tail = getattr(ctx, "continue_tail", "") or ""
    pinned_tail = ""
    cont_src = getattr(ctx, "continue_src", "") or ""
    continuing = bool(cont_tail and os.path.exists(cont_tail))
    chain_id, chain_idx = "", 0
    people: list = []
    bridge_first = False
    if continuing:
        videos = [cont_tail]
        ctx.continue_tail = ctx.continue_src = ""          # one clip per request
        people = [p for p in (getattr(ctx, "continue_people", None) or []) if p and os.path.exists(p)]
        ctx.continue_people = []
        # A clip this bot made as a Herrgott chain goes on from its own saved latent
        # (no reference, no re-encode); new people from photos still need the reference path.
        prior = video_mod.chain_of(cont_src, ctx) if (not people and video_mod.herrgott_on()) else ()
        if prior:
            chain_id, chain_idx = prior[0], prior[1] + 1
            videos, images = [], []
            # bridged after the script is cut into parts: bridged here, the script was
            # no longer found in it and «на 30 секунд» went out as one 15 s part (2026-10-13)
            bridge_first = True
            description = video_mod.CONTINUE_CTX_PREFIX + description
        # Pinned frames (same as the parts of a long script): faster and seamless, but they
        # take no pictures -- new people from photos still need the reference path.
        elif not people and video_mod.motion_context_on():
            pinned_tail, videos, images = cont_tail, [], []
            description = video_mod.CONTINUE_CTX_PREFIX + description
        else:
            description = video_mod.CONTINUE_PREFIX + description
        if people:
            # New people from photos sent after the clip: the start frame stays <Picture 1>.
            images = (images or [])[-1:] + people
            description += video_mod.new_people_clause(len(images) - len(people) + 1, len(people))

    audios = []
    # by slot, left to right; "" = that person keeps the default voice
    slots = ([] if continuing else
             [p if p and os.path.exists(p) else "" for p in (getattr(ctx, "anim_voices", None) or [])])
    slots = slots[:video_mod.MAX_REF_AUDIOS]
    if not slots and not continuing and getattr(ctx, "voice_choice", None) is None:
        # A front end without the voice question (the window) still takes «голос возьми отсюда:
        # <ссылка>»: the same engine as the bot's voice cards (voice_clone.fetch_sample).
        import tempfile
        import voice_clone
        link = voice_clone.voice_link_in(
            yes_no(ctx), str((state or {}).get("user_input_original") or (state or {}).get("user_input") or ""))
        if link:
            try:
                src = voice_clone.fetch_sample(link, os.path.join(tempfile.gettempdir(), "voice_links"))
                slots = [voice_clone.prepare_reference(ctx, src, os.path.dirname(src))[0]] if src else []
            except Exception:
                logger.warning("generate_video: the voice link could not be used", exc_info=True)
    anim = [p for p in slots if p]
    speakers = int(args.get("speakers") or 0)
    if not speakers:
        # The model forgets `speakers` (two quoted lines, speakers=0,
        # no voice question). Spoken lines in quotes are a fact, not a guess.
        speakers = min(len(re.findall(r'«[^»]{2,}»|"[^"]{2,}"|“[^”]{2,}”',
                                      args.get("description") or "")), video_mod.MAX_REF_AUDIOS)
    # a continuation speaks with the clip's own voices: its tail is the reference
    if (speakers and not anim and not continuing and not args.get("use_my_voice")
            and getattr(ctx, "voice_choice", None) == ""):
        # Asked once per clip, whatever led here (🎬 Animate, a chat request):
        # the bot puts «🎙 Свои голоса» / «▶️ Стандартные» under this reply
        # and reruns the request with the answer (tg_anim_voices).
        state["ask_voices"] = min(speakers, video_mod.MAX_REF_AUDIOS)
        return ("[NOT MADE YET] Before rendering, the user chooses the voices. Reply with "
                f"ONE short line in their language: the clip has {speakers} speaking "
                "person(s) and you will voice them with their samples or the default "
                "voices -- buttons for that come right under your message. Do not "
                "describe the clip, do not call generate_video again this turn.")
    if anim:
        # 🎙 samples collected before the clip (tg_anim_voices): one voice per
        # speaker, in the order the user was asked for -- left to right.
        audios = anim
        ctx.anim_voices = []                      # one clip per collection
        if "<Audio 1>" not in description and len(audios) < len(slots):
            nth = ["leftmost", "second from the left", "third from the left"]
            k = 0
            said = []
            for i, p in enumerate(slots):
                if p:
                    k += 1
                    said.append(f"the {nth[i]} person speaks with the voice of <Audio {k}>")
            description += " " + "; ".join(said)[:1].upper() + "; ".join(said)[1:] + "; everyone else keeps a natural voice of their own."
        elif "<Audio 1>" not in description:
            if len(audios) == 1:
                description += " The person speaks with the voice of <Audio 1>."
            else:
                tags_a = ", ".join(f"<Audio {i}>" for i in range(1, len(audios) + 1))
                description += (f" The people, from left to right, speak with the voices "
                                f"{tags_a}: the leftmost <Audio 1>, the next <Audio 2>, "
                                "and so on; each person keeps their own voice.")
    elif args.get("use_my_voice") and not continuing:
        # The chat's cloned voice (tg_tasks sets ctx.voice_ref from 🎙 Клон голоса).
        ref = getattr(ctx, "voice_ref", "") or ""
        if not (ref and os.path.exists(ref)):
            return ("[TOOL ERROR] The user has no cloned voice yet. Tell them to press "
                    "🎙 Клон голоса (Creativity menu), send a 5-12 s voice clip, then ask again. "
                    "No clip was made.")
        audios = [ref]
        if "<Audio 1>" not in description:
            description += " The person speaks with the voice of <Audio 1>."
    if audios and videos:
        # A reference clip brings its OWN soundtrack (ref_video_audio_N rides with
        # ref_video_N): with voice samples it outvoiced them -- a redo
        # «с баяном и ударами» reused the last clip and the user's voices were gone.
        logger.info("generate_video: voice samples given -> the previous clip is not a reference")
        videos = []
    # A script longer than one clip holds is rendered in parts, each the next one's
    # continuation from the last frame (the user's idea, 10-06: "split it into 5.2 s
    # blocks"): the first part here, the rest after it below.
    raw = (args.get("description") or "").strip()
    _said = str((state or {}).get("user_input_original") or (state or {}).get("user_input") or "")
    own_script = False
    dictated = False
    speech = ""             # the words the user dictated, whole: what every part together must say
    if not continuing:
        speech = video_mod.dictated_speech(_said, "", ctx)
        ctx.video_speech = speech
        words = speech if speech and video_mod._flat_line(speech) not in video_mod._flat_line(raw) else ""
        if words:
            logger.info("generate_video: the user's own speech added to the script (%d words)", len(words.split()))
            # The agent's own quoting of that speech goes: it is the same speech retold (in
            # English, cut short), so it would be said twice -- and it made the script look
            # like the bot's own, condensed to 2 parts and rendered as one (live 2026-10-20).
            dictated = True
            said_it = f'{video_mod._QUOTED.sub("", raw)} The person says, word for word: "{words}"'.strip()
            description = description.replace(raw, said_it, 1) if raw and raw in description else said_it
            raw = said_it
    if not continuing:
        # A script the user numbered and timed is theirs: the agent's retelling of it dropped
        # the lengths and the actions («3.2 с — … 2.2 с — …» became a bee in a suit).
        from prompt_guard import user_words
        own = "\n".join(ln for ln in user_words(_said).splitlines() if ln.strip())
        if video_mod.numbered_script(own):
            logger.info("generate_video: the user's own numbered script is used as written")
            description = description.replace(raw, own, 1) if raw and raw in description else own
            raw = own
            own_script = True       # written and cut by the user: there is no plan to approve
    if images and not continuing and not args.get("seconds") and not yes_no(ctx)(
            "The user asked to animate a picture: {text}. Do they say WHAT should happen in "
            "the clip (an action, a movement, a line), rather than leaving it to the bot "
            "(«на своё усмотрение», just «оживи»)?", _said, default=True):
        acted = video_mod.action_script(raw, ctx)
        if acted:
            logger.info("generate_video: script as action from the first frame: %s", acted[:300])
            description = description.replace(raw, acted, 1) if raw in description else acted
            raw = acted
    if not args.get("seconds") and not continuing:
        # «сделай видео на 30 секунд» came with no seconds from the agent and was planned
        # as three 5.2 s parts, ≈16 s (2026-10-17): the length is read from the words.
        try:
            from prompt_guard import user_words
            asked = float(intent_read(ctx)(None, user_words(_said).strip()).get("song_seconds") or 0)
        except Exception:
            asked = 0.0
        if asked > 0:
            logger.info("generate_video: %.0f s asked in the words, none in the call", asked)
            args["seconds"] = asked
    want = float(args.get("seconds") or 0)
    # The words the user dictated are never condensed or cut to the bot's own 2-part cap:
    # an 85-word monologue came out as its first sentence (live 2026-10-19).
    speech_theirs = dictated or video_mod.speech_is_users(raw, _said)
    # Over one clip's 15 s the parts follow the script: «на 30 секунд» was capped to one
    # 15 s clip and delivered without a word (2026-10-13).
    if args.get("seconds") and float(args["seconds"]) <= 15.0:
        parts = [raw]
    else:
        args.pop("seconds", None)
        parts = video_mod.split_script(raw, reads=dep(ctx, "script_read", None))
        have = sum(video_mod.estimate_seconds(p, reads=dep(ctx, "script_read", None)) for p in parts)
        if want and have > want * 1.3:
            # «на 30 секунд» came out as 11 parts, a minute (live 2026-10-13)
            tight = video_mod.condense_script(raw, have, want, ctx)
            if tight:
                logger.info("generate_video: script condensed from ~%.0f s to fit %.0f s", have, want)
                description = description.replace(raw, tight, 1) if raw in description else description
                raw = tight
                parts = video_mod.split_script(raw, reads=dep(ctx, "script_read", None))
        cap = video_mod.frames_to_seconds(video_mod.VIDEO_MAX_FRAMES_AUTO)
        if want > len(parts) * cap:
            # 30 s asked of a 4-sentence script split in two: two parts at the cap are 20 s
            finer = video_mod.split_script(raw, max_seconds=max(1.0, have / math.ceil(want / cap)), reads=dep(ctx, "script_read", None))
            if len(finer) > len(parts):
                parts = finer
        if not want and len(parts) > video_mod.AUTO_MAX_PARTS and not speech_theirs:
            # A script the bot wrote itself ran 5-8 parts, ~25 min of render, and the
            # longer it ran the more nonsense it held (owner, 2026-10-15): one or two
            # episodes, the rest is the user's ▶️ Continue.
            fit = video_mod.AUTO_MAX_PARTS * 5.2
            tight = video_mod.condense_script(raw, have, fit, ctx, sentences=2 * video_mod.AUTO_MAX_PARTS)
            if tight:
                description = description.replace(raw, tight, 1) if raw in description else description
                raw = tight
                parts = video_mod.split_script(raw, reads=dep(ctx, "script_read", None))
            if len(parts) > video_mod.AUTO_MAX_PARTS:
                parts = parts[:video_mod.AUTO_MAX_PARTS]
                cut = " ".join(parts)
                description = description.replace(raw, cut, 1) if raw in description else description
                raw = cut
            logger.info("generate_video: script kept to %d part(s) (%s)", len(parts),
                        "condensed" if tight else "cut")
    if continuing and len(parts) > 1:
        # «…орёт э, это уже есть на видео, а потом…»: the agent still opened the plan
        # with the grandma's scream, rendered again for 6 minutes (owner, 2026-10-12).
        from prompt_guard import user_words
        said = user_words(str((state or {}).get("user_input_original") or "")).strip()
        # Sentences, before the script is cut into parts: a part is a few sentences, and
        # dropping whole parts threw away «he straightens up to his full height» while the
        # user asked exactly for that.
        old = ""
        if said:
            try:
                old = (text_model(ctx)(
                    None, "Copy, word for word, ONLY the part of the user's message that they say "
                    "is ALREADY in the existing video (e.g. the words before «это уже есть на "
                    "видео» / «it is already in the video»). If they say nothing is already "
                    "there, reply NONE. No other words.", said, temperature=0, max_tokens=200)
                    or "").strip()
            except Exception:
                logger.warning("generate_video: reading what is already in the clip failed", exc_info=True)
            old = "" if old.upper().startswith("NONE") else old
            logger.info("generate_video: already in the clip per the user: %r", old[:200])

        if old:
            # One call over the numbered sentences, counting the repeats at the START only:
            # a yes/no per sentence dropped the object pulling back and the guy walking out
            # as «repeats» on one run and kept them on the next; the count came back the
            # same three times out of three.
            sentences = re.split(r"(?<=[.!?»])\s+", raw)
            try:
                n = int(re.search(r"\d+", text_model(ctx)(
                    None, "You trim a video continuation script. The existing clip already "
                    "shows: «" + old + "». The numbered sentences below are the script for what "
                    "comes AFTER the clip. Count how many sentences AT THE START only show what "
                    "the existing clip already shows (setting the scene, the same action again). "
                    "Stop counting at the first sentence where anything new happens: an object "
                    "moves or disappears, a door opens, someone appears. Reply with the number "
                    "only (0 if the first sentence is already new).",
                    "\n".join(f"{i}. {x}" for i, x in enumerate(sentences, 1)),
                    temperature=0, max_tokens=8) or "0").group(0))
            except Exception:
                logger.warning("generate_video: counting repeated sentences failed", exc_info=True)
                n = 0
            kept = sentences[min(n, len(sentences) - 1):]
            for x in sentences[:len(sentences) - len(kept)]:
                logger.info("generate_video: already in the clip, dropped: %s", x[:120])
            if len(kept) < len(sentences):
                new_raw = " ".join(kept)
                description = description.replace(raw, new_raw, 1) if raw in description else new_raw
                raw = new_raw
                parts = video_mod.split_script(raw, reads=dep(ctx, "script_read", None))
    if audios and len(parts) > 1:
        parts = video_mod.merge_silent_parts(parts)
    talking = None
    if len(parts) > 1:
        # A talking head's parts are takes of one still saying the next lines: the script
        # editor gave them «points to a map», «gestures toward the refrigerator» for want of
        # a visible action, and the head turned away from the camera (live 2026-10-20).
        talking = bool(len(images) == 1 and not videos and not continuing
                       and video_mod.is_talking_head(ctx, raw, _said, spoken=bool(speech)))
        if not talking:
            parts = video_mod.validate_script(ctx, video_mod.mark_inserts(ctx, parts),
                                              max_parts=math.ceil(want / 5.2) if want else (len(parts) if speech_theirs else video_mod.AUTO_MAX_PARTS))
        parts = video_mod.dedupe_quotes(video_mod.strip_translations(parts))
    fitted = video_mod.fit_speech(parts)
    if len(fitted) > len(parts):
        logger.info("generate_video: parts with more speech than a clip says were cut: %d -> %d", len(parts), len(fitted))
        parts = fitted
    gone = video_mod.lost_sentences(speech, parts) if speech and len(parts) > 1 else []
    if gone:
        # Two of the user's sentences never reached the plan (live 2026-10-20); which model
        # step dropped them is not the point -- the user's words are all said, as written.
        logger.warning("generate_video: the parts lost the user's words %s; re-cut from the script as written", gone)
        parts = video_mod.fit_speech(video_mod.split_script(raw, reads=dep(ctx, "script_read", None)))
    if len(parts) > 1 and raw in description:
        description = description.replace(raw, parts[0], 1)
        logger.info("generate_video: the script needs %d parts: %s", len(parts),
                    [round(video_mod.estimate_seconds(p, reads=dep(ctx, "script_read", None)), 1) for p in parts])
    else:
        parts = [raw]
    # 🎬 The plan the user approved under the card (tg_video_plan): exactly those parts.
    approved = [p for p in (getattr(ctx, "video_plan_parts", None) or []) if p]
    ctx.video_plan_parts = []
    if approved:
        ctx.video_request = "\n".join(str(p) for p in approved)
        ctx.video_speech = ""
        # the first part stands in the description now, not the whole script: looking for
        # the script dropped the continuation prefix and the Russian-speech rule, and the
        # bridge then cut the prefix's length off the part itself (live 2026-10-13)
        cur = parts[0]
        description = (description.replace(cur, approved[0], 1) if cur and cur in description
                       else approved[0])
        parts = video_mod.fit_speech(approved)
    # An asked length the script falls short of: each part plays longer, up to one clip's cap.
    secs = [video_mod.estimate_seconds(p, reads=dep(ctx, "script_read", None)) for p in parts]
    if want > 15.0 and len(parts) > 1 and want > sum(secs) * 1.15:
        per = min(video_mod.frames_to_seconds(video_mod.VIDEO_MAX_FRAMES_AUTO), want / len(parts))
        secs = [max(x, round(per, 2)) for x in secs]
    # A loop: the picture is the first and the last frame (FL2VA keyframes), one part --
    # the clip of Лёха flexing under the track had to repeat without a jump (2026-10-22).
    loop = bool(args.get("loop") and images and not videos and not continuing)
    if loop:
        images, parts = [images[-1], images[-1]], [raw]    # newest last: [0] animated an older picture
        secs = [max(video_mod.estimate_seconds(raw, reads=dep(ctx, "script_read", None)), 5.2)]
    if len(parts) > 1 and not approved and not own_script and getattr(ctx, "video_plan_ask", False):
        # Several minutes per part: the chat sees the parts, the clip length and the
        # render time first, and may drop or condense parts (user, 2026-10-10).
        state["video_plan"] = [{"text": p, "sec": x} for p, x in zip(parts, secs)]
        if continuing:      # nothing rendered: the clip and its new people wait for the ▶ rerun
            ctx.continue_tail, ctx.continue_src, ctx.continue_people = cont_tail, cont_src, people
        return ("[NOT MADE YET] The script needs several parts, and the user approves the "
                "plan first: it is shown under your message with buttons. Reply with ONE "
                "short line in their language that the plan is below. Do not describe the "
                "clip, do not call generate_video again this turn.")
    if bridge_first:
        body = description.removeprefix(video_mod.CONTINUE_CTX_PREFIX)
        description = video_mod.CONTINUE_CTX_PREFIX + _bridged(ctx, cont_src, body)
    # A fresh clip with no voices or reference clips (and at most a start photo) is part 1
    # of a Herrgott chain: its latent is kept, so the parts after it -- or a later
    # ▶️ Continue -- go on from the exact numbers instead of a reset.
    if (not continuing and not audios and not videos and len(images) <= 1
            and video_mod.herrgott_on()):
        chain_id, chain_idx = video_mod.new_chain(), 1
        if len(parts) > 1:
            parts = video_mod.bridge_script(ctx, parts)
    # Every model call of every part happens here, while the chat model is loaded; the
    # renders below then run with the card held (and the model out) start to finish.
    prepped = []
    if chain_id and len(parts) > 1:
        prepped = [video_mod.prepare_prompt(ctx, description, images=images, seconds=secs[0] if secs else 0.0,
                                           mode="i2va" if images and chain_idx <= 1 else "t2va")]
        voices = video_mod.voices_of(prepped[0][0])
        voices = f"Voices so far, the same if they speak again: {voices} " if voices else ""
        prepped += [video_mod.prepare_prompt(
            ctx, video_mod.CONTINUE_CTX_PREFIX + video_mod.with_scene(p, voices + " ".join(parts[:i])),
            seconds=secs[i] if i < len(secs) else 0.0, mode="t2va") for i, p in enumerate(parts[1:], start=1)]
    # With voices the first picture rides every part too: each part starts from the last
    # frame of the one before, and the face drifted a little further every time.
    # ...and so do all of them: with two heroes on two pictures, parts 2+ saw only the first
    # one and the second (the cat) was redrawn from the clip's tail.
    anchor = list(images[:3]) if images and not people and not continuing else []
    head = bool(not chain_id and len(anchor) == 1 and not videos
                and (talking if talking is not None else video_mod.is_talking_head(ctx, raw, _said, spoken=bool(speech))))
    head = head or loop
    if loop:
        logger.info("generate_video: loop, the picture is the first and the last frame")
    elif head:
        logger.info("generate_video: talking-head mode, the still is frame 0 of every part, the voice comes after")
    mode = video_mod.pick_mode(images, videos, audios, keyframe=head)
    if chain_id:
        mode = "i2va" if images and chain_idx <= 1 else "t2va"
    tags = video_mod.reference_tags(len(images) if mode == "ref2va" else 0, len(videos), len(audios))
    logger.info("Tool: generate_video(mode=%s, %d image(s), %d video(s)) %s",
                mode, len(images), len(videos), description[:80])

    # The whole chain holds the card: no model call runs between its parts (the bridges
    # were written above), so the chat model stays out instead of being loaded back for
    # every gap -- and the turn deadline sees the card in use, not an idle ComfyUI.
    hold = contextlib.ExitStack()
    if chain_id and len(parts) > 1:
        import comfy_client
        hold.enter_context(comfy_client.card_session("video chain"))
    with hold:
        # what generate_video's own stages say too (video._video_stage)
        ctx.video_part = f"1/{len(parts)}" if len(parts) > 1 else ""
        ctx.set_stage(f"Generating a video ({ctx.video_part})" if ctx.video_part else "Generating a video")
        first_prompt = prepped[0][0] if prepped else description
        # A static shot opening a chain on a photo: H3 sometimes covers it with drawn flowers
        # or anime faces, and every later part inherits them from the latent. Caught here by
        # the frame edges, redrawn with another seed.
        watch = bool(chain_id and images and not continuing
                     and _preserves_static_scene(ctx, ctx.video_request or first_prompt))
        best_eyes = None
        said_first = video_mod.quoted_lines(parts[0]) if head and parts else []
        speech_ok = getattr(ctx, "speech_ok", None) or _speech_ok
        try:
            for attempt in (1, 2, 3):
                result = video_mod.generate_video(
                    ctx, first_prompt,
                    images=images, videos=videos, audios=audios,
                    seconds=prepped[0][1] if prepped else (args.get("seconds") or 0.0),
                    prepared=bool(prepped),
                    aspect=args.get("aspect") or "",
                    seed=args.get("seed") if attempt == 1 else None,
                    context_video=pinned_tail,
                    chain=(chain_id, chain_idx) if chain_id else (),
                    size_from=cont_src if chain_id and continuing else "",
                    keyframe=head,
                )
                p = result.get("path")
                look = watch and p and os.path.exists(p)
                drawn = video_mod.drawn_over(p, images[0]) if look else None
                drift = video_mod.overlay_drift(p) if look else None
                eyes = video_mod.eye_drift(images[0], p, light=False) if head and p and os.path.exists(p) else None
                # part 1 had no speech check: its last word, cut by the clip's end, went to the user
                # in runs 17 and 18 (2026-10-20); a take that says it all ranks first
                missed = bool(head and said_first and p and os.path.exists(p)
                              and not speech_ok(ctx, p, said_first))
                if eyes is not None and (best_eyes is None or (missed, eyes) < best_eyes[0]):
                    best_eyes = ((missed, eyes), result)
                if ((eyes is not None and eyes > 0) or missed) and attempt < 3:
                    logger.warning("generate_video: part 1 drifted off the still's eyes (%s) or did not "
                                   "say %s; once more", eyes, said_first if missed else "")
                    continue
                if (attempt == 3 and len(parts) > 1 and
                        ((drift is not None and drift > video_mod.OVERLAY_DRIFT_MAX) or
                         (drawn is not None and drawn > video_mod.DRAWN_OVER_MAX))):
                    return ('[TOOL ERROR] The first video segment repeatedly changed the background or added objects. '
                            'Tell the user the long video could not be completed; do not claim it was created.')
                if attempt == 3 or ((drift is None or drift <= video_mod.OVERLAY_DRIFT_MAX)
                                    and (drawn is None or drawn <= video_mod.DRAWN_OVER_MAX)):
                    break
                logger.warning("video chain: part 1 is covered over (edge drift %s, drawn over %s); "
                               "once more with a new seed", drift, drawn)
            if best_eyes:
                result = best_eyes[1]       # of two takes, the one nearer the still's eyes
            if head and result.get("path") and os.path.exists(result["path"]):
                result = {**result, "path": _pull_light(video_mod, images[0], result["path"], 1)}
        except Exception as exc:
            logger.exception("generate_video crashed")
            return (f"[TOOL ERROR] Video generation failed: {exc}. Tell the user the "
                    "clip could not be made; do NOT claim one was created.")

        path = result.get("path")
        if path and not os.path.exists(path):
            logger.error("generate_video returned a non-existent path %r", path)
            path = None

        if not path:
            if result.get("status") == "cancelled":
                return ("[TOOL ERROR] The video was cancelled before it finished. Tell the "
                        "user it was stopped and no clip was produced.")
            gpu = _gpu_busy_error("Video generation")
            if gpu:
                return gpu
            detail = result.get("reason") or "the render produced no file"
            return (f"[TOOL ERROR] Video generation failed — no clip was produced. "
                    f"Reason reported by the renderer: {detail}. "
                    "You MUST tell the user clearly that the video could not be generated and "
                    "give them THIS reason in plain words; do NOT invent another cause, do NOT "
                    "blame the prompt unless the reason says so, and do NOT imply or claim that "
                    "a video was created.")

        if chain_id:
            if len(parts) > 1:
                chain_idx = _render_chain_parts(ctx, path, prepped, args, chain_id, chain_idx)
            if chain_idx > 1:
                joined = video_mod.stitch_chain(ctx, chain_id, chain_idx)
                if joined:
                    path = joined
                else:
                    logger.warning("video chain: the join failed, delivering the last part alone")
            video_mod.remember_chain(path, chain_id, chain_idx, ctx)
            if len(parts) > 1 or chain_idx > 1:
                result["seconds"] = video_mod.probe(path).get("seconds") or result.get("seconds")
    ctx.video_join_note = ""
    if chain_id and len(parts) > 1:
        ctx.video_join_note = _check_joins(ctx, getattr(ctx, "chain_joins", None) or [])
    if not chain_id and continuing and cont_src and os.path.exists(cont_src):
        joined = (video_mod.join_pinned(cont_src, path) if pinned_tail
                  else video_mod.join_continuation(cont_src, path))
        if joined:
            path = video_mod._adopt_output(joined, video_mod.OUTPUT_DIR)       # the whole thing: original + what comes next
            result["seconds"] = video_mod.probe(path).get("seconds") or result.get("seconds")
        else:
            logger.warning("continue video: the join failed, delivering the new part alone")

    if len(parts) > 1 and not chain_id:
        path = _render_remaining_parts(ctx, path, parts, args, people=[*people, *anchor], voices=audios,
                                       fresh_from=anchor[0] if head else "")
        result["seconds"] = video_mod.probe(path).get("seconds") or result.get("seconds")
    ctx.video_part = ""

    if head and audios and path:
        ctx.set_stage("Putting the voice on")
        said_in = video_mod.revoice(ctx, path, audios[0])
        if said_in:
            path = said_in
        else:
            logger.warning("generate_video: the voice could not be put on, delivering the model's own")
    # A talking head acts with its mouth only and its camera is locked: checked against the agent's
    # «the camera zooms in», the note said an action was missing and the reply told the user to
    # «pick a voice variant» (live 2026-10-20)
    ctx.video_action_note = "" if continuing or head else _check_actions(ctx, path, parts, references=images)
    if images and not continuing and not head:
        background_note = _check_static_background(ctx, video_mod, path, ctx.video_request or description)
        objects_note = ('The video may contain added objects or effects.'
                        if watch and drawn is not None and drawn > video_mod.DRAWN_OVER_MAX else '')
        ctx.video_action_note = ' '.join(note for note in (ctx.video_action_note, background_note, objects_note) if note)
    state["video_path"] = path
    state["video_status"] = "success"
    state["video_seconds"] = result.get("seconds")
    ctx.last_video_path = path
    ctx.video_made = path           # last_video_path is each PART's clip while the parts render
    ctx.last_video_prompt = ctx.video_request or description
    dep(ctx, "_remember", _remember)(ctx, state, "video", f"Video for: {ctx.last_video_prompt}", {"path": path})

    used = ""
    if tags:
        used = f" References used: {', '.join(tags)}."
    return (
        f"Video generated and saved: {path}\n"
        f"Mode: {video_mod.MODE_LABELS.get(mode, mode)}; "
        f"{result.get('seconds')}s at {result.get('width')}x{result.get('height')}, "
        f"with generated audio.{used}\n"
        # The reply told of a grandma's scream from a part the user had dropped
        # from the plan (live 2026-10-12): the approved parts are what was made.
        + ("What the clip shows is ONLY these approved plan parts (anything else in the "
           "request, such as parts the user removed, was NOT made -- never describe it): "
           + " | ".join(approved) + "\n" if approved else "")
        + ("A review of a few sampled frames suggests possible issues: " + ctx.video_action_note
           + ". This review can misread motion between frames. Tell the user these are possible issues, "
           "not confirmed failures, and let them judge the clip.\n" if getattr(ctx, "video_action_note", "") else "")
        + ("The joins between the parts were checked: " + ctx.video_join_note + ". Tell the "
           "user plainly which join is rough; do NOT call the clip seamless.\n"
           if getattr(ctx, "video_join_note", "") else "") +
        "The clip is delivered to the user automatically — just tell them it is ready "
        "in one short sentence. Do NOT call generate_video again for this turn."
    )


def redraw_whole_image(ctx, source: str, instructions: str):
    """Re-render the WHOLE picture following `instructions`. Returns (path, engine).

    Our own Ideogram picture is re-rendered from its stored boxes; anything
    else (a user's photo) goes through a mask-free FireRed edit. Shared by the
    redraw_image tool and the GUI Redraw / Style preset buttons.
    """
    new_path = None
    # Which engine ACTUALLY ran, for the audit log.
    engine = ""
    if dep(ctx, "load_layout_for", _image(ctx).load_layout_for)(source):
        ctx.set_stage("Redrawing from the picture's own layout")
        logger.info("Tool: redraw_image -> layout edit (Ideogram boxes): %r",
                    instructions[:80])
        try:
            new_path = dep(ctx, "edit_via_layout", _image(ctx).edit_via_layout)(ctx, source, instructions)
            engine = "edit_via_layout (Ideogram boxes, re-rendered from the stored layout)"
        except Exception:
            logger.exception("layout redraw failed for %s", source)
        if not new_path:
            logger.warning("layout redraw produced nothing for %s — falling "
                           "back to a pixel pipeline", source)

    if not new_path:
        # No layout: this is a picture the user brought us. A mask-free
        # whole-image FireRed edit follows the instruction while keeping the
        # photograph's own structure.
        ctx.set_stage("Redrawing the photo")
        # This is a SECOND, independent entry point into whole-frame FireRed
        # besides image_router.route_edit_request's style_transfer path — the
        # agent reaches this one directly via redraw_image(mode="redraw", ...)
        # with its OWN free-text instructions, never through the classifier.
        # Passing the agent's raw instruction straight
        # through (as before) cropped a person out of frame on a style
        # request even after route_edit_request's style_transfer path was
        # fixed, because THIS path never called it. Both entry points render a
        # style through image_style.restyle.
        # The edit classifier is not asked: «an ultra-realistic photograph of a baker…»
        # was not a style to it, went to the plain edit and the cartoon stayed a
        # cartoon. restyle's own read of the words decides, and a non-style redraw
        # is edited as written there.
        import image_style
        logger.info("Tool: redraw_image -> mask-free FireRed whole-image edit: %r",
                    instructions[:80])
        try:
            new_path = dep(ctx, "restyle", image_style.restyle)(ctx, source, instructions)
            engine = "edit_image_with_firered (whole frame, NO mask)"
        except Exception:
            logger.exception("FireRed redraw failed for %s", source)

    # Every other edit handler in this file (inpaint, transfer, fix_hands,
    # fix_artifact) runs its result through assert_deliverable before it can reach
    # the user; this one never did. A redraw delivered
    # "_INTERMEDIATE_tile_firered_00001_.png" straight to the user's chat — a
    # cropped FireRed working tile, not a finished picture. redraw/enhance can
    # fall through several engines (layout edit, whole-frame FireRed, the old model
    # controlnet, img2img) and any one of them returning a stale or scratch path
    # had nothing to catch it.
    new_path = dep(ctx, "assert_deliverable", _image(ctx).assert_deliverable)(
        new_path, where="tools._handle_redraw_image", source_path=source)

    return new_path, engine


def _handle_redraw_image(ctx, state, args: dict) -> str:
    import tools as _t   # the render seam is owned by tools; read it at CALL time
    exists = dep(ctx, "exists", os.path.exists)
    source = _edit_source(ctx, state)
    _note_source(state, source)
    if not source or not exists(source):
        return (
            "[TOOL ERROR] No image to redraw — generate an image first with generate_image, "
            "then call redraw_image. Tell the user there is no picture to work on yet."
        )
    _undone = _undo(ctx, state)
    if _undone:
        return _undone

    _over = dep(ctx, "_render_budget_exhausted", _t._render_budget_exhausted)(state, "redraw_image")
    if _over:
        return _over

    instructions = (args.get("instructions") or "").strip()
    # A 🎭 style-preset button already decided WHAT to do: its payload quotes the
    # exact instructions. Live 2026-10-29: the preset said "Change picture 1 into
    # a real photograph taken with a full-frame DSLR…", the model relayed the
    # image's own cartoon description instead -- the "style change" redrew the
    # same cartoon. The payload's quoted instructions outrank the model's args.
    _payload = (state.get("user_input") or "").lstrip()
    if _payload.startswith("[style_preset]"):
        import re as _re
        _pinned = _re.search(r'instructions="([^"]+)"', _payload)
        if _pinned and _pinned.group(1).strip():
            instructions = _pinned.group(1).strip()
    # 'enhance'/'upscale'/'restore'/'outpaint' have no engine left to run on: the
    # old model checkpoint behind them was removed from the product (Ideogram +
    # FireRed only) and its files deleted from disk. Every redraw goes through the same
    # FireRed path below; an empty instruction gets a generic quality nudge so a
    # bare "enhance" request still does something instead of a no-op.
    mode = "redraw"
    if not instructions:
        instructions = "improve the overall quality: sharpen fine detail, clean up noise and artifacts"

    # A colour conversion is exact arithmetic and must never be re-rendered,
    # whichever tool the model reaches for. Asked for black and white, the
    # generative path returned a DIFFERENT TRACTOR — in colour — and the model
    # reported failure while the picture was delivered anyway. Delegating here
    # (not only in route_edit_request) means the deterministic path wins even
    # when the request arrives as a "redraw".
    if instructions and _image(ctx).classify_edit_intent(instructions, ctx=ctx) in ("colour_convert", "transform"):
        from image import convert_colour
        import image_router
        ctx.set_stage("Converting the colours")
        logger.info("Tool: redraw_image -> exact pixel operation: %r", instructions[:80])
        new_path = (dep(ctx, "convert_colour", convert_colour)(source, instructions)
                    if _image(ctx).classify_edit_intent(instructions, ctx=ctx) == "colour_convert"
                    else image_router.transform_image(source, instructions, ctx=ctx))
        if not new_path or not exists(new_path):
            return ("[TOOL ERROR] The operation failed (a crop to the face needs a "
                    "face the detector can find). Tell the user plainly; do NOT "
                    "claim the picture was changed.")
        _set_current_image(ctx, state, new_path)
        if _image(ctx).classify_edit_intent(instructions, ctx=ctx) == "transform":
            return ("[done] The picture was rotated/mirrored/cropped exactly on its "
                    "pixels -- nothing was redrawn. The result is on screen. Briefly "
                    "confirm to the user, in their language.")
        return ("[done] The picture was converted exactly — same subject, same "
                "resolution, only the colours changed. The result is on screen. "
                "Do NOT say any part of it is still in colour: it is not. "
                "Briefly confirm to the user, in their language.")

    import image_router
    _plan = image_router.edit_plan(instructions, ctx=ctx)
    if _plan.get("kind") == "outpaint":
        # Said once, plainly: the model was left suggesting other formats and
        # offering to try again.
        return ("[TOOL ERROR] Extending the picture beyond its edges (outpaint) is not "
                "something this assistant can do: that engine was removed. Nothing was "
                "changed. Tell the user plainly it is not supported; do NOT retry and do "
                "NOT offer a different format as a substitute.")
    if _plan["reformats"]:
        # A redraw keeps the source's size: "the same, but horizontal for YouTube"
        # came back vertical and was announced as done.
        return ("[TOOL ERROR] redraw_image keeps the picture's size and orientation. "
                "To change the format, call generate_image again with the same "
                "description and the new width/height (e.g. 1680x944 for "
                "horizontal/YouTube, 944x1680 for vertical/stories).")
    # «он лысый, это Ленин»: the person may be named only in the picture's own prompt.
    import person_look
    instructions = person_look.with_look(instructions, getattr(ctx, "last_image_prompt", "") or "", ctx=ctx)
    ctx.set_stage("Redrawing the picture")
    logger.info("Tool: redraw_image(mode=%s, instructions=%r, source=%s)",
                mode, instructions[:80], source)

    # WHERE a redraw goes depends on where the picture came from.
    #   1. our own Ideogram picture  -> its stored boxes (edit_via_layout)
    #   2. the user's own photo      -> whole-frame FireRed, NO mask
    # the old model (the old controlnet/img2img fallback for everything neither of
    # these could handle) was removed from the product along with its checkpoint
    # files, so a picture that is neither of the above now surfaces as a
    # [TOOL ERROR] below instead of silently falling through to a dead engine.
    # Only a look the user NAMES this turn re-styles a redraw («сделай реалистично»);
    # «он лысый» keeps the picture's own style.
    _prev_look = getattr(ctx, "user_look", "")
    from prompt_guard import user_words
    from ideogram_layout import _look
    _said = user_words(str(state.get("user_input_original") or state.get("user_input") or "")).strip()
    ctx.user_look = _look(_said, ctx=ctx) if _said else ""
    try:
        new_path, engine = redraw_whole_image(ctx, source, instructions)
    finally:
        ctx.user_look = _prev_look
    verb = "redrawn"
    if new_path and exists(new_path):
        # keeps BEFORE for inspect_image: judged alone, «is it sharper than before?»
        # came back MISSING and a good quality pass was redone into a zoomed reframe
        _set_current_image(ctx, state, new_path)

    try:
        dep(ctx, "log_edit_decision", _image(ctx).log_edit_decision)(
            request=f"instructions={instructions!r}",
            classifier=f"mode={mode}", tool="redraw_image",
            workflow=engine or "redraw: no engine produced a file",
            returned_file=new_path, source=source)
    except Exception:
        logger.debug("could not log the redraw decision", exc_info=True)
    if not new_path or not exists(new_path):
        return (
            "[TOOL ERROR] Could not redraw the image — "
            "the ComfyUI server returned no result. Tell the user the redraw failed; "
            "do NOT claim a new image was produced."
        )

    # ctx.last_image_prompt is deliberately NOT overwritten with the bare
    # instructions: it describes the picture's subject, and a later redraw needs it.
    dep(ctx, "_remember", _remember)(ctx, state, "redraw", f"Image {verb}" + (f": {instructions}" if instructions else ""),
              {"path": new_path})
    result = f"Image {verb} and saved: {new_path}"
    return result

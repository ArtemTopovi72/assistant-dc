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
import functools
import logging
import os
import re
from typing import Optional

import config

import image as image_mod
import tool_graph
from image import (
    _INPAINT_FAILURE, _is_removal_instruction, classify_edit_intent,
    inpaint_region_with_comfy, route_edit_request,
)
from tool_context import _remember, _set_current_image

logger = logging.getLogger("assistant.tools")

def _is_absence_check(check: str) -> bool:
    """The inspect_image check asks whether something is gone or still there
    (verifying a removal). The model's yes/no; it replaced a word list
    («still», «осталась», «исчез»...). Model down -> False, the plain verdict."""
    check = (check or "").strip()
    if not check:
        return False
    if os.getenv("F5_TEST_RUN") and not os.getenv("INTENT_LIVE"):
        return bool(ABSENCE_STUB and ABSENCE_STUB(check))
    return _absence_read(check)


ABSENCE_STUB = None   # suites: ABSENCE_STUB(check) -> bool


@functools.lru_cache(maxsize=256)
def _absence_read(check: str) -> bool:
    try:
        import llm
        ans = llm.call_llm_simple(
            None, "Answer yes or no only.",
            "Does this check on a picture ask whether something was removed, is gone, or is "
            "still there? A question about a colour, size or look is no.\n\nCheck: «%s»"
            % check[:500], temperature=0.0, max_tokens=5) or ""
        return ans.strip().lower().startswith(("yes", "да"))
    except Exception:
        logger.warning("absence check: model read failed", exc_info=True)
        return False


def _note_source(state, source) -> None:
    """Record which picture an editing tool starts from, so the delivery layer
    can register the result as a VERSION of it. A free-text edit ("надень на
    кота шляпу") used to produce a picture with no parent, and the next edit
    asked "Which picture?" between the two (live, 2026-09-12, journey 21)."""
    if source:
        state["image_derived_from"] = str(source)


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

    from llm import analyze_image_with_llm
    from prompts import IMAGE_INSPECT_PROMPT
    # NO prefill. A closed-<think> prefill makes these fine-tunes return EMPTY for a
    # vision task (verified live — same bug that broke evaluate_image). Vision is also
    # flaky when a non-multimodal chat model holds the VRAM, so try twice.
    # The same measured-exposure evidence the outer judge gets: this vision
    # model calls a normally lit picture "extremely dark" and then labels every
    # element MISSING, which sent the agent into a second 70-second re-render of
    # a picture that was already right (live, 2026-09-12, the recolour edit).
    import exposure as _exposure
    _measured = _exposure.evidence(source)
    result = ""
    for _ in range(2):
        result = analyze_image_with_llm(
            ctx=ctx, image_path=source,
            user_text=check + ("\n\n" + _measured if _measured else ""),
            system_prompt=IMAGE_INSPECT_PROMPT,
        ) or ""
        if result.strip():
            break
    logger.info("inspect_image result: %s", result.strip()[:400].replace("\n", " | "))
    if _measured and result.strip() and (
            _exposure.is_dark_complaint(result)
            or _exposure.judge_is_blind([result], source)):
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
    if _is_absence_check(check):
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
    return f"Inspection of the current image:\n{result}{footer}"


def _edit_checkpoint(ctx, args: dict) -> str:
    """The model that renders an instruction edit. Always FireRed: the base
    Qwen-Image-Edit alternative (the GUI 🧬 toggle and the tools' 'engine'
    argument) was removed from the product."""
    return "firered"


def _undo(ctx, state) -> str:
    """«нет, верни как было» is the previous FILE, not a redraw of it."""
    said = f'{state.get("user_input_original") or ""} {state.get("user_input") or ""}'
    hist = getattr(ctx, "image_undo", None) or []
    import intent
    prev = hist[-1] if hist and said.strip() and intent.read(ctx, said.strip())["undo"] else ""
    if not (prev and os.path.exists(prev)):
        return ""
    ctx.image_undo = hist[:-1]
    _set_current_image(ctx, state, prev)
    return ("[done] The picture before the last change is back, exactly as it was -- "
            "nothing was redrawn. It is on screen. Briefly confirm, in the user's language.")


def _handle_inpaint_image(ctx, state, args: dict) -> str:
    source = state.get("image_path") or getattr(ctx, "last_image_path", None)
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
    # "без шарфа") and take-off/сними verbs the plain verb regex used to miss —
    # those were routed as "change the scarf to ..." and came back recoloured.
    removal = image_mod._is_removal_of(instructions, region, ctx=ctx)
    # A region that names an ABSENCE ("the area where the vase was") is not a
    # thing to remove: the thing is already gone. Live 2026-09-25 (journey #39)
    # the model re-ran a removal on that phrase after a good one; grounding
    # picked a patch of wall and the delivered picture got a pasted light square.
    if removal and image_mod._item_attributes(ctx, region).get("former_place") is True:
        return ("[TOOL ERROR] region names where something WAS, so it is already removed. "
                "Do not remove again: look at the current picture (inspect_image) and "
                "answer the user.")
    # Clear any stale still-visible verdict from a PREVIOUS removal so paths that
    # don't run the post-removal check (whole-frame mode, contained non-route edits)
    # can't inherit it and falsely report failure on a successful edit.
    if hasattr(image_mod, "_REMOVAL_VERIFY"):
        image_mod._REMOVAL_VERIFY["incomplete"] = False
        image_mod._REMOVAL_VERIFY["target"] = ""

    # Short-circuit for additive instructions: "add/insert/place X in/above/through
    # the Y" with an explicit region Y. Building synth = "change the sky to fried
    # chicken" → classify → subject_edit → inpaint_region → replaces the ENTIRE sky.
    # Instead, hand the router an explicit "add" request: it moves the boxes of
    # our own Ideogram picture, and has FireRed add the object to a user's photo.
    engine = _edit_checkpoint(ctx, args)
    manual_whole = ((getattr(ctx, "image_edit_engine", "auto") or "auto")
                    .strip().lower() == "firered_whole")
    if (not removal and not manual_whole and region
            and classify_edit_intent(instructions) == "object_insert"):
        ctx.set_stage("Inserting object")
        logger.info("Tool: inpaint_image additive-shortcut -> route_edit_request "
                    "anchor=%r instructions=%r source=%s", region, instructions[:80], source)
        synth = (instructions if region.lower() in instructions.lower()
                 else f"{instructions} ({region})")
        category, new_path = route_edit_request(ctx, source, synth)
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
        category = classify_edit_intent(synth)
        ctx.set_stage("Removing from the picture" if removal else "Editing the picture")
        logger.info("Tool: inpaint_image(region=%r, instructions=%r, removal=%s) -> intent=%s, source=%s",
                    region, instructions[:80], removal, category, source)

        _has_layout = bool(image_mod.load_layout_for(source))
        import image_lettering_remove as _lett
        _lettering_out = None
        if removal and not _has_layout and _lett.is_lettering_removal(region, instructions):
            # "Remove all the lettering": the mask is every piece the OCR reads,
            # not the one strip a segmenter finds for a phrase (journey 3,
            # 2026-09-18: "INSGATE" survived on the left). A picture with a
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
            new_path = image_mod.convert_colour(source, instructions or synth)
            edit_path = "convert_colour"
        elif classify_edit_intent(instructions or "") == "transform":
            import image_router
            new_path = image_router.transform_image(source, instructions)
            edit_path = "transform"
        elif category in ("text_edit", "text_add") or (_has_layout and category in image_mod._LAYOUT_EDITABLE):
            # A picture WE composed is edited through its boxes, and this branch
            # runs BEFORE the FireRed category list and before whole-frame mode.
            # Both of those repaint pixels, and repainting re-imagines whatever it
            # covers: asked to make a man blond, the whole-frame edit blended a
            # blonde woman's face over his. The layout moves only what was asked
            # for and redraws the scene from the same seed. The whole-frame toggle
            # is deliberately overridden — it chooses between renderers, and the
            # question here is not which renderer.
            _cat, new_path = route_edit_request(ctx, source, synth)
            edit_path = f"route_edit_request:{_cat}"
        elif manual_whole:
            # GUI "Whole frame" switch ON: bypass the category router and every masked
            # pipeline; the whole image goes through the FireRed instruction edit.
            edit_path = "inpaint_region_with_comfy:whole-frame-mode"
            new_path = inpaint_region_with_comfy(ctx, source, region, instructions,
                                                 removal=removal, engine=engine)
        elif category == "style_transfer":
            # A style change (anime/cartoon/oil painting/...) applies to the WHOLE
            # picture, never to just the region the agent happened to name (it called
            # inpaint_image with e.g. region="her"/"the girl", not "whole image").
            # Live, 2026-09-19: the contained pipeline segmented that region, cropped
            # it, re-rendered ONLY the crop in the new style, and composited it back
            # over the untouched original -- so the result came back with a hard seam
            # cutting through the head, half photorealistic and half anime. Passing an
            # empty region here forces the whole-frame branch inside
            # inpaint_region_with_comfy (no segmentation, no crop, no composite).
            edit_path = "inpaint_region_with_comfy:style-whole-frame"
            new_path = inpaint_region_with_comfy(ctx, source, "", instructions,
                                                 removal=removal, engine=engine)
        elif category in ("subject_edit", "face_edit", "clothing_edit", "product_edit",
                          "multi_op"):
            edit_path = "inpaint_region_with_comfy"
            new_path = inpaint_region_with_comfy(ctx, source, region, instructions,
                                                 removal=removal, engine=engine)
        else:
            _cat, new_path = route_edit_request(ctx, source, synth)
            edit_path = f"route_edit_request:{_cat}"

    # AGENT/UI BOUNDARY: this is the exact object handed back to the agent and then
    # the GUI (state['image_path'] -> final['image_path'] -> images_panel). Log the
    # full delivery record and HARD-REJECT any intermediate scratch tile so a
    # cropped working artifact can never be surfaced as a final result.
    logger.info("Tool: inpaint_image DELIVERY [BUILD_ID=%s] edit_path=%s source=%s -> returned=%s",
                getattr(image_mod, "IMAGE_BUILD_ID", "?"), edit_path, source, new_path)
    image_mod.log_edit_decision(
        request=f"region={region!r} instructions={instructions!r}",
        classifier=category, tool=f"inpaint_image -> {edit_path}",
        workflow=("contained-firered / firered fallback"
                  if edit_path == "inpaint_region_with_comfy"
                  else "route_edit_request specialized pipeline"),
        returned_file=new_path, source=source,
        extra={"removal": removal})
    new_path = image_mod.assert_deliverable(new_path, where="tools._handle_inpaint_image",
                                            source_path=source)
    if not new_path or not os.path.exists(new_path):
        failure = _INPAINT_FAILURE.get("reason", "server_error")
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
                "isolates just the target object, or use a narrower spatial description."
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
            return image_mod._NO_LAYOUT_NOTE
        else:
            detail = "the ComfyUI server failed to process the request"
        return (
            f"[TOOL ERROR] Could not edit the image — {detail}. "
            "Do NOT claim a new image was produced."
        )

    ctx.last_image_path = new_path
    # Fold the edit into the running subject prompt so a later redraw_image (which
    # re-renders from ctx.last_image_prompt) keeps this change instead of reverting it.
    base_prompt = (getattr(ctx, "last_image_prompt", "") or "").strip()
    if base_prompt:
        ctx.last_image_prompt = (f"{base_prompt}, (no {region}, removed)" if removal
                                 else f"{base_prompt}, ({region}: {instructions})")
    state["image_path"] = new_path
    state["image_status"] = "ok"
    _remember(ctx, state, "inpaint", f"Edited {region}: {instructions}", {"path": new_path})
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
        image_mod._LAYOUT_EDIT_PROBLEMS.clear()     # reported once, for this edit
        return (
            f"Image edited ({verb}) and saved: {new_path}\n"
            "[LETTERING CHECK FAILED] The text reader still sees problems in the new "
            "picture: " + " | ".join(_probs) + "\nDo NOT say it is done or correct. Show "
            "it, tell the user plainly that the lettering came out wrong (say what is "
            "wrong) and offer to try again.")
    return (
        f"Image edited ({verb}) and saved: {new_path}\n"
        "Before answering, call inspect_image to verify THE REQUESTED CHANGE — check "
        "only what the user asked for, not overall quality or pre-existing flaws (the "
        "mask can occasionally land on the wrong object); if the requested change is "
        "wrong, retry inpaint_image with a more specific region."
    )


def _handle_transfer_image(ctx, state, args: dict) -> str:
    """Multi-image reference transfer: take a thing from one or more reference
    images and apply it to the target (the most recent image). Reference images
    come from ctx.reference_images (everything loaded/pasted this session); the
    newest is the target, the earlier ones are the references."""
    imgs = [p for p in (getattr(ctx, "reference_images", None) or []) if p and os.path.exists(p)]
    # de-dupe preserving order
    seen = set(); imgs = [p for p in imgs if not (p in seen or seen.add(p))]
    target = state.get("image_path") or getattr(ctx, "last_image_path", None)
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
    # target, so the target's own file came back as a "reference". Live 10-06
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

    refs = []
    for i, p in enumerate(references):
        role = None
        if i < len(roles) and roles[i] in image_mod._ALL_ROLES:
            role = roles[i]
        refs.append(image_mod.ReferenceImage(p, role))

    ctx.set_stage("Transferring between images")
    logger.info("Tool: transfer_image(target=%s, refs=%d, roles=%r, instr=%r)",
                os.path.basename(target), len(refs), roles, instructions[:80])
    new_path = image_mod.plan_and_execute_transfer(ctx, target, refs, instructions)
    new_path = image_mod.assert_deliverable(new_path, where="tools._handle_transfer_image",
                                            source_path=target)
    if not new_path or not os.path.exists(new_path):
        return ("[TOOL ERROR] Could not complete the transfer — the editor returned no "
                "result. Tell the user the transfer failed; do NOT claim it worked.")
    _set_current_image(ctx, state, new_path)
    ctx.reference_images = (ctx.reference_images or []) + [new_path]
    _remember(ctx, state, "transfer", f"Reference transfer: {instructions}", {"path": new_path})
    return (
        f"Transfer complete and saved: {new_path}\n"
        "Call inspect_image to verify the transferred element is present and the person's "
        "identity is preserved before telling the user it is done."
    )


def _handle_fix_hands(ctx, state, args: dict) -> str:
    """Repair deformed hands/fingers in the current image via the MeshGraphormer
    Hand Refiner (geometry-guided depth-ControlNet inpaint of only the hand)."""
    source = state.get("image_path") or getattr(ctx, "last_image_path", None)
    _note_source(state, source)
    if not source or not os.path.exists(source):
        return ("[TOOL ERROR] No image to fix — generate or load an image first, then "
                "call fix_hands. Tell the user there is no picture to work on yet.")

    engine = _edit_checkpoint(ctx, args)
    ctx.set_stage("Fixing the hands")
    logger.info("Tool: fix_hands(source=%s, engine=%s)", source, engine)
    try:
        new_path = image_mod.fix_hands(ctx, source, engine=engine)
    except Exception as exc:
        logger.exception("fix_hands raised")
        return (f"[TOOL ERROR] The hand-repair pipeline failed with an internal error: {exc}. "
                "Tell the user it could not complete; do NOT claim the hands were fixed.")

    logger.info("Tool: fix_hands DELIVERY [BUILD_ID=%s] source=%s -> returned=%s",
                getattr(image_mod, "IMAGE_BUILD_ID", "?"), source, new_path)
    image_mod.log_edit_decision(
        request="fix hands/fingers", classifier="hand_repair",
        tool="fix_hands -> MeshGraphormer Hand Refiner",
        workflow="meshgraphormer depth-controlnet masked inpaint",
        returned_file=new_path, source=source, extra={})
    new_path = image_mod.assert_deliverable(new_path, where="tools._handle_fix_hands",
                                            source_path=source)
    if not new_path or not os.path.exists(new_path):
        return ("[TOOL ERROR] Could not fix the hands — the refiner returned no result "
                "(ComfyUI may have failed, or no hand was detected in the image). Do NOT "
                "claim the hands were fixed; ask the user to confirm a hand is visible, "
                "or try inpaint_image on the specific hand region.")

    _set_current_image(ctx, state, new_path)
    _remember(ctx, state, "fix_hands", "Repaired the hands/fingers", {"path": new_path})
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
    source = state.get("image_path") or getattr(ctx, "last_image_path", None)
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
        new_path = image_mod.edit_region_contained_via_firered(
            ctx, source, region, instruction, protect_face=True, engine=engine)
    except Exception as exc:
        logger.exception("fix_artifact raised")
        return (f"[TOOL ERROR] The artifact-repair pipeline failed with an internal error: {exc}. "
                "Tell the user it could not complete; do NOT claim it was fixed.")

    logger.info("Tool: fix_artifact DELIVERY [BUILD_ID=%s] source=%s -> returned=%s",
                getattr(image_mod, "IMAGE_BUILD_ID", "?"), source, new_path)
    image_mod.log_edit_decision(
        request=f"fix artifact region={region!r} issue={issue!r}",
        classifier="artifact_repair", tool=f"fix_artifact -> {engine}",
        workflow="contained masked inpaint (Florence/SAM region + FireRed)",
        returned_file=new_path, source=source, extra={"engine": engine})
    new_path = image_mod.assert_deliverable(new_path, where="tools._handle_fix_artifact",
                                            source_path=source)
    if not new_path or not os.path.exists(new_path):
        return ("[TOOL ERROR] Could not fix that area — the region may not have been located, "
                f"or the edit was rejected by quality checks. Do NOT claim it was fixed; ask the "
                f"user to name the spot differently (a clearer noun for where '{region}' is), or "
                "suggest the GUI '🩹 Fix artifact' button to draw the area by hand.")

    _set_current_image(ctx, state, new_path)
    # Avoid a double article ("the the smear") when the region phrase already leads with one.
    region_disp = region if region.lower().startswith(("the ", "a ", "an ")) else f"the {region}"
    _remember(ctx, state, "fix_artifact", f"Repaired {region_disp}", {"path": new_path})
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
    ts = int(__import__("time").time() * 1000)
    dest = os.path.join(str(image_mod.OUTPUT_DIR), f"webphoto_{ts}.jpg")
    try:
        image_mod.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    photo = _search.fetch_reference_photo(ctx, query, dest, require_face=want_face)
    if not photo or not os.path.exists(photo):
        return (f"[TOOL ERROR] Could not find a usable {'photo with a clear face' if want_face else 'photo'} "
                f"for '{query}'. Tell the user no suitable photo was found; do NOT claim one was posted.")

    # Make it the working image AND a reference (so a later transfer can use it too).
    ctx.last_image_path = photo
    ctx.reference_images = (ctx.reference_images or []) + [photo]
    state["image_path"] = photo
    state["image_status"] = "success"
    _remember(ctx, state, "find_photo", f"Posted a web photo for: {query}", {"path": photo})
    logger.info("Tool: find_photo(%r) -> %s", query[:60], os.path.basename(photo))
    return (
        f"Found a photo for '{query}' and posted it to the chat as the current working image: {photo}\n"
        "It is now loaded — the user can ask for an edit (e.g. 'make the dress red') and it will be "
        "inpainted on THIS photo. Tell the user the photo is ready and ask what to change."
    )


def _render_budget_exhausted(state, what: str, cost: int = 1) -> Optional[str]:
    """Spend `cost` of this turn's full renders; returns a refusal when they run out.

    The agent is told to inspect its picture and fix what is wrong "until it is
    right" — good advice with no cost attached. Live, one turn ran redraw →
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
    Latin letters (live 2026-09-28). A transliteration of the quoted name is
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


def _user_look(state) -> str:
    """The look THIS message asks for: a drawn look, "photo", "none", or "" when
    there are no user words to read (a direct call). Only this message: «нарисуй
    кота в стиле аниме», then «нарисуй собаку» drew the dog in anime too
    (owner 10-03: стиль только из текущего сообщения). A redo («ещё раз») says
    no look of its own and repeats the request before it, look included."""
    from prompt_guard import user_words
    from ideogram_layout import _look
    st = state or {}
    text = user_words(str(st.get("user_input_original") or st.get("user_input") or "")).strip()
    if not text:
        return ""
    got = _look(text)
    if got != "none":
        return got
    try:
        import intent
        redo = intent.read(None, text)["redo"] in ("same", "changed")
    except Exception:
        redo = False
    if redo:
        for m in reversed(st.get("messages") or []):
            if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str):
                t = user_words(m["content"]).strip()
                if t and t != text:
                    return _look(t)
    return "none"


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

    _over = _t._render_budget_exhausted(state, "generate_image")
    if _over:
        return _over

    description = _keep_quoted_names(description, state)
    import person_look
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
            ref_out, info = image_mod.generate_person_from_reference(ctx, description, seed=seed)
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
            _remember(ctx, state, "generate", f"Reference-person image: {description}", {"path": ref_out})
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
    ctx.user_look = _user_look(state)
    try:
        result = _t.generate_image_with_refinement(
            ctx=ctx, description=description, steps=steps, width=width, height=height, seed=seed,
        )
    finally:
        ctx.user_look = _prev_look
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
        _remember(ctx, state, "generate", f"Image for task: {description}", {"path": img_path})
        quality_note = " (base quality — refinement unavailable)" if status == "partial" else ""
        # An ACCEPTED render is final for this turn. The renderer has already
        # judged it with the same vision model the agent would use; a second
        # look does not see more, it invents -- live, 2026-09-11, a 10/10
        # picture was "fixed" for a shadow that was not there, lost its subject,
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
            # garbled name on one of them (live 2026-09-28): say what is wrong.
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
    reason = image_mod._GENERATE_FAILURE.get("reason", "server_error")
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
    started from, under its own filename. Live, 2026-09-19: pressing Animate
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


# A clip is minutes of GPU and the whole turn: it is made only when the user's
# own words ask for one. Live 10-03: «Почему он у тебя не лысый … ты нарисовал
# какого-то урода, а не Ленина» -- a complaint about a PICTURE -- started a
# video («🎬 делаю видео»), then «лимит на генерацию видео исчерпан».
_ASKS_VIDEO = re.compile(
    r"видео|видос|ролик|клип|аними|анимац|ожив|кружок|кружоч|мультфильм|мультик\b|"
    r"\bгиф|\bgif|video|clip|animat|movie|footage|\breel|shorts|фильм", re.I)


def asks_for_video(ctx, state) -> bool:
    """The user's words this turn ask for a clip, or ask to redo the last one."""
    from prompt_guard import user_words
    said = user_words(str((state or {}).get("user_input_original") or "")
                      + " " + str((state or {}).get("user_input") or "")).strip()
    if not said or _ASKS_VIDEO.search(said):     # no words to judge (a direct call): not ours to refuse
        return True
    import intent
    last = getattr(ctx, "last_video_path", None)
    if last and os.path.exists(last):
        try:
            if intent.read(None, said)["redo"] in ("same", "changed"):
                return True
        except Exception:
            pass
    # A shot script with no word «видео» is still a clip. Live 10-06: «Мужчина берёт
    # банку … говорит "Полная хуета" … бросает банку … опрокидывает стеллаж» was
    # refused here. The model reads it; a complaint about a picture stays a no.
    return intent.ask_yes(
        "A user wrote: {text}\n\nIs this a request for a video clip: a scene that unfolds "
        "over time (people act one after another, move, speak lines), not a single still "
        "picture, a complaint about a picture, or a question?", said)


def _render_remaining_parts(ctx, path: str, parts: list, args: dict) -> str:
    """Parts 2..n of a long script, each continuing the clip so far from its real
    last frame (its tail as <Video 1>, that frame as <Picture 1>), joined on. A part
    that fails or a cancel stops here: the clip made so far is still delivered."""
    import tempfile
    import tg_continue
    import video as video_mod
    work = tempfile.mkdtemp(prefix="vidparts_")
    for i, part in enumerate(parts[1:], start=2):
        if getattr(ctx, "is_cancelled", None) and ctx.is_cancelled():
            break
        ctx.set_stage(f"Generating a video ({i}/{len(parts)})")
        frame = os.path.join(work, f"seed_{i}.jpg")
        tail = os.path.join(work, f"tail_{i}.mp4")
        if not (tg_continue.seed_frame(path, frame) and tg_continue.cut_tail(path, tail)):
            logger.warning("video parts: could not take the end of part %d", i - 1)
            break
        pinned = video_mod.motion_context_on()
        try:
            if pinned:
                r = video_mod.generate_video(ctx, video_mod.CONTINUE_CTX_PREFIX + part,
                                             context_video=tail,
                                             aspect=args.get("aspect") or "", seed=args.get("seed"))
            else:
                r = video_mod.generate_video(ctx, video_mod.CONTINUE_PREFIX + part,
                                             images=[frame], videos=[tail],
                                             aspect=args.get("aspect") or "", seed=args.get("seed"))
        except Exception:
            logger.exception("video parts: part %d crashed", i)
            break
        new = r.get("path")
        if not (new and os.path.exists(new)):
            logger.warning("video parts: part %d produced no clip (%s)", i, r.get("reason"))
            break
        # pinned: the new part starts on the frame after the old one ends -- butt-join
        joined = (video_mod.join_pinned(path, new) if pinned
                  else video_mod.join_continuation(path, new))
        if not joined:
            logger.warning("video parts: joining part %d failed", i)
            break
        path = video_mod._adopt_output(joined)
        logger.info("video parts: %d/%d joined -> %s", i, len(parts), os.path.basename(path))
    return path


def _handle_generate_video(ctx, state, args: dict) -> str:
    import tools as _t   # the render seam is owned by tools; read it at CALL time
    import video as video_mod

    description = (args.get("description") or "").strip()
    if not description:
        return ("[TOOL ERROR] generate_video needs a 'description' of the shot. "
                "Re-call it describing what should be on screen, what moves, and "
                "what it should sound like.")
    if not asks_for_video(ctx, state):
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
    _over = _t._render_budget_exhausted(
        state, "generate_video",
        cost=max(1, int(getattr(config, "IMAGE_MAX_RENDERS_PER_TURN", 3) or 3)))
    if _over:
        return _over

    images, videos = [], []
    if args.get("use_current_images", True):
        images = _current_image_paths(ctx, state)
        # More than H3 can take: keep the most RECENT, which is what the user is
        # most likely talking about.
        if len(images) > video_mod.MAX_REF_IMAGES:
            images = images[-video_mod.MAX_REF_IMAGES:]
    if args.get("use_current_video"):
        lv = getattr(ctx, "last_video_path", None)
        if lv and os.path.exists(lv):
            videos = [lv]

    # ▶️ Continue video (tg_continue): the clip's tail is the reference -- its motion, faces and
    # its own sound -- and the new part is joined to the original afterwards.
    cont_tail = getattr(ctx, "continue_tail", "") or ""
    pinned_tail = ""
    cont_src = getattr(ctx, "continue_src", "") or ""
    continuing = bool(cont_tail and os.path.exists(cont_tail))
    if continuing:
        videos = [cont_tail]
        ctx.continue_tail = ctx.continue_src = ""          # one clip per request
        people = [p for p in (getattr(ctx, "continue_people", None) or []) if p and os.path.exists(p)]
        ctx.continue_people = []
        # Pinned frames (same as the parts of a long script): faster and seamless, but they
        # take no pictures -- new people from photos still need the reference path.
        if not people and video_mod.motion_context_on():
            pinned_tail, videos, images = cont_tail, [], []
            description = video_mod.CONTINUE_CTX_PREFIX + description
        else:
            description = video_mod.CONTINUE_PREFIX + description
        if people:
            # New people from photos sent after the clip: the start frame stays <Picture 1>.
            images = (images or [])[-1:] + people
            description += video_mod.new_people_clause(len(images) - len(people) + 1, len(people))

    audios = []
    anim = ([] if continuing else
            [p for p in (getattr(ctx, "anim_voices", None) or []) if p and os.path.exists(p)])
    speakers = int(args.get("speakers") or 0)
    if not speakers:
        # The model forgets `speakers` (live 10-02: two quoted lines, speakers=0,
        # no voice question). Spoken lines in quotes are a fact, not a guess.
        speakers = min(len(re.findall(r'«[^»]{2,}»|"[^"]{2,}"|“[^”]{2,}”',
                                      args.get("description") or "")), video_mod.MAX_REF_AUDIOS)
    if (speakers and not anim and not args.get("use_my_voice")
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
        audios = anim[:video_mod.MAX_REF_AUDIOS]
        ctx.anim_voices = []                      # one clip per collection
        if "<Audio 1>" not in description:
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
        # ref_video_N): with voice samples it outvoiced them -- live 10-01, a redo
        # «с баяном и ударами» reused the last clip and the user's voices were gone.
        logger.info("generate_video: voice samples given -> the previous clip is not a reference")
        videos = []
    # A script longer than one clip holds is rendered in parts, each the next one's
    # continuation from the last frame (the user's idea, 10-06: "split it into 5.2 s
    # blocks"): the first part here, the rest after it below.
    raw = (args.get("description") or "").strip()
    parts = [raw] if args.get("seconds") else video_mod.split_script(raw)
    if len(parts) > 1 and raw in description:
        description = description.replace(raw, parts[0], 1)
        logger.info("generate_video: the script needs %d parts: %s", len(parts),
                    [round(video_mod.estimate_seconds(p), 1) for p in parts])
    else:
        parts = [raw]
    mode = video_mod.pick_mode(images, videos, audios)
    tags = video_mod.reference_tags(len(images) if mode == "ref2va" else 0, len(videos), len(audios))
    logger.info("Tool: generate_video(mode=%s, %d image(s), %d video(s)) %s",
                mode, len(images), len(videos), description[:80])

    ctx.set_stage("Generating a video")
    try:
        result = video_mod.generate_video(
            ctx, description, images=images, videos=videos, audios=audios,
            seconds=args.get("seconds") or 0.0,
            aspect=args.get("aspect") or "",
            seed=args.get("seed"),
            context_video=pinned_tail,
        )
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

    if continuing and cont_src and os.path.exists(cont_src):
        joined = (video_mod.join_pinned(cont_src, path) if pinned_tail
                  else video_mod.join_continuation(cont_src, path))
        if joined:
            path = video_mod._adopt_output(joined)       # the whole thing: original + what comes next
        else:
            logger.warning("continue video: the join failed, delivering the new part alone")

    if len(parts) > 1:
        path = _render_remaining_parts(ctx, path, parts, args)
        result["seconds"] = video_mod.probe(path).get("seconds") or result.get("seconds")

    state["video_path"] = path
    state["video_status"] = "success"
    state["video_seconds"] = result.get("seconds")
    ctx.last_video_path = path
    ctx.last_video_prompt = description
    _remember(ctx, state, "video", f"Video for: {description}", {"path": path})

    used = ""
    if tags:
        used = f" References used: {', '.join(tags)}."
    return (
        f"Video generated and saved: {path}\n"
        f"Mode: {video_mod.MODE_LABELS.get(mode, mode)}; "
        f"{result.get('seconds')}s at {result.get('width')}x{result.get('height')}, "
        f"with generated audio.{used}\n"
        "The clip is delivered to the user automatically — just tell them it is ready "
        "in one short sentence. Do NOT call generate_video again for this turn."
    )


def redraw_whole_image(ctx, source: str, instructions: str):
    """Re-render the WHOLE picture following `instructions`. Returns (path, engine).

    Our own Ideogram picture is re-rendered from its stored boxes; anything
    else (a user's photo) goes through a mask-free FireRed edit. Shared by the
    redraw_image tool and the GUI Redraw / Style preset buttons, which used to
    call an old model controlnet graph whose checkpoint no longer exists.
    """
    new_path = None
    # Which engine ACTUALLY ran, for the audit log.
    engine = ""
    if image_mod.load_layout_for(source):
        ctx.set_stage("Redrawing from the picture's own layout")
        logger.info("Tool: redraw_image -> layout edit (Ideogram boxes): %r",
                    instructions[:80])
        try:
            new_path = image_mod.edit_via_layout(ctx, source, instructions)
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
        # Live, 2026-09-20: passing the agent's raw instruction straight
        # through (as before) cropped a person out of frame on a style
        # request even after route_edit_request's style_transfer path was
        # fixed, because THIS path never called it. Route a style change
        # through the same anti-crop + vision-layout-locked builder so both
        # entry points get the same protection.
        if classify_edit_intent(instructions) == "style_transfer":
            firered_instr = image_mod._firered_style_instruction(
                ctx, instructions, image_path=source)
            logger.info("Tool: redraw_image -> mask-free FireRed whole-image STYLE "
                       "edit (layout-locked): %r", instructions[:80])
        else:
            firered_instr = instructions
            logger.info("Tool: redraw_image -> mask-free FireRed whole-image edit: %r",
                        instructions[:80])
        try:
            new_path = image_mod.edit_image_with_firered(ctx, source, firered_instr)
            engine = "edit_image_with_firered (whole frame, NO mask)"
        except Exception:
            logger.exception("FireRed redraw failed for %s", source)

    # Every other edit handler in this file (inpaint, transfer, fix_hands,
    # fix_artifact) runs its result through assert_deliverable before it can reach
    # the user; this one never did. Live, 2026-09-19: a redraw delivered
    # "_INTERMEDIATE_tile_firered_00001_.png" straight to the user's chat — a
    # cropped FireRed working tile, not a finished picture. redraw/enhance can
    # fall through several engines (layout edit, whole-frame FireRed, the old model
    # controlnet, img2img) and any one of them returning a stale or scratch path
    # had nothing to catch it.
    new_path = image_mod.assert_deliverable(new_path, where="tools._handle_redraw_image",
                                            source_path=source)

    # Whole-frame re-renders (redraw/enhance) repaint the FACE too — at low res this
    # makes the person unrecognizable ("enhancing changed the face"). Composite the
    # ORIGINAL face back unless the user explicitly asked to change the face.
    #
    # A STYLE change (anime/cartoon/oil painting/…) is the one whole-frame redraw
    # where this must NOT run at all. Live, 2026-09-19: sending the same photo
    # for an anime restyle over and over kept coming back with an anime BODY and
    # the original PHOTOREALISTIC face pasted on top — preserve_identity_face's
    # own face-change detector only catches an explicit attribute edit
    # ("sunglasses", "expression"), not "in anime style", so it took the literal
    # pixel paste-back branch every time. Redrawing a face in the target style
    # is the entire point of a style request, not an unwanted side effect to
    # undo.
    if (new_path and os.path.exists(new_path)
            and classify_edit_intent(instructions) != "style_transfer"):
        new_path = image_mod.preserve_identity_face(source, new_path, instructions=instructions)
    return new_path, engine


def _handle_redraw_image(ctx, state, args: dict) -> str:
    import tools as _t   # the render seam is owned by tools; read it at CALL time
    source = state.get("image_path") or getattr(ctx, "last_image_path", None)
    _note_source(state, source)
    if not source or not os.path.exists(source):
        return (
            "[TOOL ERROR] No image to redraw — generate an image first with generate_image, "
            "then call redraw_image. Tell the user there is no picture to work on yet."
        )
    _undone = _undo(ctx, state)
    if _undone:
        return _undone

    _over = _t._render_budget_exhausted(state, "redraw_image")
    if _over:
        return _over

    mode = (args.get("mode") or "").strip().lower()
    instructions = (args.get("instructions") or "").strip()
    # 'enhance'/'upscale'/'restore'/'outpaint' used to be distinct pipelines, all
    # built on the old model checkpoint. That model was removed from the product
    # (Ideogram + FireRed only) and its files deleted from disk, so those modes
    # have no engine left to run on. Every redraw now goes through the same
    # FireRed path below; an empty instruction gets a generic quality nudge so a
    # bare "enhance" request still does something instead of a no-op.
    mode = "redraw"
    if not instructions:
        instructions = "improve the overall quality: sharpen fine detail, clean up noise and artifacts"

    # A colour conversion is exact arithmetic and must never be re-rendered,
    # whichever tool the model reaches for. Live: asked for black and white, the
    # generative path returned a DIFFERENT TRACTOR — in colour — and the model
    # reported failure while the picture was delivered anyway. Delegating here
    # (not only in route_edit_request) means the deterministic path wins even
    # when the request arrives as a "redraw".
    if instructions and classify_edit_intent(instructions) in ("colour_convert", "transform"):
        from image import convert_colour
        import image_router
        ctx.set_stage("Converting the colours")
        logger.info("Tool: redraw_image -> exact pixel operation: %r", instructions[:80])
        new_path = (convert_colour(source, instructions)
                    if classify_edit_intent(instructions) == "colour_convert"
                    else image_router.transform_image(source, instructions))
        if not new_path or not os.path.exists(new_path):
            return ("[TOOL ERROR] The operation failed (a crop to the face needs a "
                    "face the detector can find). Tell the user plainly; do NOT "
                    "claim the picture was changed.")
        _set_current_image(ctx, state, new_path)
        if classify_edit_intent(instructions) == "transform":
            return ("[done] The picture was rotated/mirrored/cropped exactly on its "
                    "pixels -- nothing was redrawn. The result is on screen. Briefly "
                    "confirm to the user, in their language.")
        return ("[done] The picture was converted exactly — same subject, same "
                "resolution, only the colours changed. The result is on screen. "
                "Do NOT say any part of it is still in colour: it is not. "
                "Briefly confirm to the user, in their language.")

    import image_router
    if image_router.edit_plan(instructions)["reformats"]:
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
    ctx.user_look = _look(_said) if _said else ""
    try:
        new_path, engine = redraw_whole_image(ctx, source, instructions)
    finally:
        ctx.user_look = _prev_look
    verb = "redrawn"
    if new_path and os.path.exists(new_path):
        ctx.last_image_path = new_path
        state["image_path"] = new_path

    try:
        image_mod.log_edit_decision(
            request=f"instructions={instructions!r}",
            classifier=f"mode={mode}", tool="redraw_image",
            workflow=engine or "redraw: no engine produced a file",
            returned_file=new_path, source=source)
    except Exception:
        pass
    if not new_path or not os.path.exists(new_path):
        return (
            "[TOOL ERROR] Could not redraw the image — "
            "the ComfyUI server returned no result. Tell the user the redraw failed; "
            "do NOT claim a new image was produced."
        )

    ctx.last_image_path = new_path
    # ctx.last_image_prompt is deliberately NOT overwritten with the bare
    # instructions: it describes the picture's subject, and a later redraw needs it.
    state["image_path"] = new_path
    state["image_status"] = "ok"
    _remember(ctx, state, "redraw", f"Image {verb}" + (f": {instructions}" if instructions else ""),
              {"path": new_path})
    result = f"Image {verb} and saved: {new_path}"
    if mode == "redraw" and instructions:
        result += (
            "\nNext step: call inspect_image to verify the requested change is actually "
            "visible before telling the user it is done; a redraw can drift from the "
            "instructions. If it did not come out, retry or say so honestly."
        )
    return result

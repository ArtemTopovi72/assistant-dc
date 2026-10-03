"""Text-to-image generation and the refinement loop.

Extracted from image.py. The generation side of the pipeline: the Ideogram
drawing-mode graph, generate_image_with_comfy (Ideogram only), the vision
evaluator, and the generate/evaluate/retry refinement loop.

SEAM NOTE -- read before adding an import here.
Everything the moved code calls that a suite might stub is reached through
`_image.` rather than imported by value: the LLM boundary (call_llm_simple,
analyze_image_with_llm, safe_json_from_llm), the sizing helpers (_snap_to_8,
_source_dims, _enforce_output_size, fix_image_params, normalize_resolution,
parse_generation_params, session_image_size, session_size_pinned), the ComfyUI
boundary (_upload_image_to_comfy, _submit_and_poll), save_layout_for, the two
prompt templates and the two workflow paths. A by-value import would bind the
same name on BOTH sides of the split, so patching `image.X` would move only
half the behaviour: the stub would silently die and the REAL call would go out
to LM Studio or ComfyUI while the suite still printed PASS.

_GENERATE_FAILURE and _ENGINE_JUDGED are module-level mutable state that
image.py, tools and the suites all read; they are mutated through the proxy so
their identity cannot drift.

The config defaults below are the exception: they are DEFAULT ARGUMENT values
and must bind at def time, so they are imported from config -- their DEFINING
module -- not from image.py.
"""
import logging
import random
from typing import Optional

from config import (DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_STEPS,
                    DEFAULT_CFG, MAX_IMAGE_REFINEMENT_ATTEMPTS)
import config as _config

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


def _ideogram_draw(ctx, prompt: str, *, width: int, height: int,
                   seed: Optional[int], timeout: int,
                   lora_name: Optional[str] = None, lora_strength: float = 1.0,
                   trigger: Optional[str] = None,
                   steps: Optional[int] = None, cfg: Optional[float] = None) -> Optional[str]:
    """Draw with Ideogram 4 — through the verify/repair loop when there is lettering.

    `steps`/`cfg` left as None means Ideogram's own turbo LoRA gate decides
    (see ideogram.generate); passing them explicitly (the "ultra" quality tier)
    turns that LoRA off and runs the plain model at full quality instead.

    A scene with no text is a single submit, exactly as before: the read-back
    would have nothing to check and the extra vision call would be pure cost.
    The moment the layout asks for words in the picture, the render is looked at,
    the lettering transcribed and compared to the string that was requested, and a
    garbled one ("POLICE" rendered as "AVCHKE") is redrawn with a box sized for
    that string and the spelling restated in the caption. That comparison is the
    only check that catches it — the scene critic sees a sign where a sign was
    asked for and calls it correct.
    """
    import ideogram
    rounds = max(0, int(getattr(_config, "IDEOGRAM_TEXT_ROUNDS", 1) or 0))
    # The layout is planned ALWAYS, not only when the lettering loop is enabled.
    # IDEOGRAM_TEXT_ROUNDS=0 used to skip this whole block and submit through
    # `ideogram.generate`, which plans its own caption and throws it away — so no
    # layout was recorded, and every later "move that box" silently fell back to
    # a pixel re-render. The rounds setting governs how hard we chase garbled
    # LETTERING; it must not decide whether the picture is editable afterwards.
    # Resolve the seed HERE rather than letting the renderer pick one privately:
    # a seed we never learn cannot be recorded, and an edit that cannot reuse the
    # seed re-rolls the picture instead of adjusting it.
    if seed is None or seed < 1:
        seed = random.randint(1, 2**31 - 1)
    # A character render skips the lettering repair loop: draw_agent has no way
    # to carry the adapter, so a repair round would re-draw the picture WITHOUT
    # the LoRA and hand back a stranger. It is also the right call on the merits
    # -- the trigger word is what leaks into rendered lettering, so a character
    # scene is the last place to go chasing text.
    if lora_name:
        rounds = 0
    try:
        layout = ideogram.plan_layout(ctx, prompt)
        # The plan is checked AS A SKETCH before the card is spent on it:
        # a missing subject, an extra one, or «the cat beside the sofa»
        # when the prompt said ON it, is moved in the boxes now, not after
        # a 90-second render the critic then rejects (draw_preview).
        import draw_preview
        layout, _pre = draw_preview.preflight(ctx, layout, prompt, width=width, height=height)
        _image._LAST_PREFLIGHT.clear(); _image._LAST_PREFLIGHT.update(_pre)
        if rounds and any(str(el.get("text") or "").strip()
                          for el in (layout.get("elements") or [])):
            import draw_agent
            logger.info("Ideogram: the scene carries lettering — drawing through "
                        "the read-back loop (%d repair round(s))", rounds)
            res = draw_agent.run(ctx, prompt, layout=layout, width=width,
                                 height=height, seed=seed, rounds=rounds,
                                 steps=steps, cfg=cfg)
            if res.get("stopped") == "refused":
                raise ideogram.ContentRefused(
                    (res.get("problems") or ["Ideogram 4 declined this prompt."])[0])
            if res.get("image"):
                # Record the REPAIRED layout, not the planned one: that is the
                # arrangement the picture actually shows, and the one a later
                # edit has to start from.
                _image.save_layout_for(res["image"], prompt, res.get("layout") or layout,
                                width=width, height=height,
                                seed=res.get("seed", seed))
                # This picture has ALREADY been judged and repaired by the loop
                # above — tell the outer refinement loop so it does not judge it
                # again on a different rubric and re-draw the whole thing.
                _verdict = ((res.get("history") or [{}])[-1].get("verdict") or {})
                _image._ENGINE_JUDGED.update({
                    "judged": True,
                    "score": int(_verdict.get("score", 0) or 0),
                    "ok": res.get("stopped") == "ok",
                    "problems": list(res.get("problems") or []),
                })
                return res["image"]
            logger.warning("Ideogram text loop produced no image (%s) — falling "
                           "back to a single submit", res.get("stopped"))
        else:
            # The layout is already planned; submitting it directly saves the
            # planner call that ideogram.generate would otherwise repeat.
            out = ideogram.generate(ctx, prompt, width=width, height=height,
                                    seed=seed, timeout=timeout,
                                    caption=ideogram.layout_to_caption(layout),
                                    lora_name=lora_name,
                                    lora_strength=lora_strength,
                                    trigger=trigger, steps=steps, cfg=cfg)
            _image.save_layout_for(out, prompt, layout, width=width, height=height,
                            seed=seed)
            return out
    except ideogram.ContentRefused:
        raise
    except Exception:
        logger.exception("Ideogram text-aware drawing failed — plain submit")
    # Last resort only: the planner itself failed, so there is no layout to
    # record and this picture will not be box-editable. Logged loudly because
    # from the user's side it looks identical to a normal render.
    logger.error("Ideogram: drawing WITHOUT a recorded layout — later box edits "
                 "on this image will fall back to pixel pipelines")
    return ideogram.generate(ctx, prompt, width=width, height=height,
                             seed=seed, timeout=timeout, lora_name=lora_name,
                             lora_strength=lora_strength, trigger=trigger,
                             steps=steps, cfg=cfg)


def generate_image_with_comfy(
        ctx,
        prompt: str,
        negative_prompt: str = "",
        steps: int = 8,
        cfg: float = 1.0,
        seed: Optional[int] = None,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        timeout: int = 1900,
        previous_image_path: Optional[str] = None,
        inpaint_denoise: float = 0.5,
        lora_name: Optional[str] = None,
        lora_strength: float = 0.9,
        trigger: Optional[str] = None,
) -> Optional[str]:
    """Draw a FRESH picture with Ideogram 4, or return None.

    Ideogram is the only drawing engine. The old model (the old txt2img graph, its
    img2img "refine through the previous picture" mode, and the fallback when
    Ideogram failed) was removed from the product together with its checkpoint,
    so `negative_prompt`, `previous_image_path` and `inpaint_denoise` are kept
    only for call-site compatibility: a refine request is refused, not drawn.
    """
    # Stale from the PREVIOUS render otherwise: a picture that did not go through
    # the judged loop would inherit the last one's "already judged" verdict and
    # skip evaluation entirely.
    _image._ENGINE_JUDGED.update({"judged": False, "score": 0, "ok": False, "problems": []})

    if not (prompt or "").strip():
        logger.error("generate_image_with_comfy: empty positive prompt — refusing to submit")
        return None
    if previous_image_path:
        logger.error("generate_image_with_comfy: img2img refine was asked for, but its "
                     "engine was removed — edit the picture with FireRed instead")
        _image._GENERATE_FAILURE["reason"] = "engine_failed"
        return None

    import ideogram
    _image._GENERATE_FAILURE["reason"] = "server_error"
    # "Максимум" (ultra) is the one quality tier that asks for the plain
    # model instead of the turbo LoRA: every other tier leaves steps/cfg
    # unset so ideogram.generate's own turbo gate fires, exactly the
    # "unchosen == None, not a default" convention used for image_aspect
    # and the music steps picker -- see resolve_steps_for_generate.
    if getattr(ctx, "image_quality", "") == "ultra":
        ideo_steps, ideo_cfg = _config.IDEOGRAM_STEPS, _config.IDEOGRAM_CFG
    else:
        ideo_steps, ideo_cfg = None, None
    try:
        result = _ideogram_draw(ctx, prompt, width=width, height=height,
                                seed=seed, timeout=timeout,
                                lora_name=lora_name,
                                lora_strength=lora_strength,
                                trigger=trigger,
                                steps=ideo_steps, cfg=ideo_cfg)
    except ideogram.ContentRefused as exc:
        logger.error("Ideogram 4 refused the prompt: %s", exc)
        _image._GENERATE_FAILURE["reason"] = "refused"
        return None
    if result:
        return result
    logger.error("Ideogram 4 generation failed — reporting it (there is no other "
                 "drawing engine to fall back to)")
    _image._GENERATE_FAILURE["reason"] = "engine_failed"
    return None


def evaluate_image(ctx, img_path: str, goal: str, current_prompt: str,
                   negative_prompt: str, steps: int, cfg: float,
                   width: int, height: int) -> dict:
    """Evaluate a generated image. Returns eval dict with verdict, score, patches, etc."""
    import exposure as _exposure
    # The judge misreads exposure in one direction (see exposure.py): told the
    # measured numbers first, it stops inventing silhouettes on most pictures;
    # the ones it still invents are dropped below.
    _measured = _exposure.evidence(img_path)
    # The final render deserves the same zoomed-tile look a picture QUESTION
    # already gets (vision_close): the judge's one overview call misses small
    # text/logos/hands/marks exactly like the analyze-photo path used to,
    # and a verdict of "success" on a botched detail never gets corrected.
    _tile_notes = ""
    try:
        with open(img_path, "rb") as _fh:
            _img_bytes = _fh.read()
        import vision_close as _vclose
        _tile_notes = _vclose.tile_notes(ctx, _img_bytes, label="the generated image")
    except Exception:
        logger.warning("evaluate_image: close look failed; judging the overview only",
                       exc_info=True)
    eval_prompt = (
        f"Original task: {goal}\n"
        f"Current prompt: {current_prompt}\n"
        f"Current negative prompt: {negative_prompt}\n"
        f"Current parameters: steps={steps}, cfg={cfg}, width={width}, height={height}\n"
        + (_measured + "\n" if _measured else "")
        + (f"Small details from zoomed-in parts of the image:\n{_tile_notes}\n"
           if _tile_notes else "")
        + "Evaluate how well the image matches the task and what needs to be adjusted."
    )

    # NO prefill. A closed-<think> prefill makes these fine-tunes return EMPTY for a
    # structured-JSON vision task (verified live) — and vision itself is flaky / often
    # unavailable when a non-multimodal chat model holds the VRAM. Try twice.
    response = ""
    for _ in range(2):
        response = _image.analyze_image_with_llm(
            ctx=ctx,
            image_path=img_path,
            user_text=eval_prompt,
            system_prompt=_image.VISION_EVAL_PROMPT,
        ) or ""
        if _image.safe_json_from_llm(response):
            break

    data = _image.safe_json_from_llm(response)
    if isinstance(data, dict) and _measured \
            and str(data.get("verdict", "")).lower() != "success" \
            and _exposure.is_dark_complaint(data.get("reason", "")):
        # Live, 2026-09-12: "the cat is extremely dark, a black silhouette"
        # about a sunlit orange cat -- three renders in a row, then an
        # apology to the user for a shadow that was not there. The pixels
        # were measured; the judge was not looking at them.
        logger.info("Image eval: darkness complaint contradicted by the measured "
                    "exposure -- accepting (%s)", str(data.get("reason", ""))[:160])
        data = dict(data, verdict="success", score=max(8, int(data.get("score") or 0)),
                    reason="matches the request (a darkness complaint was dropped: the "
                           "picture measures as normally exposed)",
                    prompt_patch="", negative_prompt_patch="", exposure_overruled=True)
    if not data:
        # FAIL SAFE, not loud. When we cannot get a verdict (empty/garbled vision
        # response — e.g. no multimodal model loaded), the OLD default was
        # score=4/"refine", which forced an endless refine spiral on every prompt and
        # made the agent re-generate from scratch. With no working judge, blindly
        # re-rolling is strictly worse than keeping the image we have, so ACCEPT it
        # and stop the loop. `eval_unavailable` lets the caller report honestly.
        logger.warning("Image eval unavailable (empty/garbled vision response) — accepting current image")
        return {
            "verdict": "success", "score": 7, "eval_unavailable": True,
            "reason": "Quality check unavailable (no vision response); keeping the image as generated.",
            "prompt_patch": "", "negative_prompt_patch": "",
            "steps": 0, "cfg": 0.0, "width": 0, "height": 0,
        }
    return data


def generate_image_with_refinement(ctx, description: str, steps: Optional[int] = None,
                                   width: Optional[int] = None, height: Optional[int] = None,
                                   seed: Optional[int] = None) -> dict:
    """
    Full generation pipeline: prompt engineering → ComfyUI → eval loop.
    Returns dict with keys: path, score, attempts, prompt, status.
    """
    # Ideogram is the only drawing engine, and it is driven by a structured
    # layout caption from ideogram.plan_layout, not by a prose prompt -- so hand
    # it the user's own words. (build_image_prompt_from_llm wrote the old model prose
    # and picked the old model parameters; it is no longer on this path.)
    dw, dh = _image.session_image_size(ctx)
    prompt = description
    llm_steps, cfg = DEFAULT_STEPS, DEFAULT_CFG   # unused by Ideogram; kept for the signature
    llm_seed = random.randint(1, 999_999_999)
    llm_width, llm_height = dw, dh

    # User params override LLM params
    final_steps = steps if steps is not None else llm_steps
    final_width = width if width is not None else llm_width
    final_height = height if height is not None else llm_height
    final_seed = seed if (seed is not None and seed > 0) else llm_seed

    # Orientation fix. A session that pinned an aspect ratio (📐 Size) keeps it —
    # see fix_image_params — but that guard used to apply ONLY when the caller
    # (the agent's own tool-call args) supplied neither width nor height. An
    # agent-guessed `width=1920, height=1080` bypassed the pin completely and
    # rendered landscape under a 9:16 pin — observed live. A pin represents an
    # explicit user choice in the 📐 picker and must outrank a value the AGENT
    # invented, exactly as it already outranks the model's prompt-text guess.
    if _image.session_size_pinned(ctx) and (width is not None or height is not None):
        final_width, final_height = _image.normalize_resolution(*_image.session_image_size(ctx))
    elif width is None and height is None:
        final_width, final_height = _image.fix_image_params(
            description, prompt, final_width, final_height,
            pinned=_image.session_size_pinned(ctx))

    negative_prompt = ""
    reason = ""
    img_path = None
    prev_img_path: Optional[str] = None
    score = 0
    # Set when the judge said a requested SUBJECT is absent/wrong (prompt_patch +
    # low score). Low-denoise img2img through the flawed image preserves the very
    # composition that LACKS the subject — the patch text rarely materializes at
    # denoise 0.35–0.65 — so those retries must regenerate from fresh noise.

    for attempt in range(1, MAX_IMAGE_REFINEMENT_ATTEMPTS + 1):
        logger.info("Image generation attempt %d/%d", attempt, MAX_IMAGE_REFINEMENT_ATTEMPTS)
        if ctx is not None and getattr(ctx, "is_cancelled", None) and ctx.is_cancelled():
            break

        # Every attempt re-draws through Ideogram with the judge's patched
        # prompt; there is no img2img "repaint the previous picture" engine any
        # more (it was the old model), and it lost the layout anyway.
        img_path = generate_image_with_comfy(
            ctx=ctx,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=final_steps,
            cfg=cfg,
            seed=final_seed,
            width=final_width,
            height=final_height,
        )

        if not img_path:
            if prev_img_path:
                logger.warning("Refinement attempt %d failed; keeping best previous result (score %d/10)",
                               attempt, score)
                return {"path": prev_img_path, "score": score, "attempts": attempt,
                        "prompt": prompt, "status": "partial"}
            return {"path": None, "score": 0, "attempts": attempt, "prompt": prompt, "status": "fail"}

        prev_img_path = img_path

        # The engine already judged and repaired this picture with a rubric built
        # for it (draw_agent: read the lettering back, critique, rearrange the
        # boxes, re-render). Judging it AGAIN here — on a different rubric, with a
        # retry path that re-plans the layout and re-renders from scratch —
        # multiplies the whole render body by up to MAX_IMAGE_REFINEMENT_ATTEMPTS
        # and swaps LM Studio models a dozen more times. See _ENGINE_JUDGED.
        if _image._ENGINE_JUDGED.get("judged"):
            _score = _image._ENGINE_JUDGED.get("score") or 0
            logger.info("Ideogram drew this through its own judge/repair loop "
                        "(score=%s, ok=%s) — skipping the second evaluation pass",
                        _score, _image._ENGINE_JUDGED.get("ok"))
            return {
                "path": img_path,
                # Its rubric is 0-10 too; fall back to a neutral pass when the
                # critic gave no number so the tool layer does not report 0/10
                # for a picture the engine was satisfied with.
                "score": _score or (8 if _image._ENGINE_JUDGED.get("ok") else 6),
                "attempts": attempt, "prompt": prompt,
                "status": "success" if _image._ENGINE_JUDGED.get("ok") else "partial",
                "reason": "; ".join(_image._ENGINE_JUDGED.get("problems") or []),
            }

        # Evaluate
        eval_data = evaluate_image(
            ctx, img_path, description, prompt, negative_prompt,
            final_steps, cfg, final_width, final_height,
        )

        verdict = str(eval_data.get("verdict", "refine")).lower()
        score = int(eval_data.get("score", 0) or 0)
        reason = str(eval_data.get("reason", "")).strip()

        logger.info("Evaluation: verdict=%s score=%d/10 reason=%s", verdict, score, reason[:300])

        if verdict == "success" or score >= 8:
            return {
                "path": img_path, "score": score, "attempts": attempt,
                "prompt": prompt, "status": "success",
                "reason": reason,
            }

        if attempt >= MAX_IMAGE_REFINEMENT_ATTEMPTS:
            break

        # Apply patches for next attempt
        prompt_patch = str(eval_data.get("prompt_patch", "")).strip()
        neg_patch = str(eval_data.get("negative_prompt_patch", "")).strip()

        if prompt_patch:
            prompt = f"{prompt}, {prompt_patch}"
        if neg_patch:
            negative_prompt = f"{negative_prompt}, {neg_patch}".strip(", ")

        new_steps = int(eval_data.get("steps", 0) or 0)
        new_cfg = float(eval_data.get("cfg", 0.0) or 0.0)
        new_width = int(eval_data.get("width", 0) or 0)
        new_height = int(eval_data.get("height", 0) or 0)

        if new_steps > 0:
            final_steps = max(4, min(new_steps, 30))  # cap runaway step counts
        if new_cfg > 0:
            cfg = max(0.5, min(new_cfg, 30.0))
        elif neg_patch and cfg <= 1.0:
            # At cfg=1.0 classifier-free guidance is off and the negative prompt is
            # mathematically inert — a defect-only refine would be a pure seed
            # reroll. Lift cfg to the top of the old model's healthy band so the patch
            # actually suppresses the named defect.
            cfg = 1.5
            logger.info("Negative patch with cfg=1.0 — raising cfg to 1.5 so it takes effect")
        # Clamp eval-suggested dimensions to the old model supported maximum.
        # Larger sizes risk GPU OOM even with 96 GB RAM (VRAM is the bottleneck).
        # Also reject pathological pairs whose total pixel count exceeds 4MP — the
        # eval LLM can hallucinate 4K suggestions it has no knowledge of VRAM for.
        _MAX_EVAL_DIM = 2720   # the old model max; above this it hits VRAM ceiling
        # 2720×1536 = 4.18MP (the old model native). Allow up to ~5MP so the native
        # resolution passes through; only reject genuinely oversized suggestions.
        _MAX_EVAL_PX  = 5_000_000
        if new_width > 0 or new_height > 0:
            cw = _image._snap_to_8(min(new_width, _MAX_EVAL_DIM)) if new_width > 0 else final_width
            ch = _image._snap_to_8(min(new_height, _MAX_EVAL_DIM)) if new_height > 0 else final_height
            if cw * ch > _MAX_EVAL_PX:
                scale = (_MAX_EVAL_PX / (cw * ch)) ** 0.5
                cw = _image._snap_to_8(int(cw * scale))
                ch = _image._snap_to_8(int(ch * scale))
                logger.info("Eval resolution %dx%d clamped to %dx%d (>4MP cap)",
                            new_width, new_height, cw, ch)
            if _image.session_size_pinned(ctx):
                # The vision judge suggests a resolution with no idea a 📐 Size
                # pin exists — proved live to flip a 9:16 pin to landscape AND to
                # exceed the normal 2048px rail via the wider eval-only cap above.
                # A pin is an explicit user choice; the judge's opinion on
                # steps/cfg/negative-prompt still applies, its opinion on
                # ORIENTATION does not. Re-orient to the pinned aspect and put it
                # back on the normal rails rather than the eval-only ones.
                pin_w, pin_h = _image.session_image_size(ctx)
                if pin_w == pin_h:
                    # A square pin has no long/short axis to match, so the
                    # landscape/portrait swap below is meaningless for it —
                    # `pin_w > pin_h` is always False, which made the swap
                    # fire on every landscape suggestion (forcing portrait)
                    # and never fire on a portrait one (staying portrait):
                    # a 1:1 pin came out portrait either way. Force square
                    # instead, keeping the judge's suggested pixel budget.
                    side = _image._snap_to_8(int((cw * ch) ** 0.5))
                    cw = ch = side
                elif (pin_w > pin_h) != (cw > ch):
                    cw, ch = ch, cw
                cw, ch = _image.normalize_resolution(cw, ch)
            final_width, final_height = cw, ch

        final_seed = random.randint(1, 999_999_999)

    return {
        "path": img_path, "score": score, "attempts": MAX_IMAGE_REFINEMENT_ATTEMPTS,
        "prompt": prompt, "status": "partial", "reason": reason,
    }

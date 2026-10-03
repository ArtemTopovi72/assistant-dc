import json
import logging
import os
import config as _cfg_env   # env_int/env_float: a bad .env value falls back, never crashes the import
import random
import re
import time
from typing import Optional, Tuple


from config import (
    WORKFLOW_FIRERED_EDIT_PATH, COMFY_URL, OUTPUT_DIR, MAX_IMAGE_REFINEMENT_ATTEMPTS,
    DEFAULT_WIDTH, DEFAULT_HEIGHT,
)
import config as _config
from prompts import VISION_EVAL_PROMPT
from llm import call_llm_simple, analyze_image_with_llm
from utils import safe_json_from_llm
# Pixel-level seam suppression, extracted to compositing.py. Re-imported here
# because several edit pipelines below drive it directly.
from compositing import (_feathered_alpha, _odd, _align_result_tile,
                         _robust_region_alpha, _seamless_clone_arr,
                         _blend_region)
# Geometry / size resolution, extracted to image_sizing.py. Re-exported here so
# callers and suites that reference image.<name> keep working.
from image_sizing import (
    _snap_to_8, session_image_size, session_size_pinned, parse_generation_params,
    detect_orientation_from_text, normalize_resolution, fix_image_params, _source_dims,
    _enforce_output_size,
)
# Region grounding / instruction language layer, extracted to image_grounding.py.
# Re-exported so image.<name> callers and monkeypatch seams keep working.
from image_grounding import (
    _INVENTORY_CACHE, _image_region_inventory, _region_phrase_variants, _ground_region_phrase,
    _extract_edit_target, _english_region, _english_instructions, _item_attributes,
    _is_removal_instruction, _is_facial, _is_removal_of, _region_present,
    _firered_instruction, _firered_style_instruction, _refine_edit_prompts, _strip_lead,
)
# Mask construction / geometry, extracted to image_masks.py. Re-exported so
# image.<name> callers and monkeypatch seams keep working.
from image_masks import (
    _dilate_mask_outward, _content_aware_alpha, _no_new_faces, _boundary_step_metric,
    _extend_mask_to_paired_changes, _extend_mask_to_removal_changes, _seam_blend_tile,
    crop_to_mask, _mask_white_frac, _significant_components, _sam3_mask_file,
    _region_mask_file,
)
# Mask quality gate, extracted to image_maskqa.py. MASK_QA_ENABLED is read
# through the module below (never by value) so a runtime flip is seen everywhere.
import image_maskqa
from image_maskqa import (
    _is_whole_face, _mask_face_fraction, _face_detector_mask, MASK_QA_ENABLED,
    _mask_quality_ok, _mask_cutout_png, _overlay_preview_png, _vlm_qa_verdict,
)
# Edit-engine registry, extracted to image_engines.py (imported as a default
# argument value by several pipelines, so it must stay import-cycle free).
from image_engines import DEFAULT_EDIT_ENGINE, _edit_engine_workflow, FIRERED_MAX_MP
# Hand/finger refiner, extracted to image_handfix.py. Re-exported so
# image.fix_hands and the stubbed seams keep resolving.
from image_handfix import _handfix_detect_graph, _handfix_region_fallback, fix_hands
# Identity preservation / face gating, extracted to image_identity.py.
from image_identity import (
    _instantid_facelock_graph, _preserve_face_after_upscale, preserve_identity_face,
)
# Reference-driven transfer (roles, reference assets, edit plan, placement),
# extracted to image_transfer.py. Re-exported so image.<name> keeps working.
from image_transfer import (
    ROLE_OBJECT, ROLE_CLOTHING, ROLE_FACE, ROLE_HAIR, ROLE_POSE, ROLE_STYLE, ROLE_IDENTITY,
    ROLE_SCENE, ROLE_LIGHTING, _ALL_ROLES, ReferenceImage, infer_reference_roles,
    extract_reference_asset, _reference_instruction, transfer_with_references,
    _ROLE_CHAIN_ORDER, build_edit_plan, _ROLE_TARGET_REGION,
    _placement_region_from_instruction, transfer_reference_contained,
    plan_and_execute_transfer, _split_person_and_scene, _composite_reference_head,
    generate_person_from_reference,
)
# Text-to-image generation + the refinement loop, extracted to image_generate.py.
# Re-exported so image.<name> callers and the monkeypatch seams keep working.
# image_generate reaches BACK into this module through a call-time proxy, so
# these names are bound here only once -- no by-value copy on the far side.
from image_generate import (
    _ideogram_draw, generate_image_with_comfy, evaluate_image,
    generate_image_with_refinement,
    )
# Background / whole-object operations, extracted to image_objects.py.
# Re-exported so image.<name> callers and the monkeypatch seams keep working.
# image_objects reaches BACK into this module through a call-time proxy, so
# these names are bound here only once -- no by-value copy on the far side.
from image_objects import (
    remove_background_with_comfy, remove_object_with_comfy,
    )
# Whole-frame transforms (relight),
# extracted to image_transforms.py. Re-exported so image.<name> callers and the
# monkeypatch seams keep working. image_transforms reaches BACK into this module
# through a call-time proxy, so these names are bound here only once.
from image_transforms import relight_image_with_comfy
# Contained / masked region editing, extracted to image_contained.py.
# Re-exported so image.<name> callers and the monkeypatch seams keep working.
# image_contained reaches BACK into this module through a call-time proxy, so
# these names are bound here only once -- no by-value copy on the far side.
from image_contained import (
    edit_region_contained_with_comfy, edit_region_contained_cropped, _contained_region_mask,
    edit_region_contained_via_firered,
)
# 16-category edit-intent router + dispatcher, extracted to image_router.py.
# Re-exported so image.<name> callers and the monkeypatch seams keep working.
# image_router reaches BACK into this module through a call-time proxy, so the
# names below are bound here only once -- there is no by-value copy on the far
# side to go split-brain.
from image_router import (
    classify_edit_intent, edit_plan, _NO_LAYOUT_NOTE, _LAYOUT_EDITABLE, edit_via_layout,
    convert_colour, route_edit_request, _contained_edit_validated,
    _orchestrate_multi_op,
)
from comfy_client import (
    _format_comfy_error, _upload_image_to_comfy, _progress_scope, _submit_and_poll,
    _poll_history, _submit_and_collect,
)

logger = logging.getLogger("assistant.image")

# Build/revision identifier so the runtime logs prove WHICH copy of this module is
# actually loaded in the live process (defeats the "stale interpreter" ambiguity).
# Derived from this file's mtime + size; logged once at import and on every edit call.


def _compute_build_id() -> str:
    try:
        st = os.stat(__file__)
        import hashlib
        h = hashlib.sha1()
        with open(__file__, "rb") as _f:
            h.update(_f.read())
        return f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(st.st_mtime))}+{h.hexdigest()[:8]}"
    except Exception:
        return "unknown"


IMAGE_BUILD_ID = _compute_build_id()
logger.info("image.py loaded — BUILD_ID=%s path=%s", IMAGE_BUILD_ID, __file__)


def log_edit_decision(*, request: str, classifier: str, tool: str, workflow: str,
                      returned_file: Optional[str], source: str = "",
                      extra: Optional[dict] = None) -> None:
    """Append one image-edit routing record to a PERSISTENT, app-independent log.

    The console logger is the only existing sink and it is not written to disk
    unless stdout happens to be redirected (it was not during the 2026-06-20
    session, which is why a dress->redraw routing could not be proven after the
    fact). This writes an always-on JSONL trail so every edit's
    REQUEST / CLASSIFIER_DECISION / SELECTED_TOOL / SELECTED_WORKFLOW /
    RETURNED_FILE is recoverable. Never raises."""
    rec = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "build_id": IMAGE_BUILD_ID,
        "REQUEST": (request or "")[:300],
        "CLASSIFIER_DECISION": classifier,
        "SELECTED_TOOL": tool,
        "SELECTED_WORKFLOW": workflow,
        "RETURNED_FILE": os.path.basename(returned_file) if returned_file else None,
        "RETURNED_DIMS": _source_dims(returned_file) if (returned_file and os.path.exists(returned_file)) else None,
        "SOURCE": os.path.basename(source) if source else None,
        "SOURCE_DIMS": _source_dims(source) if (source and os.path.exists(source)) else None,
    }
    if extra:
        rec.update(extra)
    try:
        logger.info("EDIT_DECISION %s", json.dumps(rec, ensure_ascii=False))
    except Exception:
        pass
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_DIR / "edit_routing.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("log_edit_decision: could not persist: %s", exc)


# Scratch artifacts that must NEVER be surfaced as a final result — cropped working
# tiles and QA previews. Matched by basename substring; the canonical marker is
# _INTERMEDIATE_, but a couple of older names are listed for belt-and-suspenders.
# Every one of these is a WORKING file written into OUTPUT_DIR beside the real
# renders. Deliverable results live there too and must never be listed here --
# handfix_, upscaled_faceguard_, contained-firered_ and contained-cropped_ are
# all named `final = ...` at their write sites, and adding one of those would
# make a finished picture undeliverable.
#   _srcregion_     image_masks.crop_to_mask -- the hand-drawn source region
#   _florence_proxy_ image_contained/_firered -- a proxy shown to the detector
_SCRATCH_MARKERS = ("_INTERMEDIATE_", "_firered_tile_", "_QA_cutout", "_QA_overlay",
                    "_crop_", "_cropmask_", "_srcregion_", "_florence_proxy_")


def is_intermediate_artifact(path) -> bool:
    """True if `path` is a scratch/working artifact (crop tile, QA preview) that must
    not be delivered as a final image. Used by every agent/UI delivery boundary."""
    if not path:
        return False
    name = os.path.basename(str(path))
    return any(m in name for m in _SCRATCH_MARKERS)


def is_uploaded_input(path) -> bool:
    """True if `path` is the picture the USER sent us this turn, not one we made.

    `state["image_path"]` is deliberately overloaded — tool handlers resolve their
    source from it — so it holds the uploaded photo as well as a produced one. The
    Telegram delivery guard read a non-empty value as "this turn owed the user a
    picture", so simply SENDING a photo to be described made the bot answer "I
    could not deliver the image" and log an error. Four times in one session.
    """
    if not path:
        return False
    return os.path.basename(str(path)).startswith("_working_input_")


def assert_deliverable(path: Optional[str], *, where: str, source_path: str = "") -> Optional[str]:
    """Final gate before an edited image is handed to the agent/UI.

    Loudly rejects intermediate scratch artifacts: any path containing
    ``_INTERMEDIATE_`` is a cropped working tile that must NEVER be surfaced as a
    result. Logs the full delivery record (build id, source/returned dims, path)
    so the UI-bound object is provable from the logs, then returns the path
    unchanged for valid finals or None for a rejected/invalid one.
    """
    if not path:
        return None
    src_dims = _source_dims(source_path) if source_path else None
    out_dims = _source_dims(path) if os.path.exists(path) else None
    if is_intermediate_artifact(path):
        logger.error("DELIVERY REJECTED [BUILD_ID=%s] at %s: refusing to return INTERMEDIATE "
                     "tile %s (source dims=%s, tile dims=%s). This is a cropped working "
                     "artifact, not a final result.", IMAGE_BUILD_ID, where, path, src_dims, out_dims)
        return None
    if not os.path.exists(path):
        logger.error("DELIVERY REJECTED [BUILD_ID=%s] at %s: returned path does not exist: %s",
                     IMAGE_BUILD_ID, where, path)
        return None
    # Dimension-shrink guard: a localized edit composites back over the FULL-RES
    # original, so the delivered frame MUST match the source canvas. A smaller
    # output means a crop/working tile leaked out as the final, or a workflow
    # downscale was never restored. Fail loud rather than hand the UI a shrunken
    # frame. (Allow a 1px rounding slack; reject real shrinkage.)
    if src_dims and out_dims:
        sw, sh = src_dims
        ow, oh = out_dims
        if ow < sw - 1 or oh < sh - 1:
            logger.error("DELIVERY REJECTED [BUILD_ID=%s] at %s: output %dx%d is SMALLER than "
                         "source %dx%d — a crop/downscaled tile leaked as final. path=%s",
                         IMAGE_BUILD_ID, where, ow, oh, sw, sh, path)
            return None
        if (ow, oh) != (sw, sh):
            logger.warning("DELIVERY [BUILD_ID=%s] at %s: output %dx%d != source %dx%d "
                           "(non-shrink size change) path=%s",
                           IMAGE_BUILD_ID, where, ow, oh, sw, sh, path)
    logger.info("DELIVERY OK [BUILD_ID=%s] at %s: returned=%s | source dims=%s | returned dims=%s",
                IMAGE_BUILD_ID, where, path, src_dims, out_dims)
    return path


# Records why the last inpaint_region_with_comfy call failed (cleared before each call).
# Values: "not_found" | "bad_mask" | "server_error" | "region_absent"
# Single-threaded app — module-level state is safe here; avoids changing the return type.
_INPAINT_FAILURE: dict = {"reason": "server_error"}

# Why the last FRESH generation produced nothing. "refused" means Ideogram
# declined the prompt (the user can rephrase); "engine_failed" means the render
# itself failed with the old model fallback deliberately disabled. Reporting both
# as "the server errored" sends the user to retry an identical prompt that will
# be declined again.
_GENERATE_FAILURE: dict = {"reason": "server_error"}

# Set by the Ideogram path when the picture came out of draw_agent's OWN judged
# loop (render -> read the lettering back -> critique -> rearrange the boxes ->
# render again). That loop is a complete judge-and-repair cycle with a rubric
# built for this engine, so the outer refinement loop in
# `generate_image_with_refinement` must NOT judge the same picture a second time
# with a different rubric and re-draw it from scratch on disagreement.
#
# Live 2026-08-05 ("it gets stuck, loads a second model, everything freezes"):
# one "draw me X" ran up to 3 outer attempts, each re-planning the layout and
# running draw_agent's 2 renders, each render followed by N lettering read-backs
# on TEXT_READ_MODEL and a critique on the chat model, then ANOTHER eval on the
# outer rubric — up to 6 Ideogram renders (two 9GB UNETs + a 10GB text encoder
# each) and ~18 vision calls swapping between two LM Studio models, on a 24GB
# card. It was not an infinite loop, it was a bounded loop with a multiplied
# body big enough to exhaust VRAM.
#   "judged": the engine ran its own judge/repair loop on this image
#   "score" : that loop's own 0-10 verdict, for honest reporting
#   "ok"    : whether it finished satisfied rather than out of rounds
_ENGINE_JUDGED: dict = {"judged": False, "score": 0, "ok": False, "problems": []}
# What the read-back loop still saw wrong after the LAST layout edit ([] = clean).
# The edit used to ship "ВЕ-Ч-ЕР" with three REJECTED verdicts and the tool
# result said only "Image edited" -- the bot answered "Готово" (live 2026-09-28).
_LAYOUT_EDIT_PROBLEMS: list = []
# What the pre-render layout check (draw_preview.preflight) found for the
# last Ideogram render: {"sketch", "checked", "ok", "problems", "notes", ...}.
_LAST_PREFLIGHT: dict = {}

# Records the outcome of the last removal's post-edit efficacy check. When the
# "is it still there?" verification confirms the target is STILL visible after the
# removal ran, this flags it so the tool layer can hand the agent an HONEST
# "removal did not fully succeed" signal instead of a plain success message the
# model then narrates a fabricated partial-success story around (the "I removed
# the left shoe" screenshot where both shoes were plainly still on).
_REMOVAL_VERIFY: dict = {"incomplete": False, "target": ""}


def layout_sidecar(image_path: str):
    """Path of the layout record kept beside an Ideogram render.

    PUBLIC because anything that copies, moves or renames a render has to carry
    this with it, or the picture arrives somewhere the layout cannot be found
    from -- and "передвинь кота на диван" silently falls through to the pixel
    pipelines, which repaint the frame instead of moving a box.
    """
    from pathlib import Path as _Path
    return _Path(str(image_path) + ".layout.json")


_layout_sidecar = layout_sidecar        # the old private name, still in use


def save_layout_for(image_path: Optional[str], prompt: str, layout: dict,
                    *, width: int, height: int, seed: Optional[int] = None) -> None:
    """Remember the layout an image was composed from.

    Without this the boxes exist only inside the generation call, so a later
    "fix the lettering" has nothing to drag and falls through to a whole-frame
    FireRed re-render — which does not correct letters, it destroys them. A
    sidecar file (not process state) because the request usually arrives in a
    different process: the user taps ✏️ in Telegram hours later.
    """
    if not image_path or not layout:
        return
    try:
        _layout_sidecar(image_path).write_text(json.dumps(
            {"prompt": prompt, "layout": layout, "width": width, "height": height,
             # The seed is what makes this feel like EDITING rather than rolling a
             # new picture. Re-rendering an adjusted layout on the same seed keeps
             # the framing, faces and lighting the user already accepted and moves
             # only what they asked to move; a fresh seed would hand back a
             # different picture that merely satisfies the same description.
             "seed": seed},
            ensure_ascii=False), encoding="utf-8")
    except Exception:
        # Losing the record costs a repair path, not the picture — never let it
        # take down a render that already succeeded.
        logger.exception("could not record the layout for %s", image_path)


def load_layout_for(image_path: Optional[str]) -> Optional[dict]:
    """The layout an image was composed from, or None if it wasn't ours."""
    if not image_path:
        return None
    p = _layout_sidecar(image_path)
    try:
        if not p.exists():
            return None
        rec = json.loads(p.read_text(encoding="utf-8"))
        return rec if isinstance(rec, dict) and rec.get("layout") else None
    except Exception:
        logger.exception("unreadable layout record for %s", image_path)
        return None


import contextlib as _contextlib
import contextvars as _contextvars

_EXTRA_LORA = _contextvars.ContextVar("firered_extra_lora", default=None)

# Removal adapter for FireRed; "" disables. Picked by bench/eraser_ab.py.
REMOVAL_LORA = os.getenv("FIRERED_REMOVAL_LORA", "")
REMOVAL_LORA_STRENGTH = _cfg_env.env_float("FIRERED_REMOVAL_LORA_STRENGTH", 1.0)


@_contextlib.contextmanager
def firered_extra_lora(name: Optional[str], strength: float = 1.0):
    """Every FireRed edit inside the block also loads LoRA `name`."""
    tok = _EXTRA_LORA.set((name, float(strength)) if name else None)
    try:
        yield
    finally:
        _EXTRA_LORA.reset(tok)


# Edit-on-edit drifts: FireRed was trained on single edits, and by the 4th
# chained pass colour noise and a drifting face pile up. From the
# (FIRERED_REBASE_AFTER+1)th edit in a row it edits the ORIGINAL with every
# instruction so far instead (FireRed findings 10-02). A chain longer than
# FIRERED_REBASE_MAX starts over from the current picture.
FIRERED_REBASE_AFTER = 3
FIRERED_REBASE_MAX = 6
_LINEAGE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "runtime", "firered_lineage.json")
if os.getenv("F5_TEST_RUN"):
    import tempfile as _tf
    _LINEAGE_FILE = os.path.join(_tf.mkdtemp(prefix="firered_lin_"), "lineage.json")


_ADV_SIZES = (512, 768, 1024, 1344, 1536, 2048)


def firered_ref_slot(workflow: dict, i: int) -> str:
    """Input name for reference image i (0-based) on the 187/188 encoders."""
    adv = str(workflow.get("187", {}).get("class_type", "")).startswith("TextEncodeQwenImageEditPlusAdvance")
    return ("vl_resize_image%d" if adv else "image%d") % (i + 2)


def firered_work_mp(workflow: dict, mp: float) -> None:
    """Working resolution: node 191 for the plain encoder, target_size (the
    padded square side, ~mp megapixels) for the Advance one."""
    if "191" in workflow:
        workflow["191"]["inputs"]["megapixels"] = mp
    side = min(_ADV_SIZES, key=lambda s: abs(s - (mp ** 0.5) * 1024))
    for nid in ("187", "188"):
        if "target_size" in workflow.get(nid, {}).get("inputs", {}):
            workflow[nid]["inputs"]["target_size"] = side


def _lineage() -> dict:
    try:
        with open(_LINEAGE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def edit_image_with_firered(ctx, image_path: str, instruction: str, **kw) -> Optional[str]:
    """FireRed edit; a long chain of edits is re-rendered from its original."""
    whole = not kw.get("save_prefix") and not kw.get("reference_paths")
    lin = _lineage() if whole else {}
    root, chain = image_path, []
    prev = lin.get(os.path.abspath(image_path or ""))
    if prev and os.path.exists(prev["root"]) and len(prev["chain"]) < FIRERED_REBASE_MAX:
        root, chain = prev["root"], list(prev["chain"])
    src, instr = image_path, instruction
    if len(chain) >= FIRERED_REBASE_AFTER and (instruction or "").strip():
        src = root
        instr = "; then ".join(chain + [instruction])
        logger.info("firered: edit %d in a row -> from the original with all %d instructions",
                    len(chain) + 1, len(chain) + 1)
    out = _edit_image_with_firered_once(ctx, src, instr, **kw)
    if out and whole:
        lin[os.path.abspath(out)] = {"root": os.path.abspath(root), "chain": chain + [instruction]}
        try:
            os.makedirs(os.path.dirname(_LINEAGE_FILE), exist_ok=True)
            with open(_LINEAGE_FILE, "w", encoding="utf-8") as f:
                json.dump(dict(list(lin.items())[-500:]), f, ensure_ascii=False)
        except Exception:
            logger.debug("firered lineage not saved", exc_info=True)
    return out


def _edit_image_with_firered_once(ctx, image_path: str, instruction: str,
                            *, seed: Optional[int] = None, timeout: int = 1900,
                            work_megapixels: Optional[float] = None,
                            save_prefix: Optional[str] = None,
                            reference_paths: Optional[list] = None,
                            engine: str = DEFAULT_EDIT_ENGINE) -> Optional[str]:
    """Instruction-edit an image with FireRed Image Edit 1.1 (Qwen-Image-Edit).

    ``engine`` is kept for call-site compatibility; FireRed is the only engine
    (the base Qwen-Image-Edit "qwenimage" alternative was removed).

    No mask/segmentation: the whole image is re-rendered to satisfy ``instruction``
    while preserving untouched content. Returns the saved image path or None.

    ``work_megapixels`` overrides node 191's working resolution. When None it is
    sized to the input (capped at ``FIRERED_MAX_MP``) so a large tile is no longer
    crushed to the template's 1.0 MP default — the cause of the cartoon softness.
    """
    if not image_path or not os.path.exists(image_path):
        logger.error("firered: source image not found: %s", image_path)
        return None
    if not (instruction or "").strip():
        logger.error("firered: empty instruction — refusing to submit")
        return None
    if seed is None or seed < 1:
        seed = random.randint(1, 999_999_999)

    uploaded = _upload_image_to_comfy(image_path, COMFY_URL)
    if not uploaded:
        logger.error("firered: image upload to ComfyUI failed")
        return None
    engine, wf_path = _edit_engine_workflow(engine)
    try:
        with open(wf_path, "r", encoding="utf-8") as f:
            workflow = json.load(f)
    except Exception as exc:
        logger.error("%s: failed to load workflow %s: %s", engine, wf_path, exc)
        return None

    # Patch by node id (we own the template): source image, the POSITIVE
    # instruction encoder (187 — never the empty negative 188), and the seed.
    try:
        workflow["143"]["inputs"]["image"] = uploaded
        workflow["187"]["inputs"]["prompt"] = instruction
        workflow["189"]["inputs"]["seed"] = seed
    except KeyError as exc:
        logger.error("firered: workflow template missing node %s — aborting", exc)
        return None

    # Optional reference images -> Qwen image2/image3 (multi-image conditioning).
    for i, rp in enumerate((reference_paths or [])[:2]):
        up_ref = _upload_image_to_comfy(rp, COMFY_URL) if rp and os.path.exists(rp) else None
        if not up_ref:
            logger.warning("firered: reference %d upload failed (%s) — skipped", i, rp)
            continue
        lid, sid, slot = f"ref_load_{i}", f"ref_scale_{i}", firered_ref_slot(workflow, i)
        workflow[lid] = {"class_type": "LoadImage", "_meta": {"title": f"Reference {i + 1}"},
                         "inputs": {"image": up_ref}}
        workflow[sid] = {"class_type": "ImageScaleToTotalPixels",
                         "_meta": {"title": f"Scale ref {i + 1}"},
                         "inputs": {"image": [lid, 0], "upscale_method": "lanczos",
                                    "megapixels": 1.0, "resolution_steps": 1}}
        workflow["187"]["inputs"][slot] = [sid, 0]
        workflow["188"]["inputs"][slot] = [sid, 0]

    # A task LoRA stacked on the Lightning one (removal: QIE-2511 remover
    # adapters, which FireRed -- a Qwen-Image-Edit derivative -- can load).
    _xl = _EXTRA_LORA.get()
    if _xl and "183" in workflow and "172" in workflow:
        workflow["183x"] = {"class_type": "LoraLoaderModelOnly",
                            "inputs": {"model": ["183", 0], "lora_name": _xl[0],
                                       "strength_model": _xl[1]}}
        workflow["172"]["inputs"]["model"] = ["183x", 0]
        logger.info("firered: extra LoRA %s @ %.2f", _xl[0], _xl[1])

    # When called as the tile-editor inside a contained pipeline, name the output
    # distinctly so a cropped INTERMEDIATE tile can never be mistaken for a final
    # full-frame result (both used to land as "firered_edit_*").
    if save_prefix:
        for _n in workflow.values():
            if _n.get("class_type") == "SaveImage":
                _n.setdefault("inputs", {})["filename_prefix"] = save_prefix

    # Working-resolution: size node 191 to the input (capped) instead of the flat
    # 1.0 MP template default, so a large tile keeps detail (no cartoon crush).
    dims = _source_dims(image_path)
    src_mp = (dims[0] * dims[1] / 1_000_000.0) if dims else None
    if work_megapixels is None:
        work_megapixels = min(src_mp, FIRERED_MAX_MP) if src_mp else 1.0
    # Never below 1.0 MP. TextEncodeQwenImageEditPlus rescales its reference to
    # ~1 MP on its own; a smaller working latent then disagrees with it and the
    # edit comes back ZOOMED ~1.4x (A/B 2026-09-24, 960x544 elephant: legs and
    # sand cropped at 0.52 MP, identical framing at 1.0 and at 2.0 MP). The
    # contained path's small tiles went through the same trap before being
    # composited back.
    work_megapixels = max(1.0, float(work_megapixels))
    try:
        prev_mp = workflow["191"]["inputs"].get("megapixels")
        firered_work_mp(workflow, work_megapixels)
    except KeyError:
        prev_mp = None
        logger.warning("firered: node 191 (ImageScaleToTotalPixels) missing — working MP not applied")

    # Geometry guard: rescale the decoded result back to the source canvas so the
    # saved dimensions match the input exactly (node 191 still downscaled internally).
    if dims:
        _enforce_output_size(workflow, dims[0], dims[1])

    logger.info("%s edit [BUILD_ID=%s]: input=%s dims=%s (%.2f MP) -> work_MP %.2f (was %s) seed=%d instr=%r",
                engine, IMAGE_BUILD_ID, os.path.basename(image_path), dims, (src_mp or 0.0),
                work_megapixels, prev_mp, seed, instruction[:80])
    return _submit_and_poll(ctx, workflow, timeout=timeout,
                            label=f"{engine} edit seed={seed}", exclusive=True)


# ---------------------------------------------------------------------------
# Intent router: classify a free-text edit request into one of 16 categories and
# dispatch it to the most appropriate specialized pipeline. The classifier is a
# deterministic, ordered keyword matcher (EN + RU) — no LLM round-trip needed, so
# it is fast and testable. `route_edit_request` is the single entry point the tool
# layer calls; it returns (category, output_path | None).
# ---------------------------------------------------------------------------


def inpaint_region_with_comfy(
        ctx,
        image_path: str,
        region: str,
        prompt: str,
        *,
        seed: Optional[int] = None,
        removal: bool = False,
        timeout: int = 1900,
        engine: str = DEFAULT_EDIT_ENGINE,
        on_progress=None,
        **_legacy,
) -> Optional[str]:
    """Edit one part of an image, described by ``region`` + ``prompt``.

    ``on_progress(step, total)`` — optional live sampling counter. This edit fans
    out to several ComfyUI jobs (segment, inpaint, maybe fall back); the callback is
    installed as an ambient hook so every one of them reports through it. Thin
    wrapper around ``_inpaint_region_impl``; the scope is always torn down.
    """
    with _progress_scope(on_progress):
        return _inpaint_region_impl(
            ctx, image_path, region, prompt, seed=seed, removal=removal,
            timeout=timeout, engine=engine, **_legacy)


def _inpaint_region_impl(
        ctx,
        image_path: str,
        region: str,
        prompt: str,
        *,
        seed: Optional[int] = None,
        removal: bool = False,
        timeout: int = 1900,
        engine: str = DEFAULT_EDIT_ENGINE,
        **_legacy,
) -> Optional[str]:
    """Edit one part of an image, described by ``region`` + ``prompt``.

    CONTAINED edit: because the caller already names the region, we segment exactly
    that region (Florence-2), inpaint only it, and composite the result back over the
    ORIGINAL full-resolution image. Every pixel outside the region — face, body,
    background, framing, resolution — is byte-for-byte the original, so identity is
    preserved by construction (measured cosine ~0.98 vs ~−0.03 for the old
    whole-frame FireRed re-render) and the output keeps the source dimensions.

    Falls back to whole-frame FireRed only if the contained edit fails (e.g. the
    region could not be segmented). ``**_legacy`` swallows now-unused mask knobs so
    older callers keep working. Returns the saved image path or None.
    """
    _INPAINT_FAILURE["reason"] = "server_error"
    if not image_path or not os.path.exists(image_path):
        logger.error("inpaint: source image not found: %s", image_path)
        return None
    _entry_dims = _source_dims(image_path)
    logger.info("inpaint_region ENTRY [BUILD_ID=%s]: source=%s dims=%s region=%r removal=%s",
                IMAGE_BUILD_ID, os.path.basename(image_path), _entry_dims, region, removal)
    region_en = _english_region(ctx, region) if (region or "").strip() else ""

    # Manual WHOLE-FRAME mode (sidebar 🖼 Whole frame toggle): the user explicitly
    # wants the entire image re-rendered by the instruction edit — no segmentation,
    # no mask, no composite, and NO identity guards (complete freedom, face
    # included; see the force_whole branch below). For stubborn edits where masking
    # keeps failing.
    force_whole = ((getattr(ctx, "image_edit_engine", "auto") or "auto")
                   .strip().lower() == "firered_whole")
    if force_whole:
        logger.info("inpaint: WHOLE-FRAME mode forced by user (no mask/composite)")

    # Ground the region against what's actually in the frame BEFORE redrawing: the
    # agent named the region blind, so map it onto a segmentable visible region (or
    # detect that it's absent). Defining the right region up front is far cheaper than
    # masking a phrase the segmenter can't find and falling back to a whole-frame
    # re-render. Skipped for removals/whole-frame edits (their own pipelines).
    whole = (not region_en) or region_en in ("whole", "image", "picture", "photo", "everything", "all")
    if force_whole:
        whole = True
    if not removal and not whole:
        grounded = _ground_region_phrase(ctx, image_path, region_en)
        if grounded == "absent":
            # The inventory is one vision call listing ~12 things; it left out a
            # plainly visible shoulder bag and the edit was refused (live
            # 2026-09-28). The segmenter gets the last word before "not there".
            try:
                _m = _sam3_mask_file(ctx, _upload_image_to_comfy(image_path, COMFY_URL),
                                     region_en, 0)
                _f = _mask_white_frac(_m) if _m else None
                if _f is not None and 0.001 <= _f <= 0.6:
                    logger.info("inpaint: inventory missed %r but SAM3 finds %.1f%% -- editing",
                                region_en, _f * 100)
                    grounded = region_en
            except Exception as exc:
                logger.info("inpaint: SAM3 double-check failed (%s)", exc)
        if grounded == "absent":
            logger.info("inpaint: region %r not present in image — skipping redraw", region_en)
            _INPAINT_FAILURE["reason"] = "region_absent"
            return None
        region_en = grounded
    if not removal and not whole:
        # PRIMARY: FireRed generates real content on the cropped region, composited
        # back over the original (identity preserved). FireRed is the only edit
        # engine: the masked-fill fallbacks that used to follow (the old model, then a
        # manually selected BrushNet/FLUX.1-Fill) were removed from the product.
        # The refiner rewrites the agent's terse tool args into a well-formed
        # imperative instruction + a self-contained fill prompt (templates on failure).
        instruction, _ = _refine_edit_prompts(ctx, region_en, prompt)
        out = edit_region_contained_via_firered(ctx, image_path, region_en, instruction,
                                                seed=seed, timeout=timeout, engine=engine)
        if out and os.path.exists(out):
            return out
        # A Stop between fallback renders must actually stop: each stage below is
        # its own multi-minute ComfyUI job, and without this check the chain runs
        # every remaining engine after the user cancelled.
        if ctx is not None and getattr(ctx, "is_cancelled", None) and ctx.is_cancelled():
            logger.info("inpaint: cancelled after FireRed-contained stage")
            return None
        # If the contained stages failed because the region could not be ISOLATED
        # (mask rejected / region absent / no effect), a whole-frame re-render is
        # never the right recovery — it repaints the entire person to change one
        # item ("remove her shoes" came back with a different dress and the shoes
        # still on). Surface the diagnosis to the agent instead; whole-frame stays
        # only for genuine render/server failures on an otherwise good region.
        if _INPAINT_FAILURE.get("reason") in ("bad_mask", "noisy_mask", "qa_rejected",
                                              "region_absent", "not_found"):
            logger.warning("inpaint: contained edit failed with %r for region=%r — "
                           "REFUSING whole-frame fallback (would re-render the person)",
                           _INPAINT_FAILURE.get("reason"), region_en)
            return None
        logger.warning("inpaint: contained edit failed for region=%r; falling back "
                       "to whole-frame FireRed", region_en)

    if not WORKFLOW_FIRERED_EDIT_PATH.exists():
        logger.error("inpaint: FireRed workflow missing: %s", WORKFLOW_FIRERED_EDIT_PATH)
        return None
    logger.warning("EDIT PATH=whole-frame-firered [BUILD_ID=%s] region=%r — WARNING: this "
                   "re-renders the entire frame (no composite-back); used only as last resort",
                   IMAGE_BUILD_ID, region_en)
    instruction, _ = _refine_edit_prompts(ctx, region_en or region, prompt, removal=removal)
    out = edit_image_with_firered(ctx, image_path, instruction, seed=seed, timeout=timeout,
                                  engine=engine)
    if not out:
        _INPAINT_FAILURE["reason"] = "server_error"
    elif force_whole:
        # User-selected whole-frame mode = COMPLETE freedom: no face paste-back, no
        # new-faces gate, no identity guard. The user chose this mode knowing the
        # entire frame (face included) is the model's to re-render.
        logger.info("whole-frame mode: delivering the raw re-render (no identity guards)")
    else:
        # This last-resort path re-renders the WHOLE frame, so FireRed repaints the
        # face too and the person comes back unrecognizable ("inpaint changed the
        # face"). Unless the edit deliberately targets the face, composite the
        # ORIGINAL face back so a clothing/object/background edit keeps identity.
        # A style change is the one whole-frame re-render where the face is SUPPOSED
        # to come out redrawn in the target style -- pasting the original photoreal
        # face back over it is exactly the "intermediate" look complained about live,
        # 2026-09-19 (see the matching guard in tools._handle_redraw_image).
        if (not (region and _is_facial(ctx, region))
                and classify_edit_intent(prompt) != "style_transfer"):
            out = preserve_identity_face(image_path, out, instructions=f"{region} {prompt}")
        # The re-render may have RE-COMPOSED the person (moved/resized); pasting the
        # original face back then yields two faces. Reject rather than deliver.
        if out and not _no_new_faces(image_path, out):
            _INPAINT_FAILURE["reason"] = "server_error"
            return None
        try:
            logger.info("EDIT PATH=whole-frame-firered RESULT [BUILD_ID=%s]: out %s | RETURNED PATH=%s",
                        IMAGE_BUILD_ID, _source_dims(out), out)
        except Exception:
            pass
    return out

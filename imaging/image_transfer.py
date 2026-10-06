"""Reference-driven transfer: roles, reference assets and the transfer plan.

Owns the role vocabulary (ROLE_*), the ReferenceImage carrier, role inference,
asset extraction, the role-ordered edit plan, instruction-derived placement and
the two execution paths (full-frame transfer and the contained/composite-back
one), plus reference-person generation.

Extracted from image.py. The pipelines it drives (edit_image_with_firered,
edit_region_contained_via_firered) and the telemetry it writes
(log_edit_decision, IMAGE_BUILD_ID) still live in image.py, so they are reached
through lazy module-level forwarders rather than by-value imports: that both
breaks the import cycle and keeps them patchable from either side.
"""
from config import scratch_path
import json
import logging
import os
import random
import re
import time
from typing import Optional

from config import COMFY_URL, OUTPUT_DIR, WORKFLOW_FIRERED_EDIT_PATH
from llm import call_llm_simple
from utils import safe_json_from_llm
from comfy_client import _submit_and_poll
from image_engines import FIRERED_MAX_MP
# NOTE: every other cross-module helper (_upload_image_to_comfy, _source_dims,
# _enforce_output_size, _english_instructions, _item_attributes, _strip_lead,
# _region_mask_file, _mask_quality_ok, preserve_identity_face) is deliberately
# NOT imported by value here.  Before the split these were plain globals inside
# image.py, and the suites patch them as `image.<name>`; a by-value import would
# make this module keep running the REAL helper while the test still printed
# PASS.  They are reached through `_image.` below, which resolves the CURRENT
# binding on image.py at call time.

logger = logging.getLogger("assistant.image")


class _ImageProxy:
    """Attribute proxy onto the still-monolithic image.py.

    Reading through it defers the import to call time (no cycle) and always
    resolves the CURRENT binding, so a runtime patch of image.<name> is honoured
    here even though the caller has moved out of image.py.
    """

    def __getattr__(self, name):
        import image
        return getattr(image, name)


_image = _ImageProxy()


def edit_image_with_firered(*a, **kw):
    return _image.edit_image_with_firered(*a, **kw)


def edit_region_contained_via_firered(*a, **kw):
    return _image.edit_region_contained_via_firered(*a, **kw)


def log_edit_decision(*a, **kw):
    return _image.log_edit_decision(*a, **kw)


def _split_person_and_scene(ctx, description: str):
    """Parse a 'draw <real person> <doing/wearing/scene>' description into
    (person_name, is_real_named_person, scene_clause). LLM-extracted with a
    deterministic fallback. Used by the reference-person image mode to decide
    whether to fetch a real photo and what outfit/scene to render."""
    sysp = ("You split an image request into a PERSON and a SCENE. Return STRICT JSON "
            "only: {\"person\":\"<the real, specific, named public figure being depicted, "
            "or empty>\", \"is_real_person\":true/false, \"scene\":\"<what they are wearing/"
            "doing/where, in English, or empty>\"}. is_real_person is true ONLY for a real, "
            "nameable individual (actor, musician, politician, athlete) — false for a generic "
            "subject (a cat, a woman, a knight) or a fictional character. No extra text.")
    out = call_llm_simple(ctx, sysp, f"Request: {description}", temperature=0.0,
                          max_tokens=200, prefill="<think></think>")
    data = safe_json_from_llm(out or "") or {}
    person = (data.get("person") or "").strip()
    scene = (data.get("scene") or "").strip()
    is_real = bool(data.get("is_real_person")) and bool(person)
    return person, is_real, scene


def _composite_reference_head(photo_path: str, render_path: str):
    """Paste the real person's HEAD (face + hair) from the reference ``photo`` back
    over the FireRed ``render`` through a feathered head-shaped mask.

    FireRed whole-frame re-render drifts the face/hair (a top-bun brunette came back
    as a bob with bangs — a different person). The outfit/scene we DO want from the
    render; the identity we want from the real photo. Compositing the real head back
    preserves the likeness BY CONSTRUCTION while keeping the rendered dress below the
    neck. Returns a new saved path, or the render path unchanged if no face is found.
    """
    try:
        from PIL import Image, ImageDraw, ImageFilter
        import identity_metrics as _idm
        photo = Image.open(photo_path).convert("RGB")
        render = Image.open(render_path).convert("RGB")
        if render.size != photo.size:
            render = render.resize(photo.size, Image.LANCZOS)
        fb = _idm.face_box(photo_path, pad=0.2)
        if not fb:
            logger.info("reference-head: no face in reference photo — keeping render as-is")
            return render_path
        x0, y0, x1, y1 = fb
        bw, bh = x1 - x0, y1 - y0
        W, H = photo.size
        # Generous head region: lots of room UP (hair / bun), sides for loose strands,
        # only a little DOWN (chin) so the rendered dress stays on the shoulders.
        nx0 = max(0, int(x0 - 0.5 * bw)); nx1 = min(W, int(x1 + 0.5 * bw))
        ny0 = max(0, int(y0 - 1.1 * bh)); ny1 = min(H, int(y1 + 0.25 * bh))
        mask = Image.new("L", photo.size, 0)
        ImageDraw.Draw(mask).ellipse([nx0, ny0, nx1, ny1], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(max(2, int(0.06 * bw))))
        out = render.copy()
        out.paste(photo, (0, 0), mask)
        dest = OUTPUT_DIR / f"refperson_{int(time.time() * 1000)}.png"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out.save(dest)
        logger.info("reference-head: composited real head over render -> %s", os.path.basename(str(dest)))
        return str(dest)
    except Exception as exc:
        logger.warning("reference-head: composite failed (%s) — keeping render as-is", exc)
        return render_path


def generate_person_from_reference(ctx, description: str, *,
                                   seed: Optional[int] = None, timeout: int = 1900):
    """REFERENCE-PERSON MODE (opt-in): draw a real named person by first fetching an
    actual photo of them from the web and using it as the identity reference, plus a
    read of their appearance, then rendering the requested outfit/scene onto that
    photo with FireRed (Qwen-Image-Edit). The likeness comes from the real photo, not
    a text2image guess.

    Returns (path, info) on success or (None, reason). The caller (generate_image
    handler) falls back to plain text2image when this returns None.
    """
    import search as _search
    person, is_real, scene = _image._split_person_and_scene(ctx, description)
    if not is_real:
        return None, "not_a_real_named_person"

    # 1) Fetch a real reference photo (portrait preferred).
    ts = int(time.time() * 1000)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ref_path = str(scratch_path(OUTPUT_DIR, f"_INTERMEDIATE_refphoto_{ts}.jpg"))
    photo = _search.fetch_reference_photo(ctx, f"{person} face portrait high resolution", ref_path)
    if not photo:
        logger.warning("reference-person: no usable photo for %r — falling back to text2image", person)
        return None, "no_reference_photo"

    # 2) Read their appearance (grounds the prompt; non-fatal if it fails).
    appearance = ""
    try:
        if getattr(ctx, "web_search_enabled", True):
            txt = _search.run_web_search(ctx, f"{person} physical appearance hair eyes face description")
            if txt and txt not in (_search.NO_RESULTS, _search.SEARCH_FAILED):
                appearance = txt[:600]
    except Exception as exc:
        logger.info("reference-person: appearance read skipped: %s", exc)

    # 3) Render the requested outfit/scene ONTO the real photo (identity from pixels).
    outfit = scene or _image._strip_lead(description, r"^.*?\b(" + re.escape(person) + r")\b")
    outfit = (outfit or "").strip() or "a portrait"
    instruction = (f"This is a photo of {person}. Render {person} {outfit}. "
                   f"Keep {person}'s exact facial identity, bone structure and likeness from "
                   f"this photo. Photorealistic, high detail, natural lighting.")
    logger.info("reference-person: person=%r outfit=%r photo=%s", person, outfit, os.path.basename(photo))
    out = edit_image_with_firered(ctx, photo, instruction, seed=seed, timeout=timeout)
    if not out:
        return None, "render_failed"
    # FireRed can drift the face/hair on a whole-frame re-render — paste the REAL head
    # from the reference photo back so the likeness is preserved by construction.
    final = _image._composite_reference_head(photo, out)
    return final, {"person": person, "outfit": outfit, "photo": photo,
                   "appearance_used": bool(appearance), "head_composited": final != out}


# =========================================================================== #
#  MULTI-IMAGE / REFERENCE-CONDITIONED EDITING (Phase 1)
#
#  Built on the proven Qwen-Image-Edit engine: TextEncodeQwenImageEditPlus
#  natively accepts image1 (target) + image2/image3 (references), verified via
#  /object_info. The design is model-AGNOSTIC: `transfer_with_references` builds
#  the reference-conditioned request; swapping FireRed for Flux Kontext / OmniGen2
#  / GPT-Image later means only changing the workflow builder, not the planner,
#  role model, or verification.
# =========================================================================== #

# Reference roles (what an image contributes to the edit). The agent / planner
# assigns these; pipelines stay generic (no hardcoded "hat" logic).
ROLE_TARGET = "target_image"          # the canvas being edited


ROLE_OBJECT = "object_source"         # transfer a thing (hat, product, prop)


ROLE_CLOTHING = "clothing_source"     # transfer a garment


ROLE_FACE = "face_reference"          # face to apply


ROLE_HAIR = "hairstyle_reference"     # hairstyle/shape to apply


ROLE_POSE = "pose_reference"          # body/pose guidance


ROLE_STYLE = "style_reference"        # adopt look/material/palette


ROLE_IDENTITY = "identity_reference"  # character/identity anchor


ROLE_SCENE = "scene_reference"        # background/scene to place into


ROLE_LIGHTING = "lighting_reference"  # lighting/mood to adopt


_ALL_ROLES = {ROLE_TARGET, ROLE_OBJECT, ROLE_CLOTHING, ROLE_FACE, ROLE_HAIR,
              ROLE_POSE, ROLE_STYLE, ROLE_IDENTITY, ROLE_SCENE, ROLE_LIGHTING}


# Roles whose contribution is a discrete THING that should be SAM3-extracted to a
# clean RGBA asset before generation (vs. roles that condition globally on pixels).
_EXTRACTABLE_ROLES = {ROLE_OBJECT, ROLE_CLOTHING, ROLE_HAIR}


# Keyword -> (role, default segmentation phrase) for automatic inference.
_ROLE_HINTS = [
    (ROLE_CLOTHING, "clothing", re.compile(
        r"\b(dress|shirt|jacket|coat|outfit|clothes|garment|suit|hoodie|sweater|"
        r"uniform|costume|pants|skirt|jeans|blouse)\b", re.I)),
    (ROLE_HAIR, "hair", re.compile(r"\b(hair|hairstyle|haircut|hairdo)\b", re.I)),
    (ROLE_OBJECT, "object", re.compile(
        r"\b(hat|cap|glasses|sunglasses|watch|bag|necklace|earrings?|product|bottle|"
        r"logo|prop|accessory|scarf|tie|crown|helmet|mask|furniture|chair|sofa|lamp|table)\b", re.I)),
    (ROLE_FACE, "face", re.compile(r"\b(face|facial features|likeness|look like (?:him|her|them))\b", re.I)),
    (ROLE_POSE, "person", re.compile(r"\b(pose|posture|stance|position|gesture)\b", re.I)),
    (ROLE_STYLE, "", re.compile(r"\b(style|material|texture|pattern|palette|colou?r scheme|aesthetic|vibe)\b", re.I)),
    (ROLE_SCENE, "", re.compile(r"\b(scene|background|environment|setting|room|place (?:it|them) in)\b", re.I)),
    (ROLE_LIGHTING, "", re.compile(r"\b(lighting|light|mood|shadows?|illumination)\b", re.I)),
    (ROLE_IDENTITY, "person", re.compile(r"\b(identity|character|same person)\b", re.I)),
]


class ReferenceImage:
    """One image input with a role plus the artifacts produced while processing it.

    Pixels (`path`, `extracted_asset_path`, `segmentation_mask`) are kept separate
    from structured understanding (`metadata`, `detected_objects`), per the spec.
    `extract_phrase` is the SAM3 prompt used to cut the asset out of `path`."""
    __slots__ = ("path", "role", "metadata", "extracted_asset_path",
                 "segmentation_mask", "detected_objects", "extract_phrase")

    def __init__(self, path, role=None, *, metadata=None,
                 extracted_asset_path=None, segmentation_mask=None,
                 detected_objects=None, extract_phrase=""):
        # role=None means "infer me" (filled by infer_reference_roles); an explicit
        # role is always honored and never overridden by inference.
        self.path = path
        self.role = role if role in _ALL_ROLES else None
        self.metadata = metadata or {}
        self.extracted_asset_path = extracted_asset_path
        self.segmentation_mask = segmentation_mask
        self.detected_objects = detected_objects or []
        self.extract_phrase = extract_phrase

    def effective_path(self):
        """The image the model should consume: extracted RGBA asset if present, else original."""
        return self.extracted_asset_path or self.path

    def is_extractable(self):
        return self.role in _EXTRACTABLE_ROLES

    @property
    def effective_role(self):
        """Role with the ROLE_OBJECT fallback applied (never None for execution)."""
        return self.role or ROLE_OBJECT

    def to_dict(self):
        return {"path": os.path.basename(str(self.path)), "role": self.role,
                "asset": os.path.basename(str(self.extracted_asset_path)) if self.extracted_asset_path else None,
                "detected": self.detected_objects, "metadata": self.metadata}

    def __repr__(self):
        a = "+asset" if self.extracted_asset_path else ""
        return f"ReferenceImage(role={self.role}, path={os.path.basename(str(self.path))}{a})"


def infer_reference_roles(instruction, references):
    """Assign a role (and default SAM3 extract phrase) to each reference from the
    instruction text. Heuristic auto-inference; the agent can override per-image.
    Only fills in roles left at the ROLE_OBJECT default — explicit roles are kept."""
    text = instruction or ""
    chosen, phrase = None, ""
    for role, ph, rx in _ROLE_HINTS:
        if rx.search(text):
            chosen, phrase = role, ph
            break
    for ref in references:
        if ref.role is None:                 # only fill UNASSIGNED roles
            ref.role = chosen or ROLE_OBJECT
        if not ref.extract_phrase and ref.is_extractable():
            ref.extract_phrase = phrase or {
                ROLE_CLOTHING: "clothing", ROLE_HAIR: "hair"}.get(ref.role, "object")
    return references


def extract_reference_asset(ctx, ref, *, grow=4, timeout=900):
    """PHASE 2 — detect+segment the salient thing in a reference image and write a
    clean RGBA cut-out, populating `ref.extracted_asset_path`,
    `ref.segmentation_mask`, and `ref.detected_objects`.

    Uses the existing SAM3-first mask path (`_region_mask_file`). Only meaningful
    for extractable roles (object/clothing/hair); a no-op (returns ref unchanged)
    otherwise. Never raises — on failure the ref keeps using its full image."""
    from PIL import Image
    if not ref or not ref.is_extractable():
        return ref
    if not os.path.exists(ref.path):
        logger.warning("extract: reference missing: %s", ref.path)
        return ref
    phrase = ref.extract_phrase or {ROLE_CLOTHING: "clothing", ROLE_HAIR: "hair"}.get(ref.role, "object")
    try:
        uploaded = _image._upload_image_to_comfy(ref.path, COMFY_URL)
        if not uploaded:
            return ref
        mask_path = _image._region_mask_file(ctx, uploaded, phrase, grow, seed=1, timeout=timeout)
        if not mask_path or not os.path.exists(mask_path):
            logger.warning("extract: no mask for %r in %s", phrase, os.path.basename(ref.path))
            return ref
        src = Image.open(ref.path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if mask.size != src.size:
            mask = mask.resize(src.size, Image.NEAREST)
        # Deterministic mask sanity: a scattered-speck / near-empty segmentation of the
        # reference object yields a garbage cut-out asset (the source of a nonsense
        # transfer). Reject it and keep the ref un-extracted so the caller falls back.
        _attrs = _image._item_attributes(ctx, phrase)
        _ok, _qr, _qs = _image._mask_quality_ok(mask, region_phrase=phrase,
                                         small_item=_attrs.get("small"),
                                         big_region=_attrs.get("large"))
        if not _ok:
            logger.warning("extract: rejecting %s mask for %r in %s %s",
                           _qr, phrase, os.path.basename(ref.path), _qs)
            return ref
        bbox = mask.point(lambda p: 255 if p > 24 else 0).getbbox()
        if not bbox:
            logger.warning("extract: empty mask for %r", phrase)
            return ref
        # RGBA cut-out, cropped to the asset bbox (clean reusable asset).
        rgba = src.convert("RGBA")
        rgba.putalpha(mask)
        asset = rgba.crop(bbox)
        ts = int(time.time() * 1000)
        out = scratch_path(OUTPUT_DIR, f"_asset_{ref.role}_{ts}.png")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        asset.save(out)
        ref.extracted_asset_path = str(out)
        ref.segmentation_mask = mask_path
        import numpy as _np
        cov = 100.0 * float((_np.asarray(mask) > 24).mean())
        ref.detected_objects = [{"phrase": phrase, "bbox": bbox,
                                 "coverage_pct": round(cov, 2)}]
        logger.info("extract: %s -> asset %s bbox=%s cov=%.1f%%",
                    ref.role, os.path.basename(str(out)), bbox, cov)
    except Exception as exc:
        logger.warning("extract: failed for %s: %s", os.path.basename(ref.path), exc)
    return ref


def _reference_instruction(role, n_refs):
    """Phrase the edit so the model knows image1 is the target and image2(+) are
    references. Generic across roles — no per-object hardcoding."""
    ref_clause = "image 2" if n_refs == 1 else "images 2 and 3"
    verb = {
        ROLE_CLOTHING: f"Dress the person in image 1 in the garment shown in {ref_clause}",
        ROLE_OBJECT: f"Place the object shown in {ref_clause} naturally onto the subject in image 1",
        ROLE_FACE: f"Apply the facial identity from {ref_clause} to the person in image 1",
        ROLE_HAIR: f"Give the person in image 1 the hairstyle shown in {ref_clause}",
        ROLE_POSE: f"Pose the subject in image 1 as shown in {ref_clause}",
        ROLE_STYLE: f"Restyle image 1 using the style/material from {ref_clause}",
        ROLE_SCENE: f"Place the subject of image 1 into the scene from {ref_clause}",
        ROLE_LIGHTING: f"Relight image 1 to match the lighting and mood of {ref_clause}",
        ROLE_IDENTITY: f"Render the character from {ref_clause} into image 1",
    }.get(role, f"Use {ref_clause} as a reference to edit image 1")
    keep = ("Match scale, orientation and perspective to image 1. Keep the person's "
            "identity, face, body and the rest of image 1 unchanged, matching the "
            "original style and lighting.")
    return f"{verb}. {keep}"


def transfer_with_references(ctx, target_path, references, instruction=None, *,
                             seed=None, timeout=1900, work_megapixels=None,
                             verify_identity=True):
    """Reference-conditioned edit: target (image1) + reference(s) (image2/image3).

    Reuses the FireRed/Qwen workflow, wiring each reference into the
    TextEncodeQwenImageEditPlus image2/image3 inputs (verified present). Returns
    the saved path or None. Identity is measured before/after when the target has
    a face and `verify_identity` is set; a large identity drop is logged loudly
    (the multi-image analogue of the single-image identity gate)."""
    if not target_path or not os.path.exists(target_path):
        logger.error("transfer: target not found: %s", target_path)
        return None
    refs = [r for r in (references or []) if r and os.path.exists(r.effective_path())]
    if not refs:
        logger.error("transfer: no usable reference images")
        return None
    if len(refs) > 2:
        # One Qwen pass exposes image2+image3 only. For >2 refs the PLANNER
        # (plan_and_execute_transfer) chains passes; a direct call here uses the
        # first 2 and warns rather than silently dropping the rest.
        logger.warning("transfer: %d refs in a single pass; Qwen exposes image2+image3 — "
                       "using first 2. Use plan_and_execute_transfer() to chain all of them.",
                       len(refs))
        refs = refs[:2]
    if seed is None or seed < 1:
        seed = random.randint(1, 999_999_999)

    # Role inference and the Qwen prompt need English (see plan_and_execute_transfer);
    # direct callers (GUI whole-frame mode, contained fallback) may pass Russian.
    if instruction and not instruction.isascii():
        instruction = _image._english_instructions(ctx, instruction) or instruction

    # Resolve any unassigned roles from the instruction before use.
    _image.infer_reference_roles(instruction or "", refs)

    # PHASE 2: extract a clean asset for any extractable ref that lacks one, so the
    # model is conditioned on the cut-out thing, not the whole reference photo.
    for r in refs:
        if r.is_extractable() and not r.extracted_asset_path:
            _image.extract_reference_asset(ctx, r)

    instr = (instruction or "").strip() or _image._reference_instruction(refs[0].effective_role, len(refs))

    up_target = _image._upload_image_to_comfy(target_path, COMFY_URL)
    up_refs = [_image._upload_image_to_comfy(r.effective_path(), COMFY_URL) for r in refs]
    if not up_target or not all(up_refs):
        logger.error("transfer: upload failed (target=%s refs=%s)", up_target, up_refs)
        return None

    try:
        with open(WORKFLOW_FIRERED_EDIT_PATH, "r", encoding="utf-8") as f:
            wf = json.load(f)
    except Exception as exc:
        logger.error("transfer: cannot load workflow: %s", exc)
        return None

    # target -> node 143 (image1 source), instruction -> 187, seed -> 189
    try:
        wf["143"]["inputs"]["image"] = up_target
        wf["187"]["inputs"]["prompt"] = instr
        wf["189"]["inputs"]["seed"] = seed
    except KeyError as exc:
        logger.error("transfer: workflow missing node %s", exc)
        return None

    # Add a LoadImage + ImageScaleToTotalPixels per reference and wire them into
    # image2 / image3 of BOTH the positive (187) and negative (188) encoders.
    for i, up in enumerate(up_refs):
        load_id = f"ref_load_{i}"
        scale_id = f"ref_scale_{i}"
        slot = _image.firered_ref_slot(wf, i)   # image2/3 or vl_resize_image2/3
        wf[load_id] = {"class_type": "LoadImage", "_meta": {"title": f"Reference {i + 1}"},
                       "inputs": {"image": up}}
        wf[scale_id] = {"class_type": "ImageScaleToTotalPixels",
                        "_meta": {"title": f"Scale ref {i + 1}"},
                        "inputs": {"image": [load_id, 0], "upscale_method": "lanczos",
                                   "megapixels": 1.0, "resolution_steps": 1}}
        wf["187"]["inputs"][slot] = [scale_id, 0]
        wf["188"]["inputs"][slot] = [scale_id, 0]

    # Working MP sized to the target (capped), like the single-image path.
    dims = _image._source_dims(target_path)
    src_mp = (dims[0] * dims[1] / 1_000_000.0) if dims else None
    if work_megapixels is None:
        work_megapixels = min(src_mp, FIRERED_MAX_MP) if src_mp else 1.0
    work_megapixels = max(0.5, float(work_megapixels))
    _image.firered_work_mp(wf, work_megapixels)
    for _n in wf.values():
        if _n.get("class_type") == "SaveImage":
            _n.setdefault("inputs", {})["filename_prefix"] = "transfer"
    if dims:
        _image._enforce_output_size(wf, dims[0], dims[1])

    logger.info("TRANSFER [BUILD_ID=%s]: target=%s %s | refs=%s | role=%s | work_MP=%.2f | instr=%r",
                _image.IMAGE_BUILD_ID, os.path.basename(target_path), dims,
                [repr(r) for r in refs], refs[0].effective_role, work_megapixels, instr[:80])

    cos_before = None
    if verify_identity:
        try:
            import identity_metrics as idm
            cos_before = idm.face_box(target_path)  # presence proxy
        except Exception:
            cos_before = None

    out = _submit_and_poll(ctx, wf, timeout=timeout, label=f"transfer seed={seed}",
                           exclusive=True)
    if not out:
        logger.warning("transfer: no output produced")
        return None

    # Verification: identity drift (people) + dimension integrity.
    metrics = {}
    if verify_identity and cos_before:
        try:
            import identity_metrics as idm
            cos = idm.identity_cosine(target_path, out)
            metrics["identity_cosine"] = cos
            if cos is not None and cos < 0.80:
                logger.warning("TRANSFER identity DRIFT: cosine=%.3f < 0.80 — the person in "
                               "image 1 changed too much (role=%s). Result kept but flagged.",
                               cos, refs[0].effective_role)
        except Exception as exc:
            logger.info("transfer: identity check skipped: %s", exc)

    # The Qwen multi-image transfer re-renders the WHOLE frame, so it repaints the
    # face — a clothing/object/scene transfer must NOT change the person's face.
    # Composite the original face back unless the transfer DELIBERATELY targets the
    # face/head (face/identity/hairstyle reference) or the instruction asks for it.
    if refs[0].effective_role not in (ROLE_FACE, ROLE_IDENTITY, ROLE_HAIR):
        out = _image.preserve_identity_face(target_path, out, instructions=instr)
        if "identity_cosine" in metrics:
            try:
                import identity_metrics as idm
                metrics["identity_cosine_after_guard"] = idm.identity_cosine(target_path, out)
            except Exception:
                pass

    try:
        log_edit_decision(request=f"transfer instr={instr!r}",
                          classifier=f"multi-image:{refs[0].effective_role}",
                          tool="transfer_with_references",
                          workflow="workflow_firered_edit.json + image2/image3 (Qwen multi-image)",
                          returned_file=out, source=target_path,
                          extra={"n_refs": len(refs), **metrics})
    except Exception:
        pass
    return out


# Order in which roles are applied when chaining (structure first so later, more
# detailed transfers land on a settled canvas). Lower = earlier.
_ROLE_CHAIN_ORDER = {
    ROLE_SCENE: 0, ROLE_POSE: 1, ROLE_CLOTHING: 2, ROLE_OBJECT: 3,
    ROLE_HAIR: 4, ROLE_FACE: 5, ROLE_IDENTITY: 5, ROLE_STYLE: 6, ROLE_LIGHTING: 7,
}


def build_edit_plan(target_path, references, instruction=""):
    """PHASE 4 — produce an explicit, backend-agnostic edit plan.

    Returns a dict: {target, steps:[{role, ref, instruction, extract}], n_passes}.
    Each step is one reference applied to the (evolving) target; >2 references
    become multiple chained passes so the architecture scales to N images and
    chained edits (Phase 10) without redesign."""
    refs = _image.infer_reference_roles(instruction, [r for r in references if r])
    ordered = sorted(refs, key=lambda r: _ROLE_CHAIN_ORDER.get(r.role, 4))
    steps = []
    # Group up to 2 same-spirit refs per pass (the Qwen image2/image3 budget).
    i = 0
    while i < len(ordered):
        group = ordered[i:i + 2]
        steps.append({
            "roles": [r.role for r in group],
            "refs": group,
            "extract": [r.is_extractable() for r in group],
            "instruction": _image._reference_instruction(group[0].role, len(group)),
        })
        i += 2
    plan = {"target": target_path, "n_passes": len(steps), "steps": steps,
            "user_instruction": instruction}
    logger.info("EDIT PLAN [BUILD_ID=%s]: target=%s passes=%d roles=%s",
                _image.IMAGE_BUILD_ID, os.path.basename(str(target_path)), len(steps),
                [s["roles"] for s in steps])
    return plan


# Role -> the region ON THE TARGET that the transfer should occupy. Roles here are
# localizable (mask + composite-back keeps the face/identity). Roles NOT listed
# (scene/style/pose/lighting/identity) are global and use the whole-frame pass.
_ROLE_TARGET_REGION = {
    ROLE_CLOTHING: "clothing",
    ROLE_HAIR: "hair",
    ROLE_FACE: "face",
    ROLE_OBJECT: "head and upper body",   # placement anchor for worn accessories
}


# Body parts / surfaces a placement instruction may name ("tattoo on his upper
# arm", "logo on the back of the shirt"). When the instruction explicitly names
# WHERE the object goes, that location — not the role's generic anchor — is the
# region the contained transfer must mask, or the edit lands on the wrong spot
# (the hardcoded ROLE_OBJECT="head and upper body" masks the chest, so a
# tattoo-on-arm request re-renders the chest and the arm is never touched).
_PLACEMENT_PARTS = (
    r"upper arm|forearm|shoulder|bicep|tricep|elbow|wrist|forearm|hand|"
    r"upper back|lower back|back|chest|ribs?|sternum|stomach|belly|abdomen|"
    r"thigh|calf|shin|ankle|foot|knee|neck|collarbone|hip|arm|leg"
)




_GARMENT_CACHE: dict = {}


def _garment_details(ctx, path: str) -> str:
    """One line on a garment photo: type, colour, collar, closures and every
    logo/patch/print with its colour and place. "" when the vision model is out."""
    if not path or not os.path.exists(path):
        return ""
    key = (path, os.path.getsize(path))
    if key in _GARMENT_CACHE:
        return _GARMENT_CACHE[key]
    try:
        from llm import analyze_image_with_llm
        txt = analyze_image_with_llm(
            ctx=ctx, image_path=path,
            user_text="Describe this garment in ONE line for a tailor who must copy it exactly.",
            system_prompt=("Describe only the garment: its type, colour, collar or neckline, "
                           "closures (zip, buttons) and their colour, cuffs and hem, and EVERY "
                           "logo, emblem, lettering, patch or print with its colour and exact "
                           "position (e.g. 'red embroidered monogram on the left chest'). One "
                           "line, English, no preamble, at most 60 words."),
            max_tokens=120) or ""
    except Exception as exc:
        logger.info("garment details: vision call failed (%s)", exc)
        txt = ""
    txt = re.sub(r"\s+", " ", txt).strip().strip('"').rstrip(".")[:400]
    logger.info("garment details for %s: %r", os.path.basename(path), txt)
    _GARMENT_CACHE[key] = txt
    return txt


def _placement_region_from_instruction(instruction):
    """Extract the body part / surface a placement instruction names, as a SAM3
    region phrase (e.g. "upper arm"), or None. Lets the contained transfer mask
    WHERE the user said to put the object instead of the role's generic anchor."""
    if not instruction:
        return None
    import intent
    parts = tuple(dict.fromkeys(_PLACEMENT_PARTS.split("|")))
    got = intent.ask_choice(
        "An object is put on a person in a picture: {text}\n\nWhich body part does the "
        "instruction place it on? Answer none when it names no body part.",
        instruction, parts + ("none",), "none")
    if got == "none":
        return None
    side = intent.ask_choice(
        "An object is put on a person in a picture: {text}\n\nDoes the instruction say "
        "which side of the " + got + " (upper, lower, left, right)?",
        instruction, ("upper", "lower", "left", "right", "none"), "none")
    return got if side == "none" or side in got else f"{side} {got}"


def transfer_reference_contained(ctx, target_path, ref, instruction=None, *,
                                 seed=None, timeout=1900, mask_override=None,
                                 protect_face=False):
    """Identity-preserving reference transfer: mask the target region for `ref.role`,
    re-render ONLY that region conditioned on the reference asset (Qwen image2),
    and composite back over the original — so the face/identity survive by
    construction (the whole-frame pass drifts them). Falls back to the whole-frame
    `transfer_with_references` for global roles with no target region.

    ``mask_override`` (a hand-drawn target mask from the Transfer tab) takes priority
    over the role's auto region: the contained path runs against exactly the drawn
    region, so manual masking works for any role — even those with no auto region."""
    # The placement regex and the composed FireRed instruction are English-only;
    # translate up front (ASCII no-op, so already-translated chained calls are free).
    if instruction and not instruction.isascii():
        instruction = _image._english_instructions(ctx, instruction) or instruction
    role = ref.effective_role
    role_region = _ROLE_TARGET_REGION.get(role)
    # An explicit placement in the instruction ("...onto his upper arm") is PREFERRED
    # over the role's generic anchor: otherwise ROLE_OBJECT always masks "head and
    # upper body" and a tattoo-on-arm edit re-renders the chest, never touching the
    # arm. But it is best-effort — the named part may not be visible/segmentable in
    # the target (e.g. "forearm" on a chest-up crop), so we keep a fallback chain
    # rather than letting an empty mask fail the whole transfer.
    placed = None if mask_override else _placement_region_from_instruction(instruction)
    # Region to mask: the instruction's explicit placement if the user named one,
    # otherwise the role's generic anchor. We do NOT chain placement->role_region:
    # if the user NAMED a location that is not visible/segmentable (e.g. "forearm"
    # on a chest-up crop), masking the unrelated role anchor instead would silently
    # re-render the wrong area and produce a no-op (object never appears). In that
    # case we go straight to the whole-frame pass, which renders the object on a
    # visible part. So the chain is: [named placement OR role anchor] -> whole-frame.
    if placed:
        region_candidates = [placed]
        logger.info("transfer(contained): instruction placement %r preferred over role region %r",
                    placed, role_region)
    else:
        region_candidates = [role_region] if role_region else []
    if not region_candidates and not mask_override:
        return _image.transfer_with_references(ctx, target_path, [ref], instruction,
                                        seed=seed, timeout=timeout)
    if ref.is_extractable() and not ref.extracted_asset_path:
        _image.extract_reference_asset(ctx, ref)
    ref_clause = "image 2"
    if role == ROLE_CLOTHING:
        # FireRed knows "image 2", not "the reference image", and copies only what
        # it is told: live 10-06 a navy quarter-zip with a red monogram and a flag
        # patch came out a plain navy jumper. Name the garment's details for it.
        details = _garment_details(ctx, ref.extracted_asset_path or ref.path)
        lead = (f"Replace the clothing with the garment shown in {ref_clause}"
                + (f" ({details})" if details else "") + ", fitted to the body, with all its "
                "details, logos and patches exactly as in " + ref_clause)
        extra = (instruction or "").strip()
        extra = re.sub(r"(?i)\b(?:the )?(?:reference|clothing reference) (?:image|photo|picture)", ref_clause, extra)
        extra = re.sub(r"(?i)\b(?:the )?target (?:image|photo|picture)", "image 1", extra)
        instruction = lead + (f". {extra}" if extra else "")
    instr = (instruction or "").strip() or {
        ROLE_CLOTHING: f"Replace the clothing with the garment shown in {ref_clause}, fitted to the body",
        ROLE_HAIR: f"Replace the hair with the hairstyle shown in {ref_clause}",
        ROLE_FACE: f"Apply the facial identity from {ref_clause}",
        ROLE_OBJECT: f"Add the object shown in {ref_clause}, placed and scaled naturally",
    }.get(role, f"Edit this region using {ref_clause} as reference")
    instr = (f"{instr}. Keep the rest of the image and the person's identity "
             "unchanged, matching the original style and lighting.")
    # Protect the face for non-facial manual masks (hat/clothing); never for the
    # face role itself (a deliberate face edit must reach the face).
    pf = bool(protect_face) and role != ROLE_FACE
    # Try each candidate region in turn; a None result (empty/degenerate mask for
    # that phrase) falls through to the next candidate, then to the whole-frame pass.
    out = None
    attempts = region_candidates if region_candidates else ["manual region"]
    for ridx, region in enumerate(attempts):
        out = edit_region_contained_via_firered(
            ctx, target_path, region, instr, seed=seed, timeout=timeout,
            reference_paths=[ref.effective_path()], mask_override=mask_override,
            protect_face=pf)
        if out:
            break
        if ridx + 1 < len(attempts):
            logger.warning("transfer(contained): region %r yielded no result — trying %r",
                           region, attempts[ridx + 1])
    if not out and not mask_override:
        logger.warning("transfer(contained): all regions %s failed — falling back to whole-frame pass",
                       attempts)
        out = _image.transfer_with_references(ctx, target_path, [ref], instruction,
                                       seed=seed, timeout=timeout)
    try:
        metrics = {}
        if out:
            import identity_metrics as idm
            c = idm.identity_cosine(target_path, out)
            if c is not None:
                metrics["identity_cosine"] = c
                if c < 0.80:
                    logger.warning("TRANSFER(contained) identity DRIFT cosine=%.3f<0.80 role=%s", c, role)
        log_edit_decision(request=f"contained-transfer role={role} instr={instr!r}",
                          classifier=f"multi-image-contained:{role}",
                          tool="transfer_reference_contained",
                          workflow="contained-firered + image2 (mask+composite-back)",
                          returned_file=out, source=target_path, extra=metrics)
    except Exception:
        pass
    return out


def plan_and_execute_transfer(ctx, target_path, references, instruction="", *,
                              seed=None, timeout=1900, mask_override=None,
                              protect_face=False):
    """PHASE 5 — plan, then execute the multi-image transfer stage by stage.

    Handles N references by chaining `transfer_with_references` passes (≤2 refs
    each), threading each pass's output as the next pass's target. Returns the
    final image path or None. This is the entry point the tool/router should call
    for multi-image requests; `transfer_with_references` remains the single-pass
    primitive."""
    # Everything downstream is English-only: role inference (_ROLE_HINTS), the
    # placement-region regex, and the FireRed/Qwen prompt itself. A raw Russian
    # instruction silently defaults every ref to ROLE_OBJECT and masks the generic
    # "head and upper body" anchor (the tattoo-on-chest bug). ASCII text passes
    # through untouched, so this is free for English instructions.
    if instruction and not instruction.isascii():
        instruction = _image._english_instructions(ctx, instruction) or instruction
    plan = _image.build_edit_plan(target_path, references, instruction)
    if not plan["steps"]:
        logger.error("plan_and_execute: no references to apply")
        return None
    # Flatten to one reference per pass: the CONTAINED transfer masks a single
    # target region and composites back (identity-preserving). One ref per region
    # keeps each transfer scoped and the face protected. Order follows the plan
    # (scene<pose<clothing<object<hair<face<style<lighting).
    ordered_refs = [r for step in plan["steps"] for r in step["refs"]]
    current = target_path
    last_ok = None
    for n, ref in enumerate(ordered_refs, 1):
        instr = instruction if (n == 1 and instruction.strip()) else None
        logger.info("TRANSFER pass %d/%d role=%s target=%s",
                    n, len(ordered_refs), ref.effective_role, os.path.basename(str(current)))
        # The manual mask describes a region of the ORIGINAL target; only apply it
        # on the first pass (later passes run against intermediate outputs).
        pass_mask = mask_override if (n == 1) else None
        out = _image.transfer_reference_contained(ctx, current, ref, instr,
                                           seed=seed, timeout=timeout,
                                           mask_override=pass_mask,
                                           protect_face=protect_face and n == 1)
        if not out:
            logger.warning("transfer pass %d (role=%s) produced nothing; keeping last good result",
                           n, ref.effective_role)
            break
        last_ok = out
        current = out
    return last_ok

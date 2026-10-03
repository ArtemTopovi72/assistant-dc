"""Identity preservation and face gating.

Pastes the original face back after an operation that would otherwise drift the
person (upscale, enhance, redraw), and the InstantID face-lock path for the
cases where a paste is not enough. _changes_face decides when the user
actually ASKED for the face to change, in which case preservation must stand
down.

Extracted from image.py.
"""
from config import scratch_path
import logging
import os
import random
import re
import time
from typing import Optional

from config import COMFY_URL, OUTPUT_DIR
from compositing import _blend_region
from comfy_client import _upload_image_to_comfy, _submit_and_poll
from image_masks import _seam_blend_tile

logger = logging.getLogger("assistant.image")


def _preserve_face_after_upscale(orig_path: str, upscaled_path: str, *,
                                 blend: float = 1.0) -> str:
    """Protect the ORIGINAL face from the super-resolver.

    RealESRGAN (and SR nets generally) hallucinate/sharpen facial features into a
    plasticky, often unrecognisable face — "the upscale ruined the face". The body,
    clothing and background benefit from ESRGAN; the FACE does not. So after the
    upscale we paste the real face back, taken from a plain LANCZOS enlargement of the
    original (which invents no detail and keeps identity exact), through a feathered
    ellipse, and Poisson tone-level it so there is no seam. Returns the corrected path,
    or ``upscaled_path`` unchanged when there is no detectable face / any failure
    (landscapes, objects, missing face tooling — all safely pass through).

    ``blend`` in [0,1]: 1.0 = fully faithful face (recommended); lower mixes a little
    of the ESRGAN face back in for extra crispness at some identity risk.
    """
    try:
        import identity_metrics as _idm
        from PIL import Image, ImageDraw, ImageFilter
    except Exception:
        return upscaled_path
    try:
        # EVERY face, not just the biggest one. A three-person photo used to have
        # one faithful face and two hallucinated ones ("crap performance of faces").
        boxes = _idm.face_boxes(orig_path, pad=0.30)  # pad to include forehead/hairline
        if not boxes:
            return upscaled_path                  # no face -> ESRGAN result is fine
        orig = Image.open(orig_path).convert("RGB")
        up = Image.open(upscaled_path).convert("RGB")
        ow, oh = orig.size
        tw, th = up.size
        if ow <= 0 or oh <= 0:
            return upscaled_path
        # Faithful enlargement of the original to the upscaled size (identity-exact).
        orig_up = orig.resize((tw, th), Image.LANCZOS)
        sx, sy = tw / float(ow), th / float(oh)
        face_src = orig_up if blend >= 0.999 else Image.blend(up, orig_up, blend)
        out = up.copy()
        done = []
        for (x0, y0, x1, y1) in boxes:
            X0, Y0 = int(x0 * sx), int(y0 * sy)
            X1, Y1 = int(x1 * sx), int(y1 * sy)
            bw, bh = max(1, X1 - X0), max(1, Y1 - Y0)
            # Feathered ellipse over this face (region mask + paste alpha).
            mask = Image.new("L", (tw, th), 0)
            ImageDraw.Draw(mask).ellipse([X0, Y0, X1, Y1], fill=255)
            feather = max(4, int(0.16 * min(bw, bh)))
            mask_soft = mask.filter(ImageFilter.GaussianBlur(feather))
            # Seam-suppress the faithful face into its surroundings on a padded box
            # crop (harmonize + Poisson low-freq + multi-band), then paste the merged
            # box back through the soft ellipse. Face into its own skin -> continuation.
            box = (max(0, X0 - bw // 4), max(0, Y0 - bh // 4),
                   min(tw, X1 + bw // 4), min(th, Y1 + bh // 4))
            # Blend against the RUNNING result so overlapping/adjacent faces (two
            # people shoulder to shoulder) do not undo each other's paste.
            crop_bg = out.crop(box)
            crop_face = face_src.crop(box)
            mcrop_hard = mask.crop(box).point(lambda p: 255 if p > 24 else 0)
            try:
                merged = _blend_region(crop_bg, crop_face, mcrop_hard)
            except Exception as exc:
                # One difficult face must not cost the others their protection.
                logger.warning("upscale: face blend failed for %s (%s) — pasting flat",
                               (X0, Y0, X1, Y1), exc)
                merged = crop_face
            out.paste(merged, (box[0], box[1]), mask_soft.crop(box))
            done.append((X0, Y0, X1, Y1))
        final = OUTPUT_DIR / f"upscaled_faceguard_{int(time.time() * 1000)}.png"
        out.save(final)
        logger.info("upscale: preserved %d original face(s) %s over the re-render",
                    len(done), done)
        return str(final)
    except Exception as exc:
        logger.warning("upscale: face preservation skipped (%s) — keeping ESRGAN result", exc)
        return upscaled_path


# Instructions that DELIBERATELY change the face/expression — when present, a
# whole-frame re-render (enhance/redraw) is ALLOWED to alter the face and we must
# NOT paste the original face back. Everything else (quality passes, clothing,
# background, hair, scene changes) keeps the original face identity-exact.
def _changes_face(instructions: str) -> bool:
    import intent
    return intent.ask_yes(
        "A picture of a person is edited with this instruction: {text}\n\nDoes it "
        "DELIBERATELY change the face itself (expression, smile, eyes, lips, makeup, "
        "age, gaze)? Clothes, hair, pose, background or quality are No.", instructions)


INSTANTID_CKPT = "sd_xl_base_1.0.safetensors"


INSTANTID_MODEL = "ip-adapter.bin"


INSTANTID_CONTROLNET = "instantid_controlnet.safetensors"


def _instantid_facelock_graph(target_crop_upload: str, ref_face_upload: str,
                              denoise: float, seed: int, weight: float = 0.8) -> dict:
    """SDXL + InstantID identity correction. ``image_kps`` (the TARGET crop) supplies
    the pose/expression/composition to keep; ``image`` (the REFERENCE face) supplies
    the identity to pull generation toward. A moderate img2img denoise on the
    target's OWN latent keeps whatever the whole-frame re-render already produced
    (new attribute — sunglasses, expression, lipstick) while InstantID corrects
    identity drift instead of overwriting it wholesale.
    """
    pos = "a detailed photorealistic human face, natural skin texture, sharp eyes, high detail"
    neg = "lowres, blurry, deformed, cartoon, watermark, text, different person"
    return {
        "CK": {"inputs": {"ckpt_name": INSTANTID_CKPT}, "class_type": "CheckpointLoaderSimple"},
        "L": {"inputs": {"image": target_crop_upload}, "class_type": "LoadImage"},
        "RF": {"inputs": {"image": ref_face_upload}, "class_type": "LoadImage"},
        "FA": {"inputs": {"provider": "CUDA"}, "class_type": "InstantIDFaceAnalysis"},
        "IM": {"inputs": {"instantid_file": INSTANTID_MODEL}, "class_type": "InstantIDModelLoader"},
        "CN": {"inputs": {"control_net_name": INSTANTID_CONTROLNET}, "class_type": "ControlNetLoader"},
        "P": {"inputs": {"text": pos, "clip": ["CK", 1]}, "class_type": "CLIPTextEncode"},
        "N": {"inputs": {"text": neg, "clip": ["CK", 1]}, "class_type": "CLIPTextEncode"},
        "AI": {"inputs": {"instantid": ["IM", 0], "insightface": ["FA", 0], "control_net": ["CN", 0],
                          "image": ["RF", 0], "image_kps": ["L", 0], "model": ["CK", 0],
                          "positive": ["P", 0], "negative": ["N", 0],
                          "weight": weight, "start_at": 0.0, "end_at": 1.0},
               "class_type": "ApplyInstantID"},
        "VE": {"inputs": {"pixels": ["L", 0], "vae": ["CK", 2]}, "class_type": "VAEEncode"},
        "KS": {"inputs": {"seed": seed, "steps": 25, "cfg": 4.5, "sampler_name": "dpmpp_2m",
                          "scheduler": "karras", "denoise": denoise, "model": ["AI", 0],
                          "positive": ["AI", 1], "negative": ["AI", 2],
                          "latent_image": ["VE", 0]}, "class_type": "KSampler"},
        "VD": {"inputs": {"samples": ["KS", 0], "vae": ["CK", 2]}, "class_type": "VAEDecode"},
        "9": {"inputs": {"filename_prefix": "_INTERMEDIATE_instantid-lock", "images": ["VD", 0]},
              "class_type": "SaveImage"},
    }


def _identity_lock_face_with_instantid(ctx, orig_path: str, rerendered_path: str, *,
                                       denoise: float = 0.5, seed: Optional[int] = None,
                                       timeout: int = 900) -> str:
    """Generative identity correction for a WHOLE-FRAME face-attribute edit.

    FireRed re-render the face per the instruction (sunglasses, lipstick,
    expression) with NO identity anchor at all — they are DiT architectures
    InstantID/IP-Adapter can't patch, so this can't be fixed inside their own
    generation. Instead: crop the face FireRed just rendered, use the
    ORIGINAL person's face as an InstantID reference, and regenerate just that
    crop at moderate denoise so the new content/pose survives while identity is
    corrected. Composited back through a feathered ellipse. Returns the corrected
    path, or ``rerendered_path`` unchanged on any failure/no-face (never worse
    than the uncorrected input).
    """
    try:
        import identity_metrics as _idm
        from PIL import Image, ImageDraw, ImageFilter
    except Exception:
        return rerendered_path
    try:
        ref_box = _idm.face_box(orig_path, pad=0.35)
        tgt_box = _idm.face_box(rerendered_path, pad=0.35)
        if not ref_box or not tgt_box:
            return rerendered_path
        orig = Image.open(orig_path).convert("RGB")
        tgt = Image.open(rerendered_path).convert("RGB")
        tw, th = tgt.size
        x0, y0, x1, y1 = tgt_box
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(tw, int(x1)), min(th, int(y1))
        bw, bh = x1 - x0, y1 - y0
        if bw < 16 or bh < 16:
            return rerendered_path
        cw, ch = max(8, (bw // 8) * 8), max(8, (bh // 8) * 8)  # SDXL likes multiples of 8
        x1, y1 = x0 + cw, y0 + ch
        tgt_crop = tgt.crop((x0, y0, x1, y1))
        ref_crop = orig.crop(ref_box)

        # SDXL breaks down into fractal/repeating-pattern artifacts well below its
        # comfortable working resolution (measured: a tight ~200x250 face crop fed
        # straight to VAEEncode/KSampler produced a woven/ghosted mess even though
        # the face itself was plausible) — upscale to a proper SDXL working size
        # before generating, then downscale the result back to (cw, ch) for the
        # composite so the paste position is unaffected.
        SDXL_WORK = 768
        work_scale = max(1.0, SDXL_WORK / max(cw, ch))
        ww, wh = max(8, int(cw * work_scale) // 8 * 8), max(8, int(ch * work_scale) // 8 * 8)
        tgt_work = tgt_crop.resize((ww, wh), Image.LANCZOS) if work_scale > 1.0 else tgt_crop
        ref_work = ref_crop.resize(
            (max(8, int(ref_crop.width * work_scale) // 8 * 8),
             max(8, int(ref_crop.height * work_scale) // 8 * 8)), Image.LANCZOS
        ) if work_scale > 1.0 else ref_crop

        ts = int(time.time() * 1000)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        tmp_tgt = scratch_path(OUTPUT_DIR, f"_instid_tgt_{ts}.png")
        tmp_ref = scratch_path(OUTPUT_DIR, f"_instid_ref_{ts}.png")
        tgt_work.save(tmp_tgt)
        ref_work.save(tmp_ref)

        u_tgt = _upload_image_to_comfy(str(tmp_tgt), COMFY_URL)
        u_ref = _upload_image_to_comfy(str(tmp_ref), COMFY_URL)
        if not u_tgt or not u_ref:
            return rerendered_path
        if seed is None or seed < 1:
            seed = random.randint(1, 999_999_999)
        wf = _instantid_facelock_graph(u_tgt, u_ref, denoise, seed)
        out = _submit_and_poll(ctx, wf, timeout=timeout, label=f"instantid-lock seed={seed}")
        if not out or not os.path.exists(out):
            return rerendered_path

        corrected = Image.open(out).convert("RGB")
        if corrected.size != (cw, ch):
            corrected = corrected.resize((cw, ch), Image.LANCZOS)

        # Poisson tone-level the corrected face into its surroundings before pasting
        # (same pattern as _preserve_face_after_upscale) — a plain alpha composite
        # left a visible seam where the corrected crop's background tone didn't quite
        # match the source photo's gradient, even under a soft ellipse feather.
        pad_x, pad_y = bw // 4, bh // 4
        bx0, by0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
        bx1, by1 = min(tw, x1 + pad_x), min(th, y1 + pad_y)
        box_bg = tgt.crop((bx0, by0, bx1, by1))
        box_res = box_bg.copy()
        box_res.paste(corrected, (x0 - bx0, y0 - by0))
        box_mask = Image.new("L", box_bg.size, 0)
        ImageDraw.Draw(box_mask).ellipse(
            [x0 - bx0, y0 - by0, x1 - bx0, y1 - by0], fill=255)
        toned = _seam_blend_tile(box_bg, box_res, box_mask)
        box_final = toned if toned is not None else box_res

        mask = Image.new("L", (tw, th), 0)
        ImageDraw.Draw(mask).ellipse([x0, y0, x1, y1], fill=255)
        feather = max(4, int(0.14 * min(cw, ch)))
        mask_soft = mask.filter(ImageFilter.GaussianBlur(feather))
        paste_layer = Image.new("RGB", (tw, th))
        paste_layer.paste(box_final, (bx0, by0))
        result = Image.composite(paste_layer, tgt, mask_soft)
        out_path = OUTPUT_DIR / f"identity-locked_{ts}.png"
        result.save(out_path)
        return str(out_path)
    except Exception as exc:
        logger.warning("instantid identity-lock skipped: %s", exc)
        return rerendered_path


def preserve_identity_face(orig_path: str, rerendered_path: str, *,
                           instructions: str = "") -> str:
    """Composite the ORIGINAL face back over a WHOLE-FRAME re-render (enhance /
    redraw / img2img), so a quality pass or unrelated edit never silently changes
    the person's face beyond recognition. When the user explicitly asked to change
    the face (attribute — sunglasses/lipstick/expression, not an identity swap),
    a literal pixel paste-back would undo the very edit that was requested, so this
    runs InstantID identity-lock instead (see ``_identity_lock_face_with_instantid``):
    keeps the new attribute/pose, corrects identity drift generatively rather than
    leaving it completely unprotected (which is what happened before — the face
    could drift to a different-looking person with no safeguard at all). No-op
    (returns ``rerendered_path``) when there is no detectable face or on any
    failure — never worse than doing nothing."""
    if not rerendered_path or not os.path.exists(rerendered_path):
        return rerendered_path
    if instructions and _changes_face(instructions):
        if not orig_path or not os.path.exists(orig_path):
            return rerendered_path
        try:
            locked = _identity_lock_face_with_instantid(None, orig_path, rerendered_path)
            if locked and locked != rerendered_path:
                logger.info("identity-face: InstantID identity-lock applied over face-attribute "
                            "edit (%r) -> %s", instructions[:60], os.path.basename(locked))
            else:
                logger.info("identity-face: InstantID identity-lock unavailable/no-op for "
                            "(%r) — keeping the re-render as-is", instructions[:60])
            return locked or rerendered_path
        except Exception as exc:
            logger.warning("identity-face: InstantID lock skipped (%s)", exc)
            return rerendered_path
    if not orig_path or not os.path.exists(orig_path):
        return rerendered_path
    try:
        guarded = _preserve_face_after_upscale(orig_path, rerendered_path)
        if guarded and guarded != rerendered_path:
            logger.info("identity-face: original face preserved over whole-frame re-render -> %s",
                        os.path.basename(guarded))
        return guarded or rerendered_path
    except Exception as exc:
        logger.warning("identity-face: preservation skipped (%s)", exc)
        return rerendered_path

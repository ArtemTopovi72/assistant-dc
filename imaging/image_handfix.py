"""Hand / finger refinement.

The three-tier fix_hands ladder: MeshGraphormer detect + depth-conditioned
refine, then the text-segmentation region fallback, then the contained-edit
path. Bad hands are invisible to the generic auto-detectors, which is why the
manual-mask route matters here.

Extracted from image.py.
"""
from config import scratch_path
import logging
import os
import random
import time
from typing import Optional

from config import COMFY_URL, OUTPUT_DIR
from compositing import _robust_region_alpha
from comfy_client import (_upload_image_to_comfy, _submit_and_poll,
                          _submit_and_collect)
from image_engines import DEFAULT_EDIT_ENGINE
from image_masks import _florence_mask_file, _sam3_mask_file

logger = logging.getLogger("assistant.image")


def edit_region_contained_via_firered(*args, **kwargs):
    """Forwarder to the contained-edit pipeline, which still lives in image.py.

    Resolved lazily through the module rather than bound by value at import,
    both to avoid the import cycle and so that a runtime patch of either
    image.edit_region_contained_via_firered or of this name is honoured.
    """
    import image
    return image.edit_region_contained_via_firered(*args, **kwargs)


# --- Hand / finger repair (MeshGraphormer Hand Refiner) ----------------------
# Standard ComfyUI hand fix: MeshGraphormer detects each hand, fits a 3D hand
# MESH, and renders a CORRECT-geometry depth map + an inpaint mask covering only
# the hand. A depth ControlNet then guides an SD1.5 masked inpaint so the hand is
# regenerated with anatomically plausible fingers, while every pixel outside the
# hand mask is composited back byte-for-byte. This is geometry-guided — unlike a
# plain re-render, which just paints another broken hand.
HANDFIX_CHECKPOINT = "realisticVisionV51_v51VAE-no-ema.safetensors"


HANDFIX_CONTROLNET = "control_v11f1p_sd15_depth.pth"


HANDFIX_POS = ("a realistic human hand, five fingers, correct anatomy, natural skin, "
               "detailed fingers, high quality photograph")


HANDFIX_NEG = ("extra fingers, missing fingers, fused fingers, too many fingers, "
               "mutated hands, malformed hands, deformed, bad anatomy, extra limbs, "
               "disfigured, blurry, low quality, cartoon")


def _handfix_detect_graph(uploaded_image: str, *, seed: int, mask_bbox_padding: int,
                          mask_expand: int, detect_thr: float = 0.6,
                          presence_thr: float = 0.6, resolution: int = 512) -> dict:
    """Pass 1: MeshGraphormer detect. Saves the corrected-hand DEPTH map and the
    hand MASK (as images) so Python can crop to the hand bbox for a high-res
    refine. Both outputs use the _INTERMEDIATE_ prefix so the delivery guard
    rejects them if they ever leak.

    Lower ``detect_thr``/``presence_thr`` catch DEFORMED hands (extra/fused
    fingers) that mediapipe's default 0.6 confidence misses — exactly the hands
    that need fixing."""
    return {
        "L":  {"inputs": {"image": uploaded_image}, "class_type": "LoadImage"},
        "MG": {"inputs": {"image": ["L", 0], "mask_bbox_padding": mask_bbox_padding,
                          "mask_type": "based_on_depth", "mask_expand": mask_expand,
                          "rand_seed": seed, "resolution": resolution,
                          "detect_thr": detect_thr, "presence_thr": presence_thr},
               "class_type": "MeshGraphormer-DepthMapPreprocessor"},
        "DS": {"inputs": {"filename_prefix": "_INTERMEDIATE_handdepth", "images": ["MG", 0]},
               "class_type": "SaveImage", "_meta": {"title": "depth"}},
        "MI": {"inputs": {"mask": ["MG", 1]}, "class_type": "MaskToImage"},
        "MS": {"inputs": {"filename_prefix": "_INTERMEDIATE_handmask", "images": ["MI", 0]},
               "class_type": "SaveImage", "_meta": {"title": "mask"}},
    }


def _handfix_refine_graph(crop_img: str, depth_img: str, mask_img: str, *, seed: int,
                          denoise: float, cn_strength: float, grow: int, steps: int,
                          cfg: float, pos: str, neg: str) -> dict:
    """Pass 2: depth-ControlNet guided SD1.5 masked inpaint of the HAND CROP only
    (high relative resolution -> sharp fingers). Returns just the regenerated crop
    tile (prefix _INTERMEDIATE_); Python composites it back over the original."""
    return {
        "L":  {"inputs": {"image": crop_img}, "class_type": "LoadImage"},
        "D":  {"inputs": {"image": depth_img}, "class_type": "LoadImage"},
        "LM": {"inputs": {"image": mask_img}, "class_type": "LoadImage"},
        "M":  {"inputs": {"image": ["LM", 0], "channel": "red"}, "class_type": "ImageToMask"},
        "G":  {"inputs": {"mask": ["M", 0], "expand": grow, "incremental_expandrate": 0.0,
                          "tapered_corners": True, "flip_input": False, "blur_radius": 4.0,
                          "lerp_alpha": 1.0, "decay_factor": 1.0},
               "class_type": "GrowMaskWithBlur"},
        "CK": {"inputs": {"ckpt_name": HANDFIX_CHECKPOINT},
               "class_type": "CheckpointLoaderSimple"},
        "P":  {"inputs": {"text": pos, "clip": ["CK", 1]}, "class_type": "CLIPTextEncode"},
        "N":  {"inputs": {"text": neg, "clip": ["CK", 1]}, "class_type": "CLIPTextEncode"},
        "CN": {"inputs": {"control_net_name": HANDFIX_CONTROLNET},
               "class_type": "ControlNetLoader"},
        "CA": {"inputs": {"positive": ["P", 0], "negative": ["N", 0],
                          "control_net": ["CN", 0], "image": ["D", 0],
                          "strength": cn_strength, "start_percent": 0.0,
                          "end_percent": 1.0}, "class_type": "ControlNetApplyAdvanced"},
        "DD": {"inputs": {"model": ["CK", 0]}, "class_type": "DifferentialDiffusion"},
        "VE": {"inputs": {"pixels": ["L", 0], "vae": ["CK", 2], "mask": ["G", 0],
                          "grow_mask_by": 0}, "class_type": "VAEEncodeForInpaint"},
        "KS": {"inputs": {"seed": seed, "steps": steps, "cfg": cfg,
                          "sampler_name": "dpmpp_2m", "scheduler": "karras",
                          "denoise": denoise, "model": ["DD", 0],
                          "positive": ["CA", 0], "negative": ["CA", 1],
                          "latent_image": ["VE", 0]}, "class_type": "KSampler"},
        "VD": {"inputs": {"samples": ["KS", 0], "vae": ["CK", 2]}, "class_type": "VAEDecode"},
        "S":  {"inputs": {"filename_prefix": "_INTERMEDIATE_handtile", "images": ["VD", 0]},
               "class_type": "SaveImage"},
    }


def _handfix_region_fallback(ctx, base_path: str, mg_mask, *, seed: int, timeout: int,
                             work_cap: int, pad_frac: float, denoise: float, grow: int,
                             steps: int, cfg: float, pos: str, neg: str,
                             engine: str = DEFAULT_EDIT_ENGINE) -> Optional[str]:
    """Fix hands MeshGraphormer missed: text-segment 'hand' (SAM3, Florence
    fallback), subtract the region MeshGraphormer already fixed, and SD1.5-inpaint
    the remainder with a five-finger prompt. Returns a new full-res path or None
    (nothing missed / segmentation found no hand)."""
    from PIL import Image
    import numpy as np
    try:
        orig = Image.open(base_path).convert("RGB")
        sw, sh = orig.size
    except Exception as exc:
        logger.error("handfix-fallback: cannot open base: %s", exc)
        return None
    up = _upload_image_to_comfy(base_path, COMFY_URL)
    if not up:
        return None
    hand_mask_path = (_sam3_mask_file(ctx, up, "hand", grow=8, timeout=timeout)
                      or _florence_mask_file(ctx, up, "hands", grow=8, seed=seed, timeout=timeout))
    if not hand_mask_path or not os.path.exists(hand_mask_path):
        logger.info("handfix-fallback: no hand segmented by SAM3/Florence")
        return None
    hand = np.asarray(Image.open(hand_mask_path).convert("L").resize((sw, sh), Image.NEAREST))
    miss = hand > 40
    if mg_mask is not None:
        try:
            import cv2
            mgm = mg_mask if mg_mask.size == (sw, sh) else mg_mask.resize((sw, sh), Image.NEAREST)
            mgm = (np.asarray(mgm.convert("L")) > 24).astype(np.uint8)
            mgm = cv2.dilate(mgm, np.ones((25, 25), np.uint8)) > 0
            miss = miss & (~mgm)
        except Exception:
            pass
    if int(miss.sum()) < int(sw * sh * 0.0008):
        logger.info("handfix-fallback: nothing missed (hand already covered by MeshGraphormer)")
        return None
    # Save the missed-hand mask full-res and let the FireRed contained engine
    # regenerate just that region (FireRed paints real content; an SD1.5
    # VAEEncodeForInpaint here only gray-fills the masked area).
    ts = int(time.time() * 1000)
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        miss_path = scratch_path(OUTPUT_DIR, f"_INTERMEDIATE_hfbmask_{ts}.png")
        Image.fromarray((miss * 255).astype("uint8"), "L").save(miss_path)
    except Exception as exc:
        logger.error("handfix-fallback: cannot write missed mask: %s", exc)
        return None
    hand_instr = ("Replace the hand in the marked area with one natural, anatomically "
                  "correct human hand that has exactly five separate fingers (no extra "
                  "or fused fingers), photorealistic, matching skin tone and lighting.")
    logger.info("handfix-fallback: FireRed repair of missed hand (missed px=%d)", int(miss.sum()))
    out = edit_region_contained_via_firered(
        ctx, base_path, "hand", hand_instr, mask_override=str(miss_path),
        protect_face=True, seed=seed, timeout=timeout, engine=engine)
    if out and os.path.exists(out):
        logger.info("handfix-fallback: saved %s", out)
        return out
    return None


def fix_hands(ctx, image_path: str, *, seed: Optional[int] = None,
              denoise: float = 0.7, cn_strength: float = 1.0, grow: int = 6,
              mask_bbox_padding: int = 30, mask_expand: int = 6, steps: int = 28,
              cfg: float = 6.0, work_cap: int = 768, pad_frac: float = 0.25,
              detect_thr: float = 0.3, presence_thr: float = 0.3,
              detect_resolution: int = 768, fallback: bool = True,
              mask_override: Optional[str] = None, timeout: int = 1200,
              pos: Optional[str] = None, neg: Optional[str] = None,
              engine: str = DEFAULT_EDIT_ENGINE) -> Optional[str]:
    """Repair malformed hands/fingers via the MeshGraphormer Hand Refiner.

    Two-pass, crop-based for SHARP results (a plain full-frame SD1.5 inpaint
    renders a small hand at a few latent pixels -> mush):
      1. MeshGraphormer detects hands and outputs a CORRECT-geometry depth map +
         a hand mask.
      2. Python crops the original/depth/mask to the hand bbox (padded) and
         upscales the crop; a depth-ControlNet guided SD1.5 inpaint regenerates
         the hand at high relative resolution.
      3. The refined tile is feathered back over the full-res original — every
         pixel outside the hand mask stays byte-for-byte identical.

    Returns the full-resolution repaired path, or None if ComfyUI failed or no
    hand was detected (empty mask -> caller surfaces "no hand found").
    """
    if not image_path or not os.path.exists(image_path):
        logger.error("fix_hands: source not found: %s", image_path)
        return None
    if seed is None or seed < 1:
        seed = random.randint(1, 999_999_999)
    denoise = max(0.2, min(1.0, float(denoise)))
    try:
        from PIL import Image
        import numpy as np
    except Exception as exc:
        logger.error("fix_hands: PIL/numpy unavailable: %s", exc)
        return None

    try:
        orig = Image.open(image_path).convert("RGB")
        sw, sh = orig.size
    except Exception as exc:
        logger.error("fix_hands: cannot open source: %s", exc)
        return None

    # --- MANUAL MASK path: user drew over the hand ---------------------------
    # The reliable route when every auto-detector (mediapipe/SAM3/Florence) fails
    # to recognize a severely deformed hand. Uses FireRed (instruction edit of the
    # masked region) — NOT an SD1.5 VAEEncodeForInpaint, which gray-fills the
    # masked area instead of painting a hand. FireRed generates real content and
    # the contained engine composites only the masked region back.
    if mask_override and os.path.exists(mask_override):
        hand_instr = ("Replace the hand in the marked area with one natural, "
                      "anatomically correct human hand that has exactly five separate "
                      "fingers (no extra or fused fingers), photorealistic, matching the "
                      "surrounding skin tone and lighting.")
        out = edit_region_contained_via_firered(
            ctx, image_path, "hand", hand_instr, mask_override=mask_override,
            protect_face=True, seed=seed, timeout=timeout, engine=engine)
        if out and os.path.exists(out):
            logger.info("fix_hands: manual-mask %s repair -> %s", engine, out)
            return out
        logger.warning("fix_hands: manual-mask %s repair failed", engine)
        return None

    # --- Pass 1: detect hands -> depth + mask -------------------------------
    uploaded = _upload_image_to_comfy(image_path, COMFY_URL)
    if not uploaded:
        logger.error("fix_hands: image upload failed")
        return None
    det = _submit_and_collect(
        ctx, _handfix_detect_graph(uploaded, seed=seed,
                                   mask_bbox_padding=mask_bbox_padding,
                                   mask_expand=mask_expand, detect_thr=detect_thr,
                                   presence_thr=presence_thr,
                                   resolution=detect_resolution),
        timeout=timeout, label="hand-detect")
    if not det:
        logger.warning("fix_hands: detect pass produced no output")
        return None
    # locate the depth and mask files by node id
    depth_path = mask_path = None
    for p in det.values():
        base = os.path.basename(p).lower()
        if "handdepth" in base:
            depth_path = p
        elif "handmask" in base:
            mask_path = p
    if not depth_path or not mask_path:
        logger.warning("fix_hands: missing depth/mask output (%s)", list(det.values()))
        return None

    try:
        depth = Image.open(depth_path).convert("RGB").resize((sw, sh), Image.LANCZOS)
        mask = Image.open(mask_path).convert("L").resize((sw, sh), Image.LANCZOS)
    except Exception as exc:
        logger.error("fix_hands: cannot open depth/mask: %s", exc)
        return None

    m = np.asarray(mask)
    ys, xs = np.nonzero(m > 24)
    ts = int(time.time() * 1000)
    base_path = image_path     # what the fallback composites onto (updated by MG)
    mg_done = False
    mg_mask_for_fallback = None

    if xs.size == 0:
        logger.info("fix_hands: MeshGraphormer detected no hand — trying region fallback"
                    if fallback else "fix_hands: no hand detected (empty mask)")
    else:
        mg_mask_for_fallback = mask
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        # pad the bbox so the inpaint has wrist/context, and snap inside the frame
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        pad = int(round(pad_frac * max(bw, bh)))
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(sw - 1, x1 + pad), min(sh - 1, y1 + pad)
        cw, ch = x1 - x0 + 1, y1 - y0 + 1
        if cw < 16 or ch < 16:
            logger.warning("fix_hands: degenerate hand crop %dx%d", cw, ch)
        else:
            # PIL crop is exclusive of right/lower, so use x1+1/y1+1 for exactly cw x ch.
            crop = orig.crop((x0, y0, x1 + 1, y1 + 1))
            dcrop = depth.crop((x0, y0, x1 + 1, y1 + 1))
            mcrop = mask.crop((x0, y0, x1 + 1, y1 + 1))
            scale = min(2.0, max(1.0, work_cap / max(cw, ch)))
            iw = max(64, (int(cw * scale) // 8) * 8)
            ih = max(64, (int(ch * scale) // 8) * 8)
            work_crop = crop.resize((iw, ih), Image.LANCZOS)
            work_depth = dcrop.resize((iw, ih), Image.LANCZOS)
            work_mask = mcrop.resize((iw, ih), Image.LANCZOS)
            try:
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                cpath = scratch_path(OUTPUT_DIR, f"_INTERMEDIATE_handcrop_{ts}.png")
                dpath = scratch_path(OUTPUT_DIR, f"_INTERMEDIATE_handdepthcrop_{ts}.png")
                mpath = scratch_path(OUTPUT_DIR, f"_INTERMEDIATE_handmaskcrop_{ts}.png")
                work_crop.save(cpath); work_depth.save(dpath); work_mask.convert("RGB").save(mpath)
                uc = _upload_image_to_comfy(str(cpath), COMFY_URL)
                ud = _upload_image_to_comfy(str(dpath), COMFY_URL)
                um = _upload_image_to_comfy(str(mpath), COMFY_URL)
                if uc and ud and um:
                    logger.info("Hand refiner: src %dx%d, hand bbox %dx%d -> work %dx%d, seed=%d "
                                "denoise=%.2f", sw, sh, cw, ch, iw, ih, seed, denoise)
                    refined = _submit_and_poll(
                        ctx, _handfix_refine_graph(uc, ud, um, seed=seed, denoise=denoise,
                                                   cn_strength=cn_strength, grow=grow, steps=steps,
                                                   cfg=cfg, pos=(pos or HANDFIX_POS),
                                                   neg=(neg or HANDFIX_NEG)),
                        timeout=timeout, label=f"hand-refine seed={seed}")
                    if refined and os.path.exists(refined):
                        res = Image.open(refined).convert("RGB").resize((cw, ch), Image.LANCZOS)
                        out = orig.copy()
                        out.paste(res, (x0, y0), _robust_region_alpha(mcrop, img_min=min(sw, sh)))
                        final = OUTPUT_DIR / f"handfix_{ts}.png"
                        out.save(final)
                        base_path = str(final)
                        mg_done = True
                        logger.info("fix_hands: MeshGraphormer refine -> %s", final)
                    else:
                        logger.warning("fix_hands: refine pass produced no output")
                else:
                    logger.error("fix_hands: crop upload failed")
            except Exception as exc:
                logger.error("fix_hands: MeshGraphormer refine failed: %s", exc)

    # --- Florence/SAM region fallback for hands MeshGraphormer could not detect ---
    if fallback:
        try:
            fb_out = _handfix_region_fallback(
                ctx, base_path, mg_mask_for_fallback, seed=seed, timeout=timeout,
                work_cap=work_cap, pad_frac=pad_frac, denoise=max(denoise, 0.8),
                grow=grow, steps=steps, cfg=cfg, pos=(pos or HANDFIX_POS),
                neg=(neg or HANDFIX_NEG), engine=engine)
            if fb_out and os.path.exists(fb_out):
                return fb_out
        except Exception as exc:
            logger.warning("fix_hands: region fallback failed: %s", exc)

    if mg_done:
        return base_path
    logger.info("fix_hands: no hand fixed (none detected, fallback found nothing)")
    return None

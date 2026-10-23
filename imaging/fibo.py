"""Fibo 2.0 Edit Turbo: the engine for a restyle.

Images go in as <image_1>, <image_2>, ... of the instruction; the distilled model samples 4 steps
at cfg 1. Any failure returns None and the caller falls back to Klein.
"""
import logging
import random
from typing import Optional

import config
from config import FIBO_STEPS, FIBO_UNET, KLEIN_CLIP, KLEIN_VAE

logger = logging.getLogger("assistant.image")


def edit_workflow(images: list, instruction: str, seed: int, width: int = 0, height: int = 0) -> dict:
    """images: uploaded ComfyUI names; width/height 0 = the first image's trained size."""
    enc = {"clip": ["2", 0], "prompt": instruction, "negative_prompt": "", "vae": ["3", 0],
           "width": int(width), "height": int(height), "batch_size": 1}
    wf = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": FIBO_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": KLEIN_CLIP, "type": "fibo2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": KLEIN_VAE}},
        "5": {"class_type": "TextEncodeFibo2", "inputs": enc},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["5", 0], "negative": ["5", 1], "latent_image": ["5", 2],
            "seed": int(seed), "steps": FIBO_STEPS, "cfg": 1.0, "sampler_name": "euler",
            "scheduler": "simple", "denoise": 1.0}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveImage", "inputs": {"filename_prefix": "fibo_edit", "images": ["7", 0]}},
    }
    for i, name in enumerate(images, 1):
        wf[f"load_{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        enc[f"images.image_{i}"] = [f"load_{i}", 0]
    return wf


def edit(ctx, images: list, instruction: str, *, seed: Optional[int] = None,
         timeout: int = 1900) -> Optional[str]:
    """Render `instruction` over local `images` (cited as <image_1>...); the result is sized like
    the first image. Returns the saved path or None."""
    from seams import dep
    import image as image_mod
    from image_sizing import _enforce_output_size
    if not config.FIBO_EDIT or not images or not (instruction or "").strip():
        return None
    try:
        up = dep(ctx, "_upload_image_to_comfy", image_mod._upload_image_to_comfy)
        names = [up(p, image_mod.COMFY_URL) for p in images]
        if not all(names):
            logger.warning("fibo: an image upload failed — falling back")
            return None
        dims = image_mod._source_dims(images[0])
        seed = seed if seed and seed > 0 else random.randint(1, 999_999_999)
        wf = edit_workflow(names, instruction, seed)
        if dims:
            _enforce_output_size(wf, *dims)
        logger.info("fibo edit: %d image(s) seed=%d instr=%r", len(images), seed, instruction[:160])
        return dep(ctx, "_submit_and_poll", image_mod._submit_and_poll)(
            ctx, wf, timeout=timeout, label=f"fibo edit seed={seed}", exclusive=True)
    except Exception:
        logger.exception("fibo: edit failed — falling back")
        return None

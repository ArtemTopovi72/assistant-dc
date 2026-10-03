"""Whole-frame image transforms.

Extracted from image.py. Takes an existing image and transforms the WHOLE
frame rather than a masked region: IC-Light relight. Upscale, outpaint, restore
and the face-detailer pass used to live here too; their engines (ESRGAN,
The old model, the detailer checkpoints) were removed from the product.

SEAM NOTE -- read before adding an import here.
The helpers used below (_upload_image_to_comfy, _submit_and_poll,
_valid_image_file, _source_dims, _enforce_output_size, _region_present)
were plain globals in image.py before the split and
the suites stub/read them as `image.<name>`. Importing them by value here would
bind the same name on BOTH sides of the split: patching `image.X` would move
only half the behaviour, the stub would silently die, and the REAL helper would
run while the suite still printed PASS. They are reached through `_image.`
below, which defers to call time and resolves the CURRENT binding on image.py.
"""
import logging
import os
import random
from typing import Optional

from config import COMFY_URL

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


# IC-Light relighting needs an SD1.5 base model + the IC-Light FC UNet (the latter
# auto-downloads from the open huchenlei/IC-Light-ldm repo on first use). The
# noVAE checkpoint is paired with a standalone SD1.5 VAE.
_ICLIGHT_SD15_CKPT = "realisticVisionV51_v51VAE-no-ema.safetensors"


_ICLIGHT_SD15_VAE = "vae-ft-mse-840000-ema-pruned.safetensors"


_ICLIGHT_DIRECTIONS = {
    "left": "Left Light", "right": "Right Light", "top": "Top Light",
    "above": "Top Light", "bottom": "Bottom Light", "below": "Bottom Light",
    "circle": "Circle Light", "ring": "Circle Light",
}


def relight_image_with_comfy(
        ctx, image_path: str, lighting_prompt: str,
        *, direction: str = "", seed: Optional[int] = None, timeout: int = 1900) -> Optional[str]:
    """Relight the subject with IC-Light (foreground-conditioned, SD1.5).

    IC-Light bakes the subject's latent into the UNet (c_concat) so the subject is
    preserved while its illumination is re-synthesised to match `lighting_prompt`
    (e.g. "warm golden hour sunlight from the left"). `direction` optionally seeds a
    light gradient (left/right/top/bottom/circle). Returns the saved path or None.
    """
    if not image_path or not os.path.exists(image_path):
        logger.error("relight: source image not found: %s", image_path)
        return None
    if not (lighting_prompt or "").strip():
        lighting_prompt = "natural balanced studio lighting, photorealistic"
    if seed is None or seed < 1:
        seed = random.randint(1, 999_999_999)
    # IC-Light FC needs a real light gradient: lighting="None" makes easy-use emit a
    # degenerate 1x1 lighting_image that crashes the downstream VAEEncode. Default to
    # a soft top-left key light when the request gives no explicit direction.
    lighting_dir = "Left Light"
    dl = (direction or lighting_prompt).lower()
    for key, val in _ICLIGHT_DIRECTIONS.items():
        if key in dl:
            lighting_dir = val
            break

    uploaded = _image._upload_image_to_comfy(image_path, COMFY_URL)
    if not uploaded:
        logger.error("relight: upload failed")
        return None
    neg = "lowres, bad anatomy, deformed, oversaturated, washed out, artifacts"
    wf = {
        "L": {"inputs": {"image": uploaded}, "class_type": "LoadImage"},
        "CK": {"inputs": {"ckpt_name": _ICLIGHT_SD15_CKPT}, "class_type": "CheckpointLoaderSimple"},
        "VA": {"inputs": {"vae_name": _ICLIGHT_SD15_VAE}, "class_type": "VAELoader"},
        # patch the SD1.5 model with the IC-Light FC weights + the subject latent
        "IC": {"inputs": {"mode": "Foreground", "model": ["CK", 0], "image": ["L", 0],
                          "vae": ["VA", 0], "lighting": lighting_dir,
                          "source": "Use Background Image", "remove_bg": True},
               "class_type": "easy icLightApply"},
        "POS": {"inputs": {"text": lighting_prompt.strip(), "clip": ["CK", 1]},
                "class_type": "CLIPTextEncode"},
        "NEG": {"inputs": {"text": neg, "clip": ["CK", 1]}, "class_type": "CLIPTextEncode"},
        # the lighting gradient image is the sampler's initial latent
        "EN": {"inputs": {"pixels": ["IC", 1], "vae": ["VA", 0]}, "class_type": "VAEEncode"},
        "K": {"inputs": {"seed": seed, "steps": 25, "cfg": 2.0, "sampler_name": "dpmpp_2m_sde",
                         "scheduler": "karras", "denoise": 1.0, "model": ["IC", 0],
                         "positive": ["POS", 0], "negative": ["NEG", 0],
                         "latent_image": ["EN", 0]}, "class_type": "KSampler"},
        "D": {"inputs": {"samples": ["K", 0], "vae": ["VA", 0]}, "class_type": "VAEDecode"},
        "9": {"inputs": {"filename_prefix": "relight", "images": ["D", 0]}, "class_type": "SaveImage"},
    }
    # Geometry guard: IC-Light runs at SD1.5 working resolution with no source
    # anchor; restore the original canvas so the subject isn't delivered tiny.
    dims = _image._source_dims(image_path)
    if dims:
        _image._enforce_output_size(wf, dims[0], dims[1])

    logger.info("Relight: prompt=%r dir=%s seed=%d", lighting_prompt[:60], lighting_dir, seed)
    return _image._submit_and_poll(ctx, wf, timeout=timeout, label=f"relight seed={seed}")

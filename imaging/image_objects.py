"""Background and whole-object operations.

Extracted from image.py. The pipelines that operate on the background plane or
on a whole object rather than on a painted region: background remove and
object remove (the FireRed contained path).
Background replace and object insert are FireRed instruction edits now
(image_router); their old-model graphs were removed with that model.

SEAM NOTE -- read before adding an import here.
_upload_image_to_comfy, _submit_and_poll, _enforce_output_size and
edit_region_contained_cropped were plain globals in image.py before the split
and the suites stub them as `image.<name>`. Importing them by value here would
bind the same name on BOTH sides of the split: patching `image.X` would move
only half the behaviour, the stub would silently die, and the REAL helper would
run while the suite still printed PASS. They are reached through `_image.`
below, which defers to call time and resolves the CURRENT binding on image.py.
"""
from config import scratch_path
import logging
import os
import time
import random
import re
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


def remove_background_with_comfy(
        ctx, image_path: str, *, mode: str = "transparent",
        timeout: int = 1900) -> Optional[str]:
    """Background REMOVAL — matte the subject and drop the background.

    `mode`: "transparent" (RGBA cut-out, default), "white", or "black". Uses the
    open BEN2 matte (no HF gating) with edge refinement. Returns the saved image
    path or None.
    """
    if not image_path or not os.path.exists(image_path):
        logger.error("bg-remove: source image not found: %s", image_path)
        return None
    add_bg = {"transparent": "none", "white": "white", "black": "black"}.get(mode, "none")

    uploaded = _image._upload_image_to_comfy(image_path, COMFY_URL)
    if not uploaded:
        logger.error("bg-remove: upload failed")
        return None
    wf = {
        "L": {"inputs": {"image": uploaded}, "class_type": "LoadImage"},
        "R": {"inputs": {"images": ["L", 0], "rem_mode": "BEN2", "image_output": "Hide",
                         "save_prefix": "rmbg", "torchscript_jit": False,
                         "add_background": add_bg, "refine_foreground": True},
              "class_type": "easy imageRemBg"},
        "9": {"inputs": {"filename_prefix": "bg-remove", "images": ["R", 0]},
              "class_type": "SaveImage"},
    }
    logger.info("BG-remove: mode=%s (add_background=%s)", mode, add_bg)
    return _image._submit_and_poll(ctx, wf, timeout=timeout, label=f"bg-remove {mode}")


def _with_contact_shadow(mask, frac: float = 0.5):
    """The mask smeared DOWN by `frac` of the object's height: the shadow and the
    contact patch under an object are not in its segmentation, and ObjectClear
    left a dark cat-shaped stain on the table (live 10-03).
    ponytail: assumes light from above; a long side shadow needs a shadow segmenter."""
    import numpy as np
    from PIL import Image
    a = np.asarray(mask.convert("L")) > 24
    box = mask.getbbox()
    if not box:
        return mask
    k = int((box[3] - box[1]) * frac)
    out = a.copy()
    for d in range(1, k + 1):
        out[d:] |= a[:-d]
    return Image.fromarray((out * 255).astype("uint8"))


def _remove_via_objectclear(ctx, image_path: str, target_phrase: str, *, seed: int,
                            timeout: int, max_cover: float = 1.0) -> Optional[str]:
    """Mask the target with the contained-edit masker, fill it with ObjectClear.
    None when there is no mask or ObjectClear failed."""
    try:
        import image_contained_firered as _cf
        import image_lettering_remove as _lr
        got = _cf._contained_region_mask(ctx, image_path, target_phrase, grow=18, seed=seed,
                                         timeout=timeout, protect_face=False,
                                         # the VLM cutout judge rejected a clean cat mask twice
                                         # (live 10-03) and the removal fell to FireRed, which
                                         # left a halo
                                         stage1_qa=False)
        if not got:
            return None
        _orig, mask, _box = got
        if max_cover < 1.0:
            import numpy as _np
            _cover = float((_np.asarray(mask.convert("L")) > 127).mean())
            if _cover > max_cover:
                # «white letters and icon background» selected half the picture and
                # ObjectClear rewrote the whole wall and the person (live 2026-10-08)
                logger.warning("remove-object: mask covers %.0f%% of the picture (limit %.0f%%) "
                               "-- not filled", _cover * 100, max_cover * 100)
                return None
        mask = _with_contact_shadow(mask)
        mask_path = str(scratch_path(_image.OUTPUT_DIR, f"_INTERMEDIATE_remove_mask_{int(time.time() * 1000)}.png"))
        mask.save(mask_path)
        out = _lr._fill_objectclear(ctx, image_path, mask_path, 18)
        # no smear gate here: a clean fill of a plain table measured 0.475 against
        # a busy window ring (gate 0.6), and with no fallback a rejection is only a
        # failure (live 10-03); the gate stays for lettering
        if out:
            logger.info("Remove-object: ObjectClear -> %s", os.path.basename(out))
        return out
    except Exception:
        logger.warning("remove-object: ObjectClear path raised", exc_info=True)
        return None


def remove_object_with_comfy(
        ctx, image_path: str, target_phrase: str,
        *, seed: Optional[int] = None, fill_hint: str = "", timeout: int = 1900,
        max_cover: float = 1.0) -> Optional[str]:
    """Localized object / person removal — FireRed contained edit.

    Removes only what `target_phrase` names, leaving the rest of the image as the
    ORIGINAL pixels. The PRIMARY path crops a padded tile around the target's mask,
    has FireRed (Qwen-Image-Edit) instruction-remove the object and reconstruct what
    is behind it, then feather-composites that tile back over the original — so only
    the masked region changes. This replaces the old model's masked latent-fill,
    which produced speckle/ghost artifacts when asked to paint a generic background
    over a face object (e.g. sunglasses): a turbo txt2img model cannot reconstruct
    eyes from a "seamless background" prompt.

    `fill_hint` optionally describes what should appear where the object was (e.g.
    "brick wall", "grass"); when empty FireRed reconstructs the natural surroundings.

    Returns the saved image path or None.
    """
    if not image_path or not os.path.exists(image_path):
        logger.error("remove-object: source image not found: %s", image_path)
        return None
    if not (target_phrase or "").strip():
        logger.error("remove-object: empty target phrase")
        return None
    if seed is None or seed < 1:
        seed = random.randint(1, 999_999_999)
    # the router hands «the cat»; the instructions add their own article («the the cat»)
    target_phrase = re.sub(r"^(?:the|a|an)\s+", "", target_phrase.strip(), flags=re.I) or target_phrase.strip()

    # --- PRIMARY: ObjectClear on the SAM3 mask (user's pick on the vase/cat A/B,
    # 2026-09-25: faster, 58-89 s vs 90-97 s, and preferred by eye). A fill hint
    # ("put a lamp there") needs an instruction model, so that stays on FireRed.
    if not (fill_hint or "").strip() and os.getenv("OBJECT_REMOVE_ENGINE", "objectclear") == "objectclear":
        out = _remove_via_objectclear(ctx, image_path, target_phrase, seed=seed, timeout=timeout,
                                      max_cover=max_cover)
        # no FireRed fallback: it re-lit the table and left a cat-shaped halo
        # (owner 10-03: «убери запасной FireRed»); FireRed only draws a fill_hint
        if not out:
            logger.warning("remove-object: ObjectClear failed, no fallback")
        return out

    # --- FireRed contained tile-edit (clean semantic reconstruction) ---
    if (fill_hint or "").strip():
        # "and its shadow and reflection": without it FireRed leaves a ghost of
        # the object on the floor or glass (FireRed findings, 2026-10-02)
        rm_instr = (f"Completely remove the {target_phrase} and its shadow and reflection. In its place show "
                    f"{fill_hint.strip()}, seamlessly matching the surrounding area. "
                    f"Photorealistic, natural, keep everything else identical.")
    else:
        rm_instr = (f"Completely remove the {target_phrase} and its shadow and reflection, and naturally reconstruct "
                    f"whatever is behind it so it looks like the {target_phrase} was "
                    f"never there. Photorealistic, seamless, keep everything else identical.")
    with _image.firered_extra_lora(_image.REMOVAL_LORA, _image.REMOVAL_LORA_STRENGTH):
        contained = _image.edit_region_contained_cropped(
            ctx, image_path, target_phrase, rm_instr,
            grow=18, seed=seed, timeout=timeout, engine="firered", protect_face=False,
            removal=True)
    if contained:
        logger.info("Remove-object: FireRed contained path -> %s", os.path.basename(contained))
        return contained
    # The old model masked-fill fallback that used to follow was removed with
    # that model; None means the removal failed.
    logger.warning("remove-object: FireRed contained path failed")
    return None

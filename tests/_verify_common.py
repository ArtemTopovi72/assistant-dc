"""Prompt and image-diff helper shared by the live-ComfyUI `_verify_*.py` scripts."""
import numpy as np
from PIL import Image

PROMPT = ("a red ceramic coffee mug on a wooden table, soft daylight from the left, "
          "shallow depth of field, photorealistic")


def meandiff(a, b):
    ia = np.asarray(Image.open(a).convert("RGB"), dtype=np.float64)
    ib = np.asarray(Image.open(b).convert("RGB"), dtype=np.float64)
    if ia.shape != ib.shape:
        return 999.0
    return float(np.abs(ia - ib).mean())

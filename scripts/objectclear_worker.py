"""ObjectClear fill for lettering removal. Runs in venv_eraser (diffusers 0.32).

    venv_eraser/Scripts/python scripts/objectclear_worker.py <image> <mask> <out> [grow]

Mask: white = remove. The model works at short side 512; only the masked area
(grown + feathered) is pasted back onto the original, every other pixel stays.
Chosen by the user's eye on the 26-item eraser A/B (2026-09-25).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.getenv("MODELS_EXT", os.path.join(ROOT, "models_ext"))
sys.path.insert(0, os.path.join(EXT, "objectclear_src"))

import torch
from PIL import Image, ImageFilter
from objectclear.pipelines import ObjectClearPipeline
from objectclear.utils import resize_by_short_side


def main(img_p, mask_p, out_p, grow=9):
    orig = Image.open(img_p).convert("RGB")
    mask = Image.open(mask_p).convert("L").resize(orig.size, Image.NEAREST)
    pipe = ObjectClearPipeline.from_pretrained_with_custom_modules(
        os.path.join(EXT, "objectclear"), torch_dtype=torch.float16,
        apply_attention_guided_fusion=True, variant="fp16").to("cuda")
    img = resize_by_short_side(orig, 512, resample=Image.BICUBIC)
    m = resize_by_short_side(mask, 512, resample=Image.NEAREST)
    r = pipe(prompt="remove the instance of object", image=img, mask_image=m,
             generator=torch.Generator(device="cuda").manual_seed(42),
             num_inference_steps=20, guidance_scale=2.5,
             height=img.size[1], width=img.size[0], return_attn_map=False)
    fill = r.images[0].resize(orig.size, Image.LANCZOS)
    a = mask.point(lambda v: 255 if v > 127 else 0) \
        .filter(ImageFilter.MaxFilter(grow | 1)).filter(ImageFilter.GaussianBlur(6))
    Image.composite(fill, orig, a).save(out_p)
    print("OK", out_p)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 9)

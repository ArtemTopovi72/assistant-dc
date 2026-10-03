"""FireRed colour-change A/B: does "pink" come back pink? (mega journey step 26)

    venv/Scripts/python bench/pink_ab.py

Source: runtime/pink_ab/grey_elephant.png (a real FireRed elephant with only its
own pixels desaturated). Each variant is one instruction wording, two seeds.
Scored by the mean hue/saturation inside the elephant's bounding box:
pink/hot pink sits at ~320-350 deg, the salmon/terracotta failure at ~10-25.
"""
import colorsys, os, sys, threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from PIL import Image

import image as _image
from models import Context

SRC = "runtime/pink_ab/grey_elephant.png"
KEEP = "Keep everything else in the image exactly the same, matching the original style and lighting."
VARIANTS = {
    "today": f"Change the elephant to: pink. {KEEP}",
    "vivid": ("Change the elephant to: pink. Recolor the elephant's whole skin a clear, saturated "
              f"bright pink (hot pink, like #FF69B4), not a tint of its old colour. {KEEP}"),
}
SEEDS = (11, 23)


def hue_in_box(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    box = (int(w * 230 / 700), int(h * 150 / 397), int(w * 400 / 700), int(h * 280 / 397))
    a = np.asarray(im.crop(box)).reshape(-1, 3) / 255.0
    hs = np.array([colorsys.rgb_to_hsv(*p)[:2] for p in a[::7]])
    ang = hs[:, 0] * 2 * np.pi
    mean = (np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) + 360) % 360
    return round(float(mean)), round(float(hs[:, 1].mean()), 2)


def main():
    ctx = Context(models=None, transcription_cache={}, cache_file=Path("runtime/pink_ab/c.json"),
                  asr_lock=threading.Lock(), tts_lock=threading.Lock(), model_name="", no_think=True)
    print("source", hue_in_box(SRC), flush=True)
    for name, instr in VARIANTS.items():
        for seed in SEEDS:
            out = _image.edit_image_with_firered(ctx, os.path.abspath(SRC), instr, seed=seed,
                                                 save_prefix=f"pink_{name}_{seed}")
            if not out:
                print(name, seed, "FAILED", flush=True)
                continue
            dst = f"runtime/pink_ab/{name}_{seed}.png"
            Image.open(out).save(dst)
            print(name, seed, "hue/sat", hue_in_box(dst), flush=True)


if __name__ == "__main__":
    main()

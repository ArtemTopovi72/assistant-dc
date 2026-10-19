"""LIVE: the real Klein edit with and without the LLM annotator + paste-back lock, same seed.

Needs ComfyUI :8000 and LM Studio with the vision model. Prints how many pixels OUTSIDE the edited
region changed (mean |diff| over the frame minus the annotator's boxes) and writes both results to
runtime/klein_annotate_live/.

    venv/Scripts/python -X utf8 bench/klein_annotate_live.py
"""
import os
import shutil
import sys
import threading
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "imaging"))
sys.path.insert(0, os.path.join(ROOT, "agent"))
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import config  # noqa: E402
import image  # noqa: E402
import klein_annotate  # noqa: E402
from models import Context  # noqa: E402

V2 = os.path.join(ROOT, "runtime", "klein_hard", "ig_full")
V1 = os.path.join(ROOT, "runtime", "klein_vs_ideogram", "ig_full")
OUT = os.path.join(ROOT, "runtime", "klein_annotate_live")
EDITS = [
    ("market_coat", V2 + "/s0_market.png", "change the boy's yellow raincoat to a blue one"),
    ("bookshop_cat", V2 + "/s1_bookshop.png", "remove the sleeping cat from the counter"),
    ("street_sign", V1 + "/s2_street3.png", "change the lettering on the brown cafe sign to 'БАР «ЛУНА»'"),
]


def outside_diff(src, res, boxes):
    a = np.asarray(Image.open(src).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(res).convert("RGB").resize((a.shape[1], a.shape[0])), dtype=np.int16)
    keep = np.ones(a.shape[:2], bool)
    h, w = a.shape[:2]
    for y0, x0, y1, x1 in boxes:
        keep[int(y0 / 1000 * h):int(y1 / 1000 * h) + 1, int(x0 / 1000 * w):int(x1 / 1000 * w) + 1] = False
    return float(np.abs(a - b)[keep].mean()) if keep.any() else 0.0


def main():
    os.makedirs(OUT, exist_ok=True)
    ctx = Context(models=None, transcription_cache={}, cache_file=Path("tests/_dr_cache.json"),
                  asr_lock=threading.Lock(), tts_lock=threading.Lock(),
                  model_name=config.MODEL_NAME, no_think=True, api_min_interval=0.2)
    for name, src, instr in EDITS:
        rec = klein_annotate.annotate_edit(ctx, src, instr)
        boxes = [list(map(float, b)) for b in (rec or {}).get("boxes") or []]
        print(f"== {name}: annotator {'OK' if rec else 'None'}; boxes={boxes}", flush=True)
        if rec:
            print("   prompt:", rec["prompt"][:300], flush=True)
        for label, ann, lock in (("plain", "0", "0"), ("annotated_locked", "1", "1")):
            os.environ["KLEIN_EDIT_ANNOTATE"], os.environ["KLEIN_EDIT_LOCK"] = ann, lock
            res = image._edit_image_with_firered_once(ctx, src, instr, seed=7)
            if not res:
                print(f"   {label}: no result", flush=True)
                continue
            dst = os.path.join(OUT, f"{name}_{label}.png")
            shutil.copy(res, dst)
            print(f"   {label}: outside-boxes mean|diff| = {outside_diff(src, dst, boxes):.2f}  -> {dst}", flush=True)


if __name__ == "__main__":
    main()

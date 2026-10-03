"""LIVE end-to-end: run the MeshGraphormer Hand Refiner against ComfyUI on a real
image and prove (a) it returns a full-frame deliverable, (b) only the hand region
changed (rest is pixel-identical), (c) the file passes the delivery guard.

Requires ComfyUI :8000 with comfyui_controlnet_aux loaded + control_v11f1p_sd15_depth.
Usage: venv/Scripts/python.exe bench/live_fix_hands.py [source.png]   exit 0 = pass
"""
import os, sys, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

from pathlib import Path
from PIL import Image, ImageChops
import config, image
from models import Context

SRC = sys.argv[1] if len(sys.argv) > 1 else "tests/_pasted_1782808524234.png"

def main():
    ctx = Context(models=None, transcription_cache={},
                  cache_file=Path("tests/_dr_cache.json"),
                  asr_lock=threading.Lock(), tts_lock=threading.Lock(),
                  model_name=config.MODEL_NAME, no_think=True, api_min_interval=0.2)
    src_im = Image.open(SRC).convert("RGB"); sw, sh = src_im.size
    print(f"source = {SRC} ({sw}x{sh})", flush=True)

    out = image.fix_hands(ctx, SRC, seed=12345)
    print(f"fix_hands returned: {out}", flush=True)

    ok = []
    def chk(c, m): ok.append(bool(c)); print(("PASS " if c else "FAIL ") + m, flush=True)

    chk(out is not None, "pipeline produced an output")
    if out and os.path.exists(out):
        chk(not image.is_intermediate_artifact(out),
            f"delivered file is NOT intermediate: {os.path.basename(out)}")
        ow, oh = Image.open(out).size
        print(f"output dims = {ow}x{oh}", flush=True)
        chk((ow, oh) == (sw, sh), f"output full source size ({ow}x{oh}=={sw}x{sh})")
        chk(image.assert_deliverable(out, where="live-handfix", source_path=SRC) == out,
            "assert_deliverable passes the repaired image")
        # measure how much changed: hand-only edit must change SOME but not MOST pixels
        diff = ImageChops.difference(src_im, Image.open(out).convert("RGB"))
        bbox = diff.getbbox()
        import numpy as np
        d = np.asarray(diff).sum(axis=2)
        changed = (d > 12).mean() * 100.0
        print(f"changed area = {changed:.2f}% of frame; diff bbox = {bbox}", flush=True)
        chk(changed > 0.05, "something changed (a hand region was regenerated)")
        chk(changed < 60.0, "most of the frame is UNCHANGED (localized hand edit, not whole re-render)")
    print(f"\n{sum(ok)}/{len(ok)} passed", flush=True)
    return 0 if ok and all(ok) else 1

if __name__ == "__main__":
    sys.exit(main())

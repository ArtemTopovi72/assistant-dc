"""LIVE end-to-end: run the real FireRed contained-edit pipeline against ComfyUI
and prove the delivered file is (a) full source size and (b) passes the delivery
guard, while the working tile it produces is rejected as intermediate.

Requires ComfyUI :8000 and LM Studio :1234 up. Exit 0 = verified.
"""
import os, sys, threading
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

from pathlib import Path
from PIL import Image
import config, image
from models import Context

SRC = sys.argv[1] if len(sys.argv) > 1 else \
    "tests/_asset_clothing_source_1781912058587.png"

def main():
    ctx = Context(models=None, transcription_cache={},
                  cache_file=Path("tests/_dr_cache.json"),
                  asr_lock=threading.Lock(), tts_lock=threading.Lock(),
                  model_name=config.MODEL_NAME, no_think=True, api_min_interval=0.2)
    sw, sh = Image.open(SRC).size
    print(f"source = {SRC}  ({sw}x{sh})", flush=True)

    out = image.edit_region_contained_via_firered(
        ctx, SRC, "the dress", "make the dress bright red", grow=12)
    print(f"contained-edit returned: {out}", flush=True)

    ok = []
    def chk(c, m): ok.append(bool(c)); print(("PASS " if c else "FAIL ") + m, flush=True)

    chk(out is not None, "pipeline produced an output")
    if out:
        chk(os.path.exists(out), "output file exists on disk")
        chk(not image.is_intermediate_artifact(out),
            f"delivered file is NOT an intermediate artifact: {os.path.basename(out)}")
        ow, oh = Image.open(out).size
        print(f"output dims = {ow}x{oh}", flush=True)
        chk((ow, oh) == (sw, sh), f"output is full source size ({ow}x{oh} == {sw}x{sh})")
        # the guard must pass this real final through unchanged
        chk(image.assert_deliverable(out, where="live-test", source_path=SRC) == out,
            "assert_deliverable PASSES the real full-size final")

    # the working tile that was written must be rejected by the guard
    tiles = sorted(Path(config.OUTPUT_DIR).glob("_INTERMEDIATE_firered_tile_*.png"))
    if tiles:
        t = str(tiles[-1])
        tw, th = Image.open(t).size
        print(f"latest tile = {os.path.basename(t)} ({tw}x{th})", flush=True)
        chk(image.is_intermediate_artifact(t), "working tile IS rejected as intermediate")
        chk(image.assert_deliverable(t, where="live-test", source_path=SRC) is None,
            "assert_deliverable REJECTS the working tile")
    else:
        print("WARN: no _INTERMEDIATE_firered_tile_ found (pipeline may have skipped crop path)")

    print(f"\n{sum(ok)}/{len(ok)} passed", flush=True)
    return 0 if ok and all(ok) else 1

if __name__ == "__main__":
    sys.exit(main())

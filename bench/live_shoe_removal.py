"""Live GPU end-to-end test of the removal change-region recovery fix.

Runs the REAL removal pipeline (remove_object_with_comfy -> edit_region_contained_cropped
-> FireRed tile edit -> drift-aligned, area-capped change-region composite) on the
actual "barefoot" source image and verifies the shoes are gone in the DELIVERED
full-res result, not just in the intermediate tile.

Requires ComfyUI up (COMFY_URL). LM Studio optional (_item_attributes fail-open).

    ./venv/Scripts/python.exe bench/live_shoe_removal.py
"""
import os, sys, time, threading
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import image as im

SRC = os.path.join("tests", "_working_input_1783364073455.jpg")


class _Ctx:
    """Minimal ctx the pipeline touches."""
    last_image_path = None
    last_image_prompt = ""
    image_edit_engine = "auto"
    reference_images = []
    api_min_interval = 0.0
    last_api_call_time = 0.0
    def __init__(self):
        self.api_lock = threading.Lock()
        self.memory_lock = threading.Lock()
    def set_stage(self, *a): print("  stage:", *a)
    def remember(self, *a, **k): pass
    def memory_text(self): return ""
    def is_cancelled(self): return False


def _region_still_present(path, target="shoes"):
    """Ask the same vision check the pipeline uses (fail-open None = unknown)."""
    try:
        return im._region_present(_Ctx(), path, target)
    except Exception as exc:
        print("  region-present check errored:", exc)
        return None


def main():
    assert os.path.exists(SRC), f"source missing: {SRC}"
    print("Source:", SRC, im._source_dims(SRC))
    ctx = _Ctx()
    ctx.last_image_path = SRC

    t0 = time.time()
    out = im.remove_object_with_comfy(ctx, SRC, "shoes", seed=1234, timeout=1900)
    dt = time.time() - t0
    print(f"\nremove_object_with_comfy -> {out}  ({dt:.1f}s)")
    assert out and os.path.exists(out), "pipeline returned no image"

    print("Result dims:", im._source_dims(out), " (source:", im._source_dims(SRC), ")")
    assert im._source_dims(out) == im._source_dims(SRC), "output resolution changed!"

    still = _region_still_present(out, "shoes")
    print("Shoes still present in DELIVERED result?", still)
    print("removal_verify flag:", im._REMOVAL_VERIFY)

    print("\nOpen and eyeball the result:", out)
    if still is True:
        print("RESULT: shoes STILL visible — recovery did not fully work.")
    elif still is False:
        print("RESULT: PASS — shoes removed in the delivered full-res image.")
    else:
        print("RESULT: vision check unavailable (LM Studio down) — inspect the file by eye.")


if __name__ == "__main__":
    main()

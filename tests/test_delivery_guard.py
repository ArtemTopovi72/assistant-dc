"""Regression test for the image-delivery guard: cropped working tiles and QA
previews must never pass as a final result. Guards the bug where the agent returned
the zoomed-in editing area (a 665x548 crop named _firered_tile_) instead of the
full composite. No GPU/ComfyUI needed.

Run:  venv/Scripts/python.exe tests/test_delivery_guard.py   (exit 0 = all pass)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import image

ok = []
def chk(c, m):
    ok.append(bool(c)); print(("PASS " if c else "FAIL ") + m)


def main():
    # scratch artifacts -> rejected
    for name in ("_INTERMEDIATE_firered_tile_123.png", "_firered_tile_123.png",
                 "_INTERMEDIATE_contained-crop_00001_.png", "_QA_cutout_9.png",
                 "_QA_overlay_9.png", "_crop_123.png", "_cropmask_123.png"):
        chk(image.is_intermediate_artifact(f"C:/out/{name}"),
            f"scratch rejected: {name}")
    # real results -> allowed
    for name in ("contained-firered_123.png", "contained-cropped_123.png",
                 "generated_000.png", "firered_edit_00001_.png"):
        chk(not image.is_intermediate_artifact(f"C:/out/{name}"),
            f"final allowed: {name}")
    chk(not image.is_intermediate_artifact(None), "None is not an artifact")
    chk(not image.is_intermediate_artifact(""), "empty is not an artifact")

    # assert_deliverable returns None for a tile, passes a real final through
    tile = "C:/out/_INTERMEDIATE_firered_tile_123.png"
    chk(image.assert_deliverable(tile, where="test") is None,
        "assert_deliverable rejects the firered tile")

    # the contained-firered function names its working tile with the protected prefix
    import inspect
    src = inspect.getsource(image.edit_region_contained_via_firered)
    chk("_INTERMEDIATE_firered_tile_" in src,
        "firered working tile uses the _INTERMEDIATE_ protected prefix")

    print(f"\n{sum(ok)}/{len(ok)} passed")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())

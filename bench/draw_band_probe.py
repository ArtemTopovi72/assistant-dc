"""Does a layout that tiles the frame into bands render as a COLLAGE?

Observed twice: a sane layout — sofa on the left, table in the middle, box on
the right, nothing overlapping, geometry_report clean — came back as a
three-panel collage instead of one photograph. The layouts that rendered as a
single scene had boxes whose ranges overlapped.

The hypothesis is that disjoint bands spanning the frame read as panel
boundaries. Measured against all fourteen layouts of one run BEFORE building
anything on it, and the correlation does not hold: self_omission has a wide
interior gap (x 0.45-0.60) and rendered as one clean living room, while
two_edits has no interior gap at all and came back as a checkerboard collage.
So the cut metric is not the predictor -- which leaves the other suspect below,
and is exactly why this file renders instead of reasoning. That is a claim about the renderer, so it is settled with the
renderer, not with an argument: one layout, two variants, same seed.

  A. the layout exactly as the pipeline produced it;
  B. the same boxes, nudged so neighbouring ones overlap by a few percent.

If B is one scene and A is panels, the cause is the bands and a deterministic
nudge belongs in the geometry repair. If both come back the same, the theory is
wrong and nothing should be built on it.

Run with the LLM unloaded (nothing here calls it):
    venv/Scripts/python.exe bench/draw_band_probe.py [--out DIR]
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ideogram as IG          # noqa: E402

SEED = 424242

# The move_cat layout, verbatim from the run that produced the collage.
BANDED = {
    "background": "уютная кухня, тёплый свет",
    "elements": [
        {"desc": "деревянный стол",         "x": 0.20, "y": 0.45, "w": 0.55, "h": 0.30},
        {"desc": "белая кружка с чаем",     "x": 0.28, "y": 0.15, "w": 0.23, "h": 0.27},
        {"desc": "серый кот сидит на полу", "x": 0.05, "y": 0.46, "w": 0.15, "h": 0.28},
        {"desc": "картонная коробка",       "x": 0.76, "y": 0.49, "w": 0.23, "h": 0.28},
        {"desc": "синий диван у стены",     "x": 0.00, "y": 0.42, "w": 0.24, "h": 0.46},
    ],
}


def overlapped(layout: dict, grow: float = 0.06) -> dict:
    """Widen every box a little so neighbouring ones share pixels.

    Deliberately the smallest possible change: same descriptions, same seed,
    same vertical placement, same order. Only the clean vertical cuts go away.
    """
    out = copy.deepcopy(layout)
    for el in out["elements"]:
        el["x"] = max(0.0, el["x"] - grow / 2)
        el["w"] = min(1.0 - el["x"], el["w"] + grow)
    return out


def _cuts(layout: dict) -> list:
    """Interior x positions where a vertical line crosses no element.

    INTERIOR: something has to sit on both sides of it. A gap at x=0.02 is the
    frame margin and means nothing; a gap at x=0.75 with the table on one side
    and the box on the other is a clean edge straight through the picture, and
    those are the ones that came back as panel boundaries.
    """
    els = layout.get("elements") or []
    out = []
    for i in range(1, 100):
        x = i / 100.0
        if any(el["x"] < x < el["x"] + el["w"] for el in els):
            continue
        left = any(el["x"] + el["w"] <= x for el in els)
        right = any(el["x"] >= x for el in els)
        if left and right:
            out.append(round(x, 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out = Path(args.out or ("runtime/band_probe_%d" % time.time()))
    out.mkdir(parents=True, exist_ok=True)

    # A third suspect, from a different render of the same family: a kitchen
    # that came back as an ADVERTISING BANNER -- logo, caption card, bullet
    # points -- around a mug occupying a quarter of the frame. MIN_AREA grows
    # any prop under 4% to 6%, and a mug at 23%x27% of 1024px is the size of a
    # hero product shot, which is a composition the model knows well. So: the
    # same scene with the mug left at the size a mug actually is.
    small = copy.deepcopy(BANDED)
    for el in small["elements"]:
        if "кружка" in el["desc"]:
            el.update({"x": 0.36, "y": 0.34, "w": 0.10, "h": 0.11})
    variants = [("banded", IG.normalize_layout(BANDED)),
                ("overlapped", IG.normalize_layout(overlapped(BANDED))),
                ("small_prop", IG.normalize_layout(small))]
    for name, lay in variants:
        print("%-12s clean vertical cuts at %s" % (name, _cuts(lay) or "none"))

    for name, lay in variants:
        cap = IG.layout_to_caption(lay)
        (out / (name + ".caption.json")).write_text(
            json.dumps(cap, ensure_ascii=False, indent=2), encoding="utf-8")
        t0 = time.perf_counter()
        path = IG.generate(None, "", width=1024, height=1024, seed=SEED, caption=cap)
        dt = time.perf_counter() - t0
        if path:
            dest = out / (name + Path(path).suffix)
            dest.write_bytes(Path(path).read_bytes())
            print("[draw] %-12s %6.1fs  %s" % (name, dt, dest))
        else:
            print("[draw] %-12s %6.1fs  nothing came back" % (name, dt))
    print("\nlook at both: same seed, same descriptions, only the cuts differ")
    print("  " + str(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

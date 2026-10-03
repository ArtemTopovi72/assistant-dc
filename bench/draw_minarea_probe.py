"""Where does a small element actually get swallowed?

MIN_AREA is 4% of the frame, with the repair aiming at 6.4%, on the premise
that "below ~4% an element gets swallowed or renders as a smudge". That premise
is load-bearing -- it is why a mug on a table is grown until it is the size of a
hero product shot, which is what several renders came back as: floating mugs,
advertising banners, inset panels.

And it is now in doubt: the same scene with the mug at 1.1% rendered as a clean
kitchen with a normal, clearly visible mug.

So measure it instead of arguing about it. One scene, one seed, the mug at a
sweep of sizes. Look at the results and read off where it stops being drawn.

    venv/Scripts/python.exe bench/draw_minarea_probe.py [--out DIR]
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ideogram as IG                                    # noqa: E402
from draw_band_probe import BANDED, SEED                 # noqa: E402

# Areas to try, as a fraction of the frame. 6.2% is what the repair produces
# today; 1.1% is the one that rendered well; the rest bracket the boundary.
SIZES = (0.006, 0.011, 0.02, 0.03, 0.04)


def with_mug(area: float) -> dict:
    """The kitchen, with the mug resized to `area` and left standing ON the
    table -- its bottom edge stays where it was, which is the same rule the
    geometry repair uses."""
    out = copy.deepcopy(BANDED)
    for el in out["elements"]:
        if "кружка" in el["desc"]:
            side = area ** 0.5
            bottom, cx = 0.45, el["x"] + el["w"] / 2      # rests on the table
            el["w"] = el["h"] = round(side, 3)
            el["x"] = round(cx - side / 2, 3)
            el["y"] = round(bottom - side, 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    out = Path(args.out or ("runtime/minarea_probe_%d" % time.time()))
    out.mkdir(parents=True, exist_ok=True)

    for area in SIZES:
        name = "mug_%03d" % round(area * 1000)
        lay = IG.normalize_layout(with_mug(area))
        cap = IG.layout_to_caption(lay)
        (out / (name + ".caption.json")).write_text(
            json.dumps(cap, ensure_ascii=False, indent=2), encoding="utf-8")
        t0 = time.perf_counter()
        path = IG.generate(None, "", width=1024, height=1024, seed=SEED, caption=cap)
        dt = time.perf_counter() - t0
        if path:
            dest = out / (name + Path(path).suffix)
            dest.write_bytes(Path(path).read_bytes())
            print("[draw] %-10s %.1f%% of frame  %6.1fs  %s"
                  % (name, area * 100, dt, dest))
        else:
            print("[draw] %-10s %.1f%%  nothing came back" % (name, area * 100))
    print("\n  " + str(out))
    print("  read off the smallest one whose mug is still clearly a mug")
    return 0


if __name__ == "__main__":
    sys.exit(main())

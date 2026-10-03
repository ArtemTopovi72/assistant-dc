"""How often does the same caption collage, seed to seed?

The `two_of_the_same` layout is the one scene of fourteen that came back as a
grid of stock photographs. Two causes were found and fixed upstream (MIN_AREA
inflating props, an empty style_description). The third resisted every upstream
lever that was tried:

  * wording -- eight mechanically built "this is one photograph" sentences all
    collaged, while one hand-written sentence worked and then broke when a
    single preposition inside it was changed (bench/draw_scene_floor_probe.py);
  * a richer background -- cured it in one run out of three;
  * geometry -- measured offline, the layout's box statistics (3 elements, 0.23
    coverage, no mutual overlap) are indistinguishable from `kitchen` and
    `self_omission`, both of which render as single rooms. Any geometric
    predicate that fired here would fire on those too.

So the defence is downstream: looks_like_collage() measures the render and
draw_agent re-rolls the seed. That defence has a number in it -- COLLAGE_REROLLS
-- which was chosen rather than measured. This measures it: the same caption, N
seeds, count the collages. From the observed rate p, the chance that k re-rolls
all fail is p**(k+1), which is what the constant should be set from.

Nothing here calls the LLM; ctx is None throughout, so the card is the
renderer's for the duration.

Run: venv/Scripts/python.exe bench/draw_seed_sweep.py --out DIR [--seeds 8]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import draw_agent as DA  # noqa: E402
import ideogram as IG  # noqa: E402

LAYOUT = {
    "background": "комната",
    "elements": [
        {"desc": "чёрный кот сидит на полу",  "x": 0.10, "y": 0.60, "w": 0.22, "h": 0.28},
        {"desc": "рыжий кот спит на кресле",  "x": 0.60, "y": 0.55, "w": 0.26, "h": 0.30},
        {"desc": "деревянный стол",           "x": 0.30, "y": 0.20, "w": 0.34, "h": 0.28},
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    caption = IG.layout_to_caption(IG.normalize_layout(dict(LAYOUT)))
    (out / "caption.json").write_text(
        json.dumps(caption, ensure_ascii=False, indent=2), encoding="utf-8")

    collages = renders = 0
    for n in range(args.seeds):
        seed = 12345 + n * 7919          # a prime stride, so no accidental pattern
        t0 = time.perf_counter()
        try:
            path = IG.generate(None, "", width=args.width, height=args.height,
                               caption=caption, seed=seed)
        except Exception as exc:
            print("[seed %d] FAILED %s: %s" % (seed, type(exc).__name__, exc),
                  flush=True)
            continue
        if not path:
            print("[seed %d] the renderer returned nothing" % seed, flush=True)
            continue
        dest = out / ("seed_%d%s" % (seed, Path(path).suffix))
        dest.write_bytes(Path(path).read_bytes())
        bad = DA.looks_like_collage(str(dest))
        renders += 1
        collages += bool(bad)
        print("[seed %-9d] %6.1fs  %s  %s"
              % (seed, time.perf_counter() - t0,
                 "COLLAGE    " if bad else "one picture", dest.name), flush=True)

    if not renders:
        print("\nno renders completed — nothing can be concluded")
        return 1
    p = collages / float(renders)
    print("\n%d/%d collaged  ->  p = %.2f" % (collages, renders, p))
    for k in (1, 2, 3, 4):
        print("  %d re-roll%s: chance all attempts collage = %.3f"
              % (k, " " if k == 1 else "s", p ** (k + 1)))
    print("\nCOLLAGE_REROLLS is currently %d" % DA.COLLAGE_REROLLS)
    return 0


if __name__ == "__main__":
    sys.exit(main())

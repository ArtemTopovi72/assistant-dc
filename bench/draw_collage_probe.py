"""Why one layout came back as a photo COLLAGE instead of one photograph.

In the 14-scenario render run, thirteen scenes came back as single coherent
photographs and `two_of_the_same` came back as a 2x2 grid of separate stock
photos, complete with two invented people. Its caption was not malformed: the
photographic style floor had fired, every box was in range, and the areas were
sane -- so neither of the two causes already found (MIN_AREA inflating props,
an empty style_description) explains it.

What is unusual about that layout is what it does NOT say:

    background: "комната"            one word
    high_level_description: ""       absent
    elements: a cat on the FLOOR, a cat asleep in an ARMCHAIR, a wooden table

Every element names its own surface, and the background is too thin to hold any
of them. Nothing in the caption says these are one scene, so the model
reconciled three incompatible settings the only way it could: three pictures.

Cyrillic is not the suspect -- twelve Cyrillic layouts in the same run rendered
as single rooms -- and neither is a short background on its own: `self_omission`
had two words and rendered one living room.

So this renders the same layout four ways at one seed, changing only the two
fields in question:

    as_is        the caption exactly as the run produced it
    high_level   + a synthesized one-sentence "this is a single photograph of..."
    background   + a background rich enough to hold the elements
    both

Run: venv/Scripts/python.exe bench/draw_collage_probe.py --out DIR
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ideogram as IG

SEED = 12345

BASE = {
    "background": "комната",
    "elements": [
        {"desc": "чёрный кот сидит на полу",  "x": 0.10, "y": 0.60, "w": 0.22, "h": 0.28},
        {"desc": "рыжий кот спит на кресле",  "x": 0.60, "y": 0.55, "w": 0.26, "h": 0.30},
        {"desc": "деревянный стол",           "x": 0.30, "y": 0.20, "w": 0.34, "h": 0.28},
    ],
}

HIGH_LEVEL = ("Одна фотография одной комнаты: чёрный кот на полу, рыжий кот спит "
              "в кресле, деревянный стол — все в одном кадре.")
RICH_BG = ("комната с деревянным полом, креслом у стены и окном, дневной свет")


def variants():
    yield "as_is", dict(BASE)
    yield "high_level", dict(BASE, high_level_description=HIGH_LEVEL)
    yield "background", dict(BASE, background=RICH_BG)
    yield "both", dict(BASE, background=RICH_BG, high_level_description=HIGH_LEVEL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name, layout in variants():
        caption = IG.layout_to_caption(IG.normalize_layout(layout))
        (out / (name + ".caption.json")).write_text(
            json.dumps(caption, ensure_ascii=False, indent=2), encoding="utf-8")
        t0 = time.perf_counter()
        try:
            # ctx=None: nothing in this path may call the LLM, the card is the
            # renderer's for the duration of the probe.
            path = IG.generate(None, "", width=args.width, height=args.height,
                               caption=caption, seed=SEED)
            why = path or "the renderer returned nothing"
        except Exception as exc:
            path, why = None, "%s: %s" % (type(exc).__name__, exc)
        if path:
            dest = out / (name + Path(path).suffix)
            dest.write_bytes(Path(path).read_bytes())
            why = str(dest)
        print("[draw] %-12s %6.1fs  %s" % (name, time.perf_counter() - t0, why),
              flush=True)


if __name__ == "__main__":
    main()

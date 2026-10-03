"""A grid of stock photos is not the picture the user asked for.

The `two_of_the_same` layout -- a black cat on the floor, a ginger cat in an
armchair, a wooden table, over a one-word background -- rendered as a 2x2 grid of
separate photographs, often on a transparency checkerboard, with invented people
and posters filling the spare panels.

Eight renders at one fixed seed established that WORDING is not a control
surface for this. A hand-written "one photograph of one room" sentence produced a
single coherent room and reproduced pixel for pixel across runs; every
mechanically built version of it collaged; and changing one preposition in the
sentence that worked ("в кресле" -> "на кресле") collapsed that one into a
collage as well. The output is chaotically sensitive to the exact string.

So the picture is measured instead of argued with, and a collage is redrawn on a
fresh seed. This suite pins the measurement (on images built here, so it depends
on no fixture) and the redraw.

Offline: no renderer, no LLM, no GPU.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import draw_agent as D  # noqa: E402

OK = BAD = 0


def check(name, cond, detail=""):
    global OK, BAD
    if cond:
        OK += 1
        print("PASS  " + name)
    else:
        BAD += 1
        print("FAIL  " + name + ((": " + str(detail)) if detail else ""))
        if os.environ.get("PYTEST_CURRENT_TEST"):
            raise AssertionError(str(name) + ((": " + str(detail)) if detail else ""))


import tempfile  # noqa: E402

TMP = tempfile.mkdtemp()


def save(arr, name):
    path = os.path.join(TMP, name)
    Image.fromarray(arr.astype(np.uint8)).save(path)
    return path


def photo(seed=0, size=512):
    """Something photograph-like: smooth gradients, blobs, soft noise -- edges
    that belong to objects rather than to panel borders."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    base = 90 + 60 * np.sin(x / 90.0) + 40 * np.cos(y / 70.0)
    for _ in range(6):
        cy, cx, r = rng.integers(80, size - 80, 2).tolist() + [int(rng.integers(40, 90))]
        m = (y - cy) ** 2 + (x - cx) ** 2 < r * r
        base = np.where(m, base * 0.55 + 40, base)
    base = base + rng.normal(0, 3, base.shape)
    return np.clip(np.dstack([base, base * 0.97, base * 0.92]), 0, 255)


def collage(size=512):
    """Four panels butted together: a hard seam runs the full width and height."""
    out = np.zeros((size, size, 3))
    half = size // 2
    for k, (r, c) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        p = photo(seed=k + 1, size=half)
        out[r * half:(r + 1) * half, c * half:(c + 1) * half] = p
    # the gutter itself
    out[half - 3:half + 3, :] = 250
    out[:, half - 3:half + 3] = 250
    return out


def checkerboard(size=512, tile=16):
    """The transparency checkerboard a cut-out render sits on."""
    y, x = np.mgrid[0:size, 0:size]
    v = np.where(((y // tile) + (x // tile)) % 2 == 0, 204, 235)
    out = np.dstack([v, v, v]).astype(float)
    out[120:360, 120:360] = photo(seed=9, size=240)     # one pasted object
    return out


print("=" * 66)
print("A COLLAGE IS RECOGNISED, A PHOTOGRAPH IS NOT CONDEMNED")
print("=" * 66)

check("a four-panel grid is a collage",
      D.looks_like_collage(save(collage(), "grid.png")))
check("a cut-out on a transparency checkerboard is a collage",
      D.looks_like_collage(save(checkerboard(), "checker.png")))
for s in (0, 1, 2, 3):
    check("an ordinary photograph is left alone (seed %d)" % s,
          not D.looks_like_collage(save(photo(seed=s), "photo%d.png" % s)))

flat = np.full((512, 512, 3), 128.0)
check("a flat image is not a collage", not D.looks_like_collage(save(flat, "flat.png")))
check("a file that cannot be read is not condemned",
      not D.looks_like_collage(os.path.join(TMP, "nope.png")))

# A picture whose top half is bright sky and bottom half dark ground has ONE
# strong horizontal edge, but a gradual one: it must not read as a panel border.
grad = np.zeros((512, 512, 3))
for i in range(512):
    grad[i, :, :] = 40 + i * 0.4
check("a smooth horizon is not a seam",
      not D.looks_like_collage(save(grad, "grad.png")))

print()
print("=" * 66)
print("A COLLAGE IS REDRAWN ON A FRESH SEED")
print("=" * 66)

import ideogram as _IG  # noqa: E402

GRID = save(collage(), "grid2.png")
GOOD = save(photo(seed=7), "good.png")

seeds = []
_real_gen = _IG.generate
_real_cap = _IG.layout_to_caption
_real_crit = D.critique
_real_text = D.verify_text
_IG.layout_to_caption = lambda layout: {"x": 1}
D.critique = lambda *a, **k: {"ok": True, "score": 9, "problems": [], "ops": [],
                              "source": "stub"}
D.verify_text = lambda *a, **k: {"checks": []}

LAYOUT = {"background": "комната", "elements": [{"desc": "кот", "x": .1, "y": .1,
                                                 "w": .2, "h": .2}]}
try:
    def _gen_collage_then_good(ctx, prompt, **kw):
        seeds.append(kw.get("seed"))
        return GRID if len(seeds) == 1 else GOOD

    _IG.generate = _gen_collage_then_good
    res = D.run(None, layout=dict(LAYOUT), rounds=0)
    check("a collage costs a second render", len(seeds) == 2, seeds)
    check("...on a DIFFERENT seed", len(seeds) == 2 and seeds[0] != seeds[1], seeds)

    # bench/draw_seed_sweep.py rendered the only caption that ever collaged in
    # the bench at eight seeds and got eight collages: p = 1.00. A re-roll on
    # the same geometry buys nothing and costs ~85s, so the count is now zero,
    # and this pins it -- raising it again without a measurement fails here.
    check("re-rolling the same geometry is not a rung any more",
          D.COLLAGE_REROLLS == 0, D.COLLAGE_REROLLS)
    check("and the picture handed back is the good one", res.get("image") == GOOD,
          res.get("image"))

    seeds.clear()
    _IG.generate = lambda ctx, prompt, **kw: (seeds.append(kw.get("seed")) or GOOD)
    D.run(None, layout=dict(LAYOUT), rounds=0)
    check("a picture that is fine is drawn once", len(seeds) == 1, seeds)

    seeds.clear()
    _IG.generate = lambda ctx, prompt, **kw: (seeds.append(kw.get("seed")) or GRID)
    res3 = D.run(None, layout=dict(LAYOUT), rounds=0)
    # The full ladder for a SINGLE-element layout: the first draw, two seed
    # re-rolls, the wide redraw, then the stretch-to-frame. The merge rung is
    # skipped -- there is nothing to merge -- so it does not appear here.
    # The full ladder is now: the first draw, the wide redraw, the stretch. Three
    # renders, not five -- the two re-rolls in between were measured worthless.
    check("a layout that collages on every seed stops rather than burning the card",
          len(seeds) == D.COLLAGE_REROLLS + 3, seeds)
    check("...which is three renders, not five",
          len(seeds) == 3, seeds)
    check("...and still hands back a picture rather than nothing",
          res3.get("image") == GRID, res3.get("image"))
finally:
    _IG.generate, _IG.layout_to_caption = _real_gen, _real_cap
    D.critique, D.verify_text = _real_crit, _real_text

print()
print("=" * 66)
print("A SQUARE IS WHAT A 2x2 GRID FITS INTO")
print("=" * 66)

check("a square frame is widened", D._widened(1024, 1024) == (1344, 768),
      D._widened(1024, 1024))
check("a portrait frame is widened too", D._widened(768, 1024)[0]
      > D._widened(768, 1024)[1], D._widened(768, 1024))
check("an already-wide frame is left alone", D._widened(1280, 720) == (0, 0))
check("the pixel count is roughly preserved",
      0.7 < (D._widened(1024, 1024)[0] * D._widened(1024, 1024)[1]) / (1024 * 1024) < 1.4,
      D._widened(1024, 1024))
check("a nonsense size does not raise", D._widened(0, 0) == (0, 0))

seeds_wide = []
_IG.layout_to_caption = lambda layout: {"n": len(layout.get("elements") or [])}
D.critique = lambda *a, **k: {"ok": True, "score": 9, "problems": [], "ops": [],
                              "source": "stub"}
D.verify_text = lambda *a, **k: {"checks": []}
try:
    def _grid_until_wide(ctx, prompt, **kw):
        seeds_wide.append((kw.get("width"), kw.get("height")))
        return GOOD if kw.get("width") != kw.get("height") else GRID

    _IG.generate = _grid_until_wide
    res_w = D.run(None, layout=dict(LAYOUT), rounds=0, width=1024, height=1024)
    check("the wide redraw is tried straight after the first collage",
          len(seeds_wide) == D.COLLAGE_REROLLS + 2, seeds_wide)
    check("...with no same-geometry redraw in between",
          [wh for wh in seeds_wide if wh == (1024, 1024)] == [(1024, 1024)],
          seeds_wide)
    check("...and it keeps the boxes rather than merging them",
          seeds_wide[-1] == (1344, 768), seeds_wide[-1])
    check("and the wide picture is what the caller gets", res_w.get("image") == GOOD,
          res_w.get("image"))
finally:
    _IG.generate, _IG.layout_to_caption = _real_gen, _real_cap
    D.critique, D.verify_text = _real_crit, _real_text

print()
print("=" * 66)
print("A CUT-OUT IS BOXES THAT DO NOT REACH THE EDGES")
print("=" * 66)

import ideogram_layout as _IL0  # noqa: E402

# Measured on the one scene that survived every other rung: two_edits rendered
# as a coherent kitchen squeezed into a band, with a transparency checkerboard
# above and below. Its caption was faultless; every box sat inside x 310-800,
# y 20-750 of a 1000x1000 frame. The same caption with the boxes stretched to
# the edges rendered as a full photograph at a fixed seed.
_narrow = {"background": "кухня", "elements": [
    {"desc": "стол", "x": .45, "y": .20, "w": .30, "h": .55},
    {"desc": "кружка", "x": .31, "y": .34, "w": .14, "h": .12},
    {"desc": "кот", "x": .55, "y": .04, "w": .20, "h": .20}]}
_f = _IL0.filled_layout(_narrow)
_xs = [e["x"] for e in _f["elements"]] + [e["x"] + e["w"] for e in _f["elements"]]
_ys = [e["y"] for e in _f["elements"]] + [e["y"] + e["h"] for e in _f["elements"]]
check("the elements now reach both edges",
      min(_xs) < 0.05 and max(_xs) > 0.95, (min(_xs), max(_xs)))
check("...on both axes", min(_ys) < 0.05 and max(_ys) > 0.95, (min(_ys), max(_ys)))
check("the arrangement is stretched, not rearranged",
      [e["desc"] for e in _f["elements"]] == ["стол", "кружка", "кот"]
      and _f["elements"][1]["x"] < _f["elements"][0]["x"], _f["elements"])
check("a layout that already reaches the edges is untouched",
      _IL0.filled_layout({"background": "к", "elements": [
          {"desc": "a", "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}]})["elements"][0]["w"]
      == 1.0)
check("an empty layout does not raise",
      _IL0.filled_layout({"background": "к", "elements": []})["elements"] == [])

_seen_caps = []
_IG.layout_to_caption = lambda layout: {"n": len(layout.get("elements") or []),
                                        "span": round(max(
                                            (e["x"] + e["w"]) for e in layout["elements"]) -
                                            min(e["x"] for e in layout["elements"]), 2)
                                        if layout.get("elements") else 0}
D.critique = lambda *a, **k: {"ok": True, "score": 9, "problems": [], "ops": [],
                              "source": "stub"}
D.verify_text = lambda *a, **k: {"checks": []}
try:
    def _cutout_until_filled(ctx, prompt, **kw):
        cap = kw.get("caption") or {}
        _seen_caps.append(cap)
        # the stretched layout is the one whose elements span the frame
        return GOOD if cap.get("span", 0) >= 0.9 else GRID

    _IG.generate = _cutout_until_filled
    _res_f = D.run(None, layout=dict(_narrow), rounds=0, width=1024, height=1024)
    check("the stretch is tried when everything else still looks wrong",
          _res_f.get("image") == GOOD, _res_f.get("image"))
    check("...and it keeps every element rather than merging them",
          _seen_caps[-1]["n"] == 3, _seen_caps[-1])
finally:
    _IG.generate, _IG.layout_to_caption = _real_gen, _real_cap
    D.critique, D.verify_text = _real_crit, _real_text

print()
print("=" * 66)
print("WHEN EVEN THAT COLLAGES, GIVE UP THE BOXES, NOT THE PICTURE")
print("=" * 66)

import ideogram_layout as _IL  # noqa: E402

LAY = {"background": "комната", "elements": [
    {"desc": "чёрный кот сидит на полу", "x": .1, "y": .6, "w": .22, "h": .28},
    {"desc": "рыжий кот спит на кресле", "x": .6, "y": .55, "w": .26, "h": .3},
    {"desc": "деревянный стол", "x": .3, "y": .2, "w": .34, "h": .28}]}

m = _IL.merged_layout(LAY)
check("the scene becomes ONE element", len(m["elements"]) == 1, m["elements"])
check("...covering the whole frame",
      (m["elements"][0]["x"], m["elements"][0]["y"],
       m["elements"][0]["w"], m["elements"][0]["h"]) == (0.0, 0.0, 1.0, 1.0),
      m["elements"][0])
for d in ("чёрный кот сидит на полу", "рыжий кот спит на кресле", "деревянный стол"):
    check("...and still names %r" % d[:18], d in m["elements"][0]["desc"])
check("the background leads the sentence",
      m["elements"][0]["desc"].startswith("комната"), m["elements"][0]["desc"])
check("an English scene merges in English",
      "containing:" in _IL.merged_layout(
          {"background": "a room", "elements": [{"desc": "a cat"}, {"desc": "a table"}]}
      )["elements"][0]["desc"])
one = _IL.merged_layout({"background": "комната", "elements": [{"desc": "кот"}]})
check("a single-element layout is left exactly as it was",
      one["elements"][0]["desc"] == "кот", one["elements"])

seeds = []
_IG.layout_to_caption = lambda layout: {"n": len(layout.get("elements") or [])}
D.critique = lambda *a, **k: {"ok": True, "score": 9, "problems": [], "ops": [],
                              "source": "stub"}
D.verify_text = lambda *a, **k: {"checks": []}
try:
    def _grid_until_merged(ctx, prompt, **kw):
        seeds.append(kw.get("seed"))
        # the merged caption is the one carrying a single element
        return GOOD if (kw.get("caption") or {}).get("n") == 1 else GRID

    _IG.generate = _grid_until_merged
    res = D.run(None, layout=dict(LAY), rounds=0)
    # first draw + two re-rolls + wide + stretch-to-frame + merge, in that order
    check("the merged redraw is the LAST rung, after every cheaper one",
          len(seeds) == D.COLLAGE_REROLLS + 4, seeds)
    check("and it is what the caller gets", res.get("image") == GOOD, res.get("image"))

    # The merged caption draws the safety card on some seeds. That must cost a
    # seed, never the picture already in hand.
    seeds.clear()
    calls = {"n": 0}

    def _refuse_merge(ctx, prompt, **kw):
        seeds.append(kw.get("seed"))
        if (kw.get("caption") or {}).get("n") == 1:
            calls["n"] += 1
            if calls["n"] == 1:
                raise _IG.ContentRefused("safety card")
            return GOOD
        return GRID

    _IG.generate = _refuse_merge
    res2 = D.run(None, layout=dict(LAY), rounds=0)
    check("a refused merge is retried on another seed rather than surfacing",
          res2.get("image") == GOOD, res2.get("image"))

    seeds.clear()
    _IG.generate = lambda ctx, prompt, **kw: (seeds.append(kw.get("seed")) or GRID)
    res3 = D.run(None, layout=dict(LAY), rounds=0)
    check("if even the merge collages, the caller still gets a picture",
          res3.get("image") == GRID, res3.get("image"))
finally:
    _IG.generate, _IG.layout_to_caption = _real_gen, _real_cap
    D.critique, D.verify_text = _real_crit, _real_text

print()
print("%d passed, %d failed" % (OK, BAD))
sys.exit(1 if BAD else 0)

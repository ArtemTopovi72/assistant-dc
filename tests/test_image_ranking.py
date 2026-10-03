"""A bigger picture is not a better picture.

A deck about the Kirovets tractor came back with the FLAG OF KENYA on its cover.
The search itself was fine — the engine's own top hit was a photograph of the
tractor — but image_search_urls re-sorted the candidates by PIXEL AREA, and a
4096x2160 flag is 8.8 megapixels against the right photo's 0.4. No aspect or
source bonus in that function can survive a 20x multiplier, so relevance was
thrown away every time a wallpaper appeared anywhere in the results.

Relevance is now the spine: the engine's order dominates, the aspect and source
bonuses reorder neighbours, and size is only a floor that keeps thumbnails out.

Offline: the search boundary is replaced, so no network.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import search  # noqa: E402

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


def row(url, w, h, page=""):
    return {"image": url, "url": page or url, "width": w, "height": h}


def with_rows(rows, **kw):
    real = search._ddgs_retry
    search._ddgs_retry = lambda ctx, fn, label: rows
    try:
        return search.image_search_urls(None, "кировец трактор", **kw)
    finally:
        search._ddgs_retry = real


print("=" * 66)
print("THE ENGINE'S RELEVANCE ORDER SURVIVES")
print("=" * 66)

# Exactly the shape that produced the flag: the right answer first and small,
# an enormous irrelevant wallpaper further down.
FLAG = row("https://x.ru/flag-of-kenya.jpg", 4096, 2160)
GOOD = row("https://x.ru/kirovets.jpg", 800, 533)
rows = [GOOD, row("https://x.ru/second.jpg", 930, 620), FLAG]

got = with_rows(list(rows), prefer="landscape")
check("the engine's first hit stays first", got[0].endswith("kirovets.jpg"), got[:2])
check("the 8-megapixel wallpaper does not lead", "flag" not in got[0], got[0])
check("...but it is still offered, further down", any("flag" in u for u in got), got)

check("the same holds for a portrait search",
      with_rows(list(rows), prefer="portrait")[0].endswith("kirovets.jpg"))

print()
print("=" * 66)
print("SIZE IS A DEMOTION, NOT A SCORE AND NOT A DELETION")
print("=" * 66)

TINY = row("https://x.ru/thumb.jpg", 90, 60)
got = with_rows([TINY, GOOD], prefer="landscape")
# Demoted, not deleted: dropping the small ones emptied the list on a result
# set of nothing but thumbnails, and a caller that asked for images and got none
# crashes on urls[0]. An existing suite caught that; my own placeholder check
# (len >= 0) asserted nothing and hid it.
check("a thumbnail sinks below a usable candidate the engine ranked lower",
      got and got[0].endswith("kirovets.jpg"), got)
check("...but it is not thrown away either", any("thumb" in u for u in got), got)

check("a candidate with no dimensions is kept, since unknown is not small",
      any("nodim" in u for u in with_rows(
          [row("https://x.ru/nodim.jpg", 0, 0), GOOD], prefer="landscape")))

# Everything below the floor must NOT empty the list: a caller that asked for
# images and got none crashes on urls[0]. The first version of this check
# asserted len() >= 0 -- which is to say nothing at all -- and an existing
# suite caught the crash instead.
only_tiny = with_rows([TINY, row("https://x.ru/t2.jpg", 80, 80)], prefer="landscape")
check("when everything is below the floor, the list is not emptied",
      len(only_tiny) == 2, only_tiny)
check("...and a normal-sized candidate still leads when there is one",
      with_rows([TINY, GOOD], prefer="landscape")[0] == GOOD["image"])

print()
print("=" * 66)
print("THE BONUSES REORDER NEIGHBOURS, NOT THE WHOLE LIST")
print("=" * 66)

# A preferred source one place behind may overtake; one at the bottom may not.
NEAR = row("https://upload.wikimedia.org/near.jpg", 900, 600)
got = with_rows([GOOD, NEAR], prefer="landscape")
check("a preferred source can overtake its immediate neighbour",
      got[0].endswith("near.jpg"), got)

FAR = row("https://upload.wikimedia.org/far.jpg", 900, 600)
got = with_rows([GOOD] + [row("https://x.ru/f%d.jpg" % i, 900, 600) for i in range(6)] + [FAR],
                prefer="landscape")
check("a preferred source at the bottom does NOT jump to the top",
      not got[0].endswith("far.jpg"), got[:2])

# Aspect: with landscape preferred, a portrait of equal relevance sinks.
P = row("https://x.ru/tall.jpg", 600, 900)
L = row("https://x.ru/wide.jpg", 900, 600)
check("landscape is preferred for slide artwork",
      with_rows([P, L], prefer="landscape")[0].endswith("wide.jpg"))
check("portrait is preferred for a face reference",
      with_rows([L, P], prefer="portrait")[0].endswith("tall.jpg"))

print()
print("=" * 66)
print("WATERMARKED STOCK IS STILL REFUSED")
print("=" * 66)

STOCK = row("https://cdn.example/x.jpg", 4000, 3000, page="https://www.alamy.com/photo")
got = with_rows([STOCK, GOOD], prefer="landscape")
check("an agency preview is dropped whatever its size",
      got == [GOOD["image"]], got)

STAMPED = row("https://toptexnik.ru/wp-content/uploads/2018/10/9_2.jpg", 1200, 800)
got = with_rows([STAMPED, GOOD], prefer="landscape")
# Observed on a slide: this aggregator stamps its own name across the picture in
# letters the height of the tractor cab. Same family as the agency previews.
check("an aggregator that stamps its own watermark is refused too",
      got == [GOOD["image"]], got)

print()

print()
print("=" * 66)
print("A PACKSHOT ON WHITE IS NOT A COVER")
print("=" * 66)

# Measured on the live search: the best hit for "Витамин D капсулы солнце" was
# an 800x800 photograph of one brand's carton on white. As a deck's full-bleed
# cover that is an advertisement, and the cover's white text vanishes into it.
# The signal is the BORDER: a catalogue shot isolates its object and leaves the
# edges blank. Real hits measured 0.92 for the packshot against 0.00 and 0.01
# for a tractor in a field and a city street.
import tempfile as _tf
import numpy as _np
from PIL import Image as _Im

def _make(kind):
    a = _np.zeros((256, 256, 3), dtype=_np.uint8)
    if kind == "packshot":
        a[:, :] = 255
        a[70:190, 70:190] = (90, 60, 140)          # the object, centred
    else:
        a[:, :] = (120, 110, 90)                   # a scene fills the frame
        a[40:120, 30:200] = (60, 90, 60)
    p = os.path.join(_tf.mkdtemp(), kind + ".png")
    _Im.fromarray(a).save(p)
    return p

_pack, _scene = _make("packshot"), _make("scene")
check("an object on white is recognised as a cut-out",
      search.looks_like_cutout(_pack))
check("a photograph that fills the frame is not", not search.looks_like_cutout(_scene))
check("an unreadable file is not called a cut-out",
      not search.looks_like_cutout("does-not-exist.png"))

# PAPER white is the signal, not brightness. At the first threshold (238) this
# also refused a snowfield, and that was written up as a known cost -- wrongly.
# Measured on the same images plus two snow scenes, as the share of the border
# at >=252 in every channel: packshot 0.91, snow with grain 0.03, snow under
# flat light 0.00, the two real photographs 0.00. A photograph is never 255.
_rng = _np.random.default_rng(3)
_snow = _np.clip(246 + _rng.normal(0, 4, (256, 256, 3))
                 + _np.linspace(-6, 6, 256)[:, None, None], 0, 255).astype(_np.uint8)
_snow[110:170, 40:210] = (40, 40, 45)
_sp = os.path.join(_tf.mkdtemp(), "snow.png")
_Im.fromarray(_snow).save(_sp)
check("a snowfield is a scene, not a cut-out", not search.looks_like_cutout(_sp),
      "bright is not the same as paper white")

_snow2 = _np.clip(250 + _rng.normal(0, 2, (256, 256, 3)), 0, 255).astype(_np.uint8)
_snow2[110:170, 40:210] = (50, 52, 60)
_sp2 = os.path.join(_tf.mkdtemp(), "snow_flat.png")
_Im.fromarray(_snow2).save(_sp2)
check("...even under flat overcast light", not search.looks_like_cutout(_sp2))

# ...and the threshold must still be strict enough to catch a near-white
# catalogue background, not only a pure 255 one.
_pack2 = _np.full((256, 256, 3), 253, dtype=_np.uint8)
_pack2[70:190, 70:190] = (90, 60, 140)
_pp2 = os.path.join(_tf.mkdtemp(), "pack253.png")
_Im.fromarray(_pack2).save(_pp2)
check("a 253-white catalogue background is still caught",
      search.looks_like_cutout(_pp2))

import slides as _S
check("the deck refuses a cut-out as its cover", not _S._is_scene(_pack))
check("...and accepts a real scene", _S._is_scene(_scene))

print("%d passed, %d failed" % (OK, BAD))
sys.exit(1 if BAD else 0)

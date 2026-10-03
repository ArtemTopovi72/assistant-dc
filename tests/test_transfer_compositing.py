"""Offline tests for transfer compositing / edge quality (no backend, no GPU).

These pin the FIX for the user's complaint "the edges are too transparent": the
composite alpha must keep a fully-opaque core inside the edited region and feather
ONLY a thin band at the seam — not leave a wide washed-out rim.

Run: .\\venv\\Scripts\\python.exe tests\\test_transfer_compositing.py
"""
import os
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from PIL import Image, ImageDraw
import image as im


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _box_mask(w, h, inset):
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    d.rectangle((inset, inset, w - inset, h - inset), fill=255)
    return m


def test_solid_core_thin_feather():
    """A large filled region must stay 255 in its interior; only a thin rim feathers."""
    w = h = 400
    inset = 80                       # region is the central 240x240 box
    mask = _box_mask(w, h, inset)
    alpha = im._feathered_alpha(mask, w, h)
    px = alpha.load()
    cx, cy = w // 2, h // 2
    # 1) interior is fully opaque (NOT washed out)
    _check(px[cx, cy] == 255, f"center alpha {px[cx, cy]} != 255 (washed-out core)")
    # 2) a point well inside the region (40px in from the boundary) is still solid
    _check(px[inset + 40, cy] == 255, f"interior alpha {px[inset + 40, cy]} != 255")
    # 3) outside the region is fully transparent (no bleed onto the original)
    _check(px[10, 10] == 0, f"outside alpha {px[10, 10]} != 0 (mask leaked outward)")
    # 4) the feather band is THIN: measure the ramp width along the mid-row.
    row = [px[x, cy] for x in range(w)]
    ramp = [x for x in range(w) if 0 < row[x] < 255]
    # transition pixels exist (it is feathered, not a hard step)…
    _check(ramp, "no feather at all (hard seam)")
    # …but the band is narrow on each side (<= ~30px), not a wide translucent rim.
    left_band = [x for x in ramp if x < cx]
    width = max(left_band) - min(left_band) + 1 if left_band else 0
    _check(width <= 32, f"feather band too wide ({width}px) — washed-out edge regressed")
    print(f"PASS: solid opaque core + thin feather ({width}px band)")
    return True


def test_no_full_erosion_regression():
    """Regression guard: the OLD behavior eroded by the full band, shrinking the
    opaque plateau far inside the boundary. Assert the opaque area is now the
    MAJORITY of the region (the fix), not a shrunken core."""
    w = h = 400
    inset = 80
    mask = _box_mask(w, h, inset)
    alpha = im._feathered_alpha(mask, w, h)
    region_px = mask.histogram()[255]
    opaque_px = alpha.histogram()[255]
    frac = opaque_px / float(region_px)
    _check(frac >= 0.55, f"opaque core is only {frac:.0%} of region (edge washout regressed)")
    print(f"PASS: opaque core is {frac:.0%} of the edited region (>=55%)")
    return True


def test_small_tile_still_feathers():
    """Tiny tiles must not crash and must still feather a little (graceful)."""
    w = h = 24
    mask = _box_mask(w, h, 4)
    alpha = im._feathered_alpha(mask, w, h)
    _check(alpha.size == (w, h), "alpha size changed")
    _check(alpha.load()[w // 2, h // 2] == 255, "tiny-tile core not opaque")
    print("PASS: small tile feathers without crashing")
    return True


def test_content_aware_alpha_tattoo():
    """Additive edit (ink on skin): only changed pixels composite — surrounding skin
    must stay at alpha 0 so no halo can form."""
    from PIL import Image, ImageDraw
    w = h = 200
    skin = Image.new("RGB", (w, h), (200, 170, 150))
    res = skin.copy()
    ImageDraw.Draw(res).polygon([(100, 60), (115, 95), (150, 95), (122, 115),
                                 (135, 150), (100, 128), (65, 150), (78, 115),
                                 (50, 95), (85, 95)], fill=(20, 20, 20))  # black star
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).ellipse((40, 40, 160, 160), fill=255)            # round mask
    a = im._content_aware_alpha(skin, res, mask)
    _check(a is not None, "content-aware alpha unexpectedly None for additive edit")
    ap = a.load()
    # the star center is opaque (ink composited)…
    _check(ap[100, 105] > 200, f"ink center not composited: {ap[100,105]}")
    # …but plain skin INSIDE the round mask, away from ink, stays transparent (no halo)
    _check(ap[55, 55] == 0, f"skin inside mask not transparent (halo!): {ap[55,55]}")
    _check(ap[150, 150] == 0, f"skin inside mask not transparent (halo!): {ap[150,150]}")
    print("PASS: content-aware alpha composites ink only — no halo on surrounding skin")
    return True


def test_content_aware_alpha_fullregion_fallback():
    """Whole-region edit (clothing recolor): change fills the mask, so the helper
    returns None and the caller uses the region feather (existing behavior kept)."""
    from PIL import Image, ImageDraw
    w = h = 200
    a_img = Image.new("RGB", (w, h), (40, 40, 40))         # dark shirt
    b_img = Image.new("RGB", (w, h), (140, 20, 20))        # recolored red
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rectangle((30, 30, 170, 170), fill=255)
    a = im._content_aware_alpha(a_img, b_img, mask)
    _check(a is None, "full-region recolor should fall back to region feather (got alpha)")
    print("PASS: content-aware alpha falls back to region feather for whole-region edits")
    return True


def test_odd_helper():
    _check(im._odd(2) == 3 and im._odd(3) == 3 and im._odd(8) == 9, "_odd wrong")
    print("PASS: _odd nearest-odd helper")
    return True


if __name__ == "__main__":
    tests = [test_solid_core_thin_feather, test_no_full_erosion_regression,
             test_small_tile_still_feathers, test_content_aware_alpha_tattoo,
             test_content_aware_alpha_fullregion_fallback, test_odd_helper]
    results = []
    for t in tests:
        try:
            results.append(bool(t()))
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            results.append(False)
    print("\n" + "=" * 48)
    print(f"Results: {sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)

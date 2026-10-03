"""Offscreen GUI/state tests for the Transfer tab (no backend, no GPU).

Covers: individual delete does not nuke all images, state stays consistent,
placeholder restores when empty, role/Target enforcement, mask drawing produces a
valid source-resolution L mask, and _gather threads the target mask through.

Run: QT_QPA_PLATFORM=offscreen .\\venv\\Scripts\\python.exe tests\\test_transfer_gui.py
"""
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
# This file prints ✕/✓ in its PASS lines; the Windows console codepage (cp1251
# here) cannot encode them, which failed the test for a reason unrelated to the
# code under test.
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

from PyQt5.QtWidgets import QApplication, QDialog
from PIL import Image
import gui


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


_APP = QApplication.instance() or QApplication(sys.argv)


def _img(path, w=320, h=240, color=(100, 140, 90)):
    Image.new("RGB", (w, h), color).save(path)
    return path


class _Ctx:
    def __init__(self):
        self.reference_images = []
    def is_cancelled(self): return False


class _Host:
    def __init__(self):
        self.ctx = _Ctx()
        class _IP:
            def add_image(self, p): pass
        self.images_panel = _IP()
    def _add_system(self, *a, **k): pass
    def window(self): return None


def test_mask_canvas_export():
    d = tempfile.mkdtemp()
    p = _img(os.path.join(d, "a.png"), 400, 300)
    canvas = gui.MaskCanvas(p, max_side=200)
    _check(canvas.is_empty(), "fresh canvas should be empty")
    # paint a stroke across the middle
    cx, cy = canvas.disp_w // 2, canvas.disp_h // 2
    canvas.brush = 20
    canvas.stroke_at(cx - 30, cy)
    canvas.stroke_at(cx + 30, cy)
    _check(not canvas.is_empty(), "canvas should register paint")
    out = os.path.join(d, "mask.png")
    res = canvas.export_mask(out)
    _check(res == out and os.path.exists(out), "mask not exported")
    m = Image.open(out).convert("L")
    _check(m.size == (400, 300), f"mask not at source res: {m.size}")
    _check(m.histogram()[255] > 0, "mask has no white pixels")
    # the painted center maps to white, a far corner stays black (no leak)
    _check(m.getpixel((200, 150)) > 0, "painted center not white in exported mask")
    _check(m.getpixel((5, 5)) == 0, "corner leaked white")
    # erase + clear empties it
    canvas.clear()
    _check(canvas.is_empty() and canvas.export_mask(out) is None, "clear did not empty canvas")
    print("PASS: MaskCanvas paints + exports source-resolution L mask + clears")
    return True


def test_individual_delete_keeps_others():
    d = tempfile.mkdtemp()
    paths = [_img(os.path.join(d, f"{i}.png")) for i in range(3)]
    tab = gui.TransferTab(_Host())
    for p in paths:
        tab._add_row(p)
    _check(len(tab._rows) == 3, "rows not added")
    # delete the MIDDLE one
    mid = tab._rows[1]["frame"]
    tab._remove_row(mid)
    remaining = [r["path"] for r in tab._rows]
    _check(len(remaining) == 2, f"delete removed wrong count: {remaining}")
    _check(paths[0] in remaining and paths[2] in remaining and paths[1] not in remaining,
           "individual delete corrupted the set")
    # deleting the rest one-by-one ends clean + restores placeholder
    tab._remove_row(tab._rows[0]["frame"])
    tab._remove_row(tab._rows[0]["frame"])
    _check(tab._rows == [], "rows not empty after deleting all")
    _check(tab._placeholder is not None, "placeholder not restored when empty")
    print("PASS: individual ✕ delete keeps other images + restores placeholder")
    return True


def test_target_enforcement_and_gather():
    d = tempfile.mkdtemp()
    p_target = _img(os.path.join(d, "t.png"))
    p_ref = _img(os.path.join(d, "r.png"))
    tab = gui.TransferTab(_Host())
    tab._add_row(p_ref)
    tab._add_row(p_target)            # newest is Target by default
    # make the ref a clothing source, keep target as TARGET
    tab._rows[0]["combo"].setCurrentIndex(2)   # "Clothing source"
    g = tab._gather()
    _check(g is not None, "gather returned None on a valid 2-image set")
    target, refs, mask, protect_face = g
    _check(target == p_target, f"wrong target: {target}")
    _check(len(refs) == 1 and refs[0][0] == p_ref, "refs wrong")
    _check(mask is None, "no mask drawn yet but gather returned one")
    _check(protect_face is True, "protect_face should default True")
    # assign a mask to the target row and re-gather
    canvas = gui.MaskCanvas(p_target, max_side=120); canvas.brush = 20
    canvas.stroke_at(canvas.disp_w // 2, canvas.disp_h // 2)
    mp = os.path.join(d, "tm.png"); canvas.export_mask(mp)
    trow = next(r for r in tab._rows if r["path"] == p_target)
    trow["mask"] = mp; trow["protect_face"] = False
    _, _, mask2, pf2 = tab._gather()
    _check(mask2 == mp, "target mask not threaded through _gather")
    _check(pf2 is False, "protect_face flag not threaded through _gather")
    # two Targets is rejected by _on_role_changed (single-target invariant)
    tab._rows[0]["combo"].setCurrentIndex(0)   # try to make ref a 2nd TARGET
    tab._on_role_changed(tab._rows[0]["combo"])
    targets = [r for r in tab._rows if gui._TRANSFER_ROLES[r["combo"].currentIndex()][1] == "TARGET"]
    _check(len(targets) == 1, f"single-Target invariant broken: {len(targets)} targets")
    print("PASS: Target enforcement + _gather threads target mask")
    return True


def test_source_region_crop_and_threading():
    """A source mask on a reference row crops that reference to the masked bbox, and
    _gather threads the mask so the worker can crop it (C/H: source region)."""
    import image as im
    from PIL import Image, ImageDraw
    d = tempfile.mkdtemp()
    ref = _img(os.path.join(d, "ref.png"), 400, 300, (30, 30, 30))
    # draw a bright "hat" in the top-left quadrant + a source mask around it
    rimg = Image.open(ref); ImageDraw.Draw(rimg).rectangle((40, 30, 140, 110), fill=(240, 200, 0)); rimg.save(ref)
    m = Image.new("L", (400, 300), 0); ImageDraw.Draw(m).rectangle((30, 20, 150, 120), fill=255)
    mp = os.path.join(d, "srcmask.png"); m.save(mp)
    crop = im.crop_to_mask(ref, mp)
    _check(crop and os.path.exists(crop), "crop_to_mask produced nothing")
    cw, ch = Image.open(crop).size
    _check(cw < 400 and ch < 300, f"source crop not smaller than ref: {cw}x{ch}")
    _check(120 <= cw <= 200 and 100 <= ch <= 180, f"crop bbox off: {cw}x{ch}")
    # empty mask -> None (graceful)
    em = os.path.join(d, "empty.png"); Image.new("L", (400, 300), 0).save(em)
    _check(im.crop_to_mask(ref, em) is None, "empty source mask should return None")

    # threading: a reference row with a mask carries it through _gather (3-tuple)
    tgt = _img(os.path.join(d, "t.png"))
    tab = gui.TransferTab(_Host())
    tab._add_row(ref); tab._add_row(tgt)
    rrow = next(r for r in tab._rows if r["path"] == ref); rrow["mask"] = mp
    tab._rows[0]["combo"].setCurrentIndex(3)   # Object source for the ref
    _t, refs, _m, _pf = tab._gather()
    _check(len(refs[0]) == 3 and refs[0][2] == mp, "source mask not threaded into refs spec")
    print("PASS: source-region crop_to_mask + _gather threads source mask")
    return True


def test_face_protection_on_manual_mask():
    """A manual mask covering the face is honored verbatim by default, but with
    protect_face=True the face box is subtracted (hat/clothing safety)."""
    import image as im
    from PIL import Image, ImageDraw
    import identity_metrics as idm
    d = tempfile.mkdtemp()
    # use a real desktop face if available, else skip the face-box part gracefully
    face_src = os.path.join(os.environ.get("USERPROFILE", os.path.expanduser("~")),
                            "Desktop", "photo_2026-05-20_12-40-47.jpg")
    if not os.path.exists(face_src):
        print("SKIP: face-protection (no desktop face image)")
        return True
    W, H = Image.open(face_src).size
    fb = idm.face_box(face_src, pad=0.10)
    if not fb:
        print("SKIP: face-protection (no face detected)")
        return True
    # mask the whole upper half (covers the face)
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).rectangle((0, 0, W, H // 2), fill=255)
    mp = os.path.join(d, "fullmask.png"); m.save(mp)
    # protect_face -> the returned mask must be 0 inside the face box
    got = im._contained_region_mask(None, face_src, "", grow=12, seed=1, timeout=10,
                                    mask_override=mp, protect_face=True)
    _check(got is not None, "protected mask returned None")
    _, mask, _ = got
    fx = (fb[0] + fb[2]) // 2; fy = (fb[1] + fb[3]) // 2
    _check(mask.getpixel((fx, fy)) == 0, "face center not protected with protect_face=True")
    # without protection the same point stays white (face editable, e.g. tattoo)
    got2 = im._contained_region_mask(None, face_src, "", grow=12, seed=1, timeout=10,
                                     mask_override=mp, protect_face=False)
    _, mask2, _ = got2
    _check(mask2.getpixel((fx, fy)) > 0, "face wrongly protected with protect_face=False")
    print("PASS: manual-mask face protection toggle (hat-safe vs face-tattoo)")
    return True


def test_gather_guards():
    tab = gui.TransferTab(_Host())
    _check(tab._gather() is None, "gather should fail with <2 images")
    print("PASS: _gather guards (need >=2 images)")
    return True


def test_quick_style_promotes_refs_and_fills_prompt():
    """🎭 Quick style transfer: with < 2 rows it just complains (no crash); with
    >= 2 it flips every non-Target row to style_reference, fills the canonical
    instruction, and hands off to _run('contained') -- nothing about the
    underlying transfer pipeline changes, this is only a one-click shortcut
    over the role dropdown that already existed."""
    d = tempfile.mkdtemp()
    p_target = _img(os.path.join(d, "t.png"))
    p_ref = _img(os.path.join(d, "r.png"))
    tab = gui.TransferTab(_Host())

    # < 2 rows: no crash, no _run call, just a status message
    calls = []
    tab._run = lambda mode: calls.append(mode)
    tab._quick_style()
    _check(not calls, "quick_style should not run with < 2 images")
    _check(bool(tab.status.text()), "quick_style should explain why it did nothing")

    tab._add_row(p_ref)
    tab._add_row(p_target)   # newest is Target by default
    calls.clear()
    tab._quick_style()

    style_idx = next(i for i, (_, role) in enumerate(gui._TRANSFER_ROLES)
                      if role == "style_reference")
    non_target = [r for r in tab._rows if tab._role_of(r["combo"]) != "TARGET"]
    _check(non_target and all(r["combo"].currentIndex() == style_idx for r in non_target),
           "non-Target rows not promoted to style_reference")
    _check(any(tab._role_of(r["combo"]) == "TARGET" for r in tab._rows),
           "quick_style must not disturb the single Target row")
    import gui_transfer_tab as _gtt
    _check(tab.instr.text() == _gtt._QUICK_STYLE_PROMPT, "instruction text mismatch")
    _check(calls == ["contained"], f"quick_style should call _run('contained'), got {calls}")
    print("PASS: _quick_style promotes refs to style_reference + fills prompt + runs contained")
    return True


def test_draw_mask_resolves_dialog_through_gui():
    """_draw_mask must reach MaskDrawDialog through `gui` at CALL time.

    It used gui_transfer_tab's own by-value import instead. Every suite that
    drives _draw_mask stubs `gui.MaskDrawDialog`, so that patch was a silent
    no-op: the REAL modal dialog was constructed and exec_() entered a nested
    event loop nothing ever quits -- offscreen included. test_gui_supplement4
    did not fail, it HUNG, forever, and the transfer tab's mask path was
    effectively untested.

    A hang is the worst possible regression signal, so this test does not
    reproduce it. The by-value name is replaced with a class that RAISES on
    construction: if _draw_mask ever reads it again the suite errors out in
    milliseconds instead of blocking.
    """
    import gui_transfer_tab as _tt
    d = tempfile.mkdtemp()
    src = _img(os.path.join(d, "seam_src.png"))
    maskfile = _img(os.path.join(d, "seam_mask.png"))
    tab = gui.TransferTab(_Host())
    tab._add_row(src)

    class _Fake:            # patched onto `gui` -- the seam under test
        mask_path = maskfile
        protect_face = False
        def __init__(self, *a, **k): pass
        def exec_(self): return QDialog.Accepted

    class _Trap:            # patched onto the by-value name -- must never run
        def __init__(self, *a, **k):
            raise AssertionError(
                "_draw_mask read gui_transfer_tab's by-value MaskDrawDialog; "
                "gui.MaskDrawDialog stubs no longer reach it, and the real "
                "modal dialog would hang the caller")

    og, ot = gui.MaskDrawDialog, _tt.MaskDrawDialog
    gui.MaskDrawDialog, _tt.MaskDrawDialog = _Fake, _Trap
    try:
        tab._draw_mask(tab._rows[0]["frame"])
    finally:
        gui.MaskDrawDialog, _tt.MaskDrawDialog = og, ot
    _check(tab._rows[0]["mask"] == maskfile,
           "mask was not taken from the gui-resolved dialog")
    print("PASS: _draw_mask resolves MaskDrawDialog through gui")
    return True


if __name__ == "__main__":
    tests = [test_mask_canvas_export, test_individual_delete_keeps_others,
             test_target_enforcement_and_gather, test_source_region_crop_and_threading,
             test_face_protection_on_manual_mask, test_gather_guards,
             test_quick_style_promotes_refs_and_fills_prompt,
             test_draw_mask_resolves_dialog_through_gui]
    results = []
    for t in tests:
        try:
            results.append(bool(t()))
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"FAIL: {t.__name__}: {e}")
            results.append(False)
    print("\n" + "=" * 48)
    print(f"Results: {sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)

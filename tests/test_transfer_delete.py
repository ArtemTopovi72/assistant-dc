"""Offscreen tests for INDIVIDUAL image removal in the Transfer tab (no backend, no GPU).

Covers the full deletion spec: per-row removal, isolation from other rows/masks/roles,
selection follow-on, placeholder restore, mask-file cleanup (no orphans), busy-job
safety (worker snapshots its inputs), and repeated add/delete cycles.

Run: QT_QPA_PLATFORM=offscreen .\\venv\\Scripts\\python.exe tests\\test_transfer_delete.py
"""
import os
import sys
import gc
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from PyQt5.QtWidgets import QApplication
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


def _tab(n, d):
    """A tab with n distinct images added; returns (tab, [paths])."""
    paths = [_img(os.path.join(d, f"{i}.png"), color=(40 + i * 20, 90, 120)) for i in range(n)]
    tab = gui.TransferTab(_Host())
    for p in paths:
        tab._add_row(p)
    return tab, paths


def _fake_mask(tab, row, d=None):
    """Write a real mask PNG in the dir we own so cleanup logic engages."""
    out = str(gui.OUTPUT_DIR_GUI_MASK() / f"transfer_mask_{id(row)}.png")
    Image.new("L", Image.open(row["path"]).size, 0).save(out)
    row["mask"] = out
    return out


def _roles(tab):
    return [gui._TRANSFER_ROLES[r["combo"].currentIndex()][1] for r in tab._rows]


# --------------------------------------------------------------------------- #

def test_delete_first():
    d = tempfile.mkdtemp()
    tab, paths = _tab(3, d)
    tab._remove_row(tab._rows[0]["frame"])
    rem = [r["path"] for r in tab._rows]
    _check(rem == [paths[1], paths[2]], f"delete-first wrong: {rem}")
    print("PASS: delete first image")
    return True


def test_delete_last():
    d = tempfile.mkdtemp()
    tab, paths = _tab(3, d)
    tab._remove_row(tab._rows[-1]["frame"])
    rem = [r["path"] for r in tab._rows]
    _check(rem == [paths[0], paths[1]], f"delete-last wrong: {rem}")
    print("PASS: delete last image")
    return True


def test_delete_only_restores_placeholder():
    d = tempfile.mkdtemp()
    tab, paths = _tab(1, d)
    _check(tab._placeholder is None, "placeholder should be gone with 1 image")
    tab._remove_row(tab._rows[0]["frame"])
    _check(tab._rows == [], "rows not empty")
    _check(tab._placeholder is not None, "placeholder not restored for empty list")
    _check(tab._sel is None, "selection should be cleared when empty")
    print("PASS: delete only image -> placeholder restored, selection cleared")
    return True


def test_delete_sequence_to_empty():
    d = tempfile.mkdtemp()
    tab, paths = _tab(5, d)
    while tab._rows:
        before = len(tab._rows)
        tab._remove_row(tab._rows[0]["frame"])
        _check(len(tab._rows) == before - 1, "sequential delete miscounted")
        # invariant: every surviving row still has a live frame + its own combo
        for r in tab._rows:
            _check(r["frame"] is not None and r["combo"] is not None, "row corrupted mid-sequence")
    _check(tab._placeholder is not None, "placeholder missing after deleting all")
    print("PASS: delete all 5 in sequence, invariants hold each step")
    return True


def test_delete_does_not_touch_others_roles_or_masks():
    d = tempfile.mkdtemp()
    tab, paths = _tab(4, d)
    # assign distinct roles + masks to the survivors
    tab._rows[0]["combo"].setCurrentIndex(2)   # clothing source
    tab._rows[2]["combo"].setCurrentIndex(3)   # object source
    m0 = _fake_mask(tab, tab._rows[0])
    m2 = _fake_mask(tab, tab._rows[2])
    roles_before = _roles(tab)
    # delete row 1 (a plain ref)
    tab._remove_row(tab._rows[1]["frame"])
    survivors = {r["path"]: r for r in tab._rows}
    _check(paths[1] not in survivors, "target image not removed")
    _check(set(survivors) == {paths[0], paths[2], paths[3]}, "wrong survivors")
    # roles of survivors unchanged
    _check(gui._TRANSFER_ROLES[survivors[paths[0]]["combo"].currentIndex()][1] == "clothing_source",
           "survivor role drifted")
    _check(gui._TRANSFER_ROLES[survivors[paths[2]]["combo"].currentIndex()][1] == "object_source",
           "survivor role drifted")
    # masks of survivors intact on disk + in metadata
    _check(survivors[paths[0]]["mask"] == m0 and os.path.exists(m0), "survivor mask lost")
    _check(survivors[paths[2]]["mask"] == m2 and os.path.exists(m2), "survivor mask lost")
    print("PASS: deleting one row leaves others' roles + masks untouched")
    return True


def test_delete_with_mask_cleans_file():
    d = tempfile.mkdtemp()
    tab, paths = _tab(2, d)
    m = _fake_mask(tab, tab._rows[0])
    _check(os.path.exists(m), "fake mask not written")
    tab._remove_row(tab._rows[0]["frame"])
    _check(not os.path.exists(m), "mask file orphaned after delete (leak)")
    print("PASS: deleting a masked row removes its mask file (no orphan)")
    return True


def test_delete_does_not_clean_foreign_files():
    """A mask path we did NOT write (not in outputs/masks, wrong prefix) is never deleted."""
    d = tempfile.mkdtemp()
    tab, paths = _tab(2, d)
    foreign = os.path.join(d, "user_original_mask.png")
    Image.new("L", (10, 10), 0).save(foreign)
    tab._rows[0]["mask"] = foreign
    _check(not tab._owns_mask(foreign), "_owns_mask wrongly claims a foreign file")
    tab._remove_row(tab._rows[0]["frame"])
    _check(os.path.exists(foreign), "deleted a file we do not own (data loss)")
    print("PASS: foreign/original files are never deleted on row removal")
    return True


def test_selected_row_autoselects_next():
    d = tempfile.mkdtemp()
    tab, paths = _tab(4, d)
    tab._select_row(tab._rows[1])
    _check(tab._sel["path"] == paths[1], "select did not set _sel")
    tab._remove_row(tab._rows[1]["frame"])               # delete the selected (middle)
    _check(tab._sel is not None, "no auto-selection after deleting selected row")
    _check(tab._sel["path"] == paths[2], f"did not select next: {tab._sel['path']}")
    # delete the now-selected LAST element -> selection clamps to new last
    tab._select_row(tab._rows[-1])
    tab._remove_row(tab._rows[-1]["frame"])
    _check(tab._sel is tab._rows[-1], "selection did not clamp to last after deleting last")
    print("PASS: deleting the selected row auto-selects the next (clamped)")
    return True


def test_delete_unselected_keeps_selection():
    d = tempfile.mkdtemp()
    tab, paths = _tab(4, d)
    tab._select_row(tab._rows[2])
    keep = tab._sel
    tab._remove_row(tab._rows[0]["frame"])               # delete a DIFFERENT row
    _check(tab._sel is keep and tab._sel["path"] == paths[2],
           "deleting an unselected row changed the selection")
    print("PASS: deleting a non-selected row keeps the current selection")
    return True


def test_delete_clears_stale_viz():
    d = tempfile.mkdtemp()
    tab, paths = _tab(2, d)
    # pretend the viz is showing row0's image + mask
    m = _fake_mask(tab, tab._rows[0])
    tab._set_viz("target", paths[0])
    tab._set_viz("mask", m)
    _check(tab._viz["target"]._fullpath == paths[0], "viz target not set for test")
    tab._remove_row(tab._rows[0]["frame"])
    _check(tab._viz["target"]._fullpath is None, "stale target preview left after delete")
    _check(tab._viz["mask"]._fullpath is None, "stale mask preview left after delete")
    print("PASS: deleting a row clears the viz panes it had populated")
    return True


def test_delete_while_busy_is_safe():
    """The worker snapshots target+refs at construction, so deleting rows after a job
    starts cannot mutate the running job. We assert the snapshot is independent."""
    d = tempfile.mkdtemp()
    tab, paths = _tab(3, d)
    tab._rows[0]["combo"].setCurrentIndex(2)
    g = tab._gather()
    _check(g is not None, "gather failed on a valid set")
    target, refs, tmask, pf = g
    worker = gui.TransferWorker(tab.ctx, target, list(refs), "x", "contained",
                                target_mask=tmask, protect_face=pf)
    snap_target, snap_refs = worker.target, list(worker.refs_spec)
    # now delete rows underneath the "running" worker
    tab._remove_row(tab._rows[0]["frame"])
    tab._remove_row(tab._rows[0]["frame"])
    _check(worker.target == snap_target, "worker target mutated by row deletion")
    _check(list(worker.refs_spec) == snap_refs, "worker refs mutated by row deletion")
    _check(len(tab._rows) == 1, "rows did not delete while a worker held a snapshot")
    print("PASS: deleting rows while a job holds its snapshot is safe")
    return True


def test_repeated_add_delete_then_transfer_inputs_valid():
    """After many add/delete cycles the surviving set still produces a valid _gather
    (i.e. Transfer would still run). No orphaned rows, masks, or role corruption."""
    d = tempfile.mkdtemp()
    pool = [_img(os.path.join(d, f"p{i}.png"), color=(20 + i * 10, 80, 100)) for i in range(8)]
    tab = gui.TransferTab(_Host())
    for cycle in range(5):
        for p in pool:
            tab._add_row(p)
        # delete a shifting subset each cycle
        for _ in range(3 + cycle):
            if tab._rows:
                tab._remove_row(tab._rows[len(tab._rows) // 2]["frame"])
        # no duplicate paths, no dead frames
        seen = [r["path"] for r in tab._rows]
        _check(len(seen) == len(set(seen)), f"duplicate rows after cycle {cycle}: {seen}")
        # clear between cycles, confirm empty + placeholder
        tab._clear()
        _check(tab._rows == [] and tab._sel is None, "clear left state")
        _check(tab._placeholder is not None, "clear did not restore placeholder")
    # final: a clean 2-image set still gathers
    tab._add_row(pool[0]); tab._add_row(pool[1])
    tab._rows[0]["combo"].setCurrentIndex(2)
    _check(tab._gather() is not None, "Transfer no longer runnable after add/delete churn")
    print("PASS: repeated add/delete cycles -> Transfer inputs still valid")
    return True


def test_no_orphaned_frames_after_delete():
    """Deleted frames must be detached from the layout (setParent(None)) so they are not
    leaked into the widget tree and don't reappear in the row host."""
    d = tempfile.mkdtemp()
    tab, paths = _tab(4, d)
    frames = [r["frame"] for r in tab._rows]
    for fr in list(frames):
        tab._remove_row(fr)
    _APP.processEvents()
    # every removed frame is detached from the list host
    live_children = tab._list_host.findChildren(type(frames[0]))
    leftover = [c for c in live_children if c in frames]
    _check(not leftover, f"orphaned frames still parented to list host: {len(leftover)}")
    gc.collect()
    print("PASS: removed frames are detached (no orphaned widgets in the tree)")
    return True


if __name__ == "__main__":
    tests = [
        test_delete_first,
        test_delete_last,
        test_delete_only_restores_placeholder,
        test_delete_sequence_to_empty,
        test_delete_does_not_touch_others_roles_or_masks,
        test_delete_with_mask_cleans_file,
        test_delete_does_not_clean_foreign_files,
        test_selected_row_autoselects_next,
        test_delete_unselected_keeps_selection,
        test_delete_clears_stale_viz,
        test_delete_while_busy_is_safe,
        test_repeated_add_delete_then_transfer_inputs_valid,
        test_no_orphaned_frames_after_delete,
    ]
    results = []
    for t in tests:
        try:
            results.append(bool(t()))
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"FAIL: {t.__name__}: {e}")
            results.append(False)
    print("\n" + "=" * 52)
    print(f"Results: {sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)

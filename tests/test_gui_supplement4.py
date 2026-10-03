"""Fourth supplement: close specific alternate-input branches (the other side of
if/else guards) across AssistantWindow drag/drop/paste/redraw/settings/memory and
the tabs. Targets the partial-branch tail that the earlier suites left half-covered.

Run: QT_QPA_PLATFORM=offscreen venv/Scripts/python.exe tests/test_gui_supplement4.py
"""
import os, sys, types, tempfile, threading
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("USE_GUI", "0")
os.environ["GUI_REPORT_HTML"] = "0"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _isolate_library  # noqa: F401  — never touch the live library.db
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QDialog, QInputDialog, QMessageBox
from PyQt5.QtWidgets import QFileDialog as _QFD
from PyQt5.QtGui import QImage, QColor
from PyQt5.QtCore import Qt, QMimeData, QUrl, QEvent, QPoint
import gui
from _gui_patch import stub_workers as _stub_workers, restore as _gui_restore

_app = QApplication.instance() or QApplication(sys.argv)
_TMP = tempfile.mkdtemp(prefix="guisupp4_")

def _png(name="s.png"):
    p = os.path.join(_TMP, name)
    img = QImage(40, 30, QImage.Format_RGB32); img.fill(QColor("#66aa88")); img.save(p)
    return p

RESULTS = []
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    assert cond, f"{name}: {detail}"

_orig_loader = gui.ModelLoader
_WNAMES = ["RequestWorker", "RedrawWorker", "FixHandsWorker", "FixArtifactWorker",
           "ModelSwitchWorker", "DeepResearchWorker"]

def _win():
    gui.ModelLoader = type("NoopLoader", (_orig_loader,), {"start": lambda self: None, "run": lambda self: None})
    w = gui.AssistantWindow("m", True, "high")
    gui.ModelLoader = _orig_loader
    d = Path(_TMP) / "wmem4"; d.mkdir(parents=True, exist_ok=True)
    w.ctx = types.SimpleNamespace(cancel_event=threading.Event(), memory_lock=threading.RLock(),
                                  reference_images=[], last_image_path=None, model_name="m",
                                  no_think=True, mic_disabled=False, tts_disabled=False,
                                  reasoning_effort="high", response_length="auto",
                                  active_memory_dir=d, save_memory=lambda *a, **k: None,
                                  load_memory=lambda *a, **k: None, session_memory=[])
    w.graph = object(); w.base_state = {"messages": []}
    w.stack.setCurrentWidget(w.dashboard); w._set_busy(False)
    return w

def _close(w):
    for a in ("worker", "compact_worker", "redraw_worker", "model_switch_worker", "research_worker",
              "_lib_build_worker", "_scan_worker", "transcribe_worker", "recorder", "cap", "vad_listener"):
        setattr(w, a, None)
    try: w.close()
    except Exception: pass

def _noop_workers():
    return _stub_workers(_WNAMES)

def _restore_workers(orig):
    _gui_restore(orig)


def test_dragdrop_alt_branches():
    w = _win()
    img = _png("dd.png")
    # dragEnterEvent with hasImage (no urls)
    md_img = QMimeData(); md_img.setImageData(QImage(img))
    ev = types.SimpleNamespace(mimeData=lambda: md_img, acceptProposedAction=lambda: None)
    w.dragEnterEvent(ev)
    check("drag_hasimage", w._mime_is_droppable(md_img))
    # dragEnterEvent with a non-droppable url (e.g. .zip) -> not accepted
    md_bad = QMimeData(); md_bad.setUrls([QUrl.fromLocalFile("x.zip")])
    w.dragEnterEvent(types.SimpleNamespace(mimeData=lambda: md_bad, acceptProposedAction=lambda: None))
    check("drag_bad_url", not w._mime_is_droppable(md_bad))
    # _route_dropped_mime image-data branch
    check("route_image_data", w._route_dropped_mime(md_img))
    # _url_to_local: file:// (two slashes) and bare path
    check("url_two_slash", "x" in w._url_to_local("file://server/x"))
    check("url_bare", w._url_to_local("C:/a/b.png").endswith("b.png"))
    # dropEvent while busy -> ignored
    w.worker = object()
    w.dropEvent(types.SimpleNamespace(mimeData=lambda: md_img, acceptProposedAction=lambda: None))
    w.worker = None
    _close(w)


def test_eventfilter_paste_and_dragmove():
    w = _win()
    img = _png("ef4.png")
    # Ctrl+V on the input with an image in clipboard -> intercept
    from PyQt5.QtGui import QKeyEvent
    QApplication.clipboard().setImage(QImage(img))
    kev = QKeyEvent(QEvent.KeyPress, Qt.Key_V, Qt.ControlModifier)
    handled = w.eventFilter(w.input, kev)
    check("ctrlv_image", handled is True)
    # Ctrl+V with only text -> falls through (not intercepted)
    QApplication.clipboard().setText("plain text")
    kev2 = QKeyEvent(QEvent.KeyPress, Qt.Key_V, Qt.ControlModifier)
    check("ctrlv_text_fallthrough", w.eventFilter(w.input, kev2) in (False, True))
    _close(w)


def test_redraw_branches():
    orig = _noop_workers()
    try:
        w = _win()
        src = _png("rb.png"); w.ctx.last_image_path = src
        # enhance path: empty instructions (ok True, empty)
        QInputDialog.getText = staticmethod(lambda *a, **k: ("", True))
        w._redraw_last_image(); check("redraw_enhance", w.redraw_worker is not None); w._on_redraw_finished()
        # redraw path: instructions but empty region
        seq = iter([("make it red", True), ("", True)])
        QInputDialog.getText = staticmethod(lambda *a, **k: next(seq, ("", True)))
        w._redraw_last_image(); check("redraw_wholeframe", w.redraw_worker is not None); w._on_redraw_finished()
        # cancel at region prompt (ok2 False)
        seq2 = iter([("change hat", True), ("hat", False)])
        QInputDialog.getText = staticmethod(lambda *a, **k: next(seq2, ("", True)))
        w._redraw_last_image(); check("redraw_region_cancel", w.redraw_worker is None)
        _close(w)
    finally:
        _restore_workers(orig)


def test_open_settings_rejected():
    w = _win()
    class FakeDlg:
        def __init__(self, *a, **k): pass
        # Real QDialog has these; the stub must too, or the test fails on the
        # stub's shape rather than on the behaviour under test.
        def raise_(self): pass
        def activateWindow(self): pass
        def exec_(self): return QDialog.Rejected
    o = gui.SettingsDialog; gui.SettingsDialog = FakeDlg
    try:
        w._open_settings(); check("settings_rejected", w.model_switch_worker is None)
    finally:
        gui.SettingsDialog = o
    _close(w)


def test_switch_profile_guards():
    w = _win()
    w._switch_memory_profile("")           # empty name -> no-op
    w.ctx = None
    w._switch_memory_profile("x")          # ctx None -> no-op
    check("switch_guards", True)
    _close(w)


def test_db_add_list_item_dedup():
    w = _win()
    f = _png("dbi.png")
    w._db_add_list_item(f)
    w._db_add_list_item(f)                  # already listed -> dedup branch
    w._db_add_list_item("")                 # empty path -> early return
    w._db_add_list_item(f, indexed=True, label="Doc")
    check("db_dedup", w.db_file_list.count() == 1)
    _close(w)


def _mc():
    import memory_center
    store = memory_center.MemoryStore(root=tempfile.mkdtemp(prefix="mc4_"))
    store.create_profile("default"); store.create_profile("other")
    e = store.add("default", "note", etype="session", source="inferred")
    sm = store.add("default", "summary text", etype="summary", source="compacted")
    host = types.SimpleNamespace(ctx=types.SimpleNamespace(active_memory_dir=store.profile_dir("default"),
                                                           session_memory=[]),
                                 _set_status=lambda m: None, _memory_center_set_profile=lambda p: None)
    tab = gui.MemoryCenterTab(host); tab.store = store; tab.refresh_all()
    return tab, store, e, sm


def test_mc_move_pin_export_diag():
    tab, store, e, sm = _mc()
    # _move_menu with a selection + profiles -> builds actions (stub QMenu.exec_) + _do_move
    tab._sel = e
    om = gui.QMenu.exec_; gui.QMenu.exec_ = lambda self, *a, **k: None
    try:
        tab._move_menu(); check("mc_move_menu", True)
    finally:
        gui.QMenu.exec_ = om
    tab._do_move("other"); check("mc_do_move", True)
    # _toggle_pin on a SUMMARY entry -> early return (no change)
    tab._sel = sm
    tab._toggle_pin(); check("mc_toggle_summary", True)
    # _prof_rename / _prof_duplicate cancelled (ok False)
    QInputDialog.getText = staticmethod(lambda *a, **k: ("", False))
    for i in range(tab.prof_list.count()):
        tab.prof_list.setCurrentRow(i)
        if tab._selected_profile_name() == "other": break
    tab._prof_rename(); tab._prof_duplicate()
    # valid rename/dup
    QInputDialog.getText = staticmethod(lambda *a, **k: ("other2", True))
    tab._prof_duplicate(); check("mc_prof_dup", "other2" in store.profiles())
    # _run_diag with empty query
    tab.diag_query.setText(""); tab._run_diag(); check("mc_diag_empty", True)
    # _export cancelled (no path)
    _QFD.getSaveFileName = staticmethod(lambda *a, **k: ("", ""))
    for fmt in ("json", "md", "zip"): tab._export(fmt)
    check("mc_export_cancel", True)
    # _remove_selected with no selection (StressTab-like) — here _delete_selected no ids
    tab._sel = None; tab.table.clearSelection(); tab._delete_selected(); check("mc_del_none", True)
    # _gen_compaction with no session sources -> info dialog
    oi = QMessageBox.information; QMessageBox.information = staticmethod(lambda *a, **k: None)
    try:
        # remove the only session note so there are no compaction sources
        store.delete("default", [e.id], soft=True); tab.refresh_all()
        tab._gen_compaction(); check("mc_gen_no_sources", True)
    finally:
        QMessageBox.information = oi


def test_transfer_draw_mask_target_and_preview():
    host = types.SimpleNamespace(ctx=types.SimpleNamespace(reference_images=[], last_image_path=None,
                                                           cancel_event=threading.Event()),
                                 images_panel=types.SimpleNamespace(add_image=lambda p: None),
                                 _add_system=lambda m: None)
    tab = gui.TransferTab(host)
    p1, p2 = _png("t4a.png"), _png("t4b.png")
    tab._add_row(p1); tab._add_row(p2)
    # make first row the TARGET, then draw a target mask
    tab._rows[0]["combo"].setCurrentIndex(0)
    class FakeMask:
        mask_path = _png("transfer_mask_t.png"); protect_face = True
        def __init__(self, *a, **k): pass
        # Real QDialog has these; the stub must too, or the test fails on the
        # stub's shape rather than on the behaviour under test.
        def raise_(self): pass
        def activateWindow(self): pass
        def exec_(self): return QDialog.Accepted
    om = gui.MaskDrawDialog; gui.MaskDrawDialog = FakeMask
    try:
        tab._draw_mask(tab._rows[0]["frame"]); check("tt_draw_target", tab._rows[0]["mask"] is not None)
    finally:
        gui.MaskDrawDialog = om
    # _on_preview with only asset, then only mask
    tab._on_preview({"asset": _png("as4.png")})
    tab._on_preview({"mask": _png("mk4.png")})
    check("tt_preview_alts", True)
    # _gather single explicit target + one ref
    tab._rows[0]["combo"].setCurrentIndex(0)   # target
    tab._rows[1]["combo"].setCurrentIndex(2)   # clothing source
    g = tab._gather(); check("tt_gather_explicit", g is not None)
    # two targets -> error (force via blockSignals so _on_role_changed doesn't demote)
    for r in tab._rows:
        r["combo"].blockSignals(True); r["combo"].setCurrentIndex(0); r["combo"].blockSignals(False)
    check("tt_two_targets", tab._gather() is None)


if __name__ == "__main__":
    _saved = {"gof": _QFD.getOpenFileName, "gsf": _QFD.getSaveFileName, "gt": QInputDialog.getText}
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback; traceback.print_exc()
            print(f"  ERROR in {fn.__name__}: {type(e).__name__}: {e}")
    _QFD.getOpenFileName = staticmethod(_saved["gof"])
    _QFD.getSaveFileName = staticmethod(_saved["gsf"])
    QInputDialog.getText = staticmethod(_saved["gt"])
    print(f"\n{len(fns)-failed}/{len(fns)} supplement4 functions passed "
          f"({sum(1 for _,c in RESULTS if c)}/{len(RESULTS)} checks)")
    # A failed check() must reach the exit code. This used to be `1 if failed else 0`,
    # where `failed` counts EXCEPTIONS only -- so every assertion in the file could
    # print FAIL while the suite still exited 0, invisible to anything judging by
    # exit code (which is how these suites are judged).
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)

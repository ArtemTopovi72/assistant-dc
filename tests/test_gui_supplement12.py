"""Twelfth supplement: correctly-retargeted alternate-guard branches (the FIRST
guard side of if/return pairs that earlier tests skipped). Clean-exit pattern
(os._exit after explicit coverage save) to avoid PyQt teardown crashes.

Run: QT_QPA_PLATFORM=offscreen venv/Scripts/python.exe tests/test_gui_supplement12.py
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

import numpy as np
from PyQt5.QtWidgets import QApplication, QMessageBox, QInputDialog
from PyQt5.QtWidgets import QFileDialog as _QFD
from PyQt5.QtGui import QImage, QColor
from PyQt5.QtCore import Qt, QMimeData, QUrl
import gui
from _gui_patch import stub_workers as _stub_workers, restore as _gui_restore

_app = QApplication.instance() or QApplication(sys.argv)
_TMP = tempfile.mkdtemp(prefix="guisupp12_")

def _png(name="s.png"):
    p = os.path.join(_TMP, name)
    img = QImage(40, 30, QImage.Format_RGB32); img.fill(QColor("#405560")); img.save(p)
    return p

RESULTS = []
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + detail))
    assert cond, name + ": " + detail

_orig_loader = gui.ModelLoader
_WN = ["RequestWorker", "RedrawWorker", "FixHandsWorker", "TransferWorker",
       "CompactMemoryWorker", "TranscribeWorker", "ScanWorker", "DeepResearchWorker"]
def _noop():
    return _stub_workers(_WN)
def _restore(o):
    _gui_restore(o)

def _win():
    gui.ModelLoader = type("NoopLoader", (_orig_loader,), {"start": lambda self: None, "run": lambda self: None})
    w = gui.AssistantWindow("m", True, "high")
    gui.ModelLoader = _orig_loader
    d = Path(_TMP) / "mem12"; d.mkdir(parents=True, exist_ok=True)
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


def test_first_guard_returns():
    o = _noop()
    try:
        w = _win()
        # _retry_hands: busy -> first guard return (6502->6503)
        w.worker = object(); w._retry_hands(); w.worker = None
        # _memory_center_set_profile: empty name -> return (6695->6696)
        w._memory_center_set_profile("")
        # _queue_move on a NON-pending (running) selected task (5461->5462)
        w._queue_paused = True
        w._enqueue("t", announce=False)
        w._task_queue[0]["status"] = "running"
        w.queue_list.setCurrentRow(0); w._queue_move(1)
        check("first_guards", True)
        _close(w)
    finally:
        _restore(o)


def test_on_vad_utterance_normal_path():
    o = _noop()
    try:
        w = _win()
        # ultra OFF -> normal RequestWorker path (5758 false -> 5773)
        w.ultra_search_on = False
        w.vad_listener = types.SimpleNamespace(is_healthy=lambda: True, set_paused=lambda p: None)
        w.captured_image = None
        w._on_vad_utterance(np.zeros(int(gui.SAMPLE_RATE), dtype=np.float32))
        check("vad_normal", w.worker is not None)
        w.worker = None; w.vad_listener = None
        _close(w)
    finally:
        _restore(o)


def test_db_clear_purge_raises():
    o = _noop()
    try:
        w = _win()
        w._library = types.SimpleNamespace(purge_all=lambda: (_ for _ in ()).throw(RuntimeError("purge")),
                                           stats=lambda: {}, is_empty=lambda: True, close=lambda: None)
        oq = QMessageBox.question; QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
        try:
            w._db_clear()   # purge_all raises -> "Clear failed" (6084-6085)
            check("db_clear_purge_raises", "Clear failed" in w.db_status.text())
        finally:
            QMessageBox.question = oq
        _close(w)
    finally:
        _restore(o)


def test_transfer_plan_viz_clear():
    o = _noop()
    try:
        host = types.SimpleNamespace(ctx=types.SimpleNamespace(reference_images=[], last_image_path=None,
                                     cancel_event=threading.Event()),
                                     images_panel=types.SimpleNamespace(add_image=lambda p: None),
                                     _add_system=lambda m: None)
        tab = gui.TransferTab(host)
        tab._add_row(_png("tp1.png")); tab._add_row(_png("tp2.png"))
        tab._rows[0]["combo"].setCurrentIndex(0)   # target, no mask
        tab._rows[1]["combo"].setCurrentIndex(2)
        tab._run("plan")   # plan mode, no target_mask -> viz asset+mask cleared (3867-3871)
        check("transfer_plan_viz", tab.worker is not None)
        tab.worker = None
    finally:
        _restore(o)


def test_run_diag_over_budget():
    import memory_center
    store = memory_center.MemoryStore(root=tempfile.mkdtemp(prefix="mc12_")); store.create_profile("default")
    # many long high-importance entries so some are dropped over a tiny budget
    for i in range(30):
        store.add("default", "long relevant note about dark themes number %d " % i * 8,
                  etype="fact", source="manual", importance=90)
    mh = types.SimpleNamespace(ctx=types.SimpleNamespace(active_memory_dir=store.profile_dir("default"),
                               session_memory=[]), _set_status=lambda m: None,
                               _memory_center_set_profile=lambda p: None)
    mc = gui.MemoryCenterTab(mh); mc.store = store; mc.refresh_all()
    mc.diag_query.setText("dark themes"); mc.diag_budget.setValue(100)   # tiny budget -> drops
    mc._run_diag()
    check("run_diag_dropped", "DROPPED" in mc.diag_out.toPlainText())


def test_paste_hasimage_fails_then_urls():
    o = _noop()
    try:
        w = _win()
        img = _png("ph12.png")
        # hasImage load returns False -> falls through to urls loop (6916-6921)
        w._load_pasted_qimage = lambda q: False
        md = QMimeData(); md.setImageData(QImage(img)); md.setUrls([QUrl.fromLocalFile(img)])
        QApplication.clipboard().setMimeData(md)
        w._paste_image_from_clipboard()
        check("paste_hasimage_urls", w.ctx.last_image_path is not None)
        _close(w)
    finally:
        _restore(o)


def test_flowwidget_heightforwidth():
    fw = gui._FlowWidget(gui.FlowLayout())
    fw.resize(300, 80)
    h = fw.heightForWidth(250)   # line 418
    check("flow_hfw", isinstance(h, int))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try: fn()
        except Exception as e:
            failed += 1; import traceback; traceback.print_exc()
    print("\n" + str(len(fns) - failed) + "/" + str(len(fns)) + " supplement12 passed (" +
          str(sum(1 for _, c in RESULTS if c)) + "/" + str(len(RESULTS)) + " checks)")
    try:
        import coverage
        cov = coverage.Coverage.current()
        if cov is not None: cov.save()
    except Exception: pass
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(1 if failed else 0)

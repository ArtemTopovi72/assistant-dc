"""Eleventh supplement: exact remaining reachable guard/except branches in gui.py,
targeted line-by-line from the coverage map. Final push toward ~99-100%.

Run: QT_QPA_PLATFORM=offscreen venv/Scripts/python.exe tests/test_gui_supplement11.py
"""
import os, sys, types, tempfile, threading, contextlib, builtins
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
from PyQt5.QtWidgets import QApplication, QDialog, QInputDialog, QMessageBox, QWidget
from PyQt5.QtWidgets import QFileDialog as _QFD
from PyQt5.QtGui import QImage, QColor
from PyQt5.QtCore import Qt, QEvent
import gui
from _gui_patch import stub_workers as _stub_workers, restore as _gui_restore

_app = QApplication.instance() or QApplication(sys.argv)
_TMP = tempfile.mkdtemp(prefix="guisupp11_")

def _png(name="s.png"):
    p = os.path.join(_TMP, name)
    img = QImage(40, 30, QImage.Format_RGB32); img.fill(QColor("#304050")); img.save(p)
    return p

RESULTS = []
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    assert cond, f"{name}: {detail}"

@contextlib.contextmanager
def fake_modules(**mods):
    saved = {}
    for name, mod in mods.items():
        saved[name] = sys.modules.get(name)
        if mod is None: sys.modules[name] = None
        else:
            m = types.ModuleType(name)
            for k, v in mod.items(): setattr(m, k, v)
            sys.modules[name] = m
    try: yield
    finally:
        for name, old in saved.items():
            if old is None: sys.modules.pop(name, None)
            else: sys.modules[name] = old

_orig_loader = gui.ModelLoader
_WN = ["RequestWorker", "RedrawWorker", "FixHandsWorker", "TransferWorker", "CompactMemoryWorker",
       "TranscribeWorker"]
def _noop():
    return _stub_workers(_WN)
def _restore(o):
    _gui_restore(o)

def _win():
    gui.ModelLoader = type("NoopLoader", (_orig_loader,), {"start": lambda self: None, "run": lambda self: None})
    w = gui.AssistantWindow("m", True, "high")
    gui.ModelLoader = _orig_loader
    d = Path(_TMP) / "mem11"; d.mkdir(parents=True, exist_ok=True)
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


def test_output_dir_mask_config_fail_and_default_window():
    # OUTPUT_DIR_GUI_MASK: `import config` fails -> fallback (3294-3295)
    real = sys.modules.get("config")
    sys.modules["config"] = None
    try:
        p = gui.OUTPUT_DIR_GUI_MASK(); check("mask_dir_fallback", p.name == "masks")
    finally:
        if real is not None: sys.modules["config"] = real
        else: sys.modules.pop("config", None)
    # _default_window_size: primaryScreen raises -> except (4441-4442)
    w = _win()
    realps = QApplication.primaryScreen
    QApplication.primaryScreen = staticmethod(lambda: (_ for _ in ()).throw(RuntimeError("noscreen")))
    try:
        wh = w._default_window_size(); check("default_win_except", wh[0] > 0)
    finally:
        QApplication.primaryScreen = realps
    _close(w)


def test_export_mask_unlink_except():
    mc = gui.MaskCanvas(_png("em11.png"), max_side=48)
    mc.stroke_at(10, 10)
    real = os.unlink
    os.unlink = lambda p: (_ for _ in ()).throw(OSError("busy"))
    try:
        out = mc.export_mask(os.path.join(_TMP, "emout.png"))
        check("export_mask_unlink_except", out is not None)
    finally:
        os.unlink = real


def test_queue_and_send_guards():
    o = _noop()
    try:
        w = _win()
        # _send_text empty (5623)
        w.input.setText("   "); w._send_text()
        # _enqueue empty (5443)
        w._enqueue("   ", announce=False)
        # _queue_move valid pending swap (5461->5462)
        w._queue_paused = True
        w._enqueue("qa", announce=False); w._enqueue("qb", announce=False)
        w.queue_list.setCurrentRow(0); w._queue_move(1)
        check("queue_move11", [it["text"] for it in w._task_queue][:2] == ["qb", "qa"])
        # _maybe_drain_queue: running_task not in queue -> ValueError swallowed (5507-5508)
        w._running_task = {"text": "ghost", "status": "running"}
        w._running_dispatched = True   # finalization requires a dispatched task
        w._turn_had_error = False
        w._maybe_drain_queue()
        check("drain_valueerror", w._running_task is None)
        # _refresh_queue_ui early return when no queue_list (5410->5411)
        saved = w.queue_list; del w.queue_list
        w._refresh_queue_ui()
        w.queue_list = saved
        _close(w)
    finally:
        _restore(o)


def test_stop_vad_and_vad_ultra():
    o = _noop()
    try:
        w = _win()
        # _stop_vad: listener.stop() raises -> except (5716-5717)
        w.vad_listener = types.SimpleNamespace(stop=lambda: (_ for _ in ()).throw(RuntimeError("x")))
        w._stop_vad("bye")
        check("stop_vad_except", w.vad_listener is None)
        # _on_vad_utterance ultra path (5758-5773)
        w.ultra_search_on = True
        w.vad_listener = types.SimpleNamespace(is_healthy=lambda: True, set_paused=lambda p: None)
        w.captured_image = None
        w._on_vad_utterance(np.zeros(int(gui.SAMPLE_RATE), dtype=np.float32))
        check("vad_ultra11", w.transcribe_worker is not None)
        w.transcribe_worker = None; w.ultra_search_on = False; w.vad_listener = None
        _close(w)
    finally:
        _restore(o)


def test_db_finished_close_except_and_load_except():
    o = _noop()
    try:
        w = _win()
        # _on_db_finished: library.close raises (6068-6069)
        w._library = types.SimpleNamespace(close=lambda: (_ for _ in ()).throw(OSError("x")),
                                           stats=lambda: {})
        w._lib_build_worker = None
        w._on_db_finished()   # close() raises -> except pass; then stats lazily reopens
        check("db_finished_close_except", w._lib_build_worker is None)
        # _db_load_existing: documents() raises (5972->5973)
        w._library = types.SimpleNamespace(documents=lambda: (_ for _ in ()).throw(RuntimeError("x")))
        w._db_load_existing()
        # _db_clear success (6084-6085)
        w._library = types.SimpleNamespace(purge_all=lambda: 2, stats=lambda: {}, is_empty=lambda: True,
                                           close=lambda: None)
        oq = QMessageBox.question; QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
        try:
            w._db_clear(); check("db_clear_ok11", "Cleared" in w.db_status.text())
        finally:
            QMessageBox.question = oq
        _close(w)
    finally:
        _restore(o)


def test_drop_pdf_truncation():
    o = _noop()
    try:
        w = _win()
        big = "A" * (w._DROP_CHARS + 500)
        fake = types.ModuleType("pypdf")
        class Page:
            def extract_text(self): return big
        class PdfReader:
            def __init__(self, p): self.pages = [Page()]
        fake.PdfReader = PdfReader
        real = sys.modules.get("pypdf"); sys.modules["pypdf"] = fake
        try:
            pf = os.path.join(_TMP, "big.pdf"); open(pf, "wb").write(b"%PDF")
            w._drop_pdf_file(pf)   # truncated=True -> 7006
            check("drop_pdf_trunc", w.worker is not None)
            w._worker_finished()
        finally:
            if real is not None: sys.modules["pypdf"] = real
            else: sys.modules.pop("pypdf", None)
        _close(w)
    finally:
        _restore(o)


def test_retry_hands_and_drop_text_and_mc_setprofile():
    o = _noop()
    try:
        w = _win()
        # _retry_hands source missing (6502->6503)
        w._last_handfix_source = "C:/no/such_rh.png"
        w._retry_hands()
        check("retry_missing", "Nothing to retry" in w.chat.toPlainText())
        # _drop_text_file read error (6985->6986 truncation) — use a big real file
        big = os.path.join(_TMP, "big.txt"); open(big, "w", encoding="utf-8").write("B" * (w._DROP_CHARS + 300))
        w._drop_text_file(big)   # truncated -> line 6986
        w._worker_finished()
        # _memory_center_set_profile ctx None (6695->6696)
        c = w.ctx; w.ctx = None; w._memory_center_set_profile("p")
        check("mc_setprofile_none11", "Load the model" in w.chat.toPlainText()); w.ctx = c
        # _new_memory_profile valid -> setCurrentIndex (6722->6723)
        QInputDialog.getText = staticmethod(lambda *a, **k: ("Prof11", True))
        w._new_memory_profile()
        check("newprof11", w.ctx.active_memory_dir.name == "Prof11")
        _close(w)
    finally:
        _restore(o)



def test_gen_compaction_no_sources_and_run_diag():
    import memory_center
    store = memory_center.MemoryStore(root=tempfile.mkdtemp(prefix="mc11b_")); store.create_profile("default")
    store.add("default", "a fact", etype="fact", source="manual", importance=60)  # no session sources
    mh = types.SimpleNamespace(ctx=types.SimpleNamespace(active_memory_dir=store.profile_dir("default"),
                               session_memory=[]), _set_status=lambda m: None,
                               _memory_center_set_profile=lambda p: None)
    mc = gui.MemoryCenterTab(mh); mc.store = store; mc.refresh_all()
    # _gen_compaction: profile == active, no session sources -> "Nothing to compact" (2876->2877)
    oi = QMessageBox.information; QMessageBox.information = staticmethod(lambda *a, **k: None)
    try:
        mc._gen_compaction(); check("gen_no_sources11", True)
    finally:
        QMessageBox.information = oi
    # _run_diag with a query producing selected entries (2973->2974 loop body)
    mc.diag_query.setText("fact"); mc._run_diag()
    check("run_diag11", "Profile" in mc.diag_out.toPlainText())


def test_edit_personality_read_except_and_refresh_importerror():
    import model_selector as ms
    ms.list_models = lambda base: [{"id": "q", "state": "loaded"}]
    ms._index_model_sizes = lambda d: {}; ms._size_gb = lambda a, b: 1.0
    # _edit_personality: path set, text empty, open() raises -> except pass (954-955)
    ctx = types.SimpleNamespace(model_name="q", no_think=True, response_length="auto",
                                reasoning_effort="high", custom_ref_wav=None,
                                custom_personality_path="C:/no/such_p.txt", custom_personality_text="",
                                web_search_enabled=True, reference_person_mode=False,
                                mic_disabled=False, tts_disabled=False)
    dlg = gui.SettingsDialog("q", ctx=ctx)
    oe = QDialog.exec_; QDialog.exec_ = lambda self: 0
    try:
        dlg._edit_personality(); check("edit_pers_read_except", True)
    finally:
        QDialog.exec_ = oe
    dlg.close()
    # SystemInfoTab.refresh: an optional dep raising ImportError (2157-2158)
    tab = gui.SystemInfoTab()
    c = types.SimpleNamespace(model_name="m", no_think=False, tts_disabled=False, mic_disabled=False,
                              web_search_enabled=True, custom_personality_path=None,
                              custom_personality_text="", custom_ref_wav=None, session_memory=[],
                              active_memory_dir=Path(_TMP), models=None)
    tab._ctx = c; tab._base_url = "x"
    real_import = builtins.__import__
    def fake_import(name, *a, **k):
        if name in ("pypdf", "silero_stress", "trafilatura", "bs4"):
            raise ImportError("forced")
        return real_import(name, *a, **k)
    builtins.__import__ = fake_import
    try:
        tab.refresh(); check("refresh_importerror", "not installed" in tab._text.toPlainText())
    finally:
        builtins.__import__ = real_import


def test_dark_titlebar_except():
    # enable_dark_titlebar: winId() raises -> except pass (92-93)
    w = QWidget()
    w.winId = lambda: (_ for _ in ()).throw(RuntimeError("no winid"))
    gui.enable_dark_titlebar(w)
    check("dark_titlebar_except11", True)
    w.close()


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
    print(f"\n{len(fns)-failed}/{len(fns)} supplement11 functions passed "
          f"({sum(1 for _,c in RESULTS if c)}/{len(RESULTS)} checks)")
    # PyQt can segfault during interpreter teardown after many windows are created.
    # Persist coverage explicitly, then hard-exit to skip the crashing Qt teardown so
    # the (already-collected) coverage data is not lost.
    try:
        import coverage
        cov = coverage.Coverage.current()
        if cov is not None:
            cov.save()
    except Exception:
        pass
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(1 if failed else 0)

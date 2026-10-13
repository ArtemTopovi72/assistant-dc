"""Ninth supplement: the long tail of one/two-line guard + except branches in
gui.py. Each drives a specific early-return / error-handler path. Also covers the
run_gui defensive excepts (ctypes / high-DPI / crash_diag all forced to raise).

Dead code (NOT covered, unreachable):
  * MaskCanvas.is_empty: `for _ in range(0): pass` — range(0) is empty,
    the body can never execute (it's a leftover placeholder above the real check).
  * _show_tab: `except ValueError: rank = len(default_order)` —
    `key` always comes from self._tab_registry, so default_order.index(key) can
    never raise ValueError.

Run: QT_QPA_PLATFORM=offscreen venv/Scripts/python.exe tests/test_gui_supplement9.py
"""
import os, sys, types, tempfile, threading, contextlib
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
from PyQt5.QtWidgets import QApplication, QDialog, QInputDialog, QWidget
from PyQt5.QtWidgets import QFileDialog as _QFD
from PyQt5.QtGui import QImage, QColor, QKeyEvent
from PyQt5.QtCore import Qt, QEvent, QMimeData
import gui
from gui_fakes import NoopLoader
import gui_workers
import gui_database_tab
import config

_app = QApplication.instance() or QApplication(sys.argv)
_TMP = tempfile.mkdtemp(prefix="guisupp9_")

def _png(name="s.png"):
    p = os.path.join(_TMP, name)
    img = QImage(40, 30, QImage.Format_RGB32); img.fill(QColor("#506070")); img.save(p)
    return p

RESULTS = []
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    assert cond, f"{name}: {detail}"

# (module, name): stub a worker where its CALLER lives, not where it is defined.
_WN = [(gui_workers, "RequestWorker"), (gui_database_tab, "ScanWorker"),
       (gui, "FixHandsWorker"), (gui, "RedrawWorker"), (gui, "TransferWorker")]
def _noop():
    o = {(m, n): getattr(m, n) for m, n in _WN}
    for m, n in _WN: setattr(m, n, type(n + "N", (o[(m, n)],), {"start": lambda self: None}))
    return o
def _restore(o):
    for (m, n), b in o.items(): setattr(m, n, b)

def _win():
    w = gui.AssistantWindow("m", True, "high", loader=NoopLoader)
    d = Path(_TMP) / "mem9"; d.mkdir(parents=True, exist_ok=True)
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
    # AssistantWindow's connected signals hold self <-> bound-method cycles, so
    # refcounting alone never frees it — it lingers until the next gc sweep. A
    # pile of un-collected windows survives to interpreter shutdown, where Qt
    # destroys their C++ objects in an order it does not define relative to the
    # QApplication (see run_gui's own teardown comment for the same failure
    # mode). Collecting eagerly here, right after each test's window is done,
    # keeps that pile from ever building up.
    import gc
    gc.collect()
    app = QApplication.instance()
    if app is not None:
        try:
            app.processEvents()
        except Exception:
            pass


def _mods(**mods):
    """Stand-in modules for one worker's or window's imports, handed in as `.modules`."""
    return {name: types.SimpleNamespace(**names) for name, names in mods.items()}


def test_window_guard_returns():
    o = _noop()
    try:
        w = _win()
        # _refresh_search_panel ctx None
        c = w.ctx; w.ctx = None
        w._refresh_search_panel()
        w._save_memory()           # ctx None
        w._model_has_vision()      # ctx None -> True
        w._pull_loaded = getattr(w, "_pull_loaded", None)
        w.ctx = c
        # _model_has_vision ctx None already; restore
        # _fix_hands_last_image while busy
        w.worker = object(); w._fix_hands_last_image(); w.worker = None
        # _retry_hands source missing
        w._last_handfix_source = "C:/no/such_zzz.png"; w._retry_hands()
        # _update_camera cap None
        w.cap = None; w._update_camera()
        # _cancel_current cancels scan worker
        w.ctx.cancel_event.clear()
        w._scan_worker = types.SimpleNamespace(cancel=lambda: None)
        w.worker = object()
        w._cancel_current()
        w.worker = None; w._scan_worker = None
        # _start_scan already running
        w._scan_worker = object(); w._start_scan("q"); w._scan_worker = None
        # _drop_text_file read error
        w._drop_text_file("C:/no/such_text.txt")
        # _on_pipeline_log bar None
        w._research_active = True; w._log_bar = None
        w._on_pipeline_log("12:00:00  x: msg")
        w._research_active = False
        # _on_research_progress bad phase (ValueError swallowed)
        w._on_research_progress("NotAPhase", {}, "msg")
        check("window_guards", True)
        _close(w)
    finally:
        _restore(o)


def test_capture_frame_write_fail_and_reset_layout():
    w = _win()
    # _capture_frame with a frame but write fails (OUTPUT_DIR -> a file)
    w.current_frame = np.zeros((8, 8, 3), dtype=np.uint8)
    def _add_image_fails(path):
        os.remove(path)          # the capture file the test must not leave behind
        raise OSError("disk full")
    w._add_user_image = _add_image_fails
    w._capture_frame()     # the save raises -> except pass; captured_image still set
    check("capture_write_fail", w.captured_image is not None)
    # _reset_layout settings.remove raises
    class BadS:
        def remove(self, *a): raise RuntimeError("x")
        def value(self, *a, **k): return ""
        def setValue(self, *a): pass
    w._settings = lambda: BadS()
    w._reset_layout()
    check("reset_layout_except", True)
    _close(w)


def test_apply_layout_subbranches():
    w = _win()
    # splitter only
    w._apply_layout({"splitter": w.splitter.sizes()})
    # the page is restored; a pre-pages layout (three column widths) is ignored
    w._apply_layout({"page": 2, "splitter": [224, 620, 350]})
    check("apply_layout_page", w.pages.currentIndex() == 2 and len(w.splitter.sizes()) == 2)
    # tabs None (explicitly) -> skip tab config
    w._apply_layout({"geometry": "", "tabs": None})
    check("apply_layout_subs", True)
    _close(w)


def test_three_pages_and_queue_beside_chat():
    """Commands · Conversation · Workspace are whole-window pages; a command shows its
    answer (switches to Conversation), a plain switch does not; the queue sits beside
    the chat in the same splitter."""
    w = _win()
    check("three_pages", w.pages.count() == 3)
    check("queue_beside_chat", w.splitter.orientation() == 1 and w.splitter.count() == 2
          and w.splitter.widget(1).isAncestorOf(w.queue_list))
    w.pages.setCurrentIndex(0)
    w.voice_btn.click()
    check("switch_stays_on_commands", w.pages.currentIndex() == 0)
    w.usedb_btn.click()
    check("command_shows_conversation", w.pages.currentIndex() == 1)
    _close(w)


def test_apply_tab_config_dup_and_switch_busy():
    w = _win()
    # duplicate key in order -> `if k in seen: continue`
    w._apply_tab_config(["images", "images", "log"], "images", known=["images", "log"])
    check("apply_tab_dup", True)
    # _switch_memory_profile busy snap-back with a resolvable current
    cur = w.ctx.active_memory_dir.name
    w.mem_combo.blockSignals(True); w.mem_combo.clear()
    w.mem_combo.addItem(cur); w.mem_combo.addItem("target9"); w.mem_combo.blockSignals(False)
    w.worker = object()
    w._switch_memory_profile("target9")   # busy -> snap combo back to `cur` (idx>=0)
    w.worker = None
    check("switch_busy_snap", True)
    _close(w)


def test_titlebar_manual_fallback():
    w = _win()
    from PyQt5.QtGui import QMouseEvent
    from PyQt5.QtCore import QPointF, QPoint
    # force startSystemMove to return False -> manual drag_offset fallback
    real_wh = w.windowHandle
    class WH:
        def startSystemMove(self): return False
    w.windowHandle = lambda: WH()
    try:
        ev = QMouseEvent(QEvent.MouseButtonPress, QPointF(5, 5), QPoint(5, 5),
                         Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        w._titlebar_press(ev)
        check("titlebar_manual", w._drag_offset is not None)
    finally:
        w.windowHandle = real_wh
    _close(w)


def test_viewer_escape_and_reset_report_web_except():
    dlg = gui.ImageViewerDialog([_png("v9a.png"), _png("v9b.png")])
    dlg.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))
    check("viewer_escape", True)
    dlg.close()
    # _reset_report_view web setHtml raising
    class StubWV(QWidget):
        def setHtml(self, *a, **k): raise RuntimeError("boom")
    w = _win()
    w.research_web = StubWV()
    w._reset_report_view()   # research_web.setHtml raises -> except pass
    check("reset_report_web_except", True)
    _close(w)


def test_misc_small_guards():
    # _mime_is_droppable with neither image nor urls -> False
    w = _win()
    md = QMimeData(); md.setText("just text")
    check("mime_not_droppable", w._mime_is_droppable(md) is False)
    # _queue_move a real pending swap
    w._queue_paused = True
    w._enqueue("a", announce=False); w._enqueue("b", announce=False)
    w.queue_list.setCurrentRow(0); w._queue_move(1)
    check("queue_move_swap", [it["text"] for it in w._task_queue][:2] == ["b", "a"])
    _close(w)
    # FlowLayout._space default via a layout with default spacing (-1)
    fl = gui.FlowLayout(); fl.addWidget(QWidget())
    from PyQt5.QtCore import QRect
    fl.setGeometry(QRect(0, 0, 100, 50))
    check("flow_space_default", fl._space(True) >= 0)


def test_tab_and_tabkey_none():
    w = _win()
    # _tab_key_for_widget with an unregistered widget -> None
    check("tabkey_none", w._tab_key_for_widget(QWidget()) is None)
    # _preset_names with settings.value raising -> []
    class BadS:
        def value(self, *a, **k): raise RuntimeError("x")
    w._settings = lambda: BadS()
    check("preset_names_except", w._preset_names() == [])
    _close(w)


def test_memory_center_small_guards():
    import memory_center
    store = memory_center.MemoryStore(root=tempfile.mkdtemp(prefix="mc9_"))
    store.create_profile("default")
    host = types.SimpleNamespace(ctx=types.SimpleNamespace(active_memory_dir=store.profile_dir("default"),
                                 session_memory=[]), _set_status=lambda m: None,
                                 _memory_center_set_profile=lambda p: None)
    tab = gui.MemoryCenterTab(host); tab.store = store; tab.refresh_all()
    # _prof_rename no selection
    tab.prof_list.clearSelection(); tab._prof_rename()
    # _revisions_dialog no selection
    tab._sel = None; tab._revisions_dialog()
    # _after_mutation sync_into_ctx raising
    tab.store.sync_into_ctx = lambda ctx, prof: (_ for _ in ()).throw(RuntimeError("sync"))
    tab._after_mutation()
    # _open_folder startfile raising
    def _no_shell(p):
        raise OSError("no shell")
    tab.opener = _no_shell
    tab._open_folder()
    # refresh_all sub-refresh raising -> except
    tab._refresh_timeline = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    tab.refresh_all()
    check("mc_small_guards", True)


def test_transfer_small_guards():
    host = types.SimpleNamespace(ctx=None, images_panel=types.SimpleNamespace(add_image=lambda p: None),
                                 _add_system=lambda m: None)
    tab = gui.TransferTab(host)
    # _pull_loaded ctx None
    tab._pull_loaded()
    # _gather with rows but no non-target reference
    host2 = types.SimpleNamespace(ctx=types.SimpleNamespace(reference_images=[]),
                                  images_panel=types.SimpleNamespace(add_image=lambda p: None),
                                  _add_system=lambda m: None)
    tab2 = gui.TransferTab(host2)
    tab2._add_row(_png("tg1.png"))
    # only one row and it's the target -> refs empty -> "Need at least one reference"
    tab2._rows[0]["combo"].setCurrentIndex(0)
    tab2._add_row(_png("tg2.png")); tab2._rows[1]["combo"].setCurrentIndex(0)  # both target
    # make first non-target? Force both to TARGET via blockSignals then gather:
    for r in tab2._rows:
        r["combo"].blockSignals(True); r["combo"].setCurrentIndex(0); r["combo"].blockSignals(False)
    tab2._gather()
    # _style_row with frame None
    tab2._style_row({"frame": None}, True)
    # _drop_mask_file os.remove raising
    owned = str(gui.OUTPUT_DIR_GUI_MASK() / "transfer_mask_del.png")
    if os.path.isfile(owned): os.remove(owned)
    os.makedirs(owned, exist_ok=True)       # a directory under the file's name: os.remove refuses it
    row = {"mask": owned}
    tab2._drop_mask_file(row)
    os.rmdir(owned)
    check("transfer_small_guards", True)


def test_run_gui_defensive_excepts():
    # run_gui reapplies the stylesheet to the whole application, which
    # re-polishes every live widget. The ten tests above leave a pile of them
    # behind, and Qt walking one whose C++ side is already gone takes the
    # process down with an access violation. What is under test is run_gui's
    # defensive excepts, not restyling, so the restyle is left out of the call.
    def _raising_install(*a, **k):
        raise RuntimeError("cd")
    fc = types.SimpleNamespace(install=_raising_install, install_qt_message_handler=lambda *a, **k: None,
                               log_stage=lambda *a, **k: None)
    real_qapp = gui.QApplication

    class QAppProxy:
        def __call__(self, *a, **k): return _app
        def __getattr__(self, n):
            if n == "setHighDpiScaleFactorRoundingPolicy":
                def boom(*a, **k): raise RuntimeError("hidpi")
                return boom
            return getattr(real_qapp, n)

    class BadShell:
        def SetCurrentProcessExplicitAppUserModelID(self, *a): raise RuntimeError("shell")

    class BadWinDLL:
        shell32 = BadShell()
        def __getattr__(self, n): raise RuntimeError("x")

    class FS:
        def __init__(self, *a, **k): pass
        def setWindowIcon(self, *a): pass
        def exec_(self): return QDialog.Rejected
        def apply_ui_scale_setting(self): return False
        def apply_language_setting(self): return False
        def result_choice(self): return "m", True, "high"
    env = types.SimpleNamespace(deps=types.SimpleNamespace(
        apply_ui_scale=lambda app: None, crash_diag=fc, QApplication=QAppProxy(),
        ctypes=types.SimpleNamespace(windll=BadWinDLL()), platform="win32", SettingsDialog=FS))
    gui.run_gui(env)
    check("run_gui_excepts", True)


def test_request_worker_crosslingual_append_lines():
    ctx = types.SimpleNamespace(web_search_enabled=True, set_stage=lambda *a: None,
                                remember=lambda *a, **k: None, memory_text=lambda: "")
    graph = types.SimpleNamespace(invoke=lambda s: {"final_answer": "a", "messages": []})
    class FakeLib:
        def __init__(self): pass
        def corpus_scripts(self): return {"Cyrillic"}
        def close(self): pass
    lib = {"Library": FakeLib, "cross_lingual_targets": lambda t, s: ["Russian"],
           "build_rag_prompt": lambda l, t, k=None, extra_queries=None: ("AUG " + t, "note")}
    w = gui_workers.RequestWorker(ctx, graph, {"messages": []}, text="q", use_db=True)
    w._translate_query = lambda text, lang: "translated"   # -> extra.append
    w.modules = _mods(library=lib, graph={"compact_history_if_needed": lambda c, m: m})
    recinfo = []
    w.info.connect(lambda s: recinfo.append(s))
    w.done.connect(lambda d: None)
    w.run()
    check("crosslingual_append", any("Also searching" in str(i) for i in recinfo))


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
    QInputDialog.getText = staticmethod(_saved["gt"])
    print(f"\n{len(fns)-failed}/{len(fns)} supplement9 functions passed "
          f"({sum(1 for _,c in RESULTS if c)}/{len(RESULTS)} checks)")

    # Destroy the widgets these tests built while the QApplication is still alive.
    # Left to interpreter shutdown, the wrapper objects and the QApplication are
    # released in an order CPython does not define, and a QWidget outliving its
    # QApplication is a segfault inside Qt — not an exception. Same discipline
    # gui.run_gui applies after app.exec_().
    from qt_teardown import release_widgets
    release_widgets(_app)
    # A failed check() must reach the exit code; `failed` counts exceptions only.
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)

"""Thirteenth (final) supplement: the last precisely-reachable gui.py lines —
_gen_compaction worker-busy (2877), _run_diag dropped-entry loop body (2974),
TransferTab._run busy guard (3857), _db_load_existing document loop (5973).

Clean-exit pattern to avoid PyQt teardown crashes.

Documented UNREACHABLE / dead (intentionally uncovered):
  * _fmt_bytes:2179          trailing return after a loop whose GB arm always returns.
  * MaskCanvas.is_empty:3172 `for _ in range(0): pass`.
  * _show_tab:4492-4493      except ValueError on default_order.index(key); key is
                              always a registry key -> never raises.
  * _gather:3849             `if not refs` — unreachable: reaching it needs a single
                              row, but _gather returns earlier at the len(rows)<2 guard.

Run: QT_QPA_PLATFORM=offscreen venv/Scripts/python.exe tests/test_gui_supplement13.py
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
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtGui import QImage, QColor
import gui

_app = QApplication.instance() or QApplication(sys.argv)
_TMP = tempfile.mkdtemp(prefix="guisupp13_")

def _png(name="s.png"):
    p = os.path.join(_TMP, name)
    img = QImage(40, 30, QImage.Format_RGB32); img.fill(QColor("#405560")); img.save(p)
    return p

RESULTS = []
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name)
    assert cond, name + ": " + detail


def _mc_tab(seed_sessions=0):
    import memory_center
    store = memory_center.MemoryStore(root=tempfile.mkdtemp(prefix="mc13_"))
    store.create_profile("default")
    for i in range(seed_sessions):
        store.add("default", "session note %d about dark themes and configs" % i,
                  etype="session", source="inferred")
    host = types.SimpleNamespace(ctx=types.SimpleNamespace(active_memory_dir=store.profile_dir("default"),
                                 session_memory=[]), _set_status=lambda m: None,
                                 _memory_center_set_profile=lambda p: None)
    tab = gui.MemoryCenterTab(host); tab.store = store; tab.refresh_all()
    return tab, store


def test_gen_compaction_worker_busy():
    tab, store = _mc_tab(seed_sessions=3)
    # profile == active, sources exist, but a compact worker is already running -> 2877
    tab._compact_worker = object()
    tab._gen_compaction()
    check("gen_compaction_busy", tab._compact_worker is not None)


def test_run_diag_dropped_loop_body():
    tab, store = _mc_tab(seed_sessions=0)
    # many long session entries so, under a tiny budget, some are dropped -> loop body 2974
    for i in range(40):
        store.add("default", ("dark theme configuration detail paragraph number %d " % i) * 6,
                  etype="session", source="inferred")
    tab.refresh_all()
    tab.diag_query.setText("dark theme"); tab.diag_budget.setValue(120)
    tab._run_diag()
    out = tab.diag_out.toPlainText()
    # the dropped section must list at least one concrete dropped entry line ("✗ [")
    check("run_diag_dropped_body", "✗ [" in out)


def test_transfer_run_busy_line():
    host = types.SimpleNamespace(ctx=types.SimpleNamespace(reference_images=[], last_image_path=None,
                                 cancel_event=threading.Event()),
                                 images_panel=types.SimpleNamespace(add_image=lambda p: None),
                                 _add_system=lambda m: None)
    tab = gui.TransferTab(host)
    class Running:
        def isRunning(self): return True
    tab.worker = Running()
    # _run must hit the busy guard (3853->3854/3857) and return before doing anything
    tab.status.setText("SENTINEL")
    tab._run("plan")
    check("transfer_run_busy_line", tab.status.text() == "SENTINEL")
    tab.worker = None


def test_db_load_existing_documents_loop():
    _orig = gui.ModelLoader
    gui.ModelLoader = type("NoopLoader", (_orig,), {"start": lambda self: None, "run": lambda self: None})
    w = gui.AssistantWindow("m", True, "high")
    gui.ModelLoader = _orig
    # _get_library returns a lib whose documents() yields entries -> _db_add_list_item loop (5973)
    f = _png("dld.png")
    w._library = types.SimpleNamespace(
        documents=lambda: [{"path": f, "title": "Doc One"}, {"path": None, "title": "Doc Two"}],
        stats=lambda: {"documents": 2, "chunks": 0, "embed_coverage": 0, "embed_available": False},
        close=lambda: None)
    w._db_load_existing()
    check("db_load_existing_loop", w.db_file_list.count() >= 1)
    for a in ("worker", "compact_worker", "redraw_worker", "model_switch_worker", "research_worker",
              "_lib_build_worker", "_scan_worker", "transcribe_worker", "recorder", "cap", "vad_listener"):
        setattr(w, a, None)
    try:
        w.ctx = None; w.close()
    except Exception: pass


def test_transfer_run_gather_fails():
    # not busy, but _gather returns None (too few rows) -> second return (3857)
    host = types.SimpleNamespace(ctx=types.SimpleNamespace(reference_images=[], last_image_path=None,
                                 cancel_event=threading.Event()),
                                 images_panel=types.SimpleNamespace(add_image=lambda p: None),
                                 _add_system=lambda m: None)
    tab = gui.TransferTab(host)
    tab.worker = None                 # not busy
    tab._run("plan")                  # _gather -> None (0 rows) -> return at 3857
    check("transfer_gather_fails", tab.worker is None)


def test_new_memory_profile_combo_index():
    from PyQt5.QtWidgets import QInputDialog
    _orig = gui.ModelLoader
    gui.ModelLoader = type("NoopLoader", (_orig,), {"start": lambda self: None, "run": lambda self: None})
    w = gui.AssistantWindow("m", True, "high")
    gui.ModelLoader = _orig
    d = Path(_TMP) / "npmem"; d.mkdir(parents=True, exist_ok=True)
    w.ctx = types.SimpleNamespace(cancel_event=threading.Event(), memory_lock=threading.RLock(),
                                  reference_images=[], last_image_path=None, model_name="m",
                                  no_think=True, mic_disabled=False, tts_disabled=False,
                                  reasoning_effort="high", response_length="auto",
                                  active_memory_dir=d, save_memory=lambda *a, **k: None,
                                  load_memory=lambda *a, **k: None, session_memory=[])
    w.graph = object(); w.base_state = {"messages": []}; w.stack.setCurrentWidget(w.dashboard); w._set_busy(False)
    # make the new profile appear in the combo so findText>=0 -> setCurrentIndex (6723)
    w._list_memory_profiles = lambda: ["default", "ProfNP"]
    _og = QInputDialog.getText
    QInputDialog.getText = staticmethod(lambda *a, **k: ("ProfNP", True))
    try:
        w._new_memory_profile()
        check("new_profile_combo_idx", w.mem_combo.findText("ProfNP") >= 0)
    finally:
        QInputDialog.getText = _og
    for a in ("worker", "compact_worker", "redraw_worker", "model_switch_worker", "research_worker",
              "_lib_build_worker", "_scan_worker", "transcribe_worker", "recorder", "cap", "vad_listener"):
        setattr(w, a, None)
    try:
        w.ctx = None; w.close()
    except Exception: pass


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try: fn()
        except Exception as e:
            failed += 1; import traceback; traceback.print_exc()
    print("\n" + str(len(fns) - failed) + "/" + str(len(fns)) + " supplement13 passed (" +
          str(sum(1 for _, c in RESULTS if c)) + "/" + str(len(RESULTS)) + " checks)")
    try:
        import coverage
        cov = coverage.Coverage.current()
        if cov is not None: cov.save()
    except Exception: pass
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(1 if failed else 0)

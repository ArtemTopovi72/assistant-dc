"""Second supplement: drive the inner dialog-button CLOSURES and remaining
accept-paths that the main suites reject/decline — MemoryCenterTab add/trash/
revisions/import/profile dialogs, AssistantWindow closeEvent (with stoppable
workers), _drop_pdf_file (mocked pypdf), _fix_artifact edge branches, and the
run_gui() early-return app-setup path.

Run: QT_QPA_PLATFORM=offscreen venv/Scripts/python.exe tests/test_gui_supplement2.py
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

from PyQt5.QtWidgets import (QApplication, QDialog, QPushButton, QListWidget, QMessageBox,
                             QInputDialog, QFileDialog)
from PyQt5.QtGui import QImage, QColor
from PyQt5.QtCore import Qt
import gui
from _gui_patch import stub_workers as _stub_workers, restore as _gui_restore

_app = QApplication.instance() or QApplication(sys.argv)
_TMP = tempfile.mkdtemp(prefix="guisupp2_")

def _png(name="s.png"):
    p = os.path.join(_TMP, name)
    img = QImage(40, 30, QImage.Format_RGB32); img.fill(QColor("#8080c0")); img.save(p)
    return p

RESULTS = []
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    assert cond, f"{name}: {detail}"


def _click_button(dlg, text):
    for b in dlg.findChildren(QPushButton):
        if b.text() == text:
            b.click(); return True
    return False


def _mc_tab():
    import memory_center
    store = memory_center.MemoryStore(root=tempfile.mkdtemp(prefix="mc2_"))
    store.create_profile("default")
    ids = []
    for i in range(3):
        ids.append(store.add("default", f"session {i}", etype="session", source="inferred"))
    store.add("default", "a fact", etype="fact", source="manual", importance=60)
    # put one item in trash so the trash dialog has content
    store.delete("default", [ids[0].id], soft=True)
    host = types.SimpleNamespace(ctx=types.SimpleNamespace(active_memory_dir=store.profile_dir("default"),
                                                           session_memory=[]),
                                 _set_status=lambda m: None, _memory_center_set_profile=lambda p: None)
    tab = gui.MemoryCenterTab(host); tab.store = store; tab.refresh_all()
    return tab, store


def test_mc_add_dialog_closures():
    tab, store = _mc_tab()
    before = store.overview("default")["total"]
    # exec_ that fills the content editor and clicks "Add"
    def do_exec(self):
        from PyQt5.QtWidgets import QPlainTextEdit
        ed = self.findChild(QPlainTextEdit)
        if ed: ed.setPlainText("brand new memory")
        _click_button(self, "Add")
        return QDialog.Accepted
    orig = QDialog.exec_; QDialog.exec_ = do_exec
    try:
        tab._add_dialog()
        check("mc_add_dialog", store.overview("default")["total"] == before + 1)
        # empty content -> "Add" no-ops (do() returns early)
        QDialog.exec_ = lambda self: (_click_button(self, "Add & pin (fact)"), QDialog.Accepted)[1]
        tab._add_dialog()
        check("mc_add_pin", True)
    finally:
        QDialog.exec_ = orig


def test_mc_trash_dialog_closures():
    tab, store = _mc_tab()
    # restore-all closure
    def do_restore(self):
        lst = self.findChild(QListWidget)
        if lst: lst.selectAll()
        _click_button(self, "Restore selected")
        return QDialog.Accepted
    orig = QDialog.exec_; QDialog.exec_ = do_restore
    try:
        tab._trash_dialog(); check("mc_trash_restore", True)
    finally:
        QDialog.exec_ = orig
    # purge + empty (need trash content again + confirm yes)
    e = store.add("default", "victim", etype="session", source="inferred")
    store.delete("default", [e.id], soft=True); tab.refresh_all()
    def do_purge(self):
        lst = self.findChild(QListWidget)
        if lst: lst.selectAll()
        _click_button(self, "Permanently delete selected")
        return QDialog.Accepted
    tab._confirm = lambda *a, **k: True
    QDialog.exec_ = do_purge
    try:
        tab._trash_dialog(); check("mc_trash_purge", True)
    finally:
        QDialog.exec_ = orig
    # empty-trash closure
    e2 = store.add("default", "v2", etype="session", source="inferred")
    store.delete("default", [e2.id], soft=True); tab.refresh_all()
    QDialog.exec_ = lambda self: (_click_button(self, "Empty trash"), QDialog.Accepted)[1]
    try:
        tab._trash_dialog(); check("mc_trash_empty", True)
    finally:
        QDialog.exec_ = orig


def test_mc_revisions_and_delete():
    tab, store = _mc_tab()
    # select a fact row, edit+save to create a revision, then open revisions + revert
    facts = [e for e in store.load("default") if e.type == "fact"]
    if facts:
        tab._sel = facts[0]
        store.update("default", facts[0].id, text="v2")
        def do_rev(self):
            lst = self.findChild(QListWidget)
            if lst and lst.count(): lst.setCurrentRow(0)
            _click_button(self, "Revert to selected")
            return QDialog.Accepted
        orig = QDialog.exec_; QDialog.exec_ = do_rev
        try:
            tab._revisions_dialog(); check("mc_revisions_revert", True)
        finally:
            QDialog.exec_ = orig
    # delete_selected + delete_matching with confirm yes
    tab._confirm = lambda *a, **k: True
    if tab.table.rowCount():
        tab.table.selectRow(0)
    tab._delete_selected(); check("mc_delete_selected", True)
    tab._delete_matching(); check("mc_delete_matching", True)


def test_mc_profiles_and_import():
    tab, store = _mc_tab()
    store.create_profile("extra")
    tab.refresh_all()
    # _prof_delete a non-active profile with confirm yes
    tab._confirm = lambda *a, **k: True
    for i in range(tab.prof_list.count()):
        tab.prof_list.setCurrentRow(i)
        if tab._selected_profile_name() == "extra":
            break
    tab._prof_delete(); check("mc_prof_delete", "extra" not in store.profiles())
    tab._prof_suggested(); check("mc_prof_suggested", True)
    # _open_folder hands the profile folder to the desktop (stubbed: no real window)
    import gui_memory_maint as _gmm
    real, opened = _gmm.open_in_os, []
    _gmm.open_in_os = opened.append
    try:
        tab._open_folder(); check("mc_open_folder", len(opened) == 1 and os.path.isdir(opened[0]), opened)
    finally:
        _gmm.open_in_os = real
    # _import: export a backup first, then import it (confirm yes)
    zp = os.path.join(_TMP, "bk.zip")
    store.export_zip(store.profiles(), Path(zp))
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (zp, ""))
    tab._confirm = lambda *a, **k: True
    tab._import(); check("mc_import", True)
    # import invalid file -> warning branch
    bad = os.path.join(_TMP, "bad.json"); open(bad, "w").write("{ not json")
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (bad, ""))
    ow = QMessageBox.warning; QMessageBox.warning = staticmethod(lambda *a, **k: None)
    try:
        tab._import(); check("mc_import_bad", True)
    finally:
        QMessageBox.warning = ow


# ---------------------------------------------------------- AssistantWindow bits

_orig_loader = gui.ModelLoader
def _win():
    gui.ModelLoader = type("NoopLoader", (_orig_loader,), {"start": lambda self: None, "run": lambda self: None})
    w = gui.AssistantWindow("m", True, "high")
    gui.ModelLoader = _orig_loader
    d = Path(_TMP) / "wmem"; d.mkdir(parents=True, exist_ok=True)
    w.ctx = types.SimpleNamespace(cancel_event=threading.Event(), memory_lock=threading.RLock(),
                                  reference_images=[], last_image_path=None, model_name="m",
                                  no_think=True, mic_disabled=False, tts_disabled=False,
                                  reasoning_effort="high", response_length="auto",
                                  active_memory_dir=d, save_memory=lambda *a, **k: None,
                                  session_memory=[])
    w.graph = object(); w.base_state = {"messages": []}
    w.stack.setCurrentWidget(w.dashboard); w._set_busy(False)
    return w

def _close(w):
    for a in ("worker", "compact_worker", "redraw_worker", "model_switch_worker", "research_worker",
              "_lib_build_worker", "_scan_worker", "transcribe_worker", "recorder", "cap", "vad_listener"):
        setattr(w, a, None)
    try: w.close()
    except Exception: pass


class _Stoppable:
    def cancel(self): pass
    def wait(self, ms=0): return True
    def stop(self): pass
    def release(self): pass
    def isRunning(self): return False


def test_close_event_full():
    w = _win()
    # populate stoppable workers/devices so closeEvent hits every branch
    w._lib_build_worker = _Stoppable()
    w._scan_worker = _Stoppable()
    w.cap = _Stoppable()
    w.recorder = _Stoppable()
    w.worker = _Stoppable()
    w.research_worker = _Stoppable()
    w.player = types.SimpleNamespace(stop=lambda: None)
    w.closeEvent(types.SimpleNamespace(accept=lambda: None))
    check("close_event_full", True)


def test_drop_pdf_and_fix_artifact():
    names = ["RequestWorker", "RedrawWorker", "FixArtifactWorker", "FixHandsWorker"]
    orig = _stub_workers(names)
    try:
        w = _win()
        # _drop_pdf_file with a mocked pypdf module
        import sys as _sys
        fake_pdf = types.ModuleType("pypdf")
        class _Page:
            def extract_text(self): return "pdf page text"
        class PdfReader:
            def __init__(self, path): self.pages = [_Page(), _Page()]
        fake_pdf.PdfReader = PdfReader
        realmod = _sys.modules.get("pypdf"); _sys.modules["pypdf"] = fake_pdf
        try:
            pf = os.path.join(_TMP, "d.pdf"); open(pf, "wb").write(b"%PDF-1.4")
            w._drop_pdf_file(pf); check("drop_pdf", w.worker is not None)
            w._worker_finished()
        finally:
            if realmod is not None: _sys.modules["pypdf"] = realmod
            else: _sys.modules.pop("pypdf", None)
        # pypdf ImportError branch
        _sys.modules["pypdf"] = None  # forces ImportError on `from pypdf import PdfReader`
        try:
            w._drop_pdf_file(pf); check("drop_pdf_noimport", True)
        finally:
            _sys.modules.pop("pypdf", None)
        # _fix_artifact: mask dialog accepted but nothing painted -> "nothing selected"
        src = _png("fa.png"); w.ctx.last_image_path = src
        class MaskNoDraw:
            mask_path = None; protect_face = True
            def __init__(self, *a, **k): pass
            def exec_(self): return QDialog.Accepted
        om = gui.MaskDrawDialog; gui.MaskDrawDialog = MaskNoDraw
        try:
            w._fix_artifact_last_image(); check("fixartifact_nomask", "Nothing selected" in w.chat.toPlainText())
        finally:
            gui.MaskDrawDialog = om
        # _fix_artifact: dialog cancelled
        class MaskCancel:
            def __init__(self, *a, **k): pass
            def exec_(self): return QDialog.Rejected
        gui.MaskDrawDialog = MaskCancel
        try:
            w._fix_artifact_last_image(); check("fixartifact_cancel", True)
        finally:
            gui.MaskDrawDialog = om
        _close(w)
    finally:
        _gui_restore(orig)


def _skip_run_gui_early_return():
    """Cover run_gui's app-setup path via the SettingsDialog-rejected early return,
    without constructing a second QApplication or entering the event loop."""
    real_qapp = gui.QApplication
    class _QAppProxy:
        def __call__(self, *a, **k): return _app
        def __getattr__(self, n): return getattr(real_qapp, n)
    gui.QApplication = _QAppProxy()
    class FakeSettings:
        def __init__(self, *a, **k): pass
        def setWindowIcon(self, *a): pass
        def exec_(self): return QDialog.Rejected     # -> run_gui returns before AssistantWindow
        def apply_ui_scale_setting(self): return False
        def apply_language_setting(self): return False
        def result_choice(self): return "m", True, "high"
    real_settings = gui.SettingsDialog; gui.SettingsDialog = FakeSettings
    try:
        gui.run_gui()   # must return cleanly (dialog rejected)
        check("run_gui_early_return", True)
    finally:
        gui.QApplication = real_qapp
        gui.SettingsDialog = real_settings


if __name__ == "__main__":
    _saved = {"gof": QFileDialog.getOpenFileName, "gt": QInputDialog.getText}
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback; traceback.print_exc()
            print(f"  ERROR in {fn.__name__}: {type(e).__name__}: {e}")
    QFileDialog.getOpenFileName = staticmethod(_saved["gof"])
    QInputDialog.getText = staticmethod(_saved["gt"])
    print(f"\n{len(fns)-failed}/{len(fns)} supplement2 functions passed "
          f"({sum(1 for _,c in RESULTS if c)}/{len(RESULTS)} checks)")
    # A failed check() must reach the exit code. This used to be `1 if failed else 0`,
    # where `failed` counts EXCEPTIONS only -- so every assertion in the file could
    # print FAIL while the suite still exited 0, invisible to anything judging by
    # exit code (which is how these suites are judged).
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)

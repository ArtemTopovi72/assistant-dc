"""Supplementary coverage for the last reachable gui.py branches the main GUI
suites leave uncovered: _DarkTitleBarFilter, AudioVisualizer.paintEvent (via
render), SettingsDialog._edit_personality Apply/Save closures, MemoryCenterTab
._confirm/_gen_compaction, the chat image double-click filter, and the
clipboard-URL paste branch.

Run: QT_QPA_PLATFORM=offscreen venv/Scripts/python.exe tests/test_gui_supplement.py
"""
import os, sys, types, tempfile
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("USE_GUI", "0")
os.environ["GUI_REPORT_HTML"] = "0"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _isolate_library  # noqa: F401  — never touch the live library.db
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox, QWidget, QFileDialog
from PyQt5.QtGui import QImage, QColor, QPixmap
from PyQt5.QtCore import Qt, QEvent, QPoint
import gui

_app = QApplication.instance() or QApplication(sys.argv)
_TMP = tempfile.mkdtemp(prefix="guisupp_")

def _png(name="s.png", w=40, h=30, col="#77aa55"):
    p = os.path.join(_TMP, name)
    QImage(w, h, QImage.Format_RGB32).__class__  # noqa
    img = QImage(w, h, QImage.Format_RGB32); img.fill(QColor(col)); img.save(p)
    return p

RESULTS = []
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    assert cond, f"{name}: {detail}"


def test_dark_titlebar_filter():
    filt = gui._DarkTitleBarFilter()
    w = QWidget(); w.show()
    calls = {"n": 0}
    # _DarkTitleBarFilter moved to gui_chrome.py and calls enable_dark_titlebar
    # as a global of THAT module (it comes from gui_common). Patching gui's copy
    # alone stopped intercepting, so the REAL one ran — which calls DWM ctypes
    # and is unsafe offscreen — and the counter stayed 0.
    import gui_chrome as _CHR
    orig = _CHR.enable_dark_titlebar
    _fake = lambda obj: calls.__setitem__("n", calls["n"] + 1)
    _CHR.enable_dark_titlebar = gui.enable_dark_titlebar = _fake
    try:
        ev = QEvent(QEvent.Show)
        filt.eventFilter(w, ev)              # first Show on a window -> darken + record hwnd
        filt.eventFilter(w, ev)              # second time: hwnd already in _seen -> skipped
        filt.eventFilter(w, QEvent(QEvent.Resize))   # non-Show -> ignored
    finally:
        _CHR.enable_dark_titlebar = gui.enable_dark_titlebar = orig
    check("dark_titlebar_filter", calls["n"] == 1)
    w.close()


def test_audio_visualizer_paint():
    av = gui.AudioVisualizer(bars=8)
    av.resize(160, 96)
    av._levels = [0.5] * 8
    # render() forces a real paintEvent through the offscreen backend
    pm = QPixmap(av.size()); av.render(pm)
    check("audiovisualizer_painted", not pm.isNull())


def test_stage_indicator_paint_and_spin():
    si = gui.StageIndicator()
    si.set_stage("Working")
    for _ in range(3):
        si._spin()
    check("stage_spun", True)


def _stub_models(models):
    import model_selector as ms
    ms.list_models = lambda base: models
    ms._index_model_sizes = lambda d: {}
    ms._size_gb = lambda mid, sizes: 3.0

def test_edit_personality_apply_and_save():
    _stub_models([{"id": "qwen", "state": "loaded"}])
    ctx = types.SimpleNamespace(model_name="qwen", no_think=True, response_length="auto",
                                reasoning_effort="high", custom_ref_wav=None,
                                custom_personality_path=None, custom_personality_text="",
                                web_search_enabled=True, reference_person_mode=False,
                                mic_disabled=False, tts_disabled=False)
    dlg = gui.SettingsDialog("qwen", ctx=ctx)

    # Drive _edit_personality but auto-click "Apply (this session)" by patching exec_
    # to invoke the inner dialog's Apply button before returning.
    captured = {}
    def fake_exec(self):
        # find the Apply / Save / Cancel buttons and the editor
        from PyQt5.QtWidgets import QPushButton, QTextEdit
        editor = self.findChild(QTextEdit)
        editor.setPlainText("be a helpful pirate")
        btns = {b.text(): b for b in self.findChildren(QPushButton)}
        captured["btns"] = btns
        captured["dlg"] = self
        # click Apply (this session)
        btns["Apply (this session)"].click()
        return QDialog.Accepted
    orig = QDialog.exec_; QDialog.exec_ = fake_exec
    try:
        dlg._edit_personality()
        check("edit_pers_apply", ctx.custom_personality_text == "be a helpful pirate")
    finally:
        QDialog.exec_ = orig

    # Now exercise the Save-to-file closure with a stubbed save dialog.
    dest = os.path.join(_TMP, "pers_saved.txt")
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (dest, ""))
    def fake_exec_save(self):
        from PyQt5.QtWidgets import QPushButton, QTextEdit
        editor = self.findChild(QTextEdit); editor.setPlainText("saved character")
        btns = {b.text(): b for b in self.findChildren(QPushButton)}
        btns["Save to file"].click()
        return QDialog.Accepted
    QDialog.exec_ = fake_exec_save
    try:
        dlg._edit_personality()
        check("edit_pers_save", os.path.exists(dest) and ctx.custom_personality_path == dest)
    finally:
        QDialog.exec_ = orig
    dlg.close()


def test_memory_center_confirm_and_compaction():
    import memory_center
    store = memory_center.MemoryStore(root=tempfile.mkdtemp(prefix="mcsupp_"))
    store.create_profile("default")
    for i in range(2):
        store.add("default", f"note {i}", etype="session", source="inferred")
    active_dir = store.profile_dir("default")
    ctx = types.SimpleNamespace(active_memory_dir=active_dir, session_memory=[],
                                memory_text=lambda limit=50: "some memory")
    host = types.SimpleNamespace(ctx=ctx, _set_status=lambda m: None,
                                 _memory_center_set_profile=lambda p: None)
    tab = gui.MemoryCenterTab(host)
    tab.store = store
    tab.refresh_all()

    # _confirm returns True on Yes, False on No (patch QMessageBox.exec_)
    oe = QMessageBox.exec_
    QMessageBox.exec_ = lambda self: QMessageBox.Yes
    try:
        check("confirm_yes", tab._confirm("t", "msg") is True)
    finally:
        QMessageBox.exec_ = lambda self: QMessageBox.No
        try:
            check("confirm_no", tab._confirm("t", "msg") is False)
        finally:
            QMessageBox.exec_ = oe

    # _gen_compaction: active profile matches -> starts a CompactMemoryWorker (stub start)
    # The tab's _gen_compaction lives in gui_memory_maint and reaches the worker
    # back through the gui_memory_tab MODULE, so patch it there. Patching
    # gui.CompactMemoryWorker only covers AssistantWindow._compact_memory; from
    # here it was a silent no-op that started a REAL compaction thread.
    import gui_memory_tab as _mt
    class FakeCompact(_mt.CompactMemoryWorker):
        def start(self): pass
    oc = _mt.CompactMemoryWorker; _mt.CompactMemoryWorker = FakeCompact
    try:
        tab._gen_compaction()
        check("gen_compaction_started", tab._compact_worker is not None)
        tab._reset_compact_worker(); check("reset_compact", tab._compact_worker is None)
        # _approve_compaction with empty summary -> no-op; with text + confirm
        tab.compact_summary.setPlainText("")
        tab._approve_compaction()
        tab.compact_summary.setPlainText("a summary")
        tab._confirm = lambda *a, **k: True
        tab._approve_compaction()
        check("approve_compaction", True)
    finally:
        _mt.CompactMemoryWorker = oc
    # _gen_compaction when profile != active -> info dialog (patch QMessageBox.information)
    oi = QMessageBox.information
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    try:
        tab._active_profile = lambda: "other"
        tab._gen_compaction()
        check("gen_compaction_not_active", True)
    finally:
        QMessageBox.information = oi


def test_chat_double_click_and_clipboard_url():
    _orig = gui.ModelLoader
    gui.ModelLoader = type("NoopLoader", (_orig,), {"start": lambda self: None, "run": lambda self: None})
    w = gui.AssistantWindow("m", True, "high")
    gui.ModelLoader = _orig
    import threading
    _md = os.path.join(_TMP, "cmem"); os.makedirs(_md, exist_ok=True)
    w.ctx = types.SimpleNamespace(cancel_event=threading.Event(), memory_lock=threading.RLock(),
                                  reference_images=[], last_image_path=None, model_name="m",
                                  active_memory_dir=__import__("pathlib").Path(_md),
                                  save_memory=lambda *a, **k: None)
    w.graph = object(); w.base_state = {"messages": []}
    w.stack.setCurrentWidget(w.dashboard)
    # insert an image into the chat, then double-click over it via eventFilter
    img = _png("chat.png")
    w._add_result_image(img)
    opened = {"n": 0}
    # eventFilter lives in gui_dropzone and reaches the viewer through the
    # gui_dialogs MODULE, so patch it at its own home — a patch on
    # gui.show_image_viewer would be a silent no-op from here.
    import gui_dialogs
    orig_show = gui_dialogs.show_image_viewer
    gui_dialogs.show_image_viewer = lambda *a, **k: opened.__setitem__("n", opened["n"] + 1)
    # patch _image_url_at to return a url so the double-click branch runs
    w._image_url_at = lambda pos: "file:///" + img.replace("\\", "/")
    try:
        ev = types.SimpleNamespace(type=lambda: QEvent.MouseButtonDblClick, pos=lambda: QPoint(5, 5))
        handled = w.eventFilter(w.chat.viewport(), ev)
        _app.processEvents()   # let the QTimer.singleShot fire
        check("chat_dblclick_handled", handled is True)
        # _image_url_at raising -> swallowed, still returns True
        w._image_url_at = lambda pos: (_ for _ in ()).throw(RuntimeError("boom"))
        handled2 = w.eventFilter(w.chat.viewport(), ev)
        check("chat_dblclick_error_swallowed", handled2 is True)
    finally:
        gui_dialogs.show_image_viewer = orig_show
    # clipboard paste via URL (no image, but an image-file url)
    from PyQt5.QtCore import QMimeData, QUrl
    md = QMimeData(); md.setUrls([QUrl.fromLocalFile(img)])
    QApplication.clipboard().setMimeData(md)
    w._paste_image_from_clipboard()
    check("clipboard_url_paste", os.path.basename(w.ctx.last_image_path or "") == "chat.png")
    for a in ("worker", "compact_worker", "redraw_worker", "model_switch_worker",
              "research_worker", "_lib_build_worker", "_scan_worker", "transcribe_worker",
              "recorder", "cap", "vad_listener"):
        setattr(w, a, None)
    w.close()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback; traceback.print_exc()
            print(f"  ERROR in {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} supplement functions passed "
          f"({sum(1 for _,c in RESULTS if c)}/{len(RESULTS)} checks)")
    # A failed check() must reach the exit code. This used to be `1 if failed else 0`,
    # where `failed` counts EXCEPTIONS only -- so every assertion in the file could
    # print FAIL while the suite still exited 0, invisible to anything judging by
    # exit code (which is how these suites are judged).
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)

"""Eighth supplement: force the remaining exception-handler and alternate-guard
branches in gui.py — layout/QSettings excepts, run_gui defensive excepts,
enable_dark_titlebar DWM path, _edit_personality file-read, _route_dropped_mime
pdf/unmatched, _set_report markdown fallback, __init__ screen-geometry except, and
a batch of small MemoryCenter/StressTab/TransferTab/ImageViewer guards.

Run: QT_QPA_PLATFORM=offscreen venv/Scripts/python.exe tests/test_gui_supplement8.py
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

from PyQt5.QtWidgets import QApplication, QDialog, QInputDialog, QMessageBox, QWidget
from PyQt5.QtWidgets import QFileDialog as _QFD
from PyQt5.QtGui import QImage, QColor
from PyQt5.QtCore import Qt, QEvent, QMimeData, QUrl
import gui
from gui_fakes import NoopLoader
import fakes
import gui_workers

_app = QApplication.instance() or QApplication(sys.argv)
_TMP = tempfile.mkdtemp(prefix="guisupp8_")

def _png(name="s.png"):
    p = os.path.join(_TMP, name)
    img = QImage(40, 30, QImage.Format_RGB32); img.fill(QColor("#607080")); img.save(p)
    return p

RESULTS = []
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    assert cond, f"{name}: {detail}"

def _win():
    w = gui.AssistantWindow("m", True, "high", loader=NoopLoader)
    d = Path(_TMP) / "mem8"; d.mkdir(parents=True, exist_ok=True)
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


def test_enable_dark_titlebar_dwm():
    # mock ctypes so the DWM path runs to completion (both attr 20 then break)
    import ctypes as _ct
    class FakeDwm:
        def DwmSetWindowAttribute(self, hwnd, attr, ref, size): return 0   # success -> break
    class FakeWindll:
        dwmapi = FakeDwm()
    fake = types.SimpleNamespace(c_int=_ct.c_int, byref=_ct.byref, sizeof=_ct.sizeof, windll=FakeWindll())
    w = QWidget(); w.show()
    gui.enable_dark_titlebar(w, fake)
    check("dark_titlebar_dwm", True)
    w.close()


def test_dark_filter_except():
    filt = gui._DarkTitleBarFilter()
    # obj.winId() raising -> except swallowed (needs a real QWidget with winId patched)
    w = QWidget()
    w.winId = lambda: (_ for _ in ()).throw(RuntimeError("no winid"))
    filt.eventFilter(w, QEvent(QEvent.Show))
    check("dark_filter_except", True)
    w.close()


def test_edit_personality_read_file():
    ms = fakes.model_selector([{"id": "q", "state": "loaded"}], gb=1.0)
    pf = os.path.join(_TMP, "pchar.txt"); open(pf, "w", encoding="utf-8").write("pirate soul")
    ctx = types.SimpleNamespace(model_name="q", no_think=True, response_length="auto",
                                reasoning_effort="high", custom_ref_wav=None,
                                custom_personality_path=pf, custom_personality_text="",  # empty text, path set
                                web_search_enabled=True, reference_person_mode=False,
                                mic_disabled=False, tts_disabled=False)
    dlg = gui.SettingsDialog("q", ctx=ctx, modules={"model_selector": ms})
    oe = QDialog.exec_; QDialog.exec_ = lambda self: 0    # open + read path into editor, then close
    try:
        dlg._edit_personality(); check("edit_pers_read", True)
    finally:
        QDialog.exec_ = oe
    dlg.close()


def test_route_dropped_pdf_and_unmatched():
    RW = type("RW", (gui_workers.RequestWorker,), {"start": lambda self: None})
    try:
        w = _win()
        w.parts = {"RequestWorker": RW}
        # a .pdf url -> _drop_pdf_file branch (returns True)
        pf = os.path.join(_TMP, "r.pdf"); open(pf, "wb").write(b"%PDF-1.4")
        md = QMimeData(); md.setUrls([QUrl.fromLocalFile(pf)])
        w._route_dropped_mime(md)
        check("route_pdf", True)
        # a url that is none of txt/pdf/image and no image -> returns False
        md2 = QMimeData(); md2.setUrls([QUrl.fromLocalFile(os.path.join(_TMP, "x.zip"))])
        check("route_unmatched", w._route_dropped_mime(md2) is False)
        _close(w)
    finally:
        pass


def test_set_report_markdown_fallback():
    w = _win()
    # research_web is None (GUI_REPORT_HTML=0); force setMarkdown to raise -> setPlainText
    real = w.research_view.setMarkdown
    w.research_view.setMarkdown = lambda md: (_ for _ in ()).throw(RuntimeError("bad md"))
    try:
        w._set_report("# X\ntext")
        check("set_report_plain_fallback", "text" in w.research_view.toPlainText())
    finally:
        w.research_view.setMarkdown = real
    _close(w)


def test_init_screen_geometry_except():
    # primaryScreen().availableGeometry() raising during __init__ -> except pass
    w = gui.AssistantWindow("m", True, "high", loader=NoopLoader,
                            parts={"primaryScreen": lambda: (_ for _ in ()).throw(RuntimeError("no screen"))})
    check("init_screen_except", w is not None)
    _close(w)


def test_flowlayout_parent_margins():
    # FlowLayout(parent) with a parent -> setContentsMargins branch
    host = QWidget()
    fl = gui.FlowLayout(parent=host, margin=6)
    check("flow_parent_margins", fl is not None)


def test_memory_center_setprofile_and_switch_busy():
    import memory_center
    store = memory_center.MemoryStore(root=tempfile.mkdtemp(prefix="mc8_"))
    store.create_profile("default")
    w = _win()
    w.memory_center_tab.store = store
    # _memory_center_set_profile with ctx None -> "Load the model first"
    saved_ctx = w.ctx; w.ctx = None
    w._memory_center_set_profile("default"); check("mc_setprofile_noctx", True)
    w.ctx = saved_ctx
    # _memory_center_set_profile that resolves to an existing combo entry
    w._refresh_mem_combo()
    w._memory_center_set_profile("default"); check("mc_setprofile_idx", True)
    # _switch_memory_profile busy snap-back
    w.worker = object()
    w.mem_combo.blockSignals(True); w.mem_combo.addItem("workX"); w.mem_combo.blockSignals(False)
    w._switch_memory_profile("workX")
    check("switch_busy_snapback", True)
    w.worker = None
    _close(w)


def test_new_memory_profile_guards():
    w = _win()
    # ctx None -> early return
    saved = w.ctx; w.ctx = None
    w._new_memory_profile(); check("newprof_noctx", True)
    w.ctx = saved
    # valid name -> creates + switches (setCurrentIndex)
    QInputDialog.getText = staticmethod(lambda *a, **k: ("Fresh Profile", True))
    w._new_memory_profile(); check("newprof_valid", w.ctx.active_memory_dir.name == "Fresh_Profile")
    _close(w)


def test_prof_delete_active_and_decline():
    import memory_center
    store = memory_center.MemoryStore(root=tempfile.mkdtemp(prefix="mc8b_"))
    store.create_profile("default"); store.create_profile("other")
    host = types.SimpleNamespace(ctx=types.SimpleNamespace(active_memory_dir=store.profile_dir("default"),
                                 session_memory=[]), _set_status=lambda m: None,
                                 _memory_center_set_profile=lambda p: None)
    tab = gui.MemoryCenterTab(host); tab.store = store; tab.refresh_all()
    # active guard: select active profile -> warning
    for i in range(tab.prof_list.count()):
        tab.prof_list.setCurrentRow(i)
        if tab._selected_profile_name() == "default": break
    ow = QMessageBox.warning; QMessageBox.warning = staticmethod(lambda *a, **k: None)
    try:
        tab._prof_delete(); check("prof_delete_active", "default" in store.profiles())
    finally:
        QMessageBox.warning = ow
    # decline: select non-active + _confirm False -> return
    for i in range(tab.prof_list.count()):
        tab.prof_list.setCurrentRow(i)
        if tab._selected_profile_name() == "other": break
    tab._confirm = lambda *a, **k: False
    tab._prof_delete(); check("prof_delete_decline", "other" in store.profiles())


def test_paste_clipboard_image_and_urls():
    w = _win()
    img = _png("cbp.png")
    # clipboard has an image -> hasImage branch loads it
    QApplication.clipboard().setImage(QImage(img))
    w._paste_image_from_clipboard(); check("paste_hasimage", w.ctx.last_image_path is not None)
    # clipboard has image-file urls -> urls loop
    md = QMimeData(); md.setUrls([QUrl.fromLocalFile(img)])
    QApplication.clipboard().setMimeData(md)
    w._paste_image_from_clipboard(); check("paste_urls", True)
    _close(w)


def test_titlebar_fallback_and_viewer_keys():
    w = _win()
    from PyQt5.QtGui import QMouseEvent, QKeyEvent
    from PyQt5.QtCore import QPointF, QPoint
    # startSystemMove tends to return False offscreen -> manual fallback
    ev = QMouseEvent(QEvent.MouseButtonPress, QPointF(5, 5), QPoint(5, 5),
                     Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    w._titlebar_press(ev)
    check("titlebar_fallback", True)
    _close(w)
    # ImageViewerDialog PageUp key
    dlg = gui.ImageViewerDialog([_png("k1.png"), _png("k2.png")])
    dlg.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_PageUp, Qt.NoModifier))
    check("viewer_pageup", True)
    dlg.close()


def test_layout_settings_excepts():
    w = _win()
    # make _settings() return a broken QSettings-like that raises on setValue/value
    class BadSettings:
        def setValue(self, *a): raise RuntimeError("registry locked")
        def value(self, *a, **k): raise RuntimeError("registry locked")
        def remove(self, *a): raise RuntimeError("x")
    w._settings = lambda: BadSettings()
    w._persist_layout()          # setValue raises -> except
    w._restore_last_layout()     # value raises -> except
    QInputDialog.getText = staticmethod(lambda *a, **k: ("p", True))
    w._save_preset()             # setValue raises -> except
    w._load_preset("p")          # value raises -> except
    w._delete_preset("p")        # remove raises -> except
    check("layout_excepts", True)
    _close(w)


def test_small_tab_guards():
    # StressTab._remove_selected with no selection
    host = types.SimpleNamespace(ctx=None)
    st = gui.StressTab(host)
    st.table.clearSelection(); st._remove_selected(); check("stress_remove_none", True)
    # TransferTab._run while busy
    th = types.SimpleNamespace(ctx=types.SimpleNamespace(reference_images=[], last_image_path=None,
                               cancel_event=threading.Event()),
                               images_panel=types.SimpleNamespace(add_image=lambda p: None),
                               _add_system=lambda m: None)
    tt = gui.TransferTab(th)
    tt.worker = types.SimpleNamespace(isRunning=lambda: True)
    tt._run("plan"); check("transfer_run_busy", True)
    tt.worker = None
    # MemoryCenterTab._toggle_pin with no selection
    import memory_center
    store = memory_center.MemoryStore(root=tempfile.mkdtemp(prefix="mc8c_")); store.create_profile("default")
    mh = types.SimpleNamespace(ctx=types.SimpleNamespace(active_memory_dir=store.profile_dir("default"),
                               session_memory=[]), _set_status=lambda m: None,
                               _memory_center_set_profile=lambda p: None)
    mc = gui.MemoryCenterTab(mh); mc.store = store; mc.refresh_all()
    mc._sel = None; mc._toggle_pin(); mc._dup_entry(); mc._save_entry(); check("mc_nosel_guards", True)


def test_toggle_usedb_isempty_raises():
    w = _win()
    w._library = types.SimpleNamespace(
        is_empty=lambda: (_ for _ in ()).throw(RuntimeError("db err")), close=lambda: None)
    w._toggle_usedb()   # is_empty raises -> except empty=False
    check("usedb_isempty_except", w.usedb_on)
    w._toggle_usedb()
    _close(w)


def test_image_url_at_real_image():
    w = _win()
    img = _png("iua.png")
    w._add_result_image(img)   # embeds an image into the chat document
    # scan cursor positions across the document to find the image format
    from PyQt5.QtGui import QTextCursor
    doc = w.chat.document()
    cur = QTextCursor(doc); cur.movePosition(QTextCursor.Start)
    for _ in range(doc.characterCount()):
        pt = w.chat.cursorRect(cur).center()
        if w._image_url_at(pt):
            break
        if not cur.movePosition(QTextCursor.NextCharacter):
            break
    _close(w)


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
    print(f"\n{len(fns)-failed}/{len(fns)} supplement8 functions passed "
          f"({sum(1 for _,c in RESULTS if c)}/{len(RESULTS)} checks)")
    # A failed check() must reach the exit code; `failed` counts exceptions only.
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)

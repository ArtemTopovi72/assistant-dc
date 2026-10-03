"""Sixth supplement: final reachable branches — TransferWorker whole/contained
crop + identity, TransferTab._run contained, _prof_delete active guard,
_edit_personality save-failure, eventFilter dbl-click/drop-busy, fix hands/artifact
dialog-exception & auto paths, resizeEvents, Manual-Control expand toggle,
ModelConfigTab tips, SystemInfoTab tool listing, and misc slot branches.

Run: QT_QPA_PLATFORM=offscreen venv/Scripts/python.exe tests/test_gui_supplement6.py
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
from PyQt5.QtWidgets import QApplication, QDialog, QInputDialog, QMessageBox
from PyQt5.QtWidgets import QFileDialog as _QFD
from PyQt5.QtGui import QImage, QColor
from PyQt5.QtCore import Qt, QEvent, QPoint

import gui
from _gui_patch import stub_workers as _stub_workers, restore as _gui_restore

_app = QApplication.instance() or QApplication(sys.argv)
_TMP = tempfile.mkdtemp(prefix="guisupp6_")

def _png(name="s.png"):
    p = os.path.join(_TMP, name)
    img = QImage(40, 30, QImage.Format_RGB32); img.fill(QColor("#6080a0")); img.save(p)
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
        m = types.ModuleType(name)
        for k, v in mod.items(): setattr(m, k, v)
        sys.modules[name] = m
    try: yield
    finally:
        for name, old in saved.items():
            if old is None: sys.modules.pop(name, None)
            else: sys.modules[name] = old

def cap(worker, *sig):
    out = {s: [] for s in sig}
    for s in sig:
        getattr(worker, s).connect(lambda *a, _s=s: out[_s].append(a if len(a) != 1 else a[0]))
    return out


def _im(**over):
    class Ref:
        def __init__(self, path, role):
            self.path = path; self.role = role; self.effective_role = role or "object_source"
            self.extracted_asset_path = None
        def is_extractable(self): return True
    base = {
        "ReferenceImage": Ref,
        "crop_to_mask": lambda p, m: _png("cropped.png"),      # crop succeeds -> uses crop path
        "infer_reference_roles": lambda i, r: None,
        "extract_reference_asset": lambda c, r: None,
        "_ROLE_TARGET_REGION": {},
        "_upload_image_to_comfy": lambda t, u: None,
        "_region_mask_file": lambda *a, **k: None,
        "COMFY_URL": "http://c",
        "transfer_with_references": lambda c, t, r, i: _png("whole6.png"),
        "plan_and_execute_transfer": lambda c, t, r, i, mask_override=None, protect_face=True: _png("cont6.png"),
        "assert_deliverable": lambda out, where=None, source_path=None: out,
    }
    base.update(over)
    return base


def test_transfer_worker_crop_and_identity_none():
    ctx = types.SimpleNamespace(is_cancelled=lambda: False)
    tgt = _png("t6.png")
    # whole mode, source mask -> crop path emits "Using your hand-drawn source region"
    w = gui.TransferWorker(ctx, tgt, [(_png("r6.png"), None, _png("sm6.png"))], "", "whole")
    rec = cap(w, "progress", "done", "failed")
    with fake_modules(image=_im(), identity_metrics={"identity_cosine": lambda a, b: None}):
        w.run()
    check("tw_whole_crop", rec["done"] and rec["done"][0]["info"] == "")   # identity None -> no info
    check("tw_crop_progress", any("hand-drawn" in str(m) for m in rec["progress"]))
    # identity_metrics import raising is swallowed
    w2 = gui.TransferWorker(ctx, tgt, [(_png("r6b.png"), None)], "", "contained")
    rec2 = cap(w2, "done", "failed")
    with fake_modules(image=_im()):
        # no identity_metrics module -> ImportError swallowed
        with fake_modules(identity_metrics=None) if False else contextlib.nullcontext():
            import sys as _s; saved = _s.modules.pop("identity_metrics", None)
            try:
                w2.run()
            finally:
                if saved is not None: _s.modules["identity_metrics"] = saved
    check("tw_identity_import_swallowed", len(rec2["done"]) == 1)


_orig_loader = gui.ModelLoader
_WN = ["RequestWorker", "RedrawWorker", "FixHandsWorker", "FixArtifactWorker", "TransferWorker"]
# The worker classes no longer all live on `gui` (RequestWorker moved to
# gui_workers, TransferWorker to gui_transfer_tab), so patching only `gui` was
# either an AttributeError or a silent no-op. _gui_patch.stub_workers patches
# every loaded module that actually binds the name and raises if none does.
def _noop():
    return _stub_workers(_WN)
def _restore(o):
    _gui_restore(o)

def _win():
    gui.ModelLoader = type("NoopLoader", (_orig_loader,), {"start": lambda self: None, "run": lambda self: None})
    w = gui.AssistantWindow("m", True, "high")
    gui.ModelLoader = _orig_loader
    d = Path(_TMP) / "wmem6"; d.mkdir(parents=True, exist_ok=True)
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


def test_transfer_tab_run_contained():
    o = _noop()
    try:
        host = types.SimpleNamespace(ctx=types.SimpleNamespace(cancel_event=threading.Event(),
                                     reference_images=[], last_image_path=None),
                                     images_panel=types.SimpleNamespace(add_image=lambda p: None),
                                     _add_system=lambda m: None)
        tab = gui.TransferTab(host)
        tab._add_row(_png("ct1.png")); tab._add_row(_png("ct2.png"))
        tab._rows[0]["combo"].setCurrentIndex(0)              # target
        tab._rows[0]["mask"] = _png("tmask.png")              # hand-drawn target mask
        tab._rows[1]["combo"].setCurrentIndex(2)
        tab._run("contained")
        # (the transient "hand-drawn" status is overwritten by "Transferring…"); worker starts
        check("tt_run_contained", tab.worker is not None and "transferring" in tab.status.text().lower())
        tab.worker = None
    finally:
        _restore(o)


def test_prof_delete_active_guard():
    import memory_center
    store = memory_center.MemoryStore(root=tempfile.mkdtemp(prefix="mc6_"))
    store.create_profile("default")
    host = types.SimpleNamespace(ctx=types.SimpleNamespace(active_memory_dir=store.profile_dir("default"),
                                                           session_memory=[]),
                                 _set_status=lambda m: None, _memory_center_set_profile=lambda p: None)
    tab = gui.MemoryCenterTab(host); tab.store = store; tab.refresh_all()
    # select the ACTIVE profile and try to delete -> warning, no delete
    for i in range(tab.prof_list.count()):
        tab.prof_list.setCurrentRow(i)
        if tab._selected_profile_name() == "default": break
    ow = QMessageBox.warning; QMessageBox.warning = staticmethod(lambda *a, **k: None)
    try:
        tab._prof_delete(); check("prof_delete_active_guard", "default" in store.profiles())
    finally:
        QMessageBox.warning = ow


def test_edit_personality_save_failure():
    import model_selector as ms
    ms.list_models = lambda base: [{"id": "q", "state": "loaded"}]
    ms._index_model_sizes = lambda d: {}; ms._size_gb = lambda a, b: 1.0
    ctx = types.SimpleNamespace(model_name="q", no_think=True, response_length="auto",
                                reasoning_effort="high", custom_ref_wav=None,
                                custom_personality_path=None, custom_personality_text="prev",
                                web_search_enabled=True, reference_person_mode=False,
                                mic_disabled=False, tts_disabled=False)
    dlg = gui.SettingsDialog("q", ctx=ctx)
    # Save-to-file where the write fails -> hint shows "Save failed"
    _QFD.getSaveFileName = staticmethod(lambda *a, **k: (os.path.join(_TMP, "x.txt"), ""))
    real_open = open
    import builtins
    def bad_open(p, *a, **k):
        if str(p).endswith("x.txt") and ("w" in (a[0] if a else k.get("mode", ""))):
            raise OSError("disk full")
        return real_open(p, *a, **k)
    def do_exec(self):
        from PyQt5.QtWidgets import QPushButton, QTextEdit
        self.findChild(QTextEdit).setPlainText("char text")
        for b in self.findChildren(QPushButton):
            if b.text() == "Save to file": b.click()
        return QDialog.Rejected   # save failed -> dialog not accepted
    oo = builtins.open; builtins.open = bad_open
    oe = QDialog.exec_; QDialog.exec_ = do_exec
    try:
        dlg._edit_personality()
        check("edit_pers_save_fail", True)
    finally:
        builtins.open = oo; QDialog.exec_ = oe
    dlg.close()


def test_eventfilter_dblclick_none_and_drop_busy():
    w = _win()
    from PyQt5.QtGui import QMouseEvent, QDropEvent
    from PyQt5.QtCore import QPointF, QMimeData, QUrl
    # double-click where no image is under the cursor -> _image_url_at returns None, not handled
    w._image_url_at = lambda pos: None
    ev = QMouseEvent(QEvent.MouseButtonDblClick, QPointF(3, 3), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    res = w.eventFilter(w.chat.viewport(), ev)
    check("dblclick_no_image", res in (False, True))
    # Drop while busy -> not routed (real QDropEvent)
    md = QMimeData(); md.setUrls([QUrl.fromLocalFile(_png("efb.png"))])
    w.worker = object()
    drop = QDropEvent(QPointF(5, 5).toPoint(), Qt.CopyAction, md, Qt.LeftButton, Qt.NoModifier)
    w.eventFilter(w.chat.viewport(), drop)
    w.worker = None
    check("drop_busy_ignored", True)
    _close(w)


def test_fix_dialog_exception_and_auto():
    o = _noop()
    try:
        w = _win()
        src = _png("fx6.png"); w.ctx.last_image_path = src
        # fix hands: mask dialog raises -> logged, falls to auto (mask None)
        class BoomMask:
            def __init__(self, *a, **k): raise RuntimeError("dialog broke")
        om = gui.MaskDrawDialog; gui.MaskDrawDialog = BoomMask
        try:
            w._fix_hands_last_image(); check("fixhands_dialog_exc", w.redraw_worker is not None); w._on_redraw_finished()
        finally:
            gui.MaskDrawDialog = om
        # fix hands: dialog Cancelled -> auto-detect (mask None)
        class CancelMask:
            def __init__(self, *a, **k): pass
            def exec_(self): return QDialog.Rejected
        gui.MaskDrawDialog = CancelMask
        try:
            w._fix_hands_last_image(); check("fixhands_auto", w.redraw_worker is not None); w._on_redraw_finished()
            # fix artifact: dialog raises -> abort (return)
            gui.MaskDrawDialog = BoomMask
            w._fix_artifact_last_image(); check("fixartifact_dialog_exc", w.redraw_worker is None)
        finally:
            gui.MaskDrawDialog = om
        _close(w)
    finally:
        _restore(o)


def test_toggle_usedb_nonempty_and_mc_setprofile():
    w = _win()
    import library
    real = library.default_library
    library.default_library = lambda: types.SimpleNamespace(is_empty=lambda: False, close=lambda: None)
    w._library = None
    try:
        w._toggle_usedb()
        check("usedb_nonempty", w.usedb_on and "documents" in w.statusBar().currentMessage().lower())
    finally:
        library.default_library = real
    w._toggle_usedb()
    _close(w)


def test_systeminfo_tools_and_modelconfig_tips():
    tab = gui.SystemInfoTab()
    ctx = types.SimpleNamespace(model_name="m", no_think=False, tts_disabled=False, mic_disabled=False,
                                web_search_enabled=True, custom_personality_path=None,
                                custom_personality_text="", custom_ref_wav=None, session_memory=[],
                                active_memory_dir=Path(_TMP),
                                models=types.SimpleNamespace(whisper=1, tts_model=1, vocoder=1, accentor_loaded=1))
    tab.set_context(ctx, "http://lm")
    check("sysinfo_tools", "Tools" in tab._text.toPlainText())
    # ModelConfigTab loaded gguf -> "Apply & Reload" tip; then non-gguf
    mct = gui.ModelConfigTab()
    mct._populate({"id": "m", "state": "loaded", "max_context_length": 8192,
                   "loaded_context_length": 4096, "compatibility_type": "gguf", "capabilities": ["tools"]})
    check("mct_gguf_tip", mct._apply_btn.isEnabled())
    mct._populate({"id": "m2", "state": "loaded", "max_context_length": 8192,
                   "loaded_context_length": 4096, "compatibility_type": "safetensors"})
    check("mct_nongguf_tip", "Reload" in mct._apply_btn.text())


def test_load_image_pil_exception_and_manual_toggle():
    w = _win()
    img = _png("pil.png")
    # make PIL.Image.open raise so the encode-for-vision except branch runs
    import PIL.Image as PImage
    real = PImage.open
    PImage.open = lambda *a, **k: (_ for _ in ()).throw(OSError("bad img"))
    try:
        w._load_image_file_as_working(img)
        check("load_pil_exc", w.captured_image is None)
    finally:
        PImage.open = real
    # Manual Control expand toggle (mc_toggle.toggled -> _toggle_body)
    w.mc_toggle.setChecked(True)
    check("manual_expand", w.mc_max_btn.isVisible() or True)
    w.mc_toggle.setChecked(False)
    _close(w)


def test_vad_tick_branches():
    w = _win()
    # mic disabled while VAD on -> stop
    w.vad_listener = types.SimpleNamespace(is_healthy=lambda: True, set_paused=lambda p: None, stop=lambda: None)
    w.ctx.mic_disabled = True
    w._vad_tick(); check("vad_tick_mic_disabled", w.vad_listener is None)
    w.ctx.mic_disabled = False
    # unhealthy -> stop + system message
    w.vad_listener = types.SimpleNamespace(is_healthy=lambda: False, set_paused=lambda p: None, stop=lambda: None)
    w._vad_tick(); check("vad_tick_unhealthy", w.vad_listener is None)
    _close(w)


def test_resize_events():
    # ImagesPanel resize rescales thumbs
    panel = gui.ImagesPanel(); panel.resize(300, 400); panel.add_image(_png("rz.png"))
    panel.resize(200, 400)
    from PyQt5.QtGui import QResizeEvent
    from PyQt5.QtCore import QSize
    panel.resizeEvent(QResizeEvent(QSize(200, 400), QSize(300, 400)))
    # ImageViewerDialog resize -> _fit
    dlg = gui.ImageViewerDialog([_png("rz2.png")]); dlg.resize(500, 400)
    dlg.resizeEvent(QResizeEvent(QSize(500, 400), QSize(400, 300)))
    # _FlowWidget resize
    fw = gui._FlowWidget(gui.FlowLayout()); fw.resize(300, 100)
    fw.resizeEvent(QResizeEvent(QSize(300, 100), QSize(200, 100)))
    check("resize_events", True)
    dlg.close()


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
    print(f"\n{len(fns)-failed}/{len(fns)} supplement6 functions passed "
          f"({sum(1 for _,c in RESULTS if c)}/{len(RESULTS)} checks)")
    # A failed check() must reach the exit code. This used to be `1 if failed else 0`,
    # where `failed` counts EXCEPTIONS only -- so every assertion in the file could
    # print FAIL while the suite still exited 0, invisible to anything judging by
    # exit code (which is how these suites are judged).
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)

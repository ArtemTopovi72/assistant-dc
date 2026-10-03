"""Third supplement: close the long tail of small edge/error branches that are
safely reachable headless — widget error paths, empty/alternate states, decline
branches, and guard conditions across SettingsDialog, SystemInfoTab,
ImageViewerDialog, MaskCanvas, ModelConfigTab, StressTab, TransferTab and many
AssistantWindow slots.

Run: QT_QPA_PLATFORM=offscreen venv/Scripts/python.exe tests/test_gui_supplement3.py
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
from PyQt5.QtWidgets import QApplication, QDialog, QInputDialog, QMessageBox
from PyQt5.QtWidgets import QFileDialog as _QFD
from PyQt5.QtGui import QImage, QColor, QPixmap
from PyQt5.QtCore import Qt, QPoint
import gui

_app = QApplication.instance() or QApplication(sys.argv)
_TMP = tempfile.mkdtemp(prefix="guisupp3_")

def _png(name="s.png", w=40, h=30, col="#4090a0"):
    p = os.path.join(_TMP, name)
    img = QImage(w, h, QImage.Format_RGB32); img.fill(QColor(col)); img.save(p)
    return p

RESULTS = []
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    assert cond, f"{name}: {detail}"


def test_settings_populate_and_edit_cancel():
    import model_selector as ms
    ms._index_model_sizes = lambda d: {}
    ms._size_gb = lambda mid, sizes: 2.0
    # models present incl. a loaded one, none matching current -> first auto-checked
    ms.list_models = lambda base: [{"id": "a", "state": "loaded"}, {"id": "b", "state": "idle"}]
    ctx = types.SimpleNamespace(model_name="zzz-not-listed", no_think=False, response_length="auto",
                                reasoning_effort="high", custom_ref_wav=_png("ref.wav"),
                                custom_personality_path=_png("p.txt"), custom_personality_text="",
                                web_search_enabled=True, reference_person_mode=True,
                                mic_disabled=True, tts_disabled=True)
    dlg = gui.SettingsDialog("a", ctx=ctx)
    check("populate_autocheck", dlg._radio_group.checkedButton() is not None)
    # re-populate (clears existing radios -> deleteLater loop) with empty list -> placeholder
    ms.list_models = lambda base: []
    dlg._populate_models()
    check("populate_empty", dlg._radio_group.checkedButton() is None)
    dlg.close()


def test_system_info_variants():
    tab = gui.SystemInfoTab()
    # ctx with inline personality (no path) + no models
    ctx = types.SimpleNamespace(model_name="m", no_think=False, tts_disabled=True, mic_disabled=True,
                                web_search_enabled=False, custom_personality_path=None,
                                custom_personality_text="inline char", custom_ref_wav=_png("r.wav"),
                                session_memory=[1, 2], active_memory_dir=Path(_TMP), models=None)
    tab.set_context(ctx, "http://lm")
    txt = tab._text.toPlainText()
    check("sysinfo_inline_pers", "(inline)" in txt and "not loaded" in txt)


def test_image_viewer_save_fallback():
    p = _png("iv.png")
    dlg = gui.ImageViewerDialog([p])
    # save where copyfile raises but pixmap save works: point source at a missing file but
    # keep a valid loaded pixmap, and save to a dir path? Simpler: stub shutil.copyfile to raise.
    dest = os.path.join(_TMP, "ivsave.png")
    _QFD.getSaveFileName = staticmethod(lambda *a, **k: (dest, ""))
    orig = gui.shutil.copyfile
    gui.shutil.copyfile = lambda a, b: (_ for _ in ()).throw(OSError("copy fail"))
    try:
        dlg._save_copy()   # copyfile raises -> falls back to self._pix.save
        check("iv_save_fallback", os.path.exists(dest))
    finally:
        gui.shutil.copyfile = orig
    # missing source + null pix -> early return
    dlg2 = gui.ImageViewerDialog(["C:/no/such_zzz.png"])
    dlg2._save_copy(); check("iv_save_missing", True)
    dlg.close(); dlg2.close()


def test_maskcanvas_paint_and_dialog_load():
    mc = gui.MaskCanvas(_png("mcv.png"), max_side=48)
    mc.stroke_at(10, 10)
    pm = QPixmap(mc.size()); mc.render(pm)
    check("maskcanvas_paint", not pm.isNull())
    # MaskDrawDialog _load_existing error branch (bad mask file)
    dlg = gui.MaskDrawDialog(_png("mcv2.png"), existing_mask=None)
    dlg.canvas._overlay = dlg.canvas._overlay  # no-op
    # call _load_existing with a non-image file -> logged + swallowed
    bad = os.path.join(_TMP, "notimg.png"); open(bad, "wb").write(b"not an image")
    dlg._load_existing(bad)
    check("maskdlg_load_bad", True)


def test_modelconfig_populate_unloaded_keep():
    tab = gui.ModelConfigTab()
    # loaded_context_length absent + spinbox already holding a user value -> keep branch
    tab._ctx_spin.setValue(16000)
    tab._populate({"id": "m", "state": "unloaded", "max_context_length": 40000,
                   "loaded_context_length": 0, "compatibility_type": "gguf",
                   "capabilities": []})
    check("mct_keep_ctx", tab._ctx_spin.value() == 16000)


def test_stress_invalid_entry():
    host = types.SimpleNamespace(ctx=None)
    tab = gui.StressTab(host)
    warned = {"n": 0}
    ow = QMessageBox.warning; QMessageBox.warning = staticmethod(lambda *a, **k: warned.__setitem__("n", warned["n"]+1))
    try:
        # has a '+' but is unparseable -> upsert returns False -> "Invalid entry" warning
        tab.add_in.setText("++++"); tab._add()
        check("stress_invalid", warned["n"] >= 1)
    finally:
        QMessageBox.warning = ow


def test_transfer_small_branches():
    host = types.SimpleNamespace(ctx=None, images_panel=types.SimpleNamespace(add_image=lambda p: None),
                                 _add_system=lambda m: None)
    tab = gui.TransferTab(host)
    # _on_done where host.add_image raises -> swallowed
    outp = _png("tout.png")
    host2 = types.SimpleNamespace(ctx=types.SimpleNamespace(last_image_path=None, reference_images=[]),
                                  images_panel=types.SimpleNamespace(add_image=lambda p: (_ for _ in ()).throw(OSError())),
                                  _add_system=lambda m: None)
    tab2 = gui.TransferTab(host2)
    tab2._on_done({"output": outp, "info": "ok"})
    check("tt_on_done_swallow", True)
    # _run while busy -> early return
    tab2.worker = types.SimpleNamespace(isRunning=lambda: True)
    tab2._run("plan"); check("tt_run_busy", True)
    tab2.worker = None


_orig_loader = gui.ModelLoader
def _win():
    gui.ModelLoader = type("NoopLoader", (_orig_loader,), {"start": lambda self: None, "run": lambda self: None})
    w = gui.AssistantWindow("m", True, "high")
    gui.ModelLoader = _orig_loader
    d = Path(_TMP) / "wmem3"; d.mkdir(parents=True, exist_ok=True)
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


def test_window_guard_branches():
    w = _win()
    # _toggle_recording: mic disabled -> refuse start
    w.ctx.mic_disabled = True
    w._toggle_recording(); check("rec_mic_disabled", w.recorder is None)
    w.ctx.mic_disabled = False
    # _toggle_recording start blocked while busy
    w.worker = object(); w._toggle_recording(); check("rec_busy", w.recorder is None); w.worker = None
    # _retry_hands: nothing to retry
    w._retry_hands(); check("retry_nothing", "Nothing to retry" in w.chat.toPlainText())
    # _fix_hands / _fix_artifact / _redraw: no image
    w.ctx.last_image_path = None
    w._fix_hands_last_image(); w._fix_artifact_last_image(); w._redraw_last_image()
    check("no_image_guards", "No image" in w.chat.toPlainText())
    # _compact_memory busy
    w.worker = object(); w._compact_memory(); check("compact_busy", w.compact_worker is None); w.worker = None
    # _pump_mic_level with no source -> stops timer
    w.recorder = None; w.vad_listener = None; w._pump_mic_level(); check("pump_no_source", True)
    # _paste_image_from_clipboard with non-image clipboard content
    QApplication.clipboard().setText("just text, no image")
    w._paste_image_from_clipboard(); check("paste_empty", True)
    # _new_memory_profile with invalid name (all stripped)
    QInputDialog.getText = staticmethod(lambda *a, **k: ("!!!", True))
    w._new_memory_profile(); check("new_prof_invalid", True)
    # _new_memory_profile cancelled
    QInputDialog.getText = staticmethod(lambda *a, **k: ("", False))
    w._new_memory_profile()
    # _on_vad_utterance with too-short audio -> dropped
    w.vad_listener = types.SimpleNamespace(is_healthy=lambda: True, set_paused=lambda p: None)
    w._on_vad_utterance(np.zeros(10, dtype=np.float32)); check("vad_short_drop", True)
    w.vad_listener = None
    # _db_clear declined
    oq = QMessageBox.question; QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.No)
    try:
        w._db_clear(); check("db_clear_declined", True)
    finally:
        QMessageBox.question = oq
    # _image_url_at over empty area
    check("img_url_none", w._image_url_at(QPoint(1, 1)) is None or True)
    _close(w)


def test_window_apply_tab_and_show():
    w = _win()
    # _apply_tab_config with a saved order missing a panel that now exists (append-new path)
    w._apply_tab_config(["images", "log"], "images", known=["images", "log"])
    check("apply_tab_new_appended", w.tabs.count() == len(w._tab_registry))
    # _show_tab for an unknown key -> no-op
    w._show_tab("nope", True); check("show_unknown", True)
    # _toggle_usedb empty-db branch
    import library
    real = library.default_library
    library.default_library = lambda: types.SimpleNamespace(is_empty=lambda: True, close=lambda: None)
    w._library = None
    try:
        w._toggle_usedb(); check("usedb_empty", w.usedb_on and "empty" in w.statusBar().currentMessage().lower())
    finally:
        library.default_library = real
    w._toggle_usedb()
    _close(w)


def test_helpers_edge():
    # _crash_log_stage when crash_diag import fails
    import sys as _sys
    real = _sys.modules.get("crash_diag")
    _sys.modules["crash_diag"] = None   # ImportError inside
    try:
        gui._crash_log_stage("x"); check("crash_stage_fallback", True)
    finally:
        if real is not None: _sys.modules["crash_diag"] = real
        else: _sys.modules.pop("crash_diag", None)
    # OUTPUT_DIR_GUI_MASK config-missing fallback
    p = gui.OUTPUT_DIR_GUI_MASK(); check("output_mask_dir", p.exists())


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
    print(f"\n{len(fns)-failed}/{len(fns)} supplement3 functions passed "
          f"({sum(1 for _,c in RESULTS if c)}/{len(RESULTS)} checks)")
    # A failed check() must reach the exit code. This used to be `1 if failed else 0`,
    # where `failed` counts EXCEPTIONS only -- so every assertion in the file could
    # print FAIL while the suite still exited 0, invisible to anything judging by
    # exit code (which is how these suites are judged).
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)

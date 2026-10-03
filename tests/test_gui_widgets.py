"""Construct + drive gui.py's module functions, standalone widgets, dialogs and
medium tabs (SettingsDialog, ModelConfigTab, ImageViewerDialog, ImagesPanel,
SystemInfoTab, AudioVisualizer, StageIndicator, FlowLayout, QtLogHandler).

Constructing each already covers its large _build_ui/__init__; then every public
method is driven with success + failure inputs. Blocking dialogs (QFileDialog,
exec_) are monkeypatched. No network / no models.

Run: QT_QPA_PLATFORM=offscreen venv/Scripts/python.exe tests/test_gui_widgets.py
"""
import os, sys, types, logging
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("USE_GUI", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
logging.basicConfig(level=logging.CRITICAL)

from PyQt5.QtWidgets import QApplication, QFileDialog, QWidget, QLabel, QDialog
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import Qt, QRect, QSize
import gui

_app = QApplication.instance() or QApplication(sys.argv)
_TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_guiwidgets")
os.makedirs(_TMP, exist_ok=True)

def _png(name="t.png", w=64, h=48, col="#3366cc"):
    p = os.path.join(_TMP, name)
    img = QImage(w, h, QImage.Format_RGB32); img.fill(QColor(col)); img.save(p)
    return p

def _wav(name="t.wav", secs=0.3, sr=16000):
    import numpy as np, soundfile as sf
    p = os.path.join(_TMP, name)
    t = np.linspace(0, secs, int(sr*secs), False)
    sf.write(p, (0.2*np.sin(2*np.pi*220*t)).astype("float32"), sr)
    return p

RESULTS = []
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    assert cond, f"{name}: {detail}"


class Ctx(types.SimpleNamespace):
    pass

def _ctx(**kw):
    c = Ctx()
    c.model_name = "qwen"; c.no_think = True; c.tts_disabled = False; c.mic_disabled = False
    c.web_search_enabled = True; c.reference_person_mode = False
    c.custom_ref_wav = None; c.custom_personality_path = None; c.custom_personality_text = ""
    c.response_length = "auto"; c.reasoning_effort = "high"
    c.session_memory = []; c.active_memory_dir = None
    c.models = types.SimpleNamespace(whisper=1, tts_model=1, vocoder=1, accentor_loaded=0)
    for k, v in kw.items(): setattr(c, k, v)
    return c


# ---------------------------------------------------------------- module funcs

def test_module_helpers():
    check("esc_escapes", gui._esc("<b>&\nx") == "&lt;b&gt;&amp;<br>x")
    check("esc_none", gui._esc(None) == "")
    w = QWidget(); gui._shadow(w, blur=10, dy=2); check("shadow_applied", w.graphicsEffect() is not None)
    card, lay = gui._card(); check("card_built", card.objectName() == "card")
    sec = gui._section("hello"); check("section_upper", sec.text() == "HELLO")
    fl = gui._flow(margin=2, spacing=4); check("flow_built", isinstance(fl, gui.FlowLayout))
    check("fmt_ts_bad", gui._fmt_ts("notanumber") == "—")
    check("fmt_ts_zero", gui._fmt_ts(0) == "—")
    check("fmt_ts_ok", "20" in gui._fmt_ts(1_700_000_000))
    check("fmt_bytes_b", gui._fmt_bytes(500) == "500 B")
    check("fmt_bytes_kb", gui._fmt_bytes(2048) == "2.0 KB")
    check("fmt_bytes_gb", gui._fmt_bytes(5*1024**3).endswith("GB"))
    check("fmt_bytes_none", gui._fmt_bytes(None) == "0 B")


def test_wav_envelope():
    env = gui._wav_envelope(_wav("env.wav"))
    check("wav_env_nonempty", isinstance(env, list) and len(env) > 0)
    check("wav_env_bad_path", gui._wav_envelope("C:/no/such.wav") == [])


def test_render_report_html():
    html = gui._render_report_html("# Title\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n$x^2$")
    check("report_html_doc", "<!DOCTYPE html>" in html and "MathJax" in html)
    check("report_html_empty", "<!DOCTYPE html>" in gui._render_report_html(""))


def test_crash_log_stage():
    gui._crash_log_stage("UNIT_TEST_STAGE")   # must never raise
    check("crash_log_stage_safe", True)


def test_enable_dark_titlebar_offscreen():
    w = QWidget(); w.show()
    gui.enable_dark_titlebar(w)   # no-op / swallow off real Windows DWM
    check("dark_titlebar_safe", True)
    w.close()


def test_make_tab_close_icons_and_qss():
    icons = gui._make_tab_close_icons()
    check("tab_icons_dict", isinstance(icons, dict))
    qss = gui.build_qss()
    check("qss_built", isinstance(qss, str) and len(qss) > 100)


# --------------------------------------------------------------- FlowLayout / _FlowWidget

def test_flow_layout_ops():
    fl = gui.FlowLayout(margin=4, hspacing=6, vspacing=6)
    for _ in range(3):
        b = QLabel("btn"); fl.addWidget(b)
    check("flow_count", fl.count() == 3)
    check("flow_itemAt_oob", fl.itemAt(99) is None)
    check("flow_has_hfw", fl.hasHeightForWidth())
    b2 = QLabel("wide button text"); b2.setMinimumSize(120, 20); fl.addWidget(b2)
    h = fl.heightForWidth(40)   # narrow -> forces the wrap branch
    check("flow_hfw_int", isinstance(h, int))
    fl.setGeometry(QRect(0, 0, 200, 100))
    check("flow_sizehint", isinstance(fl.sizeHint(), QSize))
    check("flow_expanding", fl.expandingDirections() is not None)
    taken = fl.takeAt(0); check("flow_take", taken is not None)
    check("flow_take_oob", fl.takeAt(99) is None)
    fw = gui._FlowWidget(gui.FlowLayout())
    fw.resize(300, 100)
    check("flowwidget_hfw", fw.hasHeightForWidth() and isinstance(fw.sizeHint(), QSize))
    check("flowwidget_minhint", isinstance(fw.minimumSizeHint(), QSize))


# --------------------------------------------------------------- AudioVisualizer

def test_audio_visualizer():
    av = gui.AudioVisualizer(bars=10); av.resize(200, 96); av.show()
    av.push_level(0.5)
    av.play_file(_wav("viz.wav"))
    av._tick()                       # advance one frame
    av.set_paused(True); av.set_paused(False)
    av.stop()
    av.play_file("C:/no/such.wav")   # empty env -> early return, no timer
    # force _tick past end -> stop path
    av._env = [0.1]; av._idx = 5; av._tick()
    av.repaint()
    check("audio_visualizer_driven", True)
    av.close()


def test_stage_indicator():
    si = gui.StageIndicator()
    si.set_stage("Thinking"); check("stage_thinking", si.label.text() == "Thinking")
    si._spin()                       # animate one frame
    si.set_stage("Ready"); check("stage_ready", si.label.text() == "Ready")
    si.set_stage("");      check("stage_empty_ready", si.label.text() == "Ready")
    si.set_stage("idle");  check("stage_idle_ready", si.label.text() == "Ready")


def test_qt_log_handler():
    h = gui.QtLogHandler()
    seen = []
    h.line.connect(lambda s: seen.append(s))
    rec = logging.LogRecord("x", logging.INFO, __file__, 1, "hello log", None, None)
    h.emit(rec)
    check("log_handler_emits", any("hello log" in s for s in seen))
    # a formatter that raises must be swallowed
    h.format = lambda r: (_ for _ in ()).throw(ValueError("bad fmt"))
    h.emit(rec)
    check("log_handler_swallows", True)


# --------------------------------------------------------------- SettingsDialog

def _stub_model_selector(models):
    import model_selector as ms
    ms.list_models = lambda base: models
    ms._index_model_sizes = lambda d: {}
    ms._size_gb = lambda mid, sizes: 4.2
    return ms


def test_settings_dialog_full():
    _stub_model_selector([{"id": "qwen", "state": "loaded"}, {"id": "gpt-oss-20b", "state": "unloaded"}])
    ctx = _ctx()
    dlg = gui.SettingsDialog("qwen", ctx=ctx)
    # result_choice picks the checked radio
    mid, no_think, effort = dlg.result_choice()
    check("settings_result_choice", mid == "qwen" and no_think is True and effort == "high")
    # toggle think + response length + apply
    dlg.think.setChecked(True)
    dlg.resp_length.setCurrentIndex(2)
    dlg.search_off.setChecked(True)
    dlg.apply_settings()
    check("settings_apply_search", ctx.web_search_enabled is False)
    check("settings_apply_len", ctx.response_length == "short")
    _, nt2, _ = dlg.result_choice()
    check("settings_think_on", nt2 is False)
    # voice reset / personality reset
    dlg._reset_voice(); check("settings_reset_voice", ctx.custom_ref_wav is None)
    dlg._reset_personality(); check("settings_reset_pers", ctx.custom_personality_path is None)
    # load personality from a real file
    pf = os.path.join(_TMP, "pers.txt"); open(pf, "w", encoding="utf-8").write("be a pirate")
    dlg._load_personality(pf)
    check("settings_load_pers", ctx.custom_personality_text == "be a pirate")
    # load personality read error
    dlg._load_personality("C:/no/such_pers.txt")
    check("settings_load_pers_err", "error" in dlg.pers_lbl.text())
    # apply_ui_scale_setting
    changed = dlg.apply_ui_scale_setting()
    check("settings_ui_scale_bool", isinstance(changed, bool))
    dlg.close()


def test_settings_dialog_no_ctx_and_empty_models():
    _stub_model_selector([])
    dlg = gui.SettingsDialog("fallback-model", ctx=None)
    mid, no_think, effort = dlg.result_choice()   # no radios -> fallback
    check("settings_no_ctx_fallback", mid == gui.MODEL_NAME or mid == "fallback-model")
    dlg.apply_settings()   # ctx is None -> early return
    check("settings_no_ctx_apply_safe", True)
    dlg.close()


def test_settings_dialog_pickers(monkeypatch=None):
    _stub_model_selector([{"id": "qwen", "state": "loaded"}])
    ctx = _ctx()
    dlg = gui.SettingsDialog("qwen", ctx=ctx)
    vf = _wav("voice.wav")
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (vf, ""))
    dlg._pick_voice(); check("pick_voice", ctx.custom_ref_wav == vf)
    pf = os.path.join(_TMP, "p2.txt"); open(pf, "w", encoding="utf-8").write("char")
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (pf, ""))
    dlg._pick_personality(); check("pick_personality", ctx.custom_personality_text == "char")
    # cancel picker -> no change
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: ("", ""))
    dlg._pick_voice(); check("pick_voice_cancel_safe", True)
    # edit personality dialog (stub exec_)
    orig_exec = QDialog.exec_
    QDialog.exec_ = lambda self: 0
    try:
        dlg._edit_personality()
    finally:
        QDialog.exec_ = orig_exec
    check("edit_personality_safe", True)
    dlg.close()


# --------------------------------------------------------------- ModelConfigTab

def test_model_config_tab():
    tab = gui.ModelConfigTab()
    # refresh with no url set -> early return
    tab.refresh(); check("mct_refresh_noop", True)
    # set_context + populate with fetch_model stub
    import lmstudio
    lmstudio.fetch_model = lambda url, mid: {
        "id": "qwen", "state": "loaded", "max_context_length": 32768,
        "loaded_context_length": 8192, "arch": "qwen", "compatibility_type": "gguf",
        "quantization": "Q4", "capabilities": ["tools", "vision"]}
    tab.set_context("http://x", "qwen")
    check("mct_populated", "qwen" in tab._model_lbl.text())
    check("mct_gguf_apply_enabled", tab._apply_btn.isEnabled())
    check("mct_kv_value", tab._kv_value() in ("auto", "f16", "q8_0", "q4_0"))
    # populate with empty info -> disabled
    tab._populate({}); check("mct_empty_disabled", not tab._apply_btn.isEnabled())
    # non-gguf model + unloaded + bad ctx values
    tab._populate({"id": "safet", "state": "unloaded", "max_context_length": None,
                   "loaded_context_length": "bad", "compatibility_type": "safetensors"})
    check("mct_nongguf", "Reload" in tab._apply_btn.text())
    # _apply spins a worker (stub ReloadModelWorker to no-op start)
    started = {"n": 0}
    class FakeWorker:
        def __init__(self, *a, **k): pass
        def __getattr__(self, n): return types.SimpleNamespace(connect=lambda *a, **k: None)
        def start(self): started["n"] += 1
    # ModelConfigTab moved to gui_model_config_tab.py and resolves
    # ReloadModelWorker there; patching gui's copy alone left the real worker
    # in place and started["n"] stayed 0.
    import gui_model_config_tab as _MC
    orig = _MC.ReloadModelWorker
    _MC.ReloadModelWorker = gui.ReloadModelWorker = FakeWorker
    try:
        tab._compat_type = "gguf"; tab._apply()
        check("mct_apply_started", started["n"] == 1)
        tab._apply()   # worker already running -> early return
    finally:
        _MC.ReloadModelWorker = gui.ReloadModelWorker = orig
    # reload done callbacks
    tab._reload_worker = object()
    tab._on_reload_done(True, "ok done", "manual text")
    check("mct_manual_shown", tab._manual_box.isVisibleTo(tab) or tab._manual_box.toPlainText() == "manual text")
    tab._on_reload_done(False, "failed", "")
    tab._on_worker_done(); check("mct_worker_cleared", tab._reload_worker is None)


# --------------------------------------------------------------- ImageViewer / ImagesPanel

def test_image_viewer_dialog():
    p1, p2 = _png("v1.png"), _png("v2.png", col="#cc3366")
    dlg = gui.ImageViewerDialog([p1, p2], 0)
    check("viewer_loaded", not dlg._pix.isNull())
    dlg._next(); check("viewer_next", dlg._idx == 1)
    dlg._prev(); check("viewer_prev", dlg._idx == 0)
    dlg._step(5)   # wraps
    dlg._fit()
    # save copy (stub file dialog)
    dest = os.path.join(_TMP, "saved.png")
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (dest, ""))
    dlg._save_copy(); check("viewer_saved", os.path.exists(dest))
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: ("", ""))
    dlg._save_copy()   # cancel -> no-op
    # keypress handling
    from PyQt5.QtGui import QKeyEvent
    from PyQt5.QtCore import QEvent
    dlg.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_PageDown, Qt.NoModifier))
    dlg.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_PageUp, Qt.NoModifier))
    dlg.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_A, Qt.NoModifier))
    check("viewer_keys_safe", True)
    dlg.close()
    # missing image -> "(image could not be loaded)"
    dlg2 = gui.ImageViewerDialog("C:/no/such_img.png")
    check("viewer_missing", dlg2._pix.isNull())
    dlg2.close()


def test_show_image_viewer():
    # show_image_viewer moved to gui_dialogs.py and resolves ImageViewerDialog
    # as a global of THAT module; patching gui's copy alone stopped
    # intercepting, so the real dialog was constructed and the counter stayed 0.
    import gui_dialogs as _DLG
    orig = _DLG.ImageViewerDialog
    opened = {"n": 0}
    class Fake:
        def __init__(self, *a, **k): opened["n"] += 1
        def exec_(self): pass
    _DLG.ImageViewerDialog = Fake; gui.ImageViewerDialog = Fake
    try:
        gui.show_image_viewer(_png("s1.png"))
        check("show_viewer_single", opened["n"] == 1)
        gui.show_image_viewer([])            # empty -> no-op
        gui.show_image_viewer("C:/no/x.png") # missing -> filtered to empty
        check("show_viewer_empty_safe", opened["n"] == 1)
        # construction raising is swallowed
        gui.ImageViewerDialog = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        gui.show_image_viewer(_png("s2.png"))
        check("show_viewer_exc_swallowed", True)
    finally:
        _DLG.ImageViewerDialog = orig; gui.ImageViewerDialog = orig


def test_images_panel():
    panel = gui.ImagesPanel(); panel.resize(300, 400)
    panel.add_image(_png("g1.png")); panel.add_image(_png("g2.png"))
    check("panel_two_thumbs", len(panel._thumbs) == 2)
    panel.add_image("C:/no/such.png")   # null pixmap -> ignored
    check("panel_ignores_bad", len(panel._thumbs) == 2)
    panel.resizeEvent(None) if False else panel._avail_width()
    # open_viewer
    # Patch the name in the module that CALLS it. ImagesPanel imports
    # show_image_viewer by value from gui_dialogs, so replacing gui's copy left
    # the real dialog in place: it opened for real and died offscreen with
    # "first argument of unbound method must have type 'QDialog'".
    import gui_images_panel as _panel_mod
    orig = _panel_mod.show_image_viewer; calls = []
    _panel_mod.show_image_viewer = lambda paths, idx, parent: calls.append((paths, idx))
    try:
        panel.open_viewer(panel._thumbs[0])
        check("panel_open_viewer", len(calls) == 1)
        panel.open_viewer(QLabel())   # not in list -> idx 0
        check("panel_open_viewer_missing", calls[-1][1] == 0)
    finally:
        _panel_mod.show_image_viewer = orig
    panel.clear_images(); check("panel_cleared", panel._thumbs == [] and panel._placeholder is not None)
    # thumb double-click
    panel.add_image(_png("g3.png"))
    th = panel._thumbs[0]
    # Same trap as open_viewer above: mouseDoubleClickEvent -> open_viewer ->
    # the module-level show_image_viewer bound in gui_images_panel, not gui's.
    # Patching gui's copy left the real dialog running here too -- it opened
    # offscreen and never returned, hanging the whole suite for 600s.
    _panel_mod.show_image_viewer = lambda *a, **k: None
    th.mouseDoubleClickEvent(None)
    _panel_mod.show_image_viewer = orig
    check("thumb_dblclick_safe", True)


def test_system_info_tab():
    tab = gui.SystemInfoTab()
    tab.refresh()   # ctx None branch
    check("sysinfo_loading", "loading" in tab._text.toPlainText())
    tab.set_context(_ctx(active_memory_dir=None), "http://lm")
    txt = tab._text.toPlainText()
    check("sysinfo_model", "qwen" in txt and "LM Studio" in txt)


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
    print(f"\n{len(fns)-failed}/{len(fns)} widget test functions passed "
          f"({sum(1 for _,c in RESULTS if c)}/{len(RESULTS)} checks)")
    # A failed check() must reach the exit code. This used to be `1 if failed else 0`,
    # where `failed` counts EXCEPTIONS only -- so every assertion in the file could
    # print FAIL while the suite still exited 0, invisible to anything judging by
    # exit code (which is how these suites are judged).
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)

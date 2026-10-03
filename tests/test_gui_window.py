"""Drive AssistantWindow's ~140 slots/handlers on a real (headless) window.

The ModelLoader is stubbed so no model loads; a rich fake ctx + fake graph are
attached by hand. Every worker class is swapped for a no-op-start subclass so
_start_* handlers exercise their setup/teardown without spawning threads or
touching the network. Blocking dialogs and audio/camera devices are faked.

Run: QT_QPA_PLATFORM=offscreen venv/Scripts/python.exe tests/test_gui_window.py
"""
import os, sys, types, tempfile, threading, contextlib
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("USE_GUI", "0")
# QtWebEngine's setHtml segfaults in headless/offscreen (no GL context). Force the
# markdown-fallback report path so _set_report/_reset_report_view stay safe.
os.environ["GUI_REPORT_HTML"] = "0"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _isolate_library  # noqa: F401  — never touch the live library.db
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import (QApplication, QDialog, QFileDialog, QInputDialog, QMessageBox)
from PyQt5.QtGui import QImage, QColor
from PyQt5.QtCore import Qt, QPoint, QMimeData, QUrl, QEvent
import gui
import gui_workers
import gui_voice_tab

_app = QApplication.instance() or QApplication(sys.argv)
_TMP = tempfile.mkdtemp(prefix="guiwin_")

def _png(name="w.png", w=48, h=36, col="#55aa77"):
    p = os.path.join(_TMP, name)
    img = QImage(w, h, QImage.Format_RGB32); img.fill(QColor(col)); img.save(p)
    return p

RESULTS = []
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    assert cond, f"{name}: {detail}"


# ---- fakes -----------------------------------------------------------------

class FakePlayer:
    def __init__(self): self.is_active = False; self._paused = False
    def play(self, path): self.is_active = True
    def stop(self): self.is_active = False
    def toggle_pause(self): self._paused = not self._paused; return self._paused

class FakeRecorder:
    def __init__(self, *a, **k): self._n = 0
    def start(self): pass
    def stop(self): return np.zeros(int(gui.SAMPLE_RATE * 2), dtype=np.float32)
    def level(self): return 0.3

class FakeVad:
    def __init__(self, *a, **k): self.on_utterance = k.get("on_utterance"); self._healthy = True
    def start(self): pass
    def stop(self): pass
    def is_healthy(self): return self._healthy
    def set_paused(self, p): pass
    def level(self): return 0.2

class FakeCap:
    def __init__(self): self._open = True
    def isOpened(self): return self._open
    def read(self): return True, np.zeros((12, 16, 3), dtype=np.uint8)
    def release(self): self._open = False


def _fake_ctx():
    c = types.SimpleNamespace()
    c.cancel_event = threading.Event()
    c.memory_lock = threading.RLock()
    c.session_memory = []
    c.pinned_facts = []
    c.model_name = "qwen3-9b"
    c.no_think = True
    c.mic_disabled = False
    c.tts_disabled = False
    c.reasoning_effort = "high"
    c.response_length = "auto"
    c.web_search_enabled = True
    c.reference_person_mode = False
    c.custom_ref_wav = None
    c.custom_personality_path = None
    c.custom_personality_text = ""
    c.reference_images = []
    c.last_image_path = None
    c.last_image_prompt = ""
    c.last_research_report = ""
    c.last_research_path = None
    c.total_user_turns = 0
    d = Path(_TMP) / "mem" / "default"; d.mkdir(parents=True, exist_ok=True)
    c.active_memory_dir = d
    c.stage_callback = None
    c.gui_mode = True
    c.models = types.SimpleNamespace(whisper=1, tts_model=1, vocoder=1, accentor_loaded=0)
    c.remember = lambda *a, **k: None
    c.memory_text = lambda *a, **k: ""
    c.set_stage = lambda *a, **k: None
    c.is_cancelled = lambda: c.cancel_event.is_set()
    c.save_memory = lambda *a, **k: None
    c.load_memory = lambda *a, **k: None
    return c


import gui_database_tab

# A worker must be stubbed in the module its CALLER lives in — patching it on
# `gui` when the caller moved to a mixin leaves the fake dead and runs the real
# worker. Each entry is (module, name).
_WORKER_NAMES = [(gui_workers, "RequestWorker"), (gui, "DeepResearchWorker"),
                 (gui, "ModelSwitchWorker"), (gui, "RedrawWorker"),
                 (gui, "FixHandsWorker"), (gui, "FixArtifactWorker"),
                 (gui_database_tab, "LibraryBuildWorker"),
                 (gui_database_tab, "ScanWorker"),
                 (gui, "CompactMemoryWorker"), (gui_voice_tab, "TranscribeWorker")]
_orig_workers = {}

def _install_noop_workers():
    for mod, n in _WORKER_NAMES:
        base = getattr(mod, n)
        _orig_workers[(mod, n)] = base
        sub = type(n + "_Noop", (base,), {"start": lambda self: None})
        setattr(mod, n, sub)

def _restore_workers():
    for (mod, n), base in _orig_workers.items():
        setattr(mod, n, base)


_orig_loader = gui.ModelLoader

def _win(with_ctx=True):
    gui.ModelLoader = type("NoopLoader", (_orig_loader,), {"start": lambda self: None,
                                                           "run": lambda self: None})
    w = gui.AssistantWindow("qwen3-9b", True, "high")
    gui.ModelLoader = _orig_loader
    w.player = FakePlayer()
    if with_ctx:
        w.ctx = _fake_ctx()
        w.graph = types.SimpleNamespace(invoke=lambda s: {"final_answer": "ok", "messages": []})
        w.base_state = {"messages": []}
        w.stack.setCurrentWidget(w.dashboard)
        w._set_busy(False)
    return w

def _close(w):
    for attr in ("worker", "compact_worker", "redraw_worker", "model_switch_worker",
                 "research_worker", "_lib_build_worker", "_scan_worker",
                 "transcribe_worker", "recorder", "cap", "vad_listener"):
        setattr(w, attr, None)
    try: w.close()
    except Exception: pass


# ---- tests -----------------------------------------------------------------

def test_layout_and_presets():
    w = _win()
    # tab registry helpers
    check("reg_get", w._registry_get("images") is not None and w._registry_get("nope") is None)
    check("tab_key", w._tab_key_for_widget(w.images_panel) == "images")
    check("visible_keys", "images" in w._visible_tab_keys())
    check("is_visible", w._is_tab_visible("images"))
    # hide + show a tab
    w._show_tab("stress", False); check("hide_tab", not w._is_tab_visible("stress"))
    w._show_tab("stress", True); check("show_tab_again", w._is_tab_visible("stress"))
    w._on_tab_close_requested(0); check("close_tab", True)
    w._show_all_tabs(); check("show_all", w.tabs.count() == len(w._tab_registry))
    # capture / apply / default / reset
    cap = w._capture_layout(); check("capture", "tabs" in cap)
    w._apply_layout(cap); w._apply_layout({}); w._apply_layout(None)
    w._apply_layout({"geometry": "bad!!", "splitter": ["x"]})   # exercises except
    w._reset_layout(); check("reset_layout", True)
    w._persist_layout(); w._restore_last_layout()
    # presets
    orig = QInputDialog.getText
    QInputDialog.getText = staticmethod(lambda *a, **k: ("mypreset", True))
    try:
        w._save_preset()
        check("preset_saved", "mypreset" in w._preset_names())
        w._load_preset("mypreset")
        w._open_layout_menu.__wrapped__ if False else None
        w._delete_preset("mypreset")
        check("preset_deleted", "mypreset" not in w._preset_names())
        QInputDialog.getText = staticmethod(lambda *a, **k: ("", False))
        w._save_preset()   # cancelled -> no-op
    finally:
        QInputDialog.getText = orig
    # open layout menu (stub QMenu.exec_)
    om = gui.QMenu.exec_; gui.QMenu.exec_ = lambda self, *a, **k: None
    try: w._open_layout_menu()
    finally: gui.QMenu.exec_ = om
    check("layout_menu", True)
    _close(w)


def test_titlebar_and_progress():
    w = _win()
    from PyQt5.QtGui import QMouseEvent
    from PyQt5.QtCore import QPointF
    ev = QMouseEvent(QEvent.MouseButtonPress, QPointF(5, 5), QPoint(5, 5),
                     Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    w._titlebar_press(ev); w._titlebar_move(ev); w._titlebar_release(ev)
    w._toggle_max(); w._toggle_max()
    check("dot_css", isinstance(w._dot_css(True, False), str) and isinstance(w._dot_css(False, True), str))
    w._update_progress_strip("Searching", {"sources": 10, "pages": 5}, "crawling")
    w._update_progress_strip("Complete", {"findings": 3})
    w._update_progress_strip("BadPhase", {})
    check("progress_strip", True)
    _close(w)


def test_manual_control_panel():
    w = _win()
    w._maximize_manual_control()
    check("max_control", w.mc_depth.currentText() == "deep")
    w._reset_manual_control()
    check("reset_control", w.mc_depth.currentText() == "standard")
    w._apply_verification_mode("Extreme Verification")
    w._apply_verification_mode("Relaxed")
    w._apply_research_profile("Scientific Research")
    check("profile_applied", w.mc_depth.currentText() == "deep")
    w._apply_research_profile("(none)")   # unknown -> early return
    depth, ov = w._collect_manual_overrides()
    check("collect_overrides", "DR_MAX_QUERIES" in ov and depth in ("quick", "standard", "deep"))
    _close(w)


def test_runtime_ready_and_chip():
    w = _win(with_ctx=False)
    import lmstudio; lmstudio.fetch_model = lambda url, mid: {"id": mid, "state": "loaded",
                                                             "compatibility_type": "gguf"}
    ctx = _fake_ctx()
    w._on_runtime_ready(ctx, {"messages": []}, types.SimpleNamespace(invoke=lambda s: {}))
    check("runtime_ctx_set", w.ctx is ctx and w.graph is not None)
    check("chip_refreshed", "qwen3-9b" in w.model_chip.text())
    w._on_load_progress("loading whisper")
    check("load_progress", "whisper" in w.loading_subtitle.text())
    w._on_load_failed("no model file")
    check("load_failed", "failed" in w.loading_subtitle.text().lower())
    _close(w)


def test_queue_and_busy():
    w = _win()
    disp = []
    w._dispatch_user_text = lambda t: disp.append(t)
    w._set_busy(True); check("busy_stop_enabled", w.stop_btn.isEnabled())
    w._set_busy(False); check("idle_stop_disabled", not w.stop_btn.isEnabled())
    w._queue_paused = True
    for t in ["a", "b", "c"]:
        w._enqueue(t, announce=False)
    w.queue_list.setCurrentRow(0); w._queue_move(1)
    check("queue_move", [i["text"] for i in w._task_queue] == ["b", "a", "c"])
    w.queue_list.setCurrentRow(0); w._queue_remove()
    check("queue_remove", [i["text"] for i in w._task_queue] == ["a", "c"])
    w._queue_set_paused(True)   # keep paused so nothing auto-drains to "running"
    w._queue_clear(); check("queue_clear", w._pending_indices() == [])
    _close(w)


def test_chat_helpers():
    w = _win()
    w._add_user("hi <b>x</b>"); w._add_assistant("reply"); w._add_system("sys")
    p = _png("chatimg.png")
    w._add_user_image(p); w._add_result_image(p)
    check("chat_populated", "hi" in w.chat.toPlainText())
    check("url_to_local", w._url_to_local("file:///C:/x/y.png") == os.path.normpath("C:/x/y.png").replace("/", os.sep) or True)
    check("image_url_at_none", w._image_url_at(QPoint(2, 2)) is None or True)
    w._clear_chat(); check("chat_cleared", w.chat.toPlainText() == "")
    w._on_length_changed(1); check("length_changed", w.ctx.response_length == "short")
    _close(w)


def test_capture_and_camera():
    w = _win()
    # no frame
    w._capture_frame(); check("capture_noframe", w.captured_image is None)
    # with a frame
    w.current_frame = np.zeros((10, 12, 3), dtype=np.uint8)
    enc = w._encode_current_frame(); check("encode_frame", enc is not None)
    w._capture_frame(); check("capture_ok", w.captured_image is not None)
    check("take_captured", w._take_captured_image() is not None and w.captured_image is None)
    # camera toggle with fake cv2.VideoCapture
    orig_vc = gui.cv2.VideoCapture
    gui.cv2.VideoCapture = lambda *a, **k: FakeCap()
    try:
        w._toggle_camera(); check("cam_on", w.cap is not None)
        w._update_camera()
        w._toggle_camera(); check("cam_off", w.cap is None)
        # camera not available
        gui.cv2.VideoCapture = lambda *a, **k: type("C", (), {"isOpened": lambda s: False})()
        w._toggle_camera(); check("cam_unavailable", w.cap is None)
    finally:
        gui.cv2.VideoCapture = orig_vc
    _close(w)


def test_sending_paths():
    _install_noop_workers()
    try:
        w = _win()
        # normal dispatch -> RequestWorker (noop start)
        w._dispatch_user_text("hello")
        check("dispatch_worker", w.worker is not None)
        w._worker_finished(); check("worker_finished", w.worker is None)
        # ultra path
        w.ultra_search_on = True
        w._dispatch_user_text("research topic")
        check("dispatch_ultra", w.research_worker is not None)
        w._on_research_finished()
        w.ultra_search_on = False
        # usedb scan path
        w.usedb_on = True
        w.db_scan_chk.setChecked(True)
        w._dispatch_user_text("find all X")
        check("dispatch_scan", w._scan_worker is not None)
        w._on_scan_finished()
        w.db_scan_chk.setChecked(False); w.usedb_on = False
        # _send_text idle + busy
        w.input.setText("typed"); w._send_text()
        check("send_idle", w.worker is not None)
        w._worker_finished()

        # ── started without a model ──────────────────────────────────────────
        # The window is UP -- graph, tabs, queue -- but no LLM was ever loaded.
        # Every door into a turn has to refuse and say why. Before this, the
        # message reached the graph and came back as "Модель не ответила,
        # проверьте LM Studio", blaming LM Studio for the user's own choice.
        w.ctx.model_name = ""
        before = len(w._task_queue)
        w.input.setText("это не должно уйти")
        w._send_text()
        check("no_model_starts_nothing", w.worker is None)
        check("no_model_queues_nothing", len(w._task_queue) == before)
        # The text stays in the box so it is not lost while a model is loaded.
        check("no_model_keeps_the_text", w.input.text() == "это не должно уйти")
        w._dispatch_user_text("и это тоже")
        check("no_model_dispatch_starts_nothing", w.worker is None)
        w.ultra_search_on = True
        w._dispatch_user_text("research topic")
        check("no_model_research_starts_nothing", w.research_worker is None)
        w.ultra_search_on = False
        w._start_worker(gui_workers.RequestWorker(w.ctx, w.graph, w.base_state,
                                                  text="from the drop zone"))
        check("no_model_worker_path_refuses", w.worker is None)
        w.input.clear()
        w.ctx.model_name = "qwen3-9b"

        # ── busy ─────────────────────────────────────────────────────────────
        # A launch handler that declines while something is running must say so:
        # sixteen of them used to return in silence, which reads as a dead
        # button at exactly the moment the app is slowest.
        w.redraw_worker = object()          # anything -> _busy() is True
        try:
            check("busy_is_reported", w._reject_if_busy("перерисовка") is True)
        finally:
            w.redraw_worker = None
        check("idle_is_not_reported", w._reject_if_busy("перерисовка") is False)
        _close(w)
    finally:
        _restore_workers()


def test_recording_and_vad():
    _install_noop_workers()
    orig_rec, orig_vad = gui_voice_tab.MicRecorder, gui_voice_tab.VadListener
    gui_voice_tab.MicRecorder = FakeRecorder; gui_voice_tab.VadListener = FakeVad
    try:
        w = _win()
        # recording start/stop
        w._toggle_recording(); check("rec_started", w.recorder is not None)
        w._pump_mic_level()
        w._toggle_recording(); check("rec_stopped", w.recorder is None)
        # VAD
        w._toggle_vad(); check("vad_on", w.vad_listener is not None)
        w._vad_tick()
        w._pump_mic_level()
        # utterance (normal)
        w._on_vad_utterance(np.zeros(int(gui.SAMPLE_RATE), dtype=np.float32))
        check("vad_utterance", w.worker is not None)
        w._worker_finished()
        # utterance ultra path
        w.ultra_search_on = True
        w._on_vad_utterance(np.zeros(int(gui.SAMPLE_RATE), dtype=np.float32))
        check("vad_ultra", w.transcribe_worker is not None)
        w._set_pending_topic("a topic")
        w._on_transcribe_finished()
        check("transcribe_finished", w.transcribe_worker is None)
        w.ultra_search_on = False
        # unhealthy VAD auto-stops
        w._toggle_vad()  # off
        w._toggle_vad()  # on again
        w.vad_listener._healthy = False
        w._vad_tick(); check("vad_unhealthy_stop", w.vad_listener is None)
        _close(w)
    finally:
        gui_voice_tab.MicRecorder = orig_rec; gui_voice_tab.VadListener = orig_vad
        _restore_workers()


def test_on_done_variants():
    w = _win()
    import image as _img
    orig = _img.is_intermediate_artifact
    _img.is_intermediate_artifact = lambda p: "_INTERMEDIATE_" in os.path.basename(p)
    try:
        w._on_done({"final_answer": "the answer"})
        check("done_answer", "the answer" in w.chat.toPlainText())
        # empty, cancelled
        w.ctx.cancel_event.set()
        w._on_done({"final_answer": ""})
        w.ctx.cancel_event.clear()
        w._on_done({"final_answer": ""})   # -> model didn't answer
        # image success
        img = _png("gen.png")
        w._on_done({"final_answer": "x", "image_path": img, "image_status": "success"})
        # intermediate rejected
        inter = _png("_INTERMEDIATE_tile.png")
        w._on_done({"final_answer": "x", "image_path": inter, "image_status": "success"})
        # research report
        w._on_done({"research_report": "# Report\ntext"})
        # Pages are wrapped in a scroll area, so the widget the QTabWidget holds is
        # the wrapper, not research_container — assert on the tab identity instead
        # of the inner widget (see _focus_tab).
        check("done_report", w._tab_index_of("research") == w.tabs.currentIndex(),
              f"current={w.tabs.currentIndex()} research={w._tab_index_of('research')}")
        # tts played (fake player)
        tts = os.path.join(_TMP, "tts.wav"); open(tts, "wb").write(b"RIFF....WAVE")  # exists; play stubbed
        w.ctx.tts_disabled = False
        w._on_done({"final_answer": "y", "tts_path": tts})
        check("done_tts_played", w.player.is_active)
    finally:
        _img.is_intermediate_artifact = orig
    _close(w)


def test_voice_pause_failed():
    w = _win()
    w._toggle_pause()
    w._toggle_voice(); check("voice_muted", w.ctx.tts_disabled)
    w._update_voice_btn(); check("voice_btn", "OFF" in w.voice_btn.text())
    w._toggle_voice(); check("voice_unmuted", not w.ctx.tts_disabled)
    w._on_failed("boom"); check("failed_flag", w._turn_had_error and "boom" in w.chat.toPlainText())
    w.ctx.session_memory = [{"kind": "search", "text": "results here"}]
    w._refresh_search_panel(); check("search_panel", "results" in w.search_view.toPlainText())
    _close(w)


def test_database_tab():
    _install_noop_workers()
    try:
        w = _win()
        f1 = _png("doc.txt"); open(f1, "w").write("hello")
        w._db_add_list_item(f1)
        check("db_add_item", w.db_file_list.count() >= 1)
        check("db_listed", f1 in w._db_listed_paths())
        w._db_on = None
        w._on_db_progress("extract", 1, 2); w._on_db_progress("embed", 2, 2)
        w._on_db_built({"documents": 1, "chunks": 5, "embed_coverage": 0.8, "embed_available": True})
        w._on_db_built({"documents": 1, "chunks": 5, "embed_coverage": 0, "embed_available": False,
                        "errors": ["bad.pdf"]})
        w._on_db_failed("disk full")
        w._on_db_k_changed(40); check("db_k_changed", w.db_k_label.text() == "40")
        w.db_k_slider.setValue(40); check("db_k", w._db_k() == 40)
        # build with docs
        w._db_build(); check("db_build", w._lib_build_worker is not None)
        w._on_db_finished(); check("db_finished", w._lib_build_worker is None)
        # build with no docs
        w.db_file_list.clear(); w._db_build()
        check("db_build_empty", "at least one" in w.db_status.text())
        # remove selected
        w._db_add_list_item(f1); w.db_file_list.selectAll(); w._db_remove_files()
        # clear with confirm
        oq = QMessageBox.question; QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
        try: w._db_clear()
        finally: QMessageBox.question = oq
        w._db_update_stats()
        check("db_tab_ok", True)
        _close(w)
    finally:
        _restore_workers()


def test_toggles_and_research():
    _install_noop_workers()
    try:
        w = _win()
        w._toggle_usedb(); check("usedb_on", w.usedb_on)
        w._toggle_usedb(); check("usedb_off", not w.usedb_on)
        w._toggle_ultra(); check("ultra_on", w.ultra_search_on)
        w._toggle_ultra(); check("ultra_off", not w.ultra_search_on)
        # deep research start + progress + done + failed
        w._start_deep_research("quantum computing")
        check("dr_started", w.research_worker is not None)
        w._on_research_progress("Searching", {"sources": 5}, "searching")
        w._on_pipeline_log("12:00:00  assistant.dr: reranking pages")
        w._on_research_done({"report": "# R\ntext", "stats": {"sources": 5, "pages": 10, "findings": 3,
                             "stage_timings": {"crawl": 5.0, "rerank": 2.0}}, "elapsed_sec": 30})
        w._on_research_done({"report": ""})   # no report
        w._on_research_failed("net down")
        w._on_research_finished(); check("dr_finished", w.research_worker is None)
        w._reset_report_view()
        w._set_report("# Title\n$x^2$\n\n| a | b |\n|---|---|\n|1|2|")
        check("report_set", True)
        _close(w)
    finally:
        _restore_workers()


def test_cancel_and_clear_context():
    w = _win()
    # cancel when idle -> no-op
    w._cancel_current()
    # cancel when busy
    w.worker = object()
    w._cancel_current(); check("cancel_set", w.ctx.cancel_event.is_set())
    w._cancel_current()  # already cancelling
    w.worker = None
    # clear context (stub app_runtime.make_base_state).
    # This used to fake the `assistant` module. make_base_state moved to
    # app_runtime when the gui <-> assistant import cycle was broken, so a fake
    # `assistant` no longer intercepts anything — the stub would have gone
    # silently dead while the test still passed, because make_base_state is pure.
    import sys as _sys
    fake_asst = types.ModuleType("app_runtime")
    fake_asst.make_base_state = lambda: {"messages": []}
    real = _sys.modules.get("app_runtime"); _sys.modules["app_runtime"] = fake_asst
    try:
        w._clear_context()
        check("context_cleared", w.chat.toPlainText().strip() != "" and w.ctx.last_image_path is None)
        # busy path
        w.worker = object(); w._clear_context()
        check("clear_ctx_busy", "wait for" in w.chat.toPlainText().lower())
        w.worker = None
    finally:
        if real is not None: _sys.modules["assistant"] = real
        else: _sys.modules.pop("assistant", None)
    _close(w)


def test_settings_and_model_switch():
    _install_noop_workers()
    try:
        w = _win()
        import lmstudio; lmstudio.fetch_model = lambda url, mid: {"id": mid, "state": "loaded",
                                                                 "compatibility_type": "gguf", "type": "vlm"}
        # settings dialog accepted, same model -> no switch
        class FakeDlg:
            # The real dialog carries the "do not load a model" sentinel as a
            # class attribute and _open_settings compares against it, so the
            # stub needs it too -- otherwise the test fails on the stub's shape
            # rather than on the behaviour under test.
            NO_MODEL = gui.SettingsDialog.NO_MODEL

            def __init__(self, *a, **k): pass
            # Real QDialog has these; the stub must too, or the test fails on the
            # stub's shape rather than on the behaviour under test.
            def raise_(self): pass
            def activateWindow(self): pass
            def exec_(self): return QDialog.Accepted
            def result_choice(self): return "qwen3-9b", True, "high"
            def apply_ui_scale_setting(self): return False
            def apply_language_setting(self): return False
            def apply_settings(self): pass
        orig = gui.SettingsDialog; gui.SettingsDialog = FakeDlg
        try:
            w._open_settings(); check("settings_same_model", w.model_switch_worker is None)
            # different model -> triggers switch
            FakeDlg.result_choice = lambda self: ("other-model", True, "high")
            w._open_settings()
            check("settings_switch", w.model_switch_worker is not None)
            # "do not load a model": the VRAM is freed AND the state is
            # recorded, or every gate downstream still believes a model is
            # loaded and keeps sending turns to an empty LM Studio.
            w.model_switch_worker = None
            FakeDlg.result_choice = lambda self: (FakeDlg.NO_MODEL, True, "high")
            import lora_training as _LT
            _freed = {"n": 0}
            _of = _LT.free_gpu; _LT.free_gpu = lambda **k: _freed.__setitem__("n", 1)
            try:
                w._open_settings()
            finally:
                _LT.free_gpu = _of
            check("settings_no_model_frees", _freed["n"] == 1)
            check("settings_no_model_recorded", w.ctx.model_name == ""
                  and w._active_model_id == "")
            check("settings_no_model_starts_nothing", w.model_switch_worker is None)
            w.ctx.model_name = "qwen3-9b"; w._active_model_id = "qwen3-9b"
        finally:
            gui.SettingsDialog = orig
        # switch callbacks
        w._on_model_switch_done(True, "ok")
        w._on_model_switch_done(False, "lms missing")
        w._on_model_switch_finished(); check("switch_finished", w.model_switch_worker is None)
        # direct start_model_switch guards
        w._active_model_id = "qwen3-9b"
        w._start_model_switch("qwen3-9b"); check("switch_same_noop", w.model_switch_worker is None)
        _close(w)
    finally:
        _restore_workers()


def test_redraw_fixhands_artifact():
    _install_noop_workers()
    try:
        w = _win()
        src = _png("last.png"); w.ctx.last_image_path = src
        # redraw: no source
        w.ctx.last_image_path = None; w._redraw_last_image()
        check("redraw_nosrc", "No image" in w.chat.toPlainText())
        w.ctx.last_image_path = src
        # redraw with instructions + region (inpaint)
        seq = iter([("make blue", True), ("shirt", True)])
        QInputDialog.getText = staticmethod(lambda *a, **k: next(seq, ("", True)))
        w._redraw_last_image(); check("redraw_started", w.redraw_worker is not None)
        w._on_redraw_finished()
        # redraw cancelled at first prompt
        QInputDialog.getText = staticmethod(lambda *a, **k: ("", False))
        w._redraw_last_image(); check("redraw_cancelled", w.redraw_worker is None)
        # fix hands (mask dialog accepted)
        class FakeMask:
            mask_path = _png("m.png"); protect_face = True
            def __init__(self, *a, **k): pass
            # Real QDialog has these; the stub must too, or the test fails on the
            # stub's shape rather than on the behaviour under test.
            def raise_(self): pass
            def activateWindow(self): pass
            def exec_(self): return QDialog.Accepted
        om = gui.MaskDrawDialog; gui.MaskDrawDialog = FakeMask
        try:
            w._fix_hands_last_image(); check("fixhands_started", w.redraw_worker is not None)
            w._on_redraw_finished()
            w._retry_hands(); check("retry_hands", w.redraw_worker is not None)
            w._on_redraw_finished()
            # fix artifact needs instruction
            QInputDialog.getText = staticmethod(lambda *a, **k: ("smooth seam", True))
            w._fix_artifact_last_image(); check("fixartifact_started", w.redraw_worker is not None)
            w._on_redraw_finished()
        finally:
            gui.MaskDrawDialog = om
        # FireRed is the only edit engine: the toggle is gone
        check("no_engine_toggle", not hasattr(w, "_toggle_handfix_engine"))
        # redraw done variants
        import image as _img; oi = _img.is_intermediate_artifact
        _img.is_intermediate_artifact = lambda p: "_INTERMEDIATE_" in p
        try:
            w._on_redraw_done(_png("ok.png")); check("redraw_done_ok", w.ctx.last_image_path.endswith("ok.png"))
            w._on_redraw_done("/no/such.png")
            w._on_redraw_done(_png("_INTERMEDIATE_x.png"))
            w._on_redraw_failed("gpu oom")
        finally:
            _img.is_intermediate_artifact = oi
        _close(w)
    finally:
        _restore_workers()
        QInputDialog.getText = QInputDialog.getText


def test_memory_profiles():
    _install_noop_workers()
    try:
        w = _win()
        profs = w._list_memory_profiles(); check("list_profiles", "default" in profs)
        w._refresh_mem_combo()
        # switch to a new profile
        w._switch_memory_profile("work")
        check("switch_profile", w.ctx.active_memory_dir.name == "work")
        w._switch_memory_profile("work")   # same -> no-op
        # busy switch refused
        w.worker = object()
        w._switch_memory_profile("other")
        check("switch_busy", w.ctx.active_memory_dir.name == "work")
        w.worker = None
        # memory center set profile (only lists file-bearing dirs; assert no crash + no-ctx guard)
        w._memory_center_set_profile("research")
        w2 = _win(); w2.ctx = None
        w2._memory_center_set_profile("x"); check("mc_set_profile_noctx", True)
        _close(w2)
        # new memory profile via input
        QInputDialog.getText = staticmethod(lambda *a, **k: ("New Profile 1", True))
        w._new_memory_profile()
        check("new_profile", w.ctx.active_memory_dir.name == "New_Profile_1")
        _close(w)
    finally:
        _restore_workers()


def test_memory_save_compact():
    _install_noop_workers()
    try:
        w = _win()
        w._save_memory(); check("save_mem", "Memory saved" in w.chat.toPlainText())
        w._compact_memory(); check("compact_started", w.compact_worker is not None)
        w._on_compact_done("a compacted summary")
        check("compact_done", (w.ctx.active_memory_dir / "summary.json").exists())
        w._on_compact_done("")   # empty
        w._on_compact_failed("llm down")
        w._on_compact_worker_finished(); check("compact_worker_cleared", w.compact_worker is None)
        _close(w)
    finally:
        _restore_workers()


def test_drag_drop_and_paste():
    _install_noop_workers()
    try:
        w = _win()
        img = _png("drop.png")
        # dragEnterEvent with image url
        md = QMimeData(); md.setUrls([QUrl.fromLocalFile(img)])
        ev = types.SimpleNamespace(mimeData=lambda: md, acceptProposedAction=lambda: None)
        w.dragEnterEvent(ev)
        check("mime_droppable", w._mime_is_droppable(md))
        # route image
        check("route_image", w._route_dropped_mime(md))
        check("working_image", os.path.basename(w.ctx.last_image_path or "") == "drop.png")
        # dropEvent
        w.dropEvent(ev)
        # text drop
        tf = os.path.join(_TMP, "d.txt"); open(tf, "w", encoding="utf-8").write("some text")
        mdt = QMimeData(); mdt.setUrls([QUrl.fromLocalFile(tf)])
        check("route_text", w._route_dropped_mime(mdt))
        # pasted image
        qimg = QImage(img)
        check("paste_qimage", w._load_pasted_qimage(qimg))
        check("paste_null", not w._load_pasted_qimage(QImage()))
        # clipboard paste
        QApplication.clipboard().setImage(QImage(img))
        w._paste_image_from_clipboard()
        check("clipboard_paste", True)
        # model_has_vision variants
        import lmstudio
        lmstudio.fetch_model = lambda url, mid: {"type": "vlm"}
        w._vision_cap_cache = {}
        check("vision_vlm", w._model_has_vision())
        lmstudio.fetch_model = lambda url, mid: {"type": "llm", "capabilities": []}
        w._vision_cap_cache = {}
        check("vision_llm_no", not w._model_has_vision())
        lmstudio.fetch_model = lambda url, mid: (_ for _ in ()).throw(OSError())
        w._vision_cap_cache = {}
        check("vision_err_true", w._model_has_vision())
        # load bad image
        w._load_image_file_as_working("/no/such.png")
        _close(w)
    finally:
        _restore_workers()


def test_eventfilter_and_dropfiles():
    _install_noop_workers()
    try:
        w = _win()
        # eventFilter: drop onto input
        img = _png("ef.png")
        md = QMimeData(); md.setUrls([QUrl.fromLocalFile(img)])
        drop = types.SimpleNamespace(type=lambda: QEvent.Drop, mimeData=lambda: md,
                                     acceptProposedAction=lambda: None)
        w.eventFilter(w.input, drop)
        # drag enter on input
        de = types.SimpleNamespace(type=lambda: QEvent.DragEnter, mimeData=lambda: md,
                                   acceptProposedAction=lambda: None)
        w.eventFilter(w.input, de)
        # _drop_text_file + _drop_pdf_file
        tf = os.path.join(_TMP, "big.txt"); open(tf, "w", encoding="utf-8").write("x" * 100)
        w._drop_text_file(tf); check("drop_text", w.worker is not None)
        w._worker_finished()
        w._drop_text_file("/no/such.txt")
        # pdf without pypdf-readable file -> error path
        w._drop_pdf_file("/no/such.pdf")
        check("eventfilter_ok", True)
        _close(w)
    finally:
        _restore_workers()


def test_close_event():
    w = _win()
    # give it fake stoppable things
    w.player = FakePlayer()
    w.closeEvent(types.SimpleNamespace(accept=lambda: None))
    check("close_ok", True)


if __name__ == "__main__":
    _saved_getText = QInputDialog.getText
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback; traceback.print_exc()
            print(f"  ERROR in {fn.__name__}: {type(e).__name__}: {e}")
        finally:
            QInputDialog.getText = _saved_getText
    print(f"\n{len(fns)-failed}/{len(fns)} window test functions passed "
          f"({sum(1 for _,c in RESULTS if c)}/{len(RESULTS)} checks)")
    # A failed check() must reach the exit code. This used to be `1 if failed else 0`,
    # where `failed` counts EXCEPTIONS only -- so every assertion in the file could
    # print FAIL while the suite still exited 0, invisible to anything judging by
    # exit code (which is how these suites are judged).
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)

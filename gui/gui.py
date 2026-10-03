"""PyQt5 desktop dashboard for the assistant.

Launched from assistant.main() when config.USE_GUI is true. Layout:

  ┌ commands + audio viz ┬ camera + chat + input ┬ Images / Search / Terminal ┐

All heavy work (Whisper, the LangGraph agent, F5-TTS) runs on background
QThreads so the window, camera preview and audio visualizer stay smooth.
"""
import copy
import json
import logging
import os
import random
import re
import shutil
import sys
import threading
import time
from pathlib import Path

# Before cv2/PyQt5: PyQt5 bundles MSVCP140 14.26 and whatever loads first wins;
# torch & co. crash in the old copy (0xc0000005). Every way into the GUI goes
# through here, not only launch_all/assistant (core/win_runtime.py).
import win_runtime
win_runtime.preload_newest_msvcp()

import cv2
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout, QGroupBox,
    QToolButton, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
    QMenu, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QRadioButton, QSpinBox,
    QStackedWidget, QStatusBar, QTableWidget, QTableWidgetItem, QTabWidget, QTextEdit,
    QVBoxLayout, QWidget,
)
from PyQt5.QtCore import QSettings

# QtWebEngine MUST be imported before any QApplication is created (PyQt5 hard rule),
# so import it at module load — long before run_gui() builds the app. Used to render
# research reports with typeset math + real tables; None => fall back to setMarkdown.
try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView as _QWebEngineView
except Exception as _webeng_exc:  # pragma: no cover - environment dependent
    # Say WHY on the way down. Without this the report viewer silently degrades
    # to markdown and the machine gives no hint which import failed.
    logging.getLogger("assistant.gui").debug(
        "QtWebEngine unavailable, falling back to markdown reports: %s", _webeng_exc)
    _QWebEngineView = None
# Escape hatch: if the WebEngine report viewer ever misbehaves on a given machine,
# set GUI_REPORT_HTML=0 to instantly revert to the (always-working) markdown view.
_REPORT_HTML_ENABLED = os.getenv("GUI_REPORT_HTML", "1").strip().lower() not in ("0", "false", "no", "off")

import config
from config import SAMPLE_RATE, LM_STUDIO_BASE, MODEL_NAME, MIC_GAIN_DB, MEMORY_DIR
from audio import AudioPlayer, MicRecorder, VadListener, transcribe_audio_array
import gui_i18n
import ui_scale
from ui_scale import px, pt, scale_style as _ss

logger = logging.getLogger("assistant.gui")

# Shared GUI toolkit (theme, stylesheet, widget helpers) extracted to
# gui_common.py so the per-tab modules can sit on it without importing gui.
# Re-exported in full ON PURPOSE. pyflakes calls FlowLayout and
# _make_tab_close_icons unused here, but the suites address all of these
# through `gui` (gui.FlowLayout appears 7 times, gui._make_tab_close_icons 5),
# and gui.py is the public face of the GUI. Trimming to satisfy the linter
# would break those callers for no gain.  # noqa: F401
import gui_common
import gui_workers
from gui_common import (
    ACCENT2, MUTED, ICON_PATH, _crash_log_stage, enable_dark_titlebar, _make_tab_close_icons,
    build_qss, _esc, _shadow, _card, _section, FlowLayout, _FlowWidget, _flow, _fmt_ts,
    _fmt_bytes,
)
# Memory Center tab, extracted to gui_memory_tab.py. CompactMemoryWorker is
# re-exported because AssistantWindow drives it as well as the tab.
from gui_memory_tab import MemoryCenterTab, CompactMemoryWorker
# Cross-tab plumbing that lives in gui_common so extracted tabs need not
# import gui.py: _ScopedCtx (four tabs) and TranscribeWorker (two).
from gui_common import TranscribeWorker
# The Madhouse (tab, grid, workers, reply/speaker logic) extracted whole.
# AssistantWindow constructs MadhouseTab; the rest are re-exported because the
# suites address them through `gui`.
# Storyboard tab (layout-first drawing and editing) with its box canvas and
# four workers. Re-exported because the suites address them through `gui`.
# Telegram panel AND its settings scope. redirect_settings is re-exported
# because tests call gui.redirect_settings(tmpdir) to keep the real BotFather
# token safe; it must stay in the same module as the flag it sets.
# Shared image dialogs (viewer + mask editor) used by AssistantWindow,
# ImagesPanel and the transfer tab, so they live in neither.
# Startup settings dialog + the UI-scale application that must happen before
# any widget is built. Both consumers (AssistantWindow, run_gui) are here, so
# tests patching gui.SettingsDialog still intercept.
import gui_settings_dialog
from gui_settings_dialog import SettingsDialog, apply_ui_scale
from gui_model_config_tab import ModelConfigTab, ReloadModelWorker
from gui_stress_tab import StressTab
from gui_music_tab import MusicTab, MusicWorker
from gui_weather_tab import WeatherTab, WeatherWorker
from gui_mashup_tab import MashupTab, MashupWorker
from gui_audio_visualizer import AudioVisualizer, _wav_envelope
from gui_log_bridge import QtLogHandler
from gui_chrome import StageIndicator, _DarkTitleBarFilter
# Re-exported: no longer called from gui.py itself (gui_research_tab imports
# its own copy — pure rendering, never patched), but the suites still call
# gui._render_report_html directly to check the generated markup.
from gui_report_html import _render_report_html
from gui_layout_mixin import LayoutMixin
from gui_database_tab import DatabaseTabMixin
from gui_voice_tab import VoiceMixin
# Drag-drop, clipboard paste and the chat/input event filter.
from gui_dropzone import DropPasteMixin
# The staged-task queue: bar, list rendering and the drain state machine.
from gui_queue import TaskQueueMixin
# The Manual Control panel (the DR_* pipeline knobs) and its reset/read logic.
from gui_manual_panel import ManualControlMixin
# The research phase strip (stepper, caption, stat chips).
from gui_research_progress import ResearchProgressMixin
# The research run + report view. Reaches _QWebEngineView, _REPORT_HTML_ENABLED
# and DeepResearchWorker back through `gui` at call time — those three are
# patched directly on this module by the suites, so the mixin cannot bind them
# by value without breaking every one of those patches.
from gui_research_tab import ResearchTabMixin
# Redraw / fix-hands / fix-artifact on the last generated image. Reaches
# RedrawWorker, FixHandsWorker, FixArtifactWorker and MaskDrawDialog back
# through `gui` at call time for the same reason.
from gui_image_fix import ImageFixMixin
# Memory-profile list/switch/create/save/compact. Reaches MEMORY_DIR and
# CompactMemoryWorker back through `gui` at call time for the same reason.
from gui_memory_profile import MemoryProfileMixin
# Writing lines into the chat transcript (text, images, video cards).
from gui_chat_view import ChatViewMixin
# The busy gate, live-thread enumeration and the teardown join.
from gui_busy_state import BusyStateMixin
# The dashboard page: the three-column layout and the tab stack.
from gui_dashboard import DashboardMixin
# Every background worker. AssistantWindow constructs most of these directly,
# so the suites that patch gui.ModelLoader et al. keep intercepting. Kept here
# even though DeepResearchWorker's only caller (gui_research_tab) now reaches
# it through `gui` rather than by name — the binding still has to exist for
# that lookup, and for gui.DeepResearchWorker patches to have anything to
# patch.
from gui_workers import (ModelLoader, ModelSwitchWorker, RedrawWorker,
                         FixHandsWorker, FixArtifactWorker,
                         DeepResearchWorker)
from gui_images_panel import ImagesPanel
from gui_system_info_tab import SystemInfoTab
from gui_dialogs import (ImageViewerDialog, show_image_viewer, MaskCanvas,
                         MaskDrawDialog, OUTPUT_DIR_GUI_MASK)
from gui_transfer_tab import TransferTab, TransferWorker, _TRANSFER_ROLES
from gui_telegram_tab import (
    TelegramTab, redirect_settings, _tg_settings, _TG_SETTINGS_SCOPE, _TG_TOKEN_KEY,
    _TG_ADMINS_KEY,
)
from gui_madhouse_tab import (
    MadhouseTab, MadhouseGrid, MadhouseReplyWorker, generate_madhouse_reply,
    choose_next_speaker, _name_is_addressed,
)


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class AssistantWindow(LayoutMixin, DatabaseTabMixin, VoiceMixin, DropPasteMixin,
                      TaskQueueMixin, ManualControlMixin, ResearchProgressMixin,
                      ResearchTabMixin, ImageFixMixin, MemoryProfileMixin, ChatViewMixin,
                      BusyStateMixin, DashboardMixin, QMainWindow):
    stage_changed = pyqtSignal(str)  # emitted (possibly from a worker thread) with the current stage
    # Emitted by the VAD listener thread with a finished utterance (np.ndarray);
    # the queued connection hops it onto the Qt main thread.
    vad_utterance = pyqtSignal(object)

    def __init__(self, model_name, no_think, reasoning_effort="high"):
        super().__init__()
        self._initial_reasoning_effort = reasoning_effort
        self.setWindowTitle("Assistant DC")
        self.setWindowIcon(QIcon(ICON_PATH))
        # Open at the scaled default, but never larger than the screen actually
        # available (a 2x-scaled default on a small panel must still fit on screen).
        w, h = self._default_window_size()
        self.resize(w, h)
        self._drag_offset = None
        self._faux_max = False      # frameless "maximize" = filled work area (see _toggle_max)

        self.ctx = self.base_state = self.graph = self.worker = None
        self.compact_worker = None
        # Drives the audio visualizer from the live mic level while recording.
        self._mic_viz_timer = QTimer(self)
        self._mic_viz_timer.setInterval(45)
        self._mic_viz_timer.timeout.connect(self._pump_mic_level)
        self.redraw_worker = None
        self.research_worker = None
        self.ultra_search_on = False
        # Document database ("Use Database" RAG mode): when ON, each typed question first
        # retrieves the most relevant passages from the local Library and injects them
        # into the prompt. Lazy-opened on first use; rebuilt fresh after each build.
        self.usedb_on = False
        self._library = None
        self._lib_build_worker = None
        self._scan_worker = None
        self._last_db_stats = {}
        # Task queue: stage multiple actions to run sequentially. Each entry is a dict
        # {"text": str, "status": pending|running|done|failed}. The queue auto-drains when
        # idle and not paused; user can add while busy, reorder, remove, clear, pause.
        # Execution-state tracking: the running task stays visible; on completion it is
        # marked done/failed and removed; _completed_count/_failed_count summarise history.
        self._task_queue = []
        self._queue_paused = False
        self._running_task = None
        self._running_dispatched = False   # has the deferred dispatch actually fired?
        self._turn_had_error = False
        self._turn_cancelled = False       # sticky: ctx.cancel_event is cleared too early
        self._completed_count = 0
        self._failed_count = 0
        self._cancelled_count = 0
        self.cap = None
        self.current_frame = None
        self.recorder = None
        self.pending_image = None
        self.captured_image = None
        self.player = AudioPlayer()
        self.cam_timer = QTimer(self)
        self.cam_timer.timeout.connect(self._update_camera)
        # Hands-free VAD mode: the listener runs on its own thread; the timer
        # keeps it deaf while a request runs or the assistant's voice plays.
        self.vad_listener = None
        self.transcribe_worker = None
        self._pending_topic = ""
        self.vad_utterance.connect(self._on_vad_utterance)
        self.vad_timer = QTimer(self)
        self.vad_timer.setInterval(250)
        self.vad_timer.timeout.connect(self._vad_tick)
        self.vad_timer.start()

        self.setAcceptDrops(True)
        self._build_ui()
        self._install_log_handler()
        # Restore the last-used layout (window geometry, splitter sizes, which panels are
        # open and in what order). Safe no-op on first run / if nothing was saved.
        self._restore_last_layout()

        self.model_switch_worker = None
        # Single source of truth for the model that is actually loaded in LM Studio.
        # Set on a confirmed load; the switch path compares against THIS (not the
        # combo / ctx, which are mutated speculatively) so reselecting the active
        # model is a guaranteed no-op and rapid clicks can't trigger overlapping loads.
        self._active_model_id = model_name
        self.loader = ModelLoader(model_name, no_think, reasoning_effort)
        self.loader.ready.connect(self._on_runtime_ready)
        self.loader.failed.connect(self._on_load_failed)
        self.loader.progress.connect(self._on_load_progress)
        self.loader.start()

    # ---- layout ----
    def _build_ui(self):
        self.stack = QStackedWidget()
        self.loading_page = self._build_loading_page()
        self.dashboard = self._build_dashboard()
        self.stack.addWidget(self.loading_page)
        self.stack.addWidget(self.dashboard)
        self.stack.setCurrentWidget(self.loading_page)

        if gui_common.NATIVE_FRAME:
            self.setCentralWidget(self.stack)
        else:
            self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
            central = QWidget()
            v = QVBoxLayout(central)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(0)
            v.addWidget(self._build_titlebar())
            v.addWidget(self.stack, 1)
            self.setCentralWidget(central)

        self.setStatusBar(QStatusBar())
        self.statusBar().setSizeGripEnabled(True)  # resize handle (works on frameless too)
        self._set_status("Loading models…")

    # NB: titlebar dragging is handled by the bar's own mousePressEvent/Move/Release
    # (set in _build_titlebar), NOT by eventFilter — AssistantWindow defines a second
    # eventFilter (the chat-drop/Ctrl-V handler) later in the class, which in Python
    # SHADOWS any earlier eventFilter. The old event-filter drag lived in that dead
    # earlier copy and never ran; this is why the window wouldn't move.

    def _build_loading_page(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.addStretch(1)
        card, clay = _card()
        card.setMaximumWidth(px(440))
        clay.setSpacing(16)
        title = QLabel("Assistant DC"); title.setObjectName("title"); title.setAlignment(Qt.AlignCenter)
        self.loading_subtitle = QLabel("Loading models, please wait…")
        self.loading_subtitle.setStyleSheet(f"color:{MUTED};"); self.loading_subtitle.setAlignment(Qt.AlignCenter)
        bar = QProgressBar(); bar.setRange(0, 0); bar.setTextVisible(False); bar.setFixedHeight(px(8))
        clay.addWidget(title); clay.addWidget(self.loading_subtitle); clay.addWidget(bar)
        _shadow(card)
        hbox = QHBoxLayout(); hbox.addStretch(1); hbox.addWidget(card); hbox.addStretch(1)
        lay.addLayout(hbox)
        lay.addStretch(1)
        return page

    def _settings(self):
        return QSettings("AssistantDC", "Layout")

    def _registry_get(self, key):
        for k, w, label in self._tab_registry:
            if k == key:
                return w, label
        return None

    def _install_log_handler(self):
        self.log_handler = QtLogHandler()
        self.log_handler.setLevel(logging.INFO)
        self.log_handler.line.connect(self._append_log)
        self.log_handler.line.connect(self._on_pipeline_log)
        asst = logging.getLogger("assistant")
        asst.addHandler(self.log_handler)
        try:
            import log_redact
            log_redact.install(self.log_handler)
        except Exception:
            pass
        # CRASH FIX: datasets/transformers (pulled in by sentence-transformers) call
        # logging.basicConfig(), adding a root StreamHandler to sys.stderr — which is
        # None under pythonw AND cp1251-encoded. A non-ASCII log char (→, Cyrillic)
        # there raised UnicodeEncodeError that escaped and killed the research worker.
        # Our QtLogHandler is attached directly here, so stop propagating to root.
        asst.propagate = False
        # Belt-and-suspenders: a logging hiccup must NEVER crash a worker thread again.
        logging.raiseExceptions = False

    def _on_pipeline_log(self, line: str):
        """Tail pipeline log lines into the Research progress strip's log bar while
        a run is active, so slow/silent stages (cross-encoder load, reranking,
        synthesis) always show a heartbeat. Rerank lines are highlighted + 'stick'
        so the rerank stage can never look like a frozen dead loop again."""
        if not getattr(self, "_research_active", False):
            return
        bar = getattr(self, "_log_bar", None)
        if bar is None:
            return
        # Strip the leading "HH:MM:SS  " timestamp the handler prepends; keep "name: msg".
        msg = line.split("  ", 1)[1] if "  " in line else line
        is_rerank = "rerank" in msg.lower()
        bar.setText(msg[:160])
        col = ACCENT2 if is_rerank else MUTED
        bar.setStyleSheet(_ss(f"color:{col}; font-family:Consolas,monospace; font-size:10px;"))
        self._log_bar_tag.setStyleSheet(_ss(f"color:{col}; font-size:11px;"))
        bar.setToolTip(msg)

    # ---- runtime ----
    def _on_runtime_ready(self, ctx, base_state, graph):
        # This slot runs on the GUI thread via a queued signal right after the model
        # loader thread finishes ("LLM ready" in the log). Each step is fenced so a
        # failure in one panel cannot escape the slot — an exception escaping here is
        # exactly what used to abort the process with 0xC0000409. Per-step lifecycle
        # logs pinpoint the offending panel in crash.log / console.
        def _step(name, fn):
            try:
                _crash_log_stage(f"_on_runtime_ready: {name}")
                fn()
            except Exception:
                logger.exception("_on_runtime_ready step failed: %s", name)

        self.ctx, self.base_state, self.graph = ctx, base_state, graph
        # The loader loaded this model exclusively → it is now the active model.
        self._active_model_id = ctx.model_name
        # Carry the main-screen reply-length selection into the freshly-built context
        # (it defaults to "auto"; the user may have changed the selector before load).
        if getattr(self, "length_combo", None) is not None:
            _step("apply response length",
                  lambda: self._on_length_changed(self.length_combo.currentIndex()))
        ctx.stage_callback = self.stage_changed.emit
        _step("switch to dashboard", lambda: self.stack.setCurrentWidget(self.dashboard))
        _step("set_busy(False)", lambda: self._set_busy(False))
        _step("clear status", lambda: self._set_status(""))
        _step("refresh chip", self._refresh_chip)
        _step("update voice btn", self._update_voice_btn)
        _step("model_config_tab.set_context",
              lambda: self.model_config_tab.set_context(LM_STUDIO_BASE, ctx.model_name))
        _step("system_info_tab.set_context",
              lambda: self.system_info_tab.set_context(ctx, LM_STUDIO_BASE))
        if hasattr(self, "memory_center_tab"):
            _step("memory_center_tab.refresh_all", self.memory_center_tab.refresh_all)
        _step("refresh mem combo", self._refresh_mem_combo)
        if hasattr(self, "telegram_tab"):
            _step("telegram autostart", self.telegram_tab.maybe_autostart)
        try:
            mem_count = len(ctx.session_memory)
            profile = ctx.active_memory_dir.name
            note = (f" Profile: '{profile}', {mem_count} memory items loaded."
                    if mem_count else f" Profile: '{profile}'.")
            if self._model_missing():
                # Do not say "ask me anything" when nothing can answer. The
                # greeting is the first thing the user reads, and a cheerful
                # one here is what makes the later silence confusing.
                self._add_system(
                    "Whisper and F5-TTS are loaded but no LLM is — as you "
                    "chose at startup, so VRAM is free. Chat and voice will "
                    "wait; the tabs (training, files, database) work. When "
                    "you need the model: ⚙ Settings → Model Config → "
                    f"Apply.{note}")
        except Exception:
            logger.exception("_on_runtime_ready: final system message failed")
        # Say it at startup, not only when a render comes back empty. A whole-
        # card job blocks every picture, video and music job in the app, and the
        # user is usually the one who started it -- being reminded once is much
        # cheaper than discovering it from a failed render ten minutes in.
        try:
            import comfy_client as _cc
            _busy_by = _cc.gpu_holder()
            if _busy_by:
                self._add_system(
                    f"🛠 The GPU is busy with «{_busy_by}» — drawing, video "
                    "and music are unavailable until it finishes. Training "
                    "progress is on the Characters tab.")
            elif not _cc.server_healthy():
                # Only when the card is FREE: during a training run ComfyUI is
                # deliberately not running, and two notices about the same
                # situation is one too many.
                self._add_system(
                    '🎨 ComfyUI is not responding — drawing, video and music will not work until it is running. Everything else works.')
        except Exception:
            logger.exception("_on_runtime_ready: gpu holder notice failed")
        _crash_log_stage("_on_runtime_ready: complete")

    def _on_load_progress(self, msg):
        self.loading_subtitle.setText(msg)

    def _on_load_failed(self, msg):
        self.loading_subtitle.setText(f"Load failed: {msg}")
        self._set_status(f"Load failed: {msg}")

    def _refresh_chip(self):
        if self.ctx:
            # An empty model_name is the "no model" choice; without a name of
            # its own the chip read "  ·  thinking ON", which looks like a
            # rendering glitch rather than a state.
            name = (self.ctx.model_name or "").strip() or 'no model'
            self.model_chip.setText(f"{name}  ·  thinking {'OFF' if self.ctx.no_think else 'ON'}")

    def _set_status(self, text):
        self.statusBar().showMessage(text)

    def _append_log(self, line):
        self.terminal.append(line)
        if self.stack.currentWidget() is self.loading_page:
            self.loading_subtitle.setText(line.split(": ", 1)[-1][:64])

    def _on_chat_scroll(self, value):
        sb = self.chat.verticalScrollBar()
        at_bottom = value >= sb.maximum() - 4
        self._scroll_down_btn.setVisible(not at_bottom)
        if not at_bottom:
            vp = self.chat.viewport()
            # Place 60px from right (clear of the scrollbar) and 40px from bottom
            self._scroll_down_btn.move(vp.width() - px(60), vp.height() - px(48))
            self._scroll_down_btn.raise_()

    # ---- chat ----

    # ---- sending ----
    def _on_length_changed(self, idx):
        """Apply the main-screen reply-length selector to the live context."""
        if self.ctx is not None and 0 <= idx < len(self._LEN_VALUES):
            self.ctx.response_length = self._LEN_VALUES[idx]
            logger.info("Response length set to %r (main screen)", self.ctx.response_length)

    def _send_text(self):
        text = self.input.text().strip()
        if not text:
            return
        # Refuse before the busy check, not after: without a model the message
        # would otherwise be queued, and a queue that can never drain is the
        # silent failure this gate exists to prevent. The text is LEFT in the
        # input box so it is not lost while the user goes to load a model.
        if self._refuse_without_model():
            return
        if self._busy():
            # Busy (model loading, database build, or a turn in flight) -> stage the message
            # in the queue instead of dropping it. It runs when the assistant frees up; the
            # queue auto-drains the moment _busy() clears (model ready / build done / turn end).
            self.input.clear()
            self._enqueue(text)
            return
        self.input.clear()
        self._dispatch_user_text(text)

    # ---- the "no model loaded" gate ----
    NO_MODEL_NOTICE = ('No model is loaded — the app started without one (VRAM is free). Pick a model: ⚙ Settings → Model Config → Apply, then try again.')

    def _model_missing(self) -> bool:
        """True when the runtime is up but no LLM was ever chosen.

        This is the state the "do not load a model" startup option creates. It
        is NOT busy -- the graph exists, the tabs work, the queue drains -- so
        without an explicit gate a message walks into a graph whose model_name
        is empty and comes back as a bare "Модель не ответила", which reads as
        LM Studio being broken rather than as the choice the user made a minute
        earlier. Answer honestly instead, and say where the switch is.
        """
        ctx = getattr(self, "ctx", None)
        return ctx is not None and not (getattr(ctx, "model_name", "") or "").strip()

    def _refuse_without_model(self) -> bool:
        """Report the missing model once and tell the caller to stop."""
        if not self._model_missing():
            return False
        self._add_system(self.NO_MODEL_NOTICE)
        self._set_status('No model loaded')
        return True

    def _dispatch_user_text(self, text):
        """Actually run a user message (shared by direct send and queued drain)."""
        self._add_user(text)
        if self._refuse_without_model():
            return
        if self.ultra_search_on:
            self._start_deep_research(text)
            return
        if self.usedb_on and getattr(self, "db_scan_chk", None) is not None \
                and self.db_scan_chk.isChecked():
            self._start_scan(text)
            return
        image = self._take_captured_image()  # attach an explicitly captured frame, if any
        self._start_worker(gui_workers.RequestWorker(self.ctx, self.graph, self.base_state,
                                         text=text, image=image, use_db=self.usedb_on,
                                         db_k=self._db_k()))

    def _start_worker(self, worker):
        # Second gate, on purpose: voice and the drop zone reach a worker
        # WITHOUT going through _dispatch_user_text, and a dropped photo that
        # silently does nothing is the same invisible failure.
        if self._refuse_without_model():
            return
        if self.ctx is not None:
            self.ctx.cancel_event.clear()  # fresh operation — clear any prior cancel
        self.worker = worker
        self._set_busy(True)
        self.stage.set_stage("Working")  # immediate feedback; graph refines the stage
        worker.recognized.connect(self._add_user)
        if hasattr(worker, "info"):
            worker.info.connect(self._add_system)
        worker.done.connect(self._on_done)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._worker_finished)
        worker.start()

    def _empty_turn_diagnosis(self) -> str:
        """Why did this turn produce nothing at all?

        One message covered four different situations, and told the user to
        check LM Studio in all of them: LM Studio really being down, the model
        being evicted from under us, the model being loaded with too small a
        context to accept a tool-bearing call, and the model simply answering
        with an empty string. Only the first is actually about LM Studio being
        started, so guessing wrong here costs the user a restart they did not
        need. Every probe is a local HTTP call with a short timeout and is
        wrapped: a diagnosis that can itself raise inside a Qt slot is worse
        than a vague message.
        """
        fallback = ('The model did not answer. Check that LM Studio is running and the model is loaded (Model tab), then try again.')
        try:
            from lmstudio import fetch_models, loaded_context_length
            models = fetch_models(LM_STUDIO_BASE)
            if not models:
                return ('LM Studio is not answering at ' + str(LM_STUDIO_BASE) +
                        ' — start it and enable the local server, then repeat the request.')
            want = (getattr(self.ctx, "model_name", "") or "").strip()
            loaded = [m.get("id") for m in models if m.get("state") == "loaded"]
            if want and want not in loaded:
                return (f"Model «{want}» is not loaded in LM Studio right now "
                        + (f"(loaded: {', '.join(str(x) for x in loaded)}). "
                           if loaded else '(nothing is loaded). ')
                        + 'Pick it again: Settings → Model Config → Apply.')
            # A model loaded at 8192 tokens rejects EVERY tool-bearing call with
            # "n_keep >= n_ctx" and returns nothing, which is indistinguishable
            # from silence unless it is named.
            ctx_len = loaded_context_length(LM_STUDIO_BASE, want) if want else 0
            # Per INSTANCE, not divided between parallel slots: that theory was
            # measured and disproved (see lmstudio.loaded_parallel). Nothing
            # here shells out to `lms ps` either -- this runs inside a Qt slot,
            # and a 30-second subprocess would freeze the window.
            if 0 < ctx_len < 16384:
                return (f"The model is loaded with a context of only {ctx_len} tokens — "
                        "not enough for the system prompt with tools, so the "
                        "request is dropped silently. Reload it with a context "
                        "of 16384 or more (Model Config tab).")
            return ('The model answered with an empty message. This happens on a long context — try again or clear the context.')
        except Exception:
            logger.exception("_empty_turn_diagnosis failed")
            return fallback

    def _on_done(self, final):
        answer = (final.get("final_answer") or "").strip()
        if answer:
            self._add_assistant(answer)
        elif not any(final.get(k) for k in ("image_status", "image_path",
                                            "research_report", "video_path",
                                            "document_path")):
            # The turn produced nothing at all. Without feedback the user just sees
            # "Ready" and silence — this happens when LM Studio is down/crashed
            # (all retries exhausted) or the user pressed Stop mid-turn.
            if self.ctx is not None and self.ctx.cancel_event.is_set():
                self._add_system('Operation stopped.')
            else:
                self._add_system(self._empty_turn_diagnosis())
        img = final.get("image_path")
        # Show only images an image TOOL produced this turn. image_status is set
        # solely by generate/redraw/inpaint handlers; the vision node also fills
        # image_path with the user's own sent photo (as the working image), and
        # that must not be re-posted as "generated".
        if (img and os.path.exists(img)
                and final.get("image_status") in ("success", "partial", "ok")):
            # Final UI gate: never surface a cropped intermediate working tile.
            import image as _img
            if _img.is_intermediate_artifact(img):
                logger.error("UI DELIVERY REJECTED: intermediate tile reached _on_done: %s", img)
                self._add_system('Internal error: the editor returned an intermediate fragment, not the finished image. The result is not shown.')
            else:
                self.images_panel.add_image(img)
                self._add_result_image(img)  # inline in chat; double-click to view full size
        vid = final.get("video_path")
        if (vid and os.path.exists(vid)
                and final.get("video_status") in ("success", "partial", "ok")):
            self._add_result_video(vid)
        doc = final.get("document_path")
        # document_path was already read above just to suppress the "empty
        # turn" fallback -- the actual file was never surfaced to the user
        # here, so a successfully created presentation/report vanished from
        # the desktop app's chat with no path shown and no way to open it.
        if (doc and os.path.exists(doc)
                and final.get("document_status") in ("success", "partial", "ok")):
            self._add_result_document(doc)
        report = final.get("research_report")
        if report:
            self._set_report(report)
            self._focus_tab("research")
            self._add_assistant("(research report ready — see the Research tab)")
        self._refresh_search_panel()
        tts = final.get("tts_path")
        tts_muted = self.ctx and self.ctx.tts_disabled
        if tts and os.path.exists(tts) and not tts_muted:
            self.viz.play_file(tts)
            self.player.play(tts)
            self.pause_btn.setText("⏸  Pause speech")

    def _toggle_pause(self):
        paused = self.player.toggle_pause()
        self.viz.set_paused(paused)
        self.pause_btn.setText("▶  Resume speech" if paused else "⏸  Pause speech")

    def _on_failed(self, msg):
        # Record failure so the queue can mark a running task ✗ (execution-state tracking).
        self._turn_had_error = True
        self._add_system(msg)

    def _worker_finished(self):
        self.worker = None
        if self.ctx is not None:
            self.ctx.cancel_event.clear()
        self.stage.set_stage("Ready")
        self._set_busy(False)
        self._set_status("")
        self._refresh_chip()

    def _toggle_ultra(self):
        self.ultra_search_on = not self.ultra_search_on
        if self.ultra_search_on:
            self.ultra_btn.setText("🔬  Ultra Search: ON")
            self.ultra_btn.setStyleSheet(f"background:{ACCENT2}; color:#06231d;")
            self.input.setPlaceholderText("Enter a research topic and press Enter…")
            self._set_status("Ultra Search ON — your next message is a research topic.")
        else:
            self.ultra_btn.setText("🔬  Ultra Search: OFF")
            self.ultra_btn.setStyleSheet("")
            self.input.setPlaceholderText("Type a message and press Enter…")
            self._set_status("Ultra Search OFF.")

    # ---- cancellation ----
    def _cancel_current(self):
        if self.ctx is None or not self._busy():
            return
        if self.ctx.cancel_event.is_set():
            return  # already cancelling — repeated Stop clicks shouldn't spam the chat
        self.ctx.cancel_event.set()
        self._turn_cancelled = True
        if self._scan_worker is not None:    # the scan loop polls its own flag, not cancel_event
            self._scan_worker.cancel()
        # Tabs that run long jobs hold their own cancel token (see _ScopedCtx), so
        # setting the global flag alone no longer reaches them.
        for attr in ("transfer_tab", "storyboard_tab", "code_tab"):
            tab = getattr(self, attr, None)
            cancel = getattr(tab, "cancel", None) if tab is not None else None
            if callable(cancel):
                try:
                    cancel()
                except Exception:
                    logger.exception("cancel failed for %s", attr)
        self._add_system("Stopping the current operation…")
        self.stage.set_stage("Cancelling")
        self._set_status("Cancelling — finishing the current step…")

    # ---- clear context ----
    def _clear_context(self):
        if self.ctx is None:
            return
        # A mid-flight turn would write the old chat history back into base_state
        # when it finishes, resurrecting the context we just cleared.
        if self._busy():
            self._add_system('Wait for the current answer to finish before clearing the context.')
            return
        with self.ctx.memory_lock:
            self.ctx.session_memory.clear()
        prof = self.ctx.active_memory_dir
        for fn in ("summary.json", "session_memory.json"):
            try:
                (prof / fn).unlink(missing_ok=True)
            except Exception as exc:
                logger.warning("Could not delete %s: %s", fn, exc)
        self.ctx.save_memory(prof)  # persist the now-empty session
        # The chat history in base_state["messages"] is what the model actually
        # "remembers" each turn — without resetting it, the next reply replays the
        # old context, ctx.remember() re-saves derivatives of it, and closeEvent
        # persists them again: the cleared "crap" comes back across sessions.
        from app_runtime import make_base_state
        self.base_state = make_base_state()
        # Drop per-conversation working context too — a full reset means the next
        # turn starts from nothing.
        self.captured_image = None
        self.ctx.last_image_path = None
        self.ctx.last_image_prompt = ""
        self.ctx.last_research_report = ""
        self.ctx.last_research_path = None
        self.ctx.total_user_turns = 0  # restart the rolling-compaction counter
        # Actually return the freed memory to the OS / CUDA allocator instead of just
        # dropping references — clearing context should lighten the process, not only
        # the screen.
        from utils import free_process_memory
        free_process_memory()
        # Wipe the visible session as well: a cleared "memory" that still shows the
        # whole dialog, generated images and the research report on screen is both
        # confusing and an information leak.
        self.chat.clear()
        self.images_panel.clear_images()
        self._reset_report_view()
        self.system_info_tab.refresh()
        self._add_system("The conversation context is cleared — history, images and reports are gone. "
                         f"The pinned facts of profile «{prof.name}» are kept and still active.")
        self._set_status("Context cleared.")

    # ---- settings ----
    def _open_settings(self):
        dlg = SettingsDialog(MODEL_NAME, ctx=self.ctx, parent=self)
        dlg.raise_()
        dlg.activateWindow()
        if dlg.exec_() == QDialog.Accepted:
            mid, no_think, effort = dlg.result_choice()
            scale_changed = dlg.apply_ui_scale_setting()
            if dlg.apply_language_setting():
                gui_i18n.set_language(ui_scale.language())   # the whole window, in place
            if mid == gui_settings_dialog.NO_MODEL:
                # "Do not load a model": the card is wanted for something else
                # (a LoRA training run needs ~18 of 24 GB, and a 19 GB chat
                # model beside it does not fit). Everything already resident is
                # evicted; the chosen model is left untouched so the next
                # deliberate pick still knows what it was.
                import lora_training as _LT   # local: gui must not pay for it at import
                _LT.free_gpu(log=self._add_system)
                # Record the state, do not just free the memory. Without this
                # ctx.model_name still named the evicted model, so every gate in
                # the app believed a model was loaded and messages went on being
                # sent to an LM Studio that had nothing resident -- which comes
                # back as "Модель не ответила", the exact confusion this option
                # was meant to remove. Clearing _active_model_id too keeps a
                # later reselect of the SAME model a real load rather than a
                # "Model already loaded." no-op.
                if self.ctx is not None:
                    self.ctx.model_name = ""
                self._active_model_id = ""
                self._refresh_chip()
                self.model_config_tab.set_context(LM_STUDIO_BASE, "")
                self.system_info_tab.refresh()
                self._set_status('No model loaded — VRAM is free.')
                self._add_system(
                    'No model is loaded, as you chose. Chat and drawing are unavailable until you pick a model in Settings.')
                return
            if self.ctx is not None:
                # Compare against the model that is ACTUALLY loaded, not ctx.model_name
                # (which we're about to overwrite). Reselecting the active model →
                # model_changed is False → no unload/reload/worker restart.
                model_changed = mid != getattr(self, "_active_model_id", self.ctx.model_name)
                # reasoning_effort takes effect on the NEXT call — no reload needed.
                self.ctx.model_name, self.ctx.no_think = mid, no_think
                self.ctx.reasoning_effort = effort
                dlg.apply_settings()
                # Keep the main-screen length selector in sync with the Settings choice.
                if getattr(self, "length_combo", None) is not None:
                    _lv = getattr(self.ctx, "response_length", "auto")
                    if _lv in self._LEN_VALUES:
                        self.length_combo.blockSignals(True)
                        self.length_combo.setCurrentIndex(self._LEN_VALUES.index(_lv))
                        self.length_combo.blockSignals(False)
                if scale_changed:
                    self._add_system(
                        "UI scale updated. Fonts and styling changed now; restart the "
                        "app for every panel to fully re-measure at the new scale.")
                # If voice was just muted, stop any ongoing playback
                if self.ctx.tts_disabled:
                    self.player.stop()
                    self.viz.stop()
                self._update_voice_btn()
                mic_ok = not self.ctx.mic_disabled and self.worker is None
                self.talk_btn.setEnabled(mic_ok)
                self._refresh_chip()
                self.model_config_tab.set_context(LM_STUDIO_BASE, mid)
                self.system_info_tab.refresh()
                if model_changed:
                    self._start_model_switch(mid)
                else:
                    self._set_status("Settings applied.")

    def _start_model_switch(self, model_id):
        """Actually load the newly chosen model in LM Studio (unloads the old one).

        Blocks chat input for the duration so a request can't hit LM Studio while
        it is mid-reload.
        """
        # Idempotency guard (defense in depth): never enter the load/unload path for
        # the model that's already active. Belt-and-suspenders with the model_changed
        # check in _open_settings, so any future caller is safe too.
        if model_id == getattr(self, "_active_model_id", None):
            self._set_status("Model already loaded.")
            return
        # Reentrancy guard: a switch is already in flight → ignore rapid repeat clicks
        # and overlapping requests rather than spawning a second worker.
        if self.model_switch_worker is not None:
            self._set_status("Model switch already in progress…")
            return
        self._add_system(f"Switching model to '{model_id}' — unloading the old one…")
        self.model_switch_worker = ModelSwitchWorker(LM_STUDIO_BASE, model_id)
        self.model_switch_worker.done.connect(self._on_model_switch_done)
        self.model_switch_worker.finished.connect(self._on_model_switch_finished)
        self.model_switch_worker.start()
        self._set_busy(True, f"Loading '{model_id}'…")
        self.stage.set_stage("Switching model")

    def _on_model_switch_done(self, ok, msg):
        if ok:
            # Commit the new model to the single source of truth only on success, so
            # a failed switch doesn't make a later reselect look like a no-op.
            self._active_model_id = self.ctx.model_name
            self._add_system(f"✓ Model loaded: {self.ctx.model_name}")
            self._set_status("Model switched.")
            # The queue puts itself on hold when there is no model (otherwise it
            # would refuse every pending item in turn). A model arriving is the
            # event that clears that condition, so lift the hold here rather
            # than making the user find the ▶ button for a pause they never
            # asked for.
            if getattr(self, "_queue_paused", False) and any(
                    it.get("status") == "pending" for it in getattr(self, "_task_queue", [])):
                self._queue_paused = False
                self._refresh_queue_ui()
                self._add_system('▶ Queue resumed — the model is loaded.')
                self._maybe_drain_queue()
        else:
            # Roll back the speculative ctx.model_name update from _open_settings so
            # the next LLM call uses the model that is ACTUALLY loaded in LM Studio.
            if self.ctx is not None:
                self.ctx.model_name = self._active_model_id
            self._add_system(
                f"⚠ Could not switch model: {msg}\n"
                "LM Studio will JIT-load it on the next message, but the previous "
                "model may stay in memory. Check that the lms CLI is installed."
            )
            self._set_status("Model switch failed — see chat.")
        self.model_config_tab.refresh()
        self.system_info_tab.refresh()

    def _on_model_switch_finished(self):
        self.model_switch_worker = None
        self._set_busy(False)
        self.stage.set_stage("Ready")

    # ---- redraw / enhance ----
    # Memory profiles (list/switch/create/save/compact) live in
    # gui_memory_profile.MemoryProfileMixin.

    # ---- drag-drop ----
    def closeEvent(self, event):
        # Logged FIRST, before any teardown that could itself fail: this is the
        # only marker distinguishing "the window was closed" from "the process
        # died", and the teardown below joins worker threads, any of which can
        # hang or raise and swallow the evidence.
        logger.info("LIFECYCLE: main window closeEvent — tearing down")
        # Remember the current layout so the next launch reopens exactly like this.
        self._persist_layout()
        self.cam_timer.stop()
        self.vad_timer.stop()
        self._stop_vad()
        if self._lib_build_worker is not None:
            self._lib_build_worker.cancel()
            self._lib_build_worker.wait(3000)
        if self._scan_worker is not None:
            self._scan_worker.cancel()
            self._scan_worker.wait(3000)
        if self.cap is not None:
            self.cap.release()
        if self.recorder is not None:
            self.recorder.stop()
        # The Database tab's reader connection was never closed. In the app that
        # is one leaked SQLite handle for the life of the process; across
        # processes it is worse -- the file stays open, so a second instance (or
        # a build started elsewhere) blocks on busy_timeout, thirty seconds per
        # statement, with the window frozen and nothing on screen to explain it.
        _lib = getattr(self, "_library", None)
        if _lib is not None:
            try:
                _lib.close()
            except Exception:
                logger.warning("closeEvent: library close failed", exc_info=True)
            self._library = None
        self.madhouse_tab.shutdown()
        for _t in ("admin_tab", "restyle_tab"):
            if hasattr(self, _t):
                getattr(self, _t).shutdown()
        if hasattr(self, "telegram_tab"):
            self.telegram_tab.shutdown()
        for _name in ("storyboard_tab", "transfer_tab"):
            _tab = getattr(self, _name, None)
            if _tab is not None and hasattr(_tab, "shutdown"):
                try:
                    _tab.shutdown()
                except Exception:
                    logger.exception("%s shutdown failed", _name)
        self.player.stop()
        # Signal any long crawl/poll loop to bail, then give background workers a
        # few seconds to finish cleanly (research can be mid-fetch).
        if self.ctx is not None:
            self.ctx.cancel_event.set()
        # Wait for EVERY live thread, not a hand-written subset. A worker that
        # outlives its widgets delivers its queued done/failed signal into
        # already-destroyed C++ objects; that only degrades to a logged
        # RuntimeError because crash_diag suppresses the PyQt abort.
        self._join_live_threads(5000)
        if self.ctx is not None:
            self.ctx.save_memory(self.ctx.active_memory_dir)
        logging.getLogger("assistant").removeHandler(self.log_handler)
        event.accept()


def run_gui() -> None:
    # Make Windows show our taskbar icon (and group) instead of the python icon.
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("AssistantDC.Voice.1")
        except Exception:
            pass

    # High-DPI MUST be enabled before the QApplication is constructed so Qt honours
    # the Windows display-scaling setting (100..300%). PassThrough keeps fractional
    # ratios (e.g. 150% -> 1.5) instead of rounding them to integers.
    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    except Exception:
        pass
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    # Required for QtWebEngine (the research report viewer) to share the GL context.
    if _QWebEngineView is not None:
        QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)

    # Route Qt's own warnings/fatals into the log + crash.log now that Qt is import-ready.
    try:
        import crash_diag
        crash_diag.install()  # idempotent — ensures the excepthook is active even if run_gui is entered directly
        crash_diag.install_qt_message_handler()
        crash_diag.log_stage("run_gui: QApplication about to be constructed")
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(ICON_PATH))
    # Darken the native title bar of every dialog/message box the app shows.
    app._dark_titlebar_filter = _DarkTitleBarFilter()
    app.installEventFilter(app._dark_titlebar_filter)

    # Resolve the effective UI scale: load the saved preference, then derive the
    # Auto factor from the screen the app actually opened on.
    ui_scale.load()
    gui_i18n.install(ui_scale.language())   # before the first widget: labels translate as they are made
    ui_scale.apply_screen(app.primaryScreen())
    logger.info("UI scale: mode=%s tv=%s auto=%.2f effective=%.2f",
                ui_scale.scale_mode(), ui_scale.tv_mode(),
                ui_scale.auto_value(), ui_scale.effective())
    apply_ui_scale(app)

    # config.STARTUP_MODEL, not MODEL_NAME: this is the model an UNATTENDED
    # launch lands on, and both ways of getting there -- the countdown inside
    # the dialog and ASSISTANT_AUTOSTART skipping it -- must agree, or the same
    # unattended restart would run a different model depending on which of the
    # two happened to fire.
    dlg = SettingsDialog(getattr(config, "STARTUP_MODEL", MODEL_NAME) or MODEL_NAME)
    dlg.setWindowIcon(QIcon(ICON_PATH))
    # ASSISTANT_AUTOSTART=1 skips the modal picker and starts on the saved
    # default model. The picker is a MODAL dialog, so with nobody at the
    # keyboard the whole launch stops here forever: models never load, and the
    # Telegram bot (started at the end of _on_runtime_ready) never comes up --
    # so an unattended restart, a watchdog relaunch, or anything driven from a
    # shell leaves a bot that is simply offline with no error anywhere to
    # explain it. Opt-in only: launched normally the dialog still appears, so
    # switching models by hand is unchanged.
    if os.getenv("ASSISTANT_AUTOSTART", "") == "1":
        # logger, not crash_diag: that import lives in a try/except above, so
        # naming it here would turn a failed import into a NameError that kills
        # the launch this flag exists to make survivable.
        logger.info("LIFECYCLE: ASSISTANT_AUTOSTART=1 — skipping the settings "
                    "dialog, starting on the default model")
    elif dlg.exec_() != QDialog.Accepted:
        # Say so on the way out. Dismissing this dialog exits the process
        # before the window is ever built and before the bot starts, and it
        # used to do that in total silence: the log simply stopped after "UI
        # scale", which is indistinguishable from a hang or a crash. A launch
        # that quietly does nothing is the hardest kind of failure to explain
        # half an hour later.
        logger.info("LIFECYCLE: settings dialog dismissed — exiting before the "
                    "window was built (the Telegram bot never starts)")
        return
    # Apply the chosen UI scale before the main window is built so every widget
    # (including the ones sized in constructors) picks up the final scale.
    dlg.apply_ui_scale_setting()
    model_name, no_think, effort = dlg.result_choice()
    if model_name == gui_settings_dialog.NO_MODEL:
        # Picked "do not load a model": start the app with the card left alone
        # and nothing resident. The window still comes up -- the point is to be
        # able to watch a training run and use the file-backed tabs, not to be
        # locked out -- but no model is loaded until one is chosen in Settings.
        import lora_training as _LT
        _LT.free_gpu(log=logger.info)
        logger.info("LIFECYCLE: starting with NO model loaded by user choice")
        model_name = ""

    win = AssistantWindow(model_name, no_think, effort)
    win.show()
    rc = app.exec_()
    # The event loop returning is the app closing, and it was silent: the only
    # trace of a shutdown was crash_diag's "process exit (atexit)", which is
    # printed for EVERY exit and says nothing about which one this was. With
    # this line a clean close is distinguishable from a crash in the log alone.
    logger.info("LIFECYCLE: Qt event loop returned rc=%s — the window was "
                "closed, shutting down", rc)

    # Tear down in a defined order. `win`, `dlg` and `app` are all locals of this
    # frame, so `sys.exit(app.exec_())` left every one of them to be released by
    # the frame teardown — in an order CPython does not define. Destroying a
    # QWidget after its QApplication has gone is a hard crash inside Qt, not a
    # Python exception: the process dies with no traceback *after* it has already
    # printed everything and apparently exited cleanly. It reproduced as a ~1-in-6
    # segfault at interpreter shutdown in the GUI suites, which is the same code
    # path a user takes every time they close the app.
    try:
        win.close()
        win.deleteLater()
        dlg.deleteLater()
        app.processEvents()      # run the deferred deletes while `app` is alive
    except Exception:
        logger.exception("teardown after exec_ failed")
    del win, dlg
    sys.exit(rc)

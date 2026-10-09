"""The dashboard layout: three full-window pages and every tab in them.

Commands · Conversation · Workspace used to be three columns squeezed side by
side; each is now a page of its own (user, 2026-09-30), with the task queue
beside the chat instead of under it. A mixin holding one long method, and that
is the honest shape of it — the dashboard is built in a single pass. Cutting it into
_build_left/_build_centre/_build_right would move code without making anything
independent, so it moved out whole instead.

Extracted because it is CONSTRUCTION, not behaviour: it wires widgets together
and returns a root. Every tab class is imported from its own module rather than
re-exported through gui, and no suite patches any of them, so there is no seam
here to keep alive.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLayout, QLineEdit, QPushButton, QScrollArea,
    QSplitter, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

from gui_audio_visualizer import AudioVisualizer
from gui_chrome import StageIndicator
from gui_common import BORDER, _card, _scroll_page, _section, _shadow
from gui_images_panel import ImagesPanel
from gui_madhouse_tab import MadhouseTab
from gui_memory_tab import MemoryCenterTab
from gui_model_config_tab import ModelConfigTab
from gui_music_tab import MusicTab
from gui_nav import NavTabs, WorkspaceNav
from gui_weather_tab import WeatherTab
from gui_storyboard_tab import StoryboardTab
from gui_stress_tab import StressTab
from gui_system_info_tab import SystemInfoTab
from gui_telegram_tab import TelegramTab
from gui_code_tab import CodeTab
from gui_characters_tab import CharactersTab
from gui_transfer_tab import TransferTab
from ui_scale import px


class DashboardMixin:
    """Builds the main dashboard page."""

    def _build_dashboard(self):
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(12)

        # header
        header = QHBoxLayout()
        title = QLabel("Assistant DC"); title.setObjectName("title")
        self.model_chip = QLabel("loading…"); self.model_chip.setObjectName("chip")
        self.stage = StageIndicator()
        self.stage_changed.connect(self.stage.set_stage)
        self.layout_btn = QPushButton("⛶  Layout"); self.layout_btn.setObjectName("ghost")
        self.layout_btn.setToolTip("Show/hide panels, save and load layout presets, or reset "
                                   "to the default layout. Tabs can also be dragged to reorder "
                                   "and closed with their ✕; drag the splitter bars to resize.")
        self.layout_btn.clicked.connect(self._open_layout_menu)
        gear = QPushButton("⚙  Settings"); gear.setObjectName("ghost"); gear.clicked.connect(self._open_settings)
        header.addWidget(title); header.addSpacing(12); header.addWidget(self.model_chip)
        header.addStretch(1); header.addWidget(self.stage); header.addStretch(1)
        header.addWidget(self.layout_btn); header.addWidget(gear)
        outer.addLayout(header)

        self.pages = QTabWidget(); self.pages.setObjectName("pages")
        outer.addWidget(self.pages, 1)

        # ---- page 1: commands ----
        left, llay = _card()
        self.talk_btn = QPushButton("🎤  Talk")
        self.talk_btn.setObjectName("ghost")
        self.talk_btn.setToolTip("Start/stop voice recording (microphone)")
        self.talk_btn.clicked.connect(self._toggle_recording)
        self.vad_btn = QPushButton("🎙  VAD: OFF")
        self.vad_btn.setObjectName("ghost")
        self.vad_btn.setToolTip("Hands-free voice mode: the assistant listens continuously and "
                                "detects by itself when you start and stop speaking — no buttons. "
                                "It goes deaf while thinking or talking, so it never hears itself.")
        self.vad_btn.clicked.connect(self._toggle_vad)
        self.cam_btn = QPushButton("📷  Camera: off")
        self.cam_btn.setObjectName("ghost")
        self.cam_btn.setToolTip("Toggle camera preview on/off")
        self.cam_btn.clicked.connect(self._toggle_camera)
        self.capture_btn = QPushButton("📸  Capture frame")
        self.capture_btn.setObjectName("ghost")
        self.capture_btn.setToolTip("Attach the current camera frame to your next message")
        self.capture_btn.clicked.connect(self._capture_frame)
        self.capture_btn.setEnabled(False)
        self.voice_btn = QPushButton("🔊  Voice: ON")
        self.voice_btn.setObjectName("ghost")
        self.voice_btn.setToolTip("Toggle voice output (TTS) on/off")
        self.voice_btn.clicked.connect(self._toggle_voice)
        self.redraw_btn = QPushButton("🎨  Redraw last")
        self.redraw_btn.setObjectName("ghost")
        self.redraw_btn.setToolTip("Redraw or enhance the last image (FireRed; our own drawings are re-rendered from their layout)")
        self.redraw_btn.clicked.connect(self._redraw_last_image)
        self.style_preset_btn = QPushButton("🎭  Style preset")
        self.style_preset_btn.setObjectName("ghost")
        self.style_preset_btn.setToolTip(
            "Convert the last image to a named art style (anime, watercolor, …) "
            "or type your own — no second reference photo needed.")
        self.style_preset_btn.clicked.connect(self._style_preset_last_image)
        self.remove_obj_btn = QPushButton('🧽  Remove object')
        self.remove_obj_btn.setObjectName("ghost")
        self.remove_obj_btn.setToolTip('Remove an object, a person or lettering from the last picture (FireRed; the rest stays as it was).')
        self.remove_obj_btn.clicked.connect(self._remove_object_last_image)
        self.outfit_btn = QPushButton('👗  Outfit')
        self.outfit_btn.setObjectName("ghost")
        self.outfit_btn.setToolTip('Change the outfit of the person in the last picture; the face and pose stay.')
        self.outfit_btn.clicked.connect(self._change_outfit_last_image)
        self.fixhands_btn = QPushButton("✋  Fix hands")
        self.fixhands_btn.setObjectName("ghost")
        self.fixhands_btn.setToolTip("Repair deformed hands/fingers in the last image.\n"
                                     "You'll be asked to draw over the bad hand (recommended — "
                                     "works even when auto-detection fails); leave it blank to "
                                     "auto-detect and fix all hands.")
        self.fixhands_btn.clicked.connect(self._fix_hands_last_image)
        self.retryhands_btn = QPushButton("🔁  Retry hand fix")
        self.retryhands_btn.setObjectName("ghost")
        self.retryhands_btn.setToolTip("Re-run the last hand fix with a NEW random seed, "
                                       "reusing the same source image and drawn region "
                                       "(the result is stochastic — retry until the fingers "
                                       "come out clean).")
        self.retryhands_btn.setEnabled(False)
        self.retryhands_btn.clicked.connect(self._retry_hands)
        self.fixartifact_btn = QPushButton("🩹  Fix artifact")
        self.fixartifact_btn.setObjectName("ghost")
        self.fixartifact_btn.setToolTip("Select-and-fix ANY awkward spot in the last image.\n"
                                        "Paint over the problem area, describe what's wrong "
                                        "(e.g. 'smooth this harsh seam', 'remove this smear', "
                                        "'fix this melted edge'), and only that region is "
                                        "regenerated — everything else stays pixel-identical.")
        self.fixartifact_btn.clicked.connect(self._fix_artifact_last_image)
        # All instruction edits (hand fix, artifact fix, region inpaint/redraw,
        # whole-frame re-render) run on FireRed, the only edit engine.
        self.whole_frame_btn = QPushButton("🖼  Whole frame: OFF")
        self.whole_frame_btn.setObjectName("ghost")
        self.whole_frame_btn.setToolTip(
            "Maskless mode: ON re-renders the ENTIRE image through the instruction "
            "edit — no segmentation, no mask, no composite, no identity guards: the "
            "model has complete freedom over the whole frame, face included. Use when "
            "masking keeps failing on a stubborn edit. OFF = normal masked/contained "
            "pipeline.")
        self.whole_frame_btn.clicked.connect(self._toggle_whole_frame)
        self.paste_img_btn = QPushButton("📋  Paste image")
        self.paste_img_btn.setObjectName("ghost")
        self.paste_img_btn.setToolTip("Load an image from the clipboard as the working image "
                                      "(or drag-drop an image file). Ctrl+V also works.")
        self.paste_img_btn.clicked.connect(self._paste_image_from_clipboard)
        self.ultra_btn = QPushButton("🔬  Ultra Search: OFF")
        self.ultra_btn.setObjectName("ghost")
        self.ultra_btn.setToolTip("Aggressive multi-source research and crawling mode.\n"
                                  "When ON, your next message is treated as a research topic: "
                                  "the assistant searches widely, reads many pages, and writes a "
                                  "full report in the Research tab (takes minutes).")
        self.ultra_btn.clicked.connect(self._toggle_ultra)
        self.usedb_btn = QPushButton("📚  Use Database: OFF")
        self.usedb_btn.setObjectName("ghost")
        self.usedb_btn.setToolTip("Chat with your uploaded documents.\n"
                                  "When ON, each question first searches your local document "
                                  "database (hybrid keyword + semantic) and feeds the most "
                                  "relevant passages to the model — so you can ask about a "
                                  "1,200-page book without it filling the context window.\n"
                                  "Upload files and press Build in the Database tab first.")
        self.usedb_btn.clicked.connect(self._toggle_usedb)
        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setObjectName("rec")
        self.stop_btn.setToolTip("Cancel the operation currently running")
        self.stop_btn.clicked.connect(self._cancel_current)
        self.stop_btn.setEnabled(False)
        clear_btn = QPushButton("🗑  Clear chat")
        clear_btn.setObjectName("ghost")
        clear_btn.setToolTip("Clear the conversation display (does not affect memory)")
        clear_btn.clicked.connect(self._clear_chat)
        self.save_mem_btn = QPushButton("💾  Save memory")
        self.save_mem_btn.setObjectName("ghost")
        self.save_mem_btn.setToolTip("Persist the current session memory to disk now")
        self.save_mem_btn.clicked.connect(self._save_memory)
        self.compact_btn = QPushButton("⚡  Compact memory")
        self.compact_btn.setObjectName("ghost")
        self.compact_btn.setToolTip("Summarise all memory items into a short paragraph via the LLM")
        self.compact_btn.clicked.connect(self._compact_memory)
        self.clear_ctx_btn = QPushButton("🧹  Clear context")
        self.clear_ctx_btn.setObjectName("ghost")
        self.clear_ctx_btn.setToolTip("Wipe the whole session context for this profile — "
                                      "memory items AND the saved summary — so nothing old is "
                                      "injected into the next reply. Does not delete other profiles.")
        self.clear_ctx_btn.clicked.connect(self._clear_context)
        self.pause_btn = QPushButton("⏸  Pause speech")
        self.pause_btn.setObjectName("ghost")
        self.pause_btn.clicked.connect(self._toggle_pause)
        self.viz = AudioVisualizer()
        mem_row = QHBoxLayout()
        mem_row.setSpacing(4)
        self.mem_combo = QComboBox()
        self.mem_combo.addItem("default")
        self.mem_combo.setToolTip("Active memory profile — select to switch (auto-saves current first)")
        self.mem_combo.currentTextChanged.connect(self._switch_memory_profile)
        mem_row.addWidget(self.mem_combo, 1)
        new_prof_btn = QPushButton("+")
        new_prof_btn.setObjectName("ghost")
        new_prof_btn.setFixedSize(px(28), px(28))
        new_prof_btn.setStyleSheet("padding: 0;")   # the ghost padding left no room: an empty square
        new_prof_btn.setToolTip("Create a new memory profile")
        new_prof_btn.clicked.connect(self._new_memory_profile)
        mem_row.addWidget(new_prof_btn)
        # The whole window is this page's, so the buttons stand in three columns
        # by what they act on instead of one long scrolling strip.
        groups = (
            ("Voice & camera", (self.talk_btn, self.vad_btn, self.viz, self.pause_btn,
                                self.voice_btn, self.cam_btn, self.capture_btn)),
            ("Images", (self.paste_img_btn, self.redraw_btn, self.style_preset_btn,
                        self.remove_obj_btn, self.outfit_btn, self.fixhands_btn,
                        self.retryhands_btn, self.fixartifact_btn, self.whole_frame_btn)),
            ("Search & memory", (self.ultra_btn, self.usedb_btn, clear_btn, mem_row,
                                 self.save_mem_btn, self.compact_btn, self.clear_ctx_btn)),
        )
        grid = QGridLayout(); grid.setHorizontalSpacing(px(18))
        for c, (title, items) in enumerate(groups):
            col = QVBoxLayout(); col.setSpacing(px(10))
            col.addWidget(_section(title))
            for it in items:
                if isinstance(it, QLayout):
                    col.addLayout(it)
                    continue
                col.addWidget(it)
                # A command answers in the chat: show it. Plain switches stay put.
                if isinstance(it, QPushButton) and it not in (self.voice_btn, self.whole_frame_btn):
                    it.clicked.connect(lambda *_: self.pages.setCurrentIndex(1))
            col.addStretch(1)
            grid.addLayout(col, 0, c)
            grid.setColumnStretch(c, 1)
        llay.addLayout(grid)
        llay.addStretch(1)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setStyleSheet("QScrollArea{border:none;background:transparent;padding:0;}")
        left_scroll.setWidget(left)
        self.pages.addTab(left_scroll, "Commands")

        # ---- page 2: camera + chat + input, the queue beside them ----
        center, clay = _card()
        self.camera_label = QLabel(); self.camera_label.setFixedHeight(px(260))
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet(f"background:#0d0e12; border:1px solid {BORDER}; border-radius:10px;")
        self.camera_label.hide()
        clay.addWidget(self.camera_label)
        self.chat = QTextEdit(); self.chat.setReadOnly(True)
        self.chat.setPlaceholderText("Write below, or press 🎤 Talk on the Commands page.")
        self.chat.viewport().installEventFilter(self)  # drops, image clicks, resize for scroll-btn
        # Deliver buttonless MouseMove events so the event filter can swap the text
        # I-beam for a pointing hand while hovering a chat image (visual cue that
        # the image is double-clickable).
        self.chat.viewport().setMouseTracking(True)
        # Floating scroll-to-bottom button — child of the viewport so it renders
        # on top of the text content (QTextEdit clips direct children behind viewport)
        self._scroll_down_btn = QPushButton("▼", self.chat.viewport())
        self._scroll_down_btn.setToolTip("Scroll to bottom")
        self._scroll_down_btn.setFixedSize(px(32), px(32))
        self._scroll_down_btn.hide()
        self._scroll_down_btn.raise_()
        self._scroll_down_btn.clicked.connect(
            lambda: self.chat.verticalScrollBar().setValue(
                self.chat.verticalScrollBar().maximum()))
        self.chat.verticalScrollBar().valueChanged.connect(self._on_chat_scroll)
        row = QHBoxLayout()
        self.input = QLineEdit(); self.input.setPlaceholderText("Type a message and press Enter…")
        self.input.returnPressed.connect(self._send_text)
        self.input.installEventFilter(self)  # intercept Ctrl+V to paste images, not just text
        # Quick reply-length selector, on the main screen so the user never has to open
        # Settings to change it. Writes ctx.response_length directly; mirrors the full
        # control in the Settings dialog. Order = increasing length.
        self._LEN_VALUES = ["ultra", "short", "auto", "long"]
        self.length_combo = QComboBox()
        self.length_combo.addItems(["1 sentence", "Short", "Auto", "Detailed"])
        self.length_combo.setToolTip(
            "Reply length: 1 sentence · Short (1–3) · Auto (model decides) · Detailed. "
            "Only affects the assistant's spoken reply — not web research or image prompts.")
        _cur = (getattr(self.ctx, "response_length", "auto") if self.ctx else "auto") or "auto"
        self.length_combo.setCurrentIndex(
            self._LEN_VALUES.index(_cur) if _cur in self._LEN_VALUES else 2)
        self.length_combo.currentIndexChanged.connect(self._on_length_changed)
        # Send runs the message now when idle, and automatically stages it in the task
        # queue when the assistant is busy (so a follow-up like "upscale the result" typed
        # mid-render is queued, never dropped). There is no separate "add to queue" button:
        # it would be identical to Send (an idle enqueue auto-runs; a busy Send auto-queues).
        self.send_btn = QPushButton("Send"); self.send_btn.clicked.connect(self._send_text)
        self.send_btn.setToolTip(
            "Send now when idle. While the assistant is busy, the message is added to the "
            "task queue instead of being dropped — queued tasks run one after another as "
            "the assistant frees up. Use the queue panel above to reorder, remove, or pause.")
        row.addWidget(self.input, 1); row.addWidget(self.length_combo)
        row.addWidget(self.stop_btn); row.addWidget(self.send_btn)
        chat_col = QWidget()
        ccl = QVBoxLayout(chat_col); ccl.setContentsMargins(0, 0, 0, 0); ccl.setSpacing(px(8))
        ccl.addWidget(self.chat, 1)
        ccl.addLayout(row)
        clay.addWidget(chat_col, 1)
        self.splitter = QSplitter(Qt.Horizontal)       # conversation | task queue
        self.splitter.addWidget(center)
        self.splitter.addWidget(self._build_queue_bar())
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setCollapsible(0, False)
        _shadow(center)
        self.pages.addTab(self.splitter, "Conversation")

        # ---- page 3: the workspace tabs ----
        right, rlay = _card(QHBoxLayout)
        self.tabs = tabs = NavTabs()
        self.images_panel = ImagesPanel()
        self.search_view = QTextEdit(); self.search_view.setReadOnly(True)
        self.search_view.setPlaceholderText("Search results will appear here.")
        self.research_container = self._build_research_tab()
        self.terminal = QTextEdit(); self.terminal.setReadOnly(True); self.terminal.setObjectName("terminal")
        self.model_config_tab = ModelConfigTab()
        self.system_info_tab = SystemInfoTab()
        self.memory_center_tab = MemoryCenterTab(self)
        self.transfer_tab = TransferTab(self)
        self.stress_tab = StressTab(self)
        self.music_tab = MusicTab(self)
        self.weather_tab = WeatherTab(self)
        from gui_voice_clone_tab import VoiceCloneTab
        self.voice_clone_tab = VoiceCloneTab(self)
        self.madhouse_tab = MadhouseTab(self)
        self.storyboard_tab = StoryboardTab(self)
        self.database_tab = self._build_database_tab()
        self.telegram_tab = TelegramTab(self)
        from gui_admin_tab import AdminTab
        from gui_restyle_tab import RestyleTab
        self.restyle_tab = RestyleTab(self)
        self.admin_tab = AdminTab(self)
        self.code_tab = CodeTab(self)
        self.characters_tab = CharactersTab(self)
        # Single source of truth for every panel: (stable_key, widget, label). Drives the
        # initial tabs, the show/hide menu, preset save/restore and reset-to-default. A
        # stable key (NOT the visible index/label) is what gets persisted, so reordering,
        # closing, or relabelling tabs can never corrupt a saved layout.
        # Every page goes through _scroll_page (see its docstring): the registry holds
        # the SCROLLABLE page, so tab lookups, presets and re-adds all agree, while the
        # attributes above still point at the real widgets for the logic.
        self._tab_registry = [
            ("images",   _scroll_page(self.images_panel),      "Images"),
            ("storyboard", _scroll_page(self.storyboard_tab),  "Storyboard"),
            ("transfer", _scroll_page(self.transfer_tab),      "Transfer"),
            ("stress",   _scroll_page(self.stress_tab),        "Stress"),
            ("music",    _scroll_page(self.music_tab),         "Music"),
            ("weather",  _scroll_page(self.weather_tab),       "Weather"),
            ("voice_clone", _scroll_page(self.voice_clone_tab), 'Voice Clone'),
            ("restyle",  _scroll_page(self.restyle_tab),       'Restyle Video'),
            ("madhouse", _scroll_page(self.madhouse_tab),      "Madhouse"),
            ("search",   _scroll_page(self.search_view),       "Search"),
            ("research", _scroll_page(self.research_container), "Research"),
            ("database", _scroll_page(self.database_tab),      "Database"),
            ("memory",   _scroll_page(self.memory_center_tab), "Memory"),
            ("model",    _scroll_page(self.model_config_tab),  "Model"),
            ("status",   _scroll_page(self.system_info_tab),   "Status"),
            ("log",      _scroll_page(self.terminal),          "Log"),
            ("telegram", _scroll_page(self.telegram_tab),      "Telegram"),
            ("admin",    _scroll_page(self.admin_tab),         'Admin'),
            ("code",     _scroll_page(self.code_tab),          "Code"),
            ("characters", _scroll_page(self.characters_tab),  'Characters'),
        ]
        for _k, _w, _label in self._tab_registry:
            tabs.addTab(_w, _label)
        tabs.setMovable(True)          # drag tabs to reorder
        tabs.setTabsClosable(True)     # each tab gets a ✕ to close it
        tabs.tabCloseRequested.connect(self._on_tab_close_requested)
        tabs.tabBar().hide()           # the grouped list on the left replaces the strip (gui_nav)
        self.workspace_nav = WorkspaceNav(tabs, self._tab_key_for_widget)
        rlay.addWidget(self.workspace_nav)
        rlay.addWidget(tabs, 1)
        _shadow(right)
        self.pages.addTab(right, "Workspace")
        self.pages.setCurrentIndex(1)
        self.splitter.setSizes(self._default_splitter_sizes())
        return root

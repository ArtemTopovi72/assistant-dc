"""The Madhouse tab's widget tree — 174 lines of pure construction.

Split out of gui_madhouse_tab.py, where it was the second half of __init__ and
made that one method 200 lines long. Nothing here decides anything: it builds
the three splitter panes (cast + transport / room / your turn), wires each
control to a method on the tab, and returns. The behaviour lives next door.

MadhouseGrid is resolved through gui_madhouse_tab at CALL time rather than
imported by value, so the re-export documented there stays the single seam a
suite would patch to stub the room renderer. Same reason gui_image_fix reaches
MaskDrawDialog through `gui`: a by-value binding across a split silently
detaches every patch aimed at the original name.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel,
                             QLineEdit, QListWidget, QPushButton, QSplitter,
                             QVBoxLayout, QWidget)

from ui_scale import px
from gui_common import MUTED, _FlowWidget, _flow, _section


class MadhouseUIMixin:
    """Builds MadhouseTab's widgets. Mixed into MadhouseTab, never used alone."""

    def _build_ui(self):
        import gui_madhouse_tab as _mh          # MadhouseGrid, at call time
        MadhouseGrid = _mh.MadhouseGrid
        # Layout: three panes in a vertical splitter — cast+transport / transcript /
        # your input — so the transcript can be dragged as large as you want. It used
        # to be a plain stack, where the fixed rows above and below squeezed the
        # transcript down to nothing in a narrow side panel.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(px(6), px(6), px(6), px(6))
        outer.setSpacing(px(6))
        self.vsplit = QSplitter(Qt.Vertical)
        self.vsplit.setChildrenCollapsible(True)   # the cast pane may be folded away
        outer.addWidget(self.vsplit, 1)

        _top = QWidget(); root = QVBoxLayout(_top)
        root.setContentsMargins(0, 0, 0, px(2)); root.setSpacing(px(8))

        # ---- cast ----------------------------------------------------------
        # Built from the SAME assets the Settings dialog offers: a reference WAV for
        # the voice (ctx.custom_ref_wav) and a personalities/*.txt file for the
        # character (ctx.custom_personality_text). Both combos also carry a
        # "Browse…" entry, exactly like the Settings "Choose…" buttons.
        root.addWidget(_section("Cast"))
        # FlowLayout throughout the control rows: this tab lives in a ~360px side
        # panel, and fixed QHBoxLayouts forced a 523px minimum width, clipping the
        # whole tab. Flow rows wrap instead, so nothing is ever cut off.
        addrow = _flow(spacing=px(6))
        self.name_in = QLineEdit(); self.name_in.setPlaceholderText("Name")
        self.name_in.setMaximumWidth(px(120))
        self.name_in.returnPressed.connect(self._add_character)
        self.voice_combo = QComboBox(); self.voice_combo.setMinimumWidth(px(104))
        self.voice_combo.setToolTip("Reference voice for F5-TTS cloning — same files the "
                                    "Settings ▸ Voice picker uses.")
        self.pers_combo = QComboBox(); self.pers_combo.setMinimumWidth(px(104))
        self.pers_combo.setToolTip("Personality file from personalities/ — same files the "
                                   "Settings ▸ Personality picker uses.")
        add_btn = QPushButton("➕  Add")
        add_btn.clicked.connect(self._add_character)
        addrow.addWidget(self.name_in)
        addrow.addWidget(self.voice_combo)
        addrow.addWidget(self.pers_combo)
        addrow.addWidget(add_btn)
        root.addWidget(_FlowWidget(addrow))
        self.voice_combo.currentIndexChanged.connect(
            lambda: self._browse_into(self.voice_combo, "voice"))
        self.pers_combo.currentIndexChanged.connect(
            lambda: self._browse_into(self.pers_combo, "personality"))

        self.cast_list = QListWidget()
        self.cast_list.setWordWrap(True)
        # No height cap: the splitter decides how much room the cast pane gets, so a
        # long cast can be shown in full by dragging instead of scrolling a stub.
        self.cast_list.setMinimumHeight(px(52))
        root.addWidget(self.cast_list, 1)

        castrow = _flow(spacing=px(6))
        files_btn = QPushButton("📄  Add from files…"); files_btn.setObjectName("ghost")
        files_btn.setToolTip("Pick a personality .txt and a voice file — that's a whole character.")
        files_btn.clicked.connect(self._add_character_from_files)
        rm_btn = QPushButton("✕  Remove"); rm_btn.setObjectName("ghost")
        rm_btn.clicked.connect(self._remove_character)
        castrow.addWidget(files_btn)
        load_btn = QPushButton("📂  Load cast…"); load_btn.setObjectName("ghost")
        load_btn.setToolTip("JSON array of {id, name, voice, personality, prompt}")
        load_btn.clicked.connect(self._load_characters)
        save_btn = QPushButton("💾  Save cast…"); save_btn.setObjectName("ghost")
        save_btn.clicked.connect(self._save_cast)
        self.cast_lbl = QLabel("no characters"); self.cast_lbl.setObjectName("chip")
        castrow.addWidget(rm_btn); castrow.addWidget(load_btn); castrow.addWidget(save_btn)
        castrow.addWidget(self.cast_lbl)
        root.addWidget(_FlowWidget(castrow))
        self._refresh_asset_combos()

        # ---- transport -----------------------------------------------------
        root.addWidget(_section("Auto-dialogue"))
        trow = _flow(spacing=px(6))
        self.start_btn = QPushButton("▶  Start")
        self.start_btn.clicked.connect(self._start_auto)
        self.pause_btn = QPushButton("⏸  Pause"); self.pause_btn.setObjectName("ghost")
        self.pause_btn.clicked.connect(self._toggle_pause)
        self.stop_btn = QPushButton("⏹  Stop"); self.stop_btn.setObjectName("ghost")
        self.stop_btn.clicked.connect(self._stop_auto)
        self.hush_btn = QPushButton("🔇  Stop speech"); self.hush_btn.setObjectName("ghost")
        self.hush_btn.setToolTip("Cut the current line short (auto-dialogue keeps running)")
        self.hush_btn.clicked.connect(self._stop_speech)
        for b in (self.start_btn, self.pause_btn, self.stop_btn, self.hush_btn):
            trow.addWidget(b)
        # Voice mode. "Own voices" clones each character's reference WAV; "Default
        # voice" speaks everyone with the assistant's normal voice (cheaper — no
        # per-character reference preprocessing); "Silent" is text only.
        # label and combo as one flow item: the flow row tops its items, a bare
        # label sat off its combo's middle
        voices_box = QWidget()
        voices_row = QHBoxLayout(voices_box)
        voices_row.setContentsMargins(0, 0, 0, 0)
        voices_row.addWidget(QLabel("Voices"))
        self.voice_mode = QComboBox()
        self.voice_mode.addItem("🔊  Own voices", "own")
        self.voice_mode.addItem("🗣  Default voice", "default")
        self.voice_mode.addItem("🔇  Silent (text only)", "off")
        self.voice_mode.setToolTip(
            "Own voices — every character speaks with its own reference clip.\n"
            "Default voice — everyone speaks with the assistant's voice.\n"
            "Silent — nothing is spoken; the room runs as text only.")
        self.voice_mode.currentIndexChanged.connect(self._on_voice_mode)
        voices_row.addWidget(self.voice_mode)
        trow.addWidget(voices_box)
        root.addWidget(_FlowWidget(trow))

        self.vsplit.addWidget(_top)

        # ---- minimap (the pane that gets the slack) -------------------------
        _mid = QWidget(); mid = QVBoxLayout(_mid)
        mid.setContentsMargins(0, 0, 0, 0); mid.setSpacing(px(2))
        mid.addWidget(_section("Room"))
        self.grid = MadhouseGrid()
        mid.addWidget(self.grid, 1)
        self.vsplit.addWidget(_mid)

        _bot = QWidget(); root = QVBoxLayout(_bot)
        root.setContentsMargins(0, px(2), 0, 0); root.setSpacing(px(8))

        # ---- manual turn ---------------------------------------------------
        # You are a participant too: pick yourself (or any character) and join by
        # typing or by talking — the mic goes through the same Whisper transcription
        # the main chat uses.
        root.addWidget(_section("You in the room"))
        namerow = _flow(spacing=px(6))
        namerow.addWidget(QLabel("My name"))
        self.me_in = QLineEdit(self.user["name"])
        self.me_in.setPlaceholderText("How the characters should address you")
        self.me_in.setToolTip("The cast is told this is the real person in the room and "
                              "is asked to address you by this name.")
        self.me_in.setMaximumWidth(px(160))
        self.me_in.textChanged.connect(self._on_my_name_changed)
        namerow.addWidget(self.me_in)
        self.router_box = QCheckBox("🧠  Smart turn-taking")
        self.router_box.setChecked(True)
        self.router_box.setToolTip(
            "On — the model decides who answers next from the cast and the transcript, "
            "and whoever is addressed by name always answers.\n"
            "Off — a random character who didn't just speak (faster: one less model "
            "call per turn; being addressed by name still wins).")
        namerow.addWidget(self.router_box)
        root.addWidget(_FlowWidget(namerow))

        mrow = QHBoxLayout()
        self.who = QComboBox(); self.who.setMinimumWidth(px(84))
        self.say_in = QLineEdit(); self.say_in.setMinimumWidth(px(70))
        self.say_in.setPlaceholderText("Type a line and press Enter…")
        self.say_in.returnPressed.connect(self._send_manual)
        send_btn = QPushButton("Send"); send_btn.clicked.connect(self._send_manual)
        # Icon-only: the label is what pushed this row (and the whole tab) past the
        # width of the side panel. The tooltip carries the explanation.
        self.talk_btn = QPushButton("🎤")
        self.talk_btn.setToolTip("Talk: click to start recording, click again to "
                                 "transcribe and post it as whoever is selected.")
        self.talk_btn.clicked.connect(self._toggle_talk)
        mrow.addWidget(self.who); mrow.addWidget(self.say_in, 1)
        mrow.addWidget(send_btn); mrow.addWidget(self.talk_btn)
        root.addLayout(mrow)

        brow = QHBoxLayout()
        clear_btn = QPushButton("🗑  Clear history"); clear_btn.setObjectName("ghost")
        clear_btn.setToolTip("Wipe the whole transcript and stop any speech. This is also "
                             "what the characters remember — after clearing, the next "
                             "replies start from a blank room. The cast is kept.")
        clear_btn.clicked.connect(self._clear_history)
        self.status_lbl = QLabel("idle"); self.status_lbl.setStyleSheet(f"color:{MUTED};")
        brow.addWidget(clear_btn); brow.addStretch(1); brow.addWidget(self.status_lbl)
        root.addLayout(brow)
        self.vsplit.addWidget(_bot)

        # Only the transcript grows when the window does; the control panes keep
        # their natural height. setSizes seeds the split before anything is shown.
        for _i, _stretch in ((0, 0), (1, 1), (2, 0)):
            self.vsplit.setStretchFactor(_i, _stretch)
        self.vsplit.setCollapsible(1, False)       # the transcript can't be folded shut
        self.vsplit.setSizes([px(210), px(430), px(120)])

        # Last: every widget exists, so this can seed the "speak as" combo with the
        # "You" participant and sync every control at once.
        self._rebuild_cast_views()

"""Music tab: type a topic, get a generated song.

Minimal by design — modeled on StressTab/MadhouseTab (a self-contained
QWidget(host) tab) and MadhouseSpeakWorker (a single QThread that does one
slow call and reports back via signals). The whole generation path
(build_structured_caption -> generate_music) runs off the UI thread so a
slow or stuck call never freezes the window.

music.py is a real engine now (MiniMax Music 3 on the local ComfyUI), not the
stub this tab was first written against — but the try/except around the call
stays load-bearing rather than becoming defensive copy-paste: a render needs
multi-GB weights, a ComfyUI carrying the Music3 nodes, and a songwriter LLM
that returns usable lyrics, and any of the three can be missing or fumble a
turn. Every one of those has to reach the user as a status message, never a
crash and never a hang.
"""
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QLineEdit, QSpinBox,
                             QPushButton, QVBoxLayout, QWidget)

import config as _cfg
from audio import AudioPlayer
from ui_scale import px
from gui_common import MUTED, _section

import logging
logger = logging.getLogger("assistant.gui")   # same channel as gui.py

# English labels for the engine's settings vocabulary. The KEYS and the
# phrases they become live in music.py (one copy, shared with the Telegram
# picker — see the comment above music.GENRES); only these display strings
# are this surface's business.
_GENRE_LABELS = {
    "auto": "Auto", "pop": "Pop", "rock": "Rock", "edm": "EDM",
    "synth": "Synth-pop", "hiphop": "Hip-hop", "rnb": "R&B", "soul": "Soul",
    "jazz": "Jazz", "folk": "Folk", "country": "Country", "metal": "Metal",
    "cinematic": "Cinematic", "lofi": "Lo-fi",
}
_TEMPO_LABELS = {
    "auto": "Auto", "slow": "Slow (~70)", "medium": "Medium (~100)",
    "fast": "Fast (~130)", "veryfast": "Very fast (~150)",
}
# Two extra rows on the tempo and length pickers, mirroring the Telegram
# menu: a typed value within the engine's range (a spin box appears next to
# the picker) and, for the length, the bot's own choice.
_CUSTOM = "custom"
_BOT_DECIDES = "auto"
_VOCAL_LABELS = {
    "auto": "Auto", "female": "Female", "male": "Male", "duet": "Duet",
    "instrumental": "Instrumental",
}


class MusicWorker(QThread):
    """Caption the topic, then generate the song — both off the UI thread.

    One worker covers both calls (rather than splitting caption/generate into
    two workers like Madhouse's reply/speak split) because there is nothing
    useful to show the user between them yet: the stub can't stream partial
    progress, and a caption with no song to follow isn't a usable interim
    result."""
    done = pyqtSignal(str, str, str)      # wav_path, lyrics, style
    failed = pyqtSignal(str)
    chose = pyqtSignal(dict)              # what the bot picked for the Auto fields

    def __init__(self, ctx, topic: str, lang: str = "ru", *,
                 prefs: dict = None, duration_s: int = 0, preset: str = "",
                 steps: int = 0):
        super().__init__()
        self.ctx, self.topic, self.lang = ctx, topic, lang
        # Snapshotted at construction, not read off the widgets in run():
        # run() is on the worker thread, and touching Qt widgets from there is
        # exactly the unhandled-slot-exception path that aborts the process
        # with no traceback.
        self.prefs = prefs or {}
        self.duration_s = duration_s
        self.preset = preset
        self.steps = int(steps or 0)

    def run(self):
        try:
            import music
            from config import MUSIC_DEFAULT_SECONDS
            duration = self.duration_s
            # Whatever is on Auto, the bot picks itself BEFORE writing (same
            # call as the Telegram side): the user's filled fields go in as
            # fixed conditions, only the blanks come back, validated.
            _pick = getattr(music, "choose_auto_params", None)
            want = {f: f not in self.prefs for f in ("genre", "tempo", "vocal")}
            if self.prefs.get("instrumental"):
                want["vocal"] = False
            want["duration"] = not duration
            chosen = {}
            if _pick and any(want.values()):
                chosen = _pick(self.ctx, self.topic, self.lang,
                               fixed=dict(self.prefs, **({"seconds": duration} if duration else {})),
                               **want) or {}
                if chosen:
                    picked = music.prefs_from(
                        genre=chosen.get("genre", ""),
                        tempo=f"bpm:{chosen['bpm']}" if chosen.get("bpm") else "",
                        vocal=chosen.get("vocal", ""))
                    self.prefs = dict(picked, **self.prefs)     # the user's own choices win
                    if not duration and chosen.get("seconds"):
                        duration = int(chosen["seconds"])
            self.chosen = chosen
            duration = duration or MUSIC_DEFAULT_SECONDS
            # Same duration for both: Music3 stops when the words run out, so
            # the songwriter needs to know how much to write, and a render
            # asking for longer than the lyrics cover just ends early.
            caption = music.build_structured_caption(
                self.ctx, self.topic, self.lang,
                duration_s=duration, prefs=self.prefs)
            # engine_available is checked for THIS preset inside generate_music;
            # the presets do not share weight files.
            lyrics = (caption or {}).get("lyrics", "")
            style = (caption or {}).get("style", "")
            path = music.generate_music(self.ctx, lyrics, style,
                                        duration_s=duration,
                                        preset=self.preset or None,
                                        steps=self.steps or None)
            if chosen:
                self.chose.emit(chosen)
            self.done.emit(path or "", lyrics, style)
        except Exception as exc:
            # Covers music.MusicUnavailable (still a RuntimeError) as well as any
            # other stub/engine failure — one message, no traceback in the UI.
            logger.exception("Music generation failed")
            self.failed.emit(str(exc))


class MusicTab(QWidget):
    """Type a topic, pick genre/tempo/vocal/length, press Generate.

    The four pickers mirror Telegram's 🎛 Song settings and resolve through the
    same music.prefs_from(), so the two surfaces cannot disagree about what a
    genre means. Auto everywhere is the default and leaves the songwriter free
    to choose what suits the topic — a setting is only pinned when the user
    actually picks one."""

    def __init__(self, host):
        super().__init__()
        self.host = host                  # AssistantWindow (for ctx)
        self.player = AudioPlayer()
        self.music_worker = None
        self._last_wav = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(px(6), px(6), px(6), px(6))
        root.setSpacing(px(8))

        root.addWidget(_section("Music"))
        help_lbl = QLabel("Describe what the song should be about — a topic, mood, "
                          "or occasion — and Generate will write lyrics and a style, "
                          "then render the song.")
        help_lbl.setWordWrap(True)
        help_lbl.setStyleSheet(f"color:{MUTED};")
        root.addWidget(help_lbl)

        row = QHBoxLayout()
        self.topic_in = QLineEdit()
        self.topic_in.setPlaceholderText("e.g.  a birthday song for a friend who loves cats")
        self.topic_in.returnPressed.connect(self._generate)
        self.gen_btn = QPushButton("🎵  Generate")
        self.gen_btn.clicked.connect(self._generate)
        row.addWidget(self.topic_in, 1)
        row.addWidget(self.gen_btn)
        root.addLayout(row)

        # ── settings ────────────────────────────────────────────────────────
        # Built from music.py's tables rather than a literal list here, so a
        # genre added to the engine shows up without touching this file (and
        # cannot appear on only one of the two surfaces).
        import music as _music
        self._combos = {}
        set_row = QHBoxLayout()
        for field, table, labels in (
                ("genre", _music.GENRES, _GENRE_LABELS),
                ("tempo", _music.TEMPOS, _TEMPO_LABELS),
                ("vocal", _music.VOCALS, _VOCAL_LABELS)):
            box = QComboBox()
            for key in table:
                # The KEY rides along as item data — the visible label is a
                # display string and must never be what gets resolved.
                box.addItem(labels.get(key, key.title()), key)
            self._combos[field] = box
            set_row.addWidget(QLabel(field.title() + ":"))
            set_row.addWidget(box)
            if field == "tempo":
                box.addItem("Custom BPM...", _CUSTOM)
                self.bpm_spin = QSpinBox()
                self.bpm_spin.setRange(*_music.BPM_RANGE)
                self.bpm_spin.setValue(100)
                self.bpm_spin.setSuffix(" BPM")
                self.bpm_spin.setToolTip(f"Pinned exactly; {_music.BPM_RANGE[0]}-{_music.BPM_RANGE[1]}.")
                self.bpm_spin.setVisible(False)
                box.currentIndexChanged.connect(
                    lambda _i, b=box: self.bpm_spin.setVisible(b.currentData() == _CUSTOM))
                set_row.addWidget(self.bpm_spin)

        # Quality preset. Labelled with its measured cost, because the whole
        # point of the choice is the trade and a bare "Maximum" hides that it
        # is ~6x slower.
        self.qual_combo = QComboBox()
        for name in _music.WEIGHT_PRESETS:
            self.qual_combo.addItem(name.title(), name)
        _names = list(_music.WEIGHT_PRESETS)
        if _music.DEFAULT_PRESET in _names:
            self.qual_combo.setCurrentIndex(_names.index(_music.DEFAULT_PRESET))
        self.qual_combo.setToolTip(
            f"Weight preset. Times are measured on this machine for a "
            f"{_music.ETA_QUOTE_SECONDS}-second song (median of completed renders). "
            "Maximum exceeds VRAM and streams, which is why it is "
            "disproportionately slower.")
        self.qual_lbl = QLabel("Quality:")
        set_row.addWidget(self.qual_lbl)
        set_row.addWidget(self.qual_combo)
        # DiT sampler steps: the second half of the "how long" trade. 20 is
        # the default (A/B 15.09 heard no loss against 30); 50 is the finest.
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(*_music.STEPS_RANGE)
        self.steps_spin.setValue(_music.MUSIC_STEPS)
        self.steps_spin.setSuffix(" steps")
        self.steps_spin.setToolTip(
            f"Sampler steps, {_music.STEPS_RANGE[0]}-{_music.STEPS_RANGE[1]}. "
            "The DiT stage is about half of a render and grows with this number.")
        set_row.addWidget(self.steps_spin)
        # «⚡ ~6 min · 💎 ~8 min · 🐘 ~23 min» for the step count on the spin --
        # the same figures the Telegram buttons carry.
        self.eta_lbl = QLabel()
        set_row.addWidget(self.eta_lbl)
        self.qual_combo.currentIndexChanged.connect(lambda _i: self._refresh_eta())
        self.steps_spin.valueChanged.connect(lambda _v: self._refresh_eta())
        self._refresh_eta()
        # Presets and steps are Music3 knobs; YuE2 has neither.
        for w in (self.qual_lbl, self.qual_combo, self.steps_spin, self.eta_lbl):
            w.setVisible(_music.MUSIC_ENGINE != "yue2")

        self.dur_combo = QComboBox()
        self.dur_combo.addItem("Bot decides", _BOT_DECIDES)
        for secs in _music.DURATIONS:
            self.dur_combo.addItem(f"{secs}s", secs)
        self.dur_combo.addItem("Custom...", _CUSTOM)
        self.dur_combo.setToolTip(
            "The song is written for this length and closes on its own outro "
            "(the render gets headroom). 'Bot decides' picks a length for the topic.")
        _default = getattr(_cfg, "MUSIC_DEFAULT_SECONDS", 60)
        if _default in _music.DURATIONS:
            self.dur_combo.setCurrentIndex(list(_music.DURATIONS).index(_default) + 1)
        set_row.addWidget(QLabel("Length:"))
        set_row.addWidget(self.dur_combo)
        self.dur_spin = QSpinBox()
        self.dur_spin.setRange(*_music.DURATION_RANGE)
        self.dur_spin.setValue(_default)
        self.dur_spin.setSuffix(" s")
        self.dur_spin.setToolTip(f"{_music.DURATION_RANGE[0]}-{_music.DURATION_RANGE[1]} seconds.")
        self.dur_spin.setVisible(False)
        self.dur_combo.currentIndexChanged.connect(
            lambda _i: self.dur_spin.setVisible(self.dur_combo.currentData() == _CUSTOM))
        set_row.addWidget(self.dur_spin)
        set_row.addStretch(1)
        root.addLayout(set_row)

        self.status = QLabel("Idle.")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color:{MUTED};")
        root.addWidget(self.status)

        play_row = QHBoxLayout()
        self.play_btn = QPushButton("▶  Play"); self.play_btn.setObjectName("ghost")
        self.play_btn.clicked.connect(self._play_last)
        self.play_btn.setEnabled(False)
        play_row.addWidget(self.play_btn)
        play_row.addStretch(1)
        root.addLayout(play_row)
        root.addStretch(1)

    # ----- actions -----------------------------------------------------------
    def _busy(self) -> bool:
        return self.music_worker is not None and self.music_worker.isRunning()

    def _settings(self) -> tuple:
        """(prefs, duration_s, preset) from the pickers, via music.py.

        Returns plain data so the worker never has to touch a widget, and goes
        through music.prefs_from rather than mapping here — the Telegram
        picker resolves the same keys the same way, and a second mapping is a
        second place for "jazz" to come to mean something else.
        """
        import music as _music
        keys = {f: (b.currentData() or "") for f, b in self._combos.items()}
        if keys.get("tempo") == _CUSTOM:
            keys["tempo"] = f"bpm:{self.bpm_spin.value()}"
        prefs = _music.prefs_from(**keys)
        d = self.dur_combo.currentData()
        if d == _CUSTOM:
            duration = int(self.dur_spin.value())
        elif d == _BOT_DECIDES:
            duration = 0                       # the worker asks the bot
        else:
            duration = int(d or 0)
        return (prefs, duration,
                self.qual_combo.currentData() or _music.DEFAULT_PRESET)

    def _refresh_eta(self):
        """Redraw the measured wait for every preset at the chosen steps,
        the current preset in bold."""
        import music as _music
        steps = int(self.steps_spin.value())
        cur = self.qual_combo.currentData() or _music.DEFAULT_PRESET
        parts = []
        for name in _music.WEIGHT_PRESETS:
            eta = _music.eta_label(_music.eta_seconds(name, steps=steps), "en")
            piece = f"{name.title()} {eta}"
            parts.append(f"<b>{piece}</b>" if name == cur else piece)
        self.eta_lbl.setText(" · ".join(parts))
        self.eta_lbl.setToolTip(f"Expected wait for a {_music.ETA_QUOTE_SECONDS}-second "
                                f"song at {steps} steps, per preset.")

    def _generate(self):
        topic = self.topic_in.text().strip()
        if not topic:
            self.status.setText('Write what the song is about first.')
            return
        ctx = getattr(self.host, "ctx", None)
        # Two different "no model" states now. ctx is None means the runtime is
        # still coming up; a ctx with an empty model_name is the deliberate
        # "start without a model" choice, and it needs a different sentence --
        # waiting will not fix it.
        if ctx is None:
            self.status.setText('Models are still loading — wait a moment.')
            return
        if not (getattr(ctx, "model_name", "") or "").strip():
            self.status.setText('No model is loaded (started without one) — pick it in Settings → Model Config → Apply.')
            return
        if self._busy():
            self.status.setText('A track is already being generated — wait for it to finish.')
            return
        self.play_btn.setEnabled(False)
        self.gen_btn.setEnabled(False)
        # Read the pickers HERE, on the UI thread, and hand the worker plain
        # data — see MusicWorker.__init__ for why it must not read widgets.
        prefs, duration, preset = self._settings()
        steps = int(self.steps_spin.value())
        self.status.setText(f"Generating song for: {topic}…")
        lang = getattr(ctx, "lang", None) or "ru"
        self.music_worker = MusicWorker(ctx, topic, lang,
                                        prefs=prefs, duration_s=duration,
                                        preset=preset, steps=steps)
        self._last_chosen = {}
        self.music_worker.chose.connect(self._on_chose)
        self.music_worker.done.connect(self._on_done)
        self.music_worker.failed.connect(self._on_failed)
        self.music_worker.finished.connect(self._on_finished)
        self.music_worker.start()

    def _on_done(self, wav_path: str, lyrics: str, style: str):
        if not wav_path:
            self.status.setText("No audio produced.")
            return
        self._last_wav = wav_path
        self.play_btn.setEnabled(True)
        text = f"Done — style: {style or '(none)'}\nSaved: {wav_path}"
        chosen = getattr(self, "_last_chosen", {}) or {}
        if chosen:
            text += chr(10) + self._chose_text(chosen)
        self.status.setText(text)

    def _on_chose(self, chosen: dict):
        self._last_chosen = dict(chosen or {})

    @staticmethod
    def _chose_text(chosen: dict) -> str:
        """'The bot chose: genre - Jazz, tempo - 92 BPM, length - 75 s' + why,
        the same report the Telegram side sends after the song."""
        items = []
        if chosen.get("genre"):
            items.append("genre - " + _GENRE_LABELS.get(chosen["genre"], chosen["genre"]))
        if chosen.get("bpm"):
            items.append(f"tempo - {chosen['bpm']} BPM")
        if chosen.get("vocal"):
            items.append("vocal - " + _VOCAL_LABELS.get(chosen["vocal"], chosen["vocal"]))
        if chosen.get("seconds"):
            items.append(f"length - {chosen['seconds']} s")
        line = "The bot chose: " + ", ".join(items)
        if chosen.get("why"):
            line += chr(10) + chosen["why"]
        return line

    def _on_failed(self, msg: str):
        self.status.setText(f"Music generation unavailable: {msg}")

    def _on_finished(self):
        self.gen_btn.setEnabled(True)
        self.music_worker = None

    def _play_last(self):
        if not self._last_wav:
            return
        try:
            self.player.play(self._last_wav)
        except Exception as exc:
            logger.exception("Music playback failed")
            self.status.setText(f"Playback failed: {exc}")

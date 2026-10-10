"""Startup settings dialog: model, endpoint, reasoning, and UI scale.

Shown before the main window exists, which is why apply_ui_scale travels with
it: the scale chosen here has to be applied to the QApplication BEFORE any
widget is built, since px()/pt() are read at construction time and a widget
already laid out will not re-scale.

Both consumers — AssistantWindow and run_gui — stay in gui.py, so gui.py
re-exports these two and the suites that patch gui.SettingsDialog (four of
them) keep intercepting exactly as before.
"""
import os

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox,
                             QDialog, QDialogButtonBox, QFileDialog, QFrame,
                             QHBoxLayout, QLabel, QPushButton, QRadioButton,
                             QScrollArea, QSpinBox, QTextEdit, QVBoxLayout,
                             QWidget)

import ui_scale
from ui_scale import pt, px, scale_style as _ss
# Imported as a MODULE, not by value: STARTUP_MODEL and the timeout are read at
# COUNTDOWN time so a test (or an env change between launches) can set them on
# config and have this dialog follow, which a by-value copy would not.
import config as _config
from config import LM_STUDIO_BASE, MODEL_NAME
from gui_common import MUTED, REC, _fit_dialog, _section, build_qss
from gui_i18n import tr

import logging
logger = logging.getLogger("assistant.gui")


# --------------------------------------------------------------------------- #
# Settings / model dialog
# --------------------------------------------------------------------------- #
# Chosen instead of a model id when the user wants the card left alone -- a
# training run owns the GPU, and loading a 19 GB chat model beside it is what
# makes the run crawl or die.
#
# Module level rather than only a class attribute: several suites replace
# SettingsDialog with a small stub, and _open_settings comparing against
# SomeStub.NO_MODEL raised AttributeError inside a Qt slot -- which is a native
# abort with no traceback, not a test failure. The comparison should not depend
# on the dialog class the caller happens to be holding.
NO_MODEL = "__no_model__"
_NO_MODEL = NO_MODEL


class SettingsDialog(QDialog):
    """Model + thinking always; voice + search settings when a ctx is given."""

    def __init__(self, default_model: str, ctx=None, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self._default_model = default_model
        self.setWindowTitle("Settings")
        _fit_dialog(self, px(600), px(660))
        import search as _search
        self._search = _search

        # The dialog can be taller than the screen (long model list + many sections),
        # which used to push the OK/Cancel buttons off the bottom edge so they couldn't
        # be clicked. Fix: put ALL settings widgets inside an outer scroll area, and
        # pin the button box BELOW it so it is always visible and clickable.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        _scroll_all = QScrollArea(self)
        _scroll_all.setWidgetResizable(True)
        _scroll_all.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        _scroll_all.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        _scroll_all.setFrameShape(QFrame.NoFrame)
        # Override the global QScrollArea skin (dark fill + border + padding) for this
        # OUTER wrapper so it doesn't paint a light/bordered box around the whole dialog
        # — the dark QDialog background must show through instead.
        _scroll_all.setStyleSheet("QScrollArea { background: transparent; border: none; padding: 0; }")
        _scroll_all.viewport().setStyleSheet("background: transparent;")
        _content = QWidget()
        _content.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(_content)  # all settings widgets are added to `lay`
        mrow = QHBoxLayout()
        mrow.addWidget(_section("Model"))
        mrow.addStretch(1)
        refresh = QPushButton("↻ Refresh"); refresh.setObjectName("ghost")
        refresh.clicked.connect(self._populate_models)
        mrow.addWidget(refresh)
        lay.addLayout(mrow)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        # Long model ids must wrap, not force a horizontal scrollbar that hides the
        # name and squashes the list into a thin strip.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setMinimumHeight(px(220))
        host = QWidget()
        host.setObjectName("radiohost")  # dark background so the model names are readable
        self._radio_layout = QVBoxLayout(host)
        self._radio_layout.setAlignment(Qt.AlignTop)
        self._radio_layout.setSpacing(2)
        scroll.setWidget(host)
        lay.addWidget(scroll, 1)
        self._radio_group = QButtonGroup(self)
        self._radios = []
        self._populate_models()

        self.think = QCheckBox("Enable thinking / reasoning (off by default; several times slower)")
        if ctx:
            self.think.setChecked(not ctx.no_think)
        lay.addWidget(self.think)
        _hint = QLabel("Off = direct mode: short answers, fewer tool calls, acts "
                       "instead of planning out loud.")
        _hint.setWordWrap(True)
        _hint.setStyleSheet(_ss(f"color:{MUTED}; font-size:11px;"))
        lay.addWidget(_hint)
        self._think_hint = _hint
        self._think_hint_default = _hint.text()
        # The radios were built before this checkbox existed; sync it now for
        # whichever model started out selected (locks it if that's a virtual).
        self._sync_think_for_model(getattr(self, "_initial_model_id", None))

        # Response length — an axis independent of the thinking toggle: it controls
        # how long the FINAL answer is (system-prompt directive + token budget),
        # not how much the model reasons. Auto lets the model size the reply itself.
        _LEN_LABELS = [("Auto (model decides)", "auto"),
                       ("Ultra-short (one sentence)", "ultra"),
                       ("Short (1–3 sentences)", "short"),
                       ("Detailed (longer, fuller)", "long")]
        lrow = QHBoxLayout()
        lrow.addWidget(QLabel("Response length"))
        self.resp_length = QComboBox()
        for label, _ in _LEN_LABELS:
            self.resp_length.addItem(label)
        self._len_values = [v for _, v in _LEN_LABELS]
        cur_len = (getattr(ctx, "response_length", "auto") if ctx else "auto") or "auto"
        self.resp_length.setCurrentIndex(
            self._len_values.index(cur_len) if cur_len in self._len_values else 0)
        self.resp_length.setToolTip(
            "How long the assistant's replies should be. Auto = sized to the question; "
            "Short = terse 1–3 sentences; Detailed = thorough, multi-paragraph answers. "
            "Independent of the thinking toggle above.")
        lrow.addWidget(self.resp_length, 1)
        lay.addLayout(lrow)

        # gpt-oss reasoning depth. Only gpt-oss models honor `reasoning_effort`
        # (low/medium/high), so the row shows only when one is in the list: next to
        # Gemma it was a knob that did nothing. Default high = deepest reasoning.
        self._effort_row = QWidget()
        erow = QHBoxLayout(self._effort_row)
        erow.setContentsMargins(0, 0, 0, 0)
        erow.addWidget(QLabel("Reasoning effort (gpt-oss)"))
        self.reasoning_effort = QComboBox()
        self.reasoning_effort.addItems(["low", "medium", "high"])
        self.reasoning_effort.setCurrentText(
            getattr(ctx, "reasoning_effort", "high") if ctx else "high")
        self.reasoning_effort.setToolTip(
            "Reasoning depth for OpenAI gpt-oss models (gpt-oss-20b / 120b): "
            "high = deepest, most thorough chains; low = fastest. Ignored by non-gpt-oss models.")
        erow.addWidget(self.reasoning_effort, 1)
        lay.addWidget(self._effort_row)
        self._effort_row.setVisible(getattr(self, "_has_gpt_oss", False))

        # ---- Interface (DPI / resolution-aware scaling) — app-level, always shown ----
        lay.addWidget(_section("Interface"))
        irow = QHBoxLayout()
        irow.addWidget(QLabel("UI Scale"))
        self.ui_scale_combo = QComboBox()
        self.ui_scale_combo.addItems(ui_scale.SCALE_OPTIONS)
        self.ui_scale_combo.setCurrentIndex(ui_scale.mode_to_combo_index(ui_scale.scale_mode()))
        auto_pct = int(round(ui_scale.auto_value() * 100))
        self.ui_scale_combo.setToolTip(
            "How large the whole interface is drawn. 'Auto' picks a sensible size from "
            f"your screen resolution and DPI (currently ~{auto_pct}%). Increase it for a "
            "4K TV viewed from the couch, or if text/controls look too small.")
        irow.addWidget(self.ui_scale_combo, 1)
        lay.addLayout(irow)

        drow = QHBoxLayout()
        drow.addWidget(QLabel("Density"))
        self.density_combo = QComboBox()
        self.density_combo.addItems(ui_scale.DENSITY_OPTIONS)
        self.density_combo.setCurrentIndex(ui_scale.density_index())
        self.density_combo.setToolTip("Compact = tighter for small screens · Comfortable = "
                                      "default · TV Mode = extra-large fonts, buttons, spacing, "
                                      "tooltips and menus for couch viewing.")
        drow.addWidget(self.density_combo, 1)
        drow.addWidget(QLabel("Font Scale"))
        self.font_combo = QComboBox()
        self.font_combo.addItems(ui_scale.FONT_OPTIONS)
        self.font_combo.setCurrentIndex(ui_scale.font_index())
        self.font_combo.setToolTip("Extra multiplier applied to text only (on top of UI Scale).")
        drow.addWidget(self.font_combo, 1)
        lay.addLayout(drow)

        lrow = QHBoxLayout()
        lrow.addWidget(QLabel("Language"))
        self.lang_combo = QComboBox()
        for code, name in ui_scale.LANGUAGES.items():
            self.lang_combo.addItem(name, code)
        self.lang_combo.setCurrentIndex(list(ui_scale.LANGUAGES).index(ui_scale.language()))
        self.lang_combo.setToolTip("Language of every button, label and status. Applies after a restart.")
        lrow.addWidget(self.lang_combo, 1)
        lay.addLayout(lrow)

        self.ui_scale_hint = QLabel("")
        self.ui_scale_hint.setStyleSheet(_ss(f"color:{MUTED}; font-size:11px;"))
        self.ui_scale_hint.setWordWrap(True)
        lay.addWidget(self.ui_scale_hint)

        if ctx is not None:
            lay.addWidget(_section("Voice"))
            vrow = QHBoxLayout()
            self.voice_lbl = QLabel(os.path.basename(ctx.custom_ref_wav) if ctx.custom_ref_wav else "default")
            vbtn = QPushButton("Choose…"); vbtn.setObjectName("ghost"); vbtn.clicked.connect(self._pick_voice)
            vclr = QPushButton("Reset"); vclr.setObjectName("ghost"); vclr.clicked.connect(self._reset_voice)
            vrow.addWidget(self.voice_lbl, 1); vrow.addWidget(vbtn); vrow.addWidget(vclr)
            lay.addLayout(vrow)

            lay.addWidget(_section("Personality"))
            prow = QHBoxLayout()
            _plbl = os.path.basename(ctx.custom_personality_path) if ctx.custom_personality_path else "none (default assistant)"
            self.pers_lbl = QLabel(_plbl)
            pbtn = QPushButton("Choose…"); pbtn.setObjectName("ghost"); pbtn.clicked.connect(self._pick_personality)
            pedit = QPushButton("Edit"); pedit.setObjectName("ghost"); pedit.clicked.connect(self._edit_personality)
            pclr = QPushButton("Reset"); pclr.setObjectName("ghost"); pclr.clicked.connect(self._reset_personality)
            prow.addWidget(self.pers_lbl, 1); prow.addWidget(pbtn); prow.addWidget(pedit); prow.addWidget(pclr)
            lay.addLayout(prow)

            lay.addWidget(_section("Web search"))
            self.search_off = QCheckBox("Disable internet search")
            self.search_off.setChecked(not bool(getattr(ctx, "web_search_enabled", True)))
            self.search_off.setToolTip(
                "When checked, the assistant never goes online — the web search and deep "
                "research tools are withheld and it answers from its own knowledge. "
                "Image generation and other tools are unaffected.")
            lay.addWidget(self.search_off)
            self.distill = QCheckBox("Distill results into grounded facts")
            self.distill.setChecked(bool(_search.SEARCH_DISTILL))
            lay.addWidget(self.distill)
            self.refperson = QCheckBox("Reference-photo mode for real people (off by default)")
            self.refperson.setChecked(bool(getattr(ctx, "reference_person_mode", False)))
            self.refperson.setToolTip(
                "When checked, drawing a real, named public figure first fetches an actual "
                "photo of them from the web and uses it as the identity reference (plus a read "
                "of their appearance), then renders the requested outfit/scene onto it — so the "
                "likeness matches the real person instead of a from-scratch guess. Requires "
                "internet search to be enabled. Slower. Generic subjects are unaffected.")
            lay.addWidget(self.refperson)
            srow = QHBoxLayout()
            srow.addWidget(QLabel("Max results"))
            self.maxres = QSpinBox(); self.maxres.setRange(1, 10); self.maxres.setValue(int(_search.SEARCH_MAX_RESULTS))
            srow.addWidget(self.maxres)
            srow.addWidget(QLabel("Region"))
            self.region = QComboBox(); self.region.addItems(["wt-wt", "us-en", "fr-fr", "ru-ru", "uk-en", "de-de"])
            self.region.setCurrentText(str(_search.SEARCH_REGION))
            srow.addWidget(self.region, 1)
            lay.addLayout(srow)

            lay.addWidget(_section("Input / Output"))
            self.mic_off = QCheckBox("Disable microphone (voice input)")
            self.mic_off.setChecked(bool(ctx.mic_disabled))
            self.mic_off.setToolTip("When checked, the Talk button is disabled and the assistant never listens to the microphone.")
            lay.addWidget(self.mic_off)
            self.voice_off = QCheckBox("Disable voice output (TTS — text only mode)")
            self.voice_off.setChecked(bool(ctx.tts_disabled))
            self.voice_off.setToolTip("When checked, the assistant replies in text only — no speech synthesis runs.")
            lay.addWidget(self.voice_off)

        # Close the scrollable content; pin the button box outside (always visible).
        _scroll_all.setWidget(_content)
        outer.addWidget(_scroll_all, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._ok_btn = buttons.button(QDialogButtonBox.Ok)
        self._ok_btn.setText("Apply / Select model")
        buttons.button(QDialogButtonBox.Cancel).setText("Cancel")   # Qt's own label skips our tr()
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self._start_autostart_countdown()

    # ── unattended startup ───────────────────────────────────────────────────
    def _start_autostart_countdown(self):
        """Start on the fallback model if nobody answers this dialog.

        This dialog is MODAL and blocks before the main window (and therefore
        the Telegram bot) exists, so a machine that is rebooted, or launched by
        anything other than a person sitting in front of it, ends up with a bot
        that is silently offline for as long as nobody notices. That happened:
        a forwarded message looked like it had crashed the app, when really the
        app had been parked on this dialog and never polled at all.

        The countdown is visible on the OK button rather than silent, and ANY
        key or click cancels it -- being yanked out of a deliberate choice
        halfway through would be worse than the problem being fixed.
        """
        self._auto_left = int(getattr(_config, "STARTUP_DIALOG_TIMEOUT_S", 30) or 0)
        self._auto_timer = None
        if self._auto_left <= 0:
            return
        self._auto_timer = QTimer(self)
        self._auto_timer.setInterval(1000)
        self._auto_timer.timeout.connect(self._autostart_tick)
        self._auto_timer.start()
        self._autostart_paint()

    def _autostart_paint(self):
        if self._auto_timer is not None:
            self._ok_btn.setText("Apply / Select model  (auto in %ds)" % self._auto_left)

    def _cancel_autostart(self):
        """A human is here -- stop the clock and say so on the button."""
        if getattr(self, "_auto_timer", None) is not None:
            self._auto_timer.stop()
            self._auto_timer = None
            self._ok_btn.setText("Apply / Select model")

    def _autostart_tick(self):
        self._auto_left -= 1
        if self._auto_left > 0:
            self._autostart_paint()
            return
        self._cancel_autostart()
        target = getattr(_config, "STARTUP_MODEL", "") or ""
        # Select the fallback model if LM Studio is actually serving it. If it
        # is not, whatever _populate_models already checked is left alone and
        # accepted -- starting on a model that is not there would fail every
        # call, which is worse than starting on the wrong one.
        picked = ""
        for rb in self._radios:
            if rb.property("model_id") == target:
                rb.setChecked(True)
                picked = target
                break
        if not picked and target:
            # STARTUP_MODEL often names a repo that several quants share
            # ("...-balanced" matches both @q5_k_m and the q5_k_p .gguf). The
            # dialog's own default checked ANY of them, and live 2026-09-22 it
            # checked q5_k_m while the Telegram bot was mid-answer on q5_k_p:
            # starting the GUI unloaded the bot's model under it ("Model
            # unloaded by user or API request") and the photo reply was lost.
            # Keep whichever matching quant is ALREADY loaded, else the first.
            matches = [rb for rb in self._radios
                       if target.lower() in str(rb.property("model_id") or "").lower()]
            matches.sort(key=lambda rb: not rb.property("model_loaded"))
            if matches:
                matches[0].setChecked(True)
                picked = matches[0].property("model_id")
                logger.info("LIFECYCLE: startup model %r matched %s (%s)", target, picked,
                            "already loaded" if matches[0].property("model_loaded")
                            else "not loaded")
        if not picked:
            btn = self._radio_group.checkedButton()
            picked = (btn.property("model_id") if btn is not None else "") or "(none)"
            logger.warning("LIFECYCLE: startup dialog timed out but %r is not in "
                           "the model list — starting on %s instead",
                           target, picked)
        logger.info("LIFECYCLE: startup dialog timed out after %ss — starting "
                    "unattended on %s",
                    getattr(_config, "STARTUP_DIALOG_TIMEOUT_S", 30), picked)
        self.accept()

    # Any interaction at all means a person is choosing; the countdown must not
    # fire under them. Covered at the DIALOG level rather than per-widget so a
    # control added later cannot silently miss it.
    def keyPressEvent(self, event):
        self._cancel_autostart()
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        self._cancel_autostart()
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        self._cancel_autostart()
        super().wheelEvent(event)

    # The sentinel lives at MODULE level (below) and is mirrored here only for
    # readability at the call sites that already have the class in hand.
    NO_MODEL = _NO_MODEL

    def _populate_models(self):
        from model_selector import list_models, _index_model_sizes, _size_gb, _MODELS_DIR
        # clear existing radios / placeholder
        for b in self._radios:
            self._radio_group.removeButton(b)
        self._radios = []
        while self._radio_layout.count():
            item = self._radio_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        models = list_models(LM_STUDIO_BASE)
        self._has_gpt_oss = any("gpt-oss" in (m.get("id") or "") for m in models or [])
        if hasattr(self, "_effort_row"):
            self._effort_row.setVisible(self._has_gpt_oss)
        sizes = _index_model_sizes(_MODELS_DIR)
        cur_model = (self.ctx.model_name if self.ctx else self._default_model)
        self._initial_model_id = cur_model

        if not models:
            lbl = QLabel("(LM Studio unreachable — start it, then click ↻ Refresh)")
            lbl.setStyleSheet(f"color:{MUTED};")
            self._radio_layout.addWidget(lbl)
            return

        # First row: leave the card alone. Placed above the models so it is the
        # first thing seen when the reason for opening this dialog is "training
        # is running and I do not want anything loaded".
        _none_row = QWidget()
        _nl = QHBoxLayout(_none_row)
        _nl.setContentsMargins(2, 2, 2, 2)
        _nl.setSpacing(8)
        _none_rb = QRadioButton()
        _none_rb.setProperty("model_id", self.NO_MODEL)
        _none_lab = QLabel('—   do not load a model (free the VRAM)')
        _none_lab.setWordWrap(True)
        _none_lab.setTextInteractionFlags(Qt.NoTextInteraction)
        _none_lab.mousePressEvent = lambda _e, b=_none_rb: b.setChecked(True)
        _nl.addWidget(_none_rb, 0, Qt.AlignTop)
        _nl.addWidget(_none_lab, 1)
        self._radio_group.addButton(_none_rb)
        self._radio_layout.addWidget(_none_row)
        self._radios.append(_none_rb)

        for m in models:
            mid = m.get("id", "")
            gb = _size_gb(mid, sizes)
            size_str = f"{gb:.1f} GB" if gb is not None else "?"
            loaded = "   ·   " + tr("loaded") if m.get("state") == "loaded" else ""
            # Row = radio (no text) + a word-wrapping label, so long model ids are
            # fully readable instead of being clipped by a horizontal scrollbar.
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(2, 2, 2, 2)
            rl.setSpacing(8)
            rb = QRadioButton()
            rb.setProperty("model_id", mid)
            rb.setProperty("model_loaded", m.get("state") == "loaded")
            lab = QLabel(f"{size_str}   {mid}{loaded}")
            lab.setWordWrap(True)
            lab.setTextInteractionFlags(Qt.NoTextInteraction)
            # clicking the label selects its radio (bigger, easier hit target)
            lab.mousePressEvent = lambda _e, b=rb: b.setChecked(True)
            rl.addWidget(rb, 0, Qt.AlignTop)
            rl.addWidget(lab, 1)
            self._radio_group.addButton(rb)
            self._radio_layout.addWidget(row)
            self._radios.append(rb)
            # LM Studio can't programmatically load a virtual (model.yaml) model —
            # ensure_exclusive() falls back to its raw base GGUF and relies on the
            # runtime no-think prefill to reproduce the virtual's baked-in
            # thinking setting. That only actually happens if ctx.no_think ends up
            # set correctly, so the checkbox must track (and lock to) whatever the
            # selected virtual declares, instead of leaving it at the user's last
            # manual choice — a mismatch there silently loses the virtual's whole
            # reason for existing (the no-think behavior).
            rb.toggled.connect(lambda checked, m=mid: checked and self._sync_think_for_model(m))
            if mid == cur_model:
                rb.setChecked(True)
        if self._radio_group.checkedButton() is None:
            # Fall back to a real model, never to "do not load" -- that option
            # must be a deliberate choice, not what happens when the current
            # model id no longer matches anything on the list.
            real = [b for b in self._radios
                    if b.property("model_id") != self.NO_MODEL]
            if real:
                real[0].setChecked(True)

    def _pick_voice(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Reference voice", "", "Audio (*.wav *.mp3 *.ogg *.m4a *.flac *.aac);;All files (*)")
        if path:
            if self.ctx:
                self.ctx.custom_ref_wav = path
            self.voice_lbl.setText(os.path.basename(path))

    def _reset_voice(self):
        if self.ctx:
            self.ctx.custom_ref_wav = None
            self.voice_lbl.setText("default")

    def _pick_personality(self):
        personalities_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "personalities")
        path, _ = QFileDialog.getOpenFileName(
            self, "Personality file", personalities_dir, "Text files (*.txt);;All files (*)")
        if path:
            self._load_personality(path)

    def _load_personality(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as exc:
            self.pers_lbl.setText(f"(read error: {exc})")
            self.pers_lbl.setStyleSheet(f"color:{REC};")
            return
        if self.ctx:
            self.ctx.custom_personality_path = path
            self.ctx.custom_personality_text = text
        self.pers_lbl.setText(os.path.basename(path))
        self.pers_lbl.setStyleSheet("")

    def _edit_personality(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Edit personality")
        _fit_dialog(dlg, px(600), px(480))
        vl = QVBoxLayout(dlg)
        editor = QTextEdit()
        editor.setPlaceholderText("Describe the character here. The assistant will embody it silently — it will never announce or reference these instructions.")
        editor.setStyleSheet(_ss("background:#1e1e1e; color:#e0e0e0; font-family:monospace; font-size:13px;"))
        if self.ctx and self.ctx.custom_personality_text:
            editor.setPlainText(self.ctx.custom_personality_text)
        elif self.ctx and self.ctx.custom_personality_path:
            try:
                with open(self.ctx.custom_personality_path, "r", encoding="utf-8") as f:
                    editor.setPlainText(f.read())
            except Exception:
                pass
        vl.addWidget(editor)
        hint = QLabel("Changes are applied immediately. To save permanently, use Save.")
        hint.setStyleSheet(_ss("color:#888; font-size:11px;"))
        vl.addWidget(hint)
        btns = QHBoxLayout()
        save_btn = QPushButton("Save to file")
        apply_btn = QPushButton("Apply (this session)")
        cancel_btn = QPushButton("Cancel")
        btns.addWidget(save_btn); btns.addWidget(apply_btn); btns.addStretch(); btns.addWidget(cancel_btn)
        vl.addLayout(btns)

        def _apply():
            if self.ctx:
                self.ctx.custom_personality_text = editor.toPlainText()
                if not self.ctx.custom_personality_path:
                    self.pers_lbl.setText("(inline)")
            dlg.accept()

        def _save():
            personalities_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "personalities")
            default_path = (self.ctx.custom_personality_path if self.ctx else None) or os.path.join(personalities_dir, "custom.txt")
            path, _ = QFileDialog.getSaveFileName(dlg, "Save personality", default_path, "Text files (*.txt)")
            if path:
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(editor.toPlainText())
                except Exception as exc:
                    hint.setText(f"Save failed: {exc}")
                    hint.setStyleSheet(_ss("color:#e23b3b; font-size:11px;"))
                    return
                if self.ctx:
                    self.ctx.custom_personality_path = path
                    self.ctx.custom_personality_text = editor.toPlainText()
                self.pers_lbl.setText(os.path.basename(path))
                self.pers_lbl.setStyleSheet("")
                dlg.accept()

        save_btn.clicked.connect(_save)
        apply_btn.clicked.connect(_apply)
        cancel_btn.clicked.connect(dlg.reject)
        dlg.exec_()

    def _reset_personality(self):
        if self.ctx:
            self.ctx.custom_personality_path = None
            self.ctx.custom_personality_text = ""
            self.pers_lbl.setText("none (default assistant)")

    def _sync_think_for_model(self, model_id):
        """Lock the thinking checkbox to whatever a selected virtual declares.

        A virtual (model.yaml) model is served under its raw base GGUF (LM Studio
        can't load virtuals programmatically — see lmstudio.resolve_virtual_model),
        so the ONLY place the virtual's baked-in thinking setting survives is this
        checkbox's resulting ctx.no_think. Leaving it at the user's last manual
        choice would silently discard the very thing a "...-NO-THINK" preset exists
        for. Non-virtual selections re-enable the checkbox for manual control.
        Network/parse errors fail open (checkbox stays enabled, untouched)."""
        if not hasattr(self, "think") or not model_id:
            return
        try:
            from lmstudio import resolve_virtual_model
            raw_key, no_think = resolve_virtual_model(LM_STUDIO_BASE, model_id)
        except Exception as exc:
            logger.debug("_sync_think_for_model(%r) failed: %s", model_id, exc)
            return
        if raw_key:
            self.think.setChecked(not no_think)
            self.think.setEnabled(False)
            self._think_hint.setText(
                f"Locked by virtual preset '{model_id}' (thinking "
                f"{'OFF' if no_think else 'ON'}) — served as base model '{raw_key}'.")
        else:
            self.think.setEnabled(True)
            self._think_hint.setText(self._think_hint_default)

    def result_choice(self):
        effort = self.reasoning_effort.currentText()
        btn = self._radio_group.checkedButton()
        if btn is not None and btn.property("model_id"):
            return btn.property("model_id"), not self.think.isChecked(), effort
        fallback = self.ctx.model_name if self.ctx else MODEL_NAME
        return fallback, not self.think.isChecked(), effort

    def apply_settings(self):
        if self.ctx is None:
            return
        self.ctx.web_search_enabled = not self.search_off.isChecked()
        self.ctx.reference_person_mode = self.refperson.isChecked()
        self._search.SEARCH_DISTILL = self.distill.isChecked()
        self._search.SEARCH_MAX_RESULTS = self.maxres.value()
        self._search.SEARCH_REGION = self.region.currentText()
        self.ctx.mic_disabled = self.mic_off.isChecked()
        self.ctx.tts_disabled = self.voice_off.isChecked()
        self.ctx.response_length = self._len_values[self.resp_length.currentIndex()]

    def apply_language_setting(self) -> bool:
        """Persist the interface language. True if it changed (applies on restart)."""
        lang = self.lang_combo.currentData()
        if lang == ui_scale.language():
            return False
        ui_scale.set_language(lang)
        ui_scale.save()
        return True

    def apply_ui_scale_setting(self) -> bool:
        """Persist + apply the chosen UI Scale / TV Mode. Returns True if the effective
        scale changed (so the caller can hint that a restart fully applies it)."""
        before = (ui_scale.effective(), ui_scale.font_scale())
        mode = ui_scale.combo_index_to_mode(self.ui_scale_combo.currentIndex())
        density = ui_scale.index_to_density(self.density_combo.currentIndex())
        font = ui_scale.index_to_font(self.font_combo.currentIndex())
        ui_scale.set_mode(mode, density, font)
        ui_scale.save()
        app = QApplication.instance()
        if app is not None:
            apply_ui_scale(app)
        return (abs(ui_scale.effective() - before[0]) > 1e-3
                or abs(ui_scale.font_scale() - before[1]) > 1e-3)


def apply_ui_scale(app) -> None:
    """(Re)apply the current UI scale to the whole application: base font + QSS.

    QSS-driven styling (fonts, paddings, buttons, tabs, menus, tooltips, scrollbars)
    updates live; widgets with fixed pixel sizes set in their constructors only pick
    up the new scale on the next launch.
    """
    app.setFont(QFont("Segoe UI", pt(10)))
    app.setStyleSheet(build_qss())

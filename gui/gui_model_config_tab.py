"""Model configuration tab: pick the LLM, its context and sampling, reload it.

ReloadModelWorker moves with the tab because the tab is its only caller — a
model reload is a multi-second call that must not block the UI thread, and it
exists for no other reason.

Note for anyone patching this in a test: gui.ReloadModelWorker is no longer the
binding that matters. The tab resolves the name in THIS module, so a fake must
be installed here (gui_model_config_tab.ReloadModelWorker) or it will not
intercept — test_gui_adversarial learned that the hard way.
"""
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel,
                             QPushButton, QSpinBox, QTextEdit, QVBoxLayout,
                             QWidget)

from ui_scale import px, scale_style as _ss
from gui_common import BORDER, MUTED, _section

import logging
logger = logging.getLogger("assistant.gui")


# --------------------------------------------------------------------------- #
# LM Studio model config tab
# --------------------------------------------------------------------------- #
class ReloadModelWorker(QThread):
    # (status_msg, manual_instructions_or_empty)
    done = pyqtSignal(bool, str, str)

    def __init__(self, base_url: str, model_id: str,
                 context_length: int, gpu_pct: int, kv_quant: str, flash_attn: bool):
        super().__init__()
        self.base_url = base_url
        self.model_id = model_id
        self.context_length = context_length
        self.gpu_pct = gpu_pct
        self.kv_quant = kv_quant
        self.flash_attn = flash_attn

    def run(self):
        try:
            from lmstudio import reload_model_full
            result = reload_model_full(
                self.base_url, self.model_id,
                self.context_length, self.gpu_pct,
                self.kv_quant, self.flash_attn,
            )
            self.done.emit(result.ok, result.message, result.manual_settings)
        except Exception as exc:
            logger.exception("Model reload failed")
            self.done.emit(False, f"Reload error: {exc}", "")


class ModelConfigTab(QWidget):
    """Read-only info + reload-config panel for the currently active model."""

    _KV_OPTIONS = [
        ('auto (keep)',  "auto"),
        ('f16 — standard',    "f16"),
        ('q8_0 — saves ~30% VRAM', "q8_0"),
        ('q4_0 — saves ~50% VRAM', "q4_0"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._base_url = ""
        self._model_id = ""
        self._compat_type = ""
        self._reload_worker = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(px(10), px(10), px(10), px(10))
        root.setSpacing(8)

        # ── header ──
        hdr = QHBoxLayout()
        self._state_dot = QLabel("○")
        self._state_dot.setFixedWidth(px(16))
        refresh_btn = QPushButton('↻ Refresh')
        refresh_btn.setObjectName("ghost")
        refresh_btn.setToolTip('Re-read the model state from LM Studio')
        refresh_btn.clicked.connect(self.refresh)
        hdr.addWidget(self._state_dot)
        hdr.addStretch(1)
        hdr.addWidget(refresh_btn)
        root.addLayout(hdr)

        self._model_lbl = QLabel('(no model)')
        self._model_lbl.setStyleSheet(_ss("font-weight:bold; font-size:12px;"))
        self._model_lbl.setWordWrap(True)
        root.addWidget(self._model_lbl)

        # ── info grid ──
        info_grid = QVBoxLayout()
        info_grid.setSpacing(2)
        self._info_lbl = QLabel()
        self._info_lbl.setWordWrap(True)
        self._info_lbl.setStyleSheet(_ss(f"color:{MUTED}; font-size:12px;"))
        info_grid.addWidget(self._info_lbl)
        root.addLayout(info_grid)

        # ── separator ──
        sep = QFrame(); sep.setFrameShape(QFrame.HLine); sep.setStyleSheet(f"color:{BORDER};")
        root.addWidget(sep)

        # ── config form ──
        root.addWidget(_section('Load configuration'))

        form = QVBoxLayout()
        form.setSpacing(6)

        # Context length
        ctx_row = QHBoxLayout()
        ctx_row.addWidget(QLabel('Context (tokens)'))
        self._ctx_spin = QSpinBox()
        self._ctx_spin.setRange(512, 1048576)
        self._ctx_spin.setSingleStep(512)
        self._ctx_spin.setToolTip('Context window length. Longer = more VRAM.')
        ctx_row.addWidget(self._ctx_spin)
        self._ctx_max_lbl = QLabel()
        self._ctx_max_lbl.setStyleSheet(_ss(f"color:{MUTED}; font-size:11px;"))
        ctx_row.addWidget(self._ctx_max_lbl)
        form.addLayout(ctx_row)

        # KV cache quantization
        self._kv_row = QHBoxLayout()
        self._kv_row.addWidget(QLabel('KV cache quantization'))
        self._kv_combo = QComboBox()
        for label, _ in self._KV_OPTIONS:
            self._kv_combo.addItem(label)
        self._kv_combo.setToolTip(
            'KV cache quantization — saves VRAM on long contexts.\nGGUF models only (llama.cpp).'
        )
        self._kv_row.addWidget(self._kv_combo, 1)
        form.addLayout(self._kv_row)

        # GPU offload
        self._gpu_row = QHBoxLayout()
        self._gpu_row.addWidget(QLabel("GPU offload (%)"))
        self._gpu_spin = QSpinBox()
        self._gpu_spin.setRange(0, 100)
        self._gpu_spin.setValue(100)
        self._gpu_spin.setToolTip('Share of layers on the GPU. 100 = all layers on the GPU (fastest).')
        self._gpu_row.addWidget(self._gpu_spin)
        form.addLayout(self._gpu_row)

        # Flash attention
        self._fa_row = QHBoxLayout()
        self._flash_chk = QCheckBox("Flash Attention")
        self._flash_chk.setToolTip('Speeds up long contexts. GGUF only.')
        self._fa_row.addWidget(self._flash_chk)
        self._fa_row.addStretch(1)
        form.addLayout(self._fa_row)

        root.addLayout(form)

        # ── apply button ──
        self._apply_btn = QPushButton("⚡  Apply & Reload")
        self._apply_btn.setObjectName("ghost")
        self._apply_btn.clicked.connect(self._apply)
        self._apply_btn.setEnabled(False)
        root.addWidget(self._apply_btn)

        # ── status / manual instructions ──
        self._status = QLabel()
        self._status.setWordWrap(True)
        self._status.setStyleSheet(_ss(f"color:{MUTED}; font-size:11px;"))
        root.addWidget(self._status)

        self._manual_box = QTextEdit()
        self._manual_box.setReadOnly(True)
        self._manual_box.setMaximumHeight(px(110))
        self._manual_box.setObjectName("terminal")
        self._manual_box.setStyleSheet(_ss("font-size:11px;"))
        self._manual_box.hide()
        root.addWidget(self._manual_box)

        root.addStretch(1)
        self._set_gguf_fields_visible(False)

    def set_context(self, base_url: str, model_id: str):
        self._base_url = base_url
        self._model_id = model_id
        self.refresh()

    def refresh(self):
        if not self._base_url or not self._model_id:
            return
        from lmstudio import fetch_model
        info = fetch_model(self._base_url, self._model_id)
        self._populate(info)

    def _populate(self, info: dict):
        if not info:
            self._model_lbl.setText('No data — is LM Studio unreachable?')
            self._state_dot.setText("○")
            self._apply_btn.setEnabled(False)
            return

        state = info.get("state", "unknown")
        loaded = state == "loaded"
        self._state_dot.setText("●" if loaded else "○")
        self._state_dot.setStyleSheet("color:#4caf50;" if loaded else f"color:{MUTED};")
        self._model_lbl.setText(info.get("id", ""))

        # The API may omit these, or return null/0 — coerce to safe ints so the
        # spinbox and the f-string formatting below never see None.
        try:
            max_ctx = int(info.get("max_context_length") or 0)
        except (TypeError, ValueError):
            max_ctx = 0
        try:
            cur_ctx = int(info.get("loaded_context_length") or 0)
        except (TypeError, ValueError):
            cur_ctx = 0
        if cur_ctx <= 0:
            # loaded_context_length is only reported while the model is resident.
            # When it's absent (model unloaded / mid-load), KEEP what the spinbox
            # already shows — i.e. the value the user just asked for — instead of
            # snapping to an arbitrary max/4 (that was the "resets to 4k" bug).
            keep = self._ctx_spin.value()
            cur_ctx = keep if keep > 512 else 4096
        arch = info.get("arch", "?")
        compat = info.get("compatibility_type", "?")
        self._compat_type = compat
        quant = info.get("quantization", "?")
        caps = ", ".join(info.get("capabilities", []) or ["—"])

        lines = [
            f"Arch: {arch}",
            f"Type: {compat}   Quant: {quant}",
            f"Max context: {max_ctx:,}",
            (f"Loaded with: {cur_ctx:,}" if loaded else 'Not loaded'),
            f"Capabilities: {caps}",
        ]
        self._info_lbl.setText("\n".join(lines))

        # Populate form with sensible defaults
        if max_ctx:
            self._ctx_spin.setMaximum(max_ctx)
            self._ctx_max_lbl.setText(f"/ {max_ctx:,}")
        self._ctx_spin.setValue(cur_ctx)

        gguf = compat.lower() == "gguf"
        self._set_gguf_fields_visible(gguf)
        self._apply_btn.setEnabled(True)
        self._apply_btn.setText("⚡  Apply & Reload" if gguf else "⚡  Reload with new context")
        self._manual_box.hide()
        if loaded:
            if gguf:
                tip = 'Press «Apply & Reload» to reload with the new settings.'
            else:
                tip = 'Not a GGUF model — only the context length can change.'
            self._status.setText(tip)
        else:
            self._status.setText('The model is not loaded — press the button to load it.')

    def _set_gguf_fields_visible(self, visible: bool):
        for layout in (self._kv_row, self._gpu_row, self._fa_row):
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item and item.widget():
                    item.widget().setVisible(visible)

    def _kv_value(self) -> str:
        idx = self._kv_combo.currentIndex()
        return self._KV_OPTIONS[idx][1]

    def _host(self):
        """The AssistantWindow, if this tab is docked in one."""
        w = self.window()
        return w if hasattr(w, "_set_busy") and hasattr(w, "_busy") else None

    def _apply(self):
        if self._reload_worker is not None:
            return
        # Applying settings unloads and reloads the model in LM Studio — exactly
        # what _start_model_switch blocks chat input for. Without the same guard a
        # turn already in flight loses its model mid-request, and a message sent
        # during the reload hits a model that is not there.
        host = self._host()
        if host is not None and host._busy():
            self._status.setText('Wait for the current operation to finish — reloading the model would interrupt it.')
            return
        self._apply_btn.setEnabled(False)
        self._status.setText('Reloading through the lms CLI…')
        self._manual_box.hide()

        gguf = self._compat_type.lower() == "gguf"
        self._reload_worker = ReloadModelWorker(
            self._base_url, self._model_id,
            context_length=self._ctx_spin.value(),
            gpu_pct=self._gpu_spin.value() if gguf else 100,
            kv_quant=self._kv_value() if gguf else "auto",
            flash_attn=self._flash_chk.isChecked() if gguf else False,
        )
        self._reload_worker.done.connect(self._on_reload_done)
        self._reload_worker.finished.connect(self._on_worker_done)
        self._reload_worker.start()
        if host is not None:
            host._set_busy(True, "Reloading the model…")

    def _on_reload_done(self, ok: bool, message: str, manual: str):
        self._status.setText(message)
        if manual:
            self._manual_box.setPlainText(manual)
            self._manual_box.show()
        else:
            self._manual_box.hide()
        if ok:
            self.refresh()

    def _on_worker_done(self):
        self._reload_worker = None
        self._apply_btn.setEnabled(True)
        host = self._host()
        if host is not None:
            # Releases the gate and drains anything the user queued meanwhile.
            host._set_busy(False)

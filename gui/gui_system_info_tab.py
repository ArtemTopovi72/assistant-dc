"""System info tab: what is installed, what is missing, what version.

A diagnostics readout, not a control. It probes for optional packages by
importing them and reporting a tick or a cross, which is why it is worth
keeping away from anything that runs at startup — the probe is the whole
feature and it is deliberately slow and failure-tolerant.
"""
import os

from PyQt5.QtWidgets import (QHBoxLayout, QPushButton, QTextEdit,
                             QVBoxLayout, QWidget)

from ui_scale import px, scale_style as _ss
from config import MEMORY_DIR
from gui_common import MUTED, _section

import logging
logger = logging.getLogger("assistant.gui")


# --------------------------------------------------------------------------- #
# System / diagnostics tab
# --------------------------------------------------------------------------- #
class SystemInfoTab(QWidget):
    """Live diagnostics panel: model state, flags, memory, deps, tools."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ctx = None
        self._base_url = ""
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(px(10), px(10), px(10), px(10))
        root.setSpacing(6)
        hdr = QHBoxLayout()
        hdr.addWidget(_section("System Status"))
        hdr.addStretch(1)
        ref_btn = QPushButton("↻ Refresh")
        ref_btn.setObjectName("ghost")
        ref_btn.setToolTip("Re-read runtime status")
        ref_btn.clicked.connect(self.refresh)
        hdr.addWidget(ref_btn)
        root.addLayout(hdr)
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setObjectName("terminal")
        self._text.setStyleSheet(_ss("font-size:11px;"))
        root.addWidget(self._text, 1)

    def set_context(self, ctx, base_url: str):
        self._ctx = ctx
        self._base_url = base_url
        self.refresh()

    def refresh(self):
        lines = []
        ctx = self._ctx

        lines.append("─── Runtime ────────────────────────────")
        if ctx:
            lines.append(f"  Model      : {ctx.model_name or '(none)'}")
            lines.append(f"  Thinking   : {'OFF' if ctx.no_think else 'ON'}")
            lines.append(f"  Voice out  : {'MUTED' if ctx.tts_disabled else 'ON'}")
            lines.append(f"  Mic input  : {'DISABLED' if ctx.mic_disabled else 'ON'}")
            lines.append(f"  Web search : {'ON' if getattr(ctx, 'web_search_enabled', True) else 'DISABLED'}")
            pers = os.path.basename(ctx.custom_personality_path) if ctx.custom_personality_path else "default"
            if ctx.custom_personality_text and not ctx.custom_personality_path:
                pers = "(inline)"
            lines.append(f"  Personality: {pers}")
            ref = os.path.basename(ctx.custom_ref_wav) if ctx.custom_ref_wav else "default"
            lines.append(f"  Voice ref  : {ref}")
        else:
            lines.append("  (loading…)")

        lines.append("")
        lines.append("─── Memory ──────────────────────────────")
        if ctx:
            profile_dir = getattr(ctx, "active_memory_dir", None) or MEMORY_DIR / "default"
            lines.append(f"  Profile    : {profile_dir.name}")
            lines.append(f"  Items      : {len(ctx.session_memory)}")
            lines.append(f"  Location   : {profile_dir}")
        else:
            lines.append("  (not loaded)")

        lines.append("")
        lines.append("─── LM Studio ───────────────────────────")
        lines.append(f"  URL        : {self._base_url or '(not set)'}")

        lines.append("")
        lines.append("─── Tools ───────────────────────────────")
        try:
            from tools import TOOLS
            for t in TOOLS:
                lines.append(f"  ✓  {t.name}")
        except Exception as exc:
            lines.append(f"  (error loading tools: {exc})")

        lines.append("")
        lines.append("─── Models ──────────────────────────────")
        if ctx and ctx.models:
            lines.append(f"  Whisper    : {'loaded' if ctx.models.whisper else 'MISSING'}")
            lines.append(f"  TTS        : {'loaded' if ctx.models.tts_model else 'MISSING'}")
            lines.append(f"  Vocoder    : {'loaded' if ctx.models.vocoder else 'MISSING'}")
            lines.append(f"  Accentor   : {'loaded' if ctx.models.accentor_loaded else 'not loaded (optional)'}")
        else:
            lines.append("  (not loaded)")

        lines.append("")
        lines.append("─── Optional Dependencies ───────────────")
        for mod, label in [
            ("pypdf", "pypdf        (PDF drag-drop)"),
            ("silero_stress", "silero-stress (TTS accents)"),
            ("cv2", "opencv       (camera mode)"),
            ("trafilatura", "trafilatura  (Ultra Search crawler)"),
            ("bs4", "beautifulsoup (Ultra Search links)"),
        ]:
            try:
                __import__(mod)
                lines.append(f"  ✓  {label}")
            except ImportError:
                lines.append(f"  ✗  {label}  ← not installed")

        self._text.setPlainText("\n".join(lines))

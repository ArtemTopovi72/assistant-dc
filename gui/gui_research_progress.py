"""The Ultra Search / deep-research progress strip.

A mixin, not free functions: the phase stepper, its caption, the stat chips
and the log bar are all widgets on THIS window. Extracted from gui.py because
the strip is pure presentation — it reads a phase name and some counters and
paints them. It starts no run and owns no worker; the run and report view
live in gui_research_tab.py, which reaches the WebEngine globals and the
worker class back through `gui` at call time because the suites patch those
three names on the gui module.

_RESEARCH_PHASES lives here because this is the only code that walks it. The
methods left in gui.py index it through self, so the MRO resolves it and
there is still exactly one copy.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QGridLayout, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget,
)

from gui_common import ACCENT2, MUTED
from ui_scale import px, scale_style as _ss


class ResearchProgressMixin:
    """Builds and updates the research phase strip."""

    # Ordered Ultra Search phases — index drives the Research progress bar.
    _RESEARCH_PHASES = ("Expanding queries", "Searching", "Crawling", "Extracting",
                        "Deduplicating", "Verifying", "Building report", "Complete")

    def _build_research_progress_strip(self):
        """A live progress 'toolbar' that stays readable in a NARROW window: a
        compact dot-stepper (8 tiny dots, never clips) + the full current phase
        name spelled out with an N/8 counter, a live sub-step caption, an
        indeterminate 'busy' pulse so long single steps (e.g. reranking 160 pages)
        never look frozen, and stat chips in a WRAPPING grid (so values like
        'sources 227' don't get cut off). Hidden until a run starts."""
        self._progress_strip = QWidget()
        v = QVBoxLayout(self._progress_strip)
        v.setContentsMargins(0, 2, 0, 2)
        v.setSpacing(px(5))

        # Dot-stepper + full current-phase name. Dots are tiny and fixed-width so
        # all 8 fit any column; the spelled-out name is what the user reads.
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(px(3))
        self._phase_dots = []
        for name in self._RESEARCH_PHASES:
            d = QLabel("●")
            d.setToolTip(name)
            d.setAlignment(Qt.AlignCenter)
            d.setStyleSheet(self._dot_css(False, False))
            self._phase_dots.append(d)
            top.addWidget(d)
        top.addSpacing(px(6))
        self._phase_label = QLabel("—")
        self._phase_label.setStyleSheet(_ss("font-weight:700; font-size:12px;"))
        top.addWidget(self._phase_label)
        top.addStretch(1)
        tw = QWidget(); tw.setLayout(top)
        v.addWidget(tw)

        # Live sub-step caption (e.g. "reranking 164 pages → top 60") + pulse.
        self._progress_caption = QLabel("")
        self._progress_caption.setStyleSheet(_ss(f"color:{ACCENT2}; font-size:11px;"))
        self._progress_caption.setWordWrap(True)
        v.addWidget(self._progress_caption)
        self._busy_pulse = QProgressBar()
        self._busy_pulse.setRange(0, 0)            # indeterminate marquee
        self._busy_pulse.setTextVisible(False)
        self._busy_pulse.setFixedHeight(px(4))
        v.addWidget(self._busy_pulse)

        # Live log bar: tails the real pipeline logger so slow/silent stages (the
        # cross-encoder load, reranking, synthesis) show a heartbeat instead of
        # looking frozen. A small monospace icon + last line; rerank lines stick
        # (stay shown) so the reranker can't look like a "dead loop" again.
        logrow = QHBoxLayout()
        logrow.setContentsMargins(0, 0, 0, 0)
        logrow.setSpacing(px(5))
        tag = QLabel("⟳")
        tag.setStyleSheet(_ss(f"color:{MUTED}; font-size:11px;"))
        self._log_bar_tag = tag
        logrow.addWidget(tag)
        self._log_bar = QLabel("")
        self._log_bar.setStyleSheet(_ss(f"color:{MUTED}; font-family:Consolas,monospace; font-size:10px;"))
        # Wrap, don't clip: long lines like "RERANK[cross]: 202->60 kept top: [...]"
        # were being chopped at the panel edge. Wrapping shows the whole tail.
        self._log_bar.setWordWrap(True)
        self._log_bar.setTextInteractionFlags(Qt.TextSelectableByMouse)
        logrow.addWidget(self._log_bar, 1)
        lw = QWidget(); lw.setLayout(logrow)
        v.addWidget(lw)

        # Stat chips in a 3-column wrapping grid so nothing is clipped at any width.
        chips = QWidget()
        g = QGridLayout(chips)
        g.setContentsMargins(0, 0, 0, 0)
        g.setHorizontalSpacing(px(6))
        g.setVerticalSpacing(px(4))
        self._stat_chips = {}
        items = (("queries", "queries"), ("sources", "sources"), ("pages", "pages"),
                 ("unique_domains", "domains"), ("findings", "findings"),
                 ("loops_completed", "loop"))
        for idx, (key, label) in enumerate(items):
            chip = QLabel(f"{label} 0")
            chip.setObjectName("chip")
            chip._label = label
            self._stat_chips[key] = chip
            g.addWidget(chip, idx // 3, idx % 3)
        v.addWidget(chips)

        self._progress_strip.hide()
        return self._progress_strip

    @staticmethod
    def _dot_css(active: bool, done: bool) -> str:
        if active:
            col, size = ACCENT2, 15        # bright + bigger = clearly "here now"
        elif done:
            col, size = "#7fd18b", 12
        else:
            col, size = MUTED, 12
        return _ss(f"color:{col}; font-size:{size}px; min-width:13px;")

    def _update_progress_strip(self, phase: str, stats: dict, message: str = ""):
        try:
            cur = self._RESEARCH_PHASES.index(phase)
        except ValueError:
            cur = -1
        for i, dot in enumerate(self._phase_dots):
            dot.setStyleSheet(self._dot_css(i == cur, 0 <= i < cur))
        n = len(self._RESEARCH_PHASES)
        if 0 <= cur < n:
            self._phase_label.setText(f"{self._RESEARCH_PHASES[cur]}  ({cur + 1}/{n})")
        for key, chip in self._stat_chips.items():
            if key in stats:
                chip.setText(f"{chip._label} {stats.get(key, 0)}")
        if message:
            self._progress_caption.setText(message)
        done = (cur == n - 1)
        # Freeze the marquee on completion (determinate full bar); otherwise keep it
        # pulsing so a long single step (reranking, synthesis) shows live activity.
        if done:
            self._busy_pulse.setRange(0, 1)
            self._busy_pulse.setValue(1)
            self._progress_caption.setText("Complete.")
        else:
            self._busy_pulse.setRange(0, 0)

    def _on_research_progress(self, phase: str, stats: dict, message: str):
        # The progress strip (phase stepper + caption + stat chips + log bar) is the
        # single source of truth during a run; the top research_status label would
        # just duplicate the phase/counts/message, so it's hidden until the final
        # summary (see _start_deep_research / _on_research_done).
        try:
            self.research_bar.setValue(self._RESEARCH_PHASES.index(phase))
        except ValueError:
            pass
        self._update_progress_strip(phase, stats, message)
        self.stage.set_stage(f"Ultra: {phase}")

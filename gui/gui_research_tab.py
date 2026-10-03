"""The research tab's run and report view — the WebEngine-bound half.

Split out of gui.py, but not a leaf: `_QWebEngineView`, `_REPORT_HTML_ENABLED`
and `DeepResearchWorker` are read through `import gui as _g` at CALL time
rather than by value. Those three names are patched directly ON the gui
module by test_gui_webengine, test_gui_supplement{4,5,9,10,12} and
test_gui_workers — a by-value import here would freeze the real WebEngine
view / real worker at import time and every one of those patches would
silently stop reaching this code (the same dead-seam failure the tools.py
render split hit and fixed the same way: reach a dependency through the
module that owns it, never by value across a split).

The progress STRIP (phase stepper, stat chips, log bar) has no such
binding and already lives in gui_research_progress.py.
"""
from PyQt5.QtWidgets import QLabel, QProgressBar, QTextEdit, QVBoxLayout, QWidget

from gui_common import MUTED
from gui_report_html import _render_report_html
from ui_scale import px, scale_style as _ss


class ResearchTabMixin:
    """Builds the research tab and drives a research run to its report."""

    def _build_research_tab(self):
        """Ultra Search panel: live progress header + manual-control panel + rendered report view."""
        import gui as _g
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)
        self.research_status = QLabel("Ultra Search idle. Toggle “🔬 Ultra Search” and "
                                      "send a topic to start a deep research run.")
        self.research_status.setWordWrap(True)
        self.research_status.setStyleSheet(_ss(f"color:{MUTED}; font-size:12px;"))
        lay.addWidget(self.research_status)
        lay.addWidget(self._build_manual_control_panel())
        lay.addWidget(self._build_research_progress_strip())
        self.research_bar = QProgressBar()
        self.research_bar.setRange(0, len(self._RESEARCH_PHASES) - 1)
        self.research_bar.setValue(0)
        self.research_bar.setTextVisible(False)
        self.research_bar.setFixedHeight(px(6))
        self.research_bar.hide()
        lay.addWidget(self.research_bar)
        self.research_view = QTextEdit()
        self.research_view.setReadOnly(True)
        self.research_view.setPlaceholderText("The research report will appear here.")
        lay.addWidget(self.research_view, 1)
        # Rich report viewer: typeset math + tables via QWebEngine when available.
        # Falls back to the QTextEdit above if WebEngine isn't installed.
        self.research_web = None
        if _g._QWebEngineView is not None and _g._REPORT_HTML_ENABLED:
            try:
                self.research_web = _g._QWebEngineView()
                self.research_web.hide()
                lay.addWidget(self.research_web, 1)
            except Exception as exc:
                _g.logger.info("QWebEngine unavailable; reports render as plain markdown: %s", exc)
                self.research_web = None
        return page

    def _start_deep_research(self, topic: str):
        import gui as _g
        if self.ctx is None:
            return
        # Reachable WITHOUT going through _dispatch_user_text: the voice tab
        # calls it straight from _on_transcribe_finished. So the model gate has
        # to be here too, or a dictated research topic disappears in silence.
        if self._refuse_without_model():
            return
        if self._busy():
            # This used to return with no trace at all. Deep research is minutes
            # of work started from a button; a request that vanishes because
            # something else was running looks like the button is broken.
            self._add_system('Busy — wait for the current task to finish and start the research again.')
            return
        self.ctx.cancel_event.clear()
        depth, overrides = self._collect_manual_overrides()
        self.research_worker = _g.DeepResearchWorker(self.ctx, topic, depth=depth, overrides=overrides)
        self.research_worker.progress.connect(self._on_research_progress)
        self.research_worker.done.connect(self._on_research_done)
        self.research_worker.failed.connect(self._on_research_failed)
        self.research_worker.finished.connect(self._on_research_finished)
        self._reset_report_view()
        self.research_bar.setValue(0)
        self.research_bar.show()
        self._research_active = True
        self._log_bar.setText("starting…")
        self.research_status.hide()  # strip is the source of truth during a run
        self._update_progress_strip(self._RESEARCH_PHASES[0], {})
        self._progress_strip.show()
        self._focus_tab("research")
        manual_note = " (manual settings applied)" if self.mc_toggle.isChecked() else ""
        self._add_system(f"🔬 Ultra Search: researching «{topic}»{manual_note} — this can take a few "
                         f"minutes. Progress is in the Research tab; press Stop to abort.")
        self.research_status.setText(f"Starting research: {topic}")
        self.research_worker.start()
        self._set_busy(True)
        self.stage.set_stage("Ultra Search")

    def _on_research_done(self, result: dict):
        report = (result.get("report") or "").strip()
        stats = result.get("stats") or {}      # key can be present and null
        self.research_status.show()  # bring back the top line for the final summary
        if report:
            self._set_report(report)
            cancelled = result.get("cancelled")
            note = " (stopped early — partial)" if cancelled else ""
            # Per-stage timing: show the 3 slowest stages so it's obvious whether
            # the run was network-bound or LLM-bound (and whether parallel I/O helped).
            timings = stats.get("stage_timings") or {}
            slow = sorted(timings.items(), key=lambda kv: kv[1], reverse=True)[:3]
            timing_line = ("\nSlowest stages: " + ", ".join(f"{k} {v:.0f}s" for k, v in slow)) if slow else ""
            self.research_status.setText(
                f"Done{note}: {stats.get('sources', 0)} sources, {stats.get('pages', 0)} pages, "
                f"{stats.get('findings', 0)} findings in {result.get('elapsed_sec', 0):.0f}s.{timing_line}")
            self._add_assistant('Done — the full report is on the Research tab.' +
                                (' The search stopped early; the report is incomplete.' if cancelled else ""))
            self._update_progress_strip(self._RESEARCH_PHASES[-1], stats)  # all pills → done
        else:
            self.research_status.setText("No report produced.")
            self._add_system('Ultra Search: could not gather material on the topic.')
        self.research_bar.hide()

    def _on_research_failed(self, msg: str):
        self.research_status.show()
        self.research_status.setText(f"Research failed: {msg}")
        self.research_bar.hide()
        self._progress_strip.hide()
        self._add_system(f"Ultra Search error: {msg}")

    def _on_research_finished(self):
        self._research_active = False
        self.research_worker = None
        if self.ctx is not None:
            self.ctx.cancel_event.clear()
        self._set_busy(False)
        self._set_status("")
        self.stage.set_stage("Ready")

    def _reset_report_view(self):
        """Clear both report viewers and show the placeholder text view (so a stale
        web report from a previous run doesn't linger when a new run starts)."""
        self.research_view.clear()
        self.research_view.show()
        if getattr(self, "research_web", None) is not None:
            try:
                self.research_web.setHtml("")
            except Exception:
                pass
            self.research_web.hide()

    def _set_report(self, markdown_text: str):
        """Render the report. Prefer the QWebEngine viewer (typeset math + real
        tables); fall back to Qt's setMarkdown, then plain text."""
        import gui as _g
        self._last_report_md = markdown_text
        if getattr(self, "research_web", None) is not None:
            try:
                from PyQt5.QtCore import QUrl
                self.research_web.setHtml(_render_report_html(markdown_text), QUrl("about:blank"))
                self.research_view.hide()
                self.research_web.show()
                return
            except Exception as exc:
                _g.logger.warning("web report render failed, falling back to markdown: %s", exc)
        self.research_web and self.research_web.hide()
        self.research_view.show()
        if hasattr(self.research_view, "setMarkdown"):
            try:
                self.research_view.setMarkdown(markdown_text)
                return
            except Exception:
                pass
        self.research_view.setPlainText(markdown_text)

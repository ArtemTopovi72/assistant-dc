"""The Manual Control panel: the DR_* pipeline knobs, exposed as widgets.

A mixin, not free functions: every method builds or reads THIS window's
widgets. Extracted from gui.py because it is a self-contained control surface
— it touches no worker, no agent and no other tab, and its only outside
contact is the live deep_research module.

That contact is deliberately made at CALL time ("import deep_research as _dr"
inside each method) rather than at module scope: the widgets are seeded from
whatever the module's DR_* globals currently are, so "untouched" always means
"current default behaviour", and a suite that patches deep_research still
intercepts from here.

The mixin needs nothing from the host but "self" to hang widgets on.
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QFrame, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QScrollArea, QSpinBox, QToolButton, QVBoxLayout,
    QWidget,
)

from gui_common import BG, MUTED, _flow
from ui_scale import px, scale_style as _ss


class ManualControlMixin:
    """Builds, resets and reads the Manual Control panel."""

    def _build_manual_control_panel(self):
        """Collapsible 'Manual Control' panel exposing the real DR_* pipeline
        knobs (deep_research.MANUAL_OVERRIDE_SPEC) so a run can be tuned per-topic
        instead of always using the hardcoded depth="standard" defaults. Widgets
        are seeded from the live deep_research module so "untouched" always means
        "current default behavior" — nothing changes unless the user changes it."""
        import deep_research as _dr

        box = QGroupBox()
        outer = QVBoxLayout(box)
        outer.setContentsMargins(6, 4, 6, 6)
        outer.setSpacing(4)

        # A wrapping FlowLayout (not a fixed QHBoxLayout): when the Research panel is
        # narrow the action buttons drop to a new line instead of being clipped
        # ("🔥 TRIPLE-CORE S…", "Reset to def…"). Each button keeps its full sizeHint.
        header = _flow(spacing=px(6))
        self.mc_toggle = QToolButton()
        self.mc_toggle.setText("▸ Manual Control (advanced)")
        self.mc_toggle.setCheckable(True)
        self.mc_toggle.setChecked(False)
        self.mc_toggle.setStyleSheet(_ss(f"color:{MUTED}; font-size:12px; border:none;"))
        header.addWidget(self.mc_toggle)
        self.mc_max_btn = QPushButton("🔥 TRIPLE-CORE SEARCH")
        self.mc_max_btn.setObjectName("rec")
        self.mc_max_btn.setToolTip(
            "Cranks EVERYTHING to the max — all source quotas maxed, every stage on, "
            "strictness set to the strictest.\nPress only in extreme need — or when you "
            "really, really feel like it.")
        self.mc_max_btn.setVisible(False)
        # Never let the text be clipped: pin a minimum width to the natural size hint.
        self.mc_max_btn.setMinimumWidth(self.mc_max_btn.sizeHint().width())
        header.addWidget(self.mc_max_btn)
        self.mc_reset_btn = QPushButton("Reset to defaults")
        self.mc_reset_btn.setObjectName("ghost")
        self.mc_reset_btn.setVisible(False)
        self.mc_reset_btn.setMinimumWidth(self.mc_reset_btn.sizeHint().width())
        header.addWidget(self.mc_reset_btn)
        outer.addLayout(header)

        body = QWidget()
        form = QFormLayout(body)
        # Left margin gives labels room so the first characters never touch (and get
        # clipped by) the panel/scrollbar edge.
        form.setContentsMargins(px(10), 4, px(6), 4)
        form.setSpacing(px(6))

        def _sub(text):
            # Indented, word-wrapping sub-label. Replaces a fragile "  text" leading-
            # space hack whose spaces were being clipped at the panel's left edge
            # (rendered "Rerank backend" as "erank backend").
            lab = QLabel(text)
            lab.setWordWrap(True)
            lab.setStyleSheet(_ss("margin-left:14px;"))
            return lab
        # In a narrow windowed column, keeping label+control on one line squeezes
        # the spin/combo to an unusable sliver. Wrap long rows so the control drops
        # below its label, and let fields grow to the full available width.
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignLeft)

        self.mc_depth = QComboBox()
        self.mc_depth.addItems(["quick", "standard", "deep"])
        self.mc_depth.setCurrentText("standard")
        form.addRow("Depth", self.mc_depth)

        def spin(name, lo, hi):
            s = QSpinBox()
            s.setRange(lo, hi)
            s.setValue(getattr(_dr, name))
            return s

        self.mc_max_queries = spin("DR_MAX_QUERIES", 1, 40)
        form.addRow("Max queries", self.mc_max_queries)
        self.mc_max_pages = spin("DR_MAX_PAGES", 1, 200)
        form.addRow("Max pages crawled", self.mc_max_pages)
        self.mc_results_per_query = spin("DR_RESULTS_PER_QUERY", 1, 30)
        form.addRow("Results per query", self.mc_results_per_query)
        self.mc_max_rounds = spin("DR_MAX_COLLECTION_ROUNDS", 1, 20)
        form.addRow("Max research loops (rounds)", self.mc_max_rounds)
        self.mc_force_exhaustive = QCheckBox("Force exhaustive search (no early stop)")
        self.mc_force_exhaustive.setChecked(_dr.DR_FORCE_EXHAUSTIVE)
        form.addRow(self.mc_force_exhaustive)

        form.addRow(QLabel("— Sources —"))
        self.mc_min_sources = spin("DR_MIN_SOURCES", 0, 500)
        form.addRow("Minimum sources required (0=off)", self.mc_min_sources)
        self.mc_target_sources = spin("DR_TARGET_SOURCES", 0, 500)
        form.addRow("Target sources count (0=off)", self.mc_target_sources)
        self.mc_max_sources = spin("DR_MAX_SOURCES", 0, 1000)
        form.addRow("Maximum sources count (0=unlimited)", self.mc_max_sources)
        self.mc_min_domains = spin("DR_MIN_UNIQUE_DOMAINS", 0, 200)
        form.addRow("Minimum unique domains (0=off)", self.mc_min_domains)
        self.mc_max_per_domain = spin("DR_MAX_PAGES_PER_DOMAIN", 0, 100)
        form.addRow("Max pages per domain (0=unlimited)", self.mc_max_per_domain)
        self.mc_domain_whitelist = QLineEdit(_dr.DR_DOMAIN_WHITELIST)
        self.mc_domain_whitelist.setPlaceholderText("comma-separated, e.g. arxiv.org, nature.com")
        form.addRow("Domain whitelist", self.mc_domain_whitelist)
        self.mc_domain_blacklist = QLineEdit(_dr.DR_DOMAIN_BLACKLIST)
        self.mc_domain_blacklist.setPlaceholderText("comma-separated, e.g. pinterest.com")
        form.addRow("Domain blacklist", self.mc_domain_blacklist)

        self.mc_src_types = {}
        types_row = QHBoxLayout()
        for label, key in (("Academic", "academic"), ("News", "news"), ("Government", "government"),
                          ("Tech docs", "technical"), ("Forums", "forum"), ("Blogs", "blog"),
                          ("Social", "social"), ("Whitepapers", "whitepaper")):
            cb = QCheckBox(label)
            cb.setChecked(key not in {t.strip() for t in _dr.DR_SOURCE_TYPES_EXCLUDE.split(",") if t.strip()})
            self.mc_src_types[key] = cb
            types_row.addWidget(cb)
        types_widget = QWidget()
        types_widget.setLayout(types_row)
        form.addRow("Include source types", types_widget)

        form.addRow(QLabel("— Retry & resilience —"))
        self.mc_query_mutation = spin("DR_QUERY_MUTATION_ATTEMPTS", 0, 3)
        form.addRow("Query mutation attempts on zero hits", self.mc_query_mutation)

        form.addRow(QLabel("— Speed (parallel I/O; 1 = serial) —"))
        self.mc_search_concurrency = spin("DR_SEARCH_CONCURRENCY", 1, 32)
        self.mc_search_concurrency.setToolTip(
            "Run this many search queries at once. Independent network calls, so this "
            "cuts wall-clock time; kept bounded to avoid search-engine rate-limit bans.")
        form.addRow("Parallel search queries", self.mc_search_concurrency)
        self.mc_fetch_concurrency = spin("DR_FETCH_CONCURRENCY", 1, 32)
        self.mc_fetch_concurrency.setToolTip(
            "Fetch this many pages at once (never two from the same domain in parallel — "
            "per-domain politeness is preserved). LLM calls stay serial regardless.")
        form.addRow("Parallel page fetches", self.mc_fetch_concurrency)

        form.addRow(QLabel("— Quality & verification —"))
        self.mc_verification_mode = QComboBox()
        self.mc_verification_mode.addItems(
            ["Relaxed", "Standard", "Strict", "Scientific", "Extreme Verification"])
        self.mc_verification_mode.setCurrentText("Standard")
        self.mc_verification_mode.currentTextChanged.connect(self._apply_verification_mode)
        form.addRow("Verification mode", self.mc_verification_mode)

        form.addRow(QLabel("— Research profile —"))
        self.mc_profile = QComboBox()
        self.mc_profile.addItems(["(none)"] + list(self._RESEARCH_PROFILES.keys()))
        self.mc_profile.currentTextChanged.connect(self._apply_research_profile)
        form.addRow("Preset", self.mc_profile)

        self.mc_multihop = QCheckBox("Multi-hop entity expansion")
        self.mc_multihop.setChecked(_dr.DR_MULTIHOP_ENABLED)
        form.addRow(self.mc_multihop)
        self.mc_contradiction = QCheckBox("Contradiction search")
        self.mc_contradiction.setChecked(_dr.DR_CONTRADICTION_ENABLED)
        form.addRow(self.mc_contradiction)
        self.mc_reflection = QCheckBox("Active reflection loop")
        self.mc_reflection.setChecked(_dr.DR_REFLECTION_ENABLED)
        form.addRow(self.mc_reflection)
        self.mc_reflection_iters = spin("DR_REFLECTION_MAX_ITERATIONS", 0, 10)
        form.addRow(_sub("Reflection max iterations"), self.mc_reflection_iters)
        self.mc_reflection_budget = spin("DR_REFLECTION_RESEARCH_BUDGET", 0, 50)
        form.addRow(_sub("Reflection query budget"), self.mc_reflection_budget)
        self.mc_hierarchical = QCheckBox("Hierarchical clustering")
        self.mc_hierarchical.setChecked(_dr.DR_HIERARCHICAL_ENABLED)
        form.addRow(self.mc_hierarchical)
        self.mc_communities = QCheckBox("Graph communities (Louvain)")
        self.mc_communities.setChecked(_dr.DR_COMMUNITIES_ENABLED)
        form.addRow(self.mc_communities)
        self.mc_graph_expansion = QCheckBox("Graph-guided deep-dive expansion")
        self.mc_graph_expansion.setChecked(_dr.DR_GRAPH_EXPANSION_ENABLED)
        form.addRow(self.mc_graph_expansion)
        self.mc_claim_merge = QCheckBox("Claim-level merge/dedup")
        self.mc_claim_merge.setChecked(_dr.DR_CLAIM_MERGE_ENABLED)
        form.addRow(self.mc_claim_merge)
        self.mc_citations = QCheckBox("Citation lineage (OpenAlex)")
        self.mc_citations.setChecked(_dr.DR_CITATIONS_ENABLED)
        form.addRow(self.mc_citations)
        self.mc_pdf = QCheckBox("PDF extraction (+ ar5iv for arXiv)")
        self.mc_pdf.setChecked(_dr.DR_PDF_ENABLED)
        form.addRow(self.mc_pdf)
        self.mc_decision_gate = QCheckBox("Decision gate (ANSWER/ASK/ABSTAIN)")
        self.mc_decision_gate.setChecked(_dr.DR_DECISION_GATE_ENABLED)
        form.addRow(self.mc_decision_gate)
        self.mc_replan = QCheckBox("Corrective re-plan on weak coverage")
        self.mc_replan.setChecked(_dr.DR_REPLAN_ENABLED)
        form.addRow(self.mc_replan)

        self.mc_rerank_enabled = QCheckBox("Second-stage reranking")
        self.mc_rerank_enabled.setChecked(_dr.DR_RERANK_ENABLED)
        form.addRow(self.mc_rerank_enabled)
        self.mc_rerank_backend = QComboBox()
        self.mc_rerank_backend.addItems(["auto", "cross", "dense", "lexical"])
        self.mc_rerank_backend.setCurrentText(_dr.DR_RERANK_BACKEND)
        form.addRow(_sub("Rerank backend"), self.mc_rerank_backend)
        self.mc_rerank_top_k = spin("DR_RERANK_TOP_K", 0, 60)
        self.mc_rerank_top_k.setSpecialValueText("all")   # 0 shows as "all"
        self.mc_rerank_top_k.setToolTip(
            "How many crawled pages to keep for briefing after relevance reranking. "
            "0 = 'all' (just reorder, drop nothing — recommended). A positive N caps "
            "briefing to the N most relevant pages.")
        form.addRow(_sub("Rerank top-K (0=all)"), self.mc_rerank_top_k)

        form.addRow(QLabel("— Report & reasoning depth —"))
        self.mc_reasoning_effort = QComboBox()
        self.mc_reasoning_effort.addItems(["auto", "low", "medium", "high"])
        self.mc_reasoning_effort.setCurrentText(getattr(_dr, "DR_REASONING_EFFORT", "auto"))
        self.mc_reasoning_effort.setToolTip(
            "Depth of analysis for the final report (gpt-oss reasoning effort). "
            "'auto' uses the model default from Settings; higher = the model thinks "
            "harder before writing.")
        form.addRow("Depth of analysis", self.mc_reasoning_effort)
        # NOTE: no token-budget boxes. All output budgets (report, per-section,
        # briefs, consolidation) now derive automatically from the model's real
        # context window — the model writes as much as the evidence supports and
        # stops on its own. There is nothing useful to cap here by hand.

        self.mc_survey_mode = QCheckBox("Survey mode (long-form scientific document)")
        self.mc_survey_mode.setChecked(bool(getattr(_dr, "DR_SURVEY_MODE", 1)))
        self.mc_survey_mode.setToolTip(
            "ON: the model first DESIGNS a document outline from the evidence, then "
            "writes each section in its own call as flowing, source-merged prose — a "
            "survey paper / monograph that can reach tens of pages. OFF: legacy "
            "single-pass fixed-template report.")
        form.addRow("", self.mc_survey_mode)
        self.mc_survey_max_sections = QSpinBox()
        self.mc_survey_max_sections.setRange(3, 24)
        self.mc_survey_max_sections.setValue(getattr(_dr, "DR_SURVEY_MAX_SECTIONS", 14))
        self.mc_survey_max_sections.setToolTip(
            "Maximum number of top-level sections the dynamic outline may contain.")
        form.addRow(_sub("Max sections"), self.mc_survey_max_sections)

        # The form is tall (40+ rows) and used to overflow the tab. Put it in a
        # scroll area with a capped height so the panel never pushes the report
        # view off-screen and every control stays reachable by scrolling.
        # Keep the form's background dark like the rest of the panel — once wrapped
        # in a QScrollArea the inner viewport can otherwise fall back to the default
        # light palette, washing out the near-white labels.
        body.setStyleSheet(_ss(f"background:{BG};"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.viewport().setStyleSheet(_ss(f"background:{BG};"))
        scroll.setWidget(body)
        scroll.setVisible(False)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMaximumHeight(px(440))
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

        def _toggle_body(checked):
            scroll.setVisible(checked)
            self.mc_reset_btn.setVisible(checked)
            self.mc_max_btn.setVisible(checked)
            self.mc_toggle.setText(("▾" if checked else "▸") + " Manual Control (advanced)")
        self.mc_toggle.toggled.connect(_toggle_body)
        self.mc_reset_btn.clicked.connect(self._reset_manual_control)
        self.mc_max_btn.clicked.connect(self._maximize_manual_control)
        return box

    def _reset_manual_control(self):
        import deep_research as _dr
        self.mc_depth.setCurrentText("standard")
        self.mc_max_queries.setValue(_dr.DR_MAX_QUERIES)
        self.mc_max_pages.setValue(_dr.DR_MAX_PAGES)
        self.mc_results_per_query.setValue(_dr.DR_RESULTS_PER_QUERY)
        self.mc_max_rounds.setValue(_dr.DR_MAX_COLLECTION_ROUNDS)
        self.mc_force_exhaustive.setChecked(_dr.DR_FORCE_EXHAUSTIVE)
        self.mc_min_sources.setValue(_dr.DR_MIN_SOURCES)
        self.mc_target_sources.setValue(_dr.DR_TARGET_SOURCES)
        self.mc_max_sources.setValue(_dr.DR_MAX_SOURCES)
        self.mc_min_domains.setValue(_dr.DR_MIN_UNIQUE_DOMAINS)
        self.mc_max_per_domain.setValue(_dr.DR_MAX_PAGES_PER_DOMAIN)
        self.mc_domain_whitelist.setText(_dr.DR_DOMAIN_WHITELIST)
        self.mc_domain_blacklist.setText(_dr.DR_DOMAIN_BLACKLIST)
        excluded = {t.strip() for t in _dr.DR_SOURCE_TYPES_EXCLUDE.split(",") if t.strip()}
        for key, cb in self.mc_src_types.items():
            cb.setChecked(key not in excluded)
        self.mc_query_mutation.setValue(_dr.DR_QUERY_MUTATION_ATTEMPTS)
        self.mc_search_concurrency.setValue(_dr.DR_SEARCH_CONCURRENCY)
        self.mc_fetch_concurrency.setValue(_dr.DR_FETCH_CONCURRENCY)
        self.mc_verification_mode.setCurrentText("Standard")
        self.mc_profile.setCurrentText("(none)")
        self.mc_multihop.setChecked(_dr.DR_MULTIHOP_ENABLED)
        self.mc_contradiction.setChecked(_dr.DR_CONTRADICTION_ENABLED)
        self.mc_reflection.setChecked(_dr.DR_REFLECTION_ENABLED)
        self.mc_reflection_iters.setValue(_dr.DR_REFLECTION_MAX_ITERATIONS)
        self.mc_reflection_budget.setValue(_dr.DR_REFLECTION_RESEARCH_BUDGET)
        self.mc_hierarchical.setChecked(_dr.DR_HIERARCHICAL_ENABLED)
        self.mc_communities.setChecked(_dr.DR_COMMUNITIES_ENABLED)
        self.mc_graph_expansion.setChecked(_dr.DR_GRAPH_EXPANSION_ENABLED)
        self.mc_claim_merge.setChecked(_dr.DR_CLAIM_MERGE_ENABLED)
        self.mc_citations.setChecked(_dr.DR_CITATIONS_ENABLED)
        self.mc_pdf.setChecked(_dr.DR_PDF_ENABLED)
        self.mc_decision_gate.setChecked(_dr.DR_DECISION_GATE_ENABLED)
        self.mc_replan.setChecked(_dr.DR_REPLAN_ENABLED)
        self.mc_rerank_enabled.setChecked(_dr.DR_RERANK_ENABLED)
        self.mc_rerank_backend.setCurrentText(_dr.DR_RERANK_BACKEND)
        self.mc_rerank_top_k.setValue(_dr.DR_RERANK_TOP_K)
        self.mc_reasoning_effort.setCurrentText(getattr(_dr, "DR_REASONING_EFFORT", "auto"))
        self.mc_survey_mode.setChecked(bool(getattr(_dr, "DR_SURVEY_MODE", 1)))
        self.mc_survey_max_sections.setValue(getattr(_dr, "DR_SURVEY_MAX_SECTIONS", 14))

    def _maximize_manual_control(self):
        """TRIPLE-CORE SEARCH: crank every knob to its maximum and every dropdown
        to its strictest setting. Each widget is clamped to its own max() so this
        stays correct if a range ever changes. This is a deliberately heavy run —
        expect it to take a long time and hammer all source quotas."""
        def _maxspin(s):
            s.setValue(s.maximum())
        self.mc_depth.setCurrentText("deep")
        for s in (self.mc_max_queries, self.mc_max_pages, self.mc_results_per_query,
                  self.mc_max_rounds, self.mc_query_mutation, self.mc_reflection_iters,
                  self.mc_reflection_budget, self.mc_rerank_top_k):
            _maxspin(s)
        self.mc_force_exhaustive.setChecked(True)
        # Parallel I/O: aggressive but NOT maxed — 8 concurrent ddgs queries invites
        # rate-limit bans (which would slow the run, not speed it). 4 is the sweet spot.
        self.mc_search_concurrency.setValue(min(4, self.mc_search_concurrency.maximum()))
        self.mc_fetch_concurrency.setValue(min(4, self.mc_fetch_concurrency.maximum()))
        # Source quotas: demand a lot, cap nothing.
        self.mc_min_sources.setValue(100)
        self.mc_target_sources.setValue(200)
        self.mc_max_sources.setValue(self.mc_max_sources.maximum())
        self.mc_min_domains.setValue(50)
        self.mc_max_per_domain.setValue(0)          # 0 = unlimited per domain
        self.mc_domain_whitelist.clear()            # don't restrict where we look
        # NB: deliberately KEEP the blacklist — "max everything" maxes quotas, it
        # must NOT silently drop the user's banned domains (safety, not a quota).
        for cb in self.mc_src_types.values():        # include every source type
            cb.setChecked(True)
        # Strictest dropdowns + every analysis stage on.
        self.mc_verification_mode.setCurrentText("Extreme Verification")
        for cb in (self.mc_multihop, self.mc_contradiction, self.mc_reflection,
                   self.mc_hierarchical, self.mc_communities, self.mc_graph_expansion,
                   self.mc_claim_merge, self.mc_citations, self.mc_pdf,
                   self.mc_decision_gate, self.mc_replan, self.mc_rerank_enabled):
            cb.setChecked(True)
        self.mc_rerank_backend.setCurrentText("cross")   # strongest reranker
        # Deepest analysis. Output budgets are auto (full context window) — no token
        # boxes to max anymore; the model writes as long as the evidence supports.
        self.mc_reasoning_effort.setCurrentText("high")
        self.mc_survey_mode.setChecked(True)
        self.mc_survey_max_sections.setValue(self.mc_survey_max_sections.maximum())
        self.research_status.setText(
            "🔥 TRIPLE-CORE SEARCH armed — everything maxed, strictness at max. "
            "This run will be long and thorough. Send a topic to launch.")

    # Mode -> (DR_GATE_MIN_STRONG, DR_GATE_MIN_CLUSTERS, contradiction_enabled,
    #          claim_merge_threshold, min_unique_domains_floor). Stricter modes
    # demand more independent confirmation before the gate will ANSWER.
    _VERIFICATION_MODES = {
        "Relaxed":              (1, 1, False, 0.70, 0),
        "Standard":             (1, 2, True,  0.55, 0),
        "Strict":               (2, 2, True,  0.45, 3),
        "Scientific":           (2, 3, True,  0.40, 5),
        "Extreme Verification": (3, 4, True,  0.30, 8),
    }

    # Each profile sets (depth, min_sources, source_types_to_exclude, verification_mode).
    _RESEARCH_PROFILES = {
        "General Research":             ("standard", 0,  "", "Standard"),
        "News Investigation":           ("standard", 10, "academic,whitepaper", "Standard"),
        "Scientific Research":          ("deep",     15, "social,forum,blog", "Scientific"),
        "Technical Research":           ("standard", 10, "social", "Strict"),
        "Market Research":              ("standard", 20, "", "Standard"),
        "Competitive Intelligence":     ("standard", 20, "academic", "Standard"),
        "Legal Research":               ("deep",     10, "social,forum,blog", "Strict"),
        "Medical Research":             ("deep",     15, "social,forum,blog", "Extreme Verification"),
        "Historical Research":          ("deep",     10, "social", "Strict"),
        "Academic Literature Review":   ("deep",     20, "social,forum,blog,news", "Scientific"),
    }

    def _apply_verification_mode(self, mode_name: str):
        """Quality/Verification preset: maps to real decision-gate + contradiction
        + claim-merge knobs (not exposed as separate widgets — the mode IS the
        control). DR_GATE_MIN_STRONG/MIN_CLUSTERS aren't in MANUAL_OVERRIDE_SPEC
        yet only via this combo, applied at collection time below."""
        self._verification_settings = self._VERIFICATION_MODES.get(mode_name)
        if self._verification_settings and self._verification_settings[4] > 0:
            self.mc_min_domains.setValue(max(self.mc_min_domains.value(),
                                             self._verification_settings[4]))
        if self._verification_settings:
            self.mc_contradiction.setChecked(self._verification_settings[2])

    def _apply_research_profile(self, name: str):
        if name not in self._RESEARCH_PROFILES:
            return
        depth, min_sources, exclude_types, vmode = self._RESEARCH_PROFILES[name]
        self.mc_depth.setCurrentText(depth)
        self.mc_min_sources.setValue(min_sources)
        excluded = {t.strip() for t in exclude_types.split(",") if t.strip()}
        for key, cb in self.mc_src_types.items():
            cb.setChecked(key not in excluded)
        self.mc_verification_mode.setCurrentText(vmode)

    def _collect_manual_overrides(self) -> dict:
        """Read the panel into (depth, overrides) for DeepResearchWorker."""
        vmode = getattr(self, "_verification_settings", None) or \
            self._VERIFICATION_MODES[self.mc_verification_mode.currentText()]
        min_strong, min_clusters, contra_on, merge_thresh, _ = vmode
        excluded_types = ",".join(k for k, cb in self.mc_src_types.items() if not cb.isChecked())
        return self.mc_depth.currentText(), {
            "DR_MAX_QUERIES": self.mc_max_queries.value(),
            "DR_MAX_PAGES": self.mc_max_pages.value(),
            "DR_RESULTS_PER_QUERY": self.mc_results_per_query.value(),
            "DR_MAX_COLLECTION_ROUNDS": self.mc_max_rounds.value(),
            "DR_FORCE_EXHAUSTIVE": self.mc_force_exhaustive.isChecked(),
            "DR_MIN_SOURCES": self.mc_min_sources.value(),
            "DR_TARGET_SOURCES": self.mc_target_sources.value(),
            "DR_MAX_SOURCES": self.mc_max_sources.value(),
            "DR_MIN_UNIQUE_DOMAINS": self.mc_min_domains.value(),
            "DR_MAX_PAGES_PER_DOMAIN": self.mc_max_per_domain.value(),
            "DR_DOMAIN_WHITELIST": self.mc_domain_whitelist.text().strip(),
            "DR_DOMAIN_BLACKLIST": self.mc_domain_blacklist.text().strip(),
            "DR_SOURCE_TYPES_EXCLUDE": excluded_types,
            "DR_QUERY_MUTATION_ATTEMPTS": self.mc_query_mutation.value(),
            "DR_SEARCH_CONCURRENCY": self.mc_search_concurrency.value(),
            "DR_FETCH_CONCURRENCY": self.mc_fetch_concurrency.value(),
            "DR_GATE_MIN_STRONG": min_strong,
            "DR_GATE_MIN_CLUSTERS": min_clusters,
            "DR_CLAIM_MERGE_THRESHOLD": merge_thresh,
            "DR_MULTIHOP_ENABLED": self.mc_multihop.isChecked(),
            "DR_CONTRADICTION_ENABLED": self.mc_contradiction.isChecked(),
            "DR_REFLECTION_ENABLED": self.mc_reflection.isChecked(),
            "DR_REFLECTION_MAX_ITERATIONS": self.mc_reflection_iters.value(),
            "DR_REFLECTION_RESEARCH_BUDGET": self.mc_reflection_budget.value(),
            "DR_HIERARCHICAL_ENABLED": self.mc_hierarchical.isChecked(),
            "DR_COMMUNITIES_ENABLED": self.mc_communities.isChecked(),
            "DR_GRAPH_EXPANSION_ENABLED": self.mc_graph_expansion.isChecked(),
            "DR_CLAIM_MERGE_ENABLED": self.mc_claim_merge.isChecked(),
            "DR_CITATIONS_ENABLED": self.mc_citations.isChecked(),
            "DR_PDF_ENABLED": self.mc_pdf.isChecked(),
            "DR_DECISION_GATE_ENABLED": self.mc_decision_gate.isChecked(),
            "DR_REPLAN_ENABLED": self.mc_replan.isChecked(),
            "DR_RERANK_ENABLED": self.mc_rerank_enabled.isChecked(),
            "DR_RERANK_BACKEND": self.mc_rerank_backend.currentText(),
            "DR_RERANK_TOP_K": self.mc_rerank_top_k.value(),
            "DR_REASONING_EFFORT": self.mc_reasoning_effort.currentText(),
            "DR_SURVEY_MODE": self.mc_survey_mode.isChecked(),
            "DR_SURVEY_MAX_SECTIONS": self.mc_survey_max_sections.value(),
        }

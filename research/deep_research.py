"""Ultra Search / deep research engine.

A separate, aggressive execution path (NOT the normal one-shot `search` tool):

    plan_queries -> collect_sources -> crawl_pages -> dedupe
                 -> per-source briefs (map) -> synthesis (reduce) -> report

It is deliberately bounded by the DR_* caps in config so it cannot run away, it
is fault-tolerant (a single bad source never aborts the run), it is cooperatively
cancellable via ``ctx.cancel_event``, and it persists its state + final report to
disk so a run survives across sessions.

Reports are returned as Markdown. In the voice assistant the full report is shown
in the GUI Research tab / written to disk; the spoken reply is just a short
summary (the model never reads a multi-page report aloud).
"""
from dr_settings import view as settings_view
import json
import logging
import re
import sys
import time
import types
from typing import Callable, Optional

import dr_settings as S
from seams import dep
from prompts import PRACTICAL_GUIDE_PROMPT, SELECTION_REPORT_PROMPT, SELECTION_VERIFY_PROMPT
from dr_settings import MANUAL_OVERRIDE_SPEC, _coerce_override
from research_cache import ExtractionCache
from citation_graph import citation_lineage_brief
from dr_urls import (
    _domain, _is_landing_url, _norm_url, _is_junk_source, _is_community_host,
    _authority_score, _source_rank, _classify_source_type, _domain_allowed,
    _is_safe_public_url, _source_type_allowed, _mutate_query, _cap_per_domain,
    _AUTH_BASE_SCORE, _AUTH_CATEGORY_SCORE, _MID_SCORE, _COMMUNITY_SCORE,
)
# Re-exported unchanged so `deep_research._classify_source_type` keeps resolving
# on this module after the move to dr_urls (callers and tests reach for it here).
_REEXPORTED_FROM_DR_URLS = (_classify_source_type, _COMMUNITY_SCORE)
from dr_lang import _DOC_LABELS, _norm_out_lang, lang_of_text, _lab, _lang_directive
# Re-exported: gui_workers calls `dr.lang_of_text(topic)` to pick the report
# language, and the output-language suite reads _DOC_LABELS off this module.
_REEXPORTED_FROM_DR_LANG = (_DOC_LABELS, lang_of_text)
from dr_timing import (
    DEPTHS, _TIMING_SEED, _TIMING_KEEP, _profile_signature,
    estimate_duration, record_run_duration,
)
# Re-exported: tg_bot quotes `_dr.estimate_duration(depth)`, and the depth
# suites read DEPTHS/_TIMING_SEED/_TIMING_KEEP off this module. All of these are
# constants or stable functions.
#
# `_TIMING_FILE` is deliberately NOT re-exported. It is the one MUTABLE seam
# here — suites repoint it at a temp dir so they do not write the live timings
# file — and a re-export would give it two homes: a suite patching
# deep_research._TIMING_FILE would leave dr_timing's copy in force and silently
# poison the user's real ETA history while still printing PASS. Patch
# dr_timing._TIMING_FILE; anything still reaching for it here gets a loud
# AttributeError.
_REEXPORTED_FROM_DR_TIMING = (DEPTHS, _TIMING_SEED, _TIMING_KEEP, _profile_signature,
                              estimate_duration, record_run_duration)
from dr_math import _repair_latex_artifacts, audit_math
# Re-exported: the pure suite and the live math probes (tests/math_live.py,
# tests/math_stage_trace.py) call `dr.audit_math` / `DR._repair_latex_artifacts`.
# Pure functions and constants; nothing here is patched, so no split state.
_REEXPORTED_FROM_DR_MATH = (_repair_latex_artifacts, audit_math)
from dr_outline import (
    _clean_report_title, _extract_outline_json, _default_outline, _repair_outline_text,
    _outline_overview, _toc, _est_tokens,
)
# Re-exported: the pure/synth/survey/quality suites read every one of these
# as `deep_research.X`. All pure functions, none of them patched anywhere,
# so there is no second copy of any mutable state here.
_REEXPORTED_FROM_DR_OUTLINE = (_clean_report_title, _extract_outline_json,
                               _default_outline, _toc, _est_tokens)
from dr_extract import (
    GATE_OK, GATE_EMPTY, GATE_THIN, GATE_CHALLENGE, GATE_SHELL, GATE_NAV,
    classify_extraction, _ar5iv_url, _inline_mathml_as_tex,
    _looks_like_unmarked_math, _fix_mojibake, _extract_text, _extract_title,
    _extract_links, _social_outlinks, _content_similarity, dedupe_pages,
    _shingles, _jaccard, _SHINGLE_CAP,
)
# Re-exported so the GATE_* verdicts and the shingle helpers keep resolving as
# `deep_research.X` after the move to dr_extract (callers and tests read them
# there).
_REEXPORTED_FROM_DR_EXTRACT = (GATE_EMPTY, GATE_THIN, GATE_CHALLENGE, GATE_SHELL,
                               _content_similarity, _shingles, _jaccard,
                               _SHINGLE_CAP)
import entities as _entities
import contradiction as _contra
import research_graph as _rgraph
import decision_gate as _gate
import clustering as _cluster
import graph_guided as _gguided
import claim_merge as _cmerge
import hierarchical as _hier
import hierarchy as _hierarchy
import communities as _communities
import formula_audit as _formula_audit
from dr_state import _slug, _save_state, _now_iso, _write_report
# Re-exported: the synth suite drives _save_state/_write_report/_now_iso and the
# pure suite _slug through this module.
_REEXPORTED_FROM_DR_STATE = (_slug, _save_state, _write_report, _now_iso)
from dr_calls import _think_call, _model_context
# Re-exported: the survey/classification suites patch `deep_research.call_llm_simple`
# — which is dr_calls' import, so they must patch it THERE. `_think_call` itself is
# read off this module by those suites, so it stays reachable here.
_REEXPORTED_FROM_DR_CALLS = (_think_call, _model_context)
from dr_progress import _Progress
from dr_policy import (
    _is_practical_request, _is_selection_request, _apply_scholarly_floor, _apply_trust_floor, _resolve_caps,
    _SCHOLARLY_CATEGORIES, _PRACTICAL_CATEGORIES, _SCHOLARLY_FLOOR_KEEP,
)
# Re-exported: dr_timing imports `_resolve_caps` from here, and the quality /
# classification / depth suites read the category sets and the trust tables as
# `deep_research.X`. All pure functions and constants — no mutable state, so no
# second copy (the DR_* knobs _resolve_caps reads live in dr_settings).
_REEXPORTED_FROM_DR_POLICY = (_is_practical_request, _apply_trust_floor, _resolve_caps,
                              _SCHOLARLY_CATEGORIES, _PRACTICAL_CATEGORIES)
# Re-exported: the fetch suite drives `_Progress` through this module.
_REEXPORTED_FROM_DR_PROGRESS = (_Progress,)
from dr_relevance import (
    _entity_terms, _term_in, _term_in_any, _relevance_hits, _min_hits, _brief_is_relevant,
    _filter_offtopic_briefs,
)
_REEXPORTED_FROM_DR_RELEVANCE = (_entity_terms, _brief_is_relevant,
                                 _filter_offtopic_briefs)
from dr_collect import (
    plan_queries, selection_queries, selection_followups, collect_sources, coverage_signals, evidence_stats,
    diversify_queries, _corrective_queries, _search_one_query, _citation_query,
    _topic_first_line, _query_lang,
)
# Re-exported: the search/quality suites drive plan_queries/collect_sources and
# read the helpers as `deep_research.X`.
#
# The MUTABLE seam here is `raw_search_results`, which dr_collect binds — a suite
# faking the search backend must patch `dr_collect.raw_search_results`. It is
# deliberately NOT re-exported, so a stale patch on this module fails loudly
# instead of quietly issuing real ddgs queries.
_REEXPORTED_FROM_DR_COLLECT = (plan_queries, collect_sources, coverage_signals,
                               evidence_stats, diversify_queries)
from dr_crawl import (
    crawl_pages, rerank_for_briefing, _fetch, _fetch_ar5iv, _fetch_page,
    _acquire_page, _extract_pdf_text, _MAX_HTML_CHARS,
)
# Re-exported: the fetch/quality suites drive every one of these through this
# module. `requests` (the network seam) is dr_crawl's own import and is patched
# there.
_REEXPORTED_FROM_DR_CRAWL = (crawl_pages, rerank_for_briefing, _fetch_page,
                             _fetch_ar5iv, _extract_pdf_text)
from dr_brief import brief_source, _parse_trust, _pack_batches, _brief_tag, _consolidate_evidence
# Re-exported: the quality suite reads brief_source/_parse_trust here. The brief
# model contact (`call_llm_simple`) is bound in dr_brief and patched THERE.
_REEXPORTED_FROM_DR_BRIEF = (brief_source, _parse_trust, _consolidate_evidence)
from dr_synthesis import (
    synthesize_report, synthesize_survey, plan_report_outline, requested_sections,
    _normalize_outline, _evidence_landscape, _select_section_evidence, _section_terms,
    _synthesize_abstract, _postprocess_report, _confidence_rule,
)
# Re-exported: the synth/survey/quality suites read all of these as
# `deep_research.X`.
_REEXPORTED_FROM_DR_SYNTHESIS = (synthesize_report, synthesize_survey,
                                 plan_report_outline, requested_sections,
                                 _normalize_outline, _postprocess_report)
from dr_assemble import (
    _report_header, _sources_appendix, _contradiction_note, _append_once,
    provenance_footer, audit_report_links, demote_failing_rows, promote_passing_rows,
    selection_form_missing, drop_empty_columns, drop_false_none_claim, rank_by_closeness, mono_color_is_color, strip_empty_fields,
    _citation_section_from_brief, _parse_cite_entry, _deterministic_digest_report,
    _build_equations_section,
)
# Re-exported: the pure/quality suites read the header/appendix/citation helpers
# as `deep_research.X`.
_REEXPORTED_FROM_DR_ASSEMBLE = (_report_header, _sources_appendix,
                                provenance_footer,
                                _citation_section_from_brief,
                                _deterministic_digest_report)

logger = logging.getLogger("assistant.research")

# --------------------------------------------------------------------------- #
# Runtime knobs — the values live in `dr_settings` (one home, see its docstring)
# and this module reads them as `S.DR_X`. The historical patch surface, however,
# is `deep_research.DR_X`: the GUI's Manual Control panel seeds its widgets from
# it and the suites patch it directly. The proxy below makes that name a live
# VIEW on dr_settings rather than a stale copy — a read forwards there, and a
# write (`deep_research.DR_MAX_SOURCES = 3`) lands there, so every pipeline
# module sees it. Module-level assignments inside THIS file are unaffected
# (STORE_NAME writes __dict__ directly and never reaches __setattr__), and an
# unknown name still raises AttributeError exactly as before.
# --------------------------------------------------------------------------- #


class _SettingsView(types.ModuleType):
    def __getattr__(self, name):
        if name in S.SETTING_NAMES:
            return getattr(S, name)
        raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")

    def __setattr__(self, name, value):
        if name in S.SETTING_NAMES:
            setattr(S, name, value)
        else:
            super().__setattr__(name, value)


sys.modules[__name__].__class__ = _SettingsView


def _stage(ctx, name: str):
    """The pipeline stage `name`: the real one, or the run's own (`ctx.deps`)."""
    return dep(ctx, name, globals()[name])


def no_briefs_report(topic: str, pages, silent: int) -> str:
    """What to tell the reader when the run produced no briefs at all.

    There are two different failures here.
    Measured live with the model unloaded mid-run: every briefing call came
    back empty and the report said eight successfully fetched pages "contained
    no facts relevant to the topic" -- blaming the sources for our own outage.
    A reader takes that to mean the subject is not covered on the web.

    One message says try a different topic; the other says try again later.
    """
    if pages and silent >= len(pages):
        return (f"# Research: {topic}\n\n"
                f"No report was written: the briefing model did not answer for "
                f"any of the {silent} pages that were crawled. The sources were "
                f"fetched successfully — this is a failure on our side, not an "
                f"absence of material.")
    return (f"# Research: {topic}\n\nPages were crawled but none "
            f"contained facts relevant to the topic.")


def _brief_pages(ctx, topic, pages, profile, run_dir, prog, *, scholarly, terms,
                 mode="confirm", cancelled=lambda: False):
    """Map a list of crawled pages to source briefs (shared by the main, multi-hop,
    and contradiction passes). Returns (briefs, dropped_irrelevant). `mode` tags
    each brief as confirmation or contradiction evidence (kept separate downstream)."""
    category = profile.get("category", "general")
    briefs, dropped = [], 0
    # A page the briefer never answered about is NOT a page without facts. Counted
    # separately so the caller can tell the two apart -- see the zero-brief branch.
    silent = 0
    sel = profile.get("selection_analysis") or {}
    # «4096x2160» is one word to the passage ranker, which matches the page's «4096 x 2160 px» by word.
    focus = re.sub(r"(\d)\s*[x×*]\s*(\d)", r"\1 \2", " ".join(
        str(v) for k in ("near_formats", "forced_interfaces") for v in sel.get(k) or []))
    for i, page in enumerate(pages, 1):
        if cancelled():
            break
        prog.update("Verifying", f"[{mode} {i}/{len(pages)}] {page['domain']}")
        brief = _stage(ctx, "brief_source")(ctx, topic, page,
                                            **({"select": True, "focus": focus} if profile.get("select") else {}))
        if not brief:
            # "" means the model produced nothing (our failure); None means it
            # read the page and judged it useless (the page's). Counting both as
            # silent would let a run of genuinely empty pages be reported as an
            # outage.
            if brief == "":
                silent += 1
            continue
        trust, body = _parse_trust(brief)
        if scholarly and not _brief_is_relevant(page.get("title", ""), body, terms):
            dropped += 1
            continue
        trust = _apply_trust_floor(trust, page["url"], category)
        briefs.append({
            "domain": page["domain"], "url": page["url"],
            "title": page.get("title", ""), "brief": body, "trust": trust,
            "mode": mode,
            "why": f"trust={trust} · mode={mode} · rerank={page.get('_rerank_relevance')}",
            "authority": _authority_score(page["url"], category),
            "cluster_size": page.get("cluster_size", 1),
            "cluster_domains": page.get("cluster_domains", [page["domain"]]),
            # Preserve display equations VERBATIM from the source text — the
            # briefing LLM reliably paraphrases/drops them (proven: LOST AT
            # BRIEFING in the stage trace), so we keep the raw blocks separately
            # and surface them in the report regardless of what the model wrote.
            # Prefer equations extracted from the FULL document at collection time
            # (page["equations"]); the page["text"] here is truncated to
            # DR_PAGE_CHARS and usually precedes a paper's key display equations.
            "equations": page.get("equations")
            or _formula_audit.extract_equations(page.get("text", "")),
        })
    if silent:
        logger.warning("%s: the briefer returned nothing for %d of %d pages",
                       mode, silent, len(pages))
        prog.bump(silent_briefs=silent)
    return briefs, dropped


def _document_title(ctx, topic: str, out_lang: str) -> Optional[str]:
    """A noun-phrase title in the document's language. The user's own words headed a
    Russian report as «Подбери цифровую видеокамеру…» — an order, not a title; the reader
    asked for «Подбор цифровой видеокамеры…». None (the cleaned request) on any failure."""
    if _norm_out_lang(out_lang) == "en" or not (topic or "").strip():
        return None
    try:
        out = dep(ctx, "_think_call", _think_call)(
            ctx, "Turn the request into the title of the document that answers it: a noun "
                 "phrase, never an imperative («Подбор камеры…», not «Подбери камеру…»), at most "
                 "12 words, in the request's language, no quotes. Return only the title."
                 + _lang_directive(out_lang),
            topic[:600], 400, temperature=0.2, retries=1) or ""
    except Exception:
        return None
    t = out.strip().splitlines()[0].strip(" «»\"'#*") if out.strip() else ""
    return t if 3 <= len(t) <= 120 else None


def _model_catalog(briefs: list, cap: int = 150, analysis: dict = None) -> str:
    """One line per product model the select briefs list, each with the page it came
    from; the first line wins for a model named on several pages. Models printing a near
    format come first: cut at 150 in crawl order, run 22's catalog lost the HE-8000-S and
    the ATP089S behind fact bullets and 2 MP cameras, and the writer never saw them."""
    from dr_collect import _PART_NO, _format_patterns
    pats = _format_patterns(analysis)
    out, seen = [], set()
    for b in briefs:
        url = b.get("url") or ""
        for ln in (b.get("brief") or "").splitlines():
            ln = ln.strip()
            m = _PART_NO.search(ln)
            if not ln.startswith(("-", "*", "•")) or not m or not _MODEL_BULLET.match(ln):
                continue
            key = ln.lstrip("-*• ").split(":")[0].strip().lower()[:60]
            if key in seen:
                continue
            seen.add(key)
            out.append(f"{ln[:260]} — {url}")
    out.sort(key=lambda ln: not any(p.search(ln) for p in pats))
    return "\n".join(out[:cap])


def models_left_out(report: str, catalog: str, analysis: dict = None) -> list:
    """Catalog lines printing a near format whose model the draft never names. Handed the
    whole catalog, run 24's writer still dropped the pvCam and the Oryx ORX-10GS-89S6C."""
    from dr_collect import _format_patterns, _PLAIN_GIGABIT
    pats = _format_patterns(analysis)
    # Only models on an interface the spec forces: 35 near-format lines were mostly USB3.
    ifaces = [re.sub(r"[\s-]", "", i).lower() for i in (analysis or {}).get("forced_interfaces") or []
              if isinstance(i, str) and i.strip() and not _PLAIN_GIGABIT.match(i.strip())]
    flat = re.sub(r"[\s-]", "", report).lower()
    out = []
    for ln in catalog.splitlines():
        name = ln.lstrip("-*• ").split(" (")[0].strip()
        squeezed = re.sub(r"[\s-]", "", ln).lower()
        if (name and any(p.search(ln) for p in pats) and re.sub(r"[\s-]", "", name).lower() not in flat
                and (not ifaces or any(i in squeezed for i in ifaces))):
            out.append(ln)
    return out[:15]


# «- ATP089S-CC (LUCID): …» is a model line; «- Sensor is a 1-inch Sony IMX255» is a fact.
_MODEL_BULLET = re.compile(r"^[-*•]\s*[^:()]{2,80}\((?:[^()]|\([^()]*\)){2,60}\)\s*:")


def hop_anchor(profile: dict, topic: str) -> str:
    """The topical anchor a follow-up query is built around.

    The DISTILLED subject, never the raw request. Searching, crawling and
    briefing are English throughout -- that is where the sources are -- and
    plan_queries already produces an English noun phrase for exactly this.
    A Russian request would otherwise produce multi-hop queries
    like "Robusta Чем отличается кофе арабика от робусты" -- half the retrieval
    passes searching in the wrong language, with the whole question glued on.

    Falls back to the raw topic only when there is no distilled one, which is
    better than searching for nothing.
    """
    core = str((profile or {}).get("core_topic") or "").strip()
    return core or (topic or "").strip()


def _expansion_pass(ctx, topic, queries, profile, caps, prog, run_dir, cache,
                    *, visited_urls, scholarly, terms, mode, label, cancelled):
    """One bounded follow-up retrieval pass (multi-hop or contradiction): search the
    given queries, drop already-seen URLs (visited-set loop protection), crawl,
    dedupe, rerank, and brief. Returns new briefs (tagged `mode`). Never raises."""
    S = settings_view(ctx)
    if not queries or cancelled():
        return []
    try:
        prog.update("Searching", f"{label}: {len(queries)} queries")
        srcs = _stage(ctx, "collect_sources")(ctx, queries, S.DR_RESULTS_PER_QUERY, prog, run_dir, profile)
        srcs = [s for s in srcs if _norm_url(s["href"]) not in visited_urls]
        if not srcs:
            logger.info("%s: no NEW sources after visited-filter", label)
            return []
        for s in srcs:
            visited_urls.add(_norm_url(s["href"]))
        # Keep each follow-up pass cheap relative to the main crawl budget.
        sub_caps = dict(caps)
        sub_caps["max_pages"] = max(4, caps.get("max_pages", S.DR_MAX_PAGES) // 2)
        pages = _stage(ctx, "crawl_pages")(ctx, srcs, sub_caps, prog, run_dir, profile, cache)
        pages = _stage(ctx, "dedupe_pages")(pages, profile.get("category", "general"))
        pages = _stage(ctx, "rerank_for_briefing")(topic, pages, profile, prog, ctx=ctx)
        briefs, _ = dep(ctx, "_brief_pages", _brief_pages)(ctx, topic, pages, profile, run_dir, prog,
                                 scholarly=scholarly, terms=terms, mode=mode,
                                 cancelled=cancelled)
        logger.info("%s: +%d briefs from %d new sources", label, len(briefs), len(srcs))
        return briefs
    except Exception as exc:
        logger.warning("%s pass failed: %s", label, exc)
        return []


def _reflect(topic, briefs, entset, contradiction_summary, gate) -> dict:
    """Reflection / error-correction (spec E): inspect the first-pass result for
    gaps. Pure + logged — drives whether a second pass is worthwhile and records
    WHY in the run state. Does not itself call the network."""
    confirm = [b for b in briefs if b.get("mode") != "contradict"]
    strong = sum(1 for b in confirm if b.get("trust") in ("PRIMARY", "SECONDARY"))
    gaps = []
    if strong == 0:
        gaps.append("no strong (primary/secondary) confirming source yet")
    if not contradiction_summary.get("has_contradictions"):
        gaps.append("no contradiction evidence gathered — provisional conclusion untested")
    if entset is not None and entset.total() == 0:
        gaps.append("no entities extracted to expand around")
    if gate.get("uncertain"):
        gaps.append("decision gate flagged uncertainty")
    out = {"gaps": gaps, "needs_second_pass": bool(gaps), "strong_confirming": strong}
    logger.info("REFLECTION: %s", out)
    return out


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_deep_research(ctx, topic: str, *, depth: str = "standard",
                      out_lang: str = "en",
                      progress: Optional[Callable[[str, dict, str], None]] = None) -> dict:
    """Run the full pipeline. Returns a dict with the report, on-disk path,
    stats, and a ``cancelled`` flag. Honors ctx.cancel_event throughout.

    `out_lang` is the language of the DOCUMENT ONLY ("en"/"ru"). Searching,
    crawling and briefing stay in English regardless — that is where the sources
    are — so a Russian report is still built on the whole English-language web.
    """
    S = settings_view(ctx)
    topic = (topic or "").strip()
    out_lang = _norm_out_lang(out_lang)
    prog = _Progress(progress)
    caps = _resolve_caps(depth, ctx=ctx)
    run_dir = S.DR_DIR / f"{int(time.time())}_{_slug(topic)}"
    started = time.time()

    if not topic:
        return {"report": "", "path": None, "stats": prog.stats, "cancelled": False,
                "error": "empty topic"}

    def cancelled():
        return ctx is not None and ctx.is_cancelled()

    # 1. Expand queries (+ classify topic for ddgs tuning)
    prog.update("Expanding queries", topic)
    queries, profile = _stage(ctx, "plan_queries")(ctx, topic, caps["max_queries"])
    _select = _is_selection_request(topic, profile, ctx=ctx)
    if _select:
        # A spec list is answered by catalog pages of any age: «product» kept a 1-year
        # window, and the 2019 5GigE IMX255 cameras that meet the specs never came up.
        profile.update(recency="any", timelimit=None, news=False, select=True)
        # Recall is the product here: at the generic caps (12 queries, 40 pages) a deep run
        # finished in 10 of its 60 minutes and never reached the 10GigE cameras.
        sel = dep(ctx, "selection_queries", selection_queries)(
            ctx, topic, n=caps["max_queries"] + 8, sink=profile)
        caps["max_pages"] = caps["max_pages"] * 3 // 2
        queries = list(dict.fromkeys(sel + queries))[:max(caps["max_queries"], len(sel) + 2)]
        logger.info("SELECTION intent: %d requirement-driven queries lead the plan", len(sel))
    prog.update("Expanding queries",
                f"{len(queries)} queries · {profile['category']}/{profile['recency']}",
                queries=len(queries))
    dep(ctx, "_save_state", _save_state)(run_dir, phase="planning", topic=topic, queries=queries, profile=profile)
    if cancelled():
        return _finish(ctx, run_dir, topic, "", prog, queries, [], started, cancelled=True)

    # 2. Collect sources
    sources = _stage(ctx, "collect_sources")(ctx, queries, S.DR_RESULTS_PER_QUERY, prog, run_dir, profile)

    # 2b. Coverage / diversity check + bounded corrective re-plan loop
    # (extended for manual control). DR_MAX_COLLECTION_ROUNDS=1 (default) reproduces
    # the original "one corrective re-plan" exactly. Raising it, or setting
    # DR_MIN_SOURCES/DR_TARGET_SOURCES/DR_MIN_UNIQUE_DOMAINS/DR_FORCE_EXHAUSTIVE,
    # makes the agent keep broadening across MULTIPLE rounds until those targets
    # are met — it will NOT stop after a small number of sources just because the
    # first pass looked "good enough" by the old heuristic. Always bounded: by
    # DR_MAX_COLLECTION_ROUNDS, by DR_MAX_SOURCES, and by a stagnation guard (a
    # round that finds zero new sources stops the loop even under force-exhaustive
    # — searching harder with the same queries can't manufacture sources that
    # don't exist).
    if S.DR_REPLAN_ENABLED and not cancelled():
        max_rounds = max(1, S.DR_MAX_COLLECTION_ROUNDS)
        for round_n in range(1, max_rounds + 1):
            if cancelled():
                break
            if S.DR_MAX_SOURCES > 0 and len(sources) >= S.DR_MAX_SOURCES:
                break
            cov = coverage_signals(sources, profile)
            domains_now = len({_domain(s["href"]) for s in sources})
            targets_unmet = (
                (S.DR_MIN_SOURCES > 0 and len(sources) < S.DR_MIN_SOURCES) or
                (S.DR_TARGET_SOURCES > 0 and len(sources) < S.DR_TARGET_SOURCES) or
                (S.DR_MIN_UNIQUE_DOMAINS > 0 and domains_now < S.DR_MIN_UNIQUE_DOMAINS)
            )
            if not (cov["weak"] or targets_unmet or S.DR_FORCE_EXHAUSTIVE):
                break
            reasons = list(cov["reasons"])
            if targets_unmet:
                reasons.append("below configured min-sources/min-domains target")
            logger.info("Coverage round %d/%d (%s) — broadening", round_n, max_rounds, reasons)
            prog.update("Expanding queries",
                        f"round {round_n}/{max_rounds}: broadening — " + "; ".join(reasons))
            extra = dep(ctx, "_corrective_queries", _corrective_queries)(ctx, topic, profile, queries, caps["max_queries"])
            if not extra:
                break
            more = _stage(ctx, "collect_sources")(ctx, extra, S.DR_RESULTS_PER_QUERY, prog, run_dir, profile)
            have = {_norm_url(s["href"]) for s in sources}
            added = [s for s in more if _norm_url(s["href"]) not in have]
            if S.DR_MAX_SOURCES > 0:
                added = added[:max(0, S.DR_MAX_SOURCES - len(sources))]
            sources.extend(added)
            sources.sort(key=lambda s: _source_rank(s["href"],
                                                     profile.get("category", "general")))
            queries = queries + extra
            unique_domains_now = len({_domain(s["href"]) for s in sources})
            logger.info("Round %d added %d queries, +%d new sources (now %d, %d domains)",
                        round_n, len(extra), len(added), len(sources), unique_domains_now)
            prog.update(queries=len(queries), sources=len(sources),
                        unique_domains=unique_domains_now, loops_completed=round_n)
            if not added:
                logger.info("Round %d added no new sources — stopping (stagnation)", round_n)
                break

    _clean_topic = _clean_report_title(profile.get("core_topic") or topic)
    if not sources:
        report = f"# Research: {_clean_topic}\n\nNo web sources could be collected for this topic."
        return _finish(ctx, run_dir, topic, report, prog, queries, [], started,
                       cancelled=cancelled())

    # 3. Crawl (with extraction cache + structured adapters)
    cache = ExtractionCache(S.DR_CACHE_DIR, enabled=S.DR_CACHE_ENABLED)
    pages = _stage(ctx, "crawl_pages")(ctx, sources, caps, prog, run_dir, profile, cache)
    prog.update("Deduplicating", "collapsing near-duplicate sources")
    pages = _stage(ctx, "dedupe_pages")(pages, profile.get("category", "general"))
    import steer as _steer    # a wish typed during the search steers reading and writing
    topic = _steer.with_notes(topic, _steer.take(ctx))
    pages = _stage(ctx, "rerank_for_briefing")(topic, pages, profile, prog, ctx=ctx)
    if not pages:
        report = (f"# Research: {_clean_topic}\n\nSources were found but none could be "
                  f"fetched or yielded readable content.")
        return _finish(ctx, run_dir, topic, report, prog, queries, [], started,
                       cancelled=cancelled())

    # 4. Per-source briefs (map). The relevance gate (scholarly) + host trust-floor
    # live in _brief_pages, shared with the multi-hop and contradiction passes.
    _scholarly = profile.get("category", "general") in _SCHOLARLY_CATEGORIES
    # Relevance terms come from the DISTILLED core_topic (the stable English entity),
    # NOT the raw request and NOT a planned query. The raw request may be a long
    # multilingual instruction ("нужны все формулы…") whose words never appear in the
    # English source pages → every brief would fail the gate → the safety net keeps
    # them ALL and the filter silently does nothing. A planned query like "arXiv Adam
    # survey 2023" injects generic words ('survey','arxiv') that match unrelated
    # sources. core_topic yields the clean entity set (e.g. structure/tensor/zenzo).
    _search_topic = profile.get("core_topic") or topic
    # Built for EVERY category, not just scholarly ones. When this was scholarly-only
    # a `local` run had no relevance filtering at all, so off-topic pages dragged in by
    # the multi-hop/contradiction passes (a federal acquisition regulation, the Deno
    # changelog, INTERPOL notices — on a bus-tour query) reached the report and were
    # counted as disconfirmation. Off-topic junk is not a scholarly-only failure mode.
    # What this enables is the FINAL sweep (_filter_offtopic_briefs below), which is
    # guarded by `if _terms` and carries its own safety net — it never empties the set,
    # so a term list that matches nothing degrades to the previous behaviour. The
    # per-page gate in _brief_pages stays scholarly-only on purpose: it drops briefs
    # one at a time with no all-fail protection.
    _terms = _entity_terms(_search_topic)
    visited_urls = {_norm_url(p["url"]) for p in pages}
    # Formula trace gate: instrument the per-stage formula trace only when the
    # run actually involves mathematics (scientific category OR detectable LaTeX
    # in the crawled pages) — avoids overhead/noise on non-math topics.
    _pages_math = "\n".join(p.get("text", "") for p in pages)
    _trace_formulas = (_scholarly or profile.get("category") == "science"
                       or _formula_audit.formula_profile(_pages_math)["latex_cmd_total"] > 0)
    if _trace_formulas:
        prog.snap_formula("EXTRACTION", _pages_math)

    briefs, dropped_irrelevant = dep(ctx, "_brief_pages", _brief_pages)(
        ctx, topic, pages, profile, run_dir, prog,
        scholarly=_scholarly, terms=_terms, mode=_contra.MODE_CONFIRM, cancelled=cancelled)
    prog.update(findings=len(briefs))
    dep(ctx, "_save_state", _save_state)(run_dir, phase="briefing", findings=len(briefs))
    if _trace_formulas:
        # BRIEFING stage = brief bodies + the equations we preserved from sources.
        prog.snap_formula("BRIEFING", "\n".join(
            b.get("brief", "") + "\n" + "\n".join(b.get("equations", []) or [])
            for b in briefs))

    # 4a. Entity-driven multi-hop expansion (spec B): extract entities from the
    # first-pass briefs and search around the salient ones. Bounded by depth +
    # the shared visited-URL set so it can never loop.
    entset = _entities.extract_from_briefs(briefs) if briefs else _entities.EntitySet()
    visited_queries = {q.lower() for q in queries}
    # A selection steers its own second rounds from the components and makers the pages
    # name; the entity hop, graph deep-dives and gap reflection searched «Teledyne»,
    # «Ethernet» and a shop's create-account page for a camera spec.
    for hop in range(2 if _select and briefs else 0):
        if cancelled():
            break
        fq = dep(ctx, "selection_followups", selection_followups)(
            ctx, topic, briefs, profile.get("selection_analysis"), visited_queries)
        if not fq:
            break
        visited_queries.update(q.lower() for q in fq)
        # Not a follow-up's half budget: these rounds reach the cameras the first one
        # could not name.
        new_briefs = dep(ctx, "_expansion_pass", _expansion_pass)(
            ctx, topic, fq, profile, dict(caps, max_pages=caps["max_pages"] * 2), prog, run_dir,
            cache, visited_urls=visited_urls, scholarly=_scholarly, terms=_terms,
            mode=_contra.MODE_CONFIRM, label=f"selection-{hop + 1}", cancelled=cancelled)
        if not new_briefs:
            break
        briefs.extend(new_briefs)
        prog.update(findings=len(briefs))
    if (S.DR_MULTIHOP_ENABLED and S.DR_MULTIHOP_DEPTH > 0 and briefs and not cancelled()
            and not _select):
        hop_target = briefs
        for hop in range(S.DR_MULTIHOP_DEPTH):
            hop_es = _entities.extract_from_briefs(hop_target)
            # profile["core_topic"], not the raw topic. Searching, crawling and
            # briefing are English throughout -- that is where the sources are --
            # and plan_queries already distils the request into an English noun
            # phrase for exactly this. Passing the raw request sent queries like
            # "Robusta Чем отличается кофе арабика от робусты" to the engine:
            # half the retrieval passes searching in the wrong language, with the
            # whole question glued on, observed live.
            hop_q = _entities.entity_queries(
                hop_anchor(profile, topic), hop_es,
                max_entities=S.DR_MULTIHOP_ENTITIES,
                per_entity=1, visited=visited_queries)
            if not hop_q:
                break
            prog.update("Searching", f"multi-hop {hop+1}/{S.DR_MULTIHOP_DEPTH}: {len(hop_q)} entity queries")
            new_briefs = dep(ctx, "_expansion_pass", _expansion_pass)(
                ctx, topic, hop_q, profile, caps, prog, run_dir, cache,
                visited_urls=visited_urls, scholarly=_scholarly, terms=_terms,
                mode=_contra.MODE_CONFIRM, label=f"multihop-{hop+1}", cancelled=cancelled)
            if not new_briefs:
                break
            briefs.extend(new_briefs)
            hop_target = new_briefs
            prog.update(findings=len(briefs))
            for nb in new_briefs:
                _entities.extract_entities(f"{nb.get('title', '')} {nb.get('brief', '')}", into=entset)

    _contra.tag_confirming(briefs)

    # 4a2. Hierarchical clustering + graph-guided deep dives (spec A/B/C). Group the
    # confirming briefs into sub-threads, build an interim graph, and let the graph +
    # cluster structure DECIDE the next retrieval: central-entity hubs, weakly-
    # supported claims, and important-but-thin major clusters get deep-dived. Bounded
    # by DR_GRAPH_EXPANSION_QUERIES + the shared visited sets so it cannot loop.
    interim_clusters = None
    if S.DR_HIERARCHICAL_ENABLED and briefs and not cancelled():
        interim_clusters = _cluster.cluster_briefs(
            briefs, threshold=S.DR_CLUSTER_THRESHOLD, major_min_size=S.DR_CLUSTER_MAJOR_MIN)
        dep(ctx, "_save_state", _save_state)(run_dir, phase="clustering_interim", clusters=interim_clusters.to_dict())
    if (S.DR_GRAPH_EXPANSION_ENABLED and S.DR_HIERARCHICAL_ENABLED and briefs and not _select
            and not cancelled()):
        interim_graph = _rgraph.build_graph(briefs)
        gx = _gguided.plan_graph_expansion(
            topic, interim_graph, interim_clusters, contradiction=None,
            visited=visited_queries, max_queries=S.DR_GRAPH_EXPANSION_QUERIES)
        dep(ctx, "_save_state", _save_state)(run_dir, phase="graph_expansion_plan", targets=gx)
        if gx:
            prog.update("Searching", f"graph-guided deep dive: {len(gx)} targets")
            gx_briefs = dep(ctx, "_expansion_pass", _expansion_pass)(
                ctx, topic, [g["query"] for g in gx], profile, caps, prog, run_dir, cache,
                visited_urls=visited_urls, scholarly=_scholarly, terms=_terms,
                mode=_contra.MODE_CONFIRM, label="graph-deepdive", cancelled=cancelled)
            briefs.extend(gx_briefs)
            for nb in gx_briefs:
                _entities.extract_entities(f"{nb.get('title', '')} {nb.get('brief', '')}", into=entset)
            prog.update(findings=len(briefs))

    # 4b. Contradiction search (spec A + E) — a SEPARATE pass that tries to disprove
    # the provisional findings, now PRIORITIZED by graph/cluster importance: major
    # threads and high-centrality entities are challenged first. Briefs tagged
    # `contradict` and surfaced apart.
    # A practical ask never shows contradictions (no gate banner, no contradiction note),
    # yet the pass still fed its pages to the writer: for a camera spec request it
    # searched «ip67 criticism OR limitations OR disputed» and brought back AI-limits
    # essays, a novel and the Oregon boundary dispute (live 2026-10-16).
    _contra_ran = bool(S.DR_CONTRADICTION_ENABLED and briefs and not cancelled()
                       and not _select and not _is_practical_request(topic, profile, ctx=ctx))
    if _contra_ran:
        # _search_topic (the distilled core_topic), NOT the raw request — the
        # templates append English angle words, so a conversational topic makes the
        # engine match on "limitations"/"benchmark" alone and returns unrelated pages.
        cq = _contra.contradiction_queries(
            _search_topic, entities=entset.rank(top_k=S.DR_MULTIHOP_ENTITIES),
            max_queries=S.DR_CONTRADICTION_QUERIES, visited=visited_queries)
        if S.DR_HIERARCHICAL_ENABLED and interim_clusters is not None:
            prioritized = _gguided.cluster_contradiction_targets(
                topic, _rgraph.build_graph(briefs), interim_clusters,
                visited=visited_queries, max_queries=S.DR_CONTRADICTION_QUERIES)
            dep(ctx, "_save_state", _save_state)(run_dir, phase="contradiction_priority", targets=prioritized)
            cq = [p["query"] for p in prioritized] + cq  # graph-prioritized first
        contra_briefs = dep(ctx, "_expansion_pass", _expansion_pass)(
            ctx, topic, cq, profile, caps, prog, run_dir, cache,
            visited_urls=visited_urls, scholarly=_scholarly, terms=_terms,
            mode=_contra.MODE_CONTRADICT, label="contradiction", cancelled=cancelled)
        _contra.tag_contradicting(contra_briefs)
        briefs.extend(contra_briefs)
        prog.update(findings=len(briefs))
    contra_sig = _contra.contradiction_summary(briefs)
    prog.stats["contradictions"] = contra_sig["count"]
    dep(ctx, "_save_state", _save_state)(run_dir, phase="contradiction", contradictions=contra_sig)

    if dropped_irrelevant:
        prog.stats["off_topic_dropped"] = dropped_irrelevant
        logger.info("Relevance gate dropped %d off-topic briefs", dropped_irrelevant)

    # 4c. Citation-graph enrichment for scholarly topics: inject a
    # PRIMARY synthetic source carrying citation counts + lineage from OpenAlex,
    # so synthesis can build the intellectual-lineage table that plain web
    # search can never supply.
    have_citations, cit_body = False, ""
    if (S.DR_CITATIONS_ENABLED and profile.get("category") in _SCHOLARLY_CATEGORIES
            and not cancelled()):
        # OpenAlex search needs a SHORT entity phrase, not the whole multi-line
        # prompt (a 2.9k-char query returns zero results -> silent no-op). Use the
        # shortest concrete planned query, else the trimmed first line of the topic.
        cite_query = _citation_query(_search_topic, queries)
        prog.update("Verifying", f"citation graph (OpenAlex): {cite_query}")
        try:
            cit = dep(ctx, "citation_lineage_brief", citation_lineage_brief)(cite_query, S.DR_CITATION_MAILTO)
        except Exception as exc:
            logger.warning("citation enrichment failed: %s", exc)
            cit = None
        logger.info("Citation enrichment query=%r -> %s",
                    cite_query, "hit" if cit else "MISS")
        if cit:
            trust, body = _parse_trust(cit)
            cit_body = body
            briefs.append({"domain": "openalex.org", "url": "https://openalex.org",
                           "title": "Citation graph (OpenAlex)", "brief": body,
                           "trust": trust, "authority": _AUTH_BASE_SCORE,
                           "cluster_size": 1, "cluster_domains": ["openalex.org"]})
            prog.update(findings=len(briefs))
            have_citations = True
            logger.info("Citation enrichment added (%d chars)", len(cit))

    if not briefs:
        report = no_briefs_report(topic, pages,
                                  prog.stats.get("silent_briefs") or 0)
        return _finish(ctx, run_dir, topic, report, prog, queries, [], started,
                       cancelled=cancelled())

    # 4c2. Claim-level dedup/merge (spec D): collapse confirming briefs that assert
    # the same thing into one representative WHILE preserving every source as
    # provenance (support_count). Contradictions are never merged in. Cuts repetition
    # in the final report and turns repetition into a corroboration signal.
    cluster_digests = []
    if S.DR_CLAIM_MERGE_ENABLED and briefs:
        briefs, merge_log = _cmerge.merge_claims(
            briefs, threshold=S.DR_CLAIM_MERGE_THRESHOLD)
        if merge_log:
            prog.stats["claims_merged"] = len(merge_log)
            dep(ctx, "_save_state", _save_state)(run_dir, phase="claim_merge", merges=merge_log)

    # 4d. Provenance graph (spec C+G): entity↔source↔claim with supported_by /
    # refuted_by edges. Built now so the reflection loop + gate can read structure;
    # rebuilt after every reflection cycle, and saved once it is final.
    graph = _rgraph.build_graph(briefs)

    def _gate_and_reflect():
        """(re)compute contradiction summary, decision gate, and reflection gaps from
        the CURRENT brief set. Returns (decision, reflection)."""
        nonlocal contra_sig
        contra_sig = _contra.contradiction_summary(briefs)
        prog.stats["contradictions"] = contra_sig["count"]
        confirm_b, _ = _contra.partition_evidence(briefs)
        stats4 = evidence_stats(confirm_b)
        amb = dep(ctx, "_gate", _gate).assess_ambiguity(topic, has_entities=entset.total() > 0)
        cov = coverage_signals([{"domain": b["domain"], "href": b["url"], "query": ""}
                                for b in confirm_b], profile)
        dec = dep(ctx, "_gate", _gate).decide(stats4, cov, contra_sig, ambiguity=amb,
                           min_strong=S.DR_GATE_MIN_STRONG, min_clusters=S.DR_GATE_MIN_CLUSTERS)
        refl = _reflect(topic, briefs, entset, contra_sig, dec) if S.DR_REFLECTION_ENABLED else {}
        return dec, refl

    # 4e+4f. Decision gate (spec D) + reflection gaps (spec C) on the first pass.
    decision, reflection = _gate_and_reflect()
    prog.stats["decision"] = decision["decision"]
    dep(ctx, "_save_state", _save_state)(run_dir, phase="decision_gate", decision=decision)
    logger.info("DECISION GATE: %s", dep(ctx, "_gate", _gate).gate_banner(decision))
    if reflection:
        prog.stats["reflection"] = reflection
        dep(ctx, "_save_state", _save_state)(run_dir, phase="reflection", reflection=reflection)

    # 4f2. ACTIVE reflection loop (spec C): turn the recorded gaps (weak/unsupported
    # claims, unresolved contradictions, thin clusters, missing entities) into NEW
    # graph-guided retrieval, fold the results back in, rebuild the graph, and
    # re-decide. Bounded by DR_REFLECTION_MAX_ITERATIONS + DR_REFLECTION_RESEARCH_BUDGET
    # and visited-set loop protection; stops early when nothing new is found.
    # Depth-scoped: a "quick" run does ONE pass, which is what the word means and
    # what its ~15-minute label promises.
    refl_budget = caps.get("reflect_budget", S.DR_REFLECTION_RESEARCH_BUDGET)
    refl_max = caps.get("reflect_iters", S.DR_REFLECTION_MAX_ITERATIONS)
    refl_iter = 0
    if S.DR_REFLECTION_ENABLED and refl_max <= 0 and reflection.get("needs_second_pass"):
        logger.info("reflection skipped at depth=%s (single-pass by design)", depth)
    while (S.DR_REFLECTION_ENABLED and reflection.get("needs_second_pass") and not _select
           and refl_iter < refl_max and refl_budget > 0
           and not cancelled()):
        refl_iter += 1
        interim_cs = (_cluster.cluster_briefs(briefs, threshold=S.DR_CLUSTER_THRESHOLD,
                      major_min_size=S.DR_CLUSTER_MAJOR_MIN) if briefs else None)
        plan = _gguided.plan_graph_expansion(
            topic, graph, interim_cs, contradiction=contra_sig,
            visited=visited_queries,
            max_queries=min(refl_budget, S.DR_GRAPH_EXPANSION_QUERIES))
        rq = [p["query"] for p in plan][:refl_budget]
        dep(ctx, "_save_state", _save_state)(run_dir, phase="reflection_iteration", iteration=refl_iter,
                    gaps=reflection.get("gaps", []), targets=plan, queries=rq)
        if not rq:
            logger.info("reflection loop: no NEW targets — stopping (loop guard)")
            break
        prog.update("Searching", f"reflection cycle {refl_iter}: {len(rq)} gap-closing queries")
        new_briefs = dep(ctx, "_expansion_pass", _expansion_pass)(
            ctx, topic, rq, profile, caps, prog, run_dir, cache,
            visited_urls=visited_urls, scholarly=_scholarly, terms=_terms,
            mode=_contra.MODE_CONFIRM, label=f"reflection-{refl_iter}", cancelled=cancelled)
        refl_budget -= len(rq)
        if not new_briefs:
            logger.info("reflection loop: cycle %d added 0 briefs — stopping", refl_iter)
            dep(ctx, "_save_state", _save_state)(run_dir, phase="reflection_result", iteration=refl_iter, added=0)
            break
        briefs.extend(new_briefs)
        for nb in new_briefs:
            _entities.extract_entities(f"{nb.get('title', '')} {nb.get('brief', '')}", into=entset)
        prog.update(findings=len(briefs))
        graph = _rgraph.build_graph(briefs)              # UPDATED graph
        decision, reflection = _gate_and_reflect()       # UPDATED decision + gaps
        dep(ctx, "_save_state", _save_state)(run_dir, phase="reflection_result", iteration=refl_iter,
                    added=len(new_briefs), total_findings=len(briefs),
                    decision=decision, reflection=reflection)
        logger.info("reflection cycle %d: +%d briefs, decision=%s, remaining gaps=%d",
                    refl_iter, len(new_briefs), decision["decision"],
                    len(reflection.get("gaps", [])))
    prog.stats["reflection_iterations"] = refl_iter

    # Final off-topic sweep (scholarly topics): the multi-hop / contradiction /
    # reflection passes can drag in noise an ambiguous entity matches loosely (the
    # "Mayor Adams" politics page on an "Adam optimizer" run). Drop briefs that name
    # NO topic entity term BEFORE they reach clustering, the contradiction note, the
    # source appendix or synthesis. Word-boundary matched; never empties the set.
    if _terms and briefs:
        briefs, _offtopic = _filter_offtopic_briefs(briefs, _terms)
        if _offtopic:
            logger.info("relevance sweep: dropped %d off-topic brief(s): %s",
                        len(_offtopic), ", ".join(b.get("domain", "?") for b in _offtopic))
            prog.stats["offtopic_dropped"] = len(_offtopic)

    # Re-merge after reflection added evidence, then finalize the graph + save it.
    if S.DR_CLAIM_MERGE_ENABLED and refl_iter and briefs:
        briefs, _ml = _cmerge.merge_claims(briefs, threshold=S.DR_CLAIM_MERGE_THRESHOLD)
        graph = _rgraph.build_graph(briefs)
    graph.save(run_dir / "graph.json")
    prog.stats["graph"] = graph.stats()
    if _trace_formulas:
        prog.snap_formula("CLAIM_MERGE_GRAPH", "\n".join(
            b.get("brief", "") + "\n" + "\n".join(b.get("equations", []) or [])
            for b in briefs))

    # 4d2. Final lexical clusters + per-cluster digests (spec A, flat layer).
    final_clusters = None
    if S.DR_HIERARCHICAL_ENABLED and briefs:
        final_clusters = _cluster.cluster_briefs(
            briefs, threshold=S.DR_CLUSTER_THRESHOLD, major_min_size=S.DR_CLUSTER_MAJOR_MIN)
        cluster_digests = _hier.cluster_digests(
            final_clusters, contradiction_domains=set(contra_sig.get("domains", [])),
            max_clusters=S.DR_CLUSTER_DIGEST_MAX)
        prog.stats["clusters"] = final_clusters.stats()
        try:
            (run_dir / "clusters.json").write_text(
                json.dumps({"clusters": final_clusters.to_dict(),
                            "digests": cluster_digests}, ensure_ascii=False, indent=2),
                encoding="utf-8")
        except Exception as exc:
            logger.debug("clusters.json save failed: %s", exc)
        dep(ctx, "_save_state", _save_state)(run_dir, phase="clustering_final",
                    clusters=final_clusters.to_dict(), digests=cluster_digests)

    # 4d3. Recursive hierarchy (spec A): root → cluster → subcluster → … saved to
    # hierarchy.json; its tree drives synthesis when it has real depth.
    tree_preamble = None
    if S.DR_HIERARCHICAL_ENABLED and briefs:
        hierarchy = _hierarchy.build_hierarchy(
            briefs, max_depth=S.DR_HIERARCHY_MAX_DEPTH,
            min_cluster_size=S.DR_HIERARCHY_MIN_CLUSTER_SIZE,
            split_threshold=S.DR_HIERARCHY_SPLIT_THRESHOLD)
        prog.stats["hierarchy"] = hierarchy.stats()
        try:
            (run_dir / "hierarchy.json").write_text(
                json.dumps(hierarchy.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8")
        except Exception as exc:
            logger.debug("hierarchy.json save failed: %s", exc)
        dep(ctx, "_save_state", _save_state)(run_dir, phase="hierarchy", stats=hierarchy.stats())
        if hierarchy.levels() >= 3:                       # use the tree only when it has depth
            tree_preamble = _hierarchy.render_tree_preamble(hierarchy)

    # 4d4. Graph communities (spec B): real modularity (Louvain) over the
    # entity/source/claim/contradiction graph, distinct from lexical clusters.
    if S.DR_COMMUNITIES_ENABLED and briefs:
        comm = _communities.detect_communities(graph)
        if final_clusters is not None:
            comm["vs_clusters"] = _communities.communities_vs_clusters(comm, final_clusters)
        prog.stats["communities"] = comm["stats"]
        try:
            (run_dir / "communities.json").write_text(
                json.dumps(comm, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.debug("communities.json save failed: %s", exc)
        dep(ctx, "_save_state", _save_state)(run_dir, phase="communities", stats=comm["stats"],
                    vs_clusters=comm.get("vs_clusters"))

    if S.DR_DECISION_GATE_ENABLED and decision["decision"] in (_gate.ASK, _gate.ABSTAIN):
        # Do not fabricate a confident answer. Surface the decision + whatever weak
        # evidence exists, plus the source appendix, and stop before synthesis.
        banner = dep(ctx, "_gate", _gate).gate_banner(decision, out_lang)
        body = (f"# {_lab(out_lang, 'research')}: {_clean_topic}\n\n> {banner}\n\n"
                + (f"**{_lab(out_lang, 'why')}:** " + "; ".join(decision["reasons"]) + "\n\n"
                   if decision["reasons"] else "")
                + (f"**{_lab(out_lang, 'clarify')}:** {decision['clarifying_question']}\n\n"
                   if decision.get("clarifying_question") else "")
                + _contradiction_note(contra_sig if _contra_ran else {}, out_lang)
                )
        # On THIS path the document is `body` — `report` does not exist yet.
        body = _append_once(body, _sources_appendix(briefs, out_lang))
        return _finish(ctx, run_dir, topic, body, prog, queries, briefs, started,
                       cancelled=cancelled())

    try:
        from rerank import release_gpu
        release_gpu()
    except Exception as exc:
        logger.debug("reranker release skipped: %s", exc)
    # 5. Synthesize (reduce). Only ask for the Citation Lineage section if we
    # actually injected OpenAlex data — otherwise the model fabricates "(OpenAlex)".
    sections = requested_sections(topic)
    if not have_citations and "Citation Lineage" in sections:
        sections.remove("Citation Lineage")
    header = _report_header(topic, briefs, prog.stats, depth,
                            core_topic=_search_topic, out_lang=out_lang,
                            title=_document_title(ctx, topic, out_lang))
    # Prefer the recursive-hierarchy scaffold when the tree has real depth (≥3
    # levels); otherwise fall back to the flat per-cluster digest preamble.
    preamble = tree_preamble or (
        _hier.render_digest_preamble(cluster_digests) if cluster_digests else None)

    report = None
    survey = None
    # PRACTICAL intent gets a bookable-options list, NOT a survey paper.
    # A survey gives an abstract and a table of contents where the user wants links,
    # prices and departure times. The category is already classified `local`; it
    # must also pick the report FORM.
    topic = _steer.with_notes(topic, _steer.take(ctx))    # last one before the writing
    _practical = _select or _is_practical_request(topic, profile, ctx=ctx)
    if _select:
        logger.info("SELECTION intent — writing a requirements x candidates matrix")
        prog.update("Building report", "matching candidates to the requirements")
        try:
            evidence = _consolidate_evidence(ctx, topic, briefs, prog)
            sel_tokens = min(S.DR_SECTION_TOKEN_CEILING, caps.get("report_tokens") or 6000) * 2
            # Prompt + answer must fit the loaded window: 150 model lines on top of the
            # briefs made a 46k-token request for 40960 and the run fell back (run 11).
            budget = max(8000, (_model_context(ctx) - sel_tokens - 1500) * 3)
            catalog = _model_catalog(briefs, analysis=profile.get("selection_analysis"))
            evidence = evidence[:max(2000, budget - len(catalog) - 4000)]
            sel_user = (
                f"User request: {topic}\n\n"
                # The arithmetic was done before the search; written from the briefs alone
                # the section gave B = W x H x bits x FPS with no number in it.
                + (f"ENGINEERING ANALYSIS made before the search (put its numbers in «What the "
                   f"numbers force», correct them if a source disagrees):\n"
                   f"{json.dumps(profile['selection_analysis'], ensure_ascii=False)}\n\n"
                   if profile.get("selection_analysis") else "")
                # From 100 briefs the writer kept five rows and dropped the IMX255 cameras
                # it had read; every model line is put in front of it, each with its page.
                + "EVERY MODEL THE PAGES STATE — judge each one and put it in the Candidates "
                  "table (meets the hard requirements or fails exactly one) or the Ruled out "
                  "list; keep the six sections exactly as instructed:\n"
                + catalog + "\n\n"
                + "Per-source briefs:\n" + evidence + _lang_directive(out_lang))
            think = dep(ctx, "_think_call", _think_call)
            report = (think(ctx, SELECTION_REPORT_PROMPT + _lang_directive(out_lang), sel_user,
                            sel_tokens, temperature=0.2, retries=1) or "").strip() or None
            missing = selection_form_missing(report) if report else []
            if missing:
                logger.warning("selection draft lacks %s — rewriting once", missing)
                again = (think(ctx, SELECTION_REPORT_PROMPT + _lang_directive(out_lang),
                               sel_user + "\n\nYour previous answer dropped or renamed these "
                               "required sections: " + "; ".join(missing) + ". Write all six "
                               "sections with exactly the headings given, in that order.",
                               sel_tokens, temperature=0.2, retries=1) or "").strip()
                if again and len(selection_form_missing(again)) < len(missing):
                    report = again
            if report:
                # Before the check too, so it rewrites the recommendation from the fixed marks.
                report = rank_by_closeness(mono_color_is_color(report))
                prog.update("Building report", "checking every verdict against its source")
                left_out = models_left_out(report, catalog, profile.get("selection_analysis"))
                if left_out:
                    logger.info("selection draft left out %d near-format models: %s", len(left_out),
                                "; ".join(ln.split(" (")[0] for ln in left_out)[:400])
                checked = (dep(ctx, "_think_call", _think_call)(
                    ctx, SELECTION_VERIFY_PROMPT + _lang_directive(out_lang),
                    f"User request: {topic}\n\nDRAFT:\n{report}\n\nEVERY MODEL THE PAGES STATE:\n"
                    f"{catalog}\n\n"
                    + ("MODELS THE DRAFT LEFT OUT (each prints a near format): put each in the "
                       "candidates table or the ruled-out list:\n" + "\n".join(left_out) + "\n\n"
                       if left_out else "")
                    + "Per-source briefs:\n"
                    f"{evidence[:max(2000, budget - len(catalog) - len(report) - 4000)]}",
                    sel_tokens, temperature=0.1, retries=1) or "").strip()
                # A rewrite that lost sections or half the text is a failed check, not a fix.
                if (len(selection_form_missing(checked)) <= len(selection_form_missing(report))
                        and len(checked) >= 0.6 * len(report)):
                    report = checked
                else:
                    logger.warning("selection verify pass dropped content (%d -> %d chars) — "
                                   "keeping the draft", len(report), len(checked))
            if report:
                from dr_pdf import plain_symbols
                report = mono_color_is_color(drop_empty_columns(plain_symbols(report)))
                report, _moved = demote_failing_rows(report)
                report, _back = promote_passing_rows(report)
                report, _ = drop_false_none_claim(report)
                report = rank_by_closeness(report)
                if _moved or _back:
                    logger.info("selection matrix: %d rows failing 2+ requirements ruled out, "
                                "%d passing rows put back", _moved, _back)
                report, _link_stats = audit_report_links(report, briefs, out_lang)
                # Every page read used to be listed: USB night-vision webcams and a search
                # ad under a camera matrix. The appendix is what the matrix rests on.
                low = report.lower()
                cited = [b for b in briefs if (b.get("url") or "\0") in report
                         or (b.get("domain") or "\0").lower() in low] or briefs
                report = _append_once(report + "\n\n", _sources_appendix(cited, out_lang))
        except Exception as exc:
            logger.warning("selection synthesis failed (%s) — falling back", exc)
            report = None
    elif _practical:
        logger.info("PRACTICAL intent (category=%r) — writing an actionable options "
                    "list instead of a survey", profile.get("category"))
        prog.update("Building report", "listing bookable options")
        try:
            report = dep(ctx, "_think_call", _think_call)(
                ctx, PRACTICAL_GUIDE_PROMPT + _lang_directive(out_lang),
                f"User request: {topic}\n\nPer-source briefs:\n"
                + _consolidate_evidence(ctx, topic, briefs, prog)
                + _lang_directive(out_lang),
                min(S.DR_SECTION_TOKEN_CEILING, caps.get("report_tokens") or 6000) * 2,
                temperature=0.2,
                # If the classifier mis-picked "practical" for a topic that is
                # actually a comparison/factual question, no amount of shrink+nudge
                # rescues the practical-guide prompt (it's the wrong template, not
                # a budget problem). Fail fast on one empty attempt instead of
                # burning 3x the 600s wall-clock guard, and fall through to the
                # survey/single-pass synthesis below.
                retries=1)
            report = (report or "").strip() or None
            if report:
                # The prompt forbids inventing or shortening a URL; this checks
                # it. See dr_assemble.audit_report_links for the run that made
                # it necessary.
                report, _link_stats = audit_report_links(report, briefs, out_lang)
                report, _dropped = strip_empty_fields(report)
                if _dropped:
                    logger.info("practical report: dropped %d empty option fields",
                                _dropped)
                if _link_stats["repaired"] or _link_stats["stripped"]:
                    logger.warning("practical report links: %d checked, %d repaired "
                                   "to the collected URL, %d stripped as unverifiable",
                                   _link_stats["checked"], _link_stats["repaired"],
                                   _link_stats["stripped"])
                report = _append_once(report + "\n\n",
                                      _sources_appendix(briefs, out_lang))
        except Exception as exc:
            logger.warning("practical synthesis failed (%s) — falling back", exc)
            report = None

    if S.DR_SURVEY_MODE and briefs and not _practical and report is None:
        # Primary product: a dynamically-outlined, section-by-section scientific
        # document (reads like a survey paper, merges sources, reaches tens of
        # pages). It writes its OWN title + abstract + TOC, so the run header is
        # NOT prepended in this path. Falls through to single-pass on empty.
        try:
            survey = dep(ctx, "synthesize_survey", synthesize_survey)(ctx, topic, briefs, prog,
                                       section_tokens=caps.get("report_tokens"),
                                       out_lang=out_lang)
        except Exception as exc:
            logger.warning("survey synthesis failed (%s) — falling back to single-pass", exc)
            survey = None
    if survey:
        body = survey
        if have_citations and cit_body:
            sec = _citation_section_from_brief(cit_body, _terms)
            if sec:
                body = body.rstrip() + "\n\n" + sec + "\n"
        gate_line = f"\n> {dep(ctx, "_gate", _gate).gate_banner(decision, out_lang)}\n" if S.DR_DECISION_GATE_ENABLED else ""
        report = (f"{body}\n{gate_line}\n"
                  f"{_contradiction_note(contra_sig if _contra_ran else {}, out_lang)}")
        # Survey mode writes its own title, abstract and contents, so the run
        # header is not prepended here -- which left the PRIMARY product with no
        # provenance at all. Measured live: a 34,000-character document that
        # never said what depth it ran at, how many sources it searched, or how
        # many pages it actually read. The footer carries the same counters, at
        # the end where it does not fight the model's own front matter.
        report = _append_once(report, provenance_footer(
            briefs, prog.stats, depth, out_lang))
        report = _append_once(report, _sources_appendix(briefs, out_lang))

    if report is None:
        report = _stage(ctx, "synthesize_report")(ctx, topic, briefs, prog,
                                   max_tokens=caps.get("report_tokens"),
                                   sections=sections, preamble=preamble,
                                   out_lang=out_lang)
    if not report and briefs:
        # Empty synthesis is usually gpt-oss spending its whole token budget on the
        # reasoning channel and never writing the final answer (observed: a 294s
        # "Building report" that produced nothing). Retry on the strongest half with
        # an effort LADDER — drop reasoning depth so the model actually WRITES — before
        # falling back to the deterministic digest. Honors "let it think" on attempt 1,
        # guarantees a real report by the last rung.
        strong = sorted(briefs, key=lambda b: {"PRIMARY": 3, "SECONDARY": 2, "COMMUNITY": 1,
                        "LOW": 0}.get(b.get("trust", "COMMUNITY"), 1), reverse=True)[:max(4, len(briefs)//2)]
        for rung in ("medium", "low"):
            logger.warning("synthesis empty — retrying at %s effort on the strongest %d briefs",
                           rung, len(strong))
            prog.update("Building report", f"retrying synthesis at {rung} reasoning effort")
            report = _stage(ctx, "synthesize_report")(ctx, topic, strong, prog,
                                       max_tokens=caps.get("report_tokens"), sections=None,
                                       preamble=preamble, out_lang=out_lang,
                                       effort=rung)
            if report:
                break
    if survey:
        pass  # already fully assembled (own title/abstract/TOC + appendix)
    elif report:
        # Deterministic cleanup: strip duplicate sections, and replace any
        # model-written lineage prose with the real OpenAlex table (counts +
        # descendants guaranteed when data exists).
        if have_citations and cit_body:
            report = _postprocess_report(
                report, drop_substrs=("influential citation", "citation lineage"))
            sec = _citation_section_from_brief(cit_body, _terms)
            if sec:
                report = report.rstrip() + "\n\n" + sec + "\n"
        else:
            report = _postprocess_report(report)
        gate_line = ""
        # A candidate matrix carries its own ✅/❌/❓ per spec; a run-wide «confidence: high»
        # above it read as a verdict on every row (camera selection, 2026-10-16).
        if S.DR_DECISION_GATE_ENABLED and not _practical:
            gate_line = f"\n> {dep(ctx, "_gate", _gate).gate_banner(decision, out_lang)}\n"
        report = (f"{header}{gate_line}\n{report}\n"
                  f"{_contradiction_note(contra_sig if _contra_ran else {}, out_lang)}")
        report = _append_once(report, _sources_appendix(briefs, out_lang))
    else:
        # Last-resort fallback: the LLM reduce produced nothing twice. Do NOT emit a
        # dead "no output" stub — we still have the hierarchical cluster digests and
        # the per-source briefs, so deliver a deterministic, evidence-bound summary
        # built from those (every point cited to its source/thread).
        logger.warning("synthesis empty after retry — emitting deterministic digest fallback")
        gate_line = f"\n> {dep(ctx, "_gate", _gate).gate_banner(decision, out_lang)}\n" if S.DR_DECISION_GATE_ENABLED else ""
        report = (f"{header}{gate_line}\n"
                  + _deterministic_digest_report(topic, cluster_digests, briefs)
                  + _contradiction_note(contra_sig if _contra_ran else {}, out_lang))
        report = _append_once(report, _sources_appendix(briefs, out_lang))

    if _trace_formulas:
        prog.snap_formula("SYNTHESIS", report)

    return _finish(ctx, run_dir, topic, report, prog, queries, briefs, started,
                   cancelled=cancelled(), depth=depth)


def _finish(ctx, run_dir, topic, report, prog, queries, briefs, started, *, cancelled,
            depth: str = "standard"):
    elapsed = time.time() - started
    # Feed the ETA the truth. Only COMPLETED runs count: a cancelled run took as
    # long as the user's patience, not as long as the work.
    if not cancelled and report:
        record_run_duration(depth, elapsed)
    prog.finalize_timings()  # flush the in-flight phase so stage_timings is complete
    if prog.stats.get("stage_timings"):
        logger.info("Stage timings (s): %s", prog.stats["stage_timings"])
    meta = {
        "topic": topic,
        "queries": queries,
        "stats": prog.stats,
        "elapsed_sec": round(elapsed, 1),
        "cancelled": cancelled,
        "sources": [{"domain": b["domain"], "url": b["url"]} for b in briefs],
    }
    # Preserve raw equations: collect the verbatim display-equation blocks pulled
    # from sources (briefing reliably drops them) and append a faithful, attributed
    # section so equations ALWAYS reach the report even when the LLM paraphrased
    # them away. Never rewritten — copied as-is, marked by source.
    if report:
        eq_section = _build_equations_section(briefs)
        if eq_section and "Key equations" not in report:
            report = report.rstrip() + "\n\n" + eq_section
    if report:
        # Deterministic LaTeX-artifact repair on the WHOLE assembled report (covers every
        # synthesis path + the equations appendix): strip ar5iv layout primitives that
        # leaked into matrices (\vskip/\cr), normalize \(..\)/\[..\] to $..$/$$..$$, and
        # unwrap domain citations the model put in \text{}. Runs before the math audit.
        report = _repair_latex_artifacts(report)
        ma = audit_math(report)
        prog.stats["math_audit"] = ma
        if not ma["clean"]:
            logger.warning("MATH AUDIT: report has math issues: %s", ma)
        # Formula fidelity (release-blocking): profile the briefs that fed
        # synthesis vs the final report. Flag (a) LaTeX/Unicode tokens that appear
        # in the report but in NO brief — candidate hallucinated math — and (b)
        # corruption (U+FFFD/mojibake/lost-backslash) that appears in the report.
        try:
            # Source profile = brief bodies + the verbatim equations we preserved
            # from the SAME sources. Without the equations, the preserved-equation
            # section we append would itself trip the hallucination check (false
            # positive) since the LLM brief body dropped that LaTeX.
            brief_text = "\n".join(
                b.get("brief", "") + "\n" + "\n".join(b.get("equations", []) or [])
                for b in briefs)
            src_prof = _formula_audit.formula_profile(brief_text)
            rep_prof = _formula_audit.formula_profile(report)
            fdiff = _formula_audit.diff_profiles(src_prof, rep_prof)
            prog.stats["formula_fidelity"] = {
                "report_clean": rep_prof["clean"],
                "introduced_cmd_tokens": fdiff["introduced_cmd_tokens"],
                "lost_cmd_tokens": fdiff["lost_cmd_tokens"],
                "new_unicode": fdiff["new_unicode"],
                "corruption_appeared": fdiff["corruption_appeared"],
            }
            if fdiff["introduced_cmd_tokens"]:
                logger.warning("FORMULA FIDELITY: report introduced LaTeX tokens not in any "
                               "brief (candidate hallucination): %s", fdiff["introduced_cmd_tokens"])
            if fdiff["corruption_appeared"]:
                logger.warning("FORMULA FIDELITY: corruption appeared in report: %s",
                               fdiff["corruption_appeared"])
        except Exception as exc:
            logger.debug("formula fidelity check failed: %s", exc)
        # Per-stage formula trace (release-blocking auditability): for scientific
        # runs, persist the full SOURCE→…→FINAL trace recorded via prog.snap_formula
        # so the exact stage where any symbol is lost/introduced/corrupted is
        # auditable from disk after the run. Empty on non-math runs (not gated in).
        try:
            snaps = list(getattr(prog, "formula_snapshots", []))
            if snaps:
                snaps.append(("FINAL_REPORT", report))
                trace = _formula_audit.build_trace(snaps)
                (run_dir / "formula_trace.json").write_text(
                    json.dumps({"topic": topic, "trace": trace}, ensure_ascii=False, indent=2),
                    encoding="utf-8")
                # compact per-stage summary into stats for quick inspection
                prog.stats["formula_trace"] = [
                    {"stage": r["stage"], "latex_tokens": r["latex_token_count"],
                     "missing": r["missing_symbols"], "introduced": r["newly_introduced_symbols"],
                     "corruption": r["corruption_appeared"]} for r in trace]
                # pinpoint the first stage where a symbol present upstream goes missing
                for r in trace[1:]:
                    if r["missing_symbols"]:
                        logger.info("FORMULA TRACE: %d symbol(s) first missing at stage %s: %s",
                                    len(r["missing_symbols"]), r["stage"], r["missing_symbols"][:8])
                        break
        except Exception as exc:
            logger.debug("formula trace failed: %s", exc)
    path = _write_report(run_dir, topic, report, meta) if report else None
    if ctx is not None and report:
        ctx.last_research_report = report
        ctx.last_research_path = path
    prog.update("Complete", "cancelled" if cancelled else "done")
    logger.info("Deep research %s in %.1fs (%d sources, %d findings)",
                "cancelled" if cancelled else "complete", elapsed,
                prog.stats["sources"], prog.stats["findings"])
    return {"report": report, "path": path, "stats": prog.stats,
            "queries": queries, "cancelled": cancelled, "elapsed_sec": elapsed}

"""Stage 5 — turn the briefs into a document (the reduce step).

Two shapes, picked by the orchestrator:

* `synthesize_report` — single-pass. One LLM call writes the whole report from
  the (optionally consolidated) evidence, sized to the model's real context
  window and the depth's token budget.
* `synthesize_survey` — SURVEY MODE. The model first plans an outline from the
  evidence landscape, then writes each section in its own call as flowing prose
  that merges sources, and finally an abstract over the finished body. That is
  what produces a tens-of-pages document instead of one call's worth of text.

Both write in the reader's language (`out_lang`): the directive rides on the
system prompt so it is the model's last word, while searching, crawling and
briefing stay in English, where the sources are.
"""
import logging
import re
from typing import Optional

import dr_settings as S
from dr_calls import _EFFORT_FROM_CONFIG, _model_context, _think_call
from dr_brief import _brief_tag, _consolidate_evidence
from dr_collect import evidence_stats
from dr_extract import _fix_mojibake
from dr_lang import _lab, _lang_directive
from dr_outline import (_clean_report_title, _default_outline, _est_tokens,
                        _extract_outline_json, _outline_overview,
                        _repair_outline_text, _strip_leading_headings, _toc)
from dr_policy import _TRUST_ORDER
from dr_progress import _Progress
from dr_relevance import _term_in
from prompts import (OUTLINE_PLANNER_PROMPT, REPORT_SYNTHESIS_PROMPT,
                     SURVEY_SECTION_PROMPT)

logger = logging.getLogger("assistant.research")


# Phase 9 — detect analyses the user's prompt explicitly asked for, so synthesis
# adds them as extra sections instead of always emitting the fixed template.
_SECTION_TRIGGERS = {
    "Chronological Timeline": ("timeline", "chronolog", "historical evolution",
                               "lineage", "evolution from", "evolved", "history of"),
    "Common Misconceptions": ("misconception", "misunderstand", "incorrect explanation",
                              "myth", "common error", "false claim", "true or false",
                              "verify whether"),
    "Method Comparison": ("compare", "comparison", "versus", " vs ", "differ from",
                          "against the"),
    "Mathematical Derivation": ("derivation", "derive", "step-by-step", "proof",
                                "eigen-analysis", "equations"),
    "Citation Lineage": ("citation", "most influential", "most-cited", "cited by",
                         "influential papers", "citation graph", "citation counts"),
    "Bibliography": ("bibliography", "references list", "reference list", "full citations"),
}


def requested_sections(topic: str) -> list:
    """Return the human-readable names of analyses the prompt explicitly requests."""
    low = (topic or "").lower()
    out = []
    for name, keys in _SECTION_TRIGGERS.items():
        if any(k in low for k in keys):
            out.append(name)
    return out


def _confidence_rule(stats: dict) -> str:
    """A short instruction block telling synthesis how confidence MUST map to the
    evidence structure (so labels are derived, not invented)."""
    tc = stats["trust_counts"]
    mix = ", ".join(f"{tc[t]} {t.lower()}" for t in _TRUST_ORDER if tc[t]) or "none"
    return (
        "EVIDENCE STRUCTURE (use this to assign confidence, do not invent it):\n"
        f"- independent evidence clusters: {stats['independent_clusters']}\n"
        f"- strong (PRIMARY/SECONDARY) sources: {stats['strong_sources']}\n"
        f"- trust mix: {mix}\n"
        "CONFIDENCE RULES (apply strictly):\n"
        "- High: backed by >= 2 INDEPENDENT clusters AND at least one PRIMARY or "
        "SECONDARY source.\n"
        "- Medium: a single strong source, OR multiple sources that only partly agree.\n"
        "- Low: only COMMUNITY/LOW sources, a single weak source, or sources that "
        "merely copy each other (a cluster of copies counts as ONE).\n"
    )


def synthesize_report(ctx, topic: str, briefs: list, prog: _Progress,
                      max_tokens: Optional[int] = None,
                      sections: Optional[list] = None,
                      preamble: Optional[str] = None,
                      out_lang: str = "en",
                      effort=_EFFORT_FROM_CONFIG) -> str:
    """Reduce per-source briefs into one structured Markdown report.

    `briefs` items: {"domain", "url", "title", "brief"}. When the combined text
    exceeds one context window, briefs are first consolidated batch-by-batch.

    `effort` overrides the report-synthesis reasoning effort (None = model default,
    or low/medium/high). Default derives it from DR_REASONING_EFFORT. The caller uses
    this to run an effort LADDER: gpt-oss at high effort can spend its whole token
    budget reasoning and emit EMPTY content, so on empty we retry at lower effort."""
    combined = _consolidate_evidence(ctx, topic, briefs, prog)

    prog.update("Building report", "writing report")
    rule = _confidence_rule(evidence_stats(briefs))
    extra = ""
    if sections:
        extra = ("\nThe request explicitly asks for these analyses — include EACH "
                 "EXACTLY ONCE as its own top-level section with a plain title (do "
                 "NOT append '(Dedicated Section)' or similar, and do NOT also place "
                 "the same content as a numbered item inside Detailed Analysis): "
                 + "; ".join(sections) + ".\n")
        if "Citation Lineage" in sections:
            extra += (
                "You ARE given an `openalex.org` brief containing real works with "
                "their citation counts (cited_by=...), years, authors and venues. "
                "The 'Citation Lineage' section is MANDATORY and MUST be a Markdown "
                "table with columns Paper | Year | Authors | Citations | Role, listing "
                "the seminal work AND its principal descendants taken verbatim from "
                "that openalex.org brief. Do not omit the citation counts.\n")
    tl = (topic or "").lower()
    if "structure tensor" in tl or "di zenzo" in tl:
        extra += (
            "\nMATH CORRECTNESS: the image/gradient structure tensor is a 2x2 "
            "symmetric matrix [[Σ Ix^2, Σ Ix·Iy],[Σ Ix·Iy, Σ Iy^2]] formed by "
            "summing the per-channel 2x2 outer products OVER the channels. The "
            "number of channels (RGB, hyperspectral bands) does NOT change the "
            "matrix size — it stays 2x2. Do NOT describe it as an N×N or n×n "
            "matrix and do NOT claim it grows with channel count.\n")
    scaffold = f"{preamble}\n" if preamble else ""
    # Let the model think as much as it wants here, but don't let reasoning STARVE the
    # output: gpt-oss spends reasoning + output from one max_tokens pool, so add a
    # dedicated reasoning headroom on top of the report budget. Effort stays at the
    # user's setting unless the caller overrides `effort`; the llm-layer guard
    # guarantees raw CoT is never emitted as the report even if the model still runs
    # out of room.
    # BUGFIX: `max_tokens` (the caller's per-depth report budget / the manual
    # "Report budget" override) and DR_REASONING_HEADROOM used to be accepted and
    # then thrown away — send_budget was DR_SECTION_TOKEN_CEILING and nothing else,
    # so quick/standard/deep and the override knob were all the same number. They
    # are now the actual budget: report budget + reasoning headroom, still clamped
    # by the ceiling and by the real context window. At stock config
    # (8000 + 40000 vs a 32000 ceiling) the ceiling still wins, so the default
    # behaviour is byte-identical; lowering either knob now has an effect.
    # DR_REPORT_TOKENS (2000) was a tiny report on a 262k window, but the whole window
    # is not the answer either — an unbounded budget just lets a degenerate loop run
    # until the 600s wall-clock guard fires. DR_SECTION_TOKEN_CEILING is the generous
    # upper bound (think + prose); the model still stops on its own well before it.
    _ctxw = _model_context(ctx)
    _report_budget = int(max_tokens) if max_tokens else S.DR_REPORT_TOKENS_STANDARD
    send_budget = min(_ctxw, S.DR_SECTION_TOKEN_CEILING,
                      max(1024, _report_budget + S.DR_REASONING_HEADROOM))
    out_reserve = min(_ctxw // 2, send_budget)
    # Keep the single-pass prompt inside the context window: trim evidence to whatever
    # the window leaves after the output reservation. (_think_call also clamps
    # max_tokens, but the PROMPT itself must fit or LM Studio returns empty.)
    # The language directive rides on the SYSTEM prompt (last, so it is the final
    # word) and is also counted in the budget below — it is not free.
    ld = _lang_directive(out_lang)
    system = REPORT_SYNTHESIS_PROMPT + ld
    fixed = (system + rule + extra + scaffold
             + f"Research topic: {topic}\n\nPer-source briefs:\n")
    ev_budget = max(1500, (_ctxw - _est_tokens(fixed) - out_reserve - 120) * 4)
    if len(combined) > ev_budget:
        combined = combined[:ev_budget].rsplit("\n", 1)[0] + "\n…(evidence truncated to fit context)"
    user = (f"Research topic: {topic}\n\n{rule}{extra}\n{scaffold}Per-source briefs:\n{combined}"
            + ld)
    report = _think_call(ctx, system, user, max_tokens=send_budget, temperature=0.3,
                         effort=effort)
    return (report or "").strip()


# --------------------------------------------------------------------------- #
# Survey synthesis: dynamic outline -> section-by-section narrative document.
# The single-pass synthesize_report above caps the document at one LLM call's
# budget; to produce a coherent tens-of-pages review the outline is planned by
# the model from the evidence, then each section is written in its own call as
# flowing prose that merges sources (never "Source A says..."). See DR_SURVEY_*.
# --------------------------------------------------------------------------- #
def _evidence_landscape(briefs: list, max_titles: Optional[int] = None) -> str:
    """A compact view of the gathered evidence for the outline planner: trust-tier
    counts, the distinct domains, and the source titles (the planner designs the
    document structure from this, not from the full brief bodies). NO arbitrary cap by
    default — the caller (plan_report_outline) trims the whole landscape to a
    context-derived budget, so we list every domain/title and let that be the limiter."""
    by_tier: dict = {}
    for b in briefs:
        by_tier.setdefault(b.get("trust", "COMMUNITY"), []).append(b)
    lines = []
    counts = ", ".join(f"{t}={len(by_tier[t])}" for t in _TRUST_ORDER if by_tier.get(t))
    lines.append(f"Sources: {len(briefs)} across tiers ({counts}).")
    doms = sorted({b["domain"] for b in briefs})
    lines.append("Domains: " + ", ".join(doms))
    lines.append("\nSource titles (tagged by trust):")
    shown = 0
    for tier in _TRUST_ORDER:
        for b in by_tier.get(tier, []):
            if max_titles is not None and shown >= max_titles:
                break
            t = (b.get("title") or b["url"]).strip()
            lines.append(f"- [{tier}] {t} ({b['domain']})")
            shown += 1
    return "\n".join(lines)


def _normalize_outline(plan: dict, topic: str) -> dict:
    """Validate/coerce a planner result into a usable outline (clamp section count,
    drop empty headings, guarantee a title). Falls back to the default skeleton."""
    if not isinstance(plan, dict):
        return _default_outline(topic)
    secs_in = plan.get("sections")
    if not isinstance(secs_in, list) or not secs_in:
        return _default_outline(topic)
    sections = []
    for s in secs_in:
        if not isinstance(s, dict):
            continue
        head = str(s.get("heading", "")).strip().lstrip("#").strip()
        if not head:
            continue
        subs = s.get("subsections")
        subs = [str(x).strip() for x in subs if str(x).strip()] if isinstance(subs, list) else []
        sections.append({"heading": head, "subsections": subs[:4],
                         "focus": str(s.get("focus", "")).strip()})
        if len(sections) >= S.DR_SURVEY_MAX_SECTIONS:
            break
    if not sections:
        return _default_outline(topic)
    title = str(plan.get("title", "")).strip() or _clean_report_title(topic)
    return {
        "title": title,
        "abstract_focus": str(plan.get("abstract_focus", "")).strip(),
        "include_math": bool(plan.get("include_math", True)),
        "include_history": bool(plan.get("include_history", True)),
        "include_applications": bool(plan.get("include_applications", True)),
        "include_open_problems": bool(plan.get("include_open_problems", True)),
        "sections": sections,
    }


def plan_report_outline(ctx, topic: str, briefs: list, prog: _Progress,
                        out_lang: str = "en") -> dict:
    """Ask the LLM to DESIGN the document's structure from the evidence landscape
    before any prose is written (title + ordered sections + subsections + focus).
    Returns a normalized outline dict; degrades to a generic skeleton on failure."""
    prog.update("Building report", "planning document outline")
    landscape = _evidence_landscape(briefs)
    # Keep the planner prompt well inside the window so the model has room to think
    # AND emit the JSON plan (leave ~half the context for output).
    land_budget = max(1200, (_model_context(ctx) // 2 - _est_tokens(OUTLINE_PLANNER_PROMPT)) * 4)
    if len(landscape) > land_budget:
        landscape = landscape[:land_budget].rsplit("\n", 1)[0] + "\n…(more sources omitted)"
    # Let the model reason about the best structure, then emit the JSON plan
    # (force_think; reasoning models need room to think AND emit the plan).
    # The planner's strings become the H1, the TOC and every section heading, so they
    # must be in the reader's language — but the JSON KEYS are the contract with
    # _normalize_outline and must stay English, or the plan parses to nothing.
    plan_lang = _lang_directive(out_lang)
    if plan_lang:
        plan_lang += ("The JSON KEYS (title, sections, heading, focus, subsections, "
                      "abstract_focus) stay EXACTLY as specified, in English. Only "
                      "their VALUES are written in the output language.\n")
    raw = _think_call(
        ctx, OUTLINE_PLANNER_PROMPT + plan_lang,
        f"Research topic:\n{topic}\n\nEVIDENCE LANDSCAPE:\n{landscape}" + plan_lang,
        max_tokens=S.DR_PLAN_TOKENS, temperature=0.4)
    plan = _normalize_outline(_extract_outline_json(raw), topic)
    logger.info("OUTLINE planned: %d sections — %s", len(plan["sections"]), plan["title"])
    return plan


def _section_terms(sec: dict) -> set:
    blob = f"{sec.get('heading', '')} {sec.get('focus', '')} {' '.join(sec.get('subsections', []))}"
    return set(re.findall(r"[a-zа-яё0-9]{4,}", blob.lower()))


def _select_section_evidence(briefs: list, sec: dict, char_budget: int) -> str:
    """Pick the briefs most relevant to THIS section (lexical overlap with its
    heading/focus/subsections, tie-broken by trust) and pack their verbatim-tagged
    text up to `char_budget`. On a small-context model this is what keeps the section
    prompt inside the window AND gives each section the evidence it actually needs,
    instead of one truncated blob shared by every section. Always returns >=1 brief."""
    terms = _section_terms(sec)
    tw = {"PRIMARY": 1.5, "SECONDARY": 1.0, "COMMUNITY": 0.3, "LOW": 0.0}

    def score(b):
        hay = f"{b.get('title', '')} {b.get('brief', '')}".lower()
        overlap = sum(1 for t in terms if _term_in(t, hay))
        has_eq = 1.0 if (b.get("equations")) else 0.0
        return overlap + tw.get(b.get("trust", "COMMUNITY"), 0.3) + has_eq
    ranked = sorted(briefs, key=score, reverse=True)
    out, used = [], 0
    for b in ranked:
        tag = _brief_tag(b)
        if out and used + len(tag) > char_budget:
            continue
        out.append(tag)
        used += len(tag) + 2
        if used >= char_budget:
            break
    if not out:                       # budget smaller than one brief: take the best, trimmed
        out = [_brief_tag(ranked[0])[:max(800, char_budget)]]
    return "\n\n".join(out)


def synthesize_survey(ctx, topic: str, briefs: list, prog: _Progress,
                      outline: Optional[dict] = None,
                      section_tokens: Optional[int] = None,
                      out_lang: str = "en") -> str:
    """Produce a coherent long-form scientific document: plan the outline (unless
    one is supplied), then write EACH section in its own LLM call as flowing,
    source-merged prose. Assembles title + abstract + TOC + sections. Returns the
    body WITHOUT the run header / sources appendix (added by the caller), or "" if
    no section produced content (caller then falls back to single-pass/digest).

    CONTEXT-AWARE: the whole call (system + outline + evidence + think + prose) must
    fit DR_MODEL_CONTEXT, or LM Studio rejects the prompt ('n_keep >= n_ctx') and
    returns empty. So evidence is trimmed per section and max_tokens is computed from
    the remaining context; on an empty section we SHRINK evidence and retry, which
    also auto-adapts when the model is loaded with a smaller context than configured."""
    if not briefs:
        return ""
    outline = outline or plan_report_outline(ctx, topic, briefs, prog, out_lang)
    _repair_outline_text(outline)   # the planner strings (title/headings/focus) can carry
    # mojibake too (e.g. a double-encoded U+202F in the title);
    # repair once here so the H1 and TOC aren't garbled.
    rule = _confidence_rule(evidence_stats(briefs))
    overview = _outline_overview(outline)

    # NO artificial length limit: budgets derive from the model's ACTUAL context.
    # The model writes each section as long as it wants (force_think) and stops on
    # its own; we only ensure the prompt + output fit the physical window. With a
    # large context (e.g. 262k) every section sees ALL the evidence and may run to
    # many thousands of words.
    ctx_tokens = _model_context(ctx)
    # The language directive is sent TWICE per section (system + user), so it must be
    # counted here or the evidence budget overruns the window on a non-English run.
    overhead_tok = (_est_tokens(SURVEY_SECTION_PROMPT) + _est_tokens(overview)
                    + _est_tokens(rule) + 2 * _est_tokens(_lang_directive(out_lang))
                    + 220)
    # Evidence budget = essentially the whole window minus a generous slice reserved
    # for the model's think-block + prose. On a big context this includes every brief.
    out_reserve = max(S.DR_SURVEY_SECTION_TOKENS, (ctx_tokens - overhead_tok) // 3)
    base_ev_chars = max(2000, (ctx_tokens - overhead_tok - out_reserve - 96) * 4)

    written = []
    n = len(outline["sections"])
    for i, sec in enumerate(outline["sections"], 1):
        if ctx is not None and ctx.is_cancelled():
            break
        prog.update("Building report", f"writing section {i}/{n}: {sec['heading']}")
        subs = ("\nSubsections to cover (use ### for each): "
                + "; ".join(sec["subsections"]) + "." if sec.get("subsections") else "")
        where = ("This is the OPENING section." if i == 1 else
                 "This is the CLOSING section." if i == n else
                 f"This is section {i} of {n}.")
        body = ""
        ev_chars = base_ev_chars
        ld = _lang_directive(out_lang)          # same directive every attempt below
        sec_system = SURVEY_SECTION_PROMPT + ld
        # Up to 3 attempts, shrinking evidence each time, so even a mis-detected /
        # too-small context still yields a real section instead of empty.
        for attempt in range(3):
            evidence = _select_section_evidence(briefs, sec, ev_chars)
            user = (f"Document title: {outline['title']}\n\n"
                    f"FULL OUTLINE (for transitions/context — write ONLY your section):\n{overview}\n\n"
                    f"SECTION TO WRITE NOW (use ## for its heading): {sec['heading']}\n"
                    f"Purpose: {sec.get('focus', '')}\n{where}{subs}\n"
                    f"Write this section in depth and at whatever length the evidence "
                    f"supports — do not self-truncate for brevity.\n\n"
                    f"{rule}\nEVIDENCE (per-source briefs):\n{evidence}")
            user += ld
            prompt_tok = _est_tokens(sec_system) + _est_tokens(user)
            # Hand the model ALL remaining context for think + prose — no length cap.
            max_out = max(1024, ctx_tokens - prompt_tok - 96)
            body = _think_call(ctx, sec_system, user, max_out,
                               temperature=0.35 + 0.05 * attempt)
            if body:
                break
            ev_chars = int(ev_chars * 0.5)    # shrink and retry
            logger.warning("section %r empty (attempt %d) — shrinking evidence to %d chars",
                           sec["heading"], attempt + 1, ev_chars)
        if body:
            if not body.lstrip().startswith("#"):
                body = f"## {sec['heading']}\n\n{body}"
            written.append(_fix_mojibake(body) or body)

    if not written:
        return ""

    body_text = "\n\n".join(written)
    abstract = _strip_leading_headings(
        _synthesize_abstract(ctx, outline, body_text, out_lang))

    parts = [f"# {outline['title']}", ""]
    if abstract:
        parts += [f"## {_lab(out_lang, 'abstract')}", "", abstract, ""]
    parts += [_toc(outline, out_lang), "", body_text]
    return "\n".join(parts).strip()


def _synthesize_abstract(ctx, outline: dict, body_text: str,
                         out_lang: str = "en") -> str:
    """Write a REAL abstract for the finished document. The outline's `abstract_focus`
    is only a one-sentence DIRECTIVE from the planner ("what the abstract should
    claim") — emitting it verbatim as the Abstract (the old behavior) gave a fake,
    one-line abstract. Instead summarize the actual assembled body into a proper
    150–250 word academic abstract. Falls back to the focus line if the call fails."""
    focus = (_fix_mojibake(outline.get("abstract_focus", "")) or
             outline.get("abstract_focus", "")).strip()
    if not body_text.strip():
        return focus
    # Trim the body to fit the window (abstract only needs the gist, not every word).
    ctx_tokens = _model_context(ctx)
    budget_chars = max(4000, (ctx_tokens * 2 // 3) * 4)
    src = body_text if len(body_text) <= budget_chars else (
        body_text[:budget_chars].rsplit("\n", 1)[0])
    system = ("You are writing the ABSTRACT of a scientific review article. Given the "
              "document's title, the abstract's intended focus, and the full body, write "
              "a single self-contained abstract of 150–250 words: state the subject, what "
              "the article establishes, the key results/structure, and why it matters. "
              "Flowing prose, no headings, no bullet points, no citations, no "
              "'this article' meta-talk beyond what an abstract normally has. Reproduce any "
              "essential equation inline in $...$ exactly. Output ONLY the abstract text."
              + _lang_directive(out_lang))
    user = (f"Title: {outline.get('title', '')}\n\nIntended focus: {focus}\n\n"
            f"FULL BODY:\n{src}")
    # No length cap — give it the whole remaining window and let it stop naturally
    # (an abstract is short by nature, but we don't impose an arbitrary ceiling).
    overhead = _est_tokens(system) + _est_tokens(user) + 96
    max_out = max(1024, ctx_tokens - overhead)
    try:
        out = _think_call(ctx, system, user, max_out, temperature=0.3)
    except Exception as exc:
        logger.warning("abstract synthesis failed (%s) — using focus line", exc)
        out = ""
    out = (_fix_mojibake(out) or out or "").strip()
    return out or focus


def _postprocess_report(md: str, drop_substrs: tuple = ()) -> str:
    """Drop duplicate sections (same normalized header kept once) and any section
    whose header contains a drop substring. Removing a header removes its body up
    to the next header. Guarantees no duplicated sections in the final report."""
    out, seen, skipping = [], set(), False
    for ln in md.split("\n"):
        m = re.match(r"^(#{1,3})\s+(.+?)\s*$", ln)
        if m:
            # Strip leading "6." / "6)" numbering so "### 6. Common Misconceptions"
            # and "# Common Misconceptions" normalize identically and dedupe.
            title = re.sub(r"^\s*\d+\s*[.)]\s*", "", m.group(2))
            # Drop a trailing parenthetical so "X" and "X (Dedicated Section)"
            # normalize identically (the model titles dups this way to satisfy a
            # "make it a dedicated section" instruction while also keeping X inline).
            title = re.sub(r"\s*\([^)]*\)\s*$", "", title)
            norm = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
            if norm in seen or any(d in norm for d in drop_substrs):
                skipping = True
                continue
            seen.add(norm)
            skipping = False
            out.append(ln)
        elif not skipping:
            out.append(ln)
    return "\n".join(out).strip()

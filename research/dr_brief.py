"""Stage 4a — turn a crawled page into a per-source brief (the map step).

`brief_source` runs the brief model over one page and returns its findings with
a self-rated trust tier (`_parse_trust`); `_consolidate_evidence` is the LOSSY
reduce that only runs when the briefs cannot fit the model's real context window
— with a large window the raw briefs are fed verbatim, so no fact and no
equation is ever compressed away needlessly.

Note for suites: the model contact has ONE home, `dr_calls.call_llm_simple`.
`brief_source` reaches it through that module rather than binding its own copy,
so a test that fakes the brief model patches `dr_calls.call_llm_simple` and the
fake is in force for every stage — not just whichever one happened to import it.
"""
import logging
import os
import re
from typing import Optional

import dr_settings as S
import dr_calls
from dr_calls import _model_context, _think_call
from dr_math import _repair_latex_artifacts
from dr_outline import _est_tokens
from dr_progress import _Progress
from prompts import SOURCE_BRIEF_PROMPT

logger = logging.getLogger("assistant.research")


# Lightweight reduce step used only when there are too many briefs to fit one
# synthesis call — consolidates a batch of briefs into dense grouped notes.
_CONSOLIDATE_PROMPT = """You compress research briefs without losing facts.

You are given several per-source briefs (each tagged with a source domain and a trust tag PRIMARY/SECONDARY/COMMUNITY/LOW) about one topic. Merge them into a dense set of grouped notes: keep every distinct fact, number, name, version, formula, benchmark and claim, and after each fact keep BOTH the source domain and its trust tag in parentheses, e.g. "(arxiv.org, PRIMARY)". Preserve any "(claim)"/"(suspect)" markers. Drop only pure repetition, but note when several independent sources agree. MATH FIDELITY: copy every formula/equation EXACTLY as written — do not paraphrase, simplify, re-derive, or drop a backslash; keep inline math in $...$ and displayed equations in $$...$$, preserving all LaTeX commands (\\frac, \\sum, \\int, \\begin{pmatrix}...), subscripts/superscripts, Greek letters and Unicode math symbols (∑ ∫ √ π ∇) verbatim; never invent or "correct" math. Plain text bullets starting with "- " (math may use $...$/$$...$$). No headings, no preamble."""


# --------------------------------------------------------------------------- #
# Stage 4 — per-source briefs (map) + synthesis (reduce)
# --------------------------------------------------------------------------- #
_TRUST_RE = re.compile(r"^\s*SOURCE:\s*(PRIMARY|SECONDARY|COMMUNITY|LOW)\b.*$",
                       re.IGNORECASE | re.MULTILINE)


def _parse_trust(brief: str) -> tuple:
    """Pull the brief model's `SOURCE: <tier>` self-rating off the first line.
    Returns (tier, body_without_the_tag_line). Defaults to COMMUNITY when the
    model omitted the tag, so an unrated source is never silently treated as
    authoritative."""
    m = _TRUST_RE.search(brief)
    if not m:
        return "COMMUNITY", brief
    tier = m.group(1).upper()
    body = (brief[:m.start()] + brief[m.end():]).strip()
    return tier, body


def brief_source(ctx, topic: str, page: dict) -> Optional[str]:
    """Extract grounded facts from one page.

    Returns the brief, or None when the model read the page and found nothing
    useful, or "" when the model produced no answer at all. The last case is a
    failure on OUR side and is counted separately -- a report that blames the
    sources for our own outage is worse than no report.
    """
    # Cap what the model is asked to read. A reasoning model deliberates in
    # proportion to its input, and this one has no brake (the <think></think>
    # prefill below is dropped for Gemma), so the full 32k page produced 17-20k
    # characters of thinking and NO brief. Measured, 3 runs each: 32000 chars ->
    # 0/3 usable at 474s median; 4000 chars -> 2/3 usable at 225s. Equations are
    # extracted from the FULL text at collection time, so nothing mathematical is
    # lost by reading less here.
    _text = page.get("text") or ""
    if S.DR_BRIEF_INPUT_CHARS and len(_text) > S.DR_BRIEF_INPUT_CHARS:
        # The 4000 characters the briefer reads are the ones about the topic,
        # not the page's first 4000 -- a page is fetched up to 32000 and its
        # head is menus and lede as often as not (search.best_passages).
        from search import best_passages
        _text = best_passages(_text, f"{topic} {page.get('title', '')}", S.DR_BRIEF_INPUT_CHARS)
        logger.debug("brief input trimmed %d -> %d chars for %s",
                     len(page.get("text") or ""), len(_text), page.get("domain"))
    user = (f"Research topic: {topic}\n\nSource domain: {page['domain']}\n"
            f"Page title: {page.get('title', '')}\n\nPage text:\n{_text}")
    # DR_BRIEF_TOKENS (320) truncated dense extractions mid-bullet — a formula-heavy
    # page can't fit its facts + verbatim equations in 320 tokens, so we were LOSING
    # evidence at the map step. Give it real room, but BOUNDED: this call runs once
    # per source, so handing over the remaining window (~250k on a 262k model) lets a
    # single degenerate loop burn the 600s wall-clock guard 20-40 times per report.
    room = _model_context(ctx) - _est_tokens(SOURCE_BRIEF_PROMPT) - _est_tokens(user) - 80
    safe = max(S.DR_BRIEF_TOKENS, min(S.DR_BRIEF_TOKEN_CEILING, room))
    # Optionally run this one call on a smaller model (see DR_BRIEF_MODEL).
    # NB the prefill below is a no-op on Gemma — llm.py strips it — which is why
    # the input cap above, not the prefill, is what actually bounds this step.
    _c = ctx
    if S.DR_BRIEF_MODEL and getattr(ctx, "model_name", "") != S.DR_BRIEF_MODEL:
        import copy as _copy
        _c = _copy.copy(ctx)
        _c.model_name = S.DR_BRIEF_MODEL
    out = dr_calls.call_llm_simple(_c, SOURCE_BRIEF_PROMPT, user,
                          temperature=0.2, max_tokens=safe, prefill="<think></think>")
    out = (out or "").strip()
    # Two different outcomes used to share one return value, and a counter built
    # on it could not tell an OUTAGE from a page that genuinely had nothing:
    #   ""   -- the model said nothing at all (empty completion): our failure.
    #   None -- the model read the page and judged it useless: the page's.
    # Both are falsy, so every existing caller behaves as before; the one that
    # cares checks `is None`.
    if not out:
        return ""
    if "NO USEFUL CONTENT" in out.upper():
        return None
    return out


def _pack_batches(entries: list, limit: int) -> list:
    """Greedily pack brief strings into batches under a char limit."""
    batches, cur, cur_len = [], [], 0
    for e in entries:
        if cur and cur_len + len(e) > limit:
            batches.append(cur)
            cur, cur_len = [], 0
        cur.append(e)
        cur_len += len(e)
    if cur:
        batches.append(cur)
    return batches


def _brief_tag(b) -> str:
    cs = b.get("cluster_size", 1)
    corro = f" | copies={cs}" if cs > 1 else ""
    # Every field is read defensively. This runs during final synthesis, after the
    # whole crawl: a brief missing a key — a source adapter that never set
    # `domain`, or state resumed from a run that predates a schema change — used
    # to raise KeyError and destroy an entire completed report at the last step.
    # A source with a blank domain is worth far more than no report at all.
    head = (f"[Source: {b.get('domain') or 'unknown'} | trust={b.get('trust', 'COMMUNITY')}"
            f"{corro}] {b.get('title') or 'untitled'}\n{b.get('brief') or ''}")
    # Attach the VERBATIM equations preserved from this source (deterministic
    # extraction, not the LLM brief body which often drops/mangles math). This is
    # what lets the section writer INTEGRATE exact formulas into the narrative —
    # introduce, define, explain — instead of only the appendix carrying them.
    # Sanitize each preserved equation so layout primitives that leaked from ar5iv
    # (\vskip/\cr/glue) don't reach the model and get echoed into the prose.
    eqs = [_repair_latex_artifacts(e).strip() for e in (b.get("equations") or []) if e and e.strip()]
    if eqs:
        # NO fixed cap on equations per source (was [:12] — a formula-dense paper has
        # far more, and capping dropped them from the narrative). Downstream context
        # fitting (_select_section_evidence char budget, _consolidate_evidence) trims
        # the whole tag to fit the window, so this never overflows; it only stops
        # losing math the writer could integrate.
        head += "\n  Verbatim equations from this source (reproduce exactly):\n"
        head += "\n".join(f"  {e}" for e in eqs)
    return head


def _consolidate_evidence(ctx, topic: str, briefs: list, prog: _Progress) -> str:
    """Tagged briefs joined into one evidence string; if it exceeds one context
    window, compress batch-by-batch first (mechanical, low effort). Shared by both
    the single-pass and the section-by-section survey synthesis paths."""
    entries = [_brief_tag(b) for b in briefs]
    combined = "\n\n".join(entries)
    # Only pre-compress when the evidence genuinely won't fit a synthesis call in the
    # model's ACTUAL window (reserve ~1/3 for think + output). With a large context
    # (e.g. 262k) this never triggers — the raw briefs are fed verbatim, avoiding a
    # lossy compression pass that can drop facts or mangle equations. An explicit
    # DR_MAX_REDUCE_CHARS env value (if set) only LOWERS the threshold, never the cap.
    reduce_chars = max(6000, (_model_context(ctx) * 2 // 3) * 4)
    _env_cap = os.getenv("DR_MAX_REDUCE_CHARS")
    if _env_cap and _env_cap.isdigit():
        reduce_chars = min(reduce_chars, int(_env_cap))
    if len(combined) <= reduce_chars:
        return combined
    prog.update("Building report", "consolidating sources")
    batches = _pack_batches(entries, reduce_chars)
    notes = []
    for i, batch in enumerate(batches, 1):
        if ctx is not None and ctx.is_cancelled():
            break
        prog.update("Building report", f"consolidating batch {i}/{len(batches)}")
        part = _think_call(
            ctx, _CONSOLIDATE_PROMPT,
            f"Topic: {topic}\n\nBriefs:\n" + "\n\n".join(batch),
            max_tokens=S.DR_PLAN_TOKENS,   # consolidation notes; NOT the full window
            temperature=0.2)
        if part and part.strip():
            notes.append(part.strip())
    return "\n\n".join(notes)[:reduce_chars * 2]

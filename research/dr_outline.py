"""Report skeleton: the outline, the table of contents, the title.

The STRUCTURE of the finished document, separate from the evidence that fills it
and from the LLM calls that write it: parse the planner's outline JSON, fall back
to a sane default when it returns nothing usable, repair mojibake in its strings,
render the overview and the table of contents, clean the report title, and the
small text utilities the assembler uses around them.

Split out of deep_research.py. Every function here is pure. Note what is NOT
here: `_normalize_outline` reads DR_SURVEY_MAX_SECTIONS and `synthesize_survey`
reads DR_SURVEY_SECTION_TOKENS, and both of those are manual-control knobs that
`deep_research.apply_overrides` swaps in that module's globals for the duration
of a run. Moving them would bind the knob at import time and silently ignore
every override, so they stay where the globals live.
"""
import re
from typing import Optional

from dr_extract import _fix_mojibake
from dr_lang import _lab
from utils import safe_json_from_llm


def _extract_outline_json(text: str) -> Optional[dict]:
    """Best-effort parse of the planner's JSON (tolerates code fences / stray prose)."""
    if not text:
        return None
    # Was: first '{' to last '}'. A reply with a brace after the object — trailing
    # prose, or the duplicate ```json block these models append — spanned two
    # objects, failed to parse, and silently dropped the planner's real outline in
    # favour of _default_outline. Verified against both shapes.
    return safe_json_from_llm(text, required_keys=("sections", "title"))


def _default_outline(topic: str) -> dict:
    """Generic survey skeleton used when the planner LLM fails or returns junk —
    keeps the section-by-section path working without a model-designed plan."""
    title = _clean_report_title(topic)
    return {
        "title": title,
        "abstract_focus": f"A synthesis of the current understanding of {title}.",
        "include_math": True, "include_history": True,
        "include_applications": True, "include_open_problems": True,
        "sections": [
            {"heading": "Introduction", "subsections": [],
             "focus": "Motivate the topic and outline the document."},
            {"heading": "Background and Foundations", "subsections": [],
             "focus": "Establish the concepts the rest of the document relies on."},
            {"heading": "Core Concepts and Methods", "subsections": [],
             "focus": "The substantive technical content, with equations integrated."},
            {"heading": "Comparison and Alternative Approaches", "subsections": [],
             "focus": "Set competing formulations side by side."},
            {"heading": "Contradictions and Open Questions", "subsections": [],
             "focus": "Surface disagreements and unresolved issues."},
            {"heading": "Applications", "subsections": [],
             "focus": "Where and how the ideas are used in practice."},
            {"heading": "Limitations and Future Directions", "subsections": [],
             "focus": "Honest limits and promising directions."},
            {"heading": "Conclusion", "subsections": [],
             "focus": "Tie the narrative together."},
        ],
    }


def _repair_outline_text(outline: dict) -> None:
    """In-place mojibake repair of the planner-generated strings (title, abstract
    focus, section headings/focus/subsections). These come straight from the LLM and
    can carry the same UTF-8-as-latin1 corruption the section bodies get — e.g. a
    double-encoded U+202F narrow no-break space turning 'Di Zenzo' into 'Diâ¯Zenzo'
    in the H1. Section bodies are already repaired on assembly; this covers the rest."""
    if not isinstance(outline, dict):
        return
    for k in ("title", "abstract_focus"):
        if outline.get(k):
            outline[k] = _fix_mojibake(outline[k]) or outline[k]
    for s in outline.get("sections", []):
        for k in ("heading", "focus"):
            if s.get(k):
                s[k] = _fix_mojibake(s[k]) or s[k]
        if s.get("subsections"):
            s["subsections"] = [_fix_mojibake(x) or x for x in s["subsections"]]


def _outline_overview(plan: dict) -> str:
    """One-line-per-section overview of the whole plan, handed to each section writer
    so it knows what the other sections cover (enables transitions, avoids overlap)."""
    lines = []
    for i, s in enumerate(plan["sections"], 1):
        subs = f" — subsections: {', '.join(s['subsections'])}" if s.get("subsections") else ""
        lines.append(f"{i}. {s['heading']}{subs}")
    return "\n".join(lines)


def _toc(plan: dict, out_lang: str = "en") -> str:
    """Markdown table of contents from the section headings (+ subsections)."""
    out = [f"## {_lab(out_lang, 'toc')}", ""]
    for i, s in enumerate(plan["sections"], 1):
        out.append(f"{i}. {s['heading']}")
        for sub in s.get("subsections", []):
            out.append(f"    - {sub}")
    return "\n".join(out)


def _est_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token for English+LaTeX). Used to keep each
    section call inside the model's context window without a tokenizer dependency."""
    return max(1, len(text or "") // 4)


def _strip_leading_headings(text: str) -> str:
    """Drop Markdown headings the model put at the START of a fragment.

    The abstract prompt says "no headings" and the model added them anyway, so the
    finished document opened with the title and "## Аннотация" printed TWICE — the
    assembler adds them, and the abstract text repeated them. A prompt cannot be
    relied on for a structural invariant; strip it instead.
    """
    lines = (text or "").splitlines()
    i = 0
    while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith("#")):
        i += 1
    return "\n".join(lines[i:]).strip()


def _clean_report_title(topic: str, max_len: int = 90) -> str:
    """A human report title from the (often messy) topic. When the topic is a boolean
    search query — "...from(\"Di Zenzo\" OR \"color tensor\") AND (...)" — the whole
    string is a terrible H1. Pick the most specific quoted phrase as the subject;
    otherwise clean + truncate the plain topic."""
    t = (topic or "").strip()
    quoted = re.findall(r'"([^"]+)"', t)
    if quoted:
        # The longest quoted phrase is almost always the precise subject.
        title = max((p.strip() for p in quoted), key=len)
    else:
        title = re.sub(r"\s+", " ", t)
    title = title.strip()
    return (title[:max_len].rstrip() + "…") if len(title) > max_len else title

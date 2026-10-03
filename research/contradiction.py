"""Contradiction search — a FIRST-CLASS pass that tries to disprove the run's own
provisional findings, kept logically separate from confirmation search (spec A).

Confirmation search asks "what supports X?"; this asks "what refutes X, what are
its limitations, failure cases, negative results, reproducibility issues, benchmark
anomalies, issue-tracker complaints, maintainer caveats?". The queries are
deterministic templates (offline-testable); the resulting briefs are TAGGED so they
are stored, counted, and surfaced separately — never silently merged into the
confirming evidence.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Sequence

logger = logging.getLogger("assistant.contradiction")

# Marker placed on briefs gathered by the contradiction pass (brief["mode"]).
MODE_CONFIRM = "confirm"
MODE_CONTRADICT = "contradict"

# Generic angles of disagreement — no domain hardcoding. {topic} = research core.
_CONTRADICTION_TEMPLATES = (
    "{topic} limitations",
    "{topic} criticism OR problems OR flaws",
    "{topic} failure cases OR does not work",
    "{topic} negative results OR debunked",
    "{topic} reproducibility issues OR could not reproduce",
    "{topic} benchmark anomalies OR misleading results",
    "{topic} issues OR bug reports OR complaints",
    "{topic} caveats OR known issues OR maintainer notes",
)

# Extra entity-anchored contradiction angles when concrete entities are known.
_ENTITY_TEMPLATES = (
    "{e} limitations OR criticism",
    "{e} bug OR issue OR regression",
    "{e} benchmark dispute OR overstated",
)


# Conversational filler that carries no search signal. A "topic" made only of these
# cannot anchor a contradiction query.
_FILLER = frozenset("""
я мне мы ты вы он она они хочу хочется нужно надо дай дайте покажи найди ищи search
как что чтобы который которые where what how i want need give me show find please
the a an of on in to for with and or but is are был была быть есть еть только just
ссылки ссылок ссылка link links пожалуйста spb
""".split())


def _usable_core(topic: str) -> str:
    """Distil `topic` into a search anchor, or "" when it carries no signal.

    Keeps word order (phrases beat bags of words for retrieval) and drops pure
    filler. Returns "" when nothing contentful survives — the caller then skips the
    pass rather than issuing template-only queries.
    """
    words = [w for w in re.findall(r"[^\W\d_]+", (topic or ""), re.UNICODE)
             if len(w) > 2 and w.lower() not in _FILLER]
    return " ".join(words[:8])


def contradiction_queries(topic: str, *, entities: Optional[Sequence[dict]] = None,
                          max_queries: int = 4, visited: Optional[set] = None) -> List[str]:
    """Build a bounded, de-duplicated set of contradiction-seeking queries.

    `topic` MUST be the distilled core topic, not the raw user request. The templates
    below append English angle words ("limitations", "benchmark anomalies", …), so a
    raw conversational topic turns each query into noise plus one strong English term
    and the search engine matches on that term ALONE. Observed for real: the first 8
    words of a Russian request about bus tours produced "…limitations" → a university
    writing guide on study limitations, a federal acquisition regulation, and SQL
    Server release notes; "…benchmark anomalies" → the Deno runtime changelog. Those
    then counted as "10 strong contradicting sources" in the report.

    `entities` is an optional ranked list of {"value":..} (from entities.rank()) so
    the pass can target the concrete things the topic is about. `visited` (lowercased
    queries already issued) is read AND updated so this never repeats earlier work."""
    visited = visited if visited is not None else set()
    core = _usable_core(topic)
    if not core:
        # No usable anchor: every query would be an unanchored English angle word.
        # Gathering nothing is strictly better than gathering unrelated pages and
        # reporting them as disconfirmation.
        logger.info("contradiction_queries: skipped — no usable core topic (%r)",
                    (topic or "")[:60])
        return []
    out: List[str] = []

    def _push(q: str) -> bool:
        q = q.strip()
        key = q.lower()
        if not q or key in visited:
            return False
        visited.add(key)
        out.append(q)
        return True

    for t in _CONTRADICTION_TEMPLATES:
        if len(out) >= max_queries:
            break
        _push(t.format(topic=core))

    # Interleave one entity-anchored contradiction query if budget remains.
    for ent in (entities or []):
        if len(out) >= max_queries:
            break
        _push(_ENTITY_TEMPLATES[0].format(e=ent.get("value", "")))

    logger.info("contradiction_queries: %d queries (topic=%r)", len(out), core[:50])
    return out[:max_queries]


def tag_confirming(briefs: Sequence[dict]) -> None:
    """Mark first-pass briefs as confirmation evidence (in place) if untagged."""
    for b in briefs:
        b.setdefault("mode", MODE_CONFIRM)


def tag_contradicting(briefs: Sequence[dict]) -> None:
    """Mark briefs gathered by the contradiction pass (in place)."""
    for b in briefs:
        b["mode"] = MODE_CONTRADICT


def partition_evidence(briefs: Sequence[dict]) -> tuple:
    """Split a brief set into (confirming, contradicting) by their mode tag."""
    confirm = [b for b in briefs if b.get("mode", MODE_CONFIRM) != MODE_CONTRADICT]
    contradict = [b for b in briefs if b.get("mode") == MODE_CONTRADICT]
    return confirm, contradict


def contradiction_summary(briefs: Sequence[dict]) -> dict:
    """Aggregate the contradiction evidence into a small, auditable signal block.

    `strength` ∈ {none, weak, moderate, strong} reflects how much *credible*
    disagreement was found — strength rises with count AND source trust, so a single
    forum gripe stays 'weak' while two PRIMARY/SECONDARY refutations read 'strong'."""
    _, contra = partition_evidence(briefs)
    n = len(contra)
    strong = sum(1 for b in contra if b.get("trust") in ("PRIMARY", "SECONDARY"))
    domains = sorted({b.get("domain", "") for b in contra if b.get("domain")})
    if n == 0:
        strength = "none"
    elif strong >= 2:
        strength = "strong"
    elif strong >= 1 or n >= 2:
        strength = "moderate"
    else:
        strength = "weak"
    return {"count": n, "strong_sources": strong, "domains": domains,
            "strength": strength, "has_contradictions": n > 0}

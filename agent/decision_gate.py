"""Abstain / ask / answer gate (spec D).

Confidence was already *computed* upstream (evidence_stats / coverage_signals /
contradiction_summary) but only advised synthesis. This module turns those signals
into an explicit decision taken BEFORE synthesis:

    ANSWER   — enough evidence to be useful (optionally flagged uncertain)
    ASK      — the request is ambiguous; ask one clarifying question first
    ABSTAIN  — evidence too weak to answer reliably; say so plainly

The decision is a pure function of the signal dicts (fully unit-testable) and
returns a fully-reasoned record so the choice is auditable in logs and the report.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

logger = logging.getLogger("assistant.decision_gate")

ANSWER = "ANSWER"
ASK = "ASK"
ABSTAIN = "ABSTAIN"

_CONTENT_STOP = {"the", "a", "an", "of", "for", "to", "and", "or", "in", "on",
                 "is", "are", "what", "how", "why", "who", "when", "where",
                 "about", "tell", "me", "give", "find", "research", "explain"}


def _content_words(topic: str) -> List[str]:
    toks = re.findall(r"[A-Za-zА-Яа-яЁё0-9][\w-]*", (topic or "").lower())
    return [t for t in toks if t not in _CONTENT_STOP and len(t) > 1]


def assess_ambiguity(topic: str, *, has_entities: bool = True) -> dict:
    """Heuristic ambiguity check. Conservative: only an empty/near-empty topic with
    no concrete entity is treated as ambiguous, so the gate asks rather than guesses
    on genuinely under-specified requests — but does not nag on normal questions."""
    words = _content_words(topic)
    reasons = []
    if len(words) == 0:
        reasons.append("empty or contentless request")
    elif len(words) == 1 and not has_entities and len(words[0]) < 4:
        reasons.append(f"single vague term '{words[0]}' with no concrete entity")
    return {"ambiguous": bool(reasons), "reasons": reasons, "content_words": words}


def decide(stats: dict, coverage: Optional[dict] = None,
           contradiction: Optional[dict] = None, *, ambiguity: Optional[dict] = None,
           min_strong: int = 1, min_clusters: int = 2) -> dict:
    """Choose ANSWER / ASK / ABSTAIN from the evidence signals.

    stats         : evidence_stats() — independent_clusters, strong_sources, max_authority
    coverage      : coverage_signals() — weak, reasons  (optional)
    contradiction : contradiction_summary() — strength, has_contradictions (optional)
    ambiguity     : assess_ambiguity() — ambiguous, reasons (optional)
    """
    coverage = coverage or {}
    contradiction = contradiction or {}
    ambiguity = ambiguity or {}
    clusters = stats.get("independent_clusters", 0)
    strong = stats.get("strong_sources", 0)
    max_auth = stats.get("max_authority", 0.0)
    reasons: List[str] = []

    # 1. ASK — the request itself is too ambiguous to research meaningfully.
    if ambiguity.get("ambiguous"):
        q = ("Your request is broad or ambiguous — could you specify the exact "
             "subject, scope, or what you want to know?")
        return {"decision": ASK, "confidence": "n/a", "uncertain": True,
                "reasons": ["ambiguous request: " + "; ".join(ambiguity.get("reasons", []))],
                "clarifying_question": q}

    # 2. ABSTAIN — there is essentially nothing to stand on.
    if clusters == 0:
        reasons.append("no usable evidence clusters")
    if strong == 0 and clusters < min_clusters and max_auth < 0.5:
        reasons.append(f"no strong sources and only {clusters} weak cluster(s)")
    if reasons:
        return {"decision": ABSTAIN, "confidence": "low", "uncertain": True,
                "reasons": reasons, "clarifying_question": None}

    # 3. ANSWER — possibly with explicit uncertainty.
    confident = (clusters >= min_clusters and strong >= min_strong)
    uncertain = (not confident)
    if coverage.get("weak"):
        uncertain = True
        reasons.append("coverage weak: " + "; ".join(coverage.get("reasons", [])))
    if contradiction.get("strength") in ("moderate", "strong"):
        uncertain = True
        reasons.append(f"unresolved contradictions ({contradiction.get('strength')})")
    confidence = "high" if confident and not uncertain else ("medium" if strong >= 1 else "low")
    if not reasons:
        reasons.append(f"{clusters} clusters, {strong} strong source(s)")
    return {"decision": ANSWER, "confidence": confidence, "uncertain": uncertain,
            "reasons": reasons, "clarifying_question": None}


_BANNER_RU = {
    "ask": "❓ Нужно уточнение — ",
    "abstain": "⚠️ Недостаточно данных для надёжного ответа — ",
    "uncertain": "с явной неопределённостью",
    "confident": "уверенно",
    "answer": "✅ Отвечаю {tag} (достоверность: {conf})",
}


def gate_banner(decision: dict, out_lang: str = "en") -> str:
    """One-line human-readable banner for the report header / logs.

    Localised: this line sits at the TOP of the finished document, so an English
    banner was the first thing a Russian reader saw above a Russian report. The
    log call site keeps the English default on purpose — operator output stays
    greppable.
    """
    d = decision.get("decision", ANSWER)
    ru = str(out_lang or "en").lower().startswith("ru")
    if d == ASK:
        q = decision.get("clarifying_question", "")
        return (_BANNER_RU["ask"] + q) if ru else f"❓ Clarification needed — {q}"
    if d == ABSTAIN:
        why = "; ".join(decision.get("reasons", []))
        return (_BANNER_RU["abstain"] + why) if ru else (
            "⚠️ Insufficient evidence to answer reliably — " + why)
    if ru:
        tag = _BANNER_RU["uncertain" if decision.get("uncertain") else "confident"]
        return _BANNER_RU["answer"].format(tag=tag, conf=decision.get("confidence"))
    tag = "with explicit uncertainty" if decision.get("uncertain") else "confidently"
    return f"✅ Answering {tag} (confidence: {decision.get('confidence')})"

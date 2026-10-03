"""Entity extraction + entity-driven query generation for multi-hop research.

Deterministic by design: extraction is regex/heuristic so it is fully unit-testable
offline and never blocks a run on an LLM call. The pipeline uses it to (a) find the
salient named entities in the first-pass briefs, and (b) mint follow-up queries that
expand research AROUND those entities — a structured multi-hop loop, not "more search".

Entity categories tracked (spec B): models, people/orgs, datasets/benchmarks,
versions, APIs/standards, issue IDs, repositories, DOIs/papers, dates.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger("assistant.entities")

# Ordered, non-overlapping patterns. First match wins per span so a DOI isn't also
# scraped as a version, etc. Each yields (category, normalized_value).
_PATTERNS = [
    ("doi",        re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")),
    ("arxiv",      re.compile(r"\barXiv:\s*\d{4}\.\d{4,5}(?:v\d+)?\b", re.IGNORECASE)),
    ("cve",        re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)),
    ("issue",      re.compile(r"(?:#|\bGH-|\bPR\s?#?|\bissue\s?#?)\d{2,6}\b", re.IGNORECASE)),
    ("rfc",        re.compile(r"\bRFC\s?\d{3,5}\b", re.IGNORECASE)),
    ("version",    re.compile(r"\bv?\d+\.\d+(?:\.\d+)?(?:-[A-Za-z0-9.]+)?\b")),
    ("repo",       re.compile(r"\b[A-Za-z0-9][\w.-]+/[A-Za-z0-9][\w.-]+\b")),
    # model/product codes: GPT-4, BGE-M3, LLaMA-2, bge-reranker-v2-m3, T5, BERT.
    ("model",      re.compile(r"\b(?:[A-Z][A-Za-z]*[A-Z0-9][A-Za-z0-9]*|[A-Za-z]+-[A-Za-z0-9.-]*\d[A-Za-z0-9.-]*)\b")),
    ("year",       re.compile(r"\b(?:19|20)\d{2}\b")),
]

# Proper-noun phrases: 1-4 Capitalized words (allows internal & / . - like "U.S.").
_PROPER_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9.&-]+)(?:\s+[A-Z][A-Za-z0-9.&-]+){0,3}\b")

# Tokens that look proper but carry no entity value — dropped from proper-noun set.
_STOP_PROPER = {
    "The", "A", "An", "This", "That", "These", "Those", "It", "We", "I", "You",
    "However", "Moreover", "Therefore", "Thus", "Source", "Primary", "Secondary",
    "Community", "Low", "Note", "Figure", "Table", "Section", "Abstract",
    "Introduction", "Conclusion", "References", "According", "While", "Although",
    "Research", "Topic", "Page", "Summary", "Overall", "Both", "Each",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
}

_CATEGORY_ORDER = ("doi", "arxiv", "cve", "issue", "rfc", "repo", "model",
                   "version", "proper", "year")


class EntitySet:
    """Extracted entities with frequency-based salience. `by_cat` maps category →
    {value: count}; `rank()` returns the most salient values across all categories."""

    __slots__ = ("by_cat",)

    def __init__(self):
        self.by_cat: Dict[str, Counter] = {c: Counter() for c in _CATEGORY_ORDER}

    def add(self, category: str, value: str, n: int = 1) -> None:
        value = (value or "").strip()
        if value:
            self.by_cat.setdefault(category, Counter())[value] += n

    def total(self) -> int:
        return sum(sum(c.values()) for c in self.by_cat.values())

    def values(self, category: str) -> List[str]:
        return [v for v, _ in self.by_cat.get(category, Counter()).most_common()]

    def rank(self, top_k: Optional[int] = None,
             prefer: Sequence[str] = ("model", "repo", "doi", "arxiv", "proper")) -> List[dict]:
        """Most salient entities first. Salience = frequency, with a category bonus
        so concrete identifiers (models/repos/papers) outrank bare proper nouns."""
        bonus = {c: (len(prefer) - i) for i, c in enumerate(prefer)}
        scored = []
        for cat, counter in self.by_cat.items():
            for val, cnt in counter.items():
                scored.append({"value": val, "category": cat,
                               "count": cnt, "score": cnt + bonus.get(cat, 0)})
        scored.sort(key=lambda e: (-e["score"], -e["count"], e["value"]))
        return scored[:top_k] if top_k else scored

    def to_dict(self) -> dict:
        return {c: dict(self.by_cat[c]) for c in _CATEGORY_ORDER if self.by_cat[c]}


def extract_entities(text: str, *, into: Optional[EntitySet] = None) -> EntitySet:
    """Extract entities from one text blob into a (new or shared) EntitySet."""
    es = into or EntitySet()
    if not text:
        return es
    claimed = []  # spans already consumed by a higher-priority pattern

    def overlaps(a, b):
        return not (a[1] <= b[0] or b[1] <= a[0])

    for cat, rx in _PATTERNS:
        for m in rx.finditer(text):
            span = (m.start(), m.end())
            if any(overlaps(span, c) for c in claimed):
                continue
            val = m.group(0).strip()
            # Drop bare years masquerading as versions, and 1-char model noise.
            if cat == "version" and re.fullmatch(r"(?:19|20)\d{2}", val):
                continue
            if cat == "model" and len(val) < 2:
                continue
            if cat == "repo" and ("/" not in val or val.count("/") > 1):
                continue
            es.add(cat, val)
            claimed.append(span)

    for m in _PROPER_RE.finditer(text):
        span = (m.start(), m.end())
        if any(overlaps(span, c) for c in claimed):
            continue
        phrase = m.group(0).strip()
        if phrase in _STOP_PROPER:
            continue
        # Single bare common word that happens to be capitalized at sentence start:
        # keep only multi-word phrases or words with internal signal (digits/&/.).
        if " " not in phrase and phrase.istitle() and phrase not in es.by_cat["model"]:
            if not re.search(r"[0-9.&-]", phrase) and len(phrase) < 4:
                continue
        es.add("proper", phrase)
    return es


def extract_from_briefs(briefs: Sequence[dict], *, text_key: str = "brief",
                        title_key: str = "title") -> EntitySet:
    """Aggregate entities across a set of source briefs (titles weighted by being
    repeated into the blob, since a title term is a strong salience signal)."""
    es = EntitySet()
    for b in briefs:
        title = str(b.get(title_key, ""))
        body = str(b.get(text_key, ""))
        extract_entities(f"{title} {title} {body}", into=es)
    return es


# Query templates per category — generic, no domain hardcoding (spec: not "more
# search", a structured expansion). {e} = entity, {topic} = the research core.
_QUERY_TEMPLATES = {
    "model":   ("{e} benchmark results", "{e} documentation", "{e} limitations"),
    "repo":    ("{e} README", "{e} issues", "{e} release notes"),
    "doi":     ("{e}",),
    "arxiv":   ("{e}",),
    "proper":  ("{e} {topic}", "{e} official"),
    "version": ("{topic} {e} changelog", "{topic} {e} release notes"),
    "cve":     ("{e} details", "{e} patch"),
    "issue":   ("{topic} {e}",),
    "rfc":     ("{e}",),
    "year":    (),  # bare years make poor standalone queries
}


def entity_queries(topic: str, entities: EntitySet, *, max_entities: int = 4,
                   per_entity: int = 1, visited: Optional[set] = None) -> List[str]:
    """Mint follow-up queries from the most salient entities. `visited` (lowercased
    queries already run) is consulted AND updated so multi-hop never repeats work."""
    visited = visited if visited is not None else set()
    core = " ".join((topic or "").split()[:6])  # short topical anchor
    out: List[str] = []
    for ent in entities.rank(top_k=max_entities):
        tmpls = _QUERY_TEMPLATES.get(ent["category"], ("{e} {topic}",))
        made = 0
        for t in tmpls:
            if made >= per_entity:
                break
            q = t.format(e=ent["value"], topic=core).strip()
            key = q.lower()
            if not q or key in visited:
                continue
            visited.add(key)
            out.append(q)
            made += 1
    logger.info("entity_queries: %d follow-up queries from top %d entities",
                len(out), max_entities)
    return out

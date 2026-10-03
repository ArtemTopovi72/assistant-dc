"""Lightweight community / cluster detection over source briefs (spec C).

The flat pipeline gathers many briefs but treats them as one undifferentiated
pile, which reads "flat" on broad topics. This module groups the briefs into
coherent sub-threads so the engine can (a) tell major threads from minor ones,
(b) allocate deep-dive follow-up budget to the important ones, and (c) synthesize
each thread before the global synthesis (hierarchical, not flat).

Design rules honored: practical + lightweight + testable, no heavyweight infra.
Clustering is a deterministic single-link agglomeration over a cheap pairwise
similarity that mixes:

    * entity overlap   (Jaccard of each brief's extracted entity-value set)
    * lexical overlap  (Jaccard of content-word shingles of title+brief)
    * domain family     (same registrable domain → a small affinity bonus)

No embeddings required (works offline in the regression harness); if dense
similarity is ever wanted it can be layered on without changing the interface.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Dict, List, Sequence, Set, Tuple

logger = logging.getLogger("assistant.clustering")

_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9][\w-]*")

# Trust tier -> evidence weight. Shared with hierarchy.py and hierarchical.py
# (both import it from here) so the weighting stays a single source of truth.
_TRUST_W = {"PRIMARY": 1.0, "SECONDARY": 0.7, "COMMUNITY": 0.4, "LOW": 0.15}
_STOP = {
    "the", "a", "an", "of", "for", "to", "and", "or", "in", "on", "is", "are",
    "be", "as", "by", "with", "that", "this", "it", "its", "was", "were", "has",
    "have", "had", "but", "not", "from", "at", "which", "their", "they", "than",
    "can", "will", "may", "also", "such", "these", "those", "into", "more",
}


def _content_words(text: str) -> List[str]:
    return [w for w in (m.group(0).lower() for m in _WORD_RE.finditer(text or ""))
            if w not in _STOP and len(w) > 2]


def _domain_family(domain: str) -> str:
    """Registrable-ish domain: last two labels (good enough to group same-site)."""
    parts = (domain or "").lower().split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else (domain or "").lower()


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / float(len(a | b)) if inter else 0.0


class Cluster:
    """One thread of the research: the briefs that belong together, plus the
    entities/domains that characterize it and a salience score."""

    __slots__ = ("id", "members", "entities", "domains", "label", "score", "major")

    def __init__(self, cid: int):
        self.id = cid
        self.members: List[int] = []        # indices into the brief list
        self.entities: Counter = Counter()  # entity-value -> count within cluster
        self.domains: Set[str] = set()
        self.label: str = ""
        self.score: float = 0.0
        self.major: bool = False

    def size(self) -> int:
        return len(self.members)

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "size": self.size(),
                "score": round(self.score, 3), "major": self.major,
                "members": list(self.members),
                "domains": sorted(self.domains)[:8],
                "top_entities": [e for e, _ in self.entities.most_common(6)]}


def _brief_signature(brief: dict) -> Tuple[Set[str], Set[str]]:
    """(entity-value set, content-word set) for one brief. Entities come from the
    deterministic extractor when available, else fall back to title+brief words."""
    ent: Set[str] = set()
    try:
        from entities import extract_entities
        es = extract_entities(f"{brief.get('title', '')} {brief.get('brief', '')}")
        ent = {e["value"].lower() for e in es.rank(top_k=12)}
    except Exception:
        ent = set()
    words = set(_content_words(f"{brief.get('title', '')} {brief.get('brief', '')}"))
    return ent, words


def _similarity(sig_a, sig_b, dom_a: str, dom_b: str,
                *, w_entity: float, w_lexical: float, domain_bonus: float) -> float:
    ent = _jaccard(sig_a[0], sig_b[0])
    lex = _jaccard(sig_a[1], sig_b[1])
    sim = w_entity * ent + w_lexical * lex
    if dom_a and dom_a == dom_b:
        sim += domain_bonus
    return sim


class ClusterSet:
    """Result of clustering: the clusters plus the brief list they index into."""

    def __init__(self, briefs: Sequence[dict], clusters: List[Cluster]):
        self.briefs = list(briefs)
        self.clusters = clusters

    def major(self) -> List[Cluster]:
        return [c for c in self.clusters if c.major]

    def minor(self) -> List[Cluster]:
        return [c for c in self.clusters if not c.major]

    def members_of(self, cluster: Cluster) -> List[dict]:
        return [self.briefs[i] for i in cluster.members]

    def stats(self) -> dict:
        return {"clusters": len(self.clusters), "major": len(self.major()),
                "minor": len(self.minor()),
                "largest": max((c.size() for c in self.clusters), default=0),
                "singletons": sum(1 for c in self.clusters if c.size() == 1)}

    def to_dict(self) -> dict:
        return {"stats": self.stats(), "clusters": [c.to_dict() for c in self.clusters]}


def cluster_briefs(briefs: Sequence[dict], *, threshold: float = 0.18,
                   w_entity: float = 0.7, w_lexical: float = 0.3,
                   domain_bonus: float = 0.15,
                   major_min_size: int = 2,
                   only_confirm: bool = True) -> ClusterSet:
    """Single-link agglomerative clustering of briefs by entity+lexical overlap.

    `only_confirm` keeps contradiction-tagged briefs out of the thread structure
    (they are weighed separately by the contradiction pass), so clusters describe
    the confirming knowledge base. Deterministic — same input → same clusters."""
    from collections import Counter

    items = [(i, b) for i, b in enumerate(briefs)
             if not (only_confirm and b.get("mode") == "contradict")]
    n = len(items)
    if n == 0:
        return ClusterSet(briefs, [])

    sigs = [_brief_signature(b) for _, b in items]
    doms = [_domain_family(b.get("domain", "")) for _, b in items]

    # Union-Find over the local item indices.
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(n):
        for j in range(i + 1, n):
            sim = _similarity(sigs[i], sigs[j], doms[i], doms[j],
                              w_entity=w_entity, w_lexical=w_lexical,
                              domain_bonus=domain_bonus)
            if sim >= threshold:
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    clusters: List[Cluster] = []
    for cid, (_root, local_idxs) in enumerate(sorted(groups.items())):
        c = Cluster(cid)
        c.members = [items[i][0] for i in local_idxs]   # back to global brief idx
        ent_counter: "Counter" = Counter()
        for i in local_idxs:
            for e in sigs[i][0]:
                ent_counter[e] += 1
            c.domains.add(doms[i])
        c.entities = ent_counter
        clusters.append(c)

    _score_and_label(clusters, briefs)
    cs = ClusterSet(briefs, clusters)
    # Mark the clusters that carry the most evidence weight as "major".
    for c in clusters:
        c.major = c.size() >= major_min_size
    if not any(c.major for c in clusters) and clusters:
        # Always have at least one major thread to deep-dive.
        max(clusters, key=lambda c: c.score).major = True
    logger.info("clustering: %s", cs.stats())
    return cs


def _score_and_label(clusters: List[Cluster], briefs: Sequence[dict]) -> None:
    """Salience = evidence weight (trust-weighted size) + entity cohesion."""
    for c in clusters:
        weight = 0.0
        for i in c.members:
            weight += _TRUST_W.get(briefs[i].get("trust", "COMMUNITY"), 0.4)
        cohesion = 0.0
        if c.entities:
            top = c.entities.most_common(1)
            cohesion = (top[0][1] / float(max(1, c.size()))) if top else 0.0
        c.score = weight + 0.5 * cohesion
        if c.entities and c.entities.most_common(1):
            c.label = c.entities.most_common(1)[0][0]
        else:
            c.label = (briefs[c.members[0]].get("domain", "") or f"cluster-{c.id}")

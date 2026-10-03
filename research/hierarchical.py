"""Cluster-level (hierarchical) synthesis layer (spec A, steps 5-6).

Flat synthesis dumps every brief into one prompt; on broad topics that reads as an
undifferentiated wall. This builds an explicit intermediate layer: each major
thread (cluster) is summarized on its own FIRST, then those cluster digests are
handed to the global synthesis as scaffolding. The tree is:

    sources → clusters → per-cluster digest → global synthesis

The per-cluster digest is deterministic (offline-testable, no LLM): it states the
thread label, its evidence weight, the entities it revolves around, the strongest
representative claims, and any contradiction touching it. Global synthesis then
writes prose ON TOP of this scaffold rather than from a flat pile, and the digest
is also saved to state so the hierarchical flow is inspectable.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import clustering as _cluster

logger = logging.getLogger("assistant.hierarchical")

# Trust-tier evidence weight: single source of truth in clustering.py, also
# reused by hierarchy.py, so all three modules agree on the weighting.
_TRUST_W = _cluster._TRUST_W


def _rep_claims(members: List[dict], k: int = 2) -> List[dict]:
    """Pick the strongest representative claims of a cluster: highest trust, then
    most-corroborated (support_count), then longest brief."""
    ranked = sorted(
        members,
        key=lambda b: (_TRUST_W.get(b.get("trust", "COMMUNITY"), 0.4),
                       b.get("support_count", 1), len(b.get("brief", ""))),
        reverse=True)
    return ranked[:k]


def cluster_digests(clusterset, *, contradiction_domains: Optional[set] = None,
                    max_clusters: int = 6, claims_per_cluster: int = 2) -> List[dict]:
    """Build a structured digest per (major-first) cluster. Returns a list of dicts
    so it is both renderable and saveable to state."""
    contradiction_domains = contradiction_domains or set()
    ordered = sorted(clusterset.clusters, key=lambda c: (c.major, c.score), reverse=True)
    digests: List[dict] = []
    for c in ordered[:max_clusters]:
        members = clusterset.members_of(c)
        reps = _rep_claims(members, claims_per_cluster)
        weight = round(sum(_TRUST_W.get(b.get("trust", "COMMUNITY"), 0.4)
                           for b in members), 2)
        touched = sorted({b.get("domain", "") for b in members} & contradiction_domains)
        digests.append({
            "id": c.id, "label": c.label, "major": c.major,
            "size": c.size(), "score": round(c.score, 2), "evidence_weight": weight,
            "entities": [e for e, _ in c.entities.most_common(6)],
            "domains": sorted(c.domains)[:6],
            "representative_claims": [
                {"domain": b.get("domain", ""), "trust": b.get("trust", ""),
                 "support_count": b.get("support_count", 1),
                 "text": (b.get("brief", "") or "")[:400]} for b in reps],
            "contradicted_domains": touched,
        })
    logger.info("hierarchical: %d cluster digests (%d major)",
                len(digests), sum(1 for d in digests if d["major"]))
    return digests


def render_digest_preamble(digests: List[dict]) -> str:
    """Render the cluster digests as a compact Markdown scaffold to prepend to the
    global synthesis input (so the model writes ON the thread structure)."""
    if not digests:
        return ""
    lines = ["THREAD STRUCTURE (synthesize per-thread first, then integrate; do "
             "NOT repeat the same point across threads):"]
    for d in digests:
        tier = "MAJOR" if d["major"] else "minor"
        ents = ", ".join(d["entities"][:5]) or "—"
        lines.append(f"\n### Thread {d['id']} [{tier}] — {d['label']} "
                     f"(weight={d['evidence_weight']}, sources={d['size']})")
        lines.append(f"Key entities: {ents}")
        if d["contradicted_domains"]:
            lines.append(f"⚠ Contradicted by: {', '.join(d['contradicted_domains'])}")
        for rc in d["representative_claims"]:
            sup = f" ×{rc['support_count']}" if rc.get("support_count", 1) > 1 else ""
            lines.append(f"- ({rc['domain']}/{rc['trust']}{sup}) {rc['text']}")
    return "\n".join(lines) + "\n"

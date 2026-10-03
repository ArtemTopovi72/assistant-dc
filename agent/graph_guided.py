"""Graph-guided + cluster-guided search expansion (spec B + E + G).

The research graph and the cluster set are no longer passive artifacts shown at
the end — here they actively decide what to retrieve next. Instead of "search
more around any entity" (flat multi-hop), this picks targets by *structure*:

    1. high-centrality entities      — the hubs the evidence revolves around
    2. unresolved contradictions     — claims with refuted_by edges to chase down
    3. weakly-supported claims        — single-source / LOW-trust claims to corroborate
    4. major clusters needing depth   — important threads with thin evidence
    5. missing links                  — central entities not yet cross-connected

Every target carries a recorded *reason* (observability) and the produced queries
are visited-set filtered so expansion is bounded and never loops. Pure functions —
fully unit-testable offline; the actual retrieval is done by the caller.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Optional, Set

logger = logging.getLogger("assistant.graph_guided")


def entity_centrality(graph) -> List[dict]:
    """Degree centrality of entity nodes = how many sources mention them. Returns
    [{value, category, degree}] sorted by degree desc (the evidence hubs)."""
    deg: Counter = Counter()
    meta: Dict[str, dict] = {}
    for e in graph.edges:
        if e["rel"] == "mentioned_in":           # entity --mentioned_in--> source
            ent = e["src"]
            deg[ent] += 1
            node = graph.nodes.get(ent, {})
            meta[ent] = {"value": node.get("value", node.get("label", "")),
                         "category": node.get("category", "proper")}
    out = [{"value": meta[nid]["value"], "category": meta[nid]["category"],
            "degree": d} for nid, d in deg.most_common() if meta.get(nid)]
    return out


def weak_claims(graph, *, low_trust=("COMMUNITY", "LOW")) -> List[dict]:
    """Claims standing on a single low-trust source — candidates for corroboration.
    Returns [{claim, source_url, trust, domain}]."""
    # Map claim -> its supporting sources.
    support: Dict[str, List[str]] = {}
    for e in graph.edges:
        if e["rel"] == "supported_by":            # claim --supported_by--> source
            support.setdefault(e["src"], []).append(e["dst"])
    weak: List[dict] = []
    for claim_id, srcs in support.items():
        cnode = graph.nodes.get(claim_id, {})
        trust = cnode.get("trust", "COMMUNITY")
        if len(srcs) <= 1 and trust in low_trust:
            snode = graph.nodes.get(srcs[0], {}) if srcs else {}
            weak.append({"claim": cnode.get("text", ""),
                         "source_url": cnode.get("source_url", ""),
                         "trust": trust, "domain": snode.get("domain", "")})
    return weak


def _push(out: List[dict], visited: Set[str], query: str, reason: str,
          target: str, max_total: int) -> bool:
    q = (query or "").strip()
    key = q.lower()
    if not q or key in visited or len(out) >= max_total:
        return False
    visited.add(key)
    out.append({"query": q, "reason": reason, "target": target})
    return True


def plan_graph_expansion(topic: str, graph, clusterset=None, *,
                         contradiction=None, visited: Optional[Set[str]] = None,
                         max_queries: int = 6, top_entities: int = 4,
                         max_weak: int = 3) -> List[dict]:
    """Decide the next round of retrieval from graph + cluster structure.

    Returns a list of {query, reason, target} records (the reasons are logged and
    saved for observability). `visited` (lowercased queries already issued) is read
    AND updated so this is loop-safe. Bounded by `max_queries`."""
    visited = visited if visited is not None else set()
    core = " ".join((topic or "").split()[:8])
    out: List[dict] = []

    # 1. High-centrality entity hubs — deepen the things the evidence revolves around.
    for ent in entity_centrality(graph)[:top_entities]:
        if len(out) >= max_queries:
            break
        v = ent["value"]
        _push(out, visited, f"{v} {core} analysis",
              f"high-centrality entity (degree={ent['degree']})", f"entity:{v}",
              max_queries)

    # 2. Unresolved contradictions — chase the disputed claims for resolution.
    if contradiction and contradiction.get("has_contradictions"):
        for dom in contradiction.get("domains", [])[:2]:
            if len(out) >= max_queries:
                break
            _push(out, visited, f"{core} {dom} response OR rebuttal OR consensus",
                  "resolve unresolved contradiction", f"contradiction:{dom}",
                  max_queries)

    # 3. Weakly-supported claims — corroborate single low-trust sources.
    for wc in weak_claims(graph)[:max_weak]:
        if len(out) >= max_queries:
            break
        anchor = (wc.get("domain") or wc.get("claim", "")[:40] or core)
        _push(out, visited, f"{core} {anchor} corroboration OR independent confirmation",
              f"weakly-supported claim ({wc.get('trust')})", f"weakclaim:{anchor}",
              max_queries)

    # 4. Major clusters that are thin on evidence — allocate deep-dive budget.
    if clusterset is not None:
        for c in sorted(clusterset.major(), key=lambda c: c.score, reverse=True):
            if len(out) >= max_queries:
                break
            label = c.label or (c.to_dict().get("top_entities") or [core])[0]
            # Only deep-dive clusters that are important but under-sourced.
            if c.size() <= 2:
                _push(out, visited, f"{label} {core} in depth",
                      f"major cluster deep-dive (size={c.size()}, score={round(c.score, 2)})",
                      f"cluster:{c.id}", max_queries)

    logger.info("graph_guided: %d expansion queries (%s)",
                len(out), "; ".join(f"{o['target']}<-{o['reason']}" for o in out[:6]))
    return out[:max_queries]


def cluster_contradiction_targets(topic: str, graph, clusterset, *,
                                  contradiction=None, visited: Optional[Set[str]] = None,
                                  max_queries: int = 4) -> List[dict]:
    """Prioritize contradiction search by graph/cluster importance (spec E): aim
    disconfirmation at the MAJOR clusters and the high-centrality entities first,
    not at random. Returns {query, reason, target} records (visited-aware)."""
    visited = visited if visited is not None else set()
    core = " ".join((topic or "").split()[:8])
    out: List[dict] = []
    # Major clusters first, ranked by salience.
    majors = sorted(clusterset.major(), key=lambda c: c.score, reverse=True) if clusterset else []
    for c in majors:
        if len(out) >= max_queries:
            break
        label = c.label or core
        _push(out, visited, f"{label} criticism OR limitations OR disputed",
              f"contradiction on major cluster (score={round(c.score, 2)})",
              f"cluster:{c.id}", max_queries)
    # Then the most central entities.
    for ent in entity_centrality(graph)[:max_queries]:
        if len(out) >= max_queries:
            break
        _push(out, visited, f"{ent['value']} overstated OR flawed OR retracted",
              f"contradiction on central entity (degree={ent['degree']})",
              f"entity:{ent['value']}", max_queries)
    logger.info("cluster_contradiction_targets: %d prioritized queries", len(out))
    return out[:max_queries]

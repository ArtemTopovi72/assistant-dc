"""Graph community detection over the research graph (spec B — real communities).

The lexical clusters group briefs by word/entity overlap. This is different: it runs
modularity-optimizing community detection (Louvain) over the ACTUAL graph topology —
the entity↔source↔claim network with provenance (supported_by) and contradiction
(refuted_by) edges. Communities therefore reflect how sources, the entities they
mention, and the claims they assert are wired together, which generally does NOT
coincide with lexical clusters (a community can span domains that never share
wording but co-mention the same entities).

Louvain here is a compact, dependency-free, deterministic implementation:
  1. each node starts in its own community;
  2. repeatedly move each node (in a fixed, sorted order) to the neighbouring
     community that yields the largest positive modularity gain, until no move helps;
  3. aggregate communities into super-nodes and repeat, until modularity stops rising.

Edge weights encode semantics: provenance/mention edges = 1.0, contradiction
(refuted_by) edges = 1.5 (a dispute is a strong structural signal binding the
disputing source to the disputed claim). The graph is treated as undirected for
modularity (each typed directed edge contributes an undirected weighted link).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Tuple

logger = logging.getLogger("assistant.communities")

_REL_WEIGHT = {
    "supported_by": 1.0, "mentioned_in": 1.0, "mentions": 1.0,
    "asserts": 1.0, "refuted_by": 1.5,
}


def _undirected_weighted(graph) -> Tuple[List[str], Dict[Tuple[str, str], float]]:
    """Collapse the typed directed research graph into an undirected weighted graph.
    Returns (sorted node list, {(u,v): weight} with u<v)."""
    w: Dict[Tuple[str, str], float] = defaultdict(float)
    nodes = set(graph.nodes.keys())
    for e in graph.edges:
        u, v = e["src"], e["dst"]
        if u == v:
            continue
        nodes.add(u); nodes.add(v)
        a, b = (u, v) if u < v else (v, u)
        w[(a, b)] += _REL_WEIGHT.get(e["rel"], 1.0)
    return sorted(nodes), dict(w)


def _modularity(comm: Dict[str, int], adj: Dict[str, Dict[str, float]],
                m2: float) -> float:
    if m2 <= 0:
        return 0.0
    deg = {n: sum(adj[n].values()) for n in adj}
    # sum over edges within communities of (A_uv - k_u k_v / 2m)
    intra = 0.0
    sigma_tot: Dict[int, float] = defaultdict(float)
    for n in adj:
        sigma_tot[comm[n]] += deg[n]
        for nb, wt in adj[n].items():
            if comm[nb] == comm[n]:
                intra += wt
    # Q = intra/2m - sum_c (sigma_tot_c / 2m)^2   (intra already double-counts)
    q = intra / m2 - sum((st / m2) ** 2 for st in sigma_tot.values())
    return q


def _louvain_level(nodes: List[str], adj: Dict[str, Dict[str, float]],
                   m2: float) -> Dict[str, int]:
    """One Louvain local-moving phase. Deterministic (fixed node order)."""
    comm = {n: i for i, n in enumerate(nodes)}
    deg = {n: sum(adj[n].values()) for n in nodes}
    sigma_tot: Dict[int, float] = defaultdict(float)
    for n in nodes:
        sigma_tot[comm[n]] += deg[n]

    improved = True
    passes = 0
    while improved and passes < 50:
        improved = False
        passes += 1
        for n in nodes:
            cn = comm[n]
            # weight from n into each neighbouring community
            k_in: Dict[int, float] = defaultdict(float)
            for nb, wt in adj[n].items():
                if nb != n:
                    k_in[comm[nb]] += wt
            # remove n from its community
            sigma_tot[cn] -= deg[n]
            best_c, best_gain = cn, 0.0
            for c, kin in sorted(k_in.items()):
                gain = kin - sigma_tot[c] * deg[n] / m2
                if gain > best_gain + 1e-12:
                    best_gain, best_c = gain, c
            # gain of staying out / returning to original is the baseline (0 via cn removed)
            sigma_tot[best_c] += deg[n]
            if best_c != cn:
                comm[n] = best_c
                improved = True
    return comm


def detect_communities(graph, *, max_levels: int = 10) -> dict:
    """Run Louvain on the research graph. Returns a dict with the community
    assignment, per-community metadata, and modularity. Deterministic."""
    nodes, w = _undirected_weighted(graph)
    if not nodes or not w:
        return {"communities": [], "modularity": 0.0,
                "stats": {"communities": 0, "nodes": len(nodes), "edges": len(w),
                          "modularity": 0.0, "levels": 0}}
    # Build adjacency (with the current node set).
    adj: Dict[str, Dict[str, float]] = {n: {} for n in nodes}
    for (a, b), wt in w.items():
        adj[a][b] = adj[a].get(b, 0.0) + wt
        adj[b][a] = adj[b].get(a, 0.0) + wt
    m2 = 2.0 * sum(w.values())

    # node -> final community label, refined across aggregation levels.
    node2final = {n: n for n in nodes}
    cur_nodes, cur_adj = nodes, adj
    prev_q = -1.0
    levels = 0
    for _ in range(max_levels):
        comm = _louvain_level(cur_nodes, cur_adj, m2)
        q = _modularity(comm, cur_adj, m2)
        # relabel communities to small ints in deterministic order
        # propagate to original nodes
        for orig, sup in list(node2final.items()):
            node2final[orig] = comm.get(sup, comm.get(node2final[orig], node2final[orig]))
        levels += 1
        # aggregate
        groups: Dict[int, List[str]] = defaultdict(list)
        for n in cur_nodes:
            groups[comm[n]].append(n)
        if len(groups) == len(cur_nodes) or q <= prev_q + 1e-9:
            break
        prev_q = q
        new_nodes = sorted(groups.keys(), key=lambda c: c)
        new_nodes = [str(c) for c in new_nodes]
        new_adj: Dict[str, Dict[str, float]] = {n: defaultdict(float) for n in new_nodes}
        for n in cur_nodes:
            for nb, wt in cur_adj[n].items():
                ca, cb = str(comm[n]), str(comm[nb])
                new_adj[ca][cb] += wt
        cur_nodes, cur_adj = new_nodes, {k: dict(v) for k, v in new_adj.items()}

    # Build final integer labels in deterministic order.
    labels = sorted(set(node2final.values()), key=lambda x: str(x))
    relabel = {lab: i for i, lab in enumerate(labels)}
    assign = {n: relabel[node2final[n]] for n in nodes}
    final_q = _modularity(assign, adj, m2)

    communities = _summarize(graph, assign)
    stats = {"communities": len(communities), "nodes": len(nodes), "edges": len(w),
             "modularity": round(final_q, 4), "levels": levels,
             "spanning": sum(1 for c in communities if c["n_domains"] > 1)}
    logger.info("communities: %s", stats)
    return {"communities": communities, "modularity": round(final_q, 4), "stats": stats,
            "assignment": {n: assign[n] for n in nodes}}


def _summarize(graph, assign: Dict[str, int]) -> List[dict]:
    """Per-community metadata: members by type, domains, entities, contradiction
    involvement (spec B: communities use entities + claims + sources + contradiction
    + provenance)."""
    by_comm: Dict[int, List[str]] = defaultdict(list)
    for nid, c in assign.items():
        by_comm[c].append(nid)
    out: List[dict] = []
    for c in sorted(by_comm):
        members = by_comm[c]
        ents, srcs, claims, doms = [], [], [], set()
        contra = 0
        for nid in members:
            node = graph.nodes.get(nid, {})
            t = node.get("type")
            if t == "entity":
                ents.append(node.get("value", node.get("label", "")))
            elif t == "source":
                srcs.append(node.get("url", nid))
                doms.add(node.get("domain", ""))
                if node.get("mode") == "contradict":
                    contra += 1
            elif t == "claim":
                claims.append(nid)
        out.append({
            "id": c, "size": len(members),
            "n_sources": len(srcs), "n_entities": len(ents), "n_claims": len(claims),
            "n_domains": len([d for d in doms if d]),
            "domains": sorted(d for d in doms if d)[:10],
            "entities": ents[:12],
            "contradiction_sources": contra,
        })
    # Biggest, most-connected communities first.
    out.sort(key=lambda x: (-x["size"], x["id"]))
    return out


def communities_vs_clusters(communities: dict, clusterset) -> dict:
    """Quantify how much the graph communities DIFFER from the lexical clusters
    (spec B acceptance: 'communities differ from lexical clusters'). Compares the
    source→group partitions and reports an agreement fraction (low = they differ)."""
    assign = communities.get("assignment", {})
    # source url -> community id (sources are 'src::<url>' nodes)
    src_comm: Dict[str, int] = {}
    for nid, cid in assign.items():
        if nid.startswith("src::"):
            src_comm[nid[len("src::"):]] = cid
    # source url -> lexical cluster id
    src_clu: Dict[str, int] = {}
    for c in getattr(clusterset, "clusters", []):
        for i in c.members:
            url = clusterset.briefs[i].get("url", "")
            if url:
                src_clu[url] = c.id
    common = [u for u in src_comm if u in src_clu]
    if len(common) < 2:
        return {"comparable_sources": len(common), "pair_agreement": None,
                "differ": None}
    # pairwise: fraction of source pairs that are grouped the SAME way by both
    same = tot = 0
    for a in range(len(common)):
        for b in range(a + 1, len(common)):
            u, v = common[a], common[b]
            tot += 1
            co = (src_comm[u] == src_comm[v])
            cl = (src_clu[u] == src_clu[v])
            if co == cl:
                same += 1
    agreement = same / tot if tot else 1.0
    return {"comparable_sources": len(common), "pairs": tot,
            "pair_agreement": round(agreement, 3),
            "differ": agreement < 0.95}

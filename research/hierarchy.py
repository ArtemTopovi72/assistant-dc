"""Recursive hierarchical clustering of briefs (spec A — true multi-level tree).

The two-level scheme (interim + final clustering) only ever produced root → cluster.
This builds a real recursion: every cluster whose evidence is large and internally
diverse spawns CHILD clusters, which can spawn their own children, down to a bounded
depth — so a broad topic yields

    root
    ├─ cluster
    │   ├─ subcluster
    │   │   ├─ subcluster
    │   │   └─ …
    │   └─ …
    └─ …

A node is split only when:
  * its depth < max_depth, and
  * it has >= 2 * min_cluster_size members (enough to divide), and
  * re-clustering its members at a TIGHTER threshold yields >1 surviving child
    (each child >= min_cluster_size); otherwise the node is a leaf.

The split threshold rises with depth (split_threshold + depth * step) so each level
is a finer-grained division than its parent. This is deterministic (same input →
same tree) and loop-safe: children are strict subsets, depth is bounded, and a level
that fails to divide its parent stops. Pure / offline-testable — no LLM, no network.
"""
from __future__ import annotations

import logging
from typing import List, Sequence

import clustering as _cluster

logger = logging.getLogger("assistant.hierarchy")

# Trust-tier evidence weight lives in clustering.py (single source of truth,
# also used by hierarchical.py) — reused here so all three modules agree.
_TRUST_W = _cluster._TRUST_W


class HNode:
    """One node in the hierarchy tree. `members` are global brief indices."""
    __slots__ = ("id", "depth", "label", "members", "score", "entities",
                 "domains", "children")

    def __init__(self, nid: str, depth: int, members: List[int]):
        self.id = nid
        self.depth = depth
        self.members = members
        self.label = ""
        self.score = 0.0
        self.entities: List[str] = []
        self.domains: List[str] = []
        self.children: List["HNode"] = []

    def size(self) -> int:
        return len(self.members)

    def is_leaf(self) -> bool:
        return not self.children

    def to_dict(self) -> dict:
        return {"id": self.id, "depth": self.depth, "label": self.label,
                "size": self.size(), "score": round(self.score, 3),
                "entities": self.entities[:6], "domains": self.domains[:6],
                "members": list(self.members),
                "children": [c.to_dict() for c in self.children]}


class Hierarchy:
    def __init__(self, root: HNode, briefs: Sequence[dict], params: dict):
        self.root = root
        self.briefs = list(briefs)
        self.params = params

    # -- measurements -------------------------------------------------------- #
    @staticmethod
    def _walk(n: HNode):
        """Pre-order traversal of the subtree rooted at `n` (shared by every
        measurement below instead of each re-implementing its own recursion)."""
        yield n
        for c in n.children:
            yield from Hierarchy._walk(c)

    def max_depth(self) -> int:
        """Deepest level that actually contains a cluster node (root = level 0).
        Equivalent to the deepest LEAF depth in the tree."""
        return max(n.depth for n in self._walk(self.root) if n.is_leaf())

    def levels(self) -> int:
        """Number of distinct cluster levels = max_depth + 1 (root counts as one)."""
        return self.max_depth() + 1

    def count_nodes(self) -> int:
        return sum(1 for _ in self._walk(self.root))

    def leaves(self) -> List[HNode]:
        return [n for n in self._walk(self.root) if n.is_leaf()]

    def stats(self) -> dict:
        return {"levels": self.levels(), "max_depth": self.max_depth(),
                "nodes": self.count_nodes() - 1,            # exclude synthetic root
                "leaves": len(self.leaves()),
                "root_children": len(self.root.children),
                "params": self.params}

    def to_dict(self) -> dict:
        return {"stats": self.stats(), "root": self.root.to_dict()}


def _describe(node: HNode, briefs: Sequence[dict]) -> None:
    """Fill label/score/entities/domains for a node from its member briefs."""
    from collections import Counter
    ents: Counter = Counter()
    doms: set = set()
    weight = 0.0
    for i in node.members:
        b = briefs[i]
        doms.add(b.get("domain", ""))
        weight += _TRUST_W.get(b.get("trust", "COMMUNITY"), 0.4)
        sig = _cluster._brief_signature(b)
        for e in sig[0]:
            ents[e] += 1
    node.entities = [e for e, _ in ents.most_common(8)]
    node.domains = sorted(d for d in doms if d)
    node.score = round(weight + 0.5 * (ents.most_common(1)[0][1] / max(1, node.size())
                                       if ents else 0.0), 3)
    node.label = (node.entities[0] if node.entities
                  else (briefs[node.members[0]].get("domain", "") or f"node-{node.id}"))


def _level_threshold(depth: int, base_threshold: float, split_threshold: float,
                     step: float) -> float:
    """Coarse at the root, finer with depth: depth-0 uses base_threshold; each deeper
    level uses split_threshold incremented by step."""
    return base_threshold if depth == 0 else split_threshold + (depth - 1) * step


def _split(node: HNode, briefs: Sequence[dict], *, max_depth: int,
           min_cluster_size: int, split_threshold: float, base_threshold: float,
           step: float) -> None:
    """Recursively divide `node` into child clusters. Loop-safe at EVERY level
    (including the root): a node becomes a leaf unless re-clustering its members
    produces >1 surviving child AND the largest child is strictly smaller than the
    node — so children are always strict subsets and depth is bounded."""
    _describe(node, briefs)
    if node.depth >= max_depth or node.size() < 2 * min_cluster_size:
        return
    sub_briefs = [briefs[i] for i in node.members]
    # Scan thresholds upward from this level's starting point: find the COARSEST
    # threshold that actually divides the node into >1 strict-subset child. A single
    # full-size group means the threshold is too loose → tighten and retry. This is
    # what gives real depth regardless of the absolute base, and is still loop-safe
    # (every accepted child is strictly smaller than the parent).
    start = _level_threshold(node.depth, base_threshold, split_threshold, step)
    groups: List[List[int]] = []
    t = start
    while t <= 0.95 + 1e-9:
        cs = _cluster.cluster_briefs(sub_briefs, threshold=t,
                                     major_min_size=min_cluster_size, only_confirm=False)
        cand = [[node.members[local] for local in c.members]
                for c in cs.clusters if len(c.members) >= min_cluster_size]
        if len(cand) >= 2 and max(len(g) for g in cand) < node.size():
            groups = cand
            break
        t = round(t + step, 4)
    if not groups:                     # no threshold divides it → leaf
        return
    groups.sort(key=lambda g: -len(g))
    for k, g in enumerate(groups):
        child = HNode(f"{node.id}.{k}", node.depth + 1, sorted(g))
        node.children.append(child)
        _split(child, briefs, max_depth=max_depth, min_cluster_size=min_cluster_size,
               split_threshold=split_threshold, base_threshold=base_threshold, step=step)


def build_hierarchy(briefs: Sequence[dict], *, max_depth: int = 4,
                    min_cluster_size: int = 2, split_threshold: float = 0.22,
                    base_threshold: float = 0.10, step: float = 0.12,
                    only_confirm: bool = True) -> Hierarchy:
    """Build the recursive tree. The root holds all confirming briefs and is split by
    the SAME guarded recursive splitter: coarse at the root (low base_threshold), finer
    at each deeper level (split_threshold + depth*step). The low base is deliberate —
    a coarse root yields a few broad threads that CAN subdivide, whereas a high base
    shatters the root and prevents real depth. Deterministic; children are strict
    subsets at every level (loop-safe)."""
    confirm = [(i, b) for i, b in enumerate(briefs)
               if not (only_confirm and b.get("mode") == "contradict")]
    root = HNode("0", 0, [i for i, _ in confirm])
    params = {"max_depth": max_depth, "min_cluster_size": min_cluster_size,
              "split_threshold": split_threshold, "base_threshold": base_threshold,
              "step": step}
    if not confirm:
        _describe_empty(root)
        return Hierarchy(root, briefs, params)
    _split(root, briefs, max_depth=max_depth, min_cluster_size=min_cluster_size,
           split_threshold=split_threshold, base_threshold=base_threshold, step=step)
    h = Hierarchy(root, briefs, params)
    logger.info("hierarchy: %s", h.stats())
    return h


def _describe_empty(root: HNode) -> None:
    root.label, root.score, root.entities, root.domains = "(empty)", 0.0, [], []


# --------------------------------------------------------------------------- #
# Synthesis scaffold from the tree
# --------------------------------------------------------------------------- #
def render_tree_preamble(hierarchy: Hierarchy, *, max_nodes: int = 24,
                         claims_per_leaf: int = 1) -> str:
    """Render the hierarchy as an indented Markdown scaffold for synthesis, so the
    model writes ON the tree (major thread → sub-thread → detail) instead of a flat
    pile. Bounded by `max_nodes` so the prompt stays small."""
    if not hierarchy.root.children:
        return ""
    briefs = hierarchy.briefs
    lines = ["RESEARCH HIERARCHY (write the report following this tree: cover each "
             "top-level thread, then its sub-threads; do NOT repeat a point across "
             "branches):"]
    budget = [max_nodes]

    def _emit(node: HNode):
        if budget[0] <= 0:
            return
        budget[0] -= 1
        indent = "  " * (node.depth - 1)
        tag = "thread" if node.depth == 1 else "sub-thread"
        ents = ", ".join(node.entities[:4]) or "—"
        lines.append(f"{indent}- [{tag} d{node.depth}] {node.label} "
                     f"(sources={node.size()}, weight={node.score}) — entities: {ents}")
        if node.is_leaf():
            reps = sorted((briefs[i] for i in node.members),
                          key=lambda b: (_TRUST_W.get(b.get("trust", "COMMUNITY"), 0.4),
                                         len(b.get("brief", ""))), reverse=True)
            for b in reps[:claims_per_leaf]:
                txt = (b.get("brief", "") or "").strip().replace("\n", " ")[:240]
                if txt:
                    lines.append(f"{indent}    · ({b.get('domain', '')}/{b.get('trust', '')}) {txt}")
        else:
            for c in node.children:
                _emit(c)

    for c in hierarchy.root.children:
        _emit(c)
    return "\n".join(lines) + "\n"

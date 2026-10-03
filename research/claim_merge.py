"""Claim-level dedup and merge (spec D).

Source-level dedup (dedupe_pages) already collapses near-duplicate *pages*, but
two different sources can still assert the same thing in slightly different words,
which makes the final report repeat itself. This pass merges briefs whose CLAIM
text is near-duplicate, while:

    * keeping full provenance — a merged claim records every source that made it
      (so corroboration is *strengthened*, never hidden), and
    * keeping contradictions separate — contradiction-tagged briefs are never
      merged into confirming ones (disagreement must stay visible).

Similarity is a deterministic content-word shingle Jaccard (offline-testable, no
embeddings). Merging confirming briefs that say the same thing also upgrades trust
context: a claim independently asserted by 3 sources is more reliable than one.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Sequence, Set, Tuple

logger = logging.getLogger("assistant.claim_merge")

_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9][\w-]*")
_STOP = {
    "the", "a", "an", "of", "for", "to", "and", "or", "in", "on", "is", "are",
    "be", "as", "by", "with", "that", "this", "it", "its", "was", "were", "has",
    "have", "had", "but", "not", "from", "at", "which", "their", "they",
}

_TRUST_RANK = {"PRIMARY": 3, "SECONDARY": 2, "COMMUNITY": 1, "LOW": 0}


def _shingles(text: str, k: int = 3) -> Set[str]:
    words = [w for w in (m.group(0).lower() for m in _WORD_RE.finditer(text or ""))
             if w not in _STOP and len(w) > 2]
    if len(words) < k:
        return set(words)
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / float(len(a | b)) if inter else 0.0


def merge_claims(briefs: Sequence[dict], *, threshold: float = 0.55,
                 only_confirm: bool = True) -> Tuple[List[dict], List[dict]]:
    """Merge near-duplicate confirming claims. Returns (merged_briefs, merge_log).

    The representative of each merged group is the highest-trust brief; the others
    are folded in as `merged_sources` (provenance preserved) and the group's
    `support_count` is recorded so synthesis can mark independently-corroborated
    claims. Contradiction briefs pass through untouched. Deterministic."""
    confirm_idx = [i for i, b in enumerate(briefs)
                   if not (only_confirm and b.get("mode") == "contradict")]
    confirm_idx_set = set(confirm_idx)
    other = [b for i, b in enumerate(briefs) if i not in confirm_idx_set]

    sigs = {i: _shingles(briefs[i].get("brief", "")) for i in confirm_idx}

    # Union-Find over confirming briefs by claim similarity.
    parent = {i: i for i in confirm_idx}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    merge_log: List[dict] = []
    for ai in range(len(confirm_idx)):
        for bi in range(ai + 1, len(confirm_idx)):
            i, j = confirm_idx[ai], confirm_idx[bi]
            sim = _jaccard(sigs[i], sigs[j])
            if sim >= threshold:
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in confirm_idx:
        groups.setdefault(find(i), []).append(i)

    merged: List[dict] = []
    for members in groups.values():
        if len(members) == 1:
            merged.append(briefs[members[0]])
            continue
        # Representative = highest trust, then longest brief (most informative).
        rep_i = max(members, key=lambda i: (_TRUST_RANK.get(briefs[i].get("trust", "COMMUNITY"), 1),
                                            len(briefs[i].get("brief", ""))))
        rep = dict(briefs[rep_i])
        folded = [i for i in members if i != rep_i]
        rep["merged_sources"] = [{"url": briefs[i].get("url", ""),
                                  "domain": briefs[i].get("domain", ""),
                                  "trust": briefs[i].get("trust", "")} for i in folded]
        # UNION preserved equations across the whole group — otherwise a unique
        # equation carried by a folded (non-representative) brief is silently lost
        # at the claim-merge stage (release-blocking: "raw equation blocks must
        # survive claim merge"). Dedupe on normalized whitespace, order-preserving.
        eqs, seen = [], set()
        for i in members:
            for eq in briefs[i].get("equations", []) or []:
                key = re.sub(r"\s+", "", eq)
                if key not in seen:
                    seen.add(key)
                    eqs.append(eq)
        if eqs:
            rep["equations"] = eqs
        rep["support_count"] = len(members)
        rep["why"] = (rep.get("why", "") +
                      f" · merged {len(folded)} duplicate claim(s) (support={len(members)})")
        merged.append(rep)
        merge_log.append({"representative": briefs[rep_i].get("url", ""),
                          "merged": [briefs[i].get("url", "") for i in folded],
                          "support_count": len(members),
                          "reason": f"claim shingle-Jaccard >= {threshold}"})

    logger.info("claim_merge: %d confirming briefs -> %d after merge (%d groups merged, "
                "%d contradiction briefs untouched)",
                len(confirm_idx), len(merged), len(merge_log), len(other))
    return merged + other, merge_log

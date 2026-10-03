"""Lightweight source / entity / claim graph for the research pipeline (spec C+G).

Practical, not academic: a dict of typed nodes + a list of typed edges, built from
the per-source briefs and the extracted entities. It connects

    entity  --mentioned_in-->  source
    source  --asserts------->  claim
    claim   --supported_by-->  source   (mode=confirm)
    claim   --refuted_by---->  source   (mode=contradict)
    source  --mentions------>  entity

so the same structure serves query expansion (salient entities), contradiction
discovery (refuted_by edges), provenance (every claim → its source + that source's
trust/freshness/mode/why), and synthesis (a compact serialized view).

No external deps. A "claim" here is one source's brief body keyed to its source —
we do not run atomic claim extraction; the source-grounded brief IS the claim unit,
which keeps provenance exact and the graph cheap.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("assistant.research_graph")


class ResearchGraph:
    def __init__(self):
        self.nodes: Dict[str, dict] = {}     # id -> {type, label, attrs...}
        self.edges: List[dict] = []          # {src, dst, rel}
        self._edge_keys = set()              # (src, dst, rel) dedup

    # -- nodes --------------------------------------------------------------- #
    def _node(self, nid: str, ntype: str, label: str, **attrs) -> str:
        if nid in self.nodes:
            self.nodes[nid].update({k: v for k, v in attrs.items() if v is not None})
        else:
            self.nodes[nid] = {"id": nid, "type": ntype, "label": label, **attrs}
        return nid

    def add_source(self, url: str, *, domain: str = "", trust: str = "COMMUNITY",
                   mode: str = "confirm", authority: float = 0.0,
                   freshness: Optional[str] = None, why: str = "") -> str:
        return self._node(f"src::{url}", "source", domain or url, url=url,
                          domain=domain, trust=trust, mode=mode,
                          authority=authority, freshness=freshness, why=why)

    def add_entity(self, value: str, category: str = "proper") -> str:
        return self._node(f"ent::{category}::{value.lower()}", "entity", value,
                          category=category, value=value)

    def add_claim(self, text: str, source_url: str, *, mode: str = "confirm",
                  trust: str = "COMMUNITY") -> str:
        cid = f"claim::{source_url}"
        return self._node(cid, "claim", (text or "")[:160], text=text,
                          source_url=source_url, mode=mode, trust=trust)

    # -- edges --------------------------------------------------------------- #
    def link(self, src: str, dst: str, rel: str) -> None:
        key = (src, dst, rel)
        if key in self._edge_keys:
            return
        self._edge_keys.add(key)
        self.edges.append({"src": src, "dst": dst, "rel": rel})

    # -- construction -------------------------------------------------------- #
    def add_brief(self, brief: dict, entity_values: Optional[List[dict]] = None) -> None:
        """Wire one source brief into the graph: its source node, its claim, the
        mode-aware claim↔source edges, and any entities it mentions."""
        url = brief.get("url", "")
        mode = brief.get("mode", "confirm")
        trust = brief.get("trust", "COMMUNITY")
        why = brief.get("why", "") or f"trust={trust}, mode={mode}"
        s = self.add_source(url, domain=brief.get("domain", ""), trust=trust,
                            mode=mode, authority=brief.get("authority", 0.0),
                            freshness=brief.get("freshness"), why=why)
        c = self.add_claim(brief.get("brief", ""), url, mode=mode, trust=trust)
        self.link(s, c, "asserts")
        self.link(c, s, "refuted_by" if mode == "contradict" else "supported_by")
        for ent in (entity_values or []):
            e = self.add_entity(ent["value"], ent.get("category", "proper"))
            self.link(e, s, "mentioned_in")
            self.link(s, e, "mentions")

    # -- queries over the graph ---------------------------------------------- #
    def sources(self) -> List[dict]:
        return [n for n in self.nodes.values() if n["type"] == "source"]

    def claims(self) -> List[dict]:
        return [n for n in self.nodes.values() if n["type"] == "claim"]

    def contradicting_sources(self) -> List[dict]:
        return [n for n in self.sources() if n.get("mode") == "contradict"]

    def provenance(self, claim_id: str) -> dict:
        """Full audit record for a claim: its text + the source backing it with
        trust/freshness/mode/why, and the relation (supported_by / refuted_by)."""
        node = self.nodes.get(claim_id, {})
        rel = next((e["rel"] for e in self.edges
                    if e["src"] == claim_id and e["rel"] in ("supported_by", "refuted_by")), None)
        src = self.nodes.get(f"src::{node.get('source_url', '')}", {})
        return {"claim": node.get("text", ""), "relation": rel,
                "source": {k: src.get(k) for k in ("url", "domain", "trust",
                                                   "authority", "freshness", "mode", "why")}}

    def stats(self) -> dict:
        st = {"nodes": len(self.nodes), "edges": len(self.edges),
              "sources": len(self.sources()), "claims": len(self.claims()),
              "entities": sum(1 for n in self.nodes.values() if n["type"] == "entity"),
              "contradicting_sources": len(self.contradicting_sources())}
        return st

    def to_dict(self) -> dict:
        return {"nodes": list(self.nodes.values()), "edges": self.edges,
                "stats": self.stats()}

    def save(self, path: Path) -> None:
        try:
            Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        except Exception as exc:
            logger.warning("research graph save failed: %s", exc)


def build_graph(briefs: List[dict], *, per_brief_entities: bool = True) -> ResearchGraph:
    """Build a graph from a brief set. Entities are extracted per brief so each
    entity links to exactly the sources that mention it (real provenance)."""
    g = ResearchGraph()
    extractor = None
    if per_brief_entities:
        try:
            from entities import extract_entities
            extractor = extract_entities
        except Exception:
            extractor = None
    for b in briefs:
        ent_vals = []
        if extractor is not None:
            es = extractor(f"{b.get('title', '')} {b.get('brief', '')}")
            ent_vals = es.rank(top_k=8)
        g.add_brief(b, ent_vals)
    logger.info("research graph: %s", g.stats())
    return g

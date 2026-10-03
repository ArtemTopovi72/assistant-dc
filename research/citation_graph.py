"""Scholarly citation-graph enrichment (Phase 9).

For research-grade questions the engine must reconstruct the intellectual
lineage — the seminal paper, its citation count, and the most-influential works
that built on it — not just summarise whatever pages a web search surfaced.
Plain web search has no citation data; this module pulls it from OpenAlex
(https://openalex.org), a free, keyless, no-rate-limit-with-mailto scholarly
graph covering 250M+ works.

`citation_lineage_brief(topic)` returns a dense, source-cited plaintext brief
(or None) that the deep-research orchestrator injects as a high-trust synthetic
"source" so the synthesis step can build the lineage table, citation counts, and
chronology the user asked for. Network-guarded: any failure returns None and the
run proceeds on web evidence alone.
"""
import logging
from typing import Optional

import requests

logger = logging.getLogger("assistant.research.citations")

_API = "https://api.openalex.org/works"
_TIMEOUT = 12


def _get(params: dict, mailto: str) -> Optional[dict]:
    try:
        params = dict(params)
        if mailto:
            params["mailto"] = mailto
        r = requests.get(_API, params=params, timeout=_TIMEOUT,
                         headers={"User-Agent": "f5-research-bot/1.0"})
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as exc:
        logger.debug("openalex request failed: %s", exc)
        return None


def _authors(work: dict, n: int = 4) -> str:
    names = [a.get("author", {}).get("display_name", "")
             for a in (work.get("authorships") or [])[:n]]
    names = [x for x in names if x]
    s = ", ".join(names)
    if len(work.get("authorships") or []) > n:
        s += " et al."
    return s


def _venue(work: dict) -> str:
    loc = work.get("primary_location") or {}
    src = loc.get("source") or {}
    return src.get("display_name", "") or ""


def _fmt(work: dict) -> str:
    title = work.get("display_name") or "(untitled)"
    year = work.get("publication_year") or "n.d."
    cites = work.get("cited_by_count", 0)
    venue = _venue(work)
    doi = (work.get("doi") or "").replace("https://doi.org/", "")
    bits = [f'"{title}" ({year})']
    auth = _authors(work)
    if auth:
        bits.append(auth)
    bits.append(f"cited_by={cites}")
    if venue:
        bits.append(venue)
    if doi:
        bits.append(f"doi:{doi}")
    return " — ".join(bits)


def citation_lineage_brief(topic: str, mailto: str = "",
                           max_works: int = 12) -> Optional[str]:
    """Build a citation-lineage brief for a scholarly topic via OpenAlex.

    Strategy: (1) most-cited works relevant to the topic = the influential
    literature; (2) the earliest high-impact work among them = the likely
    seminal paper; (3) the most-cited works that CITE that seminal paper =
    its principal descendants. Returns plaintext bullets (SOURCE line first so
    it threads through the trust pipeline) or None."""
    data = _get({"search": topic, "sort": "cited_by_count:desc",
                 "per_page": max_works,
                 "select": "id,display_name,publication_year,cited_by_count,"
                           "authorships,primary_location,doi"}, mailto)
    if not data or not data.get("results"):
        return None
    works = data["results"]

    lines = ["SOURCE: PRIMARY",
             "Citation-graph data from OpenAlex (scholarly index), not web pages:",
             "", "Most-influential works on this topic (by citation count):"]
    for w in works:
        lines.append(f"- {_fmt(w)}")

    # Seminal = earliest work, but ONLY claim "seminal" if it is genuinely
    # foundational-era (pre-2000). Otherwise the true seminal paper simply isn't
    # in OpenAlex's result set, and labelling a 2000s paper "seminal" is wrong —
    # so we present the descendants as "related high-impact works" instead.
    dated = [w for w in works if w.get("publication_year")]
    if dated:
        earliest = min(dated, key=lambda w: w["publication_year"])
        is_seminal = earliest.get("publication_year", 9999) < 2000
        if is_seminal:
            lines += ["", f"Likely seminal/earliest high-impact work: {_fmt(earliest)}"]
        wid = (earliest.get("id") or "").rsplit("/", 1)[-1]
        if wid:
            desc = _get({"filter": f"cites:{wid}", "sort": "cited_by_count:desc",
                         "per_page": 8,
                         "select": "display_name,publication_year,cited_by_count,"
                                   "authorships,primary_location,doi"}, mailto)
            if desc and desc.get("results"):
                hdr = ("Principal descendants (most-cited works citing the seminal work):"
                       if is_seminal else
                       "Related high-impact works (most-cited works citing the earliest in set):")
                lines += ["", hdr]
                for w in desc["results"]:
                    lines.append(f"- {_fmt(w)}")
    text = "\n".join(lines).strip()
    return text or None

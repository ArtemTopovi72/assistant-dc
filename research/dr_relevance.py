"""The relevance gate — is this brief actually about the topic?

The multi-hop, contradiction and reflection passes all search AROUND the topic,
so an ambiguous entity drags in noise (the "Mayor Adams" politics page on an
"Adam optimizer" run). These helpers derive the topic's entity terms from the
DISTILLED core topic and word-boundary match briefs against them.

Two deliberately different strengths:

* the per-page gate (`_brief_is_relevant`, used while briefing) is scholarly-only
  and drops briefs one at a time with no all-fail protection;
* the final sweep (`_filter_offtopic_briefs`) runs for every category but carries
  a safety net — it never empties the brief set, so a term list that matches
  nothing degrades to no filtering rather than to an empty report.
"""
import re


# Source-relevance gate (Phase-9 repair): for scholarly topics, drop briefs that
# don't mention the core entity, so off-topic papers (e.g. a Josephson-effect
# paper on a "Di Zenzo structure tensor" query) never reach the report as PRIMARY.
_RELEVANCE_STOPWORDS = {
    "paper", "papers", "original", "study", "studies", "method", "methods",
    "analysis", "review", "using", "based", "image", "images", "data", "model",
    "models", "approach", "technique", "application", "applications", "work",
    "note", "gradient",  # 'gradient' too generic alone; require a stronger term
    "tensor", "tensors",  # too generic on a tensor topic — a "tensor optimization"
                          # paper isn't on-topic; require a distinctive term like
                          # 'structure'/'zenzo'/'multichannel'/'orientation'
    # Generic research/SEO/query words that are NOT topic entities. These leak in
    # when relevance terms come from a planned query ("arXiv … survey 2023"); left
    # unfiltered, a term like 'survey' matches every "A Survey of …" citation and
    # defeats the relevance gate (real bug: off-topic OpenAlex works survived).
    "survey", "surveys", "arxiv", "comparison", "benchmark", "benchmarks",
    "tutorial", "guide", "introduction", "overview", "implementation", "practices",
    "best", "results", "performance", "comprehensive", "systematic", "advances",
}


def _entity_terms(cite_query: str) -> list:
    toks = [t for t in re.findall(r"[a-zа-яё]{4,}", (cite_query or "").lower())
            if t not in _RELEVANCE_STOPWORDS and not t.isdigit()]
    return sorted(set(toks), key=len, reverse=True)[:6]


def _term_in(term: str, hay: str) -> bool:
    """WORD-BOUNDARY membership: 'adam' matches 'adam'/'adam's' but NOT 'adams'
    (so the NYC-mayor 'Adams' politics noise stops matching an 'Adam optimizer'
    topic). Substring matching let ambiguous short entities pull in junk."""
    return re.search(rf"(?<![a-zа-яё]){re.escape(term)}(?![a-zа-яё])", hay) is not None


def _term_in_any(hay: str, terms: list) -> bool:
    return any(_term_in(t, hay) for t in terms)


def _relevance_hits(hay: str, terms: list) -> int:
    return sum(1 for t in terms if _term_in(t, hay))


def _min_hits(terms: list) -> int:
    """Require ≥2 distinct topic terms when the topic has several, so a page that
    only mentions the single ambiguous entity (e.g. 'Adam' the biblical name on an
    'Adam optimizer' run) is NOT counted relevant. With ≤2 terms, 1 hit suffices."""
    return 2 if len(terms) >= 3 else 1


def _brief_is_relevant(title: str, brief: str, terms: list) -> bool:
    if not terms:
        return True
    hay = f"{title}\n{brief}".lower()
    return _relevance_hits(hay, terms) >= _min_hits(terms)


def _filter_offtopic_briefs(briefs: list, terms: list):
    """Drop briefs that mention NO topic entity term (word-boundary). Removes the
    retrieval noise an ambiguous entity drags in (e.g. 'Mayor Adams' politics, a
    'Python roadmap' outlink) before it can pollute the report, the contradiction
    check, or the source appendix. Returns (kept, dropped). No-op without terms."""
    if not terms:
        return briefs, []
    need = _min_hits(terms)
    kept, dropped = [], []
    for b in briefs:
        hay = f"{b.get('title', '')}\n{b.get('brief', '')}".lower()
        (kept if _relevance_hits(hay, terms) >= need else dropped).append(b)
    # Safety: never strip everything (a bad term list shouldn't empty the report).
    if not kept:
        return briefs, []
    return kept, dropped

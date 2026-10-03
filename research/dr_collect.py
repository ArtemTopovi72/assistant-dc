"""Stages 1/2 — decide what to search for, and gather the candidate sources.

`plan_queries` expands one topic into diverse search queries and classifies it
(category / recency / news vertical), with a heuristic fallback so a run never
starves for queries when the planner model misbehaves. `collect_sources` then
runs those queries, filters the results by domain policy, source type and junk
host, and ranks what survives.

`coverage_signals` and `evidence_stats` are the measurements the orchestrator's
corrective re-plan loop and the decision gate read: is the evidence thin, narrow,
clustered on one host, or skewed to one language?
"""
import logging
import turn_trace
import re
from concurrent.futures import as_completed
from pathlib import Path
from typing import Optional

import dr_settings as S
from dr_calls import _think_call
from dr_lang import _LANG_RE_CYR
from dr_extract import _jaccard
from dr_policy import (_apply_scholarly_floor, _CATEGORY_RECENCY_FLOOR, _NEWS_CATEGORIES,
                       _RECENCY_RANK, _RECENCY_TIMELIMIT, _TIMELESS_CATEGORIES,
                       _TRUST_ORDER)
from dr_progress import _Progress
from dr_state import _save_state
from dr_urls import (_cap_per_domain, _domain_allowed, _is_junk_source, _mutate_query,
                     _norm_url, _source_rank, _source_type_allowed, unwrap_url)
from prompts import QUERY_PLANNER_PROMPT
from search import raw_search_results
from utils import safe_json_from_llm

logger = logging.getLogger("assistant.research")


# Some categories go stale fast no matter what the planner guessed: a job posting


def plan_queries(ctx, topic: str, max_queries: int) -> tuple:
    """Expand a topic into diverse search queries via the LLM, with a heuristic
    fallback so the run never starves for queries.

    Returns ``(queries, profile)`` where profile = {"category", "recency",
    "timelimit", "news"} drives how the ddgs query is issued (general web vs the
    news vertical, and the freshness window)."""
    user = (f"Research topic: {topic}\n\n"
            f"Return up to {max_queries} diverse search queries as JSON.")
    # force_think (not prefill): on the reasoning finetune the "<think></think>"
    # prefill backfires (unclosed think -> empty), which is exactly how a verbose
    # request fell through to the heuristic and got searched verbatim. See
    # [[reasoning-toggle-prefill]].
    # The budget must cover BOTH the CoT and the JSON answer: the planner thinks
    # (~1500 tok) before it writes, and a tight cap truncates inside <think> ->
    # strip leaves "" (verified: 900 -> empty). DR_PLAN_TOKENS is sized well above
    # CoT length; it is deliberately NOT the whole context window, which on a
    # 262k-window model is 260k tokens of rope for a repetition loop to run into.
    out = _think_call(ctx, QUERY_PLANNER_PROMPT, user,
                      max_tokens=S.DR_PLAN_TOKENS, temperature=0.4)
    queries: list = []
    category, recency = "general", "any"
    core_topic = ""
    data = safe_json_from_llm(out or "")
    if data:
        if isinstance(data.get("queries"), list):
            for q in data["queries"]:
                if isinstance(q, str) and q.strip():
                    queries.append(q.strip())
        if isinstance(data.get("category"), str):
            category = data["category"].strip().lower() or "general"
        if isinstance(data.get("recency"), str):
            recency = data["recency"].strip().lower() or "any"
        if isinstance(data.get("core_topic"), str):
            core_topic = data["core_topic"].strip()

    # Seed with the DISTILLED subject, not the raw request. A long/instruction-style
    # topic (e.g. a multi-line Russian "write me a doctoral paper that..." prompt) must
    # never become a search query verbatim — search engines can't match a 500-char
    # essay, so every page comes back as irrelevant nav junk. Only fall back to the raw
    # topic when it is already short and clean and the LLM gave us no core_topic.
    seed = core_topic or (topic if len(topic) <= 120 and "\n" not in topic else _topic_first_line(topic))
    if seed:
        logger.info("Query plan: distilled core_topic=%r (raw len=%d)", seed, len(topic or ""))
    seen, ordered = set(), []
    for q in [seed] + queries:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(q)

    if len(ordered) < 3:  # LLM gave us nothing usable — heuristic expansion
        from datetime import date
        base = seed or topic
        for suffix in ("", f" {date.today().year}", " benchmark comparison", " github",
                       " review", " release notes", " documentation", " vs alternatives"):
            cand = (base + suffix).strip()
            if cand.lower() not in seen:
                seen.add(cand.lower())
                ordered.append(cand)

    if recency not in _RECENCY_TIMELIMIT:
        recency = "any"
    # Timeless/reference topics: never recency-restrict (would hide old authorities).
    if category in _TIMELESS_CATEGORIES:
        recency = "any"
    # Enforce the per-category freshness floor (jobs/news/finance go stale fast).
    floor = _CATEGORY_RECENCY_FLOOR.get(category)
    if floor and _RECENCY_RANK[recency] > _RECENCY_RANK[floor]:
        recency = floor
    profile = {
        "category": category,
        "recency": recency,
        "timelimit": _RECENCY_TIMELIMIT[recency],
        "news": category in _NEWS_CATEGORIES,
        # The distilled subject (English noun phrase). Downstream stages that must
        # match the CRAWLED (mostly-English) pages — relevance gating, the scholarly
        # citation query, the report title — use THIS, not the raw request, which may
        # be a long multilingual instruction whose words never appear in the sources.
        "core_topic": seed or topic,
    }
    # Drop near-duplicate paraphrases so the search budget buys real coverage,
    # not the same query four times (Phase 5).
    final = diversify_queries(ordered, max_queries)
    # A floor as well as a filter. When the planner fails, the heuristic above
    # appends short suffixes to the topic -- and on a LONG topic a two-word
    # suffix barely moves the token set, so every variant scores as a paraphrase
    # of the first. Measured on a live run: eight heuristic candidates collapsed
    # to ONE, and the whole research run went out with a single query. The
    # de-duplicator was starving the guard that exists to prevent starvation.
    #
    # Only tops up from candidates already built, never invents one, and never
    # re-adds an exact duplicate.
    if len(final) < min(MIN_QUERIES, max_queries, len(ordered)):
        have = {q.lower() for q in final}
        for q in ordered:
            if q.lower() in have or len(q) > MAX_QUERY_CHARS:
                continue
            final.append(q)
            have.add(q.lower())
            if len(final) >= min(MIN_QUERIES, max_queries):
                break
        logger.info("Query plan: topped up to %d queries after diversify left %d",
                    len(final), len(final))
    logger.info("Query plan: category=%s recency=%s news=%s (%d/%d queries after diversify)",
                category, recency, profile["news"], len(final), len(ordered))
    return final, profile


def _topic_first_line(topic: str) -> str:
    """Last-resort distillation when the LLM gives no core_topic: take the first
    non-empty line, strip a leading imperative, and cap length so a runaway
    instruction never becomes the search seed."""
    first = next((ln.strip() for ln in (topic or "").splitlines() if ln.strip()), topic or "")
    first = re.sub(r"^(perform|do|conduct|provide|investigate|research|find|explain|analy[sz]e|"
                   r"сделай|напиши|нужн[оаы]|дай|расскажи|подготов[ьи])\b[ :,]*",
                   "", first, flags=re.IGNORECASE)
    return first[:120].strip()


def _citation_query(topic: str, queries: list) -> str:
    """Build a SHORT scholarly-search query (OpenAlex chokes on a multi-line
    prompt). Prefer the shortest concrete planned query that isn't the raw topic;
    fall back to the trimmed first line of the topic."""
    cands = [q for q in (queries or [])
             if q and q.strip().lower() != topic.strip().lower() and 6 <= len(q) <= 90]
    if cands:
        return min(cands, key=len)
    first = next((ln.strip() for ln in (topic or "").splitlines() if ln.strip()), topic or "")
    # Drop leading imperatives so the entity leads the query.
    first = re.sub(r"^(perform|do|conduct|provide|investigate|research|find|explain|analy[sz]e)\b[ :]*",
                   "", first.strip(), flags=re.IGNORECASE)
    return first[:90].strip(" .:-") or (topic or "")[:90]


def _corrective_queries(ctx, topic: str, profile: dict, existing: list,
                        max_queries: int) -> list:
    """Generate a small batch of BROADENING queries for the single corrective
    re-plan (Phase 3). Nudges the planner toward alternate terms, the primary
    source, and the other language, then drops anything overlapping the queries
    we already ran. Bounded and best-effort — failure just means no re-plan."""
    cat = profile.get("category", "general")
    hint = (f"{topic}\n\n[The first search was too narrow. Produce DIFFERENT, "
            f"broader queries: alternate terminology and synonyms, the original / "
            f"primary / official source, and ")
    hint += ("the other language (Russian and English both). " if cat in
             ("local", "entertainment", "people") else "an entity-free angle. ")
    hint += "Do NOT repeat these earlier queries: " + "; ".join(existing[:8]) + "]"
    try:
        extra, _ = plan_queries(ctx, hint, max_queries)
    except Exception as exc:
        logger.warning("corrective re-plan failed: %s", exc)
        return []
    have = {q.lower() for q in existing}
    fresh = [q for q in extra if q.lower() not in have and q.lower() != topic.lower()]
    return diversify_queries(fresh, max(2, max_queries // 2))


# --------------------------------------------------------------------------- #
# Stage 2 — broad source collection
# --------------------------------------------------------------------------- #
def _search_one_query(ctx, q: str, per_query: int, timelimit, use_news: bool) -> list:
    """Search one query (+ optional news vertical) with the zero-hit mutation
    retry. Pure w.r.t. shared state — safe to run on a worker thread. Never
    raises; returns [] on total failure."""
    try:
        hits = raw_search_results(ctx, q, per_query, timelimit=timelimit)
        if use_news:  # for current-events topics, also pull fresh articles
            hits += raw_search_results(ctx, q, per_query, timelimit=timelimit, news=True)
    except Exception as exc:  # one bad query must not abort the run
        logger.warning("Search failed for %r: %s", q, exc)
        hits = []
    # Retry/resilience: a zero-hit query gets mutated and retried instead of
    # silently contributing nothing (manual-control DR_QUERY_MUTATION_ATTEMPTS).
    mutation_attempt = 0
    while not hits and mutation_attempt < S.DR_QUERY_MUTATION_ATTEMPTS:
        mutated = _mutate_query(q, mutation_attempt)
        mutation_attempt += 1
        if not mutated:
            continue
        try:
            hits = raw_search_results(ctx, mutated, per_query, timelimit=timelimit)
        except Exception:
            hits = []
        if hits:
            logger.info("Query mutation recovered results: %r -> %r", q, mutated)
    return hits


def collect_sources(ctx, queries: list, per_query: int, prog: _Progress,
                    run_dir: Path, profile: Optional[dict] = None) -> list:
    """Search every query, merge + dedupe hits by normalized URL.

    ``profile`` (from plan_queries) tunes the ddgs call: a recency ``timelimit``
    and, for news topics, an extra pass over the news vertical."""
    profile = profile or {}
    timelimit = profile.get("timelimit")
    use_news = bool(profile.get("news"))
    category = profile.get("category", "general")

    seen, sources = set(), []
    dropped = 0

    def _ingest(hits, q):
        """Dedup + junk/domain/type-filter one query's hits into `sources`."""
        nonlocal dropped
        for h in hits:
            # Unwrap a translator/reader wrapper FIRST, before anything is
            # decided about this URL. A wrapped link was cited in a delivered
            # report as though tr-page.yandex.ru were the source, and it also
            # hid the real host from the dedupe key and from every trust
            # decision -- an English Wikipedia article was scored as an unknown
            # yandex subdomain.
            h["href"] = unwrap_url(h.get("href") or "")
            key = _norm_url(h["href"])
            if key in seen:
                continue
            if _is_junk_source(h["href"]):  # pirate/Q&A/SEO-farm domains
                dropped += 1
                logger.debug("Dropped junk source: %s", h["href"])
                continue
            if not _domain_allowed(h["href"], S.DR_DOMAIN_WHITELIST, S.DR_DOMAIN_BLACKLIST):
                dropped += 1
                continue
            if not _source_type_allowed(h["href"], S.DR_SOURCE_TYPES_EXCLUDE):
                dropped += 1
                continue
            seen.add(key)
            h["query"] = q
            sources.append(h)

    conc = max(1, S.DR_SEARCH_CONCURRENCY)
    if conc > 1 and len(queries) > 1:
        # Parallel search: queries are independent network calls, so a bounded pool
        # overlaps their latency (incl. the dead-engine ddgs timeouts). Hits are
        # ingested on THIS thread as futures complete — dedup stays single-threaded.
        done_n = 0
        with turn_trace.Pool(max_workers=conc) as ex:
            futs = {ex.submit(_search_one_query, ctx, q, per_query, timelimit, use_news): q
                    for q in queries}
            for fut in as_completed(futs):
                if ctx is not None and ctx.is_cancelled():
                    break
                q = futs[fut]
                done_n += 1
                prog.update("Searching", f"[{done_n}/{len(queries)}] {q}")
                try:
                    hits = fut.result()
                except Exception as exc:
                    logger.warning("Search failed for %r: %s", q, exc)
                    hits = []
                _ingest(hits, q)
                prog.update(sources=len(sources))
                _save_state(run_dir, phase="searching", queries=queries,
                            sources=[s["href"] for s in sources])
    else:
        for i, q in enumerate(queries, 1):
            if ctx is not None and ctx.is_cancelled():
                break
            prog.update("Searching", f"[{i}/{len(queries)}] {q}")
            _ingest(_search_one_query(ctx, q, per_query, timelimit, use_news), q)
            prog.update(sources=len(sources))
            _save_state(run_dir, phase="searching", queries=queries,
                        sources=[s["href"] for s in sources])
    # Crawl high-authority sources first so the page budget is spent on the most
    # credible material; ordinary sources only fill whatever budget is left.
    sources.sort(key=lambda s: _source_rank(s["href"], category))
    sources = _apply_scholarly_floor(sources, category)
    sources = _cap_per_domain(sources, S.DR_MAX_PAGES_PER_DOMAIN)
    if S.DR_MAX_SOURCES > 0:
        sources = sources[:S.DR_MAX_SOURCES]
    logger.info("Collected %d unique sources from %d queries (%d junk/filtered dropped)",
                len(sources), len(queries), dropped)
    return sources


# --------------------------------------------------------------------------- #
# Coverage / diversity check (Phase 3) + query diversification (Phase 5)
# --------------------------------------------------------------------------- #
def _query_lang(q: str) -> str:
    return "ru" if _LANG_RE_CYR.search(q or "") else "en"


MIN_QUERIES = 3
# A search engine cannot match an essay: a long "query" comes back as nav junk.
# The planner path already distils, and the top-up below must not undo that --
# it did, and an existing check caught it.
MAX_QUERY_CHARS = 130


def diversify_queries(queries: list, max_queries: int) -> list:
    """Drop near-duplicate queries (token-set Jaccard) so the planner can't waste
    the search budget on paraphrases. Order-preserving; first occurrence wins.

    A pure filter: it never invents a query and never keeps one it judged a
    paraphrase. The FLOOR that stops a run going out with a single query lives
    at the call site, which is the only place that knows how few candidates
    there were to begin with -- see plan_queries.
    """
    kept, kept_tokens = [], []
    for q in queries:
        toks = frozenset(re.sub(r"\s+", " ", q.lower()).split())
        if not toks:
            continue
        if any(_jaccard(toks, t) >= 0.8 for t in kept_tokens):
            continue
        kept.append(q)
        kept_tokens.append(toks)
        if len(kept) >= max_queries:
            break
    return kept


def coverage_signals(sources: list, profile: dict) -> dict:
    """Inspect a collected source set for weakness/skew. Returns a dict with a
    boolean ``weak`` and human-readable ``reasons`` driving the one re-plan."""
    n = len(sources)
    domains = {}
    langs = {"ru": 0, "en": 0}
    for s in sources:
        domains[s["domain"]] = domains.get(s["domain"], 0) + 1
        langs[_query_lang(s.get("query", ""))] += 1
    n_domains = len(domains)
    top_share = (max(domains.values()) / n) if n else 1.0
    reasons = []
    if n < 6:
        reasons.append(f"thin: only {n} sources collected")
    if n_domains < 3:
        reasons.append(f"narrow: only {n_domains} distinct domain(s)")
    if n and top_share > 0.6:
        reasons.append(f"clustered: one domain is {top_share:.0%} of sources")
    # Language skew only matters for inherently-local/entertainment topics where
    # the native-language web holds the real evidence.
    if profile.get("category") in ("local", "entertainment", "people") and n:
        if langs["ru"] == 0 or langs["en"] == 0:
            reasons.append("single-language coverage for a locale-sensitive topic")
    return {"weak": bool(reasons), "reasons": reasons,
            "n_sources": n, "n_domains": n_domains, "top_share": round(top_share, 2)}


# --------------------------------------------------------------------------- #
# Computed confidence inputs (Phase 5)
#
# Confidence used to be whatever label the model felt like writing. We now feed
# synthesis the hard evidence structure (independent clusters, max authority,
# trust mix) AND a rule that ties confidence levels to that structure, so "High"
# means the same thing across runs and can't be claimed on one weak source.
# --------------------------------------------------------------------------- #
def evidence_stats(briefs: list) -> dict:
    """Aggregate the evidence structure used to constrain confidence."""
    independent = len(briefs)  # each kept brief is one collapsed evidence cluster
    trust_counts = {t: 0 for t in _TRUST_ORDER}
    max_auth = 0.0
    for b in briefs:
        trust_counts[b.get("trust", "COMMUNITY")] = \
            trust_counts.get(b.get("trust", "COMMUNITY"), 0) + 1
        max_auth = max(max_auth, b.get("authority", 0.0))
    strong = trust_counts["PRIMARY"] + trust_counts["SECONDARY"]
    return {"independent_clusters": independent, "strong_sources": strong,
            "trust_counts": trust_counts, "max_authority": round(max_auth, 2)}

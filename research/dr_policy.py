"""Which sources a research run trusts, and how hard it digs.

Three related policies that decide the SHAPE of a run before any page is
fetched or any word is written:

* topic intent  — is this a "help me book/buy one" ask (an actionable options
  list) or a research question (a survey)?  `_is_practical_request`.
* source quality — scholarly topics prune the undifferentiated middle tier,
  and the brief model's self-rated trust is clamped by host provenance so a
  well-written content farm can't promote itself to PRIMARY.
* depth caps    — how many queries/pages/tokens/reflection passes a
  quick/standard/deep run gets.  `_resolve_caps`.

The caps live here rather than in dr_timing because they are computed from the
DR_* knobs, which `apply_overrides` swaps out for a single run. How LONG each
profile takes (the measured ETA) is dr_timing's job.
"""
import logging
import re

import dr_settings as S
from dr_urls import (_authority_score, _domain, _is_community_host, _is_landing_url,
                     _AUTH_BASE_SCORE, _AUTH_CATEGORY_SCORE, _MID_SCORE)

logger = logging.getLogger("assistant.research")


# --------------------------------------------------------------------------- #
# Source quality filtering
#
# The crawler used to ingest whatever the search backend returned, so a niche
# technical query could end up sourced from pirate-streaming, Q&A, or SEO
# glossary farms — junk that the synthesis step then "filled in" with
# hallucinations. We now hard-drop known-junk domains and crawl high-authority
# sources first, so the bounded page budget is spent on credible material.
# --------------------------------------------------------------------------- #


# Categories where the answer should rest on scholarly/authoritative sources and
# random blogs/tutorials/SEO add noise rather than signal (Phase 9, critique #10).
_SCHOLARLY_CATEGORIES = {"science", "legal", "health"}


# Categories whose questions are almost always "help me DO/BUY/BOOK this", not
# "explain this field to me". NOTE: "product" is deliberately NOT here — unlike
# local/shopping/travel (near-always an actionable ask), "product" also covers
# pure comparison/spec questions ("how does the K-700 differ from the K-744",
# "Kubota vs John Deere") that read like research, not a booking/purchase intent.
# A product-category topic only counts as practical when it ALSO matches the
# explicit action regex below (buy/price/ticket/etc.) — see _is_practical_request.
_PRACTICAL_CATEGORIES = {"local", "shopping", "travel"}


# Comparison/factual phrasing. When this fires, the topic is a "explain how these
# differ" question, not a "help me get/buy/book one" question — it must win over
# a practical CATEGORY guess (e.g. "product") even though a comparison can
# incidentally use practical-sounding words like "price" in passing.


# Explicit asks for concrete, actionable results. Matched against the RAW request
# (the user's own words) — the distilled core_topic deliberately strips exactly
# these verbs, so intent must be read before distillation.
def _ask_kind(topic: str) -> str:
    """compare | practical | research -- the model's read of the raw request."""
    import intent
    return intent.ask_choice(
        "A user asked for research on: {text}\n\nWhich is it? compare = it names two "
        "or more things and asks how they differ; practical = concrete options to act on (links, where to buy "
        "or book, prices, schedules, how to get there); research = an analysis or "
        "explanation.", topic or "", ("compare", "practical", "research"), "research")


def _is_practical_request(topic: str, profile: dict) -> bool:
    """True when the user wants bookable/actionable options rather than an analysis.

    Deliberately conservative: a scholarly category always wins (a legal or medical
    question mentioning "cost" is still a research question), and an explicit
    comparison/factual phrasing ("how does X differ from Y") always wins too — a
    comparison is a research question even when its category is "product" (e.g.
    two tractor models, two phones), and even if it happens to mention a
    practical-sounding word in passing. Otherwise it fires on a practical
    CATEGORY, or on an explicit ask for links/booking/prices/schedules in the raw
    request.
    """
    category = (profile or {}).get("category", "general")
    if category in _SCHOLARLY_CATEGORIES:
        return False
    kind = _ask_kind(topic)
    if kind == "compare":
        return False
    if category in _PRACTICAL_CATEGORIES:
        return True
    return kind == "practical"


_SCHOLARLY_FLOOR_MIN_AUTHORITY = 4   # only prune once we have this many strong sources


_SCHOLARLY_FLOOR_KEEP = 12           # never prune below this many sources total


def _apply_scholarly_floor(sources: list, category: str) -> list:
    """For scholarly categories, drop merely-MID hosts (random tutorials, SEO)
    once we already hold enough authority-grade sources. Conservative: only
    activates with >=4 strong sources and never trims below 12 total, so a
    thin run is never starved. Community-primary (e.g. a relevant GitHub) is
    kept — only the undifferentiated MID tier is pruned."""
    if category not in _SCHOLARLY_CATEGORIES:
        return sources
    strong = [s for s in sources if _authority_score(s["href"], category) >= _AUTH_CATEGORY_SCORE]
    if len(strong) < _SCHOLARLY_FLOOR_MIN_AUTHORITY:
        return sources
    kept = [s for s in sources
            if _authority_score(s["href"], category) > _MID_SCORE]
    if len(kept) < _SCHOLARLY_FLOOR_KEEP:
        # top up with the best of the rest to avoid starving the crawl
        rest = [s for s in sources if s not in kept]
        rest.sort(key=lambda s: _authority_score(s["href"], category), reverse=True)
        kept += rest[:_SCHOLARLY_FLOOR_KEEP - len(kept)]
    dropped = len(sources) - len(kept)
    if dropped:
        logger.info("Scholarly floor (%s): dropped %d non-scholarly sources", category, dropped)
    return kept


# --------------------------------------------------------------------------- #
# Trust floor/ceiling (Phase 4)
#
# The brief model self-labels each page PRIMARY/SECONDARY/COMMUNITY/LOW, but a
# well-written content farm reads "authoritative" to a 9B. We clamp the label by
# provenance: a genuine authority host can't be dragged below SECONDARY, and a
# community/blog/social host can't be promoted above COMMUNITY (except where it
# legitimately IS the primary source, e.g. a venue's VK group for local topics).
# --------------------------------------------------------------------------- #
_TRUST_RANK = {"PRIMARY": 0, "SECONDARY": 1, "COMMUNITY": 2, "LOW": 3}
# Trust tiers, most → least authoritative. Drives report weighting + appendix order,
# and the trust mix line every stage prints.
_TRUST_ORDER = ("PRIMARY", "SECONDARY", "COMMUNITY", "LOW")
_TRUST_LABEL = {
    "PRIMARY":   "🟢 Primary (original / official)",
    "SECONDARY": "🔵 Secondary (reputable analysis)",
    "COMMUNITY": "🟡 Community (forum / blog — leads only)",
    "LOW":       "🔴 Low trust (weakly supported)",
}


_RANK_TRUST = {v: k for k, v in _TRUST_RANK.items()}


_ENCYCLOPEDIC_HOSTS = ("wikipedia.org", "britannica.com", "scholarpedia.org")


def _apply_trust_floor(trust: str, url: str, category: str = "general") -> str:
    """Clamp the model's self-rated trust by host provenance. Returns the
    possibly-adjusted tier."""
    trust = (trust or "COMMUNITY").upper()
    if trust not in _TRUST_RANK:
        trust = "COMMUNITY"
    score = _authority_score(url, category)
    host = _domain(url)
    # Floor: established authority hosts are at least SECONDARY.
    if score >= _AUTH_BASE_SCORE and _TRUST_RANK[trust] > _TRUST_RANK["SECONDARY"]:
        trust = "SECONDARY"
    elif score >= _AUTH_CATEGORY_SCORE and _TRUST_RANK[trust] > _TRUST_RANK["SECONDARY"]:
        trust = "SECONDARY"
    # Encyclopedic/tertiary ceiling: Wikipedia/Britannica are tertiary references —
    # never a PRIMARY source no matter how the brief model rated the page.
    if any(s in host for s in _ENCYCLOPEDIC_HOSTS) and _TRUST_RANK[trust] < _TRUST_RANK["SECONDARY"]:
        trust = "SECONDARY"
    # A site-root / landing page is never an original document → never PRIMARY.
    if _is_landing_url(url) and _TRUST_RANK[trust] < _TRUST_RANK["SECONDARY"]:
        trust = "SECONDARY"
    # Ceiling: community/UGC hosts cannot be PRIMARY/SECONDARY (no self-promotion).
    if _is_community_host(url, category) and _TRUST_RANK[trust] < _TRUST_RANK["COMMUNITY"]:
        trust = "COMMUNITY"
    return trust


# --------------------------------------------------------------------------- #
# Depth profiles.
#
# The caps live here, not in dr_timing, because they are computed from the DR_*
# globals of THIS module — the ones `apply_overrides` swaps out for a single run.
# How long each profile takes (the measured ETA) is dr_timing's job.
# --------------------------------------------------------------------------- #


def _resolve_caps(depth: str) -> dict:
    """Map a depth profile to concrete caps."""
    depth = (depth or "standard").lower()
    # `reflect_iters` / `reflect_budget` are what actually separate the three
    # options in WALL-CLOCK terms. A reflection pass re-crawls and re-briefs, and
    # briefing is the single most expensive stage (measured: 22-32 of the 42
    # minutes a "quick" run was taking). A quick run is one pass by definition;
    # paying for a second is what made the 15-minute option take 42.
    if depth == "quick":
        caps = dict(max_queries=max(4, S.DR_MAX_QUERIES // 2), max_pages=max(6, S.DR_MAX_PAGES // 3),
                    depth=0, links_per_page=S.DR_LINKS_PER_PAGE,
                    report_tokens=S.DR_REPORT_TOKENS_QUICK,
                    reflect_iters=0, reflect_budget=0)
    elif depth == "deep":
        caps = dict(max_queries=S.DR_MAX_QUERIES + 4, max_pages=S.DR_MAX_PAGES + 16,
                    depth=max(S.DR_MAX_DEPTH, 1), links_per_page=S.DR_LINKS_PER_PAGE + 2,
                    report_tokens=S.DR_REPORT_TOKENS_DEEP,
                    reflect_iters=S.DR_REFLECTION_MAX_ITERATIONS,
                    reflect_budget=S.DR_REFLECTION_RESEARCH_BUDGET)
    else:
        caps = dict(max_queries=S.DR_MAX_QUERIES, max_pages=S.DR_MAX_PAGES,
                    depth=S.DR_MAX_DEPTH, links_per_page=S.DR_LINKS_PER_PAGE,
                    report_tokens=S.DR_REPORT_TOKENS_STANDARD,
                    reflect_iters=min(1, S.DR_REFLECTION_MAX_ITERATIONS),
                    reflect_budget=max(0, S.DR_REFLECTION_RESEARCH_BUDGET // 2))
    # A manual "Report budget" override wins over the per-depth default (0 = auto).
    if S.DR_REPORT_TOKENS_OVERRIDE and S.DR_REPORT_TOKENS_OVERRIDE > 0:
        caps["report_tokens"] = S.DR_REPORT_TOKENS_OVERRIDE
    return caps


# --------------------------------------------------------------------------- #
# Stage 1 — query expansion
# --------------------------------------------------------------------------- #
# How the planner's "recency" maps to a ddgs timelimit, and which categories
# are time-sensitive enough to also pull from the news vertical.
_RECENCY_TIMELIMIT = {"day": "d", "week": "w", "month": "m", "year": "y", "any": None}


_NEWS_CATEGORIES = {"news"}


# Recency ranked fresh→stale, for comparing/clamping.
_RECENCY_RANK = {"day": 0, "week": 1, "month": 2, "year": 3, "any": 4}


# or a breaking story from two years ago is noise. Clamp recency no looser than
# this floor for those categories.
_CATEGORY_RECENCY_FLOOR = {"jobs": "month", "news": "week", "finance": "month"}


# Timeless / reference categories must NEVER be recency-restricted: a ddgs
# timelimit drops every page older than the window, which for these topics is
# exactly the authoritative literature (e.g. a recency=month filter on "Di Zenzo
# structure tensor" hid the 1986 paper, the Wikipedia article, and every classic
# source, leaving only generic recently-indexed pages). Force "any" regardless of
# what the planner guessed.
_TIMELESS_CATEGORIES = {"science", "legal", "entertainment", "people", "general"}

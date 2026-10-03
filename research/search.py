"""Web search tool: ddgs fetch -> dedupe/format -> optional fact distillation."""
import re
import turn_trace
import logging
import os
import time
from typing import Optional
from urllib.parse import urlparse

from config import (
    SEARCH_MAX_RESULTS, SEARCH_REGION, SEARCH_SAFESEARCH, SEARCH_TIMELIMIT,
    SEARCH_BACKEND, SEARCH_SNIPPET_CHARS, SEARCH_TOTAL_CHARS, SEARCH_DISTILL,
    SEARCH_MAX_PER_DOMAIN,
    SEARCH_DISTILL_MAX_TOKENS, SEARCH_READ_PAGES, SEARCH_PAGE_CHARS, SEARCH_FETCH_WORKERS, SEARCH_FOLLOWUP_ROUND,
)
from prompts import SEARCHER_PROMPT
from llm import call_llm_simple
from utils import truncate_text, throttle_external_calls

logger = logging.getLogger("assistant.search")

# Sentinel results the orchestrator checks for (so it won't try to distill them).
NO_RESULTS = "No results found for your query."
SEARCH_FAILED = "Search failed."
SEARCH_BLOCKED = ("Search refused: this query was not sent to the web -- searching for "
                  "it may be illegal in Russia. Tell the user so, briefly.")

# Every query that leaves the machine is read by the model first. Russian law
# punishes the SEARCH itself for some content (KoAP 13.53, 2025: deliberately
# looking up extremist materials), so the check sits in front of ddgs, not
# behind it. Terror attacks and explosions are refused even as news: the
# owner's call (10-03) is that searching them is a risk here too.
_OUTBOUND_Q = (
    "A web search query is about to be sent from Russia: {text}\n\n"
    "Could SEARCHING for this get the user into trouble with Russian law? "
    "Be strict: when in doubt, answer yes. Yes for anything about: "
    "terror attacks, bombings, explosions or attackers (even as news or history); "
    "explosives, weapons or ammunition of any kind (making, buying, reviews, "
    "news about them); drugs of any kind; pornography or any sexual/erotic "
    "content; sex with minors; organisations, people or materials banned in "
    "Russia as extremist, terrorist or undesirable (incl. the 'LGBT movement', "
    "FBK/Navalny, IS, their books, videos, songs, symbols); calls for violence "
    "or riots; the war or the army beyond official news (losses, desertion, "
    "avoiding mobilisation); bypassing blocks (VPN, Tor, mirrors of banned "
    "sites); online casinos and betting; suicide or self-harm methods; ways to "
    "commit any crime (hacking, fraud, forged documents). "
    "No means ordinary everyday topics: weather, recipes, science, health, "
    "history (not of attacks), law texts, shopping for ordinary goods, tech.")


def outbound_blocked(query: str) -> bool:
    """True when `query` must not go to the web. The model being down blocks
    too: an unchecked query is what this guard exists to stop."""
    import intent
    # test runs without a stub must still reach their fake ddgs
    offline_test = bool(os.getenv("F5_TEST_RUN")) and not os.getenv("INTENT_LIVE")
    blocked = intent.ask_yes(_OUTBOUND_Q, query, default=not offline_test)
    if blocked:
        logger.warning("outbound search refused: %r", query[:200])
    return blocked


# A URL with no meaningful path (site root) or whose path is a bare section index
# ("/news/", "/sport/", "/tag/x"). These are navigation, not the story the user asked
# for. Kept deliberately narrow: a real article almost always has a slug or an id.
_SECTION_TAILS = frozenset({
    "news", "novosti", "sport", "politics", "world", "business", "tech", "culture",
    "category", "categories", "tag", "tags", "topic", "topics", "section", "rubric",
    "articles", "blog", "feed", "rss", "all", "latest", "index", "main", "home",
})


# Path segments that make whatever follows them an INDEX rather than a document.
_INDEX_PREFIXES = frozenset({
    "tag", "tags", "category", "categories", "topic", "topics", "rubric", "section",
    "author", "authors", "search", "archive",
})


def _is_landing_url(url: str) -> bool:
    try:
        parts = [p for p in urlparse(url).path.split("/") if p]
    except Exception:
        return False
    if not parts:                      # https://ria.ru/
        return True
    if len(parts) > 2:                 # deep path -> almost certainly a document
        return False
    tail = parts[-1].lower()
    if tail.endswith((".html", ".htm", ".php", ".shtml")):
        return False
    if any(ch.isdigit() for ch in tail):   # date/id slugs are documents
        return False
    # Index pages where the marker leads the path: /tag/ukraina, /category/sport.
    if len(parts) == 2 and parts[0].lower() in _INDEX_PREFIXES:
        return True
    return tail in _SECTION_TAILS or (len(parts) == 1 and "-" not in tail)


def _domain(url: str) -> str:
    """Bare registrable host, e.g. 'https://www.example.com/x' -> 'example.com'."""
    try:
        net = urlparse(url).netloc.lower()
        return net[4:] if net.startswith("www.") else net
    except Exception:
        return url


# Sentinel so callers can pass timelimit=None ("all time") and still be
# distinguished from "caller didn't specify, use the config default".
_UNSET = object()


def _ddgs_retry(ctx, do_query, error_label: str) -> list:
    """Run ``do_query()`` (a zero-arg callable issuing one ddgs call and returning
    a list) with throttling and one retry. Shared by the text/news search and the
    image search below so they don't drift in retry/backoff behavior: one retry,
    a 0.8s backoff, ddgs 'auto' backend rotates engines so a second try often
    succeeds where the first came back empty or raised.
    """
    out = []
    for attempt in range(2):
        if ctx is not None:
            throttle_external_calls(ctx)
        try:
            out = do_query()
            if out:
                break
        except Exception as exc:
            logger.error("%s (attempt %d): %s", error_label, attempt + 1, exc)
            if attempt == 0:
                time.sleep(0.8)
                continue
            return []
    return out


def _region_for(ctx) -> str:
    """A Russian speaker asking about «Эрмитаж» got the Norfolk, Virginia
    museum from a worldwide English search (live 2026-09-28). An explicit
    SEARCH_REGION setting wins; the worldwide default follows the reader."""
    if SEARCH_REGION != "wt-wt":
        return SEARCH_REGION
    return "ru-ru" if str(getattr(ctx, "reply_lang", "") or "").startswith("ru") else SEARCH_REGION


def raw_search_results(ctx, query: str, max_results: Optional[int] = None, *,
                       timelimit=_UNSET, news: bool = False) -> list:
    """Run a web search and return structured hits (no formatting, no distill).

    Each hit is a dict: {"title", "href", "body", "domain"}. Returns [] on failure
    or no results. Shared by the formatter below and the deep-research crawler so
    they don't drift in how queries are issued.

    Args:
        timelimit: recency window ``d``/``w``/``m``/``y`` or ``None`` for all-time.
            Left unset, falls back to ``SEARCH_TIMELIMIT``. Set this to ``w``/``m``
            for news so stale pages don't dominate.
        news: when True, query the ddgs *news* vertical (fresh, dated articles
            from news outlets) instead of the general web index.
    """
    from ddgs import DDGS
    if outbound_blocked(query):
        return []
    max_results = max_results or SEARCH_MAX_RESULTS
    tl = SEARCH_TIMELIMIT if timelimit is _UNSET else timelimit
    region = _region_for(ctx)

    def _do_query():
        with DDGS() as ddgs:
            if news:
                # news() has no `backend` arg and returns dated articles.
                return list(ddgs.news(
                    query,
                    region=region,
                    safesearch=SEARCH_SAFESEARCH,
                    timelimit=tl,
                    max_results=max_results,
                ))
            return list(ddgs.text(
                query,
                region=region,
                safesearch=SEARCH_SAFESEARCH,
                timelimit=tl,
                backend=SEARCH_BACKEND,
                max_results=max_results,
            ))

    results = _ddgs_retry(ctx, _do_query, "Search error")

    hits = []
    for r in results:
        # text() returns "href"; news() returns "url" — accept either.
        href = (r.get("href") or r.get("url") or "").strip()
        if not href:
            continue
        hit = {
            "title": (r.get("title") or "No title").strip(),
            "href": href,
            "body": " ".join((r.get("body") or "").split()),
            "domain": _domain(href),
        }
        if r.get("date"):   # news() supplies a publication date — keep it
            hit["date"] = r["date"]
        hits.append(hit)
    return hits


def _read_page(url: str) -> str:
    """Main text of one page ('' on any failure). Reuses the deep-research fetcher:
    SSRF-checked redirects, byte caps, PDFs, trafilatura extraction."""
    try:
        from dr_crawl import _fetch_page
        from dr_extract import _extract_text
        html, pdf_text = _fetch_page(url)
        text = pdf_text or (_extract_text(html, url) if html else None)
        return " ".join((text or "").split())
    except Exception as exc:
        logger.debug("read page failed %s: %s", url, exc)
        return ""


_SENT_RE = re.compile(r"(?<=[.!?…])\s+|\s*\|\s*\|\s*")


def _stems(text: str) -> set:
    # A 5-letter prefix is a crude stem that survives Russian endings
    # ("биткоина"/"биткоин"); short words are mostly stop words.
    return {w[:5] for w in re.findall(r"\w{4,}", (text or "").lower())}


def best_passages(text: str, query: str, cap: int) -> str:
    """The parts of a page that answer the query, in page order, within `cap`.

    The first `cap` characters of a page are menus and ledes as often as not;
    the price table or the answering paragraph sits further down. A short opening
    sentence stays (it says what the page is), then the sentences sharing the
    most query stems, plus their neighbours, until the budget is spent.
    """
    if len(text) <= cap:
        return text
    sents = [s for s in _SENT_RE.split(text) if s.strip()]
    q = _stems(query)
    if not q or len(sents) < 3:
        return text[:cap]
    score = [len(q & _stems(s)) for s in sents]
    # A page with no full stops in its menu yields the whole menu as "sentence"
    # one (rbc.ru): only a short opening is a heading worth keeping.
    keep, used = ({0}, len(sents[0])) if len(sents[0]) <= 200 else (set(), 0)
    for i in sorted(range(len(sents)), key=lambda i: (-score[i], i)):
        if score[i] == 0 or used >= cap:
            break
        for j in (i, i + 1):
            if j < len(sents) and j not in keep and used + len(sents[j]) <= cap:
                keep.add(j); used += len(sents[j]) + 1
    out, prev = [], -1
    for i in sorted(keep):
        if prev >= 0 and i != prev + 1:
            out.append("…")
        out.append(sents[i]); prev = i
    return " ".join(out)


# News asks for the last days; a week-old window drops the evergreen archive pages.
def _is_news(query: str) -> bool:
    import intent
    return intent.ask_yes("A web search query: {text}\n\nDoes it ask for recent news "
                          "or headlines (what happened lately)?", query)


def _read_pages(urls: list) -> dict:
    """{url: text} for the pages that could be read, fetched concurrently."""
    if not urls:
        return {}
    with turn_trace.Pool(max_workers=max(1, min(SEARCH_FETCH_WORKERS, len(urls)))) as ex:
        return {u: t for u, t in zip(urls, ex.map(_read_page, urls)) if t}


def search_duckduckgo_and_fetch(ctx, query: str, max_results: Optional[int] = None) -> str:
    """Fetch web results and format them: deduped by domain, per-snippet capped.

    Returns formatted result text, or the NO_RESULTS / SEARCH_FAILED sentinel.
    """
    if _is_news(query or ""):
        results = raw_search_results(ctx, query, max_results, timelimit="w")
    else:
        results = raw_search_results(ctx, query, max_results)
    if not results:
        return NO_RESULTS

    # Prefer DEEP links over site roots/section pages. The per-domain cap below keeps
    # sources diverse, but it used to be filled by whichever result came first — for a
    # news query that is often the outlet's homepage, so the actual articles from that
    # outlet were dropped and the user got "ria.ru" instead of the story. Sorting
    # landing pages last (stably, so relevance order is otherwise preserved) means a
    # domain's slot goes to a real article whenever the domain has one.
    results = sorted(results, key=lambda r: _is_landing_url(r.get("href") or ""))

    kept, per_domain = [], {}
    for r in results:
        href = (r.get("href") or "").strip()
        if not href:
            continue
        dom = _domain(href)
        if per_domain.get(dom, 0) >= SEARCH_MAX_PER_DOMAIN:
            continue
        per_domain[dom] = per_domain.get(dom, 0) + 1
        kept.append((r, href, dom))

    # Some sites refuse a bot (403, JS walls): candidates beyond the top are read
    # in the same parallel wave, and the first SEARCH_READ_PAGES that open win, in
    # rank order, so a refusal costs a source from further down, not a source.
    read = _read_pages([h for _, h, _ in kept[:SEARCH_READ_PAGES * 2]])
    pages = {}
    for _, h, _ in kept:
        if h in read and len(pages) < SEARCH_READ_PAGES:
            pages[h] = best_passages(read[h], query, SEARCH_PAGE_CHARS)
    if kept:
        logger.info("web search: read %d pages of %d tried", len(pages),
                    min(len(kept), SEARCH_READ_PAGES * 2))
    parts = []
    for r, href, dom in kept:
        title = (r.get("title") or "No title").strip()
        body = " ".join((r.get("body") or "").split())  # collapse whitespace/newlines
        cap = SEARCH_SNIPPET_CHARS
        if len(pages.get(href, "")) > len(body):
            body, cap = pages[href], SEARCH_PAGE_CHARS
        if len(body) > cap:
            body = body[:cap].rstrip() + "…"
        parts.append(f"[{len(parts) + 1}] {title} ({dom})\n{body}\n{href}")
        # The Telegram reply lists what was read (tg_reply_shape): a per-turn
        # register on the scoped context, absent on the bare/legacy contexts.
        _reg = getattr(ctx, "turn_sources", None)
        if isinstance(_reg, list) and not any(x.get("url") == href for x in _reg):
            _reg.append({"n": len(_reg) + 1, "title": title, "domain": dom, "url": href, "query": query})

    if not parts:
        return NO_RESULTS
    # A counted request needs whole items, and a 500-char snippet cut mid-joke
    # cannot supply one; give the same request more room overall as well.
    total_cap = SEARCH_TOTAL_CHARS + SEARCH_PAGE_CHARS * len(pages)
    if wanted_count(query):
        total_cap = max(SEARCH_TOTAL_CHARS, 12000)
    return truncate_text("\n\n".join(parts), total_cap)


def wanted_count(query: str) -> int:
    """How many items the user ordered («5 анекдотов», "list ten models"), 0 if
    they did not order a number. Read by the model: a bare "5" in "iphone 5
    price" is a name, not an order for five things."""
    import intent
    got = intent.ask_choice(
        "A web search query: {text}\n\nDoes it order a NUMBER of items (5 jokes, "
        "три рецепта, ten ideas)? Answer that number, or 0 when no count of items "
        "is ordered (a number inside a name or a price is not an order).",
        query or "", tuple(str(k) for k in range(0, 21)), "0")
    n = int(got)
    return n if 2 <= n <= 20 else 0


def distill_search_results(ctx, query: str, raw_results: str) -> str:
    """Extract grounded, source-cited facts from raw results via SEARCHER_PROMPT.

    Uses a "<think></think>" prefill so reasoning-prone fine-tunes emit the answer
    directly instead of looping inside an unclosed think block.
    """
    user_msg = f"User query: {query}\n\nSearch results:\n{raw_results}"
    # A request for N THINGS cannot be delivered inside a 400-token budget meant
    # for "1-4 sentences of distilled facts": five jokes came back as "there is
    # one about an aeroplane and one about a fence". The prompt now allows a full
    # list, so the budget has to allow it too, or the fix stops at the token cap.
    want = wanted_count(query)
    budget = SEARCH_DISTILL_MAX_TOKENS
    if want:
        budget = min(4000, max(SEARCH_DISTILL_MAX_TOKENS, 220 * want + 300))
        logger.info("search distill: request asks for %d items — budget %d tokens",
                    want, budget)
    out = call_llm_simple(
        ctx, SEARCHER_PROMPT, user_msg,
        temperature=0.2, max_tokens=budget, prefill="<think></think>",
    )
    return (out or "").strip()


# Tuned on 6 live cases: a two-part query with one part answered, an
# "Insufficient information" round, a vague fact, and three complete answers.
_FOLLOWUP_PROMPT = (
    "You check web search findings against the user's query, part by part. A query often asks "
    "for several things (\"when AND how much\", \"who won AND the score\"): every one of them must "
    "be answered by a concrete fact in the findings. \"Insufficient information\", findings about "
    "a different thing, or a missing part means the search must continue.\n"
    "If EVERY part is answered, reply exactly: NONE\n"
    "Otherwise reply with ONE short web search query for the missing part -- in the language most "
    "likely to find it. Never repeat the user's query word for word: the same words return the "
    "same results -- use other words, or English.\n"
    "Examples:\n"
    "Query: when is the new Zelda out and its price | Findings: Released 12 May 2023 -> new Zelda price\n"
    "Query: погода в Казани | Findings: Казань: +8, дождь -> NONE\n"
    "Reply with the query or NONE, nothing else.")


def followup_query(ctx, query: str, found: str) -> str:
    """The one extra search the model asks for after reading the first round, or ''."""
    try:
        out = call_llm_simple(ctx, _FOLLOWUP_PROMPT,
                              f"Query: {query} | Findings: {found[:6000]}",
                              temperature=0.1, max_tokens=60, prefill="<think></think>")
    except Exception:
        logger.exception("search follow-up check failed")
        return ""
    q = ((out or "").strip().splitlines() or [""])[0].strip().strip("\"'«»`")
    # A year the model adds from memory ("последний чемпионат" -> 2022) sends the
    # round to the past; only the user's own years stay (same rule as tools.search).
    q = re.sub(r"\s*\b(?:19|20)\d\d\b", lambda m: m.group(0) if m.group(0).strip() in (query or "") else "",
               q).strip()
    if (not q or q.upper().startswith("NONE") or len(q) > 200
            or q.lower() == (query or "").strip().lower()):
        return ""
    return q


def run_web_search(ctx, query: str) -> str:
    """Full search pipeline: fetch -> (optionally) distill into grounded facts.

    Distillation is skipped, and raw results returned, when SEARCH_DISTILL is off
    or the distilled output is empty/insufficient — so the caller always gets
    something usable.
    """
    # Five jokes cannot be extracted from five 500-character snippets. When the
    # request names a count, widen the FETCH too — the prompt and the token budget
    # are useless if the raw material was never collected.
    if outbound_blocked(query):
        return SEARCH_BLOCKED
    want = wanted_count(query)
    _qs = getattr(ctx, "turn_queries", None)
    if isinstance(_qs, list) and query not in _qs:
        _qs.append(query)
    raw = search_duckduckgo_and_fetch(
        ctx, query, max_results=max(SEARCH_MAX_RESULTS, want * 2) if want else None)
    if raw in (NO_RESULTS, SEARCH_FAILED):
        return raw
    facts = distill_search_results(ctx, query, raw) if SEARCH_DISTILL else ""
    usable = facts and "insufficient information" not in facts.lower()
    # One more round, at the model's discretion: it reads what was found and
    # either says it is enough or names the ONE query that fills the gap.
    if SEARCH_FOLLOWUP_ROUND:
        more_q = followup_query(ctx, query, facts if usable else raw)
        if more_q and not outbound_blocked(more_q):
            more = search_duckduckgo_and_fetch(ctx, more_q)
            logger.info("search follow-up round: %r -> %s", more_q,
                        "nothing" if more in (NO_RESULTS, SEARCH_FAILED) else f"{len(more)} chars")
            if more not in (NO_RESULTS, SEARCH_FAILED):
                if isinstance(_qs, list) and more_q not in _qs:
                    _qs.append(more_q)
                raw = raw + "\n\n" + more
                if SEARCH_DISTILL:
                    facts = distill_search_results(ctx, query, raw)
                    usable = facts and "insufficient information" not in facts.lower()
    return facts if usable else raw  # nothing usable distilled: the main model reads raw


# Stock-photo agencies serve WATERMARKED previews. One of these in a slide deck
# ruins it outright — a real deck came back with "alamy" stamped diagonally across
# the picture and a black agency strip along the bottom edge, which is also what
# produced the dark band under the image panel. There is no way to un-watermark
# them, so they must never be chosen in the first place.
_WATERMARK_HOSTS = (
    "alamy", "shutterstock", "gettyimages", "istockphoto", "dreamstime",
    "123rf", "depositphotos", "bigstockphoto", "canstockphoto", "agefotostock",
    "stockphoto", "adobestock", "stock.adobe", "vectorstock", "zumapress",
    "profimedia", "imago-images", "picture-alliance", "lori.ru", "photobank",
    "watermark",
    # Aggregators that stamp their OWN name across other people's photographs.
    # Observed, not guessed: toptexnik.ru put "TOPTEXNIK.RU" in letters the
    # height of the cab across a K-700 on a slide.
    "toptexnik",
)

# Marketplaces and print shops. Their pictures are PRODUCT LISTINGS: a poster
# photographed on a desk, a mug, a mock-up frame — usually with the seller's own
# caption burned into the image. A slide about the Kirovets tractor came back
# with an Etsy-style "OLIVER FARM TRACTOR patent print" complete with a
# "DIGITAL BLUEPRINTS" strip along the bottom. Never illustrate with these.
_MARKETPLACE_HOSTS = (
    "etsy.com", "etsystatic", "ebay.", "ebayimg", "amazon.", "ssl-images-amazon",
    "media-amazon", "aliexpress", "alicdn", "redbubble", "zazzle", "society6",
    "displate", "posterlounge", "allposters", "fineartamerica", "pixels.com",
    "wildberries", "ozon.ru", "avito", "taobao", "walmart", "temu.",
    "pinterest", "pinimg", "poster", "printify", "teepublic",
)


def is_watermarked_source(url: str) -> bool:
    """True if `url` is a stock agency preview or a marketplace product listing —
    the two families that make an auto-built deck look cheap."""
    u = (url or "").lower()
    return (any(h in u for h in _WATERMARK_HOSTS)
            or any(h in u for h in _MARKETPLACE_HOSTS))


# Hosts whose pictures are genuine documentary photographs, freely usable, and
# almost never overlaid with seller text. Ranked up so a good source wins over a
# merely larger one.
_PREFERRED_HOSTS = (
    "wikimedia.org", "wikipedia.org", "wikimapia", "commons.",
    "unsplash.com", "pexels.com", "pixabay.com", "flickr.com", "staticflickr",
    "nasa.gov", ".gov", ".edu", "museum",
)


def is_preferred_source(url: str) -> bool:
    u = (url or "").lower()
    return any(h in u for h in _PREFERRED_HOSTS)


def image_search_urls(ctx, query: str, *, max_results: int = 12,
                      prefer: str = "portrait") -> list:
    """Return a list of candidate image URLs (largest first) for ``query``.

    Uses the ddgs *images* vertical. ``prefer`` biases the ranking: "portrait"
    for a face reference (the original caller), "landscape" for slide artwork —
    a tall photo cropped into a wide panel loses most of its subject.
    Returns [] on failure."""
    from ddgs import DDGS
    if outbound_blocked(query):
        return []

    def _do_query():
        with DDGS() as ddgs:
            return list(ddgs.images(
                query, region=SEARCH_REGION, safesearch=SEARCH_SAFESEARCH,
                max_results=max_results))

    rows = _ddgs_retry(ctx, _do_query, "Image search error")

    # Ranking by PIXEL AREA threw the search engine's relevance away. Measured:
    # for "Кировец soviet agricultural tractor field" the engine's own top hit is
    # a photograph of the tractor at 800x533, and the deck came back with a
    # 4096x2160 flag of Kenya on its cover — 8.8 megapixels against 0.4 beats any
    # bonus this function can apply. Relevance is now the spine and size is a
    # floor, not a score.
    MIN_AREA = 160_000        # ~400x400: below this a slide picture is mush

    def _score(idx, r):
        w = int(r.get("width") or 0); h = int(r.get("height") or 0)
        if prefer == "landscape":
            bonus = 1.35 if w > h else 0.8
        else:
            bonus = 1.25 if h >= w else 1.0        # a portrait is a better face ref
        # A documentary photograph from Wikimedia beats a worse picture from a
        # random aggregator, so it may climb a few places — but not past a much
        # more relevant hit.
        if is_preferred_source(r.get("image") or "") or is_preferred_source(r.get("url") or ""):
            bonus *= 1.6
        # Too small to use is a DEMOTION, not a removal: dropping them emptied
        # the list on a result set of nothing but thumbnails, and a caller that
        # asked for images and got none crashes on urls[0]. A candidate that
        # reports no dimensions is not demoted -- unknown is not small.
        area = w * h
        if area and area < MIN_AREA:
            bonus *= 0.05
        # 0.9^rank: the engine's order is the spine, and the decay is gentle
        # enough that a bonus can overtake a NEIGHBOUR (a wide photo beats a tall
        # one for slide artwork; Wikimedia beats an aggregator) but far too
        # gentle to lift something from the bottom of the list. A 1/(rank+1)
        # decay was tried first and was too steep for any bonus to matter.
        return bonus * (0.9 ** idx)

    rows = sorted(rows, key=lambda ir: _score(rows.index(ir), ir), reverse=True)
    urls, skipped = [], 0
    for r in rows:
        u = (r.get("image") or r.get("url") or "").strip()
        if not u.lower().startswith("http"):
            continue
        # Check the PAGE as well as the image host: agencies often serve the
        # preview from a CDN whose name gives nothing away.
        if is_watermarked_source(u) or is_watermarked_source(r.get("url") or ""):
            skipped += 1
            continue
        urls.append(u)
    if skipped:
        logger.info("image search %r: skipped %d watermarked stock result(s)",
                    query[:60], skipped)
    return urls


CUTOUT_BORDER_FRAC = 0.55        # share of the border that must be paper-white
# 252, not 238. At 238 the test caught the packshot but ALSO refused a snowfield,
# and that was written up as a known cost. It was not: measured on the same
# images plus two synthetic snow scenes, a catalogue background is paper white
# (0.91 of the border at >=252) while bright snow is merely light -- 0.03 with
# grain, 0.00 under flat light, because photographic white is never 255. The
# stricter floor keeps the packshot and gives the snow back.
_CUTOUT_WHITE = 252


def looks_like_cutout(path: str) -> bool:
    """True when the picture is an object photographed on a white background.

    A shop packshot is not a scene. Measured on the live image search: the query
    "Витамин D капсулы солнце" returned, as its best hit, an 800x800 photograph
    of one brand's carton on white -- which as a deck's full-bleed cover is an
    advertisement, and its white text vanishes into the background.

    The signal is the BORDER, not the middle: a catalogue shot isolates its
    object and leaves the edges PAPER white. Measured on three real hits from
    three deck topics and on two synthetic snow scenes, as the share of the
    border at >=252 in every channel:

        packshot on white   0.91        snow, with grain    0.03
        tractor in a field  0.00        snow, flat light    0.00
        city street         0.00

    A photograph is never 255: even an overexposed snowfield sits below it.
    That gap is what the threshold sits in.

    Never raises: an unreadable file is simply not a cutout, because refusing a
    picture we could not measure would lose good ones.

    """
    try:
        import numpy as _np
        from PIL import Image as _Image
        a = _np.asarray(_Image.open(path).convert("RGB").resize((256, 256)))
    except Exception:
        return False
    b = max(2, int(256 * 0.08))
    edge = _np.concatenate([a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3),
                            a[:, :b].reshape(-1, 3), a[:, -b:].reshape(-1, 3)])
    return float((edge.min(axis=1) >= _CUTOUT_WHITE).mean()) >= CUTOUT_BORDER_FRAC


def download_image(ctx, url: str, dest_path: str, *, timeout: int = 20,
                   min_bytes: int = 4096) -> bool:
    """Download a single image URL to ``dest_path``. Returns True iff it saved a
    file that PIL can open as a real, non-trivial image (filters HTML error pages,
    1px trackers, and truncated/corrupt downloads). Never raises."""
    import requests
    try:
        if ctx is not None:
            throttle_external_calls(ctx)
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}
        # Every redirect hop checked: a result URL bouncing to 127.0.0.1 would
        # otherwise hand back a local render (ComfyUI /view) as "the photo".
        from dr_urls import safe_get
        resp = safe_get(url, headers=headers, timeout=timeout, stream=True)
        if resp is None or resp.status_code != 200:
            return False
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "image" not in ctype and not url.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            return False
        data = resp.content
        if not data or len(data) < min_bytes:
            return False
        with open(dest_path, "wb") as f:
            f.write(data)
        from PIL import Image
        with Image.open(dest_path) as im:
            im.verify()                       # detect truncated/corrupt
        with Image.open(dest_path) as im:     # re-open (verify leaves it unusable)
            w, h = im.size
        return w >= 200 and h >= 200
    except Exception as exc:
        logger.info("download_image failed (%s): %s", url, exc)
        return False


def _has_clear_face(path: str) -> bool:
    """True iff a face detector finds a single reasonably-sized face — the gate that
    stops a face-less search hit (a product photo, logo, landscape) from being used
    as a 'person' reference. Fails OPEN (returns True) only if the detector itself is
    unavailable, so the feature still works where face tooling isn't installed."""
    try:
        import identity_metrics as _idm
    except Exception:
        return True  # detector unavailable — don't block the feature
    try:
        fb = _idm.face_box(path, pad=0.0)
        if not fb:
            return False
        from PIL import Image
        with Image.open(path) as im:
            W, H = im.size
        fw, fh = (fb[2] - fb[0]), (fb[3] - fb[1])
        # Require the face to be a non-trivial part of the frame (filters tiny
        # incidental faces in a busy product shot).
        return fw * fh >= 0.02 * W * H
    except Exception:
        return False


def _is_grayscale(path: str, *, sat_thresh: float = 10.0) -> bool:
    """True if the image is effectively black-and-white (very low mean saturation).
    A grayscale portrait makes a recolor edit look wrong (a blue dress on a B&W photo),
    so the reference fetch prefers a colour photo when one is available."""
    try:
        from PIL import Image
        import numpy as np
        with Image.open(path) as im:
            hsv = im.convert("RGB").convert("HSV")
            s = np.asarray(hsv)[:, :, 1].astype("float32")
        return float(s.mean()) < sat_thresh
    except Exception:
        return False


def fetch_reference_photo(ctx, query: str, dest_path: str, *, tries: int = 10,
                          require_face: bool = True, prefer_color: bool = True,
                          prefer: str = "portrait") -> Optional[str]:
    """Image-search ``query`` and download the first candidate that is a VALID image
    AND (when ``require_face``) actually contains a clear face — so a face-less search
    hit (e.g. a smartwatch that ranked high) is never used as a person reference.
    When ``prefer_color`` (default), a COLOUR photo is preferred over a black-and-white
    one (a recolor edit on a grayscale photo looks wrong); grayscale is accepted only
    if no colour candidate is found. Saves the winner to ``dest_path``; returns the
    path or None.
    """
    urls = image_search_urls(ctx, query, prefer=prefer)
    if not urls:
        logger.warning("fetch_reference_photo: no image results for %r", query)
        return None
    checked = 0
    gray_fallback = None                      # first valid-but-grayscale candidate
    import shutil
    for u in urls[:tries]:
        if not download_image(ctx, u, dest_path):
            continue
        checked += 1
        if require_face and not _has_clear_face(dest_path):
            logger.info("fetch_reference_photo: skipping face-less candidate %r", u[:80])
            continue
        if prefer_color and _is_grayscale(dest_path):
            if gray_fallback is None:         # stash the first usable B&W, keep looking for colour
                gray_fallback = dest_path + ".gray.jpg"
                try:
                    shutil.copyfile(dest_path, gray_fallback)
                except Exception:
                    gray_fallback = None
            logger.info("fetch_reference_photo: grayscale candidate held as fallback %r", u[:80])
            continue
        logger.info("fetch_reference_photo: accepted colour face photo %r -> %s", u[:80], dest_path)
        return dest_path
    if gray_fallback:                          # no colour photo found — use the B&W one
        try:
            shutil.copyfile(gray_fallback, dest_path)
            os.remove(gray_fallback)
        except Exception:
            pass
        logger.info("fetch_reference_photo: no colour photo; using grayscale fallback for %r", query)
        return dest_path
    logger.warning("fetch_reference_photo: no candidate with a clear face in %d checked for %r",
                   checked, query)
    return None

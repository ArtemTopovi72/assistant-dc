"""Stages 2b/3 — fetch the pages, and pick which ones get briefed.

`_fetch_page` is the polite, bounded single-page getter (safe-URL check, size
caps, PDF and ar5iv/HTML-math paths); `_acquire_page` layers the extraction
cache, the structured source adapters and the social-post readers on top of it;
`crawl_pages` runs the whole bounded breadth-first crawl with its per-domain
caps and its link budget; `rerank_for_briefing` then orders what came back so
the expensive brief model is spent on the most relevant pages first.

Everything here is fault-tolerant by design: one bad source can never abort a
run, and the run stays cooperatively cancellable throughout.
"""
import logging
import turn_trace
import random
import re
import time
from collections import deque
from concurrent.futures import as_completed
from pathlib import Path
from typing import Optional

import requests

import dr_settings as S
import formula_audit as _formula_audit
from dr_extract import (GATE_NAV, GATE_OK, classify_extraction, _ar5iv_url,
                        _extract_links, _extract_text, _extract_title,
                        _inline_mathml_as_tex, _looks_like_unmarked_math,
                        _social_outlinks)
from dr_progress import _Progress
from dr_state import _save_state
from dr_urls import (_authority_score, _domain, _domain_allowed, _is_landing_url,
                     _is_safe_public_url, _norm_url, safe_get)
from social import is_social_url, social_kind, fetch_social_posts
from source_adapters import fetch_via_adapter
from utils import throttle_external_calls

logger = logging.getLogger("assistant.research")


# Briefs are LOSSILY compressed (batch consolidation) before single-pass synthesis
# ONLY when they won't fit the model's ACTUAL context window — computed at runtime in
# `_consolidate_evidence` from `_model_context(ctx)` (reserve ~1/3 for think+output),
# optionally lowered by a DR_MAX_REDUCE_CHARS env value. With a large context the raw
# briefs are fed verbatim (no lossy pass that could drop a fact or mangle an equation).
_MAX_HTML_CHARS = 2_000_000


# --------------------------------------------------------------------------- #
# Stage 3 — crawling
# --------------------------------------------------------------------------- #
def _fetch(url: str) -> Optional[str]:
    """Fetch one URL, returning HTML text or None. Never raises. Streams with a
    hard byte cap so a huge or hostile page can't exhaust memory."""
    resp = None
    try:
        resp = safe_get(
            url, getter=requests.get, timeout=S.DR_PAGE_TIMEOUT, stream=True,
            headers={"User-Agent": S.DR_USER_AGENT, "Accept-Language": "en,ru;q=0.8"},
        )
        if resp is None:
            return None
        if resp.status_code != 200:
            logger.debug("fetch %s -> HTTP %s", url, resp.status_code)
            return None
        ctype = (resp.headers.get("content-type") or "").lower()
        if "html" not in ctype and "text" not in ctype and ctype:
            return None  # skip PDFs, images, binaries
        chunks, total = [], 0
        for chunk in resp.iter_content(16384):
            if not chunk:
                continue
            chunks.append(chunk)
            total += len(chunk)
            if total >= _MAX_HTML_CHARS:
                break
        raw = b"".join(chunks)
        # requests reports ISO-8859-1 when the server omits charset (RFC default);
        # for real pages that's almost always wrong (Cyrillic turns to mojibake) —
        # prefer UTF-8 in that case.
        enc = resp.encoding
        if not enc or enc.lower() in ("iso-8859-1", "latin-1"):
            enc = "utf-8"
        return raw.decode(enc, errors="replace")[:_MAX_HTML_CHARS]
    except Exception as exc:
        logger.debug("fetch failed %s: %s", url, exc)
        return None
    finally:
        if resp is not None:
            resp.close()


def _fetch_ar5iv(url: str) -> Optional[str]:
    """Fetch the ar5iv HTML rendering of an arXiv paper with formulas inlined
    as LaTeX. Returns None (caller falls back to PDF) on any failure."""
    ar5iv = _ar5iv_url(url)
    if not ar5iv:
        return None
    resp = None
    try:
        resp = requests.get(
            ar5iv, timeout=S.DR_PAGE_TIMEOUT, allow_redirects=True,
            headers={"User-Agent": S.DR_USER_AGENT})
        if resp.status_code != 200:
            return None
        raw_html = resp.text
        if "<math" not in raw_html:  # not a real ar5iv render (e.g. error page)
            return None
        return _inline_mathml_as_tex(raw_html)
    except Exception as exc:
        logger.debug("ar5iv fetch failed %s: %s", url, exc)
        return None
    finally:
        if resp is not None:
            resp.close()


def _extract_pdf_text(raw: bytes) -> Optional[str]:
    """Extract text from a PDF byte string via pypdf (Phase 9). Returns None on
    any failure or for image-only/scanned PDFs (no extractable text layer)."""
    try:
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw))
        parts = []
        for page in reader.pages[:S.DR_PDF_MAX_PAGES]:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                continue
        text = re.sub(r"\n{3,}", "\n\n", "\n".join(parts)).strip()
        if text and _looks_like_unmarked_math(text):
            text = (
                "[NOTE: this PDF's text layer has no LaTeX/math markup — equations below "
                "may have lost fraction bars, exponents, or operators during extraction; "
                "treat any formula as approximate and flag uncertain math rather than "
                "reproducing it as exact LaTeX.]\n\n" + text)
        return text or None
    except Exception as exc:
        logger.debug("pdf extract failed: %s", exc)
        return None


def _scrub(text: str, url: str) -> str:
    """Strangers' pages: cut chunks aimed at the AI (agent/injection_scan.py; fails open)."""
    try:
        import injection_scan
        return injection_scan.scrub(text, url)
    except Exception:
        return text


def _fetch_page(url: str) -> tuple:
    """Fetch one URL. Returns (html, pdf_text): exactly one is non-None on
    success, both None on failure. PDFs (scholarly papers, arXiv) are downloaded
    and text-extracted instead of being skipped as 'not HTML' (Phase 9)."""
    if not _is_safe_public_url(url):
        logger.debug("blocked non-public/unsafe URL: %s", url)
        return None, None
    if "arxiv.org" in url.lower():
        ar5iv_html = _fetch_ar5iv(url)
        if ar5iv_html is not None:
            return ar5iv_html, None
    resp = None
    try:
        # Every redirect hop re-checked (dr_urls.safe_get), not just the first URL.
        resp = safe_get(getter=requests.get, url=url, timeout=S.DR_PAGE_TIMEOUT, stream=True,
            headers={"User-Agent": S.DR_USER_AGENT, "Accept-Language": "en,ru;q=0.8"},
        )
        if resp is None:
            return None, None
        if resp.status_code != 200:
            logger.debug("fetch %s -> HTTP %s", url, resp.status_code)
            return None, None
        ctype = (resp.headers.get("content-type") or "").lower()
        is_pdf = "application/pdf" in ctype or url.lower().split("?")[0].endswith(".pdf")
        if not is_pdf and "html" not in ctype and "text" not in ctype and ctype:
            return None, None  # skip images/binaries we can't read
        cap = S.DR_PDF_MAX_BYTES if is_pdf else _MAX_HTML_CHARS
        chunks, total = [], 0
        for chunk in resp.iter_content(16384):
            if not chunk:
                continue
            chunks.append(chunk)
            total += len(chunk)
            if total >= cap:
                break
        raw = b"".join(chunks)
        # Content sniff: some servers send PDFs as octet-stream / wrong type.
        if not is_pdf and raw[:5] == b"%PDF-":
            is_pdf = True
        if is_pdf:
            if not S.DR_PDF_ENABLED:
                return None, None
            return None, _extract_pdf_text(raw)
        enc = resp.encoding
        if not enc or enc.lower() in ("iso-8859-1", "latin-1"):
            enc = "utf-8"
        return raw.decode(enc, errors="replace")[:_MAX_HTML_CHARS], None
    except Exception as exc:
        logger.debug("fetch failed %s: %s", url, exc)
        return None, None
    finally:
        if resp is not None:
            resp.close()


def _acquire_page(url, depth, seed, category, use_adapters, cache) -> dict:
    """Fetch+extract+gate ONE url. No shared mutable state (besides the file-per-URL
    cache, whose writes are atomic per distinct URL), so it is safe to run on a
    worker thread. Returns {url, depth, page, html, reason, social}: `page` is set
    only on usable gated content; `reason` carries a quarantine code on rejection;
    `html` is set only on a live HTML fetch (enables the link/social crawl)."""
    res = {"url": url, "depth": depth, "page": None, "html": None,
           "reason": None, "social": False}
    # Social walls (VK/Telegram) need API/server-rendered extraction, not scraping.
    if is_social_url(url):
        res["social"] = True
        posts = fetch_social_posts(url)
        if posts:
            res["page"] = {"url": url, "domain": _domain(url),
                           "title": seed.get("title") or f"{social_kind(url)} posts",
                           "text": posts[:S.DR_PAGE_CHARS]}
        return res

    page, html = None, None
    # 1. Structured API adapter (Wikipedia/arXiv/Crossref): clean canonical text.
    if use_adapters:
        ad = fetch_via_adapter(url, category)
        if ad and ad.get("text"):
            status, _ = classify_extraction(
                ad["text"], None, ad.get("title") or seed.get("title", ""))
            if status == GATE_OK:
                page = {"url": url, "domain": ad.get("domain") or _domain(url),
                        "title": ad.get("title") or seed.get("title", ""),
                        "text": ad["text"][:S.DR_PAGE_CHARS]}
                logger.debug("adapter %s served %s", ad.get("via"), url)
    # 2. Extraction cache (skips fetch + trafilatura on a fresh hit).
    if page is None and cache is not None:
        cached = cache.get(url, category)
        if cached:
            status, _ = classify_extraction(
                cached.get("text"), None, cached.get("title") or seed.get("title", ""))
            if status == GATE_OK:
                page = {"url": url, "domain": _domain(url),
                        "title": cached.get("title") or seed.get("title", ""),
                        "text": cached["text"][:S.DR_PAGE_CHARS],
                        "equations": _formula_audit.extract_equations(cached["text"])}
    # 3. Live fetch + extract (HTML or PDF) + content-quality gate.
    if page is None:
        if cache is not None and cache.failed_recently(url):
            res["reason"] = "fetch_failed"
            return res
        html, pdf_text = _fetch_page(url)
        if html is None and pdf_text is None:
            res["reason"] = "fetch_failed"
            if cache is not None:
                cache.mark_failed(url)
            return res
        if pdf_text is not None:
            text = pdf_text  # html stays None -> no link/social crawl off a PDF
            budget = S.DR_PDF_TEXT_CHARS
        else:
            text = _extract_text(html, url)
            budget = S.DR_PAGE_CHARS
        title = seed.get("title") or _extract_title(html)
        status, reason = classify_extraction(text, html, title)
        if status != GATE_OK:
            res["reason"] = reason
            logger.debug("gate %s -> %s", reason, url)
            if cache is not None:
                cache.mark_failed(url)
            return res
        page = {"url": url, "domain": _domain(url), "title": title,
                "text": _scrub(text[:budget], url),
                "equations": _formula_audit.extract_equations(text)}
        if cache is not None:
            cache.put(url, title, text)
    # Reject site-root / section-landing pages (no document content).
    if _is_landing_url(page["url"]):
        res["reason"] = GATE_NAV
        logger.debug("gate %s (landing url) -> %s", GATE_NAV, page["url"])
        return res
    res["page"], res["html"] = page, html
    return res


def crawl_pages(ctx, sources: list, caps: dict, prog: _Progress, run_dir: Path,
                profile: Optional[dict] = None, cache=None) -> list:
    """Breadth-first crawl of the seed sources (optionally following in-domain
    links up to caps['depth']). Returns a list of page dicts; bad pages are
    quarantined with a reason code (Phase 1), not silently dropped. A structured
    API adapter (Phase 7) and the extraction cache (Phase 6) short-circuit the
    network fetch where they apply. Fetches run in bounded waves (one URL per
    domain per wave for politeness) when DR_FETCH_CONCURRENCY > 1; bookkeeping
    (visited/pages/queue/quarantine) stays single-threaded."""
    max_pages = caps["max_pages"]
    max_depth = caps["depth"]
    links_per_page = caps["links_per_page"]
    category = (profile or {}).get("category", "general")
    use_adapters = S.DR_ENABLE_ADAPTERS

    queue = deque((s["href"], 0) for s in sources)
    meta = {s["href"]: s for s in sources}
    visited, pages = set(), []
    quarantine: dict = {}  # reason_code -> count (visible in stats/logs)

    conc = max(1, S.DR_FETCH_CONCURRENCY)
    while queue and len(pages) < max_pages:
        if ctx is not None and ctx.is_cancelled():
            break
        # Build a wave of up to `conc` unvisited URLs, at most ONE per domain so a
        # single host is never fetched in parallel (per-domain politeness); extra
        # same-domain URLs are deferred to a later wave, BFS order preserved.
        wave, wave_doms, deferred = [], set(), []
        while queue and len(wave) < conc:
            url, depth = queue.popleft()
            k = _norm_url(url)
            if k in visited:
                continue
            dom = _domain(url)
            if dom in wave_doms:
                deferred.append((url, depth))
                continue
            visited.add(k)
            wave.append((url, depth))
            wave_doms.add(dom)
        if deferred:
            queue.extendleft(reversed(deferred))
        if not wave:
            continue

        if ctx is not None:
            throttle_external_calls(ctx)

        # Fetch the wave. Concurrent only when asked AND there's >1 URL; the
        # per-URL work (_acquire_page) holds no shared mutable state.
        if conc > 1 and len(wave) > 1:
            with turn_trace.Pool(max_workers=len(wave)) as ex:
                results = list(ex.map(
                    lambda wd: _acquire_page(wd[0], wd[1], meta.get(wd[0], {}),
                                             category, use_adapters, cache), wave))
        else:
            results = [_acquire_page(u, d, meta.get(u, {}), category, use_adapters, cache)
                       for u, d in wave]

        # Bookkeeping is strictly single-threaded here.
        for r in results:
            if len(pages) >= max_pages:
                break
            page = r["page"]
            if page is None:
                if r["reason"]:
                    quarantine[r["reason"]] = quarantine.get(r["reason"], 0) + 1
                continue
            pages.append(page)
            suffix = " (social)" if r["social"] else ""
            # bump, not update: this wave counts its own pages from zero, and
            # a later wave used to overwrite the run's total with its own size.
            prog.bump("Crawling",
                      f"[{len(pages)}/{max_pages}] {page['domain']}{suffix}",
                      pages=1)
            html, durl, ddepth = r["html"], r["url"], r["depth"]
            # Hop out to the org's VK/Telegram + follow in-domain links. Only
            # possible with the page's HTML (not adapter/cache/PDF). Re-apply the
            # domain whitelist/blacklist on every queued hop (a social outlink
            # would otherwise bypass a ban set on the search-hit filter).
            if html is not None:
                for link in _social_outlinks(html, durl)[:S.SOCIAL_OUTLINKS_PER_PAGE]:
                    if (_norm_url(link) not in visited and len(pages) < max_pages
                            and _domain_allowed(link, S.DR_DOMAIN_WHITELIST, S.DR_DOMAIN_BLACKLIST)):
                        queue.append((link, ddepth + 1))
                if ddepth < max_depth and len(pages) < max_pages:
                    for link in _extract_links(html, durl)[:links_per_page]:
                        if (_norm_url(link) not in visited
                                and _domain_allowed(link, S.DR_DOMAIN_WHITELIST, S.DR_DOMAIN_BLACKLIST)):
                            queue.append((link, ddepth + 1))
            if len(pages) % 4 == 0:
                _save_state(run_dir, phase="crawling", pages=[p["url"] for p in pages])

        # Politeness between waves. Add jitter under concurrency to desync from
        # rate limiters; serial behavior (conc==1) is exactly the old fixed delay.
        if S.DR_FETCH_DELAY > 0:
            extra = random.uniform(0, S.DR_FETCH_JITTER) if conc > 1 else 0.0
            time.sleep(S.DR_FETCH_DELAY + extra)

    if quarantine:
        prog.stats["quarantined"] = dict(quarantine)
        logger.info("Crawl quarantined %d pages by reason: %s",
                    sum(quarantine.values()), quarantine)
    if cache is not None:
        prog.stats["cache"] = cache.stats()
        logger.info("Extraction cache: %s", cache.stats())
    logger.info("Crawled %d pages", len(pages))
    return pages


# --------------------------------------------------------------------------- #
# Second-stage reranking (rerank.py) — pick the strongest evidence to brief
# --------------------------------------------------------------------------- #
def rerank_for_briefing(topic: str, pages: list, profile: dict, prog: _Progress) -> list:
    """Reorder deduped pages by semantic relevance to the topic (fused with the
    existing host-authority score) and keep only the top-K for the briefing/map
    step. This is the explicit reranker stage: the LLM never sees the full raw
    retrieval set, only the best candidates. A pure pass-through when disabled or
    when there are already few pages."""
    if not S.DR_RERANK_ENABLED or not pages:
        return pages
    category = profile.get("category", "general")
    top_k = S.DR_RERANK_TOP_K if S.DR_RERANK_TOP_K > 0 else len(pages)
    top_k = min(top_k, len(pages))  # can't keep more than we have ("7 → top 60" was a lie)
    if len(pages) <= top_k and S.DR_RERANK_AUTHORITY_WEIGHT == 0:
        return pages  # nothing to drop and relevance-only order isn't requested
    try:
        from rerank import rerank_pages
    except Exception as exc:
        logger.warning("reranker unavailable (%s); keeping authority order", exc)
        return pages
    prog.update("Deduplicating", f"reranking {len(pages)} pages → top {top_k}")

    def _authority(page: dict) -> float:
        return _authority_score(page.get("url", ""), category)

    # Score against title + a generous head of the page text (the reranker caps
    # internally); keep it cheap and deterministic.
    for p in pages:
        p["_rr_text"] = f"{p.get('title', '')}\n{(p.get('text') or '')[:1200]}"
    # Rerank against the DISTILLED core_topic, not the raw user message. The raw
    # text is a conversational request ("я хочу ссылки на конкретные экскурсии как
    # в яндексе...", typos included) — as a cross-encoder query that is mostly noise,
    # and every page scored ~0.01-0.05, i.e. the ranking was effectively arbitrary.
    # core_topic is the clean entity phrase the query planner already extracted.
    # Same source of truth as the relevance-term filter in run_deep_research.
    rr_query = (profile.get("core_topic") or "").strip() or topic
    ordered = rerank_pages(
        rr_query, pages, top_k=top_k, text_key="_rr_text",
        authority_fn=_authority, authority_weight=S.DR_RERANK_AUTHORITY_WEIGHT,
        backend=S.DR_RERANK_BACKEND)
    for p in ordered:
        p.pop("_rr_text", None)
    backend = "?"
    try:
        from rerank import get_reranker
        backend = get_reranker(S.DR_RERANK_BACKEND).backend
    except Exception:
        pass
    kept = [f"{p['domain']}({p.get('_rerank_relevance', 0):.2f})" for p in ordered[:6]]
    logger.info("RERANK[%s]: %d->%d q=%r kept top: %s",
                backend, len(pages), len(ordered), rr_query[:70], kept)
    prog.update("Deduplicating", f"kept top {len(ordered)} by relevance [{backend}]")
    return ordered

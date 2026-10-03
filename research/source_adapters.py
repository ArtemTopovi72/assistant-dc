"""Structured source adapters for JS-heavy / API-backed domains (Phase 7).

Static HTML extraction fails or degrades on several high-value domains. Where a
domain exposes a *keyless* structured endpoint, we fetch the clean canonical
text from that endpoint instead of scraping rendered HTML. This is strictly
better than headless rendering for these sources: faster, no anti-bot, and the
text is the real article, not a DOM snapshot.

Scope is deliberately narrow — only domains validated by evaluation as both
high-value AND well-served by a keyless API:

    * Wikipedia (any language)  -> MediaWiki action API plain-text extract.
      The single biggest entertainment/reference quality lever (the Кочегар and
      Di Zenzo runs lean heavily on Wikipedia) and 100% reliable via the API.
    * arXiv (arxiv.org)         -> arXiv Atom API (title + abstract).
    * DOI / Crossref (doi.org)  -> Crossref REST works metadata + abstract.

Everything else returns None and the caller falls back to normal fetch+extract.
JS sites with NO keyless API (kinopoisk, okko, dzen, venue SPAs) are left to the
deferred headless-browser phase (P8); we do not scrape them here.

Every adapter is network-guarded and returns None on ANY failure so a flaky API
never aborts a run.
"""
import functools
import logging
import os
import re
from typing import Optional
from urllib.parse import urlparse, unquote

import requests

logger = logging.getLogger("assistant.research.adapters")

_TIMEOUT = 12
# Wikimedia's User-Agent policy wants a way to reach the operator. Since
# 2026-09 the action API and the full-page REST routes answer 429 to a UA
# without one (measured: same request, contact added -> 200). The contact is
# the operator's to choose, so it comes from the environment, never a default.
_WIKI_CONTACT = os.getenv("WIKIMEDIA_CONTACT", "").strip()
_HEADERS = {"User-Agent": "f5-research-bot/1.0 (research assistant"
                          + (f"; {_WIKI_CONTACT}" if _WIKI_CONTACT else "") + ")",
            "Accept": "application/json"}
_warned_no_contact = False
_MAX_TEXT = 8000


def _domain(url: str) -> str:
    try:
        net = urlparse(url).netloc.lower()
        return net[4:] if net.startswith("www.") else net
    except Exception:
        return url


def _adapter_guard(name: str):
    """Every adapter is network-guarded and returns None on ANY failure so a
    flaky API never aborts a run — shared here instead of a try/except
    duplicated in each fetch_* function."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(url):
            try:
                return fn(url)
            except Exception as exc:
                logger.debug("%s adapter failed %s: %s", name, url, exc)
                return None
        return wrapper
    return deco


def adapter_for(url: str, category: str = "general") -> Optional[str]:
    """Return the name of the adapter that applies to this URL, or None.
    Pure routing — no network — so it is cheap and unit-testable."""
    host = _domain(url)
    if host.endswith("wikipedia.org") and "/wiki/" in url:
        return "wikipedia"
    if host == "arxiv.org" and re.search(r"/(abs|pdf)/", url):
        return "arxiv"
    if host == "doi.org":
        return "crossref"
    return None


# --------------------------------------------------------------------------- #
# Wikipedia (MediaWiki action API — keyless, any language)
# --------------------------------------------------------------------------- #
def _wiki_lang_title(url: str):
    p = urlparse(url)
    lang = (p.netloc.split(".")[0] or "en").lower()
    m = re.search(r"/wiki/(.+)$", p.path)
    if not m:
        return None, None
    title = unquote(m.group(1)).replace("_", " ").split("#")[0]
    return lang, title


@_adapter_guard("wikipedia")
def fetch_wikipedia(url: str) -> Optional[dict]:
    lang, title = _wiki_lang_title(url)
    if not title:
        return None
    api = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query", "format": "json", "prop": "extracts",
        "explaintext": "1", "redirects": "1", "titles": title,
    }
    r = requests.get(api, params=params, headers=_HEADERS, timeout=_TIMEOUT)
    if r.status_code != 200:
        # Throttled (429 without a contact) or down: the lead section from the
        # cached summary route is still better than a silent nothing.
        global _warned_no_contact
        if r.status_code == 429 and not _WIKI_CONTACT and not _warned_no_contact:
            _warned_no_contact = True
            logger.warning("Wikipedia API throttled this client (429): set "
                           "WIKIMEDIA_CONTACT in .env (an e-mail or URL) to get "
                           "full articles; falling back to lead summaries")
        return _wiki_summary(lang, title, url)
    pages = r.json().get("query", {}).get("pages", {})
    for _, page in pages.items():
        extract = (page.get("extract") or "").strip()
        if extract:
            return {"title": page.get("title", title),
                    "text": extract[:_MAX_TEXT],
                    "domain": _domain(url), "via": "wikipedia-api"}
    return _wiki_summary(lang, title, url)


def _wiki_summary(lang: str, title: str, url: str) -> Optional[dict]:
    """The article's lead section via the REST summary route (served from
    cache, not throttled like the action API)."""
    try:
        r = requests.get(
            f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/"
            + title.replace(" ", "_"),
            headers=_HEADERS, timeout=_TIMEOUT)
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception:
        return None
    extract = (data.get("extract") or "").strip()
    if not extract:
        return None
    return {"title": data.get("title", title), "text": extract[:_MAX_TEXT],
            "domain": _domain(url), "via": "wikipedia-summary"}


# --------------------------------------------------------------------------- #
# arXiv (Atom API)
# --------------------------------------------------------------------------- #
@_adapter_guard("arxiv")
def fetch_arxiv(url: str) -> Optional[dict]:
    m = re.search(r"/(?:abs|pdf)/([^?#/]+?)(?:v\d+)?(?:\.pdf)?$", url)
    if not m:
        return None
    arxiv_id = m.group(1)
    r = requests.get("https://export.arxiv.org/api/query",
                     params={"id_list": arxiv_id, "max_results": 1},
                     headers={"User-Agent": _HEADERS["User-Agent"]},
                     timeout=_TIMEOUT)
    if r.status_code != 200:
        return None
    xml = r.text
    title = re.search(r"<entry>.*?<title>(.*?)</title>", xml, re.DOTALL)
    summary = re.search(r"<summary>(.*?)</summary>", xml, re.DOTALL)
    if not summary:
        return None
    t = re.sub(r"\s+", " ", title.group(1)).strip() if title else arxiv_id
    body = re.sub(r"\s+", " ", summary.group(1)).strip()
    return {"title": t, "text": f"{t}\n\nAbstract: {body}"[:_MAX_TEXT],
            "domain": "arxiv.org", "via": "arxiv-api"}


# --------------------------------------------------------------------------- #
# Crossref (DOI metadata + abstract)
# --------------------------------------------------------------------------- #
@_adapter_guard("crossref")
def fetch_crossref(url: str) -> Optional[dict]:
    p = urlparse(url)
    doi = p.path.lstrip("/")
    if not doi:
        return None
    r = requests.get(f"https://api.crossref.org/works/{doi}",
                     headers=_HEADERS, timeout=_TIMEOUT)
    if r.status_code != 200:
        return None
    msg = r.json().get("message", {})
    title = "; ".join(msg.get("title", []) or []) or doi
    parts = [title]
    authors = msg.get("author") or []
    if authors:
        names = ", ".join(
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in authors[:8])
        parts.append(f"Authors: {names}")
    venue = "; ".join(msg.get("container-title", []) or [])
    if venue:
        parts.append(f"Published in: {venue}")
    year = (msg.get("published", {}).get("date-parts", [[None]])[0][0])
    if year:
        parts.append(f"Year: {year}")
    abstract = msg.get("abstract")
    if abstract:
        parts.append("Abstract: " + re.sub(r"<[^>]+>", "", abstract).strip())
    text = "\n".join(parts).strip()
    if len(text) < 40:
        return None
    return {"title": title, "text": text[:_MAX_TEXT],
            "domain": "doi.org", "via": "crossref-api"}


_ADAPTERS = {"wikipedia": fetch_wikipedia, "arxiv": fetch_arxiv,
             "crossref": fetch_crossref}


def fetch_via_adapter(url: str, category: str = "general") -> Optional[dict]:
    """Try the structured adapter for this URL. Returns a page dict
    {title, text, domain, via} or None (no adapter, or the API failed —
    caller then falls back to normal fetch+extract)."""
    name = adapter_for(url, category)
    if not name:
        return None
    fn = _ADAPTERS.get(name)
    if not fn:
        return None
    return fn(url)

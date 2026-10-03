"""URL and source classification for the deep-research pipeline.

Pure, side-effect-free predicates and scores over a URL / search result:
canonicalisation (`_norm_url`, `_domain`), junk + landing-page detection,
authority scoring, source-type classification, the SSRF guard
(`_is_safe_public_url`), and the whitelist/blacklist/per-domain-cap filters.

Split out of deep_research.py. Nothing here reads a DR_* module global — every
knob arrives as an argument — so these functions carry no configuration state
and `deep_research.apply_overrides` has nothing to patch on this module.
"""
import ipaddress
import logging
import re
import socket
from typing import Optional
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _domain(url: str) -> str:
    try:
        net = urlparse(url).netloc.lower()
        return net[4:] if net.startswith("www.") else net
    except Exception:
        return url


_LANDING_SEGMENTS = {"en", "ru", "zh", "index", "home", "siteindex", "site-index",
                     "journals", "about", "search", "browse", "login", "signin"}


def _is_landing_url(url: str) -> bool:
    """True if the URL is a site root / section landing page (e.g. nature.com/,
    nature.com/siteindex, scholar.xjtu.edu.cn/en/) rather than an actual document.
    Landing pages have no real content and must never become evidence — let alone
    a PRIMARY source. A real paper/article has a deep, document-like path."""
    try:
        path = urlparse(url).path.strip("/")
    except Exception:
        return False
    if not path:
        return True  # bare host root
    low = "/" + path.lower()
    if any(low.endswith(t) for t in _LANDING_PATH_TAILS):
        return True  # account/login/register endpoint at any depth
    segs = [s for s in path.split("/") if s]
    if len(segs) == 1 and (segs[0].lower() in _LANDING_SEGMENTS or len(segs[0]) <= 3):
        return True
    return False


# Reader/translator/cache wrappers that carry the REAL page in a query
# parameter. Observed in a delivered report's source list:
#   tr-page.yandex.ru/translate?lang=en-ru&url=https://en.wikipedia.org/wiki/Tannin
# cited as if it were the source. It is not: the citation must point at the page
# somebody can check, and a wrapper also hides the real host from every trust
# and dedupe decision made downstream (that Wikipedia article scored as an
# unknown yandex subdomain).
_URL_WRAPPERS = (
    ("tr-page.yandex.ru", "url"),
    ("translate.google.com", "u"),
    ("translate.yandex.ru", "url"),
    ("webcache.googleusercontent.com", "q"),
    ("r.jina.ai", ""),            # path-embedded, handled below
    ("12ft.io", ""),
)


def unwrap_url(url: str) -> str:
    """The real page behind a translator/reader wrapper, or the url unchanged.

    Only unwraps to an absolute http(s) URL, and only one level: a wrapper of a
    wrapper is pathological and not worth chasing.
    """
    try:
        p = urlparse(url or "")
    except Exception:
        return url
    host = (p.netloc or "").lower()
    for marker, param in _URL_WRAPPERS:
        if marker not in host:
            continue
        if param:
            vals = parse_qs(p.query or "").get(param) or []
            cand = (vals[0] if vals else "").strip()
        else:
            # r.jina.ai/https://example.org/x — the target is the path itself.
            cand = (p.path or "").lstrip("/")
        if cand.startswith(("http://", "https://")):
            return cand
    return url


def _norm_url(url: str) -> str:
    """Normalize for dedupe: drop fragment, query, trailing slash; lowercase host."""
    try:
        p = urlparse(url)
        path = p.path.rstrip("/") or "/"
        return f"{p.scheme.lower()}://{p.netloc.lower()}{path}"
    except Exception:
        return url


# Substrings matched against the host. Near-useless for factual research:
# pirate streaming, Q&A / opinion farms, auto-generated glossary/SEO farms,
# scraped study-note and slide dumps, pure link aggregators.
_JUNK_HOST_SUBSTR = (
    # pirate video / streaming (SPECIFIC names only — do NOT use broad substrings
    # like "kino-" here: that would wrongly nuke legit film databases such as
    # kino-teatr.ru. kinopoisk.ru / kinopoisk.ru reviews are legitimate and pass.)
    "lordfilm", "lordserial", "svonok", "kinogo", "hdrezka", "rezka.",
    "filmix", "seasonvar", "zerx.tv", "kinokrad", "baskino", "kinokong",
    # Q&A / opinion / advice farms
    "bolshoyvopros", "otvet.mail.ru", "answers.com", "ask.fm", "quora.com",
    "yahoo.com/answers", "iwillplay",
    # SEO glossary / term / definition farms
    "termbasehub", "definitions.net", "yourdictionary", "examples.com",
    # scraped academic / slide dumps (paywalled or low-trust copies)
    "coursehero", "studocu", "scribd", "slideshare", "slideserve", "slideplayer",
    "studfile", "studyraid", "slidetodoc", "fdocuments",
    # social/aggregator noise
    "pinterest.", "fandom.com",
    # AI-spun / unrelated SEO spam that surfaced on technical queries
    "aimangatranslator", "aimanga",
)


# Always-good hosts regardless of topic: encyclopedic + government/edu primaries.
_BASE_AUTHORITY = (
    "wikipedia.org", "britannica.com", ".gov", ".edu", ".ac.uk", "europa.eu",
)


# Per-category authority hosts. The planner classifies the topic; we then crawl
# the hosts that ACTUALLY hold the best evidence for THAT kind of question first.
# This is what makes the engine a universal fighter instead of a science-only
# tool — a company-trust query prioritizes Glassdoor/Trustpilot, a job query
# prioritizes LinkedIn/careers pages, a health query prioritizes WHO/Mayo, etc.
_CATEGORY_AUTHORITY = {
    "science": (
        "arxiv.org", "doi.org", "ieee.org", "ieeexplore", "acm.org", "dl.acm.org",
        "springer", "sciencedirect", "nature.com", "mdpi.com", "openreview.net",
        "semanticscholar.org", "researchgate.net", "ncbi.nlm.nih.gov", "pubmed",
        "scholar.google", "jmlr.org", "neurips.cc", "proceedings.mlr.press",
        "mathworks.com", "wolfram",
    ),
    "news": (
        "reuters.com", "apnews.com", "bbc.co", "bbc.com", "npr.org",
        "nytimes.com", "washingtonpost.com", "theguardian.com", "wsj.com",
        "bloomberg.com", "ft.com", "economist.com", "axios.com", "politico.com",
        "aljazeera.com", "cnbc.com",
    ),
    "product": (
        "github.com", "github.io", "readthedocs", "rtings.com", "notebookcheck",
        "tomshardware.com", "anandtech.com", "techradar.com", "theverge.com",
        "wirecutter", "nytimes.com/wirecutter", "dpreview.com", "gsmarena.com",
    ),
    "company": (  # reputation / trust / "is X a good company"
        "glassdoor.", "trustpilot.", "comparably.com", "bbb.org", "indeed.com/cmp",
        "linkedin.com/company", "crunchbase.com", "g2.com", "ambitionbox.com",
        "sec.gov", "reuters.com", "bloomberg.com",
    ),
    "jobs": (  # vacancies / hiring / "is this vacancy real"
        "linkedin.com", "indeed.", "glassdoor.", "lever.co", "greenhouse.io",
        "ashbyhq.com", "workday", "wellfound.com", "levels.fyi", "hh.ru",
        "careers.", "jobs.",
    ),
    "finance": (
        "sec.gov", "bloomberg.com", "reuters.com", "ft.com", "wsj.com",
        "morningstar.com", "marketwatch.com", "investor.", "nasdaq.com",
    ),
    "people": (
        "linkedin.com", "imdb.com", "muckrack.com", "forbes.com/profile",
    ),
    "local": (  # venues, cultural centres, posters/afisha — often only on socials
        "tripadvisor.", "yelp.", "openstreetmap.org", "google.com/maps",
        "vk.com", "t.me", "telegram.me", "afisha.ru", "timepad.ru", "kassir.ru",
        "2gis.ru", "yandex.ru/maps",
    ),
    "health": (
        "who.int", "cdc.gov", "nih.gov", "ncbi.nlm.nih.gov", "mayoclinic.org",
        "nhs.uk", "cochrane.org", "medlineplus.gov", "clevelandclinic.org",
    ),
    "legal": (
        "eur-lex.europa.eu", "law.cornell.edu", "justia.com", "courtlistener.com",
        "supremecourt.gov", "findlaw.com",
    ),
    "entertainment": (  # films, series, books, music, games — incl. Russian sources
        "imdb.com", "kinopoisk.ru", "rottentomatoes.com", "metacritic.com",
        "letterboxd.com", "themoviedb.org", "kino-teatr.ru", "film.ru",
        "goodreads.com", "allmusic.com", "discogs.com", "rateyourmusic.com",
        "criterion.com", "afisha.ru", "mubi.com",
    ),
}


def _is_junk_source(url: str) -> bool:
    host = _domain(url)
    return any(s in host for s in _JUNK_HOST_SUBSTR)


# --------------------------------------------------------------------------- #
# Graded authority scoring (Phase 4)
#
# The old _source_rank was binary (0 = authority, 1 = everything else), so a
# personal blog and a major trade publication sorted identically and a fluent
# SEO page could self-promote to SECONDARY via the brief model. We now score
# every host on a graded scale and use that both to RANK (crawl best first) and
# as a TRUST FLOOR/CEILING that the model's self-label cannot override.
# --------------------------------------------------------------------------- #
_AUTH_BASE_SCORE = 0.95      # encyclopaedic / gov / edu — trusted on any topic


_AUTH_CATEGORY_SCORE = 0.85  # the right authority for THIS category


_MID_SCORE = 0.50            # unknown but not obviously bad


_COMMUNITY_SCORE = 0.35      # forum / blog / social — leads, not proof


_JUNK_SCORE = 0.0


# Hosts that are community/UGC/blog platforms: useful as leads but must never be
# rated above COMMUNITY (a fluent post here is still not a primary source). VK /
# Telegram are deliberately EXCLUDED for local/people topics (see _is_community).
_COMMUNITY_HOSTS = (
    "reddit.com", "habr.com", "medium.com", "blogspot.", "wordpress.",
    "livejournal.com", "dzen.ru", "zen.yandex", "pikabu.ru", "vc.ru",
    "stackexchange.com", "stackoverflow.com", "x.com", "twitter.com",
    "facebook.com", "vk.com", "t.me", "telegram.me", "ok.ru", "tumblr.com",
    "substack.com", "tproza.ru", "stihi.ru",
)


def _is_community_host(url: str, category: str = "general") -> bool:
    """Whether a host is community/UGC for trust-capping purposes. Social
    networks are NOT capped for local/people topics — for a venue's poster the
    VK group IS the primary source."""
    host = _domain(url)
    if category in ("local", "people") and any(
            s in host for s in ("vk.com", "t.me", "telegram.me", "ok.ru")):
        return False
    return any(s in host for s in _COMMUNITY_HOSTS)


def _authority_score(url: str, category: str = "general") -> float:
    """Graded host authority in [0, 1]; higher = crawl/trust first.
    Junk is filtered before this, but we still score it 0 defensively."""
    host = _domain(url)
    if _is_junk_source(url):
        return _JUNK_SCORE
    if any(s in host for s in _BASE_AUTHORITY):
        return _AUTH_BASE_SCORE
    if any(s in host for s in _CATEGORY_AUTHORITY.get(category, ())):
        return _AUTH_CATEGORY_SCORE
    if _is_community_host(url, category):
        return _COMMUNITY_SCORE
    return _MID_SCORE


def _source_rank(url: str, category: str = "general") -> int:
    """Back-compat binary rank derived from the graded score (0 = authority for
    THIS category, crawl first; 1 = ordinary). Used by the source sort."""
    return 0 if _authority_score(url, category) >= _AUTH_CATEGORY_SCORE else 1


# --------------------------------------------------------------------------- #
# Manual control — source-type classification + domain filters (real filters
# applied in collect_sources, not cosmetic). Built from the SAME domain lists
# already used for authority scoring above, so a host is never classified one
# way for trust and another way for inclusion.
# --------------------------------------------------------------------------- #
_GOV_MARKERS = (".gov", ".mil", ".gov.uk", ".gov.au")


_FORUM_HOSTS = ("reddit.com", "stackoverflow.com", "stackexchange.com", "quora.com",
                "superuser.com", "serverfault.com", "pikabu.ru", "answers.")


_BLOG_HOSTS = ("medium.com", "blogspot.", "wordpress.", "substack.com", "dzen.ru",
               "zen.yandex", "livejournal.com", "tumblr.com", "habr.com")


_SOCIAL_HOSTS = ("x.com", "twitter.com", "facebook.com", "vk.com", "t.me",
                  "telegram.me", "instagram.com", "tiktok.com", "ok.ru")


_TECH_DOC_HOSTS = ("readthedocs.io", "github.io", "devdocs.io", "docs.", "developer.")


_SOURCE_TYPES = ("academic", "news", "government", "technical", "forum", "blog",
                 "social", "whitepaper")


def _classify_source_type(url: str) -> str:
    """One of _SOURCE_TYPES, or 'general' for anything that doesn't match a
    recognizable pattern (general sources are never excluded by the Sources
    panel's per-type toggles)."""
    host = _domain(url)
    if any(host.endswith(m) for m in _GOV_MARKERS):
        return "government"
    if host.endswith(".edu") or any(s in host for s in _CATEGORY_AUTHORITY["science"]):
        return "academic"
    if any(s in host for s in _CATEGORY_AUTHORITY["news"]):
        return "news"
    if any(s in host for s in _SOCIAL_HOSTS):
        return "social"
    if any(s in host for s in _FORUM_HOSTS):
        return "forum"
    if any(s in host for s in _BLOG_HOSTS):
        return "blog"
    if any(s in host for s in _TECH_DOC_HOSTS) or host == "github.com":
        return "technical"
    if url.lower().split("?")[0].endswith(".pdf"):
        return "whitepaper"
    return "general"


def _domain_allowed(url: str, whitelist: str, blacklist: str) -> bool:
    """Domain whitelist/blacklist (comma-separated substrings of the host).
    Blacklist always wins; an empty whitelist means 'no restriction'."""
    host = _domain(url)
    black = [d.strip().lower() for d in (blacklist or "").split(",") if d.strip()]
    if any(d in host for d in black):
        return False
    white = [d.strip().lower() for d in (whitelist or "").split(",") if d.strip()]
    if white and not any(d in host for d in white):
        return False
    return True


_BLOCKED_HOSTS = {"localhost", "metadata.google.internal",
                  "metadata", "instance-data"}


def _is_safe_public_url(url: str) -> bool:
    """SSRF guard: only fetch http(s) URLs that resolve to a PUBLIC address.
    Rejects non-web schemes (file://, gopher://, ...) and any host that resolves
    to a loopback/private/link-local/reserved IP — so a stray search result or
    redirect can't make the crawler reach localhost, the LAN, or the
    169.254.169.254 cloud-metadata endpoint. Fails closed (any error → unsafe)."""
    try:
        p = urlparse(url)
    except Exception:
        return False
    if p.scheme.lower() not in ("http", "https"):
        return False
    host = (p.hostname or "").lower()
    if (not host or host in _BLOCKED_HOSTS
            or host.endswith(".local") or host.endswith(".internal")):
        return False
    try:
        infos = socket.getaddrinfo(host, p.port or (443 if p.scheme == "https" else 80),
                                   proto=socket.IPPROTO_TCP)
    except Exception:
        return False
    if not infos:
        return False
    for *_unused, sockaddr in infos:
        try:
            addr = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            return False
        if (addr.is_private or addr.is_loopback or addr.is_link_local
                or addr.is_reserved or addr.is_multicast or addr.is_unspecified):
            return False
    return True


def safe_get(url: str, max_redirects: int = 5, getter=None, **kw):
    """requests.get that checks EVERY hop with _is_safe_public_url.

    requests' own allow_redirects=True only let the first URL be checked: a
    public page answering 302 -> http://127.0.0.1:8000/view?... (ComfyUI, other
    users' renders) or :1234 (LM Studio) was fetched and its body handed on.
    Returns the final Response, or None when any hop is unsafe or the chain is
    too long. Network errors propagate like requests.get's."""
    import requests
    from urllib.parse import urljoin
    kw["allow_redirects"] = False
    for _ in range(max_redirects + 1):
        if not _is_safe_public_url(url):
            logger.debug("blocked non-public/unsafe URL: %s", url)
            return None
        resp = (getter or requests.get)(url, **kw)
        loc = (getattr(resp, "headers", None) or {}).get("location")
        if getattr(resp, "is_redirect", False) and loc:
            resp.close()
            url = urljoin(url, loc)
            continue
        return resp
    return None


def _source_type_allowed(url: str, exclude: str) -> bool:
    excluded = {t.strip().lower() for t in (exclude or "").split(",") if t.strip()}
    if not excluded:
        return True
    return _classify_source_type(url) not in excluded


def _mutate_query(q: str, attempt: int) -> Optional[str]:
    """Deterministic query mutation for the retry/resilience knob: a zero-hit
    query gets reformulated instead of just being dropped. Returns None once
    `attempt` exceeds the available mutation strategies."""
    words = q.split()
    if attempt == 0:
        stripped = re.sub(r'["\'“”‘’]', "", q).strip()
        return stripped if stripped != q else (q + " overview")
    if attempt == 1 and len(words) > 2:
        return " ".join(words[:-1])  # drop the most specific trailing term
    if attempt == 2:
        return q + " explained"
    return None


def _cap_per_domain(sources: list, max_per_domain: int) -> list:
    """Keep at most `max_per_domain` sources from any single host, preserving
    the existing (authority-sorted) order. 0 = unlimited."""
    if not max_per_domain or max_per_domain <= 0:
        return sources
    counts, out = {}, []
    for s in sources:
        host = _domain(s["href"])
        counts[host] = counts.get(host, 0) + 1
        if counts[host] <= max_per_domain:
            out.append(s)
    return out


# URL path tails that mark an account/landing endpoint regardless of depth.
_LANDING_PATH_TAILS = ("/register", "/signup", "/sign-up", "/login", "/signin",
                       "/sign-in", "/user/register", "/account", "/subscribe")

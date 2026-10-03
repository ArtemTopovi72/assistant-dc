"""Page-content extraction, quality gating and near-duplicate collapse.

Everything between "we have some HTML/PDF bytes" and "we have usable text":
HTML -> text/title/links (`_extract_text`, `_extract_title`, `_extract_links`),
mojibake repair, MathML -> TeX inlining, the ar5iv URL rewrite, the
unmarked-math heuristic, the social-outlink hop, the extraction quality gate
(`classify_extraction` + its GATE_* verdicts) and shingle-based near-duplicate
clustering (`dedupe_pages`).

Split out of deep_research.py. Every function here is pure: it reads no DR_*
module global and performs no network I/O — the fetchers stay in deep_research
and hand their bytes in. The `logger` is the same "assistant.research" logger the
rest of the pipeline writes to, so log capture is unchanged.
"""
import logging
import re
from typing import Optional
from urllib.parse import urljoin

from social import is_social_url
from dr_urls import _authority_score, _domain, _norm_url

logger = logging.getLogger("assistant.research")


# --------------------------------------------------------------------------- #
# Content-quality gate (Phase 1)
#
# Static extraction frequently returns NOT "nothing" but a misleading shell:
# an anti-bot/CAPTCHA interstitial, a JS skeleton with only nav chrome, or a few
# hundred chars of boilerplate. Those used to flow straight into the brief step
# and become "evidence" (e.g. the hal.science CAPTCHA page in the science run).
# This gate classifies each extraction and quarantines the bad ones with an
# explicit reason code so failures are visible in logs/stats, never silent.
# --------------------------------------------------------------------------- #
GATE_OK = "ok"


GATE_EMPTY = "empty_extraction"


GATE_THIN = "thin_extraction"


GATE_CHALLENGE = "challenge_page"


GATE_SHELL = "shell_page"


GATE_NAV = "nav_landing_page"


# Generic landing/index/legal page titles — these are NOT documents and must
# never become evidence (the Di Zenzo run let "PMC Home" / "PMC Journal List" /
# "Academia.edu | Privacy" through and even tagged them PRIMARY).
_NAV_TITLE_MARKERS = (
    "home page", "| home", "home -", "home |", "privacy", "journal list",
    "sign in", "log in", "create account", "table of contents", "browse",
    "all journals", "categories", "site map", "sitemap", "terms of service",
    "cookie", "subscribe", "newsletter", "404", "not found", "search results",
    "register", "sign up", "signup", "registration", "user login", "my account",
)


# Conservative minimum real-content length. Kept low so legitimately short pages
# (a venue contact block, a short bio) are NOT nuked — the earlier kino-teatr.ru
# lesson: over-aggressive filtering is worse than letting a thin page through.
_GATE_MIN_CHARS = 180


# A challenge/interstitial marker only condemns a page when the page is ALSO
# short — a real article that merely mentions "captcha" must survive.
_GATE_CHALLENGE_MAX_CHARS = 1500


# Shell detection: a large HTML document that extracts to almost no text is a
# JS app that rendered nothing server-side (kinopoisk/okko/dzen/SPA docs).
_GATE_SHELL_HTML_MIN = 60_000


_GATE_SHELL_TEXT_MAX = 220


# Anti-bot / interstitial / placeholder fingerprints (substring match, lowercased).
_CHALLENGE_MARKERS = (
    "captcha", "are you a robot", "are you human", "checking your browser",
    "verifying you are human", "cloudflare", "ddos-guard", "ddos protection",
    "anubis proof-of-work", "proof-of-work", "badbrowser", "access denied",
    "enable javascript", "please enable js", "javascript is required",
    "доступ ограничен", "проверка браузера", "подтвердите, что вы не робот",
    "включите javascript", "request unsuccessful", "incapsula",
)


def classify_extraction(text: Optional[str], html: Optional[str] = None,
                        title: Optional[str] = None) -> tuple:
    """Classify an extraction result. Returns (status, reason_code).
    status == GATE_OK means the text is safe to brief; anything else is
    quarantined. Conservative by design — see thresholds above."""
    if not text or not text.strip():
        return GATE_EMPTY, GATE_EMPTY
    body = text.strip()
    low = body.lower()
    n = len(body)
    # Challenge / interstitial: marker present AND the page is short (real
    # articles that merely discuss captchas are long and survive).
    if n <= _GATE_CHALLENGE_MAX_CHARS and any(m in low for m in _CHALLENGE_MARKERS):
        return GATE_CHALLENGE, GATE_CHALLENGE
    # JS shell: a big HTML doc that yielded almost no extractable text.
    if html is not None and len(html) >= _GATE_SHELL_HTML_MIN and n <= _GATE_SHELL_TEXT_MAX:
        return GATE_SHELL, GATE_SHELL
    # Nav / index / legal landing page: a generic title, OR short text with almost
    # no prose (very few sentence-ending periods = a menu/list, not an article).
    t = (title or "").lower().strip()
    if t and any(m in t for m in _NAV_TITLE_MARKERS):
        return GATE_NAV, GATE_NAV
    periods = body.count(". ") + body.count(".\n") + body.count("? ") + body.count("! ")
    if n < 2500 and periods < max(2, n // 500):
        return GATE_NAV, GATE_NAV
    # Plain too-thin content.
    if n < _GATE_MIN_CHARS:
        return GATE_THIN, GATE_THIN
    return GATE_OK, GATE_OK


_ARXIV_ID_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf|html)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", re.I)


_MATHML_RE = re.compile(r'<math\b([^>]*)>.*?</math>', re.S)


_ALTTEXT_RE = re.compile(r'alttext="((?:[^"\\]|\\.)*)"')


def _ar5iv_url(url: str) -> Optional[str]:
    """arXiv abs/pdf URL -> ar5iv.org HTML rendering, which embeds the ORIGINAL
    LaTeX source for every formula in the math element's alttext attribute
    (MathML+TeX dual rendering). PDF text extraction has no such structure and
    flattens fractions/matrices into glyph soup with no math markup at all; the
    HTML route is strictly more faithful when it's available."""
    m = _ARXIV_ID_RE.search(url)
    return f"https://ar5iv.org/abs/{m.group(1)}" if m else None


def _inline_mathml_as_tex(html: str) -> str:
    """Replace ar5iv <math alttext="...">...</math> nodes with their literal
    $...$/$$...$$ LaTeX source so a plain-text extractor (trafilatura strips
    <math> tags entirely, losing the formula) preserves the real equation."""
    import html as _htmllib

    def _sub(m):
        attrs = m.group(1)
        alt = _ALTTEXT_RE.search(attrs)
        if not alt:
            return ""  # no TeX source recoverable -> drop rather than keep raw MathML
        tex = _htmllib.unescape(alt.group(1))
        return f" $${tex}$$ " if 'display="block"' in attrs else f" ${tex}$ "

    return _MATHML_RE.sub(_sub, html)


_BARE_MATH_HINTS = re.compile(
    r"softmax|matrix|tensor|gradient|eigenvalue|differentiable|optimi[sz]ation|"
    r"\bvector\b|∑|∫|√|∇|≤|≥|≈|±|×", re.I)


def _looks_like_unmarked_math(text: str) -> bool:
    """True when the text reads like it came from a math-heavy source (softmax,
    matrix, summation glyphs...) but carries ZERO $/$$ LaTeX delimiters — i.e. the
    PDF text layer had real formulas and pypdf flattened them into bare prose with
    no markup at all (confirmed live: a NeurIPS PDF's attention-mechanism section
    came back as "divide each by√dk" with no $ anywhere). audit_math alone can't
    catch this because 0 delimiters trivially balance; this is the extraction-time
    companion check that flags the loss BEFORE an LLM ever sees the text."""
    if "$" in text:
        return False
    hits = len(_BARE_MATH_HINTS.findall(text))
    return hits >= 3


def _fix_mojibake(text: Optional[str]) -> Optional[str]:
    """Repair encoding mojibake (UTF-8 decoded as latin-1/cp1252: 'Î»'→'λ', 'â'→'—',
    'Ã©'→'é'). Pages mis-declare their charset, so math/Greek/punctuation arrives
    garbled and poisons briefs + synthesis. ftfy is the canonical fixer; no-op if
    it's unavailable or the text is already clean."""
    if not text:
        return text
    try:
        import ftfy
        return ftfy.fix_text(text)
    except Exception:
        return text


def _extract_text(html: str, url: str) -> Optional[str]:
    try:
        import trafilatura
        text = trafilatura.extract(
            html, include_comments=False, include_tables=True,
            favor_recall=True, url=url,
        )
        return _fix_mojibake(text.strip()) if text else None
    except Exception as exc:
        logger.debug("extract failed %s: %s", url, exc)
        return None


def _extract_title(html: str) -> str:
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        if soup.title and soup.title.string:
            return soup.title.string.strip()[:160]
    except Exception:
        pass
    return ""


def _extract_links(html: str, base_url: str) -> list:
    """Same-domain absolute http(s) links, deduped, in document order."""
    out, seen = [], set()
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        base_dom = _domain(base_url)
        for a in soup.find_all("a", href=True):
            absu = urljoin(base_url, a["href"])
            if not absu.startswith(("http://", "https://")):
                continue
            if _domain(absu) != base_dom:
                continue
            key = _norm_url(absu)
            if key in seen:
                continue
            seen.add(key)
            out.append(absu)
    except Exception:
        pass
    return out


def _social_outlinks(html: str, base_url: str) -> list:
    """Cross-domain VK/Telegram links found on a page — the 'go to their VK group'
    hop. An org's site links out to its social wall, where the live poster/afisha
    actually lives; in-domain crawling would never reach it."""
    out, seen = [], set()
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", href=True):
            absu = urljoin(base_url, a["href"])
            if not absu.startswith(("http://", "https://")) or not is_social_url(absu):
                continue
            key = _norm_url(absu)
            if key in seen:
                continue
            seen.add(key)
            out.append(absu)
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------- #
# Near-duplicate content collapse (Phase 2)
#
# The old dedupe hashed the leading 500 chars, so it only caught BYTE-identical
# reposts. Syndicated wire stories, lightly-edited copies, and the same article
# on N mirror domains all slipped through as "independent" sources — and the
# synthesis step then read N agreeing copies as strong consensus. We now cluster
# pages by word-shingle Jaccard similarity, keep ONE representative per cluster
# (highest authority, then longest), and annotate it with the cluster size +
# member domains so synthesis counts CLUSTERS, not raw URLs, as corroboration.
# --------------------------------------------------------------------------- #
_DUP_SIM = 0.70              # >= this similarity = same underlying content


_SHINGLE_N = 4               # word n-gram size


_SHINGLE_CAP = 400           # cap shingles per page (bounds O(n^2) cost)


_OVERLAP_MIN_SHINGLES = 10   # only trust the containment metric on non-tiny pages


def _shingles(text: str) -> frozenset:
    words = re.sub(r"\s+", " ", (text or "").lower()).split()
    if len(words) < _SHINGLE_N:
        return frozenset(words)
    grams = {" ".join(words[i:i + _SHINGLE_N])
             for i in range(len(words) - _SHINGLE_N + 1)}
    if len(grams) > _SHINGLE_CAP:  # deterministic subsample by sort
        grams = set(sorted(grams)[:_SHINGLE_CAP])
    return frozenset(grams)


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / len(a | b)


def _content_similarity(a: frozenset, b: frozenset) -> float:
    """Max of Jaccard and the overlap (containment) coefficient. Jaccard alone
    under-counts a lightly-edited or trimmed copy when texts are short; the
    overlap coefficient (|A∩B| / min) catches "one is ~contained in the other".
    Containment is only trusted when both pages are non-tiny, so a stray shared
    phrase between two short distinct pages can't force a false merge."""
    j = _jaccard(a, b)
    inter = len(a & b)
    if inter and min(len(a), len(b)) >= _OVERLAP_MIN_SHINGLES:
        return max(j, inter / min(len(a), len(b)))
    return j


def dedupe_pages(pages: list, category: str = "general") -> list:
    """Collapse near-duplicate pages into evidence clusters.

    Returns one representative page per cluster, each annotated with
    ``cluster_size`` (number of independent copies) and ``cluster_domains``
    (their provenance). Representative = highest authority, then longest text."""
    sigs = [_shingles(p["text"]) for p in pages]
    n = len(pages)
    cluster_of = [-1] * n
    clusters: list = []  # each: list of member indices
    for i in range(n):
        if cluster_of[i] != -1:
            continue
        cid = len(clusters)
        cluster_of[i] = cid
        members = [i]
        for j in range(i + 1, n):
            if cluster_of[j] != -1:
                continue
            if _content_similarity(sigs[i], sigs[j]) >= _DUP_SIM:
                cluster_of[j] = cid
                members.append(j)
        clusters.append(members)

    out = []
    for members in clusters:
        rep = max(members, key=lambda k: (
            _authority_score(pages[k]["url"], category), len(pages[k]["text"])))
        page = dict(pages[rep])
        page["cluster_size"] = len(members)
        page["cluster_domains"] = sorted({pages[k]["domain"] for k in members})
        out.append(page)
    if len(out) < n:
        logger.info("Dup-collapse: %d pages -> %d evidence clusters", n, len(out))
    return out

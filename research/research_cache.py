"""URL extraction cache for the deep-research crawler (Phase 6).

Caches the *extracted* result of a URL (clean text + title + content hash) on
disk, keyed by normalized URL, with a category-aware TTL. A cache hit skips both
the network fetch AND the trafilatura extraction — the two slowest non-LLM steps
— so repeat runs and overlapping queries get much faster without re-hammering
sites. Freshness-sensitive categories (news/local/finance) use a short TTL so a
current-events run never serves a stale page; stable reference topics cache for
days. Content-hash is stored so callers can detect when a page changed.

Best-effort and fault-tolerant: any cache error degrades to a miss, never raises
into the engine.
"""
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("assistant.research.cache")

# TTL (seconds) by topic category. Conservative: short where freshness matters.
_TTL_BY_CATEGORY = {
    "news": 3 * 3600,        # 3h — breaking stories move fast
    "finance": 6 * 3600,
    "local": 12 * 3600,      # posters/afisha change but not minute-to-minute
    "jobs": 24 * 3600,
    "company": 3 * 86400,
    "product": 5 * 86400,
    "health": 14 * 86400,
    "people": 14 * 86400,
    "science": 30 * 86400,   # papers don't change
    "legal": 30 * 86400,
    "entertainment": 30 * 86400,
    "general": 7 * 86400,
}
_DEFAULT_TTL = 7 * 86400


def ttl_for(category: str) -> int:
    return _TTL_BY_CATEGORY.get((category or "general").lower(), _DEFAULT_TTL)


def _key(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8", "ignore")).hexdigest()


def content_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", "ignore")).hexdigest()


class ExtractionCache:
    """Tiny file-per-URL JSON cache. Thread-safety is not required: the crawler
    is single-threaded, and a torn write just degrades to a miss."""

    def __init__(self, cache_dir: Path, enabled: bool = True):
        self.dir = Path(cache_dir)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        self.stores = 0
        self.skipped = 0

    def get(self, url: str, category: str = "general") -> Optional[dict]:
        """Return the cached extraction dict if present and within TTL, else None.
        Dict shape: {url, title, text, content_hash, ts}."""
        if not self.enabled:
            return None
        path = self.dir / f"{_key(url)}.json"
        try:
            if not path.exists():
                self.misses += 1
                return None
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
            age = time.time() - rec.get("ts", 0)
            if age > ttl_for(category):
                self.misses += 1
                return None  # stale — caller will refetch and overwrite
            self.hits += 1
            return rec
        except Exception as exc:
            logger.debug("cache get failed %s: %s", url, exc)
            self.misses += 1
            return None

    def put(self, url: str, title: str, text: str) -> None:
        if not self.enabled:
            return
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            rec = {"url": url, "title": title, "text": text,
                   "content_hash": content_hash(text), "ts": time.time()}
            path = self.dir / f"{_key(url)}.json"
            tmp = self.dir / f"{_key(url)}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False)
            tmp.replace(path)
            self.stores += 1
        except Exception as exc:
            logger.debug("cache put failed %s: %s", url, exc)

    # A URL that just failed (fetch error/timeout or content-gate reject) is
    # skipped for this long, so a repeat run does not sit through the same
    # timeouts again.
    FAIL_TTL = 30 * 60

    def failed_recently(self, url: str) -> bool:
        if not self.enabled:
            return False
        try:
            path = self.dir / f"{_key(url)}.fail"
            if path.exists() and time.time() - path.stat().st_mtime < self.FAIL_TTL:
                self.skipped += 1
                return True
        except Exception:
            pass
        return False

    def mark_failed(self, url: str) -> None:
        if not self.enabled:
            return
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / f"{_key(url)}.fail").write_text(url, encoding="utf-8")
        except Exception as exc:
            logger.debug("cache mark_failed %s: %s", url, exc)

    def invalidate(self, url: str) -> None:
        try:
            (self.dir / f"{_key(url)}.json").unlink(missing_ok=True)
        except Exception:
            pass

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "stores": self.stores,
                "skipped_failed": self.skipped}

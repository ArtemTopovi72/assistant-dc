"""Coverage for research_cache.py: category TTLs, key hashing, and the
file-per-URL ExtractionCache (get/put/invalidate/stats, hit/miss/stale/error
paths). Pure filesystem, no network/GPU.
Run: venv/Scripts/python.exe tests/test_research_cache.py
"""
import os, sys, tempfile, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
from pathlib import Path
import research_cache as RC

RESULTS = []
import os as _os
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + detail))
    if not cond and _os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))

_TMP = Path(tempfile.mkdtemp(prefix="rescache_"))


def test_ttl_for():
    check("ttl_news_short", RC.ttl_for("news") == 3 * 3600)
    check("ttl_science_long", RC.ttl_for("science") == 30 * 86400)
    check("ttl_default_general", RC.ttl_for("general") == RC._DEFAULT_TTL)
    check("ttl_unknown_category_default", RC.ttl_for("totally_unknown") == RC._DEFAULT_TTL)
    check("ttl_case_insensitive", RC.ttl_for("NEWS") == RC._TTL_BY_CATEGORY["news"])
    check("ttl_none_defaults", RC.ttl_for(None) == RC._DEFAULT_TTL)


def test_key_and_content_hash():
    k1 = RC._key("https://example.com/a")
    k2 = RC._key("https://example.com/a")
    k3 = RC._key("https://example.com/b")
    check("key_deterministic", k1 == k2)
    check("key_differs_by_url", k1 != k3)
    check("content_hash_deterministic", RC.content_hash("abc") == RC.content_hash("abc"))
    check("content_hash_empty_ok", isinstance(RC.content_hash(None), str))


def test_cache_disabled():
    cache = RC.ExtractionCache(_TMP / "disabled", enabled=False)
    check("cache_disabled_get_none", cache.get("https://x.com/a") is None)
    cache.put("https://x.com/a", "T", "text")
    check("cache_disabled_put_noop", not (_TMP / "disabled").exists())


def test_cache_miss_then_put_then_hit():
    cache = RC.ExtractionCache(_TMP / "c1")
    check("cache_initial_miss", cache.get("https://x.com/a") is None)
    check("cache_stats_miss_recorded", cache.stats()["misses"] == 1)

    cache.put("https://x.com/a", "Title A", "Some text content")
    check("cache_stats_store_recorded", cache.stats()["stores"] == 1)

    rec = cache.get("https://x.com/a")
    check("cache_hit_returns_record", rec is not None and rec["title"] == "Title A")
    check("cache_stats_hit_recorded", cache.stats()["hits"] == 1)


def test_cache_stale_entry():
    cache = RC.ExtractionCache(_TMP / "c2")
    cache.put("https://x.com/stale", "T", "text")
    # manually rewrite the ts field to be older than the TTL
    path = cache.dir / f"{RC._key('https://x.com/stale')}.json"
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec["ts"] = time.time() - (RC.ttl_for("general") + 1000)
    path.write_text(json.dumps(rec), encoding="utf-8")
    out = cache.get("https://x.com/stale", category="general")
    check("cache_stale_entry_returns_none", out is None)
    check("cache_stale_counts_as_miss", cache.stats()["misses"] == 1)


def test_cache_corrupt_file():
    cache = RC.ExtractionCache(_TMP / "c3")
    cache.dir.mkdir(parents=True, exist_ok=True)
    path = cache.dir / f"{RC._key('https://x.com/corrupt')}.json"
    path.write_text("not valid json{{{", encoding="utf-8")
    out = cache.get("https://x.com/corrupt")
    check("cache_corrupt_file_returns_none", out is None)
    check("cache_corrupt_counts_as_miss", cache.stats()["misses"] == 1)


def test_cache_invalidate():
    cache = RC.ExtractionCache(_TMP / "c4")
    cache.put("https://x.com/inv", "T", "text")
    check("cache_invalidate_file_exists_before", cache.get("https://x.com/inv") is not None)
    cache.invalidate("https://x.com/inv")
    check("cache_invalidate_removes_entry", cache.get("https://x.com/inv") is None)
    cache.invalidate("https://x.com/never-existed")  # missing_ok path
    check("cache_invalidate_missing_file_noraise", True)


def test_cache_put_failure_swallowed():
    cache = RC.ExtractionCache(Path(str(_TMP) + "\x00bad_dir"))
    cache.put("https://x.com/a", "T", "text")
    check("cache_put_failure_swallowed", cache.stats()["stores"] == 0)


def test_cache_category_specific_ttl():
    cache = RC.ExtractionCache(_TMP / "c5")
    cache.put("https://x.com/news1", "N", "news text")
    path = cache.dir / f"{RC._key('https://x.com/news1')}.json"
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec["ts"] = time.time() - (2 * 3600)  # 2h old: stale for news (3h TTL) is still fresh
    path.write_text(json.dumps(rec), encoding="utf-8")
    out = cache.get("https://x.com/news1", category="news")
    check("cache_news_ttl_fresh_within_window", out is not None)
    rec["ts"] = time.time() - (4 * 3600)  # 4h old: stale for news (3h TTL)
    path.write_text(json.dumps(rec), encoding="utf-8")
    out2 = cache.get("https://x.com/news1", category="news")
    check("cache_news_ttl_stale_past_window", out2 is None)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback; traceback.print_exc()
            print("  ERROR in " + fn.__name__ + ": " + type(e).__name__ + ": " + str(e))
    passed = sum(1 for _, c in RESULTS if c)
    print(f"\n{len(fns)-failed}/{len(fns)} functions, {passed}/{len(RESULTS)} checks passed")
    # A failed check() must reach the exit code. This used to be `1 if failed else 0`,
    # where `failed` counts EXCEPTIONS only -- so every assertion in the file could
    # print FAIL while the suite still exited 0, invisible to anything judging by
    # exit code (which is how these suites are judged).
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)

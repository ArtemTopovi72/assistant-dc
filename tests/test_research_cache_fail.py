"""A URL that just failed is skipped for FAIL_TTL; then it is tried again."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research_cache import ExtractionCache


def test_failed_url_is_skipped_then_expires(tmp_path):
    c = ExtractionCache(tmp_path)
    assert not c.failed_recently("http://x/a")
    c.mark_failed("http://x/a")
    assert c.failed_recently("http://x/a") and not c.failed_recently("http://x/b")
    assert c.stats()["skipped_failed"] == 1
    old = time.time() - c.FAIL_TTL - 5
    f = next(tmp_path.glob("*.fail"))
    os.utime(f, (old, old))
    assert not c.failed_recently("http://x/a")

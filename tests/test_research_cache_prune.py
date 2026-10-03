"""The research page cache drops entries nothing can serve any more."""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("F5_TEST_RUN", "1")

import research_cache as RC  # noqa: E402


def _age(p, seconds):
    t = time.time() - seconds
    os.utime(p, (t, t))


def test_expired_entries_and_old_fail_marks_are_removed(tmp_path):
    c = RC.ExtractionCache(tmp_path)
    c.put("https://a.example/fresh", "t", "fresh text")
    c.put("https://a.example/old", "t", "old text")
    c.mark_failed("https://a.example/bad")
    old = tmp_path / f"{RC._key('https://a.example/old')}.json"
    bad = tmp_path / f"{RC._key('https://a.example/bad')}.fail"
    _age(old, max(RC._TTL_BY_CATEGORY.values()) + 60)
    _age(bad, c.FAIL_TTL + 60)
    assert c.prune(force=True) == 2
    assert not old.exists() and not bad.exists()
    assert c.get("https://a.example/fresh") is not None


def test_prune_runs_at_most_daily(tmp_path):
    RC.ExtractionCache(tmp_path).put("https://a.example/x", "t", "x")
    p = tmp_path / f"{RC._key('https://a.example/x')}.json"
    _age(p, 10 ** 8)
    RC.ExtractionCache(tmp_path)               # stamped by the first construction already
    assert p.exists()
    (tmp_path / ".pruned").unlink()
    RC.ExtractionCache(tmp_path)
    assert not p.exists()


def test_a_missing_dir_is_fine(tmp_path):
    assert RC.ExtractionCache(tmp_path / "nope").prune(force=True) == 0

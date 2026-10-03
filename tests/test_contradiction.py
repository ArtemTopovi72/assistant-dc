"""Coverage for contradiction.py: contradiction-query template generation
(topic + entity-anchored, budget/dedup-bounded), confirm/contradict tagging,
evidence partitioning, and the contradiction-strength summary. Pure functions.
Run: venv/Scripts/python.exe tests/test_contradiction.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import contradiction as CT

RESULTS = []
import os as _os
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + detail))
    if not cond and _os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))


def test_contradiction_queries_basic():
    out = CT.contradiction_queries("adam optimizer", max_queries=4)
    check("contradiction_queries_bounded_by_max", len(out) == 4)
    check("contradiction_queries_topic_substituted", all("adam optimizer" in q for q in out))


def test_contradiction_queries_entity_anchored():
    entities = [{"value": "AdamW"}, {"value": "RMSProp"}]
    out = CT.contradiction_queries("optimizer comparison", entities=entities, max_queries=10)
    check("contradiction_queries_includes_entity_query", any("AdamW" in q for q in out))
    check("contradiction_queries_multiple_entities", any("RMSProp" in q for q in out))


def test_contradiction_queries_dedup_via_visited():
    visited = {"optimizer limitations"}
    out = CT.contradiction_queries("optimizer", max_queries=8, visited=visited)
    check("contradiction_queries_skips_visited", "optimizer limitations" not in [q.lower() for q in out])
    check("contradiction_queries_visited_updated", len(visited) > 1)


def test_contradiction_queries_empty_topic():
    # A topic with no search signal must produce NO queries: the templates only
    # append angle words ("limitations", "benchmark anomalies"), so an empty
    # anchor would issue template-only searches and pull in unrelated junk
    # sources. _usable_core gates that, and the caller skips the pass.
    check("contradiction_queries_empty_topic_returns_nothing",
          CT.contradiction_queries("", max_queries=3) == [])
    check("contradiction_queries_filler_only_topic_returns_nothing",
          CT.contradiction_queries("покажи мне пожалуйста ссылки", max_queries=3) == [])
    # …but a topic that DOES carry signal still yields queries, so the gate
    # cannot pass by rejecting everything.
    check("contradiction_queries_real_topic_still_returns",
          len(CT.contradiction_queries("adam optimizer", max_queries=3)) == 3)


def test_contradiction_queries_zero_max():
    out = CT.contradiction_queries("topic", max_queries=0)
    check("contradiction_queries_zero_max_empty", out == [])


def test_tag_confirming():
    briefs = [{"title": "A"}, {"title": "B", "mode": "contradict"}]
    CT.tag_confirming(briefs)
    check("tag_confirming_sets_untagged", briefs[0]["mode"] == CT.MODE_CONFIRM)
    check("tag_confirming_preserves_existing", briefs[1]["mode"] == "contradict")


def test_tag_contradicting():
    briefs = [{"title": "A", "mode": "confirm"}, {"title": "B"}]
    CT.tag_contradicting(briefs)
    check("tag_contradicting_overwrites_all", all(b["mode"] == CT.MODE_CONTRADICT for b in briefs))


def test_partition_evidence():
    briefs = [{"mode": "confirm"}, {"mode": "contradict"}, {}]  # untagged defaults to confirm
    confirm, contradict = CT.partition_evidence(briefs)
    check("partition_confirm_count", len(confirm) == 2)
    check("partition_contradict_count", len(contradict) == 1)


def test_contradiction_summary_none():
    out = CT.contradiction_summary([{"mode": "confirm"}])
    check("summary_no_contradictions", out["strength"] == "none" and out["has_contradictions"] is False)


def test_contradiction_summary_weak():
    briefs = [{"mode": "contradict", "trust": "COMMUNITY", "domain": "reddit.com"}]
    out = CT.contradiction_summary(briefs)
    check("summary_weak_single_low_trust", out["strength"] == "weak")


def test_contradiction_summary_moderate_by_count():
    briefs = [{"mode": "contradict", "trust": "COMMUNITY", "domain": "a.com"},
              {"mode": "contradict", "trust": "LOW", "domain": "b.com"}]
    out = CT.contradiction_summary(briefs)
    check("summary_moderate_by_count", out["strength"] == "moderate")


def test_contradiction_summary_moderate_by_one_strong():
    briefs = [{"mode": "contradict", "trust": "PRIMARY", "domain": "a.com"}]
    out = CT.contradiction_summary(briefs)
    check("summary_moderate_one_strong", out["strength"] == "moderate")


def test_contradiction_summary_strong():
    briefs = [{"mode": "contradict", "trust": "PRIMARY", "domain": "a.com"},
              {"mode": "contradict", "trust": "SECONDARY", "domain": "b.com"}]
    out = CT.contradiction_summary(briefs)
    check("summary_strong_two_strong_sources", out["strength"] == "strong")
    check("summary_domains_sorted", out["domains"] == ["a.com", "b.com"])


def test_contradiction_summary_domain_dedup_and_missing():
    briefs = [{"mode": "contradict", "trust": "LOW", "domain": "a.com"},
              {"mode": "contradict", "trust": "LOW", "domain": "a.com"},
              {"mode": "contradict", "trust": "LOW"}]  # no domain key
    out = CT.contradiction_summary(briefs)
    check("summary_domain_dedup", out["domains"] == ["a.com"])
    check("summary_count_includes_all", out["count"] == 3)


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

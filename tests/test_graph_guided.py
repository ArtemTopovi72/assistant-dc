"""Coverage for graph_guided.py: entity centrality ranking, weak-claim
detection, and the graph/cluster-guided expansion planners (entity hubs,
contradiction domains, weak claims, thin-major-clusters; plus the
cluster-prioritized contradiction targets). Pure functions against small fake
graph/clusterset objects (no real research_graph/clustering dependency needed
for structural coverage).
Run: venv/Scripts/python.exe tests/test_graph_guided.py
"""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import graph_guided as GG

RESULTS = []
import os as _os
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + detail))
    if not cond and _os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))


class FakeGraph:
    def __init__(self, nodes=None, edges=None):
        self.nodes = nodes or {}
        self.edges = edges or []


class FakeCluster:
    def __init__(self, id, label, score, size, top_entities=None):
        self.id = id
        self.label = label
        self.score = score
        self._size = size
        self._top_entities = top_entities or []
    def size(self):
        return self._size
    def to_dict(self):
        return {"top_entities": self._top_entities}


class FakeClusterSet:
    def __init__(self, majors):
        self._majors = majors
    def major(self):
        return self._majors


def test_entity_centrality():
    graph = FakeGraph(
        nodes={"e1": {"value": "Adam", "category": "proper"},
              "e2": {"value": "RMSProp", "category": "proper"}},
        edges=[
            {"rel": "mentioned_in", "src": "e1", "dst": "s1"},
            {"rel": "mentioned_in", "src": "e1", "dst": "s2"},
            {"rel": "mentioned_in", "src": "e2", "dst": "s1"},
            {"rel": "supported_by", "src": "c1", "dst": "s1"},  # ignored, wrong rel
        ])
    out = GG.entity_centrality(graph)
    check("entity_centrality_sorted_desc", out[0]["value"] == "Adam" and out[0]["degree"] == 2)
    check("entity_centrality_second_entry", out[1]["value"] == "RMSProp" and out[1]["degree"] == 1)

    graph_missing_meta = FakeGraph(nodes={}, edges=[{"rel": "mentioned_in", "src": "ghost", "dst": "s1"}])
    out2 = GG.entity_centrality(graph_missing_meta)
    check("entity_centrality_missing_node_meta_included_with_defaults", len(out2) == 1 and out2[0]["value"] == "")


def test_weak_claims():
    graph = FakeGraph(
        nodes={"c1": {"trust": "COMMUNITY", "text": "some claim", "source_url": "u1"},
              "c2": {"trust": "PRIMARY", "text": "strong claim"},
              "c3": {"trust": "LOW", "text": "weak claim2"},
              "s1": {"domain": "reddit.com"}},
        edges=[
            {"rel": "supported_by", "src": "c1", "dst": "s1"},  # single low-trust -> weak
            {"rel": "supported_by", "src": "c2", "dst": "s1"},  # primary -> not weak
            {"rel": "supported_by", "src": "c3", "dst": "s1"},
            {"rel": "supported_by", "src": "c3", "dst": "s1"},  # 2 sources -> not weak (even if low trust)... actually same src twice
        ])
    out = GG.weak_claims(graph)
    check("weak_claims_single_low_trust_included", any(w["claim"] == "some claim" for w in out))
    check("weak_claims_primary_excluded", not any(w["claim"] == "strong claim" for w in out))


def test_weak_claims_custom_low_trust_set():
    graph = FakeGraph(
        nodes={"c1": {"trust": "SECONDARY", "text": "claim", "source_url": ""}},
        edges=[{"rel": "supported_by", "src": "c1", "dst": "s1"}])
    out = GG.weak_claims(graph, low_trust=("SECONDARY",))
    check("weak_claims_custom_low_trust", len(out) == 1)


def test_push_helper():
    out, visited = [], set()
    check("push_adds_new", GG._push(out, visited, "a query", "reason", "target", 5))
    check("push_dedup_skips", not GG._push(out, visited, "A QUERY", "reason2", "target2", 5))
    check("push_empty_query_rejected", not GG._push(out, visited, "   ", "r", "t", 5))
    check("push_respects_max_total", not GG._push(out, visited, "another", "r", "t", 1))


def test_plan_graph_expansion_entity_hubs():
    graph = FakeGraph(
        nodes={"e1": {"value": "Adam", "category": "proper"}},
        edges=[{"rel": "mentioned_in", "src": "e1", "dst": "s1"}])
    out = GG.plan_graph_expansion("adam optimizer", graph, max_queries=6)
    check("plan_expansion_entity_hub_included", any(o["target"] == "entity:Adam" for o in out))


def test_plan_graph_expansion_contradictions():
    graph = FakeGraph()
    out = GG.plan_graph_expansion(
        "topic", graph, contradiction={"has_contradictions": True, "domains": ["a.com", "b.com"]},
        max_queries=6)
    check("plan_expansion_contradiction_domains", any("contradiction:a.com" in o["target"] for o in out)
          and any("contradiction:b.com" in o["target"] for o in out))


def test_plan_graph_expansion_weak_claims():
    graph = FakeGraph(
        nodes={"c1": {"trust": "COMMUNITY", "text": "weak claim text"}},
        edges=[{"rel": "supported_by", "src": "c1", "dst": "s1"}])
    out = GG.plan_graph_expansion("topic", graph, max_queries=6)
    check("plan_expansion_weak_claim_included", any("weakclaim:" in o["target"] for o in out))


def test_plan_graph_expansion_major_clusters():
    graph = FakeGraph()
    clusters = FakeClusterSet([FakeCluster("cl1", "Thread A", 0.9, 2)])
    out = GG.plan_graph_expansion("topic", graph, clusterset=clusters, max_queries=6)
    check("plan_expansion_major_cluster_thin", any("cluster:cl1" in o["target"] for o in out))

    big_cluster = FakeClusterSet([FakeCluster("cl2", "", 0.5, 5, top_entities=["fallback label"])])
    out2 = GG.plan_graph_expansion("topic", graph, clusterset=big_cluster, max_queries=6)
    check("plan_expansion_big_cluster_not_deep_dived", not any("cluster:cl2" in o["target"] for o in out2))


def test_plan_graph_expansion_bounded_by_max_queries():
    graph = FakeGraph(
        nodes={f"e{i}": {"value": f"Entity{i}"} for i in range(10)},
        edges=[{"rel": "mentioned_in", "src": f"e{i}", "dst": "s1"} for i in range(10)])
    out = GG.plan_graph_expansion("topic", graph, max_queries=3, top_entities=10)
    check("plan_expansion_bounded", len(out) == 3)


def test_plan_graph_expansion_no_contradiction_flag():
    graph = FakeGraph()
    out = GG.plan_graph_expansion("topic", graph, contradiction={"has_contradictions": False})
    check("plan_expansion_no_contradiction_no_targets", not any("contradiction:" in o["target"] for o in out))


def test_cluster_contradiction_targets():
    graph = FakeGraph(
        nodes={"e1": {"value": "Adam"}},
        edges=[{"rel": "mentioned_in", "src": "e1", "dst": "s1"}])
    clusters = FakeClusterSet([FakeCluster("cl1", "Thread A", 0.9, 3)])
    out = GG.cluster_contradiction_targets("topic", graph, clusters, max_queries=4)
    check("cluster_contradiction_has_cluster_target", any("cluster:cl1" in o["target"] for o in out))
    check("cluster_contradiction_has_entity_target", any("entity:Adam" in o["target"] for o in out))


def test_cluster_contradiction_targets_no_clusterset():
    graph = FakeGraph(nodes={"e1": {"value": "X"}},
                      edges=[{"rel": "mentioned_in", "src": "e1", "dst": "s1"}])
    out = GG.cluster_contradiction_targets("topic", graph, None, max_queries=4)
    check("cluster_contradiction_no_clusterset_still_has_entities", any("entity:X" in o["target"] for o in out))


def test_cluster_contradiction_targets_bounded():
    graph = FakeGraph(
        nodes={f"e{i}": {"value": f"E{i}"} for i in range(10)},
        edges=[{"rel": "mentioned_in", "src": f"e{i}", "dst": "s1"} for i in range(10)])
    clusters = FakeClusterSet([FakeCluster(f"cl{i}", f"L{i}", 0.5, 1) for i in range(10)])
    out = GG.cluster_contradiction_targets("topic", graph, clusters, max_queries=2)
    check("cluster_contradiction_bounded", len(out) == 2)


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

"""Coverage for decision_gate.py: content-word extraction, ambiguity
assessment, the ANSWER/ASK/ABSTAIN decision function across all reason
combinations, and the banner formatter. Pure functions.
Run: venv/Scripts/python.exe tests/test_decision_gate.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import decision_gate as DG

RESULTS = []
import os as _os
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + detail))
    if not cond and _os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))


def test_content_words():
    check("content_words_filters_stopwords", DG._content_words("what is the capital of France") == ["capital", "france"])
    check("content_words_empty", DG._content_words("") == [])
    check("content_words_none", DG._content_words(None) == [])
    check("content_words_single_char_dropped", DG._content_words("a b c d") == [])


def test_assess_ambiguity():
    out = DG.assess_ambiguity("")
    check("ambiguity_empty_topic", out["ambiguous"] is True and "empty" in out["reasons"][0])
    out2 = DG.assess_ambiguity("cat", has_entities=False)
    check("ambiguity_single_vague_no_entity", out2["ambiguous"] is True)
    out3 = DG.assess_ambiguity("cat", has_entities=True)
    check("ambiguity_single_word_but_has_entity_not_ambiguous", out3["ambiguous"] is False)
    out4 = DG.assess_ambiguity("explain quantum tunneling in depth")
    check("ambiguity_normal_topic_not_ambiguous", out4["ambiguous"] is False)
    out5 = DG.assess_ambiguity("elephant", has_entities=False)
    check("ambiguity_long_single_word_not_ambiguous", out5["ambiguous"] is False)


def test_decide_ask():
    out = DG.decide({"independent_clusters": 0, "strong_sources": 0, "max_authority": 0.0},
                    ambiguity={"ambiguous": True, "reasons": ["empty request"]})
    check("decide_ask_on_ambiguous", out["decision"] == DG.ASK and out["clarifying_question"])


def test_decide_abstain_no_clusters():
    out = DG.decide({"independent_clusters": 0, "strong_sources": 0, "max_authority": 0.0})
    check("decide_abstain_no_clusters", out["decision"] == DG.ABSTAIN)
    check("decide_abstain_reasons_mention_no_evidence", "no usable evidence" in out["reasons"][0])


def test_decide_abstain_weak_evidence():
    out = DG.decide({"independent_clusters": 1, "strong_sources": 0, "max_authority": 0.2},
                    min_clusters=2)
    check("decide_abstain_weak_evidence", out["decision"] == DG.ABSTAIN)


def test_decide_answer_confident():
    out = DG.decide({"independent_clusters": 3, "strong_sources": 2, "max_authority": 0.9},
                    min_strong=1, min_clusters=2)
    check("decide_answer_confident", out["decision"] == DG.ANSWER and out["confidence"] == "high"
          and out["uncertain"] is False)


def test_decide_answer_uncertain_low_clusters():
    out = DG.decide({"independent_clusters": 1, "strong_sources": 1, "max_authority": 0.9},
                    min_strong=1, min_clusters=2)
    check("decide_answer_uncertain_confidence_medium", out["decision"] == DG.ANSWER
          and out["uncertain"] is True and out["confidence"] == "medium")


def test_decide_answer_weak_coverage():
    out = DG.decide({"independent_clusters": 3, "strong_sources": 2, "max_authority": 0.9},
                    coverage={"weak": True, "reasons": ["thin: only 3 sources"]},
                    min_strong=1, min_clusters=2)
    check("decide_answer_weak_coverage_uncertain", out["uncertain"] is True)
    check("decide_answer_weak_coverage_reason_included", any("coverage weak" in r for r in out["reasons"]))


def test_decide_answer_contradiction_moderate():
    out = DG.decide({"independent_clusters": 3, "strong_sources": 2, "max_authority": 0.9},
                    contradiction={"has_contradictions": True, "strength": "moderate"},
                    min_strong=1, min_clusters=2)
    check("decide_answer_contradiction_uncertain", out["uncertain"] is True)
    check("decide_answer_contradiction_reason", any("contradictions" in r for r in out["reasons"]))


def test_decide_answer_contradiction_strong():
    out = DG.decide({"independent_clusters": 3, "strong_sources": 2, "max_authority": 0.9},
                    contradiction={"has_contradictions": True, "strength": "strong"},
                    min_strong=1, min_clusters=2)
    check("decide_answer_contradiction_strong_uncertain", out["uncertain"] is True)


def test_decide_answer_low_confidence_no_strong():
    out = DG.decide({"independent_clusters": 3, "strong_sources": 0, "max_authority": 0.6},
                    min_strong=1, min_clusters=2)
    check("decide_answer_low_confidence_no_strong", out["confidence"] == "low")


def test_decide_default_reason_when_none_set():
    out = DG.decide({"independent_clusters": 5, "strong_sources": 3, "max_authority": 0.9},
                    min_strong=1, min_clusters=2)
    check("decide_default_reason_present", "clusters" in out["reasons"][0])


def test_gate_banner():
    ask = {"decision": DG.ASK, "clarifying_question": "What do you mean?"}
    check("banner_ask", "Clarification needed" in DG.gate_banner(ask) and "What do you mean?" in DG.gate_banner(ask))
    abstain = {"decision": DG.ABSTAIN, "reasons": ["no evidence"]}
    check("banner_abstain", "Insufficient evidence" in DG.gate_banner(abstain) and "no evidence" in DG.gate_banner(abstain))
    answer_confident = {"decision": DG.ANSWER, "confidence": "high", "uncertain": False}
    check("banner_answer_confident", "confidently" in DG.gate_banner(answer_confident))
    answer_uncertain = {"decision": DG.ANSWER, "confidence": "medium", "uncertain": True}
    check("banner_answer_uncertain", "explicit uncertainty" in DG.gate_banner(answer_uncertain))
    check("banner_defaults_to_answer_when_missing", "confidently" in DG.gate_banner({}) or "uncertainty" in DG.gate_banner({}))


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

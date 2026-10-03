"""The jina listwise adapter must map its sorted output back to input order."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import library as L


class _FakeJina:
    def rerank(self, query, docs):
        # sorted by score, like the real model: best first
        scored = [{"index": i, "relevance_score": float(len(d)), "document": d} for i, d in enumerate(docs)]
        return sorted(scored, key=lambda r: -r["relevance_score"])


def _adapter():
    a = L._ListwiseReranker.__new__(L._ListwiseReranker)
    a._model = _FakeJina()
    return a


def test_scores_come_back_in_input_order():
    assert _adapter().predict([("q", "aaa"), ("q", "a"), ("q", "aa")]) == [3.0, 1.0, 2.0]


def test_empty_pool():
    assert _adapter().predict([]) == []


def test_rerank_puts_best_first():
    class R:
        def __init__(self, t): self.text = t
    pool = [R("x"), R("xxxxxxxxxx"), R("xx")]
    assert L.rerank("q", pool, 1, cross_encoder=_adapter())[0].text == "xxxxxxxxxx"


def test_load_failure_degrades_to_none(monkeypatch):
    L._cross_encoder_cache.pop("nope/jina-reranker-v3-missing", None)
    assert L.load_cross_encoder("nope/jina-reranker-v3-missing") is None

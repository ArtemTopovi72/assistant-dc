"""Offline regressions for persisted knowledge and cache ingestion."""
import json
import sqlite3

import pytest

import knowledge as K


class _Vectors(K.Embedder):
    def __init__(self, model):
        super().__init__(model=model)

    def available(self, recheck=30.0):
        return True

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


@pytest.mark.parametrize("numpy", [True, False])
def test_model_switch_uses_fts_until_current_vectors_exist(tmp_path, numpy):
    kb = K.KnowledgeBase(tmp_path / "kb.db", embedder=_Vectors("old"),
                         np=K._np if numpy else None)
    try:
        kb.add_document("doc", "source", "Orchids need water.")
        kb.embedder = _Vectors("new")
        assert kb.search_semantic("unrelated subject") == []
        hits = kb.search("Orchids")
        assert len(hits) == 1 and hits[0].why == "BM25 keyword"
        assert kb.ensure_embeddings() == 1
        assert len(kb.search_semantic("Orchids")) == 1
    finally:
        kb.close()


def test_failed_document_replacement_preserves_old_chunks(tmp_path):
    kb = K.KnowledgeBase(tmp_path / "kb.db", embedder=K.Embedder(enabled=False))
    try:
        kb.add_document("doc", "source", "Original orchids", embed=False)
        kb._conn.execute("""CREATE TRIGGER reject_replacement BEFORE INSERT ON chunks
            WHEN new.text = 'Replacement text'
            BEGIN SELECT RAISE(ABORT, 'write rejected'); END""")
        with pytest.raises(sqlite3.IntegrityError, match="write rejected"):
            kb.add_document("doc", "source", "Replacement text", embed=False)
        # A later successful ingestion must never commit a half-replaced document.
        kb.add_document("doc", "other", "Other flowers", embed=False)
        assert [hit.text for hit in kb.search("orchids", mode="fts")] == ["Original orchids"]
    finally:
        kb.close()


def test_cache_import_skips_malformed_records_and_keeps_valid_sources(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    records = [[], {"url": "bad", "text": ["not text"]},
               {"url": ["not a URL"], "text": "Bad source"},
               {"url": "https://example.org", "text": "Orchids need water."}]
    for index, record in enumerate(records):
        (cache / f"{index}.json").write_text(json.dumps(record), encoding="utf-8")
    kb = K.KnowledgeBase(tmp_path / "kb.db", embedder=K.Embedder(enabled=False))
    try:
        result = kb.import_research_cache(cache, embed=False)
        assert result["errors"] == 3
        assert result["documents_indexed"] == 1
        assert len(kb.search("Orchids", mode="fts")) == 1
    finally:
        kb.close()

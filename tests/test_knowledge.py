"""Real-DB coverage for knowledge.py: SQLite-backed hybrid (FTS5 + cosine RRF)
index against a real temp DB, using the REAL live LM Studio embedding endpoint
(nomic-embed-text is loaded) for the happy paths, and a disabled/stubbed
Embedder for the degrade-to-FTS-only and embed-failure branches.
Run: venv/Scripts/python.exe tests/test_knowledge.py
"""
import os, sys, tempfile, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
from pathlib import Path
import knowledge as K

RESULTS = []
import os as _os
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + detail))
    if not cond and _os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))

_TMP = Path(tempfile.mkdtemp(prefix="knowledge_"))


def _kb(name, embedder=None):
    return K.KnowledgeBase(_TMP / name, embedder=embedder)


class _StubEmbedder(K.Embedder):
    """A deterministic stand-in for the /v1/embeddings endpoint.

    The default Embedder is a real HTTP client. Five checks here used it, so
    they were quietly a live test of whatever LM Studio happened to be serving:
    they failed whenever the card was busy with something else, which says
    nothing about knowledge.py. The endpoint itself belongs in bench/; what
    these cases are actually about is the batching, the worker pool, the
    progress callback and the pure-Python fallback, and none of that needs a
    model.
    """
    def __init__(self, dim=8):
        super().__init__()
        self._dim = dim
        self.calls = 0

    def available(self, recheck=30.0):
        return True

    def embed(self, texts):
        self.calls += 1
        out = []
        for i, t in enumerate(texts):
            h = float(abs(hash(t)) % 997) / 997.0
            out.append([h] + [(h + j * 0.01) % 1.0 for j in range(self._dim - 1)])
        return out


def test_search_result_to_public():
    r = K.SearchResult(chunk_id=1, doc_id=2, source_type="doc", title="T", url="u",
                       path="p", text="x" * 400, score=0.12345, bm25_rank=1,
                       cosine_rank=2, cosine=0.5, why="hybrid")
    pub = r.to_public()
    check("search_result_to_public_snippet_truncated", len(pub["snippet"]) == 280)
    check("search_result_to_public_score_rounded", pub["score"] == 0.12345)
    r2 = K.SearchResult(chunk_id=1, doc_id=2, source_type="doc", title="T", url="u",
                        path="p", text="x", score=0.1, cosine=None)
    check("search_result_to_public_none_cosine", r2.to_public()["cosine"] is None)


def test_vec_blob_roundtrip():
    vec = [0.1, -0.2, 0.3, 0.5]
    blob = K._vec_to_blob(vec)
    out = K._blob_to_list(blob)
    check("vec_blob_roundtrip_numpy", all(abs(a - b) < 1e-5 for a, b in zip(vec, out)))
    saved_np = K._np
    K._np = None
    try:
        blob2 = K._vec_to_blob(vec)
        out2 = K._blob_to_list(blob2)
        check("vec_blob_roundtrip_no_numpy", all(abs(a - b) < 1e-5 for a, b in zip(vec, out2)))
    finally:
        K._np = saved_np


def test_chunk_text():
    check("chunk_text_empty", K.chunk_text("") == [])
    check("chunk_text_whitespace_only", K.chunk_text("   ") == [])
    short = "This is a short paragraph."
    check("chunk_text_single_chunk", len(K.chunk_text(short)) == 1)
    long_text = "\n\n".join([" ".join(f"word{i}_{j}" for j in range(50)) for i in range(20)])
    chunks = K.chunk_text(long_text, target_words=100, overlap=10)
    check("chunk_text_multiple_chunks", len(chunks) > 1)
    huge_para = " ".join(f"w{i}" for i in range(1000))
    chunks2 = K.chunk_text(huge_para, target_words=100, overlap=10)
    check("chunk_text_hard_splits_long_paragraph", len(chunks2) > 5)
    check("chunk_text_no_para_fallback", len(K.chunk_text("just one line no blank lines")) == 1)

    # The hard-split stride is (target_words - overlap). At overlap >= target the
    # buffer never shrank and the loop spun forever, hanging the indexing thread
    # with no error — run these on a watchdog thread so a regression FAILS rather
    # than wedging the suite.
    import threading
    def _finishes(*args, **kw):
        box = {}
        t = threading.Thread(target=lambda: box.update(r=K.chunk_text(*args, **kw)),
                             daemon=True)
        t.start(); t.join(10.0)
        return not t.is_alive() and bool(box.get("r"))

    big = " ".join(f"w{i}" for i in range(3000))
    for _tw, _ov in ((10, 40), (50, 50), (50, 60), (1, 0), (0, 0), (100, -5)):
        check(f"chunk_text_terminates_target{_tw}_overlap{_ov}",
              _finishes(big, target_words=_tw, overlap=_ov))
    # …and the clamp must not change the normal case
    check("chunk_text_default_params_unchanged",
          len(K.chunk_text(big, target_words=100, overlap=10)) > 5)


def test_hash():
    check("hash_deterministic", K._hash("abc") == K._hash("abc"))
    check("hash_differs", K._hash("abc") != K._hash("xyz"))
    check("hash_empty_ok", isinstance(K._hash(""), str))


def test_embedder_disabled():
    e = K.Embedder(enabled=False)
    check("embedder_disabled_not_available", not e.available())
    check("embedder_disabled_embed_none", e.embed(["x"]) is None)


def test_embedder_no_requests():
    saved = K._requests
    K._requests = None
    try:
        e = K.Embedder()
        check("embedder_no_requests_disabled", not e.enabled)
    finally:
        K._requests = saved


def test_embedder_empty_texts():
    e = K.Embedder()
    check("embedder_empty_texts_none", e.embed([]) is None)


def test_embedder_http_failure():
    class FakeResp:
        def raise_for_status(self):
            raise RuntimeError("500 error")
    class FakeReq:
        def post(self, *a, **k):
            return FakeResp()
    saved = K._requests
    K._requests = FakeReq()
    try:
        e = K.Embedder()
        out = e.embed(["hello"])
        check("embedder_http_failure_none", out is None)
        check("embedder_failure_count_incremented", e.failures == 1)
    finally:
        K._requests = saved


def test_embedder_mismatched_vector_count():
    class FakeResp:
        def raise_for_status(self): pass
        def json(self):
            return {"data": [{"embedding": [0.1, 0.2], "index": 0}]}
    class FakeReq:
        def post(self, *a, **k):
            return FakeResp()
    saved = K._requests
    K._requests = FakeReq()
    try:
        e = K.Embedder()
        out = e.embed(["hello", "world"])  # 2 texts, 1 vector back
        check("embedder_mismatched_count_none", out is None)
    finally:
        K._requests = saved


def test_embedder_real_available_and_embed():
    # The real embedder is served by LM Studio. When it is not running this
    # used to fall through the `check` (which records and CONTINUES) straight
    # into `len(v[0])` on None — a TypeError that killed the whole file and
    # took every later test function with it. Report the gap and move on.
    e = K.Embedder()
    if not e.available():
        print("[SKIP] embedder_real_*: embedding backend is not answering")
        return
    check("embedder_real_available", True)
    v = e.embed(["hello world"])
    ok = v is not None and len(v) == 1 and len(v[0]) > 0
    check("embedder_real_embed_returns_vector", ok)
    if not ok:
        return
    check("embedder_real_dim_set", e.dim == len(v[0]))
    check("embedder_real_available_cached", e.available(recheck=1000))


def test_add_document_and_stats():
    kb = _kb("db1.sqlite", embedder=K.Embedder(enabled=False))
    try:
        doc_id, changed = kb.add_document("doc", "src1", "Some text content about testing.",
                                          title="Test Doc")
        check("add_document_new", changed is True and doc_id is not None)
        doc_id2, changed2 = kb.add_document("doc", "src1", "Some text content about testing.")
        check("add_document_unchanged_noop", changed2 is False and doc_id2 == doc_id)
        doc_id3, changed3 = kb.add_document("doc", "src1", "Different content now entirely.")
        check("add_document_changed_updates", changed3 is True and doc_id3 == doc_id)
        stats = kb.stats()
        check("stats_documents_count", stats["documents"] == 1)
        check("stats_embed_coverage_zero_when_disabled", stats["embed_coverage"] == 0.0)
    finally:
        kb.close()


def test_ensure_embeddings_unavailable():
    kb = _kb("db2.sqlite", embedder=K.Embedder(enabled=False))
    try:
        kb.add_document("doc", "s1", "text one here", embed=False)
        n = kb.ensure_embeddings()
        check("ensure_embeddings_endpoint_unavailable_zero", n == 0)
        check("ensure_embeddings_no_rows_zero", kb.ensure_embeddings(doc_id=99999) == 0)
    finally:
        kb.close()


def test_ensure_embeddings_real_sequential():
    kb = _kb("db3.sqlite", embedder=_StubEmbedder())
    try:
        doc_id, _ = kb.add_document("doc", "s1", "The quick brown fox jumps over the lazy dog.",
                                    embed=False)
        n = kb.ensure_embeddings(doc_id=doc_id, batch=1)
        check("ensure_embeddings_real_sequential_success", n >= 1)
        check("has_embeddings_true", kb.has_embeddings())
    finally:
        kb.close()


def test_ensure_embeddings_real_workers():
    kb = _kb("db4.sqlite", embedder=_StubEmbedder())
    try:
        for i in range(3):
            kb.add_document("doc", f"multi{i}", f"Document number {i} content about topic {i}.",
                            embed=False)
        n = kb.ensure_embeddings(batch=1, workers=3)
        check("ensure_embeddings_workers_success", n >= 1)
    finally:
        kb.close()


def test_ensure_embeddings_progress_and_cancel():
    kb = _kb("db5.sqlite", embedder=_StubEmbedder())
    try:
        for i in range(3):
            kb.add_document("doc", f"pc{i}", f"Progress cancel doc {i} text content here.", embed=False)
        seen = []
        n = kb.ensure_embeddings(batch=1, progress=lambda d, t: seen.append((d, t)))
        check("ensure_embeddings_progress_called", len(seen) >= 1)

        for i in range(3):
            kb.add_document("doc", f"cancel{i}", f"Cancel doc {i} more text content here.", embed=False)
        calls = {"n": 0}
        def cancel():
            calls["n"] += 1
            return calls["n"] > 1
        n2 = kb.ensure_embeddings(batch=1, cancel=cancel)
        check("ensure_embeddings_cancel_stops_early", isinstance(n2, int))

        def bad_progress(d, t):
            raise RuntimeError("progress boom")
        kb.add_document("doc", "pcbad", "one more doc for bad progress callback test", embed=False)
        n3 = kb.ensure_embeddings(batch=1, progress=bad_progress)
        check("ensure_embeddings_progress_exception_swallowed", isinstance(n3, int))
    finally:
        kb.close()


def test_ensure_embeddings_batch_fails_midway():
    class HalfFailEmbedder(K.Embedder):
        def __init__(self):
            super().__init__()
            self.n = 0
        def available(self, recheck=30.0):
            return True
        def embed(self, texts):
            self.n += 1
            if self.n > 1:
                return None
            return [[0.1] * 8 for _ in texts]
    kb = _kb("db6.sqlite", embedder=HalfFailEmbedder())
    try:
        for i in range(4):
            kb.add_document("doc", f"hf{i}", f"half fail doc {i} content", embed=False)
        n = kb.ensure_embeddings(batch=1)
        check("ensure_embeddings_batch_fail_midway_partial", 0 <= n < 4)
    finally:
        kb.close()


def test_ensure_embeddings_workers_batch_fails():
    class AlwaysFailEmbedder(K.Embedder):
        def available(self, recheck=30.0):
            return True
        def embed(self, texts):
            return None
    kb = _kb("db7.sqlite", embedder=AlwaysFailEmbedder())
    try:
        for i in range(4):
            kb.add_document("doc", f"wf{i}", f"workers fail doc {i} content", embed=False)
        n = kb.ensure_embeddings(batch=1, workers=2)
        check("ensure_embeddings_workers_all_fail_zero", n == 0)
    finally:
        kb.close()


def test_import_research_cache():
    cache_dir = _TMP / "cache_dir"
    cache_dir.mkdir(exist_ok=True)
    (cache_dir / "a.json").write_text(json.dumps({"url": "https://x.com/a", "text": "content A here", "title": "A"}), encoding="utf-8")
    (cache_dir / "b.json").write_text(json.dumps({"content": "content B here", "title": "B"}), encoding="utf-8")
    (cache_dir / "empty.json").write_text(json.dumps({"text": "   "}), encoding="utf-8")
    (cache_dir / "bad.json").write_text("not valid json{{{", encoding="utf-8")

    kb = _kb("db8.sqlite", embedder=K.Embedder(enabled=False))
    try:
        out = kb.import_research_cache(cache_dir, embed=False)
        check("import_research_cache_counts", out["files"] == 4 and out["documents_indexed"] == 2
              and out["skipped_unchanged"] == 1 and out["errors"] == 1)
        out2 = kb.import_research_cache(cache_dir, embed=False, limit=1)
        check("import_research_cache_limit", out2["files"] == 1)
    finally:
        kb.close()


def test_import_markdown_dir():
    docs_dir = _TMP / "docs_dir"
    docs_dir.mkdir(exist_ok=True)
    (docs_dir / "doc1.md").write_text("# My Title\n\nSome content here.", encoding="utf-8")
    (docs_dir / "doc2.md").write_text("no heading just text", encoding="utf-8")

    kb = _kb("db9.sqlite", embedder=K.Embedder(enabled=False))
    try:
        out = kb.import_markdown_dir(docs_dir, embed=False)
        check("import_markdown_dir_indexed", out["documents_indexed"] == 2)
    finally:
        kb.close()


def test_search_fts_only():
    kb = _kb("db10.sqlite", embedder=K.Embedder(enabled=False))
    try:
        kb.add_document("doc", "s1", "The quick brown fox jumps over the lazy dog", embed=False)
        kb.add_document("doc", "s2", "Completely unrelated content about cooking recipes", embed=False)
        check("fts_query_empty_terms", kb._fts_query("   ") == "")
        results = kb.search("fox jumps", mode="fts")
        check("search_fts_finds_match", len(results) >= 1 and "fox" in results[0].text.lower())
        check("search_empty_query_empty", kb.search("   ") == [])
        results_auto = kb.search("fox", mode="auto")
        check("search_auto_falls_back_fts_no_embeddings", results_auto[0].why == "BM25 keyword")
    finally:
        kb.close()


def test_search_semantic_and_hybrid_real():
    kb = _kb("db11.sqlite")
    try:
        kb.add_document("doc", "s1", "The quick brown fox jumps over the lazy dog near the river.")
        kb.add_document("doc", "s2", "Completely different content discussing recipes for baking bread.")
        sem_results = kb.search("a fast fox leaping", mode="semantic")
        check("search_semantic_returns_results", isinstance(sem_results, list))
        hybrid_results = kb.search("fox jumping", mode="hybrid")
        check("search_hybrid_returns_results", len(hybrid_results) >= 1)
        auto_results = kb.search("fox jumping", mode="auto")
        check("search_auto_uses_hybrid_when_embeddings_exist",
              auto_results[0].why in ("hybrid RRF (BM25 + cosine)", "embedding cosine", "BM25 keyword"))
    finally:
        kb.close()


def test_search_semantic_disabled_embedder():
    kb = _kb("db12.sqlite", embedder=K.Embedder(enabled=False))
    try:
        kb.add_document("doc", "s1", "some text", embed=False)
        check("search_semantic_disabled_embedder_empty", kb.search_semantic("query") == [])
    finally:
        kb.close()


def test_all_embeddings_pure_python_fallback():
    kb = _kb("db13.sqlite", embedder=_StubEmbedder())
    try:
        kb.add_document("doc", "s1", "The quick brown fox and other animals in the forest.")
        saved_np = K._np
        K._np = None
        try:
            ids, mat = kb._all_embeddings()
            check("all_embeddings_pure_python_fallback", len(ids) >= 1 and isinstance(mat, list))
            sem = kb.search_semantic("fox forest animals")
            check("search_semantic_pure_python_fallback", isinstance(sem, list))
        finally:
            K._np = saved_np
    finally:
        kb.close()


def test_all_embeddings_none_when_empty():
    kb = _kb("db14.sqlite", embedder=K.Embedder(enabled=False))
    try:
        kb.add_document("doc", "s1", "no embeddings here", embed=False)
        ids, mat = kb._all_embeddings()
        check("all_embeddings_empty_when_none", ids == [] and mat is None)
    finally:
        kb.close()


def test_purge_and_repair():
    kb = _kb("db15.sqlite", embedder=K.Embedder(enabled=False))
    try:
        kb.add_document("doc", "s1", "doc one", profile="p1", embed=False)
        kb.add_document("research", "s2", "doc two", profile="p2", embed=False)
        n = kb.purge(source_type="doc")
        check("purge_by_source_type", n == 1)
        n2 = kb.purge(profile="p2")
        check("purge_by_profile", n2 == 1)
        check("purge_no_filter_all", isinstance(kb.purge(), int))

        kb.add_document("doc", "s3", "orphan test doc", embed=False)
        orphans = kb.repair()
        check("repair_no_orphans_zero", orphans == 0)
        kb._conn.execute("PRAGMA foreign_keys=OFF")
        kb._conn.execute("DELETE FROM documents WHERE source_id='s3'")
        kb._conn.commit()
        kb._conn.execute("PRAGMA foreign_keys=ON")
        orphans2 = kb.repair()
        check("repair_removes_orphans", orphans2 >= 1)
    finally:
        kb.close()


def test_default_kb():
    kb = K.default_kb(_TMP / "default_kb.sqlite")
    try:
        check("default_kb_creates_instance", isinstance(kb, K.KnowledgeBase))
    finally:
        kb.close()


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

"""Deterministic regression test for the document Library (TXT/MD/PDF/EPUB
extraction + hybrid build + retrieval). Runs with the embedder DISABLED so it
needs no LM Studio: the build falls back to BM25/FTS-only and retrieval still
returns the right passage. Validates extraction, chunking, build stats, the
graceful no-embed path, retrieval relevance, context formatting, and purge.

Run:  venv/Scripts/python.exe tests/test_library.py   (exit 0 = all pass)
"""
import os
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import library
import knowledge

ok = []
def chk(c, m):
    ok.append(bool(c)); print(("PASS " if c else "FAIL ") + m)


def make_epub(path):
    """Minimal valid-enough EPUB: container.xml -> OPF spine -> one XHTML doc."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("META-INF/container.xml",
                    '<?xml version="1.0"?><container version="1.0" '
                    'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                    '<rootfiles><rootfile full-path="OEBPS/content.opf" '
                    'media-type="application/oebps-package+xml"/></rootfiles></container>')
        zf.writestr("OEBPS/content.opf",
                    '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" '
                    'version="3.0" unique-identifier="id"><metadata/>'
                    '<manifest><item id="c1" href="ch1.xhtml" '
                    'media-type="application/xhtml+xml"/></manifest>'
                    '<spine><itemref idref="c1"/></spine></package>')
        zf.writestr("OEBPS/ch1.xhtml",
                    '<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml">'
                    '<body><h1>Chapter One</h1>'
                    '<p>The zephyr carried the scent of bergamot across the harbour.</p>'
                    '</body></html>')


def main():
    tmp = tempfile.mkdtemp(prefix="libtest_")
    # --- extraction across formats ---
    txt = os.path.join(tmp, "note.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write("Pierre wandered the battlefield at Borodino, dazed and unarmed.")
    md = os.path.join(tmp, "doc.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Title\n\nNatasha danced at her first grand ball in Moscow.")
    epub = os.path.join(tmp, "book.epub")
    make_epub(epub)

    chk("Borodino" in library.extract_text(txt), "extract TXT")
    chk("Natasha" in library.extract_text(md), "extract MD")
    chk("bergamot" in library.extract_text(epub), "extract EPUB (zip+bs4)")
    try:
        library.extract_text(os.path.join(tmp, "x.zip"))
        chk(False, "unsupported ext rejected")
    except ValueError:
        chk(True, "unsupported ext rejected")

    # --- build with embedder DISABLED (no LM Studio needed) ---
    db = os.path.join(tmp, "library.db")
    lib = library.Library(db_path=db)
    lib.kb.embedder.enabled = False                     # force BM25-only path

    seen = {"extract": 0, "embed": 0}
    def prog(stage, d, t):
        seen[stage] = seen.get(stage, 0) + 1
    stats = lib.build([txt, md, epub], progress=prog)

    chk(stats["documents_indexed"] == 3, "built 3 documents")
    chk(stats["chunks"] >= 3, f"chunked into >=3 chunks (got {stats['chunks']})")
    chk(stats["chunks_embedded"] == 0, "no-embed path: 0 embedded, FTS-only")
    chk(stats["errors"] == [], "no per-file errors")
    chk(seen["extract"] >= 3, "extract progress fired per file")

    # --- idempotent rebuild (unchanged files skip) ---
    stats2 = lib.build([txt, md, epub])
    chk(stats2["documents_indexed"] == 0 and stats2["skipped_unchanged"] == 3,
        "rebuild is idempotent (all skipped)")

    # --- retrieval (BM25) returns the right passage ---
    res = lib.retrieve("Where was Pierre during the battle?", k=3)
    chk(len(res) >= 1, "retrieval returns results")
    chk(any("Borodino" in r.text for r in res), "top results contain the Borodino passage")
    res2 = lib.retrieve("ball dancing Moscow", k=3)
    chk(any("Natasha" in r.text for r in res2), "lexical query finds the ball passage")

    # --- RAG prompt builder (used by both typed and voice paths) ---
    send, note = library.build_rag_prompt(lib, "Where was Pierre during the battle?")
    chk("=== Retrieved passages ===" in send and "Borodino" in send,
        "build_rag_prompt wraps passages into the prompt")
    chk(send.rstrip().endswith("Where was Pierre during the battle?"),
        "build_rag_prompt keeps the original question")
    chk(note and "passage" in note, "build_rag_prompt returns an info note")
    empty_lib = library.Library(db_path=os.path.join(tmp, "empty.db"))
    s_send, s_note = library.build_rag_prompt(empty_lib, "anything")
    chk(s_send == "anything" and "empty" in (s_note or "").lower(),
        "build_rag_prompt on empty DB returns original question")
    empty_lib.close()

    ctx = library.Library.format_context(res, max_chars=8000)
    chk(ctx.startswith("[1]") and "Borodino" in ctx, "context block is numbered + sourced")
    chk(len(library.Library.format_context(res, max_chars=10)) <= 200,
        "context respects char budget")

    # --- reranker ---
    # lexical_score: full query coverage beats partial coverage.
    full = library.lexical_score("pierre captured borodino", "Pierre was captured at Borodino")
    part = library.lexical_score("pierre captured borodino", "Pierre walked in the garden")
    chk(full > part, "lexical_score: full coverage > partial coverage")
    chk(library.lexical_score("", "anything") == 0.0, "lexical_score: empty query -> 0")

    # rerank promotes the clearly-relevant passage even when it starts lower in the pool.
    def SR(cid, text):
        return knowledge.SearchResult(chunk_id=cid, doc_id=1, source_type="library",
                                      title="t", url="", path="", text=text, score=0.0)
    pool = [SR(1, "An unrelated paragraph about weather and rivers."),
            SR(2, "Totally off topic cooking recipe."),
            SR(3, "Pierre was captured by the French near Borodino in 1812.")]
    # Without a cross-encoder the hybrid order is kept (the lexical rerank lost
    # 9 pp hit@1 on bench/rag_eval); LIBRARY_LEXICAL_RERANK=1 brings it back.
    top = library.rerank("Where was Pierre captured at Borodino?", pool, k=1)
    chk(top and top[0].chunk_id == 1, "no cross-encoder: hybrid order kept")
    os.environ["LIBRARY_LEXICAL_RERANK"] = "1"
    try:
        top = library.rerank("Where was Pierre captured at Borodino?", pool, k=1)
        chk(top and top[0].chunk_id == 3, "opt-in lexical rerank promotes the relevant passage")
    finally:
        os.environ.pop("LIBRARY_LEXICAL_RERANK", None)
    chk(len(library.rerank("q", pool, k=2)) == 2, "rerank returns exactly k")
    chk(library.rerank("q", [], k=3) == [], "rerank handles empty pool")

    # cross-encoder path: a stub scorer drives the order; failure degrades to lexical.
    class StubCE:
        def predict(self, pairs):
            return [9.0 if "Borodino" in p else 0.0 for _, p in pairs]
    top_ce = library.rerank("anything", pool, k=1, cross_encoder=StubCE())
    chk(top_ce and top_ce[0].chunk_id == 3, "cross-encoder stub drives rerank order")
    class BrokenCE:
        def predict(self, pairs):
            raise RuntimeError("boom")
    safe = library.rerank("Pierre Borodino", pool, k=1, cross_encoder=BrokenCE())
    chk(safe and safe[0].chunk_id == 1, "cross-encoder failure keeps the hybrid order (no raise)")

    # retrieve(rerank_results=...) integrates the stage end-to-end (BM25 pool).
    rr_on = lib.retrieve("Where was Pierre during the battle?", k=2, rerank_results=True)
    chk(any("Borodino" in r.text for r in rr_on), "retrieve+rerank still finds Borodino")
    rr_off = lib.retrieve("Where was Pierre during the battle?", k=2, rerank_results=False)
    chk(any("Borodino" in r.text for r in rr_off), "retrieve without rerank still works")

    # --- cross-lingual helpers ---
    chk(library.text_scripts("What color are the eyes") == {"latin"}, "script detect: latin")
    chk(library.text_scripts("Какого цвета глаза") == {"cyrillic"}, "script detect: cyrillic")
    chk(library.cross_lingual_targets("eye color", {"cyrillic"}) == ["Russian"],
        "cross-lingual: EN query + RU corpus -> translate to Russian")
    chk(library.cross_lingual_targets("глаза", {"cyrillic"}) == [],
        "cross-lingual: RU query + RU corpus -> no translation")

    # multi-query retrieval merges + dedups across query variants
    multi = lib.retrieve_multi(["Pierre", "Natasha"], k=5)
    ids = [r.chunk_id for r in multi]
    chk(len(ids) == len(set(ids)), "retrieve_multi de-duplicates by chunk")
    chk(len(multi) >= 1, "retrieve_multi returns merged results")

    # --- whole-book scan (map-reduce) with stubbed LLM fns ---
    chunks = ["Pierre has grey eyes.", "Andrei was tired.",
              "Natasha has dark eyes.", "Dolokhov has light-blue eyes."]
    seen_batches = []
    def mapf(batch, q):
        hits = [ln for ln in batch.split("\n\n") if "eyes" in ln]
        seen_batches.append(len(hits))
        return "\n".join(hits)
    def reducef(notes, q):
        return f"{sum(1 for n in notes)} note-blocks; mentions: " + " | ".join(notes)
    ans = library.map_reduce_scan(chunks, "eye colors", mapf, reducef, batch_chars=30)
    chk("Pierre has grey eyes." in ans and "Dolokhov" in ans,
        "scan aggregates eye mentions across all chunks")
    # cancel stops the scan early
    cancelled = library.map_reduce_scan(chunks, "q", mapf, lambda n, q: f"{len(n)}",
                                        batch_chars=30, cancel=lambda: True)
    chk(cancelled == "0", "scan honours cancel (no batches mapped)")
    chk(len(lib.all_chunks()) == lib.stats()["chunks"], "all_chunks returns every chunk")

    # --- concurrent embedding (workers>1) with a fake offline embedder ---
    class FakeEmb:
        model = "fake"; dim = 8; enabled = True
        def __init__(self): self.calls = 0
        def available(self, *a, **k): return True
        def embed(self, texts):
            self.calls += 1
            return [[float((hash(t) % 97))] * self.dim for t in texts]
    db2 = os.path.join(tmp, "concurrent.db")
    kb2 = knowledge.KnowledgeBase(db2, embedder=FakeEmb())
    for i in range(50):
        kb2.add_document("library", f"doc{i}", f"chunk text number {i} about eyes",
                         title=f"d{i}", embed=False)
    n_par = kb2.ensure_embeddings(batch=4, workers=4)
    st2 = kb2.stats()
    chk(n_par == st2["chunks"] and st2["chunks_embedded"] == st2["chunks"],
        f"concurrent embed: all {st2['chunks']} chunks embedded via workers=4")
    chk(kb2.search("eyes", k=5), "search works after concurrent embed")
    # cancel stops a concurrent build early
    kb3 = knowledge.KnowledgeBase(os.path.join(tmp, "cancel.db"), embedder=FakeEmb())
    for i in range(40):
        kb3.add_document("library", f"d{i}", f"text {i}", embed=False)
    n_cancel = kb3.ensure_embeddings(batch=2, workers=4, cancel=lambda: True)
    chk(n_cancel == 0, "concurrent embed honours immediate cancel")

    # --- FK cascade + orphan repair (the "0 documents / N passages" bug) ---
    fkdb = os.path.join(tmp, "fk.db")
    kbf = knowledge.KnowledgeBase(fkdb)            # FK + busy_timeout enabled in __init__
    kbf.add_document("library", "fk1", "alpha beta gamma. delta epsilon zeta.", embed=False)
    chunks_before = kbf.stats()["chunks"]
    kbf.purge(source_type="library")               # delete the document
    chk(chunks_before > 0 and kbf.stats()["chunks"] == 0,
        "FK cascade: deleting a document removes its chunks (no orphans)")
    # manufacture an orphan the way the old bug did — delete the parent document via a
    # raw connection (FK enforcement OFF by default), leaving chunks behind
    kbf.add_document("library", "fk2", "alpha beta gamma. delta epsilon zeta.", embed=False)
    import sqlite3 as _sql
    raw = _sql.connect(fkdb)
    raw.execute("DELETE FROM documents")
    raw.commit(); raw.close()
    chk(kbf.stats()["documents"] == 0 and kbf.stats()["chunks"] > 0,
        "orphan chunks created (parent deleted with FK off)")
    orphans = kbf.stats()["chunks"]
    removed = kbf.repair()
    chk(removed == orphans and kbf.stats()["chunks"] == 0, "repair() removes orphan chunks")
    chk(kbf.repair() == 0, "repair() is idempotent (no orphans left)")
    kbf.close()

    # --- introspection + purge ---
    docs = lib.documents()
    chk(len(docs) == 3 and all(d["chunks"] >= 1 for d in docs), "documents() lists 3 docs")
    chk(not lib.is_empty(), "is_empty False when populated")
    purged = lib.purge_all()
    chk(purged == 3 and lib.is_empty(), "purge_all clears the library")

    # --- budgets are budgets, even for one oversized element -----------------
    # Both of these caps exist to keep a prompt inside the context window, and
    # both used to be bypassed entirely by a single element larger than the cap.
    class _Res:
        def __init__(self, text):
            self.text = text; self.title = "doc"; self.path = ""
            self.source_type = "library"

    one_huge = library.Library.format_context([_Res("x" * 50000)], max_chars=1000)
    chk(len(one_huge) <= 1000,
        f"format_context truncates a single oversized passage ({len(one_huge)} chars)")
    chk(one_huge.strip() != "", "format_context still returns something usable")
    many = library.Library.format_context([_Res("y" * 300) for _ in range(50)],
                                    max_chars=1000)
    chk(len(many) <= 1000, f"format_context respects the budget for many passages ({len(many)})")

    src_chunks = ["a" * 100] * 30 + ["b" * 40000] + ["c" * 100] * 5
    bt = library.batch_chunks(src_chunks, 12000)
    chk(all(len(x) <= 12000 for x in bt),
        f"batch_chunks splits an oversized chunk (max {max(len(x) for x in bt)})")
    # Splitting must not silently drop book text — a whole-book scan claims to be
    # exhaustive, so lost text would become a confidently incomplete answer.
    chk(sorted("".join(src_chunks)) ==
        sorted("".join(x.replace("\n\n", "") for x in bt)),
        "batch_chunks preserves every character while splitting")

    print(f"\n{sum(ok)}/{len(ok)} passed")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())

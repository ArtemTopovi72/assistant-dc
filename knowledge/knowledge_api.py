"""The knowledge/RAG service boundary.

This module defines — and ONLY defines — the contract between the three
consumers of the document/knowledge layer and its storage substrate. It holds
no state, opens no database, and imports nothing from `library`/`knowledge`, so
it is safe to import from anywhere (including a standalone service process that
has no PyQt5 and no LM Studio).

WHY A BOUNDARY HERE
-------------------
`library.py` (ingest + retrieval policy) and `knowledge.py` (SQLite + FTS5
hybrid index with RRF fusion) already form one bounded context over one storage
substrate, consumed by three independent front ends:

  * the desktop GUI's Database tab   — gui_database_tab.py, gui_workers.py
  * the Telegram bot                 — tg_library.py, tg_tasks.py
  * the deep-research pipeline       — knowledge.import_research_cache()

Each front end opens its own SQLite connection on its own thread (a SQLite
connection may only be used by its creating thread), which is exactly the
coupling a service boundary removes.

THE CALLS THAT ACTUALLY CROSS THE BOUNDARY
------------------------------------------
Derived by grepping the real call sites, not from what looked tidy:

  ingest / build
    gui_workers.LibraryBuildWorker.run   lib.build(paths, progress=, cancel=)
    tg_library._index_document           lib.build([staged], source_ids=, titles=)

  search
    library.build_rag_prompt             lib.retrieve(q, k=) / lib.retrieve_multi(qs, k=)
      called from gui_workers.RequestWorker (GUI) and tg_tasks (bot)

  stats / introspection
    gui_database_tab                     lib.stats(), lib.documents(), lib.is_empty()
    gui_supplement suites                lib.stats(), lib.documents(), lib.purge_all()
    tg_library._library_stats            lib.documents()
    tg_library._send_library_list        lib.documents()
    gui_workers.RequestWorker            lib.corpus_scripts()   (cross-lingual routing)
    gui_workers.ScanWorker               lib.all_chunks(paths)  (whole-book scan)

  delete
    gui_database_tab / tg_library._clear_library    lib.purge_all()

  lifecycle
    every consumer                       lib.close()

WHAT IS *NOT* ON THE BOUNDARY (deliberately client-side / internal)
-------------------------------------------------------------------
  library.build_rag_prompt(lib, question)
      Pure prompt assembly over whatever `lib` returned. It calls only boundary
      methods (is_empty / retrieve / retrieve_multi) and does no I/O, so it stays
      a client-side helper and works unchanged against any backend.

  library.map_reduce_scan(chunks, question, map_fn, reduce_fn)
      Takes two LLM CALLBACKS. Callbacks cannot cross a process boundary, and
      the LLM lives with the caller, not with the index. Stays client-side; it
      is fed by the boundary call all_chunks().

  library.cross_lingual_targets / text_scripts / rerank / lexical_score /
  load_cross_encoder / extract_text / batch_chunks
      Pure functions over text. No storage access. Client-side.

  knowledge.KnowledgeBase internals (Embedder, RRF fusion, FTS5 schema,
  ensure_embeddings, repair, add_document, purge, import_research_cache)
      Server-side implementation detail. No consumer outside library.py and the
      maintenance script reindex_embeddings.py touches them.

  research_cache.ExtractionCache
      Listed as part of this layer, but it is NOT: it is a file-per-URL JSON
      cache on local disk, opened and closed inside deep_research.py's crawler
      loop, and it does not share the SQLite substrate at all. Its only contact
      with the knowledge layer is that KnowledgeBase.import_research_cache()
      reads such a directory BY PATH. Putting it behind a network hop would add
      one round-trip per URL to a component whose whole purpose is to avoid
      per-URL latency. It is therefore left in-process and out of this contract.
      (See "NOT MIGRATED" in the branch notes.)

RETURN SHAPES
-------------
Everything crossing the boundary is JSON-serialisable so the same contract can
be served over HTTP:
  * search hits           -> list of dicts, keys in SEARCH_HIT_FIELDS below
  * ingest / stats        -> plain dicts of str/int/float/list
  * documents             -> list of {title, path, chunks, embedded}
  * corpus_scripts        -> set of script names (serialised as a list)
  * all_chunks            -> list of str

`knowledge.SearchResult` is a dataclass of exactly these fields, so the
in-process backend can return the dataclass itself (callers use attribute
access: r.title, r.path, r.text, r.score, r.chunk_id) and the HTTP backend
reconstructs the same dataclass from JSON. Attribute access is the contract;
do not make callers depend on it being a dict.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, runtime_checkable

# Marker grepped by the contract suites instead of a function name, so that
# moving any single method does not break the anchor.
KNOWLEDGE_API_MARKER = "knowledge-service-boundary-v1"

#: Fields of a search hit. Mirrors knowledge.SearchResult's dataclass fields.
SEARCH_HIT_FIELDS = (
    "chunk_id", "doc_id", "source_type", "title", "url", "path", "text",
    "score", "bm25_rank", "cosine_rank", "cosine", "why",
)

#: Fields of one row of documents().
DOCUMENT_FIELDS = ("title", "path", "chunks", "embedded")

#: Operation names the HTTP backend exposes. One route per boundary method.
OPERATIONS = (
    "ingest", "search", "search_multi", "documents", "stats",
    "corpus_scripts", "all_chunks", "purge_all", "is_empty",
)


@runtime_checkable
class KnowledgeClient(Protocol):
    """What a knowledge backend must provide, in-process or over HTTP.

    Implementations are NOT thread-safe: the in-process backend wraps a SQLite
    connection owned by its creating thread. Open one per thread per operation,
    exactly as the existing consumers already do, and close it in a finally.
    """

    # -- ingest ------------------------------------------------------------ #
    def ingest(self, paths: Sequence[str], *,
               progress: Optional[Callable[[str, int, int], None]] = None,
               cancel: Optional[Callable[[], bool]] = None,
               source_ids: Optional[Dict[str, str]] = None,
               titles: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Index every path. Returns the build stats dict:
        {files, documents_indexed, skipped_unchanged, chunks_embedded, errors, ...stats()}

        progress(stage, done, total) with stage in {'extract','embed'};
        cancel() -> True stops early. Both are LOCAL callbacks: an out-of-process
        backend polls them from the client side, it does not call back over the
        wire.
        """

    # -- search ------------------------------------------------------------ #
    def search(self, query: str, k: Optional[int] = None, *,
               rerank_results: bool = True,
               pool: Optional[int] = None) -> List[Any]:
        """Hybrid BM25+embedding search, reranked. Returns SearchResult-shaped
        objects with attribute access (see SEARCH_HIT_FIELDS)."""

    def search_multi(self, queries: Sequence[str], k: Optional[int] = None, *,
                     pool: Optional[int] = None) -> List[Any]:
        """Search several query variants (question + translations), merge the
        pools deduped by chunk_id, rerank against queries[0]."""

    # -- stats / introspection --------------------------------------------- #
    def documents(self) -> List[Dict[str, Any]]:
        """One row per indexed document: DOCUMENT_FIELDS."""

    def stats(self) -> Dict[str, Any]:
        """Index-wide counters: documents, chunks, chunks_embedded, embed_coverage, ..."""

    def is_empty(self) -> bool:
        """True when this library holds no documents."""

    def corpus_scripts(self, sample: int = 50) -> set:
        """Writing systems present in the corpus, for cross-lingual query routing."""

    def all_chunks(self, paths: Optional[Sequence[str]] = None) -> List[str]:
        """Every chunk's text in document order. Feeds the whole-book scan."""

    # -- delete ------------------------------------------------------------ #
    def purge_all(self) -> int:
        """Remove every document this library owns. Returns rows deleted."""

    # -- lifecycle --------------------------------------------------------- #
    def close(self) -> None:
        """Release the connection / session. Must be idempotent and never raise."""


class KnowledgeBackendError(RuntimeError):
    """Raised by a remote backend when the service reports a failure.

    In-process failures propagate as whatever library/knowledge raised, so that
    existing except-clauses in the consumers keep behaving identically.
    """

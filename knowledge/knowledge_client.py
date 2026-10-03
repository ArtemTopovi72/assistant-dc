"""Client abstraction for the knowledge/RAG layer (see knowledge_api.py).

Every consumer of the document layer — the desktop GUI's Database tab, the
Telegram bot, and the whole-book scan — goes through `open_library()` here
instead of constructing `library.Library` itself. The backend is chosen by the
env var KNOWLEDGE_BACKEND and defaults to the in-process one, so nothing about
the running product changes unless it is explicitly switched.

    KNOWLEDGE_BACKEND=inprocess   (default) -> InProcessKnowledgeClient
    KNOWLEDGE_BACKEND=http                  -> HttpKnowledgeClient
    KNOWLEDGE_SERVICE_URL=http://127.0.0.1:8790   (http backend only)

WHY THE LAZY `import library` INSIDE EVERY CONSTRUCTOR
------------------------------------------------------
It is not cosmetic and it is not an import-cycle dodge. The GUI and worker
suites stub the document layer by swapping sys.modules['library'] for a module
that defines only the handful of names the real code path touches. Resolving
`library` at CALL time keeps that seam alive now that the caller has moved into
this module, and calling exactly the same entry points with exactly the same
argument shapes (positional db_path; build() without source_ids/titles when
they are None; corpus_scripts() with no argument; all_chunks(paths) positional)
keeps those stubs valid. Do not "tidy" these call shapes.

The in-process client also keeps the legacy method names `build`, `retrieve`
and `retrieve_multi` as aliases of `ingest`, `search` and `search_multi`. That
is deliberate: `library.build_rag_prompt(lib, question)` duck-types on
is_empty/retrieve/retrieve_multi, so a client can be handed straight to it and
works against either backend without a second prompt-assembly implementation.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Sequence

from knowledge_api import KnowledgeClient, KnowledgeBackendError  # noqa: F401  (re-export)

__all__ = ["open_library", "InProcessKnowledgeClient", "backend_name",
           "KnowledgeClient", "KnowledgeBackendError", "map_reduce_scan",
           "cross_lingual_targets", "build_rag_prompt"]

ENV_BACKEND = "KNOWLEDGE_BACKEND"
ENV_URL = "KNOWLEDGE_SERVICE_URL"
DEFAULT_URL = "http://127.0.0.1:8790"


def backend_name() -> str:
    """Which backend open_library() would pick right now. Read at call time so a
    test (or the differential suite) can flip the env var between calls."""
    v = (os.environ.get(ENV_BACKEND) or "inprocess").strip().lower()
    return "http" if v in ("http", "https", "remote", "service") else "inprocess"


def open_library(db_path: Optional[str] = None, *, default: bool = False) -> KnowledgeClient:
    """Open a knowledge client for one thread and one operation.

    db_path: the index file. None means "this process's default library".
    default: use library.default_library() rather than library.Library() for the
        None case — the GUI Database tab's reader connection, which several
        suites patch by name.

    The returned object is NOT thread-safe (SQLite connections belong to their
    creating thread). Close it in a finally, exactly as the callers already do.
    """
    if backend_name() == "http":
        from knowledge_service import HttpKnowledgeClient   # local import: stdlib-only module
        return HttpKnowledgeClient(os.environ.get(ENV_URL) or DEFAULT_URL, db_path=db_path)
    return InProcessKnowledgeClient(db_path, default=default)


# --------------------------------------------------------------------------- #
# In-process backend
# --------------------------------------------------------------------------- #
class InProcessKnowledgeClient:
    """The default backend: a thin, behaviour-preserving delegator to library.Library.

    It adds no policy of its own. Every method forwards one-to-one, so switching
    a consumer from `library.Library(...)` to `open_library(...)` cannot change
    what that consumer observes — including which exceptions it sees, which is
    why nothing here is wrapped in try/except.
    """

    backend = "inprocess"

    def __init__(self, db_path: Optional[str] = None, *, default: bool = False):
        import library                       # call-time resolution: see module docstring
        if db_path is None:
            self._lib = library.default_library() if default else library.Library()
        else:
            self._lib = library.Library(db_path)

    # -- ingest ------------------------------------------------------------ #
    def ingest(self, paths: Sequence[str], *,
               progress: Optional[Callable[[str, int, int], None]] = None,
               cancel: Optional[Callable[[], bool]] = None,
               source_ids: Optional[Dict[str, str]] = None,
               titles: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        # Only forward the optional kwargs that were actually supplied: the
        # worker suites' fake Library.build() accepts progress/cancel only.
        kw: Dict[str, Any] = {}
        if progress is not None:
            kw["progress"] = progress
        if cancel is not None:
            kw["cancel"] = cancel
        if source_ids is not None:
            kw["source_ids"] = source_ids
        if titles is not None:
            kw["titles"] = titles
        return self._lib.build(list(paths), **kw)

    # -- search ------------------------------------------------------------ #
    def search(self, query: str, k: Optional[int] = None, *,
               rerank_results: bool = True, pool: Optional[int] = None) -> List[Any]:
        return self._lib.retrieve(query, k=k, rerank_results=rerank_results, pool=pool)

    def search_multi(self, queries: Sequence[str], k: Optional[int] = None, *,
                     pool: Optional[int] = None) -> List[Any]:
        return self._lib.retrieve_multi(list(queries), k=k, pool=pool)

    # -- stats / introspection --------------------------------------------- #
    def documents(self) -> List[Dict[str, Any]]:
        return self._lib.documents()

    def stats(self) -> Dict[str, Any]:
        return self._lib.stats()

    def is_empty(self) -> bool:
        return self._lib.is_empty()

    def corpus_scripts(self, sample: int = 50) -> set:
        # No-argument call: the RequestWorker suite's fake defines corpus_scripts(self).
        return self._lib.corpus_scripts() if sample == 50 else self._lib.corpus_scripts(sample)

    def all_chunks(self, paths: Optional[Sequence[str]] = None) -> List[str]:
        return self._lib.all_chunks(list(paths) if paths else paths)

    # -- delete ------------------------------------------------------------ #
    def purge_all(self) -> int:
        return self._lib.purge_all()

    # -- lifecycle --------------------------------------------------------- #
    def close(self) -> None:
        self._lib.close()

    # -- legacy aliases ----------------------------------------------------- #
    # library.build_rag_prompt() duck-types on these; keeping them means one
    # prompt-assembly implementation serves both backends.
    def build(self, paths, **kw):
        return self.ingest(paths, **kw)

    def retrieve(self, query, k=None, **kw):
        return self.search(query, k=k, **kw)

    def retrieve_multi(self, queries, k=None, **kw):
        return self.search_multi(queries, k=k, **kw)


# --------------------------------------------------------------------------- #
# Client-side helpers (documented in knowledge_api as NOT crossing the boundary)
# --------------------------------------------------------------------------- #
def build_rag_prompt(client, question: str, k: Optional[int] = None,
                     extra_queries: Optional[List[str]] = None):
    """Retrieve for `question` through `client` and assemble the RAG prompt.

    Pure assembly over boundary results — no storage access of its own — so it
    is identical for every backend. Delegates to library.build_rag_prompt at
    call time so the module stub used by the worker suites still applies.
    """
    import library
    return library.build_rag_prompt(client, question, k=k, extra_queries=extra_queries)


def cross_lingual_targets(client, text: str) -> List[str]:
    """Languages to also search in, given the scripts present in this corpus."""
    import library
    return library.cross_lingual_targets(text, client.corpus_scripts())


def map_reduce_scan(chunks, question, map_fn, reduce_fn, *, progress=None, cancel=None):
    """Whole-book scan. Client-side by necessity: map_fn/reduce_fn are LLM
    callbacks and the LLM lives with the caller, not with the index."""
    import library
    return library.map_reduce_scan(chunks, question, map_fn, reduce_fn,
                                   progress=progress, cancel=cancel)

"""Out-of-process backend for the knowledge layer: a small HTTP service plus a
client that implements the same contract (knowledge_api.KnowledgeClient).

Nothing selects it unless you ask for it:

    set KNOWLEDGE_BACKEND=http
    set KNOWLEDGE_SERVICE_URL=http://127.0.0.1:8790
    venv\\Scripts\\python.exe knowledge_service.py --port 8790

The default remains the in-process backend, so an unconfigured install is
byte-for-byte the product it was before this branch.

DEPENDENCIES
------------
Server: stdlib only (http.server + json). FastAPI/uvicorn ARE installed in this
venv, but stdlib is enough for nine JSON routes and keeps the service startable
in a bare interpreter — which matters, because the whole point of extracting it
is that it need not carry PyQt5, torch, or the rest of the monolith's baggage.
Client: `requests`, already a hard dependency of knowledge.py.

WIRE PROTOCOL
-------------
    GET  /health              -> {"ok": true, "backend": "http", "ops": [...]}
    POST /op/<operation>      -> {"ok": true, "result": <json>}
                              or {"ok": false, "error": "<Type: message>"}
    POST /shutdown            -> {"ok": true}   (so a test can stop it cleanly)

Request body: {"db_path": <str|null>, "default": <bool>, "args": {...}}.
`operation` is one of knowledge_api.OPERATIONS.

CONNECTION MODEL — read this before "optimising" it
---------------------------------------------------
A SQLite connection may only be used by the thread that created it, and
ThreadingHTTPServer hands each request to a fresh thread. So the service opens
a Library, serves the one operation, and closes it, inside the request thread.
That is a real cost on ingest, and it is the correct default: a cached
connection would be used from the wrong thread on the very next request and
fail intermittently, which is far worse than being slower. If this ever needs
to be fast, the fix is a single dedicated DB thread fed by a queue, not a cache.

WHAT THE HTTP BACKEND CANNOT DO, AND SAYS SO
--------------------------------------------
* progress(stage, done, total) during ingest. Callbacks do not cross a process
  boundary. The client emits one 'extract' tick before the call and one 'embed'
  tick with the final counts after it, so a progress bar still moves, but the
  intermediate granularity is gone.
* cancel() mid-ingest. The client checks cancel() before dispatching and
  returns an empty stats dict if it is already set; once the request is in
  flight it runs to completion.
Both are documented rather than faked, because a fake would silently change
what a cancelling user observes.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from http.server import ThreadingHTTPServer
from service_http import JsonHandler
from typing import Any, Callable, Dict, List, Optional, Sequence

from knowledge_api import OPERATIONS, KnowledgeBackendError

logger = logging.getLogger("assistant.knowledge.service")

DEFAULT_PORT = 8790
#: Test/ops override applied to config.EMBED_BASE at service start, so a caller
#: can pin the embedding endpoint of the service process (the differential
#: suite points it at a dead port to force the deterministic FTS-only path).
ENV_EMBED_BASE = "KNOWLEDGE_SERVICE_EMBED_BASE"


# --------------------------------------------------------------------------- #
# Server side
# --------------------------------------------------------------------------- #
def _open_local(db_path: Optional[str], default: bool = False):
    """Open the real in-process library inside the current request thread."""
    import library
    if db_path is None:
        return library.default_library() if default else library.Library()
    return library.Library(db_path)


def _dispatch(op: str, db_path: Optional[str], default: bool,
              args: Dict[str, Any]) -> Any:
    """Run one boundary operation and return a JSON-serialisable result."""
    if op not in OPERATIONS:
        raise KnowledgeBackendError(f"unknown operation: {op}")
    lib = _open_local(db_path, default)
    try:
        if op == "ingest":
            kw: Dict[str, Any] = {}
            if args.get("source_ids") is not None:
                kw["source_ids"] = args["source_ids"]
            if args.get("titles") is not None:
                kw["titles"] = args["titles"]
            return lib.build(list(args.get("paths") or []), **kw)
        if op == "search":
            hits = lib.retrieve(args["query"], k=args.get("k"),
                                rerank_results=bool(args.get("rerank_results", True)),
                                pool=args.get("pool"))
            return [_hit_to_json(h) for h in hits]
        if op == "search_multi":
            hits = lib.retrieve_multi(list(args.get("queries") or []),
                                      k=args.get("k"), pool=args.get("pool"))
            return [_hit_to_json(h) for h in hits]
        if op == "documents":
            return lib.documents()
        if op == "stats":
            return lib.stats()
        if op == "is_empty":
            return bool(lib.is_empty())
        if op == "corpus_scripts":
            return sorted(lib.corpus_scripts(int(args.get("sample", 50))))
        if op == "all_chunks":
            return lib.all_chunks(args.get("paths"))
        if op == "purge_all":
            return int(lib.purge_all())
        raise KnowledgeBackendError(f"unhandled operation: {op}")   # pragma: no cover
    finally:
        try:
            lib.close()
        except Exception:
            logger.debug("close failed after %s", op, exc_info=True)


def _hit_to_json(h: Any) -> Dict[str, Any]:
    """Full field set, not SearchResult.to_public() — to_public truncates text to
    a 280-char snippet and rounds the score, which would make the two backends
    disagree on exactly the payload the RAG prompt is built from."""
    from knowledge_api import SEARCH_HIT_FIELDS
    return {f: getattr(h, f, None) for f in SEARCH_HIT_FIELDS}


class _Handler(JsonHandler):
    server_version = "KnowledgeService/1"

    def do_GET(self):                       # noqa: N802 (stdlib naming)
        if self.path.rstrip("/") in ("/health", ""):
            self._send(200, {"ok": True, "backend": "http", "ops": list(OPERATIONS)})
        else:
            self._send(404, {"ok": False, "error": f"no such path: {self.path}"})

    def do_POST(self):                      # noqa: N802
        path = self.path.rstrip("/")
        if path == "/shutdown":
            self._send(200, {"ok": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        if not path.startswith("/op/"):
            self._send(404, {"ok": False, "error": f"no such path: {self.path}"})
            return
        op = path[len("/op/"):]
        try:
            req = self._read_body()
        except Exception as exc:
            self._send(400, {"ok": False, "error": f"bad request: {exc}"})
            return
        try:
            result = _dispatch(op, req.get("db_path"), bool(req.get("default")),
                               req.get("args") or {})
        except Exception as exc:
            logger.exception("knowledge service op %s failed", op)
            self._send(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            return
        self._send(200, {"ok": True, "result": result})


def make_server(port: int = DEFAULT_PORT, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """Bind and return an unstarted server. Pass port=0 to let the OS pick one
    (read it back from srv.server_address[1])."""
    return ThreadingHTTPServer((host, port), _Handler)


def serve(port: int = DEFAULT_PORT, host: str = "127.0.0.1") -> None:
    srv = make_server(port, host)
    logger.info("knowledge service listening on http://%s:%d", host, srv.server_address[1])
    print(f"knowledge service listening on http://{host}:{srv.server_address[1]}", flush=True)
    try:
        srv.serve_forever()
    finally:
        srv.server_close()


# --------------------------------------------------------------------------- #
# Client side
# --------------------------------------------------------------------------- #
class HttpKnowledgeClient:
    """knowledge_api.KnowledgeClient over HTTP. Same methods, same return
    shapes, same attribute-access search hits."""

    backend = "http"

    def __init__(self, url: str, db_path: Optional[str] = None,
                 *, default: bool = False, timeout: float = 600.0):
        self.url = url.rstrip("/")
        self.db_path = None if db_path is None else str(db_path)
        self.default = bool(default)
        self.timeout = timeout
        self._session = None

    # -- transport --------------------------------------------------------- #
    def _post(self, op: str, **args) -> Any:
        import requests
        if self._session is None:
            self._session = requests.Session()
        payload = {"db_path": self.db_path, "default": self.default, "args": args}
        try:
            r = self._session.post(f"{self.url}/op/{op}", json=payload, timeout=self.timeout)
        except Exception as exc:
            raise KnowledgeBackendError(f"knowledge service unreachable at {self.url}: {exc}") from exc
        try:
            body = r.json()
        except Exception as exc:
            raise KnowledgeBackendError(f"knowledge service returned non-JSON ({r.status_code})") from exc
        if not body.get("ok"):
            raise KnowledgeBackendError(str(body.get("error") or f"HTTP {r.status_code}"))
        return body.get("result")

    def health(self) -> Dict[str, Any]:
        import requests
        r = requests.get(f"{self.url}/health", timeout=10)
        return r.json()

    # -- ingest ------------------------------------------------------------ #
    def ingest(self, paths: Sequence[str], *,
               progress: Optional[Callable[[str, int, int], None]] = None,
               cancel: Optional[Callable[[], bool]] = None,
               source_ids: Optional[Dict[str, str]] = None,
               titles: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        paths = list(paths)
        # Cancel is honoured only before dispatch — see the module docstring.
        if cancel is not None and cancel():
            return {"files": 0, "documents_indexed": 0, "skipped_unchanged": 0,
                    "chunks_embedded": 0, "errors": []}
        if progress is not None:
            progress("extract", 0, len(paths))
        out = self._post("ingest", paths=paths, source_ids=source_ids, titles=titles)
        if progress is not None:
            progress("embed", int(out.get("chunks_embedded") or 0),
                     int(out.get("chunks_embedded") or 0))
        return out

    # -- search ------------------------------------------------------------ #
    def search(self, query: str, k: Optional[int] = None, *,
               rerank_results: bool = True, pool: Optional[int] = None) -> List[Any]:
        return [_json_to_hit(d) for d in self._post(
            "search", query=query, k=k, rerank_results=rerank_results, pool=pool)]

    def search_multi(self, queries: Sequence[str], k: Optional[int] = None, *,
                     pool: Optional[int] = None) -> List[Any]:
        return [_json_to_hit(d) for d in self._post(
            "search_multi", queries=list(queries), k=k, pool=pool)]

    # -- stats / introspection --------------------------------------------- #
    def documents(self) -> List[Dict[str, Any]]:
        return self._post("documents")

    def stats(self) -> Dict[str, Any]:
        return self._post("stats")

    def is_empty(self) -> bool:
        return bool(self._post("is_empty"))

    def corpus_scripts(self, sample: int = 50) -> set:
        return set(self._post("corpus_scripts", sample=sample))

    def all_chunks(self, paths: Optional[Sequence[str]] = None) -> List[str]:
        return self._post("all_chunks", paths=list(paths) if paths else None)

    # -- delete ------------------------------------------------------------ #
    def purge_all(self) -> int:
        return int(self._post("purge_all"))

    # -- lifecycle --------------------------------------------------------- #
    def close(self) -> None:
        s, self._session = self._session, None
        if s is not None:
            try:
                s.close()
            except Exception:
                pass

    # -- legacy aliases (see knowledge_client) ------------------------------ #
    def build(self, paths, **kw):
        return self.ingest(paths, **kw)

    def retrieve(self, query, k=None, **kw):
        return self.search(query, k=k, **kw)

    def retrieve_multi(self, queries, k=None, **kw):
        return self.search_multi(queries, k=k, **kw)


def _json_to_hit(d: Dict[str, Any]):
    """Rebuild a knowledge.SearchResult so callers keep attribute access and the
    existing rerank/format_context code paths work untouched."""
    import knowledge
    return knowledge.SearchResult(**d)


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Standalone knowledge/RAG service.")
    ap.add_argument("--port", type=int, default=int(os.environ.get("KNOWLEDGE_SERVICE_PORT", DEFAULT_PORT)))
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    embed_base = os.environ.get(ENV_EMBED_BASE)
    if embed_base:
        # Applied before any Library is built, so every request thread inherits it.
        import config
        config.EMBED_BASE = embed_base
    serve(args.port, args.host)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

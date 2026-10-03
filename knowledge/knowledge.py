"""knowledge.py — durable hybrid knowledge index (P2 retrieval layer).

A single SQLite file (`knowledge.db`) that turns the otherwise-throwaway research
page cache (`memory/research/_cache/*.json`) into a queryable knowledge base.

Design (per docs/knowledge_architecture.md — approved):
  * `documents` — one row per source page/file (metadata + full text + hash).
  * `chunks`    — text split into retrieval-sized pieces; FTS5 indexes these.
  * `chunks_fts`— external-content FTS5 virtual table (BM25 keyword search),
                  kept in sync with `chunks` via triggers.
  * embeddings  — float32 BLOB stored per chunk, produced on demand by the
                  embedding model **already loaded in LM Studio**
                  (`config.EMBED_MODEL`, default BGE-M3 / 1024-dim) via
                  `/v1/embeddings`. No new heavy dependency, no resident VRAM.

Retrieval is **hybrid**: FTS5/BM25 (precision on names/numbers) fused with cosine
similarity over embeddings (recall on paraphrase) using Reciprocal Rank Fusion.

Hard requirements honored:
  * survives restart  — it's a plain SQLite file.
  * survives upgrades — schema versioned in a `meta` table; additive migrations.
  * handles missing embeddings — chunks may have NULL embedding; cosine simply
    skips them and the row is still reachable via FTS.
  * handles embedding-API failure — `embed()` returns None on any error; search
    degrades to FTS-only. No turn ever blocks on the embed endpoint.
  * NO hard dependency on vector search — numpy is used if present for a fast
    matrix cosine, otherwise a pure-Python cosine is used.
"""
from __future__ import annotations

import hashlib
import json
import turn_trace
import logging
import os
import re
import sqlite3
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger("assistant.knowledge")

try:                                # numpy is available in the venv; degrade if not.
    import numpy as _np
except Exception:                   # pragma: no cover - defensive
    _np = None

try:
    import requests as _requests
except Exception:                   # pragma: no cover - defensive
    _requests = None

SCHEMA_VERSION = 1

# Embedding model/dim come from config so there is ONE place to change them.
# They used to be hardcoded here, in default_kb() and in library.py, which meant
# a "switch" silently left two of the three on the old model.
try:                                # pragma: no cover - config is always present
    import config as _cfg
    EMBED_MODEL_DEFAULT = getattr(_cfg, "EMBED_MODEL", "text-embedding-bge-m3")
    EMBED_DIM_DEFAULT   = int(getattr(_cfg, "EMBED_DIM", 1024))
    EMBED_BASE_DEFAULT  = getattr(_cfg, "EMBED_BASE", "http://localhost:1234")
except Exception:                   # pragma: no cover - defensive
    EMBED_MODEL_DEFAULT = "text-embedding-bge-m3"
    EMBED_DIM_DEFAULT   = 1024
    EMBED_BASE_DEFAULT  = "http://localhost:1234"
_RRF_K = 60                         # standard Reciprocal Rank Fusion constant
_FLOAT_FMT = "<%df"                 # little-endian float32 array


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #
@dataclass
class SearchResult:
    chunk_id: int
    doc_id: int
    source_type: str
    title: str
    url: str
    path: str
    text: str
    score: float                    # fused score (higher = better)
    bm25_rank: Optional[int] = None
    cosine_rank: Optional[int] = None
    cosine: Optional[float] = None
    why: str = ""

    def to_public(self) -> dict:
        return {
            "chunk_id": self.chunk_id, "doc_id": self.doc_id,
            "source_type": self.source_type, "title": self.title,
            "url": self.url, "path": self.path,
            "snippet": (self.text or "")[:280],
            "score": round(self.score, 5),
            "bm25_rank": self.bm25_rank, "cosine_rank": self.cosine_rank,
            "cosine": None if self.cosine is None else round(self.cosine, 4),
            "why": self.why,
        }


# --------------------------------------------------------------------------- #
# Embedding helper (LM Studio /v1/embeddings) — degrades to None on any failure
# --------------------------------------------------------------------------- #
class Embedder:
    """Thin client for an OpenAI-compatible /v1/embeddings endpoint.

    Never raises into callers: any network/model error returns None so retrieval
    falls back to FTS-only. Availability is probed lazily and cached briefly.
    """

    def __init__(self, base_url: str = "",
                 model: str = "",
                 enabled: bool = True, timeout: float = 30.0):
        base_url = base_url or EMBED_BASE_DEFAULT
        model = model or EMBED_MODEL_DEFAULT
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.enabled = enabled and _requests is not None
        self.timeout = timeout
        self.dim: Optional[int] = None
        self._avail: Optional[bool] = None
        self._avail_ts = 0.0
        self.calls = 0
        self.failures = 0

    def available(self, recheck: float = 30.0) -> bool:
        if not self.enabled:
            return False
        now = time.time()
        if self._avail is not None and (now - self._avail_ts) < recheck:
            return self._avail
        ok = self.embed(["ping"]) is not None
        self._avail = ok
        self._avail_ts = now
        return ok

    def embed(self, texts: Sequence[str]) -> Optional[List[List[float]]]:
        """Return a list of vectors for `texts`, or None on any failure."""
        if not self.enabled or not texts:
            return None
        try:
            self.calls += 1
            r = _requests.post(
                f"{self.base_url}/v1/embeddings",
                json={"model": self.model, "input": list(texts)},
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json().get("data", [])
            vecs = [d["embedding"] for d in sorted(data, key=lambda d: d.get("index", 0))]
            if vecs:
                self.dim = len(vecs[0])
            if len(vecs) != len(texts):
                logger.warning("embed: got %d vectors for %d inputs", len(vecs), len(texts))
                return None
            return vecs
        except Exception as exc:
            self.failures += 1
            logger.debug("embed failed: %s", exc)
            self._avail = False
            self._avail_ts = time.time()
            return None


# --------------------------------------------------------------------------- #
# BLOB (de)serialization
# --------------------------------------------------------------------------- #
def _vec_to_blob(vec: Sequence[float]) -> bytes:
    if _np is not None:
        return _np.asarray(vec, dtype="<f4").tobytes()
    return struct.pack(_FLOAT_FMT % len(vec), *vec)


def _blob_to_list(blob: bytes) -> List[float]:
    n = len(blob) // 4
    if _np is not None:
        return _np.frombuffer(blob, dtype="<f4").tolist()
    return list(struct.unpack(_FLOAT_FMT % n, blob))


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def _embed_input(row) -> str:
    """Text actually embedded for a chunk row: title header + body (see ensure_embeddings)."""
    title = (row["title"] or "").strip()
    if not title or os.getenv("EMBED_TITLE_HEADER", "0") != "1":
        return row["text"]
    return os.path.splitext(title)[0] + "\n\n" + row["text"]


def chunk_text(text: str, target_words: int = 240, overlap: int = 40) -> List[str]:
    """Split text into ~target_words chunks with a small overlap, on paragraph
    boundaries where possible. Robust to empty/short input."""
    text = (text or "").strip()
    if not text:
        return []
    # The hard-split step below advances by (target_words - overlap). At overlap
    # >= target_words that stride is <= 0, so the buffer never shrinks and the
    # loop spins forever — an indexing call that never returns and takes the
    # worker thread with it. Clamp instead of trusting the caller.
    target_words = max(1, int(target_words))
    overlap = max(0, min(int(overlap), target_words - 1))
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        paras = [text]
    chunks: List[str] = []
    buf: List[str] = []
    buf_n = 0
    for para in paras:
        words = para.split()
        if not words:
            continue
        if buf_n + len(words) <= target_words or not buf:
            buf.extend(words)
            buf_n += len(words)
        else:
            chunks.append(" ".join(buf))
            tail = buf[-overlap:] if overlap and len(buf) > overlap else []
            buf = list(tail) + words
            buf_n = len(buf)
        # a single very long paragraph: hard-split it
        while buf_n > target_words * 2:
            chunks.append(" ".join(buf[:target_words]))
            buf = buf[target_words - overlap:]
            buf_n = len(buf)
    if buf:
        chunks.append(" ".join(buf))
    return [c for c in chunks if c.strip()]


def _hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", "ignore")).hexdigest()


# --------------------------------------------------------------------------- #
# The knowledge base
# --------------------------------------------------------------------------- #
class KnowledgeBase:
    def __init__(self, db_path: Path | str, embedder: Optional[Embedder] = None,
                 embed_dim: int = EMBED_DIM_DEFAULT):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder if embedder is not None else Embedder()
        self.embed_dim = embed_dim
        # timeout + busy_timeout: when another connection (e.g. a GUI reader) holds the
        # file, wait for it instead of failing instantly with "database is locked".
        self._conn = sqlite3.connect(str(self.path), timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        # SQLite does NOT enforce foreign keys unless asked — without this, ON DELETE
        # CASCADE silently does nothing and deleting a document orphans its chunks
        # (the "0 documents · N passages" bug).
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    # -- schema ------------------------------------------------------------- #
    def _init_schema(self) -> None:
        c = self._conn
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

            CREATE TABLE IF NOT EXISTS documents (
                id          INTEGER PRIMARY KEY,
                source_type TEXT NOT NULL,
                source_id   TEXT UNIQUE,      -- stable key (url / path) for upsert
                url         TEXT DEFAULT '',
                path        TEXT DEFAULT '',
                title       TEXT DEFAULT '',
                profile     TEXT DEFAULT '',
                hash        TEXT DEFAULT '',
                created     REAL,
                updated     REAL,
                meta        TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id          INTEGER PRIMARY KEY,
                doc_id      INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                ord         INTEGER NOT NULL,
                text        TEXT NOT NULL,
                n_words     INTEGER DEFAULT 0,
                embedding   BLOB,
                embed_model TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_noembed ON chunks(id) WHERE embedding IS NULL;

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text,
                content='chunks',
                content_rowid='id',
                tokenize='porter unicode61'
            );

            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
                INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
            END;
            """
        )
        c.execute("INSERT OR IGNORE INTO meta(key,value) VALUES('schema_version',?)",
                  (str(SCHEMA_VERSION),))
        c.commit()

    # -- ingestion ---------------------------------------------------------- #
    def add_document(self, source_type: str, source_id: str, text: str,
                     title: str = "", url: str = "", path: str = "",
                     profile: str = "", meta: Optional[dict] = None,
                     embed: bool = True) -> Tuple[int, bool]:
        """Insert/replace a document and (re)chunk it. Returns (doc_id, changed).
        If the source text hash is unchanged, this is a no-op (returns changed=False)."""
        text = text or ""
        h = _hash(text)
        now = time.time()
        cur = self._conn.execute("SELECT id, hash FROM documents WHERE source_id=?", (source_id,))
        row = cur.fetchone()
        if row and row["hash"] == h:
            return row["id"], False     # unchanged — skip re-chunk/re-embed

        if row:
            doc_id = row["id"]
            self._conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            self._conn.execute(
                "UPDATE documents SET source_type=?,url=?,path=?,title=?,profile=?,"
                "hash=?,updated=?,meta=? WHERE id=?",
                (source_type, url, path, title, profile, h, now,
                 json.dumps(meta or {}, ensure_ascii=False), doc_id))
        else:
            cur = self._conn.execute(
                "INSERT INTO documents(source_type,source_id,url,path,title,profile,"
                "hash,created,updated,meta) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (source_type, source_id, url, path, title, profile, h, now, now,
                 json.dumps(meta or {}, ensure_ascii=False)))
            doc_id = cur.lastrowid

        chunks = chunk_text(text)
        for ordi, ch in enumerate(chunks):
            self._conn.execute(
                "INSERT INTO chunks(doc_id,ord,text,n_words) VALUES(?,?,?,?)",
                (doc_id, ordi, ch, len(ch.split())))
        self._conn.commit()
        if embed:
            self.ensure_embeddings(doc_id=doc_id)
        return doc_id, True

    def ensure_embeddings(self, doc_id: Optional[int] = None, batch: int = 16,
                          max_chunks: Optional[int] = None, progress=None,
                          cancel=None, workers: int = 1) -> int:
        """Embed chunks that lack an embedding. Returns count embedded.
        Safe no-op when the embed endpoint is unavailable (returns 0).

        progress(done, total): optional callback after each committed batch (for a
        UI progress bar on a large build). cancel(): optional predicate; when it
        returns True the loop stops (already-embedded chunks are kept — the rest stay
        FTS-only and can be embedded on a later run).

        workers: number of concurrent embedding requests in flight. The embedding is
        computed by the (GPU-backed) LM Studio endpoint, not locally; the only cost
        here is HTTP round-trip latency, so overlapping several requests keeps the
        endpoint's GPU fed instead of idling between sequential batches. SQLite writes
        stay on this thread (one connection, no cross-thread use). workers=1 = the
        original sequential behaviour."""
        # Chunks embedded by a DIFFERENT model are stale, not done. The old query
        # was `WHERE embedding IS NULL` alone, so switching EMBED_MODEL re-embedded
        # nothing: every vector stayed on the previous model and was silently
        # compared against new-model query vectors — degraded retrieval with no
        # error anywhere. embed_model IS NULL covers rows written before that
        # column was populated.
        # Contextual header: the chunk is embedded as "<document title>\n\n<text>"
        # (Anthropic contextual retrieval, the LLM-free variant). A bare chunk
        # loses which document/topic it belongs to. Off by default: no gain on bench/rag_eval (hit@1 0.675 vs 0.688). EMBED_TITLE_HEADER=1 enables.
        q = ("SELECT chunks.id AS id, chunks.text AS text, "
             "COALESCE(documents.title, '') AS title FROM chunks "
             "LEFT JOIN documents ON documents.id = chunks.doc_id "
             "WHERE (embedding IS NULL OR embed_model IS NULL OR embed_model <> ?)")
        params: list = [self.embedder.model]
        if doc_id is not None:
            q += " AND chunks.doc_id=?"
            params.append(doc_id)
        q += " ORDER BY chunks.id"
        if max_chunks:
            q += f" LIMIT {int(max_chunks)}"
        rows = self._conn.execute(q, params).fetchall()
        if not rows:
            return 0
        if not self.embedder.available():
            logger.info("ensure_embeddings: embed endpoint unavailable; %d chunks left FTS-only",
                        len(rows))
            return 0
        total = len(rows)
        done = 0
        batches = [rows[i:i + batch] for i in range(0, total, batch)]

        def _commit(sub, vecs):
            nonlocal done
            for r, v in zip(sub, vecs):
                self._conn.execute(
                    "UPDATE chunks SET embedding=?, embed_model=? WHERE id=?",
                    (_vec_to_blob(v), self.embedder.model, r["id"]))
                done += 1
            self._conn.commit()
            if progress is not None:
                try:
                    progress(done, total)
                except Exception:
                    pass

        if workers and workers > 1 and len(batches) > 1:
            # Concurrent in-flight requests; embed() (a plain POST) releases the GIL
            # during I/O so threads genuinely overlap. Process in waves of `workers`
            # batches so in-flight requests stay bounded and cancel is checked between
            # waves. Writes happen here, serially (single SQLite connection).
            stop = False
            with turn_trace.Pool(max_workers=workers) as ex:
                for w in range(0, len(batches), workers):
                    if cancel is not None and cancel():
                        stop = True
                        break
                    wave = batches[w:w + workers]
                    results = list(ex.map(
                        lambda b: self.embedder.embed([_embed_input(r) for r in b]), wave))
                    for sub, vecs in zip(wave, results):
                        if vecs is None:
                            logger.warning("ensure_embeddings: batch failed; stopping (FTS-only for rest)")
                            stop = True
                            break
                        _commit(sub, vecs)
                    if stop:
                        break
            if stop:
                logger.info("ensure_embeddings: stopped after %d/%d chunks", done, total)
        else:
            for sub in batches:
                if cancel is not None and cancel():
                    logger.info("ensure_embeddings: cancelled after %d/%d chunks", done, total)
                    break
                vecs = self.embedder.embed([_embed_input(r) for r in sub])
                if vecs is None:
                    logger.warning("ensure_embeddings: batch failed; stopping (FTS-only for rest)")
                    break
                _commit(sub, vecs)
        if self.embedder.dim:
            self.embed_dim = self.embedder.dim
        return done

    # -- importers ---------------------------------------------------------- #
    def import_research_cache(self, cache_dir: Path | str, embed: bool = True,
                              limit: Optional[int] = None) -> dict:
        """Load memory/research/_cache/*.json into the index. Idempotent by URL+hash."""
        cache_dir = Path(cache_dir)
        files = sorted(cache_dir.glob("*.json"))
        if limit:
            files = files[:limit]
        added = changed = skipped = errors = 0
        for f in files:
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                errors += 1
                continue
            url = rec.get("url") or f.stem
            text = rec.get("text") or rec.get("content") or ""
            if not text.strip():
                skipped += 1
                continue
            _, ch = self.add_document(
                source_type="research", source_id=url, text=text,
                title=rec.get("title") or url, url=url,
                meta={"content_hash": rec.get("content_hash"), "ts": rec.get("ts"),
                      "cache_file": f.name},
                embed=False)              # embed in one batched pass below
            if ch:
                added += 1
            else:
                skipped += 1
        if embed:
            changed = self.ensure_embeddings()
        return {"files": len(files), "documents_indexed": added,
                "skipped_unchanged": skipped, "errors": errors,
                "chunks_embedded": changed, **self.stats()}

    def import_markdown_dir(self, docs_dir: Path | str, embed: bool = True) -> dict:
        docs_dir = Path(docs_dir)
        added = 0
        for f in sorted(docs_dir.glob("*.md")):
            try:
                text = f.read_text(encoding="utf-8")
            except Exception:
                continue
            title = next((ln.lstrip("# ").strip() for ln in text.splitlines()
                          if ln.strip().startswith("#")), f.stem)
            _, ch = self.add_document("doc", str(f), text, title=title,
                                      path=str(f), embed=False)
            added += int(ch)
        embedded = self.ensure_embeddings() if embed else 0
        return {"documents_indexed": added, "chunks_embedded": embedded}

    # -- retrieval ---------------------------------------------------------- #
    @staticmethod
    def _fts_query(query: str) -> str:
        """Build a safe FTS5 MATCH expression: OR of quoted terms (prefix-matched)."""
        terms = [t for t in re.split(r"\W+", query or "") if len(t) > 1]
        if not terms:
            return ""
        return " OR ".join(f'"{t}"*' for t in terms)

    def search_fts(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Return [(chunk_id, bm25_score_positive)] best-first."""
        match = self._fts_query(query)
        if not match:
            return []
        try:
            rows = self._conn.execute(
                "SELECT rowid, bm25(chunks_fts) AS b FROM chunks_fts "
                "WHERE chunks_fts MATCH ? ORDER BY b LIMIT ?", (match, k)).fetchall()
        except sqlite3.OperationalError as exc:
            logger.debug("fts query failed: %s", exc)
            return []
        # bm25() returns more-negative = better; flip to positive descending.
        return [(r["rowid"], -float(r["b"])) for r in rows]

    def _all_embeddings(self, dim: Optional[int] = None):
        """Vectors comparable with the CURRENT model, as (ids, normalised matrix).

        Two filters, both load-bearing during a model migration:
          * embed_model — a vector from another model is meaningless against this
            model's query vector even when the dimensions happen to agree.
          * byte length — belt and braces. Mixing 768- and 1024-dim blobs made
            np.stack raise ValueError("all input arrays must have the same shape")
            and every semantic search died. A half-migrated index must degrade to
            FTS-only, never crash.
        """
        rows = self._conn.execute(
            "SELECT id, embedding FROM chunks "
            "WHERE embedding IS NOT NULL AND embed_model = ?",
            (self.embedder.model,)).fetchall()
        if not rows:
            # No vectors from this model yet (fresh switch, mid-reindex): fall back
            # to whatever is stored, still length-filtered, so an index that was
            # never migrated keeps working until it is.
            rows = self._conn.execute(
                "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL"
            ).fetchall()
        if rows:
            want = dim * 4 if dim else None
            if want is None:
                # majority length wins — the current model's vectors
                lengths: dict = {}
                for r in rows:
                    lengths[len(r["embedding"])] = lengths.get(len(r["embedding"]), 0) + 1
                want = max(lengths, key=lengths.get)
            dropped = [r for r in rows if len(r["embedding"]) != want]
            if dropped:
                logger.warning("ignoring %d chunk vector(s) of a different dimension "
                               "— reindex to make them searchable again", len(dropped))
            rows = [r for r in rows if len(r["embedding"]) == want]
        ids = [r["id"] for r in rows]
        if not ids:
            return [], None
        if _np is not None:
            mat = _np.stack([_np.frombuffer(r["embedding"], dtype="<f4") for r in rows])
            norms = _np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return ids, mat / norms
        return ids, [_blob_to_list(r["embedding"]) for r in rows]

    def search_semantic(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Return [(chunk_id, cosine)] best-first, or [] if embeddings unavailable."""
        qv = self.embedder.embed([query]) if self.embedder.enabled else None
        if not qv:
            return []
        # Size the candidate set to the query vector itself, so stored vectors of
        # any other dimension are excluded rather than blowing up the stack.
        ids, mat = self._all_embeddings(dim=len(qv[0]))
        if not ids:
            return []
        if _np is not None:
            q = _np.asarray(qv[0], dtype="<f4")
            qn = q / (_np.linalg.norm(q) or 1.0)
            sims = mat @ qn
            order = _np.argsort(-sims)[:k]
            return [(ids[i], float(sims[i])) for i in order]
        # pure-python fallback
        q = qv[0]
        qnorm = sum(x * x for x in q) ** 0.5 or 1.0
        scored = []
        for cid, vec in zip(ids, mat):
            dot = sum(a * b for a, b in zip(q, vec))
            vnorm = sum(x * x for x in vec) ** 0.5 or 1.0
            scored.append((cid, dot / (qnorm * vnorm)))
        scored.sort(key=lambda t: -t[1])
        return scored[:k]

    def search(self, query: str, k: int = 10, mode: str = "auto",
               pool: int = 30) -> List[SearchResult]:
        """Hybrid search.

        mode: 'fts' | 'semantic' | 'hybrid' | 'auto' (hybrid if embeddings exist,
        else fts). Hybrid fuses BM25 and cosine rankings with Reciprocal Rank
        Fusion (no tuning, robust to scale differences).
        """
        query = (query or "").strip()
        if not query:
            return []
        use_sem = mode in ("semantic", "hybrid")
        if mode == "auto":
            use_sem = self.has_embeddings() and self.embedder.enabled
            mode = "hybrid" if use_sem else "fts"

        fts = [] if mode == "semantic" else self.search_fts(query, pool)
        sem = self.search_semantic(query, pool) if use_sem else []

        if mode == "fts" or not sem:
            ranked = [(cid, sc, i + 1, None, None) for i, (cid, sc) in enumerate(fts)][:k]
            why = "BM25 keyword"
        elif mode == "semantic" or not fts:
            ranked = [(cid, sc, None, i + 1, sc) for i, (cid, sc) in enumerate(sem)][:k]
            why = "embedding cosine"
        else:
            # Reciprocal Rank Fusion
            fts_rank = {cid: i + 1 for i, (cid, _) in enumerate(fts)}
            sem_rank = {cid: i + 1 for i, (cid, _) in enumerate(sem)}
            sem_cos = {cid: s for cid, s in sem}
            fused: dict = {}
            for cid, r in fts_rank.items():
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (_RRF_K + r)
            for cid, r in sem_rank.items():
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (_RRF_K + r)
            order = sorted(fused, key=lambda c: -fused[c])[:k]
            ranked = [(cid, fused[cid], fts_rank.get(cid), sem_rank.get(cid),
                       sem_cos.get(cid)) for cid in order]
            why = "hybrid RRF (BM25 + cosine)"

        return self._hydrate(ranked, why)

    def _hydrate(self, ranked, why) -> List[SearchResult]:
        out: List[SearchResult] = []
        for cid, score, br, cr, cos in ranked:
            row = self._conn.execute(
                "SELECT c.id, c.doc_id, c.text, d.source_type, d.title, d.url, d.path "
                "FROM chunks c JOIN documents d ON d.id=c.doc_id WHERE c.id=?", (cid,)
            ).fetchone()
            if not row:
                continue
            out.append(SearchResult(
                chunk_id=row["id"], doc_id=row["doc_id"], source_type=row["source_type"],
                title=row["title"] or "", url=row["url"] or "", path=row["path"] or "",
                text=row["text"] or "", score=float(score),
                bm25_rank=br, cosine_rank=cr, cosine=cos, why=why))
        return out

    # -- maintenance / introspection --------------------------------------- #
    def has_embeddings(self) -> bool:
        r = self._conn.execute("SELECT 1 FROM chunks WHERE embedding IS NOT NULL LIMIT 1").fetchone()
        return r is not None

    def stats(self) -> dict:
        c = self._conn
        docs = c.execute("SELECT COUNT(*) n FROM documents").fetchone()["n"]
        chunks = c.execute("SELECT COUNT(*) n FROM chunks").fetchone()["n"]
        embedded = c.execute("SELECT COUNT(*) n FROM chunks WHERE embedding IS NOT NULL").fetchone()["n"]
        by_type = {r["source_type"]: r["n"] for r in c.execute(
            "SELECT source_type, COUNT(*) n FROM documents GROUP BY source_type")}
        # Which model produced the stored vectors. Without this a migration is
        # invisible: coverage stays at 100% while every vector is from the old
        # model and unusable.
        by_model = {(r["embed_model"] or "?"): r["n"] for r in c.execute(
            "SELECT embed_model, COUNT(*) n FROM chunks "
            "WHERE embedding IS NOT NULL GROUP BY embed_model")}
        current = int(by_model.get(self.embedder.model, 0))
        return {"documents": docs, "chunks": chunks, "chunks_embedded": embedded,
                "embed_coverage": round(embedded / chunks, 3) if chunks else 0.0,
                "by_source_type": by_type, "embed_model": self.embedder.model,
                "embed_by_model": by_model,
                "embed_stale": max(0, embedded - current),
                "embed_current": current,
                "embed_available": self.embedder.enabled}

    def reindex_embeddings(self, *, drop_stale: bool = True, **kw) -> dict:
        """Re-embed everything that is not on the current model.

        drop_stale: clear vectors from other models first, so a failure part-way
        through leaves rows with NO embedding (searchable via FTS) rather than a
        vector that silently belongs to the wrong model.
        """
        before = self.stats()
        if drop_stale:
            self._conn.execute(
                "UPDATE chunks SET embedding=NULL, embed_model=NULL "
                "WHERE embedding IS NOT NULL AND "
                "(embed_model IS NULL OR embed_model <> ?)", (self.embedder.model,))
            self._conn.commit()
        embedded = self.ensure_embeddings(**kw)
        after = self.stats()
        return {"model": self.embedder.model, "re_embedded": embedded,
                "before": before, "after": after}

    def purge(self, source_type: Optional[str] = None, profile: Optional[str] = None) -> int:
        q = "DELETE FROM documents WHERE 1=1"
        params: list = []
        if source_type:
            q += " AND source_type=?"; params.append(source_type)
        if profile:
            q += " AND profile=?"; params.append(profile)
        n = self._conn.execute(q, params).rowcount
        self._conn.commit()
        self._conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
        return n

    def repair(self) -> int:
        """Delete chunks whose parent document is gone (orphans left by an old delete
        that ran without FK enforcement), then rebuild the FTS index from the live
        chunks. Returns the number of orphan chunks removed. Idempotent."""
        n = self._conn.execute(
            "DELETE FROM chunks WHERE doc_id NOT IN (SELECT id FROM documents)").rowcount
        self._conn.commit()
        try:
            self._conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
            self._conn.commit()
        except Exception:
            logger.exception("repair: FTS rebuild failed")
        if n:
            logger.info("repair: removed %d orphan chunks", n)
        return n

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


# Module-level default base url import is deferred to avoid a hard config dep.
def default_kb(db_path: Optional[Path] = None) -> KnowledgeBase:
    base = EMBED_BASE_DEFAULT
    model = EMBED_MODEL_DEFAULT
    try:
        import config
        base = getattr(config, "EMBED_BASE", None) or getattr(config, "LM_STUDIO_BASE", base)
        model = getattr(config, "EMBED_MODEL", model)
    except Exception:
        pass
    if db_path is None:
        db_path = Path("memory") / "knowledge.db"
    return KnowledgeBase(db_path, embedder=Embedder(base_url=base, model=model))

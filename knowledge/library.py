"""library.py — a local *document library* you can chat with.

Upload large text documents (TXT, Markdown, PDF, EPUB), build a searchable index,
and retrieve the most relevant passages for a question so the model can answer from
a 1,200-page book without the whole thing ever touching its context window.

The retrieval engine is the project's existing hybrid index (`knowledge.KnowledgeBase`):
BM25/FTS5 lexical search fused with embedding cosine via Reciprocal Rank Fusion.
This module only adds (a) text extraction from document files and (b) a thin,
build-with-progress + retrieve-and-format wrapper, stored in its OWN SQLite file
(`memory/library.db`) so it never mixes with the research-cache knowledge base.

Design choices honoured (match the rest of the codebase):
  * survives restart/upgrade — plain SQLite file, additive schema (inherited).
  * no new heavy dependency — PDF via pypdf (already used), EPUB via the stdlib
    zipfile + beautifulsoup (already used); embeddings via the LM Studio endpoint.
  * degrades gracefully — if the embed endpoint is down, build/search fall back to
    BM25-only; extraction errors are per-file and never abort the whole build.
"""
from __future__ import annotations

import logging
import os
import re
import zipfile
from pathlib import Path
from typing import Callable, List, Optional

import knowledge

logger = logging.getLogger("assistant.library")

SUPPORTED_EXTS = {".txt", ".text", ".log", ".md", ".markdown", ".pdf", ".epub", ".docx"}
SOURCE_TYPE = "library"
DEFAULT_DB = Path("memory") / "library.db"


def redirect_db(path) -> None:
    """Point the default library at another file. FOR TESTS.

    Mirrors tg_bot.redirect_data_dir, and exists for the same reason: a suite
    that drives the Database tab opened the LIVE memory/library.db. When the
    real app is running it holds that file, so the suite blocked inside
    _init_schema on SQLite's busy_timeout -- 30 seconds per statement, which is
    what "the test suite hangs" actually was. Read at call time by Library(),
    so assigning here is enough.
    """
    global DEFAULT_DB
    DEFAULT_DB = Path(path)


# Retrieval depth. Tuned for "ask a large book" recall without blowing a small context
# window: pull this many passages and cap the injected context at this many chars. Both
# overridable via config (LIBRARY_RETRIEVE_K / LIBRARY_CONTEXT_CHARS) for big-context rigs.
DEFAULT_K = 25
DEFAULT_CONTEXT_CHARS = 32000
_CHARS_PER_PASSAGE = 2600       # budget headroom per requested passage (RU chunk ~2k chars)
_MAX_CONTEXT_CHARS = 200000     # absolute safety ceiling regardless of slider
DEFAULT_SCAN_BATCH_CHARS = 12000   # whole-book scan: text per map call (bigger = fewer LLM calls)


def _cfg(name: str, default):
    try:
        import config
        v = getattr(config, name, default)
        return v if v else default
    except Exception:
        return default


# --------------------------------------------------------------------------- #
# Cross-lingual helpers — a Russian book can't be keyword-matched by an English
# question ("eyes" != "глаза"). We detect the corpus script(s) and (in the worker)
# translate the query into those languages, then retrieve over all query variants.
# --------------------------------------------------------------------------- #
_SCRIPT_RANGES = [
    ("cyrillic", 0x0400, 0x04FF), ("greek", 0x0370, 0x03FF),
    ("han", 0x4E00, 0x9FFF), ("kana", 0x3040, 0x30FF),
    ("arabic", 0x0600, 0x06FF), ("hebrew", 0x0590, 0x05FF),
    ("latin", 0x0041, 0x024F),
]
_SCRIPT_LANG = {"cyrillic": "Russian", "greek": "Greek", "han": "Chinese",
                "kana": "Japanese", "arabic": "Arabic", "hebrew": "Hebrew",
                "latin": "English"}


def text_scripts(text: str, min_ratio: float = 0.08) -> set:
    """Scripts that make up at least `min_ratio` of the alphabetic characters."""
    counts: dict = {}
    total = 0
    for ch in text or "":
        o = ord(ch)
        for name, lo, hi in _SCRIPT_RANGES:
            if lo <= o <= hi:
                counts[name] = counts.get(name, 0) + 1
                total += 1
                break
    if not total:
        return set()
    return {n for n, c in counts.items() if c / total >= min_ratio}


def cross_lingual_targets(query: str, corpus_scripts: set) -> List[str]:
    """Languages to translate `query` into: those present in the corpus but not in
    the query's own script(s)."""
    qs = text_scripts(query)
    out = []
    for s in corpus_scripts:
        if s not in qs and s in _SCRIPT_LANG and _SCRIPT_LANG[s] not in out:
            out.append(_SCRIPT_LANG[s])
    return out


# --------------------------------------------------------------------------- #
# Whole-book scan (map-reduce) — for exhaustive "find all X" questions that top-k
# retrieval can't answer. Reads EVERY chunk in batches (map: extract relevant facts),
# then synthesizes (reduce). Slow but complete. LLM calls are injected so the core
# orchestration is testable without a model.
# --------------------------------------------------------------------------- #
def batch_chunks(chunks: List[str], batch_chars: int) -> List[str]:
    batch_chars = max(1, int(batch_chars))
    batches: List[str] = []
    cur: List[str] = []
    n = 0
    for ch in chunks:
        if n + len(ch) > batch_chars and cur:
            batches.append("\n\n".join(cur))
            cur, n = [], 0
        # A chunk larger than a whole batch used to be emitted verbatim, so the
        # budget that exists to keep a map call inside the context window was
        # silently exceeded (a 40k-char chunk against a 12k budget). Split it
        # rather than truncate: this feeds the whole-book scan, and dropping text
        # would make an "exhaustive" answer quietly incomplete.
        while len(ch) > batch_chars:
            batches.append(ch[:batch_chars])
            ch = ch[batch_chars:]
        if not ch:
            continue
        if n + len(ch) > batch_chars and cur:
            batches.append("\n\n".join(cur))
            cur, n = [], 0
        cur.append(ch)
        n += len(ch)
    if cur:
        batches.append("\n\n".join(cur))
    return batches


def map_reduce_scan(chunks: List[str], question: str, map_fn, reduce_fn, *,
                    batch_chars: Optional[int] = None, progress=None, cancel=None) -> str:
    """map_fn(batch_text, question)->notes ('' = nothing relevant); reduce_fn(notes,
    question)->final answer. progress(done, total); cancel()->bool stops early."""
    batch_chars = batch_chars or int(_cfg("LIBRARY_SCAN_BATCH_CHARS", DEFAULT_SCAN_BATCH_CHARS))
    batches = batch_chunks(chunks, batch_chars)
    notes: List[str] = []
    total = len(batches)
    for i, b in enumerate(batches):
        if cancel is not None and cancel():
            break
        if progress is not None:
            progress(i, total)
        note = map_fn(b, question)
        if note and note.strip():
            notes.append(note.strip())
    if progress is not None:
        progress(total, total)
    return reduce_fn(notes, question)


# --------------------------------------------------------------------------- #
# Text extraction
# --------------------------------------------------------------------------- #
def _extract_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    parts: List[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception as exc:                       # one bad page must not kill the file
            logger.debug("pdf page extract failed in %s: %s", path.name, exc)
    return "\n\n".join(parts)


def _extract_epub(path: Path) -> str:
    """EPUB = a ZIP of XHTML documents. Read the spine order from the OPF, strip
    tags with BeautifulSoup. Self-contained (no ebooklib dependency)."""
    from bs4 import BeautifulSoup
    out: List[str] = []
    with zipfile.ZipFile(str(path)) as zf:
        names = zf.namelist()
        # Find the OPF (package) file via META-INF/container.xml; fall back to a scan.
        opf_name = None
        try:
            container = zf.read("META-INF/container.xml").decode("utf-8", "replace")
            csoup = BeautifulSoup(container, "xml")
            rootfile = csoup.find("rootfile")
            if rootfile and rootfile.get("full-path"):
                opf_name = rootfile["full-path"]
        except Exception:
            pass
        ordered: List[str] = []
        if opf_name and opf_name in names:
            try:
                opf = zf.read(opf_name).decode("utf-8", "replace")
                osoup = BeautifulSoup(opf, "xml")
                base = os.path.dirname(opf_name)
                manifest = {it.get("id"): it.get("href")
                            for it in osoup.find_all("item") if it.get("id")}
                for ref in osoup.find_all("itemref"):
                    href = manifest.get(ref.get("idref"))
                    if not href:
                        continue
                    full = os.path.normpath(os.path.join(base, href)).replace("\\", "/")
                    if full in names:
                        ordered.append(full)
            except Exception as exc:
                logger.debug("epub OPF parse failed in %s: %s", path.name, exc)
        if not ordered:                                # fallback: every (x)html in zip order
            ordered = [n for n in names
                       if n.lower().endswith((".xhtml", ".html", ".htm"))]
        for name in ordered:
            try:
                html = zf.read(name).decode("utf-8", "replace")
                text = BeautifulSoup(html, "html.parser").get_text("\n")
                if text.strip():
                    out.append(text)
            except Exception as exc:
                logger.debug("epub doc extract failed (%s in %s): %s", name, path.name, exc)
    return "\n\n".join(out)


def _extract_docx(path: Path) -> str:
    """Paragraphs and table cells, in document order (python-docx)."""
    import docx
    d = docx.Document(str(path))
    out = [p.text for p in d.paragraphs if p.text.strip()]
    for t in d.tables:
        for row in t.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                out.append(" | ".join(cells))
    return "\n".join(out)


def extract_text(path: str | Path) -> str:
    """Extract plain text from a supported document. Raises ValueError for an
    unsupported extension; lets extraction errors propagate to the caller (which
    records them per-file)."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".txt", ".text", ".log", ".md", ".markdown"):
        return _extract_txt(path)
    if ext == ".pdf":
        return _extract_pdf(path)
    if ext == ".epub":
        return _extract_epub(path)
    if ext == ".docx":
        return _extract_docx(path)
    raise ValueError(f"unsupported document type: {ext or '(none)'}")


# --------------------------------------------------------------------------- #
# Reranking — sharpen the order of the hybrid candidate pool before the top-k
# reach the model. Two implementations:
#   * lexical (default, deterministic, no model): query-term coverage + density.
#     Always available, instant, no VRAM — safe default on a shared 12 GB GPU.
#   * cross-encoder (opt-in): a sentence-transformers CrossEncoder jointly scores
#     (query, passage). Loaded lazily only when configured; falls back to lexical
#     on any import/load/scoring error so retrieval never breaks.
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_RRF_K = 60  # match knowledge.py's fusion constant for the rank-prior term


def _terms(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def lexical_score(query: str, passage: str) -> float:
    """A transparent [0,1] relevance signal: how many distinct query terms the
    passage covers (weighted high) plus how densely they occur (weighted low)."""
    q = set(_terms(query))
    if not q:
        return 0.0
    p = _terms(passage)
    if not p:
        return 0.0
    qset = set(p) & q
    coverage = len(qset) / len(q)
    hits = sum(1 for w in p if w in q)
    density = min(hits / len(p) * 5.0, 1.0)      # saturate quickly
    return 0.75 * coverage + 0.25 * density


_cross_encoder_cache: dict = {}


class _ListwiseReranker:
    """jina-reranker-v3.x is listwise (one query, all documents in one pass) and
    is not a sentence-transformers CrossEncoder. This adapter exposes the same
    predict([(query, passage), ...]) contract rerank() already uses; every pair
    in a call shares one query, so the whole pool is scored in one pass."""

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModel
        device = os.getenv("LIBRARY_CE_DEVICE", "cpu")
        self._model = AutoModel.from_pretrained(
            model_name, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            trust_remote_code=True).to(device).eval()

    def predict(self, pairs):
        if not pairs:
            return []
        query = pairs[0][0]
        docs = [p for _, p in pairs]
        out = [0.0] * len(docs)
        for r in self._model.rerank(query, docs):
            out[r["index"]] = float(r["relevance_score"])
        return out


class _YesLogitReranker:
    """zerank-2 ships for sentence-transformers 5.x (LogitScore module); on our
    3.x CrossEncoder loads it as a sequence classifier with a random head. Score
    it by hand: its chat template (query as system, document as user), then the
    last-position logit of the "Yes" token."""

    def __init__(self, model_name: str):
        import json
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._dev = os.getenv("LIBRARY_CE_DEVICE", "cpu")
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._tok.padding_side = "left"
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16 if self._dev == "cuda" else torch.float32
        ).to(self._dev).eval()
        with open(os.path.join(model_name, "1_LogitScore", "config.json")) as f:
            self._yes = json.load(f)["true_token_id"]

    def predict(self, pairs, batch: int = 8):
        import torch
        out = []
        for i in range(0, len(pairs), batch):
            texts = [self._tok.apply_chat_template(
                [{"role": "query", "content": q}, {"role": "document", "content": d}],
                tokenize=False, add_generation_prompt=True) for q, d in pairs[i:i + batch]]
            enc = self._tok(texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=4096).to(self._dev)
            with torch.no_grad():
                logits = self._model(**enc).logits[:, -1, self._yes]
            out += logits.float().tolist()
        return out


def load_cross_encoder(model_name: str):
    """Lazily load a sentence-transformers CrossEncoder, cached by name. Returns
    the model or None if unavailable (package missing / load failure)."""
    if not model_name:
        return None
    if model_name in _cross_encoder_cache:
        return _cross_encoder_cache[model_name]
    model = None
    if "jina-reranker-v3" in model_name or "zerank" in model_name:
        try:
            model = (_ListwiseReranker if "jina" in model_name else _YesLogitReranker)(model_name)
            logger.info("library: custom reranker loaded: %s", model_name)
        except Exception as exc:
            logger.warning("library: custom reranker unavailable (%s); using lexical rerank", exc)
        _cross_encoder_cache[model_name] = model
        return model
    try:
        from sentence_transformers import CrossEncoder
        # CPU by default: the card is full (chat model + TTS), and on 12 candidates
        # the CPU costs ~0.45 s a question. LIBRARY_CE_DEVICE=cuda opts back in.
        device = os.getenv("LIBRARY_CE_DEVICE", "cpu")
        model = CrossEncoder(model_name, device=device)
        logger.info("library: cross-encoder reranker loaded on %s: %s", device, model_name)
    except Exception as exc:
        logger.warning("library: cross-encoder unavailable (%s); using lexical rerank", exc)
    _cross_encoder_cache[model_name] = model
    return model


def rerank(query: str, results: List["knowledge.SearchResult"], k: int,
           *, cross_encoder=None) -> List["knowledge.SearchResult"]:
    """Reorder `results` (a hybrid candidate pool) and return the top-k.

    Fuses each candidate's existing hybrid rank (a stable prior) with a relevance
    score from the cross-encoder when supplied, else the lexical signal. Stable:
    ties keep the incoming order. Never raises — a cross-encoder failure degrades
    to lexical for that call.
    """
    if not results:
        return []
    rel: List[float]
    if cross_encoder is not None:
        try:
            scores = cross_encoder.predict([(query, r.text or "") for r in results])
            rel = [float(s) for s in scores]
            lo, hi = min(rel), max(rel)
            rng = (hi - lo) or 1.0
            rel = [(s - lo) / rng for s in rel]      # normalise to [0,1]
        except Exception:
            logger.exception("cross-encoder predict failed; keeping the hybrid order")
            return list(results[:k])
    elif os.getenv("LIBRARY_LEXICAL_RERANK") == "1":
        rel = [lexical_score(query, r.text or "") for r in results]
    else:
        # The lexical rerank made retrieval WORSE than the hybrid order it was
        # reordering (bench/rag_eval, 80 ru questions: hit@1 0.675 -> 0.588,
        # hit@5 0.963 -> 0.838). Without a cross-encoder, trust the hybrid.
        return list(results[:k])

    ranked = []
    for i, (r, rscore) in enumerate(zip(results, rel)):
        prior = 1.0 / (_RRF_K + i + 1)               # trust the hybrid order as a prior
        ranked.append((prior + 0.5 * rscore, i, r))
    ranked.sort(key=lambda t: (-t[0], t[1]))          # stable on ties via original index
    return [r for _, _, r in ranked[:k]]


# --------------------------------------------------------------------------- #
# Library (build + retrieve)
# --------------------------------------------------------------------------- #
class Library:
    """A document library backed by a hybrid (BM25 + embedding) index.

    NOTE on threading: a SQLite connection may only be used on the thread that
    created it. Create one Library on the GUI thread for retrieval, and a separate
    Library inside the build worker thread. They share the same file (WAL mode),
    so committed documents are visible to the reader connection.
    """

    def __init__(self, db_path: Optional[Path] = None):
        base = knowledge.EMBED_BASE_DEFAULT
        model = knowledge.EMBED_MODEL_DEFAULT
        ce_model = ""
        try:
            import config
            base = getattr(config, "EMBED_BASE", "") or \
                getattr(config, "LM_STUDIO_BASE", base) or base
            model = getattr(config, "EMBED_MODEL", model) or model
            # Neural reranker on CPU (config.LIBRARY_CROSS_ENCODER); off in test runs.
            ce_model = "" if os.getenv("F5_TEST_RUN") else (getattr(config, "LIBRARY_CROSS_ENCODER", "") or "")
        except Exception:
            pass
        self.kb = knowledge.KnowledgeBase(
            db_path or DEFAULT_DB,
            embedder=knowledge.Embedder(base_url=base, model=model))
        # Loaded on the first search that reranks, not here: the GUI opens its
        # Library on the UI thread at startup to list documents, and loading
        # here froze the window for the whole reranker download (minutes on a
        # first start, ~30 s of retries offline) for a tab that never reranks.
        self._ce_model = ce_model

    @property
    def _cross_encoder(self):
        if "_ce" not in self.__dict__:
            name = self.__dict__.get("_ce_model", "")
            self.__dict__["_ce"] = load_cross_encoder(name) if name else None
        return self.__dict__["_ce"]

    @_cross_encoder.setter
    def _cross_encoder(self, model):
        self.__dict__["_ce"] = model

    # -- build ------------------------------------------------------------- #
    def build(self, paths: List[str], *,
              progress: Optional[Callable[[str, int, int], None]] = None,
              cancel: Optional[Callable[[], bool]] = None,
              source_ids: Optional[dict] = None,
              titles: Optional[dict] = None) -> dict:
        """Ingest every file, then embed all chunks in one batched pass.

        progress(stage, done, total): 'extract' during ingestion (done=file index),
        then 'embed' during embedding (done=chunks embedded). cancel(): stop early.
        source_ids: optional {path: source_id} override. Dedup keys on source_id,
        which defaults to os.path.abspath(p) — fine for a stable on-disk file, but
        wrong for a caller that stages every ingest under a fresh temp path (e.g.
        the Telegram bot's per-upload tempdir): without a stable override, a
        re-upload of the same file gets a new abspath every time and is never
        recognised as the same document, so it duplicates instead of replacing.
        titles: optional {path: display title} override, for a caller that had to
        sanitize the on-disk basename (e.g. stripping characters Windows rejects
        in a path) but still wants the ORIGINAL name shown to the user.
        Returns a stats dict (documents/chunks/embed coverage + per-file errors).
        """
        self.kb.repair()   # self-heal any orphan chunks from an old FK-less delete
        added = skipped = 0
        errors: List[str] = []
        total_files = len(paths)
        for i, p in enumerate(paths):
            if cancel is not None and cancel():
                break
            name = (titles or {}).get(p) or os.path.basename(p)
            if progress is not None:
                progress("extract", i, total_files)
            try:
                text = extract_text(p)
            except Exception as exc:
                logger.warning("library build: extract failed for %s: %s", name, exc)
                errors.append(f"{name}: {exc}")
                continue
            if not text.strip():
                errors.append(f"{name}: no extractable text")
                continue
            try:
                sid = (source_ids or {}).get(p) or os.path.abspath(p)
                _, changed = self.kb.add_document(
                    source_type=SOURCE_TYPE, source_id=sid,
                    text=text, title=name, path=os.path.abspath(p), embed=False)
                added += int(changed)
                skipped += int(not changed)
            except Exception as exc:
                logger.exception("library build: ingest failed for %s", name)
                errors.append(f"{name}: {exc}")
        # One batched embedding pass over everything still missing an embedding.
        embedded = 0
        if cancel is None or not cancel():
            embedded = self.kb.ensure_embeddings(
                batch=int(_cfg("LIBRARY_EMBED_BATCH", 128)),
                workers=int(_cfg("LIBRARY_EMBED_WORKERS", 4)),
                progress=(lambda d, t: progress("embed", d, t)) if progress else None,
                cancel=cancel)
        out = {"files": total_files, "documents_indexed": added,
               "skipped_unchanged": skipped, "chunks_embedded": embedded,
               "errors": errors}
        out.update(self.kb.stats())
        return out

    # -- retrieve ---------------------------------------------------------- #
    def retrieve(self, query: str, k: Optional[int] = None, *, rerank_results: bool = True,
                 pool: Optional[int] = None) -> List[knowledge.SearchResult]:
        """Hybrid search, then rerank a larger candidate pool down to the top-k.

        rerank_results: apply the reranker (lexical by default, cross-encoder if
        configured). pool: candidate count to rerank (default max(k*4, 30))."""
        k = k or int(_cfg("LIBRARY_RETRIEVE_K", DEFAULT_K))
        if not rerank_results:
            return self.kb.search(query, k=k, mode="auto")
        # bench/rag_eval: a cross-encoder over 12 candidates matches 30 in quality
        # (hit@1 0.863, hit@5 1.0) at 0.45 s/question on CPU instead of 1.1 s.
        pool = pool or (max(k + 7, 12) if self._cross_encoder is not None else max(k * 4, 30))
        cands = self.kb.search(query, k=pool, mode="auto")
        if len(cands) <= k:
            return cands
        return rerank(query, cands, k, cross_encoder=self._cross_encoder)

    def retrieve_multi(self, queries: List[str], k: Optional[int] = None,
                       pool: Optional[int] = None) -> List[knowledge.SearchResult]:
        """Retrieve over several query variants (e.g. the question + its translations),
        merge the candidate pools (dedup by chunk), and rerank against the FIRST query."""
        queries = [q for q in queries if q and q.strip()]
        if not queries:
            return []
        if len(queries) == 1:
            return self.retrieve(queries[0], k=k, pool=pool)
        k = k or int(_cfg("LIBRARY_RETRIEVE_K", DEFAULT_K))
        per = pool or max(k * 4, 30)
        seen: dict = {}
        for q in queries:
            for r in self.kb.search(q, k=per, mode="auto"):
                seen.setdefault(r.chunk_id, r)
        merged = list(seen.values())
        if len(merged) <= k:
            return merged
        return rerank(queries[0], merged, k, cross_encoder=self._cross_encoder)

    def corpus_scripts(self, sample: int = 50) -> set:
        """Scripts present across a sample of indexed chunks (for cross-lingual routing)."""
        rows = self.kb._conn.execute(
            "SELECT c.text FROM chunks c JOIN documents d ON d.id=c.doc_id "
            "WHERE d.source_type=? LIMIT ?", (SOURCE_TYPE, sample)).fetchall()
        scripts: set = set()
        for r in rows:
            scripts |= text_scripts(r["text"])
        return scripts

    def all_chunks(self, paths: Optional[List[str]] = None) -> List[str]:
        """Every chunk's text in document order (optionally only for the given files).
        Used by the whole-book scan."""
        q = ("SELECT c.text FROM chunks c JOIN documents d ON d.id=c.doc_id "
             "WHERE d.source_type=?")
        params: list = [SOURCE_TYPE]
        if paths:
            ap = [os.path.abspath(p) for p in paths]
            q += " AND d.path IN (%s)" % ",".join("?" * len(ap))
            params += ap
        q += " ORDER BY c.doc_id, c.ord"
        return [r["text"] for r in self.kb._conn.execute(q, params).fetchall()]

    @staticmethod
    def format_context(results: List[knowledge.SearchResult],
                       max_chars: Optional[int] = None) -> str:
        """Render retrieved passages as a numbered, source-labelled context block,
        truncated to a character budget so it can never blow the context window."""
        max_chars = max_chars or int(_cfg("LIBRARY_CONTEXT_CHARS", DEFAULT_CONTEXT_CHARS))
        blocks: List[str] = []
        used = 0
        for i, r in enumerate(results, 1):
            src = r.title or os.path.basename(r.path) or r.source_type or "document"
            body = (r.text or "").strip()
            header = f"[{i}] {src}"
            piece = f"{header}\n{body}"
            if used + len(piece) > max_chars:
                if blocks:
                    break
                # The FIRST passage was emitted whole however long it was, so a
                # single oversized chunk blew the budget completely (50k chars
                # against a 1k cap) — the one thing this cap exists to prevent.
                room = max(0, max_chars - len(header) - 2)
                if room:
                    blocks.append(f"{header}\n{body[:room]}")
                break
            blocks.append(piece)
            used += len(piece)
        return "\n\n".join(blocks)

    # -- introspection / maintenance --------------------------------------- #
    def documents(self) -> List[dict]:
        rows = self.kb._conn.execute(
            "SELECT d.title, d.path, COUNT(c.id) n_chunks, "
            "SUM(c.embedding IS NOT NULL) n_embedded "
            "FROM documents d LEFT JOIN chunks c ON c.doc_id=d.id "
            "WHERE d.source_type=? GROUP BY d.id ORDER BY d.title",
            (SOURCE_TYPE,)).fetchall()
        return [{"title": r["title"], "path": r["path"],
                 "chunks": r["n_chunks"] or 0, "embedded": r["n_embedded"] or 0}
                for r in rows]

    def stats(self) -> dict:
        return self.kb.stats()

    def purge_all(self) -> int:
        return self.kb.purge(source_type=SOURCE_TYPE)

    def is_empty(self) -> bool:
        r = self.kb._conn.execute(
            "SELECT 1 FROM documents WHERE source_type=? LIMIT 1", (SOURCE_TYPE,)).fetchone()
        return r is None

    def close(self) -> None:
        self.kb.close()


SMALL_LIBRARY_CHUNKS = int(_cfg("LIBRARY_SMALL_CHUNKS", 6) or 6)


def _whole_small_library(lib: "Library") -> List[knowledge.SearchResult]:
    """Every chunk of a library no bigger than SMALL_LIBRARY_CHUNKS, as results."""
    try:
        rows = lib.kb._conn.execute(
            "SELECT c.id AS chunk_id, c.doc_id, d.title, d.path, c.text FROM chunks c "
            "JOIN documents d ON d.id=c.doc_id WHERE d.source_type=? "
            "ORDER BY c.doc_id, c.ord LIMIT ?", (SOURCE_TYPE, SMALL_LIBRARY_CHUNKS + 1)).fetchall()
    except Exception:
        return []
    if not rows or len(rows) > SMALL_LIBRARY_CHUNKS:
        return []
    return [knowledge.SearchResult(chunk_id=r["chunk_id"], doc_id=r["doc_id"], source_type=SOURCE_TYPE,
                                   title=r["title"] or "", url="", path=r["path"] or "",
                                   text=r["text"] or "", score=0.0, why="whole small library")
            for r in rows]


# The RAG wrapper's fixed head and the seam before the user's own words. The
# entry translator (graph_language) splits on these: it must translate the
# QUESTION alone, never the whole wrapper -- live 2026-09-18 00:44 the model
# "translated" 6 KB of passages + «Что нарисовано?» into «The documents do not
# specify what is drawn.» and the user's question was gone before the turn began.
RAG_HEAD = "Passages retrieved from the user's documents are below."
RAG_QUESTION_SEAM = "\n\nQuestion: "


def build_rag_prompt(lib: "Library", question: str, k: Optional[int] = None,
                     extra_queries: Optional[List[str]] = None):
    """Retrieve passages for `question` (plus any `extra_queries`, e.g. translations)
    and wrap it into a RAG prompt.

    Returns (send_text, info) where `info` is a short human status line (or None).
    Pure and Qt-free so it can run on a worker thread; never raises into the caller
    in a way that would block a turn — on any retrieval miss it returns the original
    question unchanged with an explanatory info line.
    """
    if lib.is_empty():
        return question, "📚 The database is empty — answering without it. Add documents in the Database tab and Build."
    k = k or int(_cfg("LIBRARY_RETRIEVE_K", DEFAULT_K))
    queries = [question] + [q for q in (extra_queries or []) if q and q.strip()]
    results = lib.retrieve_multi(queries, k=k) if len(queries) > 1 else lib.retrieve(question, k=k)
    if not results:
        # A small library is its own best retrieval. "можно ли с собакой?"
        # against a one-page contract that says "домашних животных" was an
        # FTS miss (no embeddings under tests), the question went to the
        # model bare, and it answered from whatever else was in the chat
        # (live 2026-09-13, mega journey). When the whole library fits the
        # passage budget, hand it over instead of nothing.
        results = _whole_small_library(lib)
        if not results:
            return question, "📚 No relevant passages found in the database for this query."
    # Size the char budget to the requested passage count so the slider actually controls
    # how many passages are injected (the static default would otherwise truncate large k).
    budget = min(_MAX_CONTEXT_CHARS,
                 max(int(_cfg("LIBRARY_CONTEXT_CHARS", DEFAULT_CONTEXT_CHARS)),
                     len(results) * _CHARS_PER_PASSAGE))
    context = Library.format_context(results, max_chars=budget)
    srcs = ", ".join(sorted({(r.title or os.path.basename(r.path) or "document")
                             for r in results}))
    prompt = (
        RAG_HEAD + " If the question is ABOUT "
        "the documents (their contents, terms, numbers, people), answer from the passages "
        "only — do not use web search or outside knowledge for such claims.\n"
        "Rules:\n"
        "- If the question is about the documents but the passages do not contain the "
        "answer, say plainly that the documents do not specify it. Do NOT guess or fill "
        "in from general knowledge.\n"
        "- If the question is NOT about the documents at all (the passages merely happen to "
        "share a word with it — e.g. a general-knowledge, date, weather or tool request), "
        "ignore the passages and answer normally, using tools when appropriate; do not "
        "say the documents are silent on it.\n"
        "- Make only claims directly supported by the passages; do not invent specifics "
        "(names, numbers, colours, dates) that are not present in them.\n"
        "- If the question asks for something exhaustive that the passages only partly "
        "cover, answer for what is present and state that the rest is not specified.\n\n"
        f"=== Retrieved passages ===\n{context}\n=== End passages ==="
        f"{RAG_QUESTION_SEAM}{question}")
    return prompt, f"📚 Using {len(results)} passage(s) from: {srcs}"


def default_library(db_path: Optional[Path] = None) -> Library:
    return Library(db_path)

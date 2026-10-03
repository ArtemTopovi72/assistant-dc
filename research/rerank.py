"""Pluggable second-stage reranker for the deep-research pipeline.

WHERE THIS SITS: after retrieval + dedupe, BEFORE per-source briefing. The LLM
only briefs the top-K *reranked* pages, so synthesis reads the strongest
evidence instead of the full raw retrieval set (spec #3 — "the final LLM should
only read the top-ranked sources, not the full raw retrieval set").

Relevance (this module) is kept SEPARATE from source scoring (authority/trust,
which lives in deep_research). The two only meet at the very end, where the
final order FUSES semantic relevance with caller-supplied authority via
Reciprocal Rank Fusion — so neither signal alone dominates and either can be
tuned without touching the other.

Backends, resolved in priority order (first available wins), all swappable:
  1. "cross"   — a cross-encoder reranker (BGE: BAAI/bge-reranker-* via
                 sentence_transformers.CrossEncoder) if installed. True
                 (query, passage) relevance; the spec's preferred option.
  2. "dense"   — cosine over the BGE-M3 embeddings already served by LM Studio
                 (`knowledge.Embedder`). No new dependency, no resident VRAM.
  3. "lexical" — deterministic IDF-weighted token overlap (BM25-lite). Always
                 available; also the offline/regression backend.

Nothing here raises into the pipeline: every backend degrades to the next on
failure, and the whole stage is a no-op pass-through if it cannot score.
"""
from __future__ import annotations

import logging
import math
import os
import re
import time
from typing import Callable, List, Optional, Sequence

logger = logging.getLogger("assistant.rerank")

# Device: CPU reranking is painfully slow (~64s for 14 pairs on this box), so prefer
# CUDA when a GPU is actually present — the bge-reranker (~2.3GB) scores in well under
# a second there. Falls back to CPU when CUDA is unavailable. Override either way with
# DR_RERANK_DEVICE=cpu|cuda (explicit env always wins).


def _default_rerank_device() -> str:
    dev = os.getenv("DR_RERANK_DEVICE")
    if dev:
        return dev
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


_RERANK_DEVICE = _default_rerank_device()

# A live research run must NOT block on a multi-GB HuggingFace download. By default
# the cross-encoder is used ONLY if it is already cached; otherwise we degrade to
# dense/lexical (no download) and tell the user how to pre-fetch it. Set
# DR_RERANK_ALLOW_DOWNLOAD=1 to permit the one-time ~2.3GB download mid-run.
_ALLOW_DOWNLOAD = os.getenv("DR_RERANK_ALLOW_DOWNLOAD", "0").lower() in ("1", "true", "yes", "on")


def _model_is_cached(repo_id: str) -> bool:
    """True if the cross-encoder snapshot is already in the local HF cache, so
    instantiating it won't trigger a network download."""
    try:
        from huggingface_hub import try_to_load_from_cache
        for fname in ("config.json", "model.safetensors", "pytorch_model.bin"):
            path = try_to_load_from_cache(repo_id, fname)
            if isinstance(path, str):
                return True
        return False
    except Exception:
        return False


_TOKEN_RE = re.compile(r"[a-zA-Zа-яёА-ЯЁ0-9]+")
_RRF_K = 60  # standard Reciprocal Rank Fusion constant (matches knowledge.py)


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return num / (na * nb)


class Reranker:
    """Scores passages against a query. Backend is chosen once, lazily.

    Construct with backend="auto" (default) to probe cross → dense → lexical, or
    pin a specific backend for tests/benchmarks. `cross_model` selects the
    cross-encoder checkpoint so the model can be swapped without code changes.
    """

    def __init__(self, backend: str = "auto", *, embedder=None,
                 cross_model: str = "BAAI/bge-reranker-v2-m3"):
        self.requested = backend
        self.cross_model = cross_model
        self._embedder = embedder
        self._ce = None                 # lazy CrossEncoder instance
        self.backend = self._resolve(backend)

    # -- backend resolution -------------------------------------------------- #
    def _resolve(self, backend: str) -> str:
        # Explicit "cross" must ACTUALLY load the cross-encoder (it used to return
        # "cross" while leaving self._ce None → relevance() silently fell back to
        # lexical, so "cross" was a no-op). Now it loads it or degrades cleanly.
        if backend == "cross":
            if self._try_cross():
                return "cross"
            logger.warning("cross-encoder requested but unavailable; using %s",
                           "dense" if self._embedder_available() else "lexical")
            return "dense" if self._embedder_available() else "lexical"
        if backend in ("dense", "lexical"):
            return backend
        # auto: prefer the strongest backend that is actually usable.
        if self._try_cross():
            return "cross"
        if self._embedder_available():
            return "dense"
        return "lexical"

    def _try_cross(self) -> bool:
        try:
            from sentence_transformers import CrossEncoder  # noqa: F401
        except Exception:
            return False
        # Don't hang a live run on a 2.3GB download. If the model isn't cached and
        # the user hasn't opted in, degrade cleanly instead of blocking (this was
        # the "infinite reranking / dead loop with no logs").
        if not _ALLOW_DOWNLOAD and not _model_is_cached(self.cross_model):
            logger.warning(
                "cross-encoder %s is NOT cached; skipping it to avoid a ~2.3GB "
                "download mid-run. Falling back to dense/lexical. To use it, pre-fetch "
                "the model or set DR_RERANK_ALLOW_DOWNLOAD=1.", self.cross_model)
            return False
        try:
            # Loud + timed: the first call may download ~2.3GB and the load can
            # stall under VRAM pressure — without this log the stage looked like an
            # "infinite reranking" hang with no output. CPU by default (see top).
            logger.info("loading cross-encoder %s on %s (first use may download ~2.3GB)…",
                        self.cross_model, _RERANK_DEVICE)
            t0 = time.time()
            self._ce = CrossEncoder(self.cross_model, device=_RERANK_DEVICE)
            logger.info("cross-encoder ready in %.1fs (device=%s)", time.time() - t0, _RERANK_DEVICE)
            return True
        except Exception as exc:
            logger.info("cross-encoder %s unavailable: %s", self.cross_model, exc)
            self._ce = None
            return False

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from knowledge import Embedder
                self._embedder = Embedder()
            except Exception as exc:
                logger.info("dense backend: no embedder: %s", exc)
                self._embedder = False  # sentinel: tried and failed
        return self._embedder or None

    def _embedder_available(self) -> bool:
        emb = self._get_embedder()
        try:
            return bool(emb) and emb.available()
        except Exception:
            return False

    # -- relevance scoring --------------------------------------------------- #
    def relevance(self, query: str, texts: Sequence[str]) -> List[float]:
        """Return one relevance score per text (higher = more relevant). Never
        raises; falls back to lexical if the chosen backend fails at call time."""
        if not texts:
            return []
        try:
            if self.backend == "cross" and self._ce is not None:
                return self._relevance_cross(query, texts)
            if self.backend == "dense":
                out = self._relevance_dense(query, texts)
                if out is not None:
                    return out
        except Exception as exc:
            logger.warning("rerank backend %s failed, falling back to lexical: %s",
                           self.backend, exc)
        return self._relevance_lexical(query, texts)

    def _relevance_cross(self, query: str, texts: Sequence[str]) -> List[float]:
        pairs = [(query, t[:2000]) for t in texts]
        logger.info("cross-encoder scoring %d pairs (device=%s)…", len(pairs), _RERANK_DEVICE)
        t0 = time.time()
        scores = self._ce.predict(pairs, batch_size=32, show_progress_bar=False)
        logger.info("cross-encoder scored %d pairs in %.1fs", len(pairs), time.time() - t0)
        return [float(s) for s in scores]

    def _relevance_dense(self, query: str, texts: Sequence[str]) -> Optional[List[float]]:
        emb = self._get_embedder()
        if not emb:
            return None
        vecs = emb.embed([query] + [t[:2000] for t in texts])
        if not vecs or len(vecs) != len(texts) + 1:
            return None
        qv, dvs = vecs[0], vecs[1:]
        return [_cosine(qv, dv) for dv in dvs]

    def _relevance_lexical(self, query: str, texts: Sequence[str]) -> List[float]:
        """IDF-weighted query-term coverage. Deterministic; query terms that are
        rare across the candidate set count for more (BM25-lite, no tuning)."""
        q_terms = set(_tokenize(query))
        if not q_terms:
            return [0.0] * len(texts)
        doc_tokens = [_tokenize(t) for t in texts]
        n = len(texts)
        df = {term: 0 for term in q_terms}
        for toks in doc_tokens:
            present = set(toks)
            for term in q_terms:
                if term in present:
                    df[term] += 1
        idf = {term: math.log(1.0 + n / (1 + df[term])) for term in q_terms}
        scores = []
        for toks in doc_tokens:
            if not toks:
                scores.append(0.0)
                continue
            counts = {}
            for tk in toks:
                if tk in q_terms:
                    counts[tk] = counts.get(tk, 0) + 1
            # saturating term frequency * idf, normalized by doc length.
            s = sum(idf[t] * (c / (c + 1.0)) for t, c in counts.items())
            scores.append(s / math.sqrt(len(toks)))
        return scores

    # -- public rerank ------------------------------------------------------- #
    def rerank(self, query: str, items: List[dict], *, top_k: Optional[int] = None,
               text_key: str = "text", authority_fn: Optional[Callable[[dict], float]] = None,
               authority_weight: float = 0.5) -> List[dict]:
        """Reorder `items` by fused relevance + authority and keep the top_k.

        Each returned item is annotated in place with `_rerank_relevance`,
        `_rerank_score` and `_rerank_rank` for observability. `authority_fn`
        maps an item to an authority score in [0,1]; when omitted, ordering is
        relevance-only. Fusion is RRF so the two scales never need calibrating.
        """
        if not items:
            return []
        texts = [str(it.get(text_key, "")) for it in items]
        rel = self.relevance(query, texts)
        for it, r in zip(items, rel):
            it["_rerank_relevance"] = round(float(r), 5)

        # Rank by relevance (desc). RRF needs ranks, not raw scores.
        rel_order = sorted(range(len(items)), key=lambda i: rel[i], reverse=True)
        rel_rank = {i: pos for pos, i in enumerate(rel_order)}

        if authority_fn is not None and authority_weight > 0:
            auth = [float(authority_fn(it)) for it in items]
            auth_order = sorted(range(len(items)), key=lambda i: auth[i], reverse=True)
            auth_rank = {i: pos for pos, i in enumerate(auth_order)}
        else:
            auth_rank = None

        fused = []
        for i, it in enumerate(items):
            s = 1.0 / (_RRF_K + rel_rank[i])
            if auth_rank is not None:
                s += authority_weight * (1.0 / (_RRF_K + auth_rank[i]))
            it["_rerank_score"] = round(s, 6)
            fused.append((s, i))
        fused.sort(key=lambda t: (-t[0], t[1]))

        ordered = [items[i] for _, i in fused]
        for pos, it in enumerate(ordered):
            it["_rerank_rank"] = pos
        if top_k is not None and top_k > 0:
            ordered = ordered[:top_k]
        logger.info("rerank[%s]: %d -> %d (q=%r)", self.backend, len(items),
                    len(ordered), (query or "")[:60])
        return ordered


# Module-level convenience: one reranker per process (cross-encoder load is heavy).
_DEFAULT: Optional[Reranker] = None


def get_reranker(backend: str = "auto", *, embedder=None,
                 cross_model: str = "BAAI/bge-reranker-v2-m3") -> Reranker:
    global _DEFAULT
    if backend != "auto" or embedder is not None:
        return Reranker(backend, embedder=embedder, cross_model=cross_model)
    if _DEFAULT is None:
        _DEFAULT = Reranker("auto", cross_model=cross_model)
    return _DEFAULT


def rerank_pages(query: str, pages: List[dict], *, top_k: Optional[int] = None,
                 text_key: str = "text",
                 authority_fn: Optional[Callable[[dict], float]] = None,
                 authority_weight: float = 0.5, backend: str = "auto",
                 embedder=None) -> List[dict]:
    """Convenience wrapper used by the research pipeline. Pure pass-through if
    reranking can't run, so callers never need a guard."""
    try:
        rr = get_reranker(backend, embedder=embedder)
        return rr.rerank(query, pages, top_k=top_k, text_key=text_key,
                         authority_fn=authority_fn, authority_weight=authority_weight)
    except Exception as exc:
        logger.warning("rerank_pages failed (%s); keeping original order", exc)
        return pages[:top_k] if (top_k and top_k > 0) else pages

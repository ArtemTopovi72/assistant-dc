"""Just-in-time fact selection: the saved facts THIS turn needs, not all of them.

Sending every pinned fact (up to 40) on every turn is mostly noise: most turns
need one or two, and the rest the model reads as standing instructions, the
same way an NPC's summarised joke turned into a rule it kept obeying.

Selection, no LLM on the path:
  * a small store (<= JIT_FACTS_ALL) goes in whole: nothing to gain;
  * otherwise the facts nearest the message (BGE-M3 cosine, floor + top-k),
    plus the newest few: a fact said a minute ago is context whatever the words;
  * no embedder -> every fact, i.e. no selection at all. Selection may
    only ever drop noise, never be the reason a known fact went missing.
Original order is kept, so "newest are last" in the prompt still holds.
"""
import logging
import math
import os
from typing import List, Sequence

logger = logging.getLogger("assistant." + __name__)

_EMBEDDER = None
_CACHE: dict = {}          # embedding source identity, fact text -> unit vector


def _cfg(name, default):
    return type(default)(os.getenv(name, default))


def enabled() -> bool:
    v = os.getenv("JIT_FACTS")
    if v is not None:
        return v.strip() == "1"
    return not os.getenv("F5_TEST_RUN")


def _unit(v):
    if (not isinstance(v, (list, tuple)) or not v
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in v)):
        raise ValueError('Invalid embedding vector')
    n = sum(x * x for x in v) ** 0.5
    if not n or not math.isfinite(n):
        raise ValueError('Invalid embedding norm')
    return [x / n for x in v]


def _embed(texts: Sequence[str]):
    global _EMBEDDER
    if _EMBEDDER is None:
        import knowledge
        _EMBEDDER = knowledge.Embedder(timeout=10.0)
    return _EMBEDDER.embed(list(texts))


def select(facts: List[str], query: str, *, k: int = None, newest: int = None,
           floor: float = None, keep_all_upto: int = None, embed=None) -> List[str]:
    k = _cfg("JIT_FACTS_K", 6) if k is None else k
    newest = _cfg("JIT_FACTS_NEWEST", 2) if newest is None else newest
    floor = _cfg("JIT_FACTS_FLOOR", 0.35) if floor is None else floor
    keep_all_upto = _cfg("JIT_FACTS_ALL", 8) if keep_all_upto is None else keep_all_upto
    if len(facts) <= keep_all_upto or not (query or "").strip():
        return list(facts)
    try:
        q = (embed or _embed)([query])
        if not q:
            return list(facts)
        if len(q) != 1:
            raise ValueError('Mismatched query embeddings')
        q = _unit(q[0])
        owner = getattr(embed, '__self__', embed) if embed is not None else _EMBEDDER
        identity = (getattr(owner, 'base_url', None), getattr(owner, 'model', None))
        if identity == (None, None):
            identity = embed
        keys = {f: (identity, f) for f in dict.fromkeys(facts)}
        local = {}
        absent = object()
        try:
            for fact, key in keys.items():
                cached = _CACHE.get(key, absent)
                if cached is not absent:
                    vec = _unit(cached)
                    if len(vec) != len(q):
                        raise ValueError('Incompatible cached embeddings')
                    local[fact] = vec
        except (TypeError, ValueError, OverflowError):
            for key in keys.values():
                _CACHE.pop(key, None)
            return list(facts)
        missing = [f for f in keys if f not in local]
        if missing:
            vecs = (embed or _embed)(missing)
            if not vecs:
                return list(facts)
            if len(vecs) != len(missing):
                raise ValueError('Mismatched fact embeddings')
            vecs = [_unit(v) for v in vecs]
            if any(len(v) != len(q) for v in vecs):
                raise ValueError('Incompatible fact embeddings')
            local.update(zip(missing, vecs))
            _CACHE.update((keys[f], v) for f, v in zip(missing, vecs))
        sims = [(sum(a * b for a, b in zip(q, local[f])), i) for i, f in enumerate(facts)]
    except Exception:
        logger.warning("jit facts: embedding failed -- keeping every fact", exc_info=True)
        return list(facts)
    keep = {i for s, i in sorted(sims, reverse=True)[:k] if s >= floor}
    keep |= set(range(max(0, len(facts) - newest), len(facts)))
    return [f for i, f in enumerate(facts) if i in keep]

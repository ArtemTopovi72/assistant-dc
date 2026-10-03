"""Just-in-time fact selection: the saved facts THIS turn needs, not all of them.

Pinned facts used to ride along in full on every turn (up to 40). Most turns
need one or two; the rest is noise the model reads as standing instructions,
the same way an NPC's summarised joke turned into a rule it kept obeying.

Selection, no LLM on the path:
  * a small store (<= JIT_FACTS_ALL) goes in whole: nothing to gain;
  * otherwise the facts nearest the message (BGE-M3 cosine, floor + top-k),
    plus the newest few: a fact said a minute ago is context whatever the words;
  * no embedder -> every fact, i.e. exactly the old behaviour. Selection may
    only ever drop noise, never be the reason a known fact went missing.
Original order is kept, so "newest are last" in the prompt still holds.
"""
import logging
import os
from typing import List, Sequence

logger = logging.getLogger(__name__)

_EMBEDDER = None
_CACHE: dict = {}          # fact text -> unit vector


def _cfg(name, default):
    return type(default)(os.getenv(name, default))


def enabled() -> bool:
    v = os.getenv("JIT_FACTS")
    if v is not None:
        return v.strip() == "1"
    return not os.getenv("F5_TEST_RUN")


def _unit(v):
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


def _embed(texts: Sequence[str]):
    global _EMBEDDER
    if _EMBEDDER is None:
        import knowledge
        _EMBEDDER = knowledge.Embedder(timeout=10.0)
    vecs = _EMBEDDER.embed(list(texts))
    return [_unit(v) for v in vecs] if vecs else None


def select(facts: List[str], query: str, *, k: int = None, newest: int = None,
           floor: float = None, keep_all_upto: int = None) -> List[str]:
    k = _cfg("JIT_FACTS_K", 6) if k is None else k
    newest = _cfg("JIT_FACTS_NEWEST", 2) if newest is None else newest
    floor = _cfg("JIT_FACTS_FLOOR", 0.35) if floor is None else floor
    keep_all_upto = _cfg("JIT_FACTS_ALL", 8) if keep_all_upto is None else keep_all_upto
    if len(facts) <= keep_all_upto or not (query or "").strip():
        return list(facts)
    try:
        missing = [f for f in dict.fromkeys(facts) if f not in _CACHE]
        if missing:
            vecs = _embed(missing)
            if not vecs:
                return list(facts)
            _CACHE.update(zip(missing, vecs))
        q = _embed([query])
        if not q:
            return list(facts)
        q = q[0]
        sims = [(sum(a * b for a, b in zip(q, _CACHE[f])), i) for i, f in enumerate(facts)]
    except Exception:
        logger.warning("jit facts: embedding failed -- keeping every fact", exc_info=True)
        return list(facts)
    keep = {i for s, i in sorted(sims, reverse=True)[:k] if s >= floor}
    keep |= set(range(max(0, len(facts) - newest), len(facts)))
    return [f for i, f in enumerate(facts) if i in keep]

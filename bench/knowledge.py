"""Retrieval benchmark on the REAL research cache (knowledge.db).

Compares four retrieval methods on the same 1897-chunk / 126-doc corpus:
  1. keyword  — the EXISTING system retriever (memory_center.KeywordRetriever),
                applied per-chunk, as a faithful baseline of "current behaviour".
  2. fts      — SQLite FTS5 / BM25 only.
  3. semantic — embedding cosine only (LM Studio nomic).
  4. hybrid   — RRF fusion of FTS5 + cosine (knowledge.search mode='hybrid').

Ground truth is by TOPIC membership (a substring predicate over each document's
full text), so paraphrase/concept queries — whose wording does NOT contain the
topic keywords — still have correct relevance labels. That is exactly what
separates keyword/FTS (precision) from semantic (recall).

Metrics per query: Recall@10, Precision@10, MRR, latency(ms). Reports per-type
and overall averages. Uses the prebuilt memory/knowledge_test.db.
"""
import os, sys, time, statistics
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import knowledge
import memory_center

# DB and embedding model are overridable so the same corpus can be benchmarked
# under two embedding models (BENCH_DB=<copy> EMBED_MODEL=<other>) — the model
# name flows through config into knowledge.Embedder's default.
DB = Path(os.getenv("BENCH_DB", "")) or (_ROOT / "memory" / "knowledge_test.db")
K = 10

kb = knowledge.KnowledgeBase(DB, embedder=knowledge.Embedder())
print(f"# corpus : {DB.name}")
print(f"# model  : {kb.embedder.model}")
conn = kb._conn

# --- load full doc text (concatenate its chunks) for ground-truth labelling - #
doctext = {}
for r in conn.execute("SELECT doc_id, group_concat(text,' ') t FROM chunks GROUP BY doc_id"):
    doctext[r["doc_id"]] = (r["t"] or "").lower()
all_docs = list(doctext.keys())


def gt(pred):
    """Ground-truth relevant doc ids: those whose full text satisfies pred()."""
    return {d for d in all_docs if pred(doctext[d])}


def has(*subs):
    return lambda t: all(s in t for s in subs)


def any_of(*subs):
    return lambda t: any(s in t for s in subs)


# (query, type, ground-truth predicate)
QUERIES = [
    ("Harris corner detector",                    "exact",      any_of("harris corner")),
    ("Wolfram Mathematica computation",           "exact",      any_of("wolfram")),
    ("Wildberries marketplace",                   "exact",      any_of("wildberries")),
    ("mechancal keybord revews",                  "fuzzy",      any_of("keyboard")),
    ("strucutre tensor anisitropic difusion",     "fuzzy",      any_of("structure tensor","diffusion tensor","anisotropic")),
    ("how to find edges and corners in pictures", "paraphrase", any_of("corner detection","edge detection","corner detector")),
    ("rules and laws governing artificial intelligence", "paraphrase", any_of("ai regulation","regulation of artificial intelligence","ai act","regulation")),
    ("online shopping store in russia",           "paraphrase", any_of("wildberries")),
    ("best clicky switches for fast typing",      "concept",    any_of("keyboard")),
    ("orientation estimation from image gradients","concept",   any_of("structure tensor","gradient","orientation")),
    ("tensor methods for medical MRI image analysis","cross",   any_of("diffusion tensor","biomedical","mri","dt-mri")),
    ("colorization of images using tensors",      "cross",      any_of("coloriz","color tensor","color image")),
]

# --- keyword baseline: current KeywordRetriever applied per chunk ----------- #
KW = memory_center.KeywordRetriever()
_chunks = [(r["id"], r["doc_id"], r["text"]) for r in
           conn.execute("SELECT id,doc_id,text FROM chunks")]


class _E:  # minimal shim with the attributes KeywordRetriever.score reads
    __slots__ = ("text", "importance")
    def __init__(self, t): self.text = t; self.importance = 50


def retrieve_keyword(q, k):
    scored = []
    for cid, did, txt in _chunks:
        s, _ = KW.score(_E(txt), q)
        if s > 0:
            scored.append((s, cid, did))
    scored.sort(key=lambda x: -x[0])
    return [(cid, did) for _, cid, did in scored[:k]]


def chunks_to_docs(pairs):
    """Top-k chunk results -> ranked unique doc ids (first occurrence wins)."""
    seen, out = set(), []
    for cid, did in pairs:
        if did not in seen:
            seen.add(did); out.append(did)
    return out


def method(name, q, k):
    t = time.time()
    if name == "keyword":
        pairs = retrieve_keyword(q, k)
    else:
        res = kb.search(q, k=k, mode=name)
        pairs = [(r.chunk_id, r.doc_id) for r in res]
    dt = (time.time() - t) * 1000
    return chunks_to_docs(pairs), dt


def metrics(ranked_docs, relevant):
    if not relevant:
        return None
    topk = ranked_docs[:K]
    hits = [d for d in topk if d in relevant]
    recall = len(set(topk) & relevant) / min(len(relevant), K)
    precision = len(hits) / K
    mrr = 0.0
    for i, d in enumerate(ranked_docs):
        if d in relevant:
            mrr = 1.0 / (i + 1); break
    return recall, precision, mrr


METHODS = ["keyword", "fts", "semantic", "hybrid"]
agg = {m: {"recall": [], "precision": [], "mrr": [], "lat": []} for m in METHODS}
bytype = {}

print(f"Corpus: {len(all_docs)} docs, {len(_chunks)} chunks | k={K}\n")
hdr = f"{'query':42} {'type':10} " + " ".join(f"{m:>20}" for m in METHODS)
print(hdr); print("-" * len(hdr))
for q, typ, pred in QUERIES:
    relevant = gt(pred)
    cells = []
    for m in METHODS:
        ranked, dt = method(m, q, K)
        r, p, mrr = metrics(ranked, relevant)
        agg[m]["recall"].append(r); agg[m]["precision"].append(p)
        agg[m]["mrr"].append(mrr); agg[m]["lat"].append(dt)
        bytype.setdefault(typ, {mm: [] for mm in METHODS})[m].append((r, mrr))
        cells.append(f"R{r:.2f} M{mrr:.2f} {dt:4.0f}ms")
    print(f"{q[:42]:42} {typ:10} " + " ".join(f"{c:>20}" for c in cells))
    print(f"{'  relevant docs in corpus: '+str(len(relevant)):42}")

print("\n==== OVERALL AVERAGES ====")
print(f"{'method':10} {'Recall@10':>10} {'Prec@10':>9} {'MRR':>7} {'med.lat(ms)':>12}")
for m in METHODS:
    a = agg[m]
    print(f"{m:10} {statistics.mean(a['recall']):>10.3f} {statistics.mean(a['precision']):>9.3f} "
          f"{statistics.mean(a['mrr']):>7.3f} {statistics.median(a['lat']):>12.0f}")

print("\n==== RECALL@10 BY QUERY TYPE ====")
print(f"{'type':12} " + " ".join(f"{m:>10}" for m in METHODS))
for typ, d in bytype.items():
    row = [statistics.mean([x[0] for x in d[m]]) for m in METHODS]
    print(f"{typ:12} " + " ".join(f"{v:>10.3f}" for v in row))

kb.close()

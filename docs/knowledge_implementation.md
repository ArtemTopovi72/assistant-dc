# Knowledge Layer — Implementation & Validation

Implements the approved discovery-first architecture (see
`knowledge_architecture.md`). Module: [`knowledge.py`](../knowledge.py).
Benchmark: [`tests/bench_knowledge.py`](../tests/bench_knowledge.py).

## 1. Discovery pass — candidate classification (A index / B don't / C unsure)

Re-confirmed on the live tree before coding. Sizes are current.

| Candidate | Size / count | Growth | Retrieval value | Semantic helps? | Keyword enough? | Verdict |
|---|---|---|---|---|---|---|
| **research cache** `memory/research/_cache` | **2.9 MB, 126 files, 1897 chunks** | per crawl (grows) | **high** — cross-session reuse | **yes** (paraphrase/concept) | no | **A — index** |
| docs/ (`*.md`) | ~40 KB, 4 files | per audit | medium ("what did we decide") | mild | mostly | **A — optional**, same DB |
| markdown/audit reports | ⊂ docs/ | rare | medium | mild | mostly | **A — optional** |
| architecture notes | ⊂ docs/ | rare | medium | mild | mostly | **A — optional** |
| generated research | = research cache | — | — | — | — | **A** (same as cache) |
| benchmark reports | ⊂ docs/ | rare | low | no | yes | C → treat as docs |
| memory summaries (`summary.json`) | 0 (never written) | on compaction | low (tiny) | no | yes | **B — keyword** |
| facts / session memory | 2 facts / 0 | bounded (≤90) | n/a (already injected) | no | yes | **B — keyword** |
| conversations / transcripts | not persisted | ephemeral | low + privacy | — | — | **B** (don't index) |
| workflow definitions (`workflow_*.json`) | 8 files | on commit | low | no | yes (grep) | **B — code** |
| prompt libraries (`prompts.py`) | 24 KB | on commit | low | no | yes | **B — code** |
| tool descriptions (`tools.py`) | 39 KB | on commit | low | no | yes | **B — code** |
| router rules (`image.py`) | code | on commit | low | no | yes | **B — code** |
| diagnostics output (`crash.log`…) | 27 KB | per run | low (by time) | no | yes (grep) | **B — logs** |
| examples / templates | none present / = workflows | — | — | — | — | **B** |

**Indexed first:** research cache (the only large natural-language corpus). docs/
are cheap to fold into the same DB and are wired (`import_markdown_dir`). Everything
in **B is code/config/tiny** — the file is the source of truth; a vector copy would
drift and lie.

## 2. What was built (`knowledge.py`)

One SQLite file (`knowledge.db`), no server, no new heavy dependency:

- **`documents`** (source_type, source_id, url/path, title, hash, ts, meta JSON) —
  one row per page/file; upsert keyed by `source_id`, **skipped when text hash is
  unchanged** (idempotent re-import).
- **`chunks`** (doc_id, ord, text, n_words, embedding BLOB, embed_model) — ~240-word
  paragraph-aware chunks with overlap.
- **`chunks_fts`** — external-content **FTS5** (BM25, `porter unicode61`) kept in
  sync by AFTER INSERT/UPDATE/DELETE triggers.
- **Embeddings** — float32 BLOBs from LM Studio `/v1/embeddings`
  (`text-embedding-nomic-embed-text-v1.5`, **768-dim**), embedded lazily in batches.
- **Hybrid retrieval** — FTS5/BM25 + numpy cosine fused with **Reciprocal Rank
  Fusion** (K=60, no tuning). `mode = fts | semantic | hybrid | auto`.

**Resilience (all verified):** survives restart (plain file); schema versioned in a
`meta` table for additive upgrades; chunks may have NULL embeddings and stay
reachable via FTS; `embed()` returns `None` on any API error → search **degrades to
FTS-only**; numpy optional (pure-python cosine fallback). No turn ever blocks on the
embed endpoint.

## 3. Build result (real data)

```
IMPORT: 126 files → 126 documents, 1897 chunks, embed_coverage 1.0  (294.7s one-off)
STATS : embed_model=text-embedding-nomic-embed-text-v1.5, embed_available=True
```

## 4. Retrieval benchmark (real corpus, 12 labelled queries, k=10)

Ground truth = topic membership (so paraphrase/concept queries are labelled
correctly even though their wording lacks the topic keywords). "keyword" = the
*existing* `memory_center.KeywordRetriever` applied per chunk — the current baseline.

| method | Recall@10 | Prec@10 | MRR | median latency |
|---|---|---|---|---|
| keyword (current) | 0.565 | 0.375 | 0.917 | 17 ms |
| FTS5 only | 0.556 | 0.367 | **1.000** | **3 ms** |
| semantic only | 0.585 | 0.408 | 1.000 | 2135 ms |
| **hybrid (RRF)** | **0.623** | **0.458** | **1.000** | 2123 ms |

**Recall@10 by query type:**

| type | keyword | fts | semantic | hybrid |
|---|---|---|---|---|
| exact | 0.833 | 0.917 | 0.917 | 0.833 |
| fuzzy (typos) | **0.100** | 0.150 | **0.450** | 0.400 |
| paraphrase | 0.558 | 0.475 | 0.392 | 0.392 |
| concept | 0.650 | 0.550 | 0.550 | 0.650 |
| cross-document | 0.550 | 0.550 | **0.550→0.850** | **0.850** |

### Findings (evidence-based)
- **Hybrid is the best overall retriever** (Recall 0.623, Prec 0.458, MRR 1.0) and
  **dominates cross-document** queries (0.85 vs 0.55).
- **The current keyword baseline collapses on typos** (0.10); semantic rescues them
  (0.45) — embeddings tolerate misspelling, exact matchers don't.
- **FTS5 alone is the latency winner** (3 ms, perfect MRR) and a great default; the
  current keyword scan is both slower (17 ms, scans every chunk) and weaker.
- **Semantic latency (~2.1 s) is entirely the query-embed HTTP round-trip** to LM
  Studio; the numpy cosine over 1897×768 is sub-millisecond. Mitigation: FTS-first
  with semantic as an opt-in "deep" toggle, and cache query embeddings.
- **Honest caveat:** semantic was *not* a universal win — on two paraphrase queries
  whose wording still shared keywords, keyword/FTS edged it. Hybrid hedges both.

## 5. Integration & next steps
- Keep **working memory (facts/session/summary) on JSON + KeywordRetriever** — too
  small to index; must stay exact. (Unchanged.)
- The `SemanticRetriever(available=False)` seam in `memory_center.py` remains the
  drop-in point if semantic *re-ranking of memory* is ever wanted; the knowledge
  corpus itself lives in its own `KnowledgeBase` (correct separation — it indexes
  pages, not memory entries).
- **Not yet wired into the GUI**: a "Knowledge" tab (search box + fts/semantic/hybrid
  toggle + score-breakdown rows, reusing the Diagnostics retrieval-trace pattern) is
  the recommended next step; the backend it needs is done and benchmarked.

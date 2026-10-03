# Knowledge & Retrieval Architecture — Discovery + Proposal

Discovery pass over the real project (not assumptions). Conclusion up front:
**do not add a vector database.** The injected-memory corpus is tiny; the one
genuinely large text corpus (the research cache) is the only thing that would
benefit from semantic retrieval, and that is best served by **SQLite + FTS5 +
an embedding column scored with numpy**, using the embedding model **already
present in LM Studio** (`text-embedding-nomic-embed-text-v1.5`). No server, no
new heavy dependency, no VRAM contention with the 12 GB RTX 3060.

---

## 1. Inventory of current knowledge sources (what actually exists)

| # | Source | Where | Stored as | Size now | Change rate |
|---|---|---|---|---|---|
| A | Pinned facts | `memory/<profile>/facts.json` | JSON list `{id,ts,text,importance,source,tags}` | **2 facts / 526 B** | per user "remember" (rare); cap 40 |
| B | Session memory | `memory/<profile>/session_memory.json` | JSON list `{ts,kind,text,meta}` | **`[]` (empty)** | per turn; `deque(maxlen=50)` |
| C | Summary | `memory/<profile>/summary.json` | JSON `{ts,text}` | **absent (never compacted)** | on compaction only |
| D | Memory Center sidecars | `memory/<profile>/.mc_meta/.mc_audit.jsonl/.mc_trash` | JSON / JSONL | **none yet** | on edit/delete/revision |
| E | Research page cache | `memory/research/_cache/*.json` | per-URL JSON (text+title+hash) | **3.6 MB**, files 60–350 KB | per crawl; TTL'd by `research_cache.py` |
| F | Transcription cache | `transcriptions.json` | dict hash→text | 721 B | per ASR of a new ref clip |
| G | Prompt templates | `prompts.py` | 12 Python string constants | 24 KB | only on commit |
| H | Tool descriptions | `tools.py` | `ToolSpec` + pydantic `*Args` Fields | 39 KB | only on commit |
| I | System/personality prompt | `prompts.py` (+ user personality files) | string consts / `.txt` | — | rare |
| J | Workflow definitions | `workflow_*.json`, `*.json` | ComfyUI graph JSON | 2–36 KB each (8 files) | only on commit |
| K | Crash/diagnostic logs | `crash.log`, `faulthandler.log` | append-only text | 27 KB | on crash / per run |
| L | Docs / reports | `docs/*.md` | Markdown | ~10 KB each | per audit |
| M | UI settings | `ui_settings.json` | JSON `{scale_mode,density,font_scale}` | 70 B | on settings change |
| N | Router / model descriptions | `image.py` `_INTENT_RULES`, `lmstudio.py` | compiled regex / code | — | only on commit |

Installed libs: `sqlite3`, `numpy`, `sklearn` ✅. Absent: chromadb, faiss,
sentence-transformers, sqlite-vec, lancedb, rank_bm25, whoosh.
LM Studio already serves an embeddings model (`text-embedding-nomic-embed-text-v1.5`).

---

## 2. Classification & per-source recommendation

**Keep structured, do NOT vectorize, do NOT full-text-index:**
- **A facts, B session, C summary** — at 2 facts / 0 session items with hard caps
  (50 + 40 ≈ 90 items max ever injected), this is *trivially* keyword/substring
  searchable in microseconds. The existing `KeywordRetriever` (token coverage +
  phrase + importance) is already more than enough. Embedding 90 short strings
  adds latency, a model round-trip, an index to keep in sync, and staleness risk
  for **zero** measured benefit. Keep JSON as source of truth.
- **F transcription cache, M UI settings** — pure key/value config. JSON. Leave alone.
- **D sidecars** — audit/trash/revisions. Append-only JSON/JSONL is correct;
  audit is a log, not a search corpus.

**Keep structured + add full-text (keyword) only — never vectorize:**
- **G prompts, H tool descriptions, J workflows, N router rules** — these are
  **code and config; the file is the source of truth.** Vectorizing them creates a
  second, drifting copy that goes stale the moment someone edits the file, and
  semantic search over your *own 12 tool specs* is pointless (you know their
  names). If you ever want a "which prompt/tool/workflow does X" lookup, a
  build-time keyword/grep index (or just `ripgrep`) is the right tool. Embeddings
  here are a net negative.
- **K logs, L docs** — grep/FTS at most. Logs are diagnostic, read by timestamp.

**The one real semantic-retrieval candidate:**
- **E research page cache (3.6 MB and growing)** — this is the only corpus that is
  (a) large, (b) natural language, (c) where a user query rarely matches exact
  keywords, and (d) valuable to reuse across sessions. Today it's treated as a
  *throwaway fetch-skipping cache* (keyed by URL, TTL'd). Promoting it to a
  queryable knowledge base is the highest-value retrieval upgrade in the project.
  Strategy: **hybrid** — FTS5/BM25 for precision + embeddings for recall, fused.

---

## 3. Surprising findings

1. **The memory system you built a whole Memory Center for currently holds 2
   facts and zero session items.** The retrieval "problem" is hypothetical at
   today's scale; the caps (50/40) mean it can never grow into a vector-DB
   problem. Effort spent vectorizing memory would be misallocated.
2. **Your largest text corpus is the research cache (3.6 MB), and it's treated as
   disposable.** That's the actual knowledge asset hiding in plain sight.
3. **You already have an embedding model in the stack** (LM Studio nomic-embed).
   Semantic search needs *no new dependency and no resident VRAM* — call
   `/v1/embeddings` on demand; nomic is ~140 MB and cheap.
4. **`memory_center.py` already designed for this**: `Retriever`/`KeywordRetriever`
   live, `SemanticRetriever(available=False)` stub, `search(semantic=...)`
   degrades safely. The seam to drop in embeddings exists.
5. **`summary.json` has never been written** — the compaction path is unexercised
   in practice, so "safe compaction" is currently theoretical.

---

## 4. Recommended architecture (from first principles)

### The problem, stated honestly
There are **two different problems** wearing one coat:
- **(P1) Working memory** — the ≤90 facts/notes injected into each LLM turn.
  Needs: correctness, recency/importance ranking, pinning, edit/audit, profile
  separation, safe compaction. Size is bounded and small. **Not a search-scale
  problem.**
- **(P2) Knowledge reuse** — the growing research corpus (and potentially docs).
  Needs: recall over paraphrased queries across sessions. **This is the retrieval
  problem.**

Conflating them into "put memory in a vector DB" solves a problem you don't have
(P1) while ignoring the one you do (P2).

### Recommended design
**Do NOT adopt ChromaDB / a standalone vector DB.** Reasons grounded in this repo:
single-user desktop app, 12 GB shared VRAM, everything else is flat files, and a
server adds ops/backup/version-skew burden (you already hit a comfy-aimdo version
drift). Instead:

**Layer 1 — Working memory (P1): keep JSON + keyword. Improve ranking, not storage.**
- Source of truth stays `facts.json` / `session_memory.json` / `summary.json`
  (human-readable, atomic writes, easy backup — all already done).
- Retrieval = existing `KeywordRetriever`, extended with recency + importance +
  pinned-always-injected. No embeddings needed at this scale. If you ever want
  semantic *re-ranking* of memory, gate it behind the existing
  `SemanticRetriever` seam and reuse Layer 2's embedding helper — but treat it as
  optional polish, not core.

**Layer 2 — Knowledge index (P2): one SQLite file, hybrid retrieval.**
- `knowledge.db` with:
  - a `documents` table (id, source_type, url/path, profile, title, text, hash,
    created, ttl, metadata JSON),
  - an **FTS5** virtual table over `title+text` (BM25 keyword/full-text, built in,
    zero deps),
  - an `embeddings` table: `doc_id → BLOB(float32[768])` from LM Studio nomic.
- Query = **hybrid fusion**: run FTS5 BM25 *and* cosine over embeddings (numpy on
  the candidate set — you don't need ANN for thousands of chunks), combine with
  Reciprocal Rank Fusion. FTS gives precision on names/numbers; embeddings give
  recall on paraphrase. RRF needs no tuning.
- Populate it from the **existing research cache** (E) on write — turn the
  throwaway cache into durable, queryable knowledge. Optionally also index docs (L).
- Chunk pages (~512–1000 tokens) before embedding; store chunk→doc backref.

Why SQLite+FTS5+numpy over a vector DB: single file (trivial backup/restore,
matches your existing JSON-file ethos), transactional, no server, FTS5 is
first-class, and at your scale (thousands→tens-of-thousands of chunks) a flat
numpy cosine over FTS-prefiltered candidates is *faster* than spinning up ANN and
far simpler than ChromaDB. If you ever exceed ~100k chunks, add `sqlite-vec`
(same file, drop-in ANN) — no migration.

### 5. What to vectorize first (shortlist, highest value)
1. **Research page cache (E)** — the only corpus where semantic recall pays off;
   already 3.6 MB and growing; reuse across sessions is real value.
2. *(maybe)* **Docs (L)** — small, but cheap to fold into the same index for a
   "what did we decide about X" lookup.
That's it. Everything else is too small, too structured, or code.

### 6. What must stay structured / NOT vectorized
- Facts / session / summary (A/B/C) — JSON + keyword; too small, must stay exact.
- Tool specs, prompts, workflows, router rules (G/H/J/N) — **code is the source of
  truth**; a vector copy drifts and lies.
- Logs, transcription cache, UI settings, audit/trash sidecars (K/F/M/D) —
  diagnostic/config/append-only; grep or direct read.

### 7. UI to manage it well
- The **Memory Center already covers P1** (explorer, edit, pin, revisions, trash,
  compaction review, audit, diagnostics) — keep it; it's the right surface.
- Add for P2: a **Knowledge tab** — search box with a keyword/semantic/hybrid
  toggle, result list with source + snippet + score breakdown (BM25 vs cosine),
  "open source," TTL/refresh controls, and per-profile scoping. Reuse the
  Diagnostics "retrieval trace" pattern already built so users *see* why a result
  ranked where it did. An "index status" line (docs, chunks, embedding model,
  last build) belongs in the Overview card.

### 8. What to prototype first
1. A `knowledge.py` module: SQLite schema + FTS5 + an `embed(texts)` helper that
   calls LM Studio `/v1/embeddings` (graceful degrade to FTS-only if the embed
   model isn't loaded — mirrors the existing `SemanticRetriever.available` seam).
2. A one-shot importer that loads the existing `memory/research/_cache/*.json`
   into `knowledge.db` (chunk → FTS row → embedding).
3. A hybrid `search(query)` returning RRF-fused results; wire it behind
   `MemoryStore.search(semantic=True)` and a new Knowledge tab.
   Measure recall on ~10 real past research queries before building more.

### 9. Risks & tradeoffs
- **Embedding/source drift**: never treat the index as truth — always link back to
  the JSON/file/DB row; rebuildable from source. Store the source hash; re-embed
  on change.
- **Embedding model availability**: LM Studio may not have the embed model loaded
  (it competes for the same single-model-exclusive slot you use for the LLM).
  Mitigation: batch-embed lazily, cache vectors in the DB, and degrade to FTS-only
  when the embed endpoint is down — never block a turn on it.
- **VRAM**: don't add a resident sentence-transformers model; the 12 GB is already
  shared by LLM+Whisper+F5. On-demand nomic via LM Studio (or CPU) is the safe path.
- **Privacy/footprint**: research cache may hold scraped personal data; keep it
  per-profile, honor TTL, and make "purge knowledge" a first-class action.
- **Over-engineering**: the biggest risk is building retrieval infrastructure for
  P1, which doesn't need it. Resist. Ship Layer 2 for research; leave memory as JSON.

---

## One-line answer
Keep memory as JSON + keyword (it's tiny and must stay exact); build **one SQLite
`knowledge.db` with FTS5 + numpy-cosine hybrid retrieval** over the **research
cache**, embedded by the **nomic model already in LM Studio**. No ChromaDB, no
vector server, no new heavy dependency.

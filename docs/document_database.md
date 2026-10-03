# Document Database (chat with your own files)

Upload large text documents, build a searchable index, and ask questions answered
from their contents — without the document ever filling the model's context window.
Built for the "upload War & Peace (1,200 pages) → Build → ask about chapter X" use case.

## How to use
1. **Database tab** → **➕ Add files** — pick TXT, Markdown, PDF, or EPUB files.
2. **🛠 Build Database** — extracts text, chunks it, and builds the hybrid index
   (keyword + semantic). A progress bar shows extraction then embedding; it runs on a
   worker thread so the UI stays responsive. Re-building is idempotent (unchanged files
   are skipped).
3. Turn on **📚 Use Database** (left command panel, under Ultra Search).
4. Ask questions normally. Each question first retrieves the most relevant passages and
   the answer is generated from them. A system line shows which documents were used.

## Architecture

The retrieval engine is the project's existing hybrid index (`knowledge.KnowledgeBase`)
— **no ChromaDB / new vector DB was added** because that layer already does exactly the
requested pipeline and is battle-tested:

- **Chunking** — `knowledge.chunk_text` (~240-word passages, 40-word overlap).
- **BM25 / keyword** — SQLite **FTS5** (`chunks_fts`, porter+unicode61) for exact names,
  numbers, and phrases.
- **Semantic** — embeddings from the LM Studio `/v1/embeddings` endpoint
  (`text-embedding-nomic-embed-text-v1.5`, 768-d), stored as float32 BLOBs. No new heavy
  dependency, no resident VRAM.
- **Fusion** — **Reciprocal Rank Fusion** of the BM25 and cosine rankings (`mode="auto"`
  → hybrid when embeddings exist, else BM25-only).
- **Rerank** — `retrieve()` pulls a larger candidate pool (`max(k*4, 20)`) and reranks it
  down to the top-k. Default is a **deterministic lexical reranker** (query-term coverage +
  density; instant, no model, no VRAM — safe on a shared 12 GB GPU). An **opt-in
  cross-encoder** (`config.LIBRARY_CROSS_ENCODER` = a sentence-transformers CrossEncoder
  model name) jointly scores (query, passage) when configured; it loads lazily and **falls
  back to lexical on any import/load/scoring failure**, so retrieval never breaks. The
  hybrid rank is kept as a stable prior fused with the rerank score.

`library.py` adds only what `knowledge.py` lacked for this feature:

| Piece | Where |
|---|---|
| Text extraction (TXT/MD via read; PDF via `pypdf`; EPUB via stdlib `zipfile` + `bs4`, spine-ordered) | `library.extract_text` |
| Build with progress + cancel, per-file error isolation, own SQLite file `memory/library.db` | `library.Library.build` |
| Retrieve + format a numbered, source-labelled, char-budgeted context block | `library.Library.retrieve` / `.format_context` |
| Chunk-level progress + cancel during embedding | added params on `knowledge.KnowledgeBase.ensure_embeddings` (backward-compatible) |

GUI (`gui.py`): `LibraryBuildWorker` (QThread), `_build_database_tab` + `_db_*` handlers,
the `📚 Use Database` toggle (`_toggle_usedb`). RAG injection happens **inside
`RequestWorker`** (worker thread): when `use_db` is set, after the input text is known
(typed, or transcribed from voice) it calls `library.build_rag_prompt` to wrap the
question with the retrieved passages, sets that as the model's `user_input`, and emits an
`info` line shown via `_add_system`. Memory and the on-screen user bubble keep the
*original* text. Doing this in the worker means retrieval never blocks the UI and **typed
and voice queries are augmented uniformly** (VAD/Talk paths pass `use_db=self.usedb_on`).

## Closed-book: no web in database mode
A "Use Database" question must be answered from the user's documents, not the internet.
So a DB-mode turn is **closed-book**: `RequestWorker` sets `ctx.web_search_enabled = False`
for that turn only (then restores the user's setting), which makes the agent loop withhold
the `search` and `find_photo` tools (graph.py). The RAG prompt also forbids outside
knowledge and tells the model to say "the documents do not specify it" rather than guess —
fixing an observed case where the model web-searched and fabricated specifics (eye colours
of *War and Peace* characters that Tolstoy never states) instead of answering from the PDF.

## Build speed (embedding throughput)
Embeddings are computed by the GPU-backed LM Studio `/v1/embeddings` endpoint, not on the
local CPU — the build's only local cost is HTTP round-trip latency. So the build embeds
with **bigger batches and concurrent in-flight requests** to keep that GPU fed instead of
idling between sequential calls: `KnowledgeBase.ensure_embeddings(batch=, workers=)`,
driven for the library by `LIBRARY_EMBED_BATCH` (default 128) and `LIBRARY_EMBED_WORKERS`
(default 4). Requests run in waves of `workers` (bounded in-flight, cancel checked between
waves); SQLite writes stay on the calling thread. Set `LIBRARY_EMBED_WORKERS=1` to force
the old sequential behaviour.

**Sweet spot (measured, 300 real chunks vs live LM Studio + RTX 3060):** throughput
saturates at **~15.6 chunks/s** — the endpoint is GPU-bound, so beyond a point more
concurrency/batch doesn't help. `batch=16/workers=1` = 4.9 ch/s; the **default
batch=128/workers=4 ≈ 15.5 ch/s (3.2× faster)**, matching the plateau with only 4
concurrent connections. Going to workers=8–12 or batch=256 gains nothing measurable and
just loads the server harder.

## Robustness / determinism
- **Concurrency / "database is locked"** — every `KnowledgeBase` connection sets
  `busy_timeout=30s` (wait, don't fail, when another connection holds the file), and the
  GUI releases its reader connection before a build so the build worker owns the DB.
- **Foreign keys + orphan repair** — connections set `PRAGMA foreign_keys=ON` so deleting a
  document actually cascades to its chunks (without it SQLite silently orphaned them — the
  "0 documents · N passages" bug). `KnowledgeBase.repair()` deletes any pre-existing orphan
  chunks and rebuilds the FTS index; it runs at the start of every build, so a corrupted DB
  self-heals on the next Build.
- **Embedding model offline** → build and search both fall back to BM25-only; the tab
  says so and a later rebuild adds embeddings. No turn ever blocks on the embed endpoint.
- **Bad / unreadable file** → recorded per-file in `errors`, the rest of the build proceeds.
- **Threading** — the build worker uses its own SQLite connection (connections can't cross
  threads); the GUI reader connection is dropped after a build so the next query sees the
  new data (WAL).
- **Retrieval never breaks a turn** — any failure in `_augment_with_database` falls back to
  sending the original question unchanged.
- **Context budget** — `format_context` truncates to a character budget, so a huge corpus
  can never blow the context window.

## Tests
`tests/test_library.py` — **18/18**, runs with the embedder disabled (no LM Studio):
extraction across TXT/MD/EPUB, unsupported-type rejection, build stats, the FTS-only
fallback path, idempotent rebuild, BM25 retrieval relevance, context formatting + budget,
introspection, and purge. The Database tab + RAG injection are additionally validated
headless (offscreen Qt): tab construction, retrieval block assembly, and the empty-DB
passthrough.

## Recall for big books
Three layers improve recall on large documents:
1. **Passages-per-question slider** — the Database tab has a *"Passages per question"*
   slider (1–100, **default 25**) controlling how many retrieved passages are fed to the
   model per Use-Database question. The value flows through `RequestWorker(db_k=)` →
   `build_rag_prompt(k=)`; the context budget auto-sizes to the chosen count
   (`_CHARS_PER_PASSAGE`, capped at `_MAX_CONTEXT_CHARS`) so the slider really controls how
   many passages are injected, not just retrieved. `LIBRARY_RETRIEVE_K` (default 25) is the
   non-GUI fallback.
2. **Cross-lingual query** — `library.cross_lingual_targets` compares the corpus script(s)
   (`corpus_scripts`) to the question's; if they differ (e.g. Russian book, English
   question), `RequestWorker._translate_query` translates the query into the corpus
   language(s) via the chat model and `retrieve_multi` searches all variants and merges.
   This fixes the BM25 half being useless across languages ("eyes" ≠ "глаза").
3. **Whole-book scan** — the *"Whole-book scan for the next question"* checkbox in the
   Database tab. With Use Database ON, the question routes to `ScanWorker`: a map-reduce
   that reads EVERY chunk in batches (`library.map_reduce_scan`, `LIBRARY_SCAN_BATCH_CHARS`
   default 12000), extracts relevant facts per batch, then synthesizes one answer. Complete
   but slow (many model calls); cancellable via Stop. Use it for exhaustive "find all X"
   questions that top-k retrieval can't answer.

## Limitations / future work
- Whole-book scan cost scales with book size (one model call per ~12k-char batch); a
  1,200-page book is many calls (minutes). It's deliberately opt-in per question.
- The cross-encoder reranker is opt-in (off by default for VRAM safety); enabling it loads
  a model. The default lexical reranker is a heuristic, not a learned relevance model.
- The database is shared across memory profiles (one `library.db`); per-profile libraries
  are a possible extension.

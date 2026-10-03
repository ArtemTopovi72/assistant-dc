"""Re-embed every stored index after an EMBED_MODEL change.

Changing the embedding model invalidates every vector already on disk: a query
vector from the new model cannot be compared with passage vectors from the old
one. The code degrades safely on its own — `ensure_embeddings` treats a foreign
`embed_model` as work to do, and searches ignore vectors of a different
dimension — but until this runs, semantic recall is only as good as whatever has
already been re-embedded, and the rest of the corpus is reachable by FTS/BM25
alone.

Covers all three families of index in the project:
  * memory/knowledge.db          — the research knowledge layer
  * memory/*.db + library DBs    — the GUI Database tab
  * tg_libraries/lib_<chat>.db   — per-user Telegram document libraries

Usage
  python reindex_embeddings.py              # re-embed everything that is stale
  python reindex_embeddings.py --dry-run    # report what would be done
  python reindex_embeddings.py --path X.db  # one database only
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import knowledge

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("reindex")

BASE_DIR = Path(__file__).resolve().parents[1]


def discover() -> list[Path]:
    """Every SQLite file that carries a `chunks` table with embeddings."""
    seen: dict[Path, None] = {}
    candidates = list((BASE_DIR / "memory").glob("*.db"))
    candidates += list((BASE_DIR / "tg_libraries").glob("*.db"))
    candidates += list(BASE_DIR.glob("*.db"))
    out = []
    for p in candidates:
        p = p.resolve()
        if p in seen:
            continue
        seen[p] = None
        try:
            import sqlite3
            conn = sqlite3.connect(str(p))
            has = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks'"
            ).fetchone()
            conn.close()
            if has:
                out.append(p)
        except Exception:
            continue
    return out


def report(path: Path, kb) -> dict:
    st = kb.stats()
    logger.info("  chunks=%d embedded=%d current=%d stale=%d",
                st["chunks"], st["chunks_embedded"],
                st["embed_current"], st["embed_stale"])
    for model, n in sorted(st["embed_by_model"].items(), key=lambda kv: -kv[1]):
        mark = "  <- current" if model == st["embed_model"] else "  (stale)"
        logger.info("    %-42s %6d%s", model, n, mark)
    return st


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--path", action="append", default=[],
                    help="specific database(s); default: auto-discover")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    logger.info("Target model : %s (%d-dim) at %s",
                config.EMBED_MODEL, config.EMBED_DIM, config.EMBED_BASE)

    probe = knowledge.Embedder()
    if not probe.available():
        logger.error("Embedding endpoint is not answering for model %r — start "
                     "LM Studio and make sure that model is downloaded.",
                     config.EMBED_MODEL)
        return 2
    vec = probe.embed(["dimension probe"])
    got = len(vec[0]) if vec else 0
    if got != config.EMBED_DIM:
        logger.error("Model returns %d-dim vectors but EMBED_DIM=%d — fix config "
                     "before reindexing.", got, config.EMBED_DIM)
        return 2
    logger.info("Endpoint OK, %d-dim vectors.\n", got)

    paths = [Path(p).resolve() for p in args.path] or discover()
    if not paths:
        logger.info("No indexes found.")
        return 0

    total_re = 0
    for path in paths:
        logger.info("%s", path)
        kb = None
        try:
            kb = knowledge.KnowledgeBase(path)
            st = report(path, kb)
            if not st["embed_stale"] and st["chunks_embedded"] == st["chunks"]:
                logger.info("  already current — skipped\n")
                continue
            if args.dry_run:
                logger.info("  DRY RUN — would re-embed %d chunk(s)\n",
                            st["chunks"] - st["embed_current"])
                continue
            t0 = time.time()
            res = kb.reindex_embeddings(batch=args.batch, workers=args.workers)
            total_re += res["re_embedded"]
            logger.info("  re-embedded %d chunk(s) in %.1fs",
                        res["re_embedded"], time.time() - t0)
            after = res["after"]
            logger.info("  now: current=%d stale=%d coverage=%.1f%%\n",
                        after["embed_current"], after["embed_stale"],
                        100.0 * after["embed_coverage"])
        except Exception:
            logger.exception("  FAILED for %s\n", path)
        finally:
            if kb is not None:
                try: kb.close()
                except Exception: pass

    logger.info("Done. %d chunk(s) re-embedded.", total_re)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

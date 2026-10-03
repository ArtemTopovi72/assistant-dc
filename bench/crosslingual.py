"""Cross-lingual retrieval A/B: RUSSIAN queries against an ENGLISH corpus.

This is the case the embedding switch was made for. The monolingual benchmark
(bench_knowledge.py) cannot show it: both models were trained on English, so
English-to-English retrieval flatters the weaker one.

Ground truth reuses bench_knowledge's predicate style — a document is relevant
if its full text contains one of the given substrings — so the labels are
language-independent and identical for both models.

Usage:
  EMBED_MODEL=... BENCH_DB=... python bench/crosslingual.py
"""
import os
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import knowledge

DB = Path(os.getenv("BENCH_DB", "")) or (_ROOT / "memory" / "knowledge_test.db")
K = 10

kb = knowledge.KnowledgeBase(DB, embedder=knowledge.Embedder())
conn = kb._conn
print(f"# corpus : {DB.name}")
print(f"# model  : {kb.embedder.model}")

doctext = {}
for r in conn.execute("SELECT doc_id, group_concat(text,' ') t FROM chunks GROUP BY doc_id"):
    doctext[r["doc_id"]] = (r["t"] or "").lower()
all_docs = list(doctext.keys())


def any_of(*subs):
    return lambda t: any(s in t for s in subs)


# (russian query, english gloss, ground-truth predicate)
QUERIES = [
    ("детектор углов Харриса", "Harris corner detector",
     any_of("harris corner")),
    ("Вольфрам Математика вычисления", "Wolfram Mathematica computation",
     any_of("wolfram")),
    ("Вайлдберриз маркетплейс", "Wildberries marketplace",
     any_of("wildberries")),
    ("как найти края и углы на изображении", "how to find edges and corners",
     any_of("corner detection", "edge detection", "corner detector")),
    ("законы и правила регулирования искусственного интеллекта",
     "rules and laws governing AI",
     any_of("ai regulation", "regulation of artificial intelligence", "ai act", "regulation")),
    ("интернет-магазин в России", "online shopping store in russia",
     any_of("wildberries", "ozon", "marketplace")),
    ("лучшие механические переключатели для быстрой печати",
     "best clicky switches for fast typing",
     any_of("keyboard", "switch")),
    ("оценка ориентации по градиентам изображения",
     "orientation estimation from image gradients",
     any_of("structure tensor", "gradient", "orientation")),
    ("тензорные методы для медицинской МРТ",
     "tensor methods for medical MRI",
     any_of("diffusion tensor", "biomedical", "mri", "dt-mri")),
    ("раскрашивание изображений с помощью тензоров",
     "colorization of images using tensors",
     any_of("coloriz", "color tensor", "color image")),
]


def chunks_to_docs(pairs):
    """chunk hits -> ranked doc ids, best rank wins, order preserved."""
    seen, out = set(), []
    for cid, _score in pairs:
        row = conn.execute("SELECT doc_id FROM chunks WHERE id=?", (cid,)).fetchone()
        if row and row["doc_id"] not in seen:
            seen.add(row["doc_id"])
            out.append(row["doc_id"])
    return out


def metrics(ranked, relevant):
    if not relevant:
        return None
    top = ranked[:K]
    hits = [d for d in top if d in relevant]
    recall = len(hits) / min(len(relevant), K)
    prec = len(hits) / K
    mrr = 0.0
    for i, d in enumerate(top, 1):
        if d in relevant:
            mrr = 1.0 / i
            break
    return recall, prec, mrr


rows = []
print(f"\n{'russian query':<46} {'sem R@10':>9} {'P@10':>6} {'MRR':>6}  {'fts R@10':>9}")
print("-" * 84)
for ru, gloss, pred in QUERIES:
    relevant = {d for d in all_docs if pred(doctext[d])}
    sem = metrics(chunks_to_docs(kb.search_semantic(ru, k=60)), relevant)
    fts = metrics(chunks_to_docs(kb.search_fts(ru, k=60)), relevant)
    if sem is None:
        print(f"{ru[:44]:<46}  (no relevant docs — skipped)")
        continue
    rows.append((sem, fts))
    print(f"{ru[:44]:<46} {sem[0]:>9.2f} {sem[1]:>6.2f} {sem[2]:>6.2f}  {fts[0]:>9.2f}")

if rows:
    n = len(rows)
    print("-" * 84)
    print(f"{'AVERAGE (' + str(n) + ' queries)':<46} "
          f"{sum(r[0][0] for r in rows)/n:>9.3f} "
          f"{sum(r[0][1] for r in rows)/n:>6.3f} "
          f"{sum(r[0][2] for r in rows)/n:>6.3f}  "
          f"{sum(r[1][0] for r in rows)/n:>9.3f}")
kb.close()

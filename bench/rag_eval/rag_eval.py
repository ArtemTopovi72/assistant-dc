"""Document-library retrieval eval: no rerank vs lexical rerank vs cross-encoder.

Corpus: 8 ru-wiki articles (bench/rag_eval/corpus). For a stratified sample of
chunks the house model writes ONE question whose answer is only in that chunk
(cached in questions.json, so the LLM is needed once). Gold = that chunk's text.
Metrics: hit@1, hit@3, hit@5, MRR over the top-5 the library would hand the model.

    venv/Scripts/python bench/rag_eval/rag_eval.py [--n 80]
"""
import argparse, json, os, random, sys, tempfile, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import library as L

HERE = os.path.dirname(os.path.abspath(__file__))
QFILE = os.path.join(HERE, "questions.json")


def make_questions(lib, n):
    import llm
    chunks = [c for c in lib.all_chunks() if len(c) > 400]
    random.Random(7).shuffle(chunks)
    out = []
    for c in chunks:
        if len(out) >= n:
            break
        q = llm.call_llm_simple(
            None, "Ты составляешь вопросы для проверки поиска по документам.",
            "Прочитай фрагмент и задай ОДИН конкретный вопрос на русском, ответ на который "
            "есть ТОЛЬКО в этом фрагменте (факт, число, имя, дата). Перефразируй — не копируй "
            "слова фрагмента дословно. Выведи только вопрос.\n\nФрагмент:\n" + c[:1800],
            max_tokens=400, temperature=0.3)
        q = (q or "").strip().split("\n")[0]
        if q.endswith("?") and 15 < len(q) < 250:
            out.append({"q": q, "gold": c})
            print(len(out), q, flush=True)
    json.dump(out, open(QFILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return out


def score(lib, qs, mode):
    hits = {1: 0, 3: 0, 5: 0}
    mrr = 0.0
    t0 = time.time()
    for it in qs:
        if mode == "none":
            res = lib.retrieve(it["q"], k=5, rerank_results=False)
        else:
            res = lib.retrieve(it["q"], k=5)
        texts = [r.text for r in res]
        rank = next((i + 1 for i, t in enumerate(texts) if t.strip() == it["gold"].strip()
                     or it["gold"].strip()[:200] in t), None)
        for k in hits:
            hits[k] += bool(rank and rank <= k)
        mrr += (1.0 / rank) if rank else 0.0
    n = len(qs)
    return {f"hit@{k}": round(v / n, 3) for k, v in hits.items()} | {
        "mrr": round(mrr / n, 3), "ms_per_q": round(1000 * (time.time() - t0) / n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    a = ap.parse_args()
    db = Path(tempfile.gettempdir()) / "rag_eval_library.db"
    if db.exists():
        db.unlink()
    lib = L.Library(db)
    paths = sorted(str(p) for p in Path(HERE, "corpus").glob("*.txt"))
    print(lib.build(paths))
    qs = json.load(open(QFILE, encoding="utf-8")) if os.path.exists(QFILE) else make_questions(lib, a.n)
    print(f"{len(qs)} questions")
    res = {"none": score(lib, qs, "none"), "lexical": score(lib, qs, "lexical")}
    import config
    model = os.getenv("CE_MODEL", "BAAI/bge-reranker-v2-m3")
    lib._cross_encoder = L.load_cross_encoder(model)
    res["cross:" + model] = score(lib, qs, "cross") if lib._cross_encoder else {"error": "not loaded"}
    for k, v in res.items():
        print(f"{k:40s} {v}")
    json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

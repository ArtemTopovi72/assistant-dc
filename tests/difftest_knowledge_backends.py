"""Differential harness: the SAME knowledge operations through both backends.

WHY THIS EXISTS. Step 3 of the service extraction added a second implementation
of knowledge_api.KnowledgeClient. Two implementations of one contract rot apart
silently — the in-process one keeps working, the HTTP one drifts, and every
existing suite stays green because every existing suite runs in-process. This
harness runs the identical sequence of boundary operations against
InProcessKnowledgeClient and HttpKnowledgeClient over two freshly built,
identical indexes and asserts the results are equal, field by field.

Precedent: tests/difftest_refactor.py, which captures output for an external
diff across two checkouts. This one is self-checking instead, because both
implementations exist in the same tree and can be run side by side.

DETERMINISM. Embeddings are the only nondeterministic input (they need a live
LM Studio and a GPU). Both sides are therefore pointed at a dead embedding
endpoint, which knowledge.Embedder degrades to FTS-only — the documented
fallback path, deterministic, offline, and it touches no GPU. That means this
proves the two backends agree on ingest/BM25 retrieval/stats/delete, and does
NOT prove they agree on the embedding-backed half of hybrid search.

SAFETY. Every index lives under a fresh tempfile.mkdtemp(); no default DB path
is ever used, so this cannot see or touch a real user library. The service runs
on a NON-default port (8791, vs the 8790 default) and is shut down through its
/shutdown route, with terminate/kill as a backstop in a finally.

Run:  venv/Scripts/python.exe -u tests/difftest_knowledge_backends.py
Judge by EXIT CODE (0 = the backends agree).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import logging
logging.disable(logging.WARNING)

PORT = 8791                                   # NOT the 8790 default
DEAD_EMBED = "http://127.0.0.1:9"             # forces the FTS-only degrade path
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

_fails = []
_passes = 0


def check(name, cond, detail=""):
    global _passes
    if cond:
        _passes += 1
        print(f"[PASS] {name}")
    else:
        _fails.append(name)
        print(f"[FAIL] {name}  {detail}")


def same(name, a, b):
    check(name, a == b, f"\n  inproc: {json.dumps(a, ensure_ascii=False, default=str)[:400]}"
                        f"\n  http  : {json.dumps(b, ensure_ascii=False, default=str)[:400]}")


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
DOCS = {
    "alpha.txt": ("Alpha beta gamma delta. The quick brown fox jumps over the lazy dog.\n\n"
                  "Retrieval fuses lexical and dense ranks with reciprocal rank fusion.\n"),
    "bravo.md": ("# Bravo\n\nGamma delta epsilon zeta. Chunking splits on paragraph "
                 "boundaries where possible.\n\nThe lazy dog sleeps.\n"),
    "cyrillic.txt": ("Альфа бета гамма. Быстрая коричневая лиса прыгает через ленивую собаку.\n\n"
                     "Поиск объединяет лексический и плотный ранги.\n"),
}

QUERIES = ["lazy dog", "reciprocal rank fusion", "гамма", "nothing matches this at all"]


def write_fixtures(d):
    paths = []
    for name, body in DOCS.items():
        p = os.path.join(d, name)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(body)
        paths.append(p)
    return sorted(paths)


def hits_to_rows(hits):
    """Compare the fields the consumers actually read, plus the ids and score."""
    return [{"chunk_id": h.chunk_id, "doc_id": h.doc_id, "source_type": h.source_type,
             "title": h.title, "text": h.text, "score": round(float(h.score), 6),
             "bm25_rank": h.bm25_rank, "cosine_rank": h.cosine_rank, "why": h.why}
            for h in hits]


def scrub_docs(rows, root):
    """documents() returns absolute paths; the two runs index the same files from
    the same directory, so only the db file differs — but scrub anyway so a
    future change of layout cannot make this pass or fail for the wrong reason."""
    out = []
    for r in rows:
        r = dict(r)
        r["path"] = os.path.basename(r.get("path") or "")
        out.append(r)
    return sorted(out, key=lambda r: (r.get("title") or "", r["path"]))


def scrub_stats(st):
    """Drop keys that legitimately differ between two processes (e.g. which
    embed model each side thinks it has configured) — everything counted stays."""
    keep = ("documents", "chunks", "chunks_embedded", "embed_coverage", "by_source_type")
    return {k: st[k] for k in keep if k in st}


# --------------------------------------------------------------------------- #
# the operation script — one function, run twice
# --------------------------------------------------------------------------- #
def run_ops(client, paths, root):
    out = {}
    out["is_empty_before"] = client.is_empty()
    st = client.ingest(paths)
    out["ingest"] = {k: st.get(k) for k in
                     ("files", "documents_indexed", "skipped_unchanged",
                      "chunks_embedded", "errors")}
    out["ingest_stats"] = scrub_stats(st)
    out["is_empty_after"] = client.is_empty()
    out["documents"] = scrub_docs(client.documents(), root)
    out["stats"] = scrub_stats(client.stats())
    out["corpus_scripts"] = sorted(client.corpus_scripts())
    out["all_chunks"] = client.all_chunks()
    out["all_chunks_filtered"] = client.all_chunks([paths[0]])
    for q in QUERIES:
        out[f"search::{q}"] = hits_to_rows(client.search(q, k=3))
        out[f"search_norerank::{q}"] = hits_to_rows(
            client.search(q, k=3, rerank_results=False))
    out["search_multi"] = hits_to_rows(client.search_multi(QUERIES[:3], k=4))
    # re-ingest the same files: must be recognised as unchanged, not duplicated
    st2 = client.ingest(paths)
    out["reingest"] = {k: st2.get(k) for k in
                       ("files", "documents_indexed", "skipped_unchanged", "errors")}
    out["purge_all"] = client.purge_all()
    out["is_empty_after_purge"] = client.is_empty()
    out["documents_after_purge"] = client.documents()
    return out


# --------------------------------------------------------------------------- #
def main():
    root = tempfile.mkdtemp(prefix="kbdiff_")
    proc = None
    http_client = None
    try:
        paths = write_fixtures(root)

        # ---- in-process side -------------------------------------------- #
        os.environ.pop("KNOWLEDGE_BACKEND", None)
        import config
        config.EMBED_BASE = DEAD_EMBED          # deterministic FTS-only, offline
        import knowledge_client
        check("default backend is in-process", knowledge_client.backend_name() == "inprocess")
        ip = knowledge_client.open_library(os.path.join(root, "inproc.db"))
        check("in-process client type", type(ip).__name__ == "InProcessKnowledgeClient",
              type(ip).__name__)
        try:
            a = run_ops(ip, paths, root)
        finally:
            ip.close()

        # ---- HTTP side ---------------------------------------------------- #
        env = dict(os.environ)
        env["KNOWLEDGE_SERVICE_EMBED_BASE"] = DEAD_EMBED
        env["PYTHONPATH"] = os.pathsep.join([REPO, os.path.join(REPO, "core"), env.get("PYTHONPATH", "")])
        proc = subprocess.Popen([PY, "-u", os.path.join(REPO, "knowledge/knowledge_service.py"),
                                 "--port", str(PORT)],
                                cwd=REPO, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        os.environ["KNOWLEDGE_BACKEND"] = "http"
        os.environ["KNOWLEDGE_SERVICE_URL"] = f"http://127.0.0.1:{PORT}"
        check("env selects the http backend", knowledge_client.backend_name() == "http")

        import requests
        up = False
        for _ in range(120):                    # up to ~60s: first import pulls torch et al
            if proc.poll() is not None:
                break
            try:
                if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=1).json().get("ok"):
                    up = True
                    break
            except Exception:
                time.sleep(0.5)
        if not up:
            print(proc.stdout.read() if proc.stdout else "(no service output)")
            check("service started on the non-default port", False, f"port {PORT}")
            return 1
        check("service started on the non-default port", True)

        http_client = knowledge_client.open_library(os.path.join(root, "http.db"))
        check("http client type", type(http_client).__name__ == "HttpKnowledgeClient",
              type(http_client).__name__)
        b = run_ops(http_client, paths, root)

        # ---- compare ------------------------------------------------------ #
        # A comparison of two empty result sets agrees perfectly and proves
        # nothing. Refuse to be vacuous.
        check("fixtures actually indexed", a["ingest"]["documents_indexed"] == len(DOCS),
              str(a["ingest"]))
        check("retrieval actually returned passages",
              all(a[f"search::{q}"] for q in QUERIES[:3]),
              str({q: len(a[f"search::{q}"]) for q in QUERIES}))
        check("the no-match query really matched nothing",
              a["search::nothing matches this at all"] == [])
        check("re-ingest deduped instead of duplicating",
              a["reingest"]["documents_indexed"] == 0
              and a["reingest"]["skipped_unchanged"] == len(DOCS), str(a["reingest"]))

        same("operation set is identical", sorted(a), sorted(b))
        for key in sorted(a):
            same(f"agree: {key}", a[key], b.get(key))

    finally:
        try:
            if http_client is not None:
                http_client.close()
        except Exception:
            pass
        if proc is not None and proc.poll() is None:
            try:
                import requests
                requests.post(f"http://127.0.0.1:{PORT}/shutdown", timeout=5)
            except Exception:
                pass
            try:
                proc.wait(timeout=15)
            except Exception:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except Exception:
                    proc.kill()
        os.environ.pop("KNOWLEDGE_BACKEND", None)
        os.environ.pop("KNOWLEDGE_SERVICE_URL", None)
        shutil.rmtree(root, ignore_errors=True)

    print(f"\n{_passes} passed, {len(_fails)} failed")
    if _fails:
        print("FAILED: " + ", ".join(_fails))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

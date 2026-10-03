"""A document-backed turn: the entry translator must translate the user's
QUESTION only, never the RAG wrapper around it (live 2026-09-18 00:44 the
whole wrapper was 'translated' into an answer and the question vanished)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("F5_TEST_RUN", "1")
import library, graph_language as GL

PASSED = FAILED = 0
def check(name, cond, extra=""):
    global PASSED, FAILED
    if cond: PASSED += 1; print("PASS ", name)
    else: FAILED += 1; print("FAIL ", name, extra)

seen = []
GL.call_llm_simple = lambda ctx, sysp, text, **kw: (seen.append(text), "What is drawn here?")[-1]
wrapper = (library.RAG_HEAD + " rules...\n=== Retrieved passages ===\nlots of English text\n"
           "=== End passages ===" + library.RAG_QUESTION_SEAM + "Что здесь нарисовано?")
out = GL._translate_to_english(object(), wrapper)
check("only the question tail went to the translator", seen == ["Что здесь нарисовано?"], seen)
check("the wrapper is kept and the tail replaced",
      out.startswith(library.RAG_HEAD) and out.endswith(library.RAG_QUESTION_SEAM + "What is drawn here?"), out[-60:])

class _Lib:
    def is_empty(self): return False
    def retrieve(self, q, k=None):
        import knowledge
        return [knowledge.SearchResult(chunk_id=1, doc_id=1, source_type="library", title="doc",
                                       url="", path="d.md", text="chunk text", score=1.0, why="")]
p, info = library.build_rag_prompt(_Lib(), "Что думаешь?")
check("build_rag_prompt uses the shared head and seam",
      p.startswith(library.RAG_HEAD) and p.endswith(library.RAG_QUESTION_SEAM + "Что думаешь?"))

# «Скажи а» -> "Say, uh..." (live 2026-09-17 20:56): a one- or two-word
# message is passed through untranslated, and a bare "say" no longer sends
# the turn to the vision model.
seen.clear()
check("a two-word quip is not translated", GL._translate_to_english(object(), "Скажи а") == "Скажи а" and not seen)
# «Скажи а» vs "what does it say" about a picture: bench/intent_picture_live.py
src = open(os.path.join(os.path.dirname(__file__), "..", "bot/tg_tasks.py"), encoding="utf-8").read()
check("a message pointing at a picture skips document retrieval",
      "_about_a_picture" in src and 'getattr(task, "image_id", "")' in src)
print(f"\n{PASSED} passed, {FAILED} failed"); sys.exit(1 if FAILED else 0)

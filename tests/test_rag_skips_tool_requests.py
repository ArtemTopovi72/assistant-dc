"""With document search on, tool-shaped requests bypass retrieval.

Live 2026-09-13 (mega journey, steps 58-60): after a rental PDF was indexed,
'нарисуй витрину пекарни с вывеской «ХЛЕБ У ДОМА»' was wrapped into the RAG
prompt ("Using 1 passage(s) from dogovor.pdf", "answer ONLY from the passages,
do not use web search") — and so was every later drawing, weather and song
request in the chat. Questions about the documents must still hit retrieval.
"""
import os, sys, inspect
os.environ.setdefault("F5_TEST_RUN", "1")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tg_tasks as T

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

# Which messages the documents could answer is the model's read (intent
# doc_question); the phrases run live in bench/intent_docs_live.py.
import intent
DOCS = {"можно ли с собакой?", "за сколько дней надо предупредить о выезде?"}
intent.STUB = lambda t: {"doc_question": t in DOCS}
for t in ("[upscale]", "https://ru.wikipedia.org/wiki/Омлет", "👍",
          "сократи до 3 пунктов:\n\nКомпания «Ромашка» объявила о запуске новых электросамокатов.",
          "перескажи: " + "слово " * 70,
          __import__("prompt_guard").wrap_quoted("the user forwarded this message from someone else",
                                                 "Внимание! перекрывают движение"),
          "нарисуй витрину пекарни"):
    check("bypasses retrieval: " + t[:40], T._plainly_not_a_doc_question(t))
for t in DOCS:
    check("still retrieves: " + t, not T._plainly_not_a_doc_question(t))
# owner 10-03: three logs answered from old library fragments, none opened
intent.STUB = lambda t: {"doc_question": True}      # the model would call it a doc question
check("sandbox files are read, not retrieved",
      T._plainly_not_a_doc_question("в чём ошибка?\n"
                                    "[The file 'a.log', 'b.log' are now in your working folder]"))
intent.STUB = lambda t: {"doc_question": t in DOCS}
intent.STUB = None
check("no read (model down) bypasses: the plain loop, not a wrong RAG wrap",
      T._plainly_not_a_doc_question("можно ли с собакой?"))

src = inspect.getsource(T)
check("the RAG gate consults the helper",
      "if sess.use_docs and not _about_a_picture and not _plainly_not_a_doc_question(user_input):" in src)

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)

"""The RAG wrapper is docs-first, not docs-only.

Live 2026-09-13 (mega journey re-run, step 89): with a rental PDF indexed,
'сколько дней до нового года?' got an FTS hit on the contract (it mentions
'дней') and the old wrapper — "answer ONLY from the passages… say the
documents do not specify it" — made the bot reply that the documents do not
say how many days are left until New Year. Weak lexical hits must not turn
every general question into a documents-are-silent refusal.
"""
import os, sys, pathlib, tempfile
os.environ.setdefault("F5_TEST_RUN", "1")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tg_bot as T
DATA = tempfile.mkdtemp(prefix="ragfirst_")
T.redirect_data_dir(DATA)
import knowledge

class _StubEmbedder:
    def __init__(self, *a, **k):
        self.model = "stub"; self.base_url = "stub"; self.enabled = False; self.dim = None
    def available(self, recheck=30.0): return False
    def embed(self, texts, **k): return [None] * len(texts)
    def embed_one(self, text, **k): return None
knowledge.Embedder = _StubEmbedder
import library

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

d = pathlib.Path(DATA)
contract = d / "dogovor.txt"
contract.write_text("ДОГОВОР АРЕНДЫ. Срок аренды 12 месяцев. О выезде уведомить за 30 дней. "
                    "Залог возвращается в течение 10 дней после выезда.", encoding="utf-8")
lib = library.Library(d / "lib.db")
lib.build([str(contract)])

q = "сколько дней до нового года?"
prompt, info = library.build_rag_prompt(lib, q)
check("a lexical hit on 'дней' still wraps the question", prompt != q, info)
check("the question stays at the end", prompt.rstrip().endswith(q))
low = prompt.lower()
check("the wrapper no longer says ONLY the passages", "using only the retrieved" not in low)
check("the wrapper no longer forbids web search outright", "do not use web search," not in low)
check("questions ABOUT the documents are answered from the passages",
      "about the documents" in low and "answer from the passages" in low)
check("the documents-are-silent rule is scoped to document questions",
      "if the question is about the documents but the passages do not contain" in low)
check("an unrelated question is answered normally, tools allowed",
      "not about the documents" in low and "answer normally" in low and "using tools" in low)
check("...and must not be met with 'the documents are silent'",
      "do not\nsay the documents are silent" in low.replace("do not say", "do not\nsay")
      or "do not say the documents are silent" in low)
lib.close()

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)

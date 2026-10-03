"""A retrieval miss on a tiny library hands over the whole library.

Live 2026-09-13 (mega journey, step 26): "можно ли с собакой?" against a
one-chunk rental contract that says "домашних животных" was an FTS miss
(no embeddings), so the question reached the model bare and it answered
from the chat's other business (re-running the previous image edit).
"""
import os, sys, pathlib, tempfile
os.environ.setdefault("F5_TEST_RUN", "1")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tg_bot as T
DATA = tempfile.mkdtemp(prefix="smalllib_")
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
contract.write_text("ДОГОВОР АРЕНДЫ. Срок аренды 12 месяцев. Содержание домашних животных "
                    "допускается только с письменного согласия арендодателя. О выезде "
                    "уведомить за 30 дней.", encoding="utf-8")
lib = library.Library(d / "lib.db")
lib.build([str(contract)])

prompt, info = library.build_rag_prompt(lib, "можно ли с собакой?")
check("an FTS miss on a one-document library still wraps the question", prompt != "можно ли с собакой?", info)
check("...with the document's text", "домашних животных" in prompt)
check("...and the question at the end", prompt.rstrip().endswith("можно ли с собакой?"))
check("the status says passages were used", info and "passage" in info, info)

prompt, info = library.build_rag_prompt(lib, "за сколько дней уведомить о выезде?")
check("a normal hit still works", "30 дней" in prompt)

# A big library is NOT dumped wholesale.
big = d / "big.txt"
big.write_text("\n\n".join(f"Глава {i}. " + ("Текст про совершенно другое. " * 60) for i in range(40)), encoding="utf-8")
lib.build([str(contract), str(big)])
n = len(lib.all_chunks())
check("the fixture is larger than the small-library cap", n > library.SMALL_LIBRARY_CHUNKS, n)
prompt, info = library.build_rag_prompt(lib, "zzqx nonsense query")
check("a miss on a big library stays a miss", prompt == "zzqx nonsense query", info)
lib.close()

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)

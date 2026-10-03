"""An explicit request for English is not translated back to Russian.

Live 2026-09-12 (journey 18): "и то же самое по-английски" got an English
reply from the model, and _match_reply_language -- seeing a Cyrillic turn
with a Latin answer -- translated it back. The model had obeyed; the guard undid it.
"""
import os, sys
os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import graph_language as L, intent
# Which messages ask for an English reply is the model's read (agent/intent.py);
# the phrases run live in bench/intent_lang_live.py.
EN = {"и то же самое по-английски", "переведи это на английский", "ответь на английском",
      "say it in English", "translate that into English"}
intent.STUB = lambda t: {"reply_language": "en"} if t in EN else None

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

for q in EN:
    check("asks for English: " + q, L.asks_for_english(q))
for q in ["как будет по-английски «подоконник»?", "привет", ""]:
    check("does not: " + q, not L.asks_for_english(q))

calls = []
L.call_llm_simple = lambda *a, **k: calls.append(1) or "Спасибо, передам соседям."
class C:
    def is_cancelled(self): return False
ans = "Thank you for the information! Understood, I will pass it on to the neighbours."
out = L._match_reply_language(C(), ans, "и то же самое по-английски")
check("an English answer to an explicit English request is kept", out == ans and not calls, (out, calls))
out = L._match_reply_language(C(), ans, "напиши короткий вежливый ответ автору")
check("...but a plain Russian turn with an English answer is still translated back", out.startswith("Спасибо") and calls)

print("\n%d/%d checks passed" % (OK, OK + BAD))

sys.exit(1 if BAD else 0)


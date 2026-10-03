"""After forget_facts("all") the chat history's copy of the facts is void.

Live 2026-09-13 (mega run 3, step 61): 'забудь всё, что я про себя
рассказывал' dropped the saved facts, the bot confirmed — and 'напомни, на
что у меня аллергия?' got 'У вас аллергия на орехи' straight from the
'remember: …' message still sitting in the history. Context.facts_forgotten
now carries a note into every turn until a new fact is remembered.
"""
import os, sys, threading
from pathlib import Path
os.environ.setdefault("F5_TEST_RUN", "1")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import graph_compose as C
from models import Context

def _ctx():
    return Context(models=None, transcription_cache={}, cache_file=Path("tests/_ff.json"),
                   asr_lock=threading.Lock(), tts_lock=threading.Lock(), no_think=True)

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

ctx = _ctx()
ctx.pin_fact("The user's name is Marat; vegetarian; allergic to nuts.")
check("a fresh context has nothing forgotten", ctx.facts_forgotten is False)
msg, _, _ = C._compose_user_message(ctx, {"user_input": "what am I allergic to?"})
check("with a saved fact the turn carries the fact", "allergic to nuts" in msg and "VOID" not in msg)

n = ctx.forget_facts("all")
check("forget_facts('all') drops the fact", n == 1 and not ctx.pinned_facts)
check("...and raises the flag", ctx.facts_forgotten is True)
msg, _, _ = C._compose_user_message(ctx, {"user_input": "remind me, what am I allergic to?"})
check("the next turn says the earlier statements are void",
      C.FORGOTTEN_NOTE in msg and "allergic" not in msg.split(C.FORGOTTEN_NOTE)[0], msg[:200])
check("the note says what to answer", "nothing saved" in C.FORGOTTEN_NOTE)
msg, _, _ = C._compose_user_message(ctx, {"user_input": "tell a joke"})
check("the note persists across turns", C.FORGOTTEN_NOTE in msg)

ctx.pin_fact("The user's cat is called Barsik.")
check("a new remember_fact clears the flag", ctx.facts_forgotten is False)
msg, _, _ = C._compose_user_message(ctx, {"user_input": "what is my cat's name?"})
check("...and the turn is back to Saved facts", "Barsik" in msg and C.FORGOTTEN_NOTE not in msg)

ctx2 = _ctx()
ctx2.pin_fact("likes tea"); ctx2.pin_fact("has a dog")
ctx2.forget_facts("dog")
check("a targeted forget does not void the whole history", ctx2.facts_forgotten is False and len(ctx2.pinned_facts) == 1)

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)

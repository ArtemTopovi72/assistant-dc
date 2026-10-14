"""A bare path in a Russian chat got an English reply: ASCII skips the translate step,
so no language directive reached the model. The turn now points at the chat's language."""
import os, sys, threading
from pathlib import Path
os.environ.setdefault("F5_TEST_RUN", "1")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import graph_compose as C
from models import Context

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

def ctx():
    return Context(models=None, transcription_cache={}, cache_file=Path("tests/_bp.json"),
                   asr_lock=threading.Lock(), tts_lock=threading.Lock(), no_think=True)

RU = [{"role": "user", "content": "посмотри файл на рабочем столе"}, {"role": "assistant", "content": "ок"}]
PATH = r"C:\Users\Artem\Desktop\notes.txt"
msg, _, _ = C._compose_user_message(ctx(), {"user_input": PATH, "messages": RU})
check("a bare path in a Russian chat carries the language directive", "посмотри файл" in msg and "same language" in msg, msg[:300])
msg, _, _ = C._compose_user_message(ctx(), {"user_input": PATH, "messages": [{"role": "user", "content": "look at my file"}]})
check("an English chat gets no directive", "same language" not in msg)
msg, _, _ = C._compose_user_message(ctx(), {"user_input": "what is in " + PATH, "messages": RU})
check("a message with words of its own gets none", "only a path" not in msg)
check("links and ids count as no words", C._has_no_words("https://example.com/a") and not C._has_no_words("hello"))
print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)

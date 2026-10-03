"""An English reply is sent as text beside the voice note.

Live 2026-09-12 (journey 6): an English user's "describe the picture" came
back as a voice note only, and the Russian F5 voice speaks English by
transliteration -- the reader had no way to get the actual sentence.
"""
import os, sys
os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import tg_bot

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

f = tg_bot._needs_text_beside_voice
check("an English sentence needs the text", f("A dramatic lighthouse stands on a jagged cliff."))
check("a Russian sentence does not", not f("Маяк стоит на скале во время шторма."))
check("Russian with one Latin brand name does not", not f("Я использую Python для скриптов."))
check("an empty reply does not", not f(""))
check("a short Latin token alone does not", not f("OK"))

src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(tg_bot.__file__))), "bot/tg_tasks.py"), encoding="utf-8").read()
check("delivery consults it beside the fenced-code rule",
      'Every reply goes out as text too' in src)

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)

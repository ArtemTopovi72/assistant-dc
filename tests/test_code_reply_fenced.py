"""Source code is fenced, shown as code, and never spoken.

Live 2026-09-12 (journey 24): "напиши на python функцию…" took the fast path,
whose prompt forbids markdown outright, so the code arrived as prose in the
chat. Now: coding requests reach the tool loop, both prompts carve out the
fenced-block exception, the TTS cleaner drops fenced blocks, and a reply with
a fence is sent as text even when the voice note went out.
"""
import os, sys
os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import graph, prompts, audio

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

# Which phrases need a tool is read by the model: bench/intent_live.py.

full = prompts.build_system_prompt()
lite = prompts.build_system_prompt_lite()
check("full prompt: code goes in a fenced block", "``` fenced block" in full and "never read aloud" in full)
check("lite prompt too", "``` fenced block" in lite)

spoken = audio.clean_text("Вот функция.\n```python\ndef f(x):\n    return x\n```\nОна возвращает аргумент.")
check("fenced code is not voiced", "def f" not in spoken and "return" not in spoken, spoken)
check("the prose around it is", "Вот функция" in spoken and "возвращает" in spoken, spoken)
check("plain text untouched", audio.clean_text("привет мир") == "привет мир")

src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(graph.__file__))), "bot/tg_tasks.py"), encoding="utf-8").read()
check("a fenced reply is sent as text beside the voice note", 'Every reply goes out as text too' in src)

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)

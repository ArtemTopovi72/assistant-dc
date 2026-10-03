"""tool_descriptions._RUN_CODE_DESC must tell the model to actually call
run_code when a request also wants to see/save/send a result, not just print
the source as a chat answer.

Live 2026-09-18: "напиши код на python с graphviz, который рисует блок-схему
... сохрани картинку и покажи её мне" (write code that draws a flowchart,
save the picture, show it) got answered with the graphviz source pasted into
the chat reply plus a "how this works" paragraph -- no run_code call, no
file, no picture. journeys.json for that run: step 1 problems =
["no sendPhoto", "working folder lacks ['*.png', '*.py']", "no photo to
look at"]. The description already said "never paste the code into your
reply", which was not enough to override "write code that does X" being read
as the deliverable rather than the method.

Run: venv/Scripts/python.exe tests/test_run_code_desc_execute_not_explain.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import tool_descriptions as td

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

desc = td._RUN_CODE_DESC

check("still warns against pasting code into the chat reply",
      "never paste the code into your reply" in desc.lower())
check("explicitly says 'write code that does X' describes the method, not the goal",
      "describes the method, not the" in desc.lower())
check("explicitly says to call run_code and deliver what it produced",
      "call run_code and deliver" in desc.lower())
check("mentions the see/save/send trigger phrase",
      "see, save, or send" in desc.lower())

print(f"\n{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)

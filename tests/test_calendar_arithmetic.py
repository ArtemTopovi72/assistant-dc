"""Calendar questions must reach `calculate` and be computed, not guessed.

Live 2026-09-12 (journey 14): "сколько дней до нового года?" and "сколько мне
полных лет?" took the fast path and the model answered from an invented
"Сегодня 23 мая 2024 года" — 222 days, 34 years. Three things fix it: the
lite prompt now states today's date, the trigger regex routes calendar
arithmetic to the tool loop, and calculate has today()/date()/days_between/
years_between/weekday so the model never counts in its head.
"""
import os, sys, datetime as dt
os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import graph
import tools
import prompts
import tool_descriptions

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

# Which phrases need a tool is read by the model: bench/intent_live.py.

class _Ctx: pass
calc = lambda e: tools._handle_calculate(_Ctx(), {}, {"expression": e})
today = dt.date.today()
check("today() is the real date", calc("today()") == today.isoformat(), calc("today()"))
ny = (dt.date(today.year + 1, 1, 1) - today).days
check("days until New Year", calc("days_between(today(), date(%d,1,1))" % (today.year + 1)) == str(ny))
age = today.year - 1990 - ((today.month, today.day) < (3, 3))
check("age in full years is birthday-aware", calc("years_between(date(1990,3,3), today())") == str(age))
check("birthday not yet reached this year counts one less",
      calc("years_between(date(2000,12,31), date(2026,9,12))") == "25")
check("birthday reached counts full", calc("years_between(date(2000,9,12), date(2026,9,12))") == "26")
check("weekday name", calc("weekday(date(2026,9,12))") == "Saturday")
check("negative span when b is earlier", calc("days_between(date(2026,1,10), date(2026,1,1))") == "-9")
check("attribute access still forbidden", "[TOOL ERROR]" in calc("(date(2027,1,1) - today()).days"))
check("dunder still forbidden", "[TOOL ERROR]" in calc("today().__class__"))

lite = prompts.build_system_prompt_lite()
check("lite prompt carries today's date", today.isoformat() in lite and today.strftime("%A") in lite)
full = prompts.build_system_prompt()
check("full prompt carries the weekday too", today.strftime("%A") in full)
check("calculate's description teaches the date helpers",
      "days_between" in tool_descriptions._CALCULATE_DESC and "years_between" in tool_descriptions._CALCULATE_DESC)

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)

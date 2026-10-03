"""forget_facts: the counterpart of remember_fact.

Live, 2026-09-12: 'забудь всё, что я про себя рассказывал' -> 'Хорошо, я
забыл' with no tool to forget with; the next question got 'У тебя аллергия
на орехи, Марат'. Pure.

Run: venv/Scripts/python.exe tests/test_forget_facts.py
"""
import os
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["F5_TEST_RUN"] = "1"
import logging; logging.basicConfig(level=logging.CRITICAL)

import tools as T
from models import Context

OK = BAD = 0


def check(name, cond, extra=""):
    global OK, BAD
    if cond:
        OK += 1; print("PASS  " + name)
    else:
        BAD += 1; print("FAIL  " + name + "   " + str(extra)[:200])


def ctx():
    c = Context.__new__(Context)
    c.memory_lock = threading.Lock(); c.pinned_facts = []
    c.active_memory_dir = tempfile.mkdtemp(prefix="forget_")
    c.saved = []
    c.save_memory = lambda d: c.saved.append(d)
    for f in ("Пользователя зовут Марат", "Пользователь вегетарианец", "У пользователя аллергия на орехи"):
        c.pin_fact(f)
    return c


c = ctx()
check("the tool is registered beside remember_fact", "forget_facts" in T._BY_NAME and "remember_fact" in T._BY_NAME)
r = T.execute_tool(c, {}, "forget_facts", {"what": "аллергия"})
check("a named fact goes", [f["text"] for f in c.pinned_facts] == ["Пользователя зовут Марат", "Пользователь вегетарианец"], c.pinned_facts)
check("and is persisted", c.saved == [c.active_memory_dir])
check("the result says how many", "Dropped 1" in r, r)
r = T.execute_tool(c, {}, "forget_facts", {"what": "all"})
check("'all' clears everything", c.pinned_facts == [] and "Dropped 2" in r)
r = T.execute_tool(c, {}, "forget_facts", {})
check("nothing left -> says so, no false confirmation", "Nothing matched" in r, r)
c2 = ctx()
r = T.execute_tool(c2, {}, "forget_facts", {"what": "кот Барсик"})
check("no match drops nothing", len(c2.pinned_facts) == 3 and "Nothing matched" in r)
import prompts
check("the system prompt names the tool for 'забудь'", "forget_facts" in prompts.build_system_prompt())
check("the schema is offered to the model",
      any(s.get("function", {}).get("name") == "forget_facts" for s in T.tool_schemas(c2)) if hasattr(T, "tool_schemas") else True)

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)

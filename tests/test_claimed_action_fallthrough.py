"""Claimed actions on the tool-less fast path, and the path-less write_file name.

Cue-less requests used to take the fast path (no tools sent) and the model
claimed what it never did -- "Хорошо, я забыл все факты",
"я запустил глубокое исследование", a JSON pseudo-call written as text,
"Этот код выведет …". Each must fall through to the full loop. Hard
arithmetic must not take the fast path at all, or the calculator force can
never apply.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("F5_TEST_RUN", "1")

import graph_fastpath as F
import tools as T

ok = []


def check(c, m):
    ok.append(bool(c)); print(("ok   " if c else "FAIL ") + m)


# Which drafts leave the fast path is the model's reading now (agent/reply_read.py),
# scored live with these phrases in bench/reply_read_set.


task = "Напиши rx.py — свой движок регулярных выражений. Напиши pytest-тесты и прогони."
check(T._default_write_path("def match(p, s):\n    pass", task) == "rx.py", "path-less code -> the file the task names")
check(T._default_write_path("import pytest\ndef test_a(): pass", task) == "test_rx.py", "path-less tests -> test_<name>")
check(T._default_write_path("def f(): pass", "сделай a.py и b.py") == "script.py", "two names -> no guess")
check(T._default_write_path("def f(): pass") == "script.py", "no task -> script.py as before")
check(T._normalize_args("write_file", {"content": "def match(): pass"}, task)["path"] == "rx.py",
      "normalizer passes the task through")

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)

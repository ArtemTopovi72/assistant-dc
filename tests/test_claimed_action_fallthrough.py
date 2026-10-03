"""Claimed actions on the tool-less fast path, and the path-less write_file name.

Routing bench 2026-09-24: cue-less requests took the fast path (no tools sent)
and the model claimed what it never did -- "Хорошо, я забыл все факты",
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


R = F._FAST_PATH_ACTION_CLAIM_RE
claims = [
    "Хорошо, я забыл все факты о вашей работе.",
    'Конечно!\n{\n  "action": "dalle.text2im",\n  "action_input": "{}"\n}',
    "Для разбора рынка я запустил глубокое исследование.",
    "Вот несколько вариантов промптов для генерации таких обоев.",
    "```python\nprint(1)\n```\nЭтот код выведет 1.",
    "Я могу создавать видео, но для этого мне нужно описание.",
    "Сейчас нарисую закат над морем!",
    "Запомнил: у тебя аллергия на орехи. Готово, я сохранил это.",
    "I have deleted those facts.",
]
for a in claims:
    check(R.search(a), f"falls through: {a[:50]!r}")

chat = [
    "Привет! Рад тебя слышать, как прошёл день?",
    "Спасибо, рад помочь!",
    "Я могу помочь с чем угодно, если расскажешь подробнее.",
    "Кофе лучше пить утром: кортизол к обеду снижается.",
    "Ищу слова, чтобы описать это... просто красиво.",
]
for a in chat:
    check(not R.search(a), f"stays on the fast path: {a[:50]!r}")


task = "Напиши rx.py — свой движок регулярных выражений. Напиши pytest-тесты и прогони."
check(T._default_write_path("def match(p, s):\n    pass", task) == "rx.py", "path-less code -> the file the task names")
check(T._default_write_path("import pytest\ndef test_a(): pass", task) == "test_rx.py", "path-less tests -> test_<name>")
check(T._default_write_path("def f(): pass", "сделай a.py и b.py") == "script.py", "two names -> no guess")
check(T._default_write_path("def f(): pass") == "script.py", "no task -> script.py as before")
check(T._normalize_args("write_file", {"content": "def match(): pass"}, task)["path"] == "rx.py",
      "normalizer passes the task through")

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)

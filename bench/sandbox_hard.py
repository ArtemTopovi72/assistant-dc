"""Hard coding tasks, scored by HIDDEN tests the model never sees.

    venv/Scripts/python bench/sandbox_hard.py [--reps 1] [--case calc] [--dump dir]

Reuses sandbox_e2e's runner. Each task is small enough to fit a turn but has
edge cases a sloppy implementation misses; the score is how many hidden checks
pass, not only pass/fail.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sandbox_e2e as E                      # noqa: E402
from code_sandbox import AGENT_DIR           # noqa: E402


def _hidden(box, code: str, need: int):
    """Run hidden checks inside the sandbox; each prints 'OK <name>' or 'BAD'."""
    import code_runner
    res = code_runner.run_python(box, "import os, sys; sys.path.insert(0, os.getcwd())\n" + code,
                                 timeout=120, allow_host=False)
    out = res.output or ""
    ok = sum(1 for l in out.splitlines() if l.startswith("OK "))
    bad = [l for l in out.splitlines() if l.startswith("BAD")][:3]
    return ok >= need, f"{ok}/{need} hidden checks" + (("; " + " | ".join(bad))[:220] if bad else
                                                      ("" if ok >= need else "; " + out[-160:].replace("\n", " ")))


_HARNESS = '''
def T(name, fn):
    try:
        r = fn()
        print(("OK " if r is not False else "BAD ") + name)
    except Exception as e:
        print("BAD " + name + ": " + type(e).__name__ + " " + str(e)[:80])
def raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    except Exception as e:
        raise AssertionError("wrong exception " + type(e).__name__)
    return False
'''

# ── 1. expression evaluator from scratch ───────────────────────────────────

def _check_calc(box, state):
    if not (box.root / "calc.py").exists():
        return False, "no calc.py"
    return _hidden(box, _HARNESS + '''
import ast as _ast, calc
src = open("calc.py", encoding="utf-8").read()
T("no eval", lambda: "eval(" not in src and "exec(" not in src)
E = calc.evaluate
T("prec", lambda: E("2+3*4") == 14)
T("parens", lambda: E("(2+3)*4") == 20)
T("pow right-assoc", lambda: E("2^3^2") == 512)
T("unary", lambda: E("-2^2") == -4)
T("unary nested", lambda: E("--3") == 3)
T("float", lambda: abs(E("7/2") - 3.5) < 1e-9)
T("vars", lambda: E("x*y+1", {"x": 3, "y": 4}) == 13)
T("funcs", lambda: E("max(1, min(5, 3), 2)") == 3)
T("sqrt", lambda: abs(E("sqrt(16)+1") - 5) < 1e-9)
T("spaces", lambda: E("  1 +\\t2 ") == 3)
T("div0", lambda: raises(lambda: E("1/0")))
T("unknown var", lambda: raises(lambda: E("z+1")))
T("unbalanced", lambda: raises(lambda: E("(1+2")))
T("garbage", lambda: raises(lambda: E("1 + * 2")))
T("trailing", lambda: raises(lambda: E("1 2")))
T("unknown func", lambda: raises(lambda: E("foo(1)")))
''', 16)

# ── 2. three subtle bugs across modules ────────────────────────────────────

def _inventory(box):
    box.write_text("inv/__init__.py", "")
    box.write_text("inv/store.py",
        "class Store:\n"
        "    def __init__(self, items=[]):\n"
        "        self.items = items\n\n"
        "    def add(self, sku, qty, price):\n"
        "        for it in self.items:\n"
        "            if it['sku'] == sku:\n"
        "                it['qty'] += qty\n"
        "                return\n"
        "        self.items.append({'sku': sku, 'qty': qty, 'price': price})\n\n"
        "    def remove(self, sku, qty):\n"
        "        for it in self.items:\n"
        "            if it['sku'] == sku:\n"
        "                if it['qty'] < qty:\n"
        "                    raise ValueError('not enough ' + sku)\n"
        "                it['qty'] -= qty\n"
        "                if it['qty'] == 0:\n"
        "                    self.items.remove(it)\n"
        "                return\n"
        "        raise KeyError(sku)\n")
    box.write_text("inv/report.py",
        "def page(items, n, size):\n"
        "    \"\"\"1-based page n of the items sorted by sku.\"\"\"\n"
        "    items = sorted(items, key=lambda i: i['sku'])\n"
        "    return items[n * size:(n + 1) * size]\n\n\n"
        "def value(items):\n"
        "    \"\"\"Total stock value in kopecks (price is roubles, may be fractional).\"\"\"\n"
        "    return int(sum(i['qty'] * i['price'] for i in items) * 100)\n")
    box.write_text("tests/test_inv.py",
        "from inv.store import Store\nfrom inv.report import page, value\n\n\n"
        "def test_independent_stores():\n"
        "    a = Store(); a.add('x', 1, 1.0)\n"
        "    assert Store().items == []\n\n\n"
        "def test_first_page():\n"
        "    s = Store()\n"
        "    for k in 'dcba':\n        s.add(k, 1, 1.0)\n"
        "    assert [i['sku'] for i in page(s.items, 1, 2)] == ['a', 'b']\n\n\n"
        "def test_value():\n"
        "    s = Store(); s.add('a', 3, 19.99)\n"
        "    assert value(s.items) == 5997\n")
    return "inv"


def _check_inventory(box, state):
    ok, out = E._pytest_green(box)
    t = box.read_text("tests/test_inv.py")
    if not ok:
        return False, "tests still fail: " + out.replace("\n", " ")[-150:]
    if "5997" not in t or "['a', 'b']" not in t:
        return False, "edited the tests"
    return _hidden(box, _HARNESS + '''
from inv.store import Store
from inv.report import page, value
def pg():
    s = Store()
    for k in "edcba": s.add(k, 1, 1.0)
    return [i["sku"] for i in page(s.items, 3, 2)] == ["e"] and page(s.items, 4, 2) == []
T("last page", pg)
T("value rounding", lambda: value([{"qty": 1, "price": 0.29}, {"qty": 7, "price": 1.1}]) == 799)
T("iso", lambda: (Store().add("q", 1, 1.0), Store().items)[1] == [])
def rm():
    s = Store(); s.add("a", 2, 1.0); s.remove("a", 2)
    return s.items == []
T("remove all", rm)
''', 4)

# ── 3. streaming log aggregation with a spec ───────────────────────────────

def _logs(box):
    lines = []
    import random
    rnd = random.Random(7)
    for i in range(400):
        m = rnd.randrange(0, 5)
        ep = rnd.choice(["/api/a", "/api/b", "/login"])
        st = rnd.choice([200, 200, 200, 404, 500])
        ms = rnd.randrange(5, 900)
        lines.append(f"2026-09-23T10:{m:02d}:{i % 60:02d}Z {ep} {st} {ms}ms")
    lines.insert(50, "garbage line")
    lines.insert(120, "2026-09-23T10:01:00Z /api/a notanumber 5ms")
    box.write_text("access.log", "\n".join(lines) + "\n")
    return "access.log"


def _check_logs(box, state):
    if not (box.root / "logstat.py").exists():
        return False, "no logstat.py"
    return _hidden(box, _HARNESS + '''
import logstat, re, math
lines = open("access.log", encoding="utf-8").read().splitlines()
good = []
for l in lines:
    m = re.fullmatch(r"(\\S+T(\\d\\d):(\\d\\d):\\d\\dZ) (\\S+) (\\d{3}) (\\d+)ms", l)
    if m: good.append((m.group(2)+":"+m.group(3), m.group(4), int(m.group(5)), int(m.group(6))))
def p95(v):
    v = sorted(v); return v[max(0, math.ceil(0.95 * len(v)) - 1)]
exp = {}
for mi, ep, st, ms in good:
    exp.setdefault((mi, ep), []).append((st, ms))
r = logstat.summarize("access.log")
T("skipped", lambda: r["skipped"] == 2)
T("keys", lambda: set(r["buckets"]) == {f"{k[0]} {k[1]}" for k in exp})
def vals():
    for (mi, ep), v in exp.items():
        b = r["buckets"][f"{mi} {ep}"]
        if b["count"] != len(v): return False
        if b["errors"] != sum(1 for s, _ in v if s >= 500): return False
        if b["p95_ms"] != p95([ms for _, ms in v]): return False
    return True
T("values", vals)
T("order", lambda: list(r["buckets"]) == sorted(r["buckets"]))
''', 4)


# ── 4. a regex engine, fuzzed against Python's re ──────────────────────────

def _check_regex(box, state):
    if not (box.root / "rx.py").exists():
        return False, "no rx.py"
    return _hidden(box, _HARNESS + '''
import re, random, rx
src = open("rx.py", encoding="utf-8").read()
T("no re module", lambda: not re.search(r"^\\s*(import re\\b|from re\\b)", src, re.M))
P = ["a*b", "(ab|cd)+e?", "[a-c]+x", "[^ab]*", "a.c", "^(a|b)*c$", "x?y+z*",
     "(a*)*b", "((ab)|c)*d", "[a-z]+[0-9]?", "a|b|", "(a|ab)(c|bcd)d", "\\\\.a\\\\*"]
rnd = random.Random(3)
def fuzz(fn_ours, fn_ref, name):
    bad = 0
    for p in P:
        for _ in range(60):
            s = "".join(rnd.choice("abcdxyz0.*") for _ in range(rnd.randrange(0, 9)))
            try:
                if bool(fn_ours(p, s)) != bool(fn_ref(p, s)):
                    bad += 1
            except Exception:
                bad += 1
    print(("OK " if bad == 0 else "BAD ") + name + f" ({bad} mismatches)")
fuzz(rx.match, re.fullmatch, "fullmatch fuzz")
fuzz(rx.search, re.search, "search fuzz")
def slow():
    import time; t = time.time()
    rx.match("(a*)*b", "a" * 30)
    return time.time() - t < 2
T("no exponential blowup", slow)
''', 4)


CASES = [
    dict(id="calc", setup=E._empty,
         text="Напиши calc.py с функцией evaluate(expr, variables=None) — вычислитель "
              "выражений БЕЗ eval/exec: + - * /, ^ (степень, правоассоциативная, "
              "приоритет выше унарного минуса, т.е. -2^2 = -4), унарный минус, скобки, "
              "переменные из словаря, функции min, max (любое число аргументов) и sqrt. "
              "На любую ошибку (деление на ноль, неизвестная переменная или функция, "
              "синтаксис) — ValueError. Напиши pytest-тесты и прогони их.",
         check=_check_calc),
    dict(id="inventory", setup=_inventory,
         text="В проекте inv падают тесты. Там несколько багов в разных файлах. Найди и "
              "почини ВСЕ (тесты не трогай), подумай о граничных случаях, прогони тесты.",
         check=_check_inventory),
    dict(id="logstat", setup=_logs,
         text="Напиши logstat.py с функцией summarize(path) для access.log. Строка: "
              "'<ISO-время>Z <endpoint> <status> <N>ms'. Кривые строки пропускай и считай. "
              "Верни {'skipped': int, 'buckets': {...}}, где ключ — 'ЧЧ:ММ endpoint', "
              "ключи отсортированы, значение — {'count', 'errors' (status >= 500), "
              "'p95_ms' (nearest-rank: отсортировать, взять элемент ceil(0.95*n)-1)}. "
              "Проверь на файле.",
         check=_check_logs),
    dict(id="regex", setup=E._empty,
         text="Напиши rx.py — свой движок регулярных выражений БЕЗ модуля re: "
              "match(pattern, text) (совпадение со всей строкой) и search(pattern, text) "
              "(где-то в строке). Поддержка: литералы, '.', '*', '+', '?', альтернатива "
              "'|', группы (), классы [a-c], [^ab], якоря ^ и $, экранирование '\\'. "
              "Семантика как у Python re. Важно: без экспоненциального перебора — "
              "match('(a*)*b', 'a'*30) должен отвечать мгновенно (используй NFA/Thompson). "
              "Напиши pytest-тесты и прогони.",
         check=_check_regex),
]

if __name__ == "__main__":
    E.CASES[:] = CASES
    sys.exit(E.main())

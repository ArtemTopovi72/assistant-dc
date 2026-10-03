"""Regression: the `calculate` tool must reject every process-hanging / OOM vector
while still evaluating ordinary scalar math.

Bomb classes (all materialise a giant int inside a C call BEFORE eval can raise, so a
try/except cannot save the process):
  - factorial/comb/perm/prod/ldexp on large args
  - bit-shift bombs (1<<10**9)
  - sequence-repetition bombs ([0]*10**9)
  - single huge exponent (10**100000)
  - RIGHT-nested power tower (a**b**c)
  - LEFT-nested power tower ((a**b)**c)  <-- BUG #9, found by self-red-team:
    the original guard only inspected the RIGHT operand of `**`, so a left-nested
    tower had a constant <=1000 exponent at every node and sailed past both the tower
    and exponent checks; ((10**900)**900)**900 = 10**729_000_000 hung the process.

Run: venv/Scripts/python.exe tests/test_calculate_dos.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
import tools


def _calc(expr):
    return tools.execute_tool(object(), {}, "calculate", {"expression": expr})


BOMBS = [
    "factorial(9999999)", "comb(1000000, 500000)", "perm(1000000, 3)",
    "prod(range(1))", "ldexp(1, 10**9)",
    "1 << 10**9", "[0]*10**9", "(1,)*10**9",
    "10**100000", "2**10**9",
    "2**10**5",                 # right-nested tower
    "(10**50)**50",             # left-nested tower (small exponents)
    "((10**900)**900)**900",    # left-nested tower that actually OOMs
    "10**900**900",             # right-nested
    "(9**9)**9",                # left-nested
    "().__class__",             # sandbox-escape attempt (dunder)
]

# These MUST evaluate to a real number quickly.
OK = {
    "2+2": "4", "(15*1.2)/3": "6.0", "sqrt(144)": "12.0", "2**10": "1024",
    "10**3": "1000", "2**1000": None, "log(100,10)": None, "3**3": "27",
    "abs(-5)": "5", "round(3.14159, 2)": "3.14",
}


def test_bombs_blocked():
    bad = []
    for e in BOMBS:
        t0 = time.time()
        out = _calc(e)
        dt = time.time() - t0
        blocked = isinstance(out, str) and out.lstrip().startswith("[TOOL ERROR]")
        if not blocked or dt > 5:
            bad.append((e, blocked, round(dt, 2), out[:60]))
    assert not bad, f"UNBLOCKED / SLOW bomb vectors: {bad}"
    print(f"PASS all {len(BOMBS)} bomb vectors blocked as [TOOL ERROR] within time bound")


def test_plain_math_still_works():
    bad = []
    for e, want in OK.items():
        out = _calc(e)
        if out.lstrip().startswith("[TOOL ERROR]"):
            bad.append((e, out[:60]))
        elif want is not None and out.strip() != want:
            bad.append((e, f"got {out!r} want {want!r}"))
    assert not bad, f"plain math wrongly rejected / wrong: {bad}"
    print(f"PASS all {len(OK)} plain-math expressions evaluate correctly")


if __name__ == "__main__":
    test_bombs_blocked()
    test_plain_math_still_works()
    print("\ndone")

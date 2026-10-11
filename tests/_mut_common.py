"""Scaffolding shared by the `_mut_*.py` mutation harnesses."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def utf8_stdout():
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass


def run_suite(suite, env, timeout):
    return subprocess.run([sys.executable, suite], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=timeout)


def first_fail(r):
    """Text of the suite's first FAIL line, or '(crashed)' when it printed none."""
    fails = [l for l in (r.stdout or "").splitlines() if l.startswith("FAIL")]
    return fails[0][6:].strip() if fails else "(crashed)"

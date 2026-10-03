"""Run every suite in tests/, both kinds, and report one total.

Half of this directory is pytest modules and half is standalone scripts that
execute at import and end in `sys.exit(1 if BAD else 0)`. That mix is why
`pytest tests/` does not work at all: collection imports the first script, the
script calls sys.exit, and pytest aborts the whole session with INTERNALERROR
before running anything. So there has been no single command that runs the
suite -- which is exactly the condition under which a suite quietly rots.

This runs each file the way that file expects to be run, in its own process,
and prints one summary. A file is treated as a script when it calls sys.exit at
module level; everything else goes to pytest.

    venv/Scripts/python.exe tests/run_all.py [-k SUBSTRING] [--timeout N]
                                             [--list] [-j N]
"""
import argparse
import ast
import concurrent.futures as _cf
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# A script announces itself by exiting at module level (or from an `if`/`try`
# guard AT module level -- still no new scope, unlike `def`/`class`).
#
# This used to be the regex `^sys\.exit\(` (only a line whose FIRST character
# is "sys.exit("), which missed two real, fully-offline, check()-based
# suites' shapes: `print(...); sys.exit(...)` compound statements (the exit
# is not at column 0 even though it is still module level -- test_
# picture_question.py, test_rag_question_translation.py) and `sys.exit(1)`
# nested one level inside a module-level `if`/`if __name__ == "__main__":`
# block (test_library_fixes.py, test_tg_fwd_voice_batch.py, test_deck_
# informativeness.py, test_tg_style_button.py). All of them silently fell
# through every branch below to "skip" and were never run by run_all.py at
# all -- caught 2026-09-18 when a brand-new test file (test_tg_style_
# button.py) turned out to be missing from run_all's own summary.
def _has_module_level_exit(src: str) -> bool:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False

    def scan(stmts) -> bool:
        for n in stmts:
            call = n.value if isinstance(n, ast.Expr) else None
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "exit" and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "sys"):
                return True
            # `if`/`try`/`while`/`for` at module level are not a new scope --
            # `def`/`class` are, and are deliberately not recursed into.
            if isinstance(n, (ast.If, ast.Try, ast.While, ast.For)):
                for field in ("body", "orelse", "finalbody"):
                    if scan(getattr(n, field, None) or []):
                        return True
        return False

    return scan(tree.body)


def classify(path):
    try:
        src = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return "skip"
    # An explicit opt-out for a script that must NEVER auto-run in a full
    # sweep -- a live external-service audit (test_agent_dialogue.py needs an
    # already-loaded LM Studio model and makes real calls to it) or a
    # manual/interactive tool with no pass/fail of its own (test_vad.py
    # listens to a real microphone). Checked first so it wins even though
    # both of those also match the module-level-exit shape below.
    if re.search(r"^NOT_IN_RUN_ALL\s*=\s*True", src, re.M):
        return "skip"
    # An explicit opt-out, for a suite whose tests must run in a stated ORDER.
    # pytest runs test_ functions in definition order and ignores the list in
    # main(), so test_research_quality's last test -- the one that reloads
    # config and deep_research -- ran three tests early, and the reload took
    # pytest's own output capture down with it at teardown.
    if re.search(r"^RUN_AS_SCRIPT\s*=\s*True", src, re.M):
        return "script"
    if _has_module_level_exit(src):
        return "script"
    # A file with its own main() behind a __main__ guard is a script too, even
    # though nothing exits at import. Handing those to pytest is worse than
    # useless: their check() helpers only COUNT, so every collected test_
    # function returns normally no matter what it saw and the suite reports
    # green while asserting nothing. Measured on test_image_size_choice: with
    # pytest, replacing the size lookup with a hardcoded 960x544 still passed.
    if (re.search(r'^if __name__ == .__main__.:', src, re.M)
            and re.search(r'^def main\(', src, re.M)
            and not re.search(r'^(async )?def test_', src, re.M)):
        return "script"
    if re.search(r"^(async )?def test_|^class Test", src, re.M):
        return "pytest"
    # Module-level asserts are a script's verdict: an uncaught AssertionError
    # exits 1. These were skipped, so test_latin_question_russian_reply sat red
    # through a "397/397 green" sweep.
    if re.search(r"^assert\b", src, re.M):
        return "script"
    return "skip"


def counting_only(name):
    """Does this file's check() helper merely tally, instead of raising?"""
    try:
        tree = ast.parse(open(os.path.join(HERE, name), encoding="utf-8").read())
    except (OSError, SyntaxError):
        return False
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in ("check", "chk", "_check", "ok"):
            body = ast.dump(n)
            return "Raise(" not in body and "Assert(" not in body
    return False


_FAIL_LINE = re.compile(r"\s*(\[?FAIL\b|FAILED\b|FAIL:|E   |\w*Error\b|Traceback)")


def run_one(name, kind, timeout):
    path = os.path.join(HERE, name)
    if kind == "script":
        cmd = [sys.executable, path]
    else:
        cmd = [sys.executable, "-m", "pytest", path, "-q",
               "-p", "no:cacheprovider"]
    # Marks every child as a test process. lmstudio._lms_unload_all refuses to
    # run under it: one suite exercised the context-overflow path, which answers
    # by shelling out to `lms unload --all`, and every full run silently
    # unloaded the operator's model. Set at the runner so a suite written
    # tomorrow is covered without remembering to opt in.
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
               F5_TEST_RUN="1")
    # Same idea for the network: an unstubbed LLM/ComfyUI call in a suite hit
    # the operator's live server (test_long_video made 48 real vision calls,
    # 2.5 min, in the middle of a night bench). Dead port unless asked.
    if not os.getenv("RUN_ALL_LIVE"):
        env.update(LM_STUDIO_BASE="http://127.0.0.1:9", COMFY_URL="http://127.0.0.1:9")
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, cwd=ROOT, env=env, timeout=timeout,
                           capture_output=True, text=True,
                           errors="replace", stdin=subprocess.DEVNULL)
        code, out = p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        code, out = -9, "TIMEOUT after %ss" % timeout
    # A suite that needs LM Studio or ComfyUI and cannot reach them (or cannot
    # fit beside them on one card) has not failed -- it did not run. Several
    # say so with exit 2 and a CANNOT RUN line. Reported apart, so a busy card
    # does not read as a regression.
    if code == 2 or "CANNOT RUN" in out:
        code = "skip"
    return name, kind, code, time.perf_counter() - t0, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", default="", help="only files containing this")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("-j", type=int, default=24)  # 28 logical CPUs; owner 10-03: 24
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    # difftest_* counts too. The two backend difftests are the ONLY thing that
    # compares the in-process and out-of-process implementations of knowledge
    # and deep research against each other, and the prefix alone kept their 60
    # checks out of every sweep -- so the summary said "every suite green"
    # while the service boundary was unguarded. difftest_refactor.py is a
    # snapshot dumper with no verdict; classify() returns "skip" for it and the
    # filter below drops it, which is why this needs no special case.
    names = sorted(f for f in os.listdir(HERE)
                   if f.startswith(("test_", "difftest_")) and f.endswith(".py")
                   and (not args.k or args.k in f))
    work = [(n, classify(os.path.join(HERE, n))) for n in names]
    work = [(n, k) for n, k in work if k != "skip"]

    if args.list:
        for n, k in work:
            print("%-8s %s" % (k, n))
        print("\n%d files (%d scripts, %d pytest)"
              % (len(work), sum(k == "script" for _, k in work),
                 sum(k == "pytest" for _, k in work)))
        return 0

    bad, slow, skipped = [], [], []
    done = 0
    with _cf.ThreadPoolExecutor(max_workers=args.j) as pool:
        futs = [pool.submit(run_one, n, k, args.timeout) for n, k in work]
        for fut in _cf.as_completed(futs):
            name, kind, code, dt, out = fut.result()
            done += 1
            mark = {0: "ok  ", "skip": "skip"}.get(code, "FAIL")
            if code == "skip":
                skipped.append(name)
            elif code != 0:
                bad.append((name, code, out))
            if dt > 60:
                slow.append((name, dt))
            print("[%3d/%3d] %s %-52s %6.1fs" % (done, len(work), mark, name, dt),
                  flush=True)

    print("\n" + "=" * 72)
    for name, code, out in sorted(bad):
        print("\n--- %s (exit %s) ---" % (name, code))
        lines = out.strip().splitlines()
        tail = lines[-25:]
        # The tail alone can be all log noise (stderr comes last), which hid
        # the one FAIL line that said why a suite went red on CI.
        why = [ln for ln in lines[:-25] if _FAIL_LINE.match(ln)][:15]
        if why:
            print("\n".join(why))
            print("  ...")
        print("\n".join(tail))
    if slow:
        print("\nslowest:")
        for name, dt in sorted(slow, key=lambda r: -r[1])[:10]:
            print("  %6.1fs  %s" % (dt, name))
    if skipped:
        print("")
        print("%d suites did not run (they need a live LM Studio / ComfyUI):"
              % len(skipped))
        for n in sorted(skipped):
            print("   " + n)

    hollow = [n for n, k in work if k == "pytest" and counting_only(n)]
    if hollow:
        # Not a failure, but the number below is smaller than it looks: these
        # files define a check() that only COUNTS. pytest collects their test_
        # functions, every one returns normally whatever check() saw, and the
        # suite reports green while asserting nothing. Verified by mutation on
        # test_image_size_choice, which passed with a hardcoded 960x544 in the
        # size path. The fix per file is one line -- raise under
        # PYTEST_CURRENT_TEST -- but their check() signatures differ, so they
        # are listed rather than rewritten in bulk.
        print("")
        print("%d suites report green without asserting "
              "(counting check(), collected by pytest):" % len(hollow))
        for n in hollow:
            print("   " + n)

    print("")
    print("%d/%d suites green" % (len(work) - len(bad) - len(skipped),
                                  len(work) - len(skipped)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

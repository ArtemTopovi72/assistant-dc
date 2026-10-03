"""tests/run_all.py's own classify() must not silently drop real suites.

Live 2026-09-18: adding test_tg_style_button.py (a real, offline, check()-
based suite whose sys.exit lives under `if __name__ == "__main__":`) turned
out to be MISSING from run_all's own printed summary -- classify()'s old
`^sys\\.exit\\(` regex only matched a line whose first character is the call,
so both "indented under a module-level if" and "compound `x; sys.exit(...)`"
shapes fell through every branch to "skip" and never ran at all. A sweep
that silently drops suites is worse than one that is merely slow: it reports
green while asserting nothing for the dropped files.

Run: venv/Scripts/python.exe tests/test_run_all_classify.py
"""
import os, sys, importlib.util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("run_all", os.path.join(HERE, "run_all.py"))
run_all = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_all)

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")


def classify_src(src: str) -> str:
    import tempfile
    p = os.path.join(tempfile.mkdtemp(prefix="classify_"), "t.py")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(src)
    return run_all.classify(p)


# Compound statement: `print(...); sys.exit(...)` on one line.
check("compound `; sys.exit(` at module level is a script",
      classify_src("import sys\nprint('x'); sys.exit(1 if False else 0)\n") == "script")

# sys.exit nested one level inside a module-level `if` block.
check("sys.exit under a module-level `if` is a script",
      classify_src("import sys\nBAD = 0\nif BAD:\n    sys.exit(1)\n") == "script")

# sys.exit under `if __name__ == '__main__':` (test_tg_style_button.py's shape).
check("sys.exit under `if __name__ == '__main__':` is a script",
      classify_src("import sys\nBAD = 0\nif __name__ == '__main__':\n"
                   "    sys.exit(0 if BAD == 0 else 1)\n") == "script")

# sys.exit inside a function definition must NOT count -- that is a helper
# guard, not the suite announcing its own final verdict (test_vad.py has
# nothing like this at all, but a false positive here would misclassify any
# ordinary script that merely imports a module with an early-exit CLI guard).
check("sys.exit inside a def is NOT a module-level exit",
      classify_src("import sys\ndef helper():\n    sys.exit(1)\n") == "skip")

# The explicit opt-out must win even when the shape would otherwise match.
check("NOT_IN_RUN_ALL wins over a module-level exit",
      classify_src("NOT_IN_RUN_ALL = True\nimport sys\nif True:\n    sys.exit(1)\n") == "skip")

# The suites this bug actually hid, verified against the real files on disk.
REAL_SCRIPTS = ["test_tg_style_button.py", "test_picture_question.py",
                "test_rag_question_translation.py", "test_library_fixes.py",
                "test_tg_fwd_voice_batch.py", "test_deck_informativeness.py",
                "test_vad_synthetic.py"]
for name in REAL_SCRIPTS:
    path = os.path.join(HERE, name)
    check(f"{name} is classified as a runnable script, not skipped",
          os.path.exists(path) and run_all.classify(path) == "script")

# The two legitimate opt-outs must still be skipped.
check("test_agent_dialogue.py (live LM Studio audit) opts out via NOT_IN_RUN_ALL",
      run_all.classify(os.path.join(HERE, "test_agent_dialogue.py")) == "skip")
check("test_vad.py (interactive mic tool, no assertions) is still skip",
      run_all.classify(os.path.join(HERE, "test_vad.py")) == "skip")

print(f"\n{OK}/{OK + BAD} checks passed")
sys.exit(1 if BAD else 0)

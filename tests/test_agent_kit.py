"""The coding-agent kit: code_outline, run_tests, update_plan, undo_edit.

Driven through tools.execute_tool (the real dispatcher + pydantic args), with a
real Sandbox and a real Python subprocess for the tests -- stubs would hide
exactly the argv / exit-code handling that run_tests exists for.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest

import tools as T
import sandbox_access as A
import tool_code_handlers as H
import tool_retrieval
from code_sandbox import Sandbox


class _Ctx:
    def __init__(self, sandbox=None, user=None):
        self.sandbox = sandbox
        self.sandbox_user = user


class _User:
    def __init__(self, level):
        self.prefs = {"sandbox": level}


@pytest.fixture
def box(tmp_path):
    return Sandbox(tmp_path / "sbx")


@pytest.fixture
def ctx(box):
    return _Ctx(box, _User(A.HOST))   # CI has no Linux engine; with one, HOST still runs in it


def _call(ctx, name, **args):
    return T.execute_tool(ctx, {}, name, args)


# --- registration -----------------------------------------------------------

def test_registered_everywhere():
    names = {t.name for t in T.TOOLS}
    for n in ("code_outline", "run_tests", "update_plan", "undo_edit"):
        assert n in names and n in H.CODE_TOOL_NAMES, n
    assert "run_tests" in H.EXECUTING_TOOL_NAMES
    assert "code_outline" in tool_retrieval._CODE_KIT


# --- code_outline -----------------------------------------------------------

def test_outline_lists_python_and_java(ctx, box):
    box.write_text("app/core.py", "class Shop:\n    def buy(self):\n        pass\n\ndef main():\n    pass\n")
    box.write_text("src/Mod.java", "public class Mod {\n    public void onInit() {\n    }\n}\n")
    out = _call(ctx, "code_outline")
    assert "app/core.py" in out and "1: class Shop" in out and "2:   def buy()" in out
    assert "5: def main()" in out
    assert "src/Mod.java" in out and "class Mod" in out and "onInit()" in out
    assert ".agent" not in out


def test_outline_reports_a_broken_python_file(ctx, box):
    box.write_text("bad.py", "def x(:\n")
    assert "SyntaxError" in _call(ctx, "code_outline", path="bad.py")


def test_outline_on_empty_folder(ctx):
    assert "No source files" in _call(ctx, "code_outline")


# --- undo_edit --------------------------------------------------------------

def test_undo_restores_the_previous_edit_and_steps_back(ctx, box):
    box.write_text("a.py", "x = 1\n")
    _call(ctx, "edit_file", path="a.py", old="x = 1", new="x = 2")
    _call(ctx, "edit_file", path="a.py", old="x = 2", new="x = 3")
    out = _call(ctx, "undo_edit", path="a.py")
    assert "restored" in out and box.read_text("a.py") == "x = 2\n", out
    _call(ctx, "undo_edit", path="a.py")
    assert box.read_text("a.py") == "x = 1\n"
    assert "[TOOL ERROR]" in _call(ctx, "undo_edit", path="a.py")


def test_undo_of_a_created_file_removes_it(ctx, box):
    _call(ctx, "write_file", path="new.py", content="print(1)\n")
    out = _call(ctx, "undo_edit", path="new.py")
    assert "removed" in out and not (box.root / "new.py").exists(), out


def test_a_refused_edit_leaves_no_undo_point(ctx, box):
    box.write_text("a.py", "x = 1\n")
    assert "[TOOL ERROR]" in _call(ctx, "edit_file", path="a.py", old="nope", new="y")
    assert "[TOOL ERROR]" in _call(ctx, "undo_edit", path="a.py")


def test_undo_keeps_a_bounded_history(ctx, box):
    box.write_text("a.txt", "0")
    for i in range(H._UNDO_KEEP + 5):
        _call(ctx, "write_file", path="a.txt", content=str(i + 1))
    slot = H._undo_slot(box, "a.txt")
    assert len(list(slot.iterdir())) == H._UNDO_KEEP


# --- update_plan ------------------------------------------------------------

def test_plan_is_saved_and_shown_in_the_record(ctx, box):
    out = _call(ctx, "update_plan", steps=[
        {"step": "unpack mods.rar", "status": "done"},
        {"step": "fix the tag file", "status": "doing"},
        {"step": "build the jar"}])
    assert "1/3 done" in out, out
    rec = H.read_record(box)
    assert "[x] 1. unpack mods.rar" in rec and "[>] 2. fix the tag file" in rec
    assert "[ ] 3. build the jar" in rec


def test_finished_plan_is_not_shown(ctx, box):
    _call(ctx, "update_plan", steps=[{"step": "a", "status": "done"}])
    assert "plan" not in H.read_record(box).lower()


def test_two_doing_steps_are_refused(ctx):
    out = _call(ctx, "update_plan", steps=[{"step": "a", "status": "doing"},
                                           {"step": "b", "status": "doing"}])
    assert "[TOOL ERROR]" in out and "ONE" in out


def test_bad_status_is_rejected_by_the_schema(ctx):
    assert "[TOOL ERROR]" in _call(ctx, "update_plan", steps=[{"step": "a", "status": "wip"}])


# --- run_tests (real subprocess; installs pytest into the sandbox if absent) ---

def test_run_tests_pulls_out_the_failure(ctx, box):
    box.write_text("calc.py", "def add(a, b):\n    return a - b\n")
    box.write_text("test_calc.py", "from calc import add\n\ndef test_add():\n    assert add(1, 2) == 3\n\ndef test_ok():\n    assert True\n")
    out = _call(ctx, "run_tests")
    assert out.startswith("[TOOL ERROR] Tests failed"), out[:300]
    assert "FAILED test_calc.py::test_add" in out and "1 failed, 1 passed" in out, out
    box.write_text("calc.py", "def add(a, b):\n    return a + b\n")
    out = _call(ctx, "run_tests")
    assert out.startswith("All tests passed") and "2 passed" in out, out[:300]


def test_run_tests_with_no_tests_says_how_to_write_one(ctx, box):
    box.write_text("calc.py", "x = 1\n")
    out = _call(ctx, "run_tests")
    assert "No tests were found" in out and "test_*.py" in out, out


def test_run_tests_refused_without_the_code_grant(box):
    out = _call(_Ctx(box, _User(A.FILES)), "run_tests")
    assert "[TOOL ERROR]" in out and "not run" in out


def test_run_tests_refuses_a_path_outside_the_folder(ctx):
    assert "[TOOL ERROR]" in _call(ctx, "run_tests", path="../../etc")


def test_digest_parses_pytest_summary():
    out = ("FAILED t.py::test_a - assert 1 == 2\nERROR t.py::test_b - ImportError\n"
           "==== 1 failed, 3 passed, 1 error in 0.1s ====\n")
    d = H._test_digest(out)
    assert "Result: 1 failed, 3 passed, 1 error in 0.1s" in d
    assert "FAILED t.py::test_a -- assert 1 == 2" in d and "ERROR t.py::test_b" in d


# --- retrieval --------------------------------------------------------------

def test_run_tests_imports_the_package_from_a_tests_folder(ctx, box):
    # Real-model run 2026-09-23: tests/ without __init__.py could not import the
    # package beside it, every run was "ERROR tests/test_cart.py".
    box.write_text("shop/__init__.py", "")
    box.write_text("shop/money.py", "def cents(r):\n    return int(r * 100)\n")
    box.write_text("tests/test_money.py", "from shop.money import cents\n\ndef test_c():\n    assert cents(2) == 200\n")
    out = _call(ctx, "run_tests")
    assert out.startswith("All tests passed") and "Result: 1 passed" in out, out[:400]


def test_digest_reads_the_unframed_quiet_summary():
    assert "Result: 1 failed, 2 passed in 0.12s" in H._test_digest("1 failed, 2 passed in 0.12s\n")


# --- mechanics ported from SWE-agent / Aider / mini-swe-agent -----------------

def test_read_file_is_a_window_on_a_long_file(ctx, box):
    box.write_text("big.py", "".join(f"x{i} = {i}\n" for i in range(1, 1001)))
    out = _call(ctx, "read_file", path="big.py")
    assert "lines 1-120 of 1000" in out and "  120| x120 = 120" in out
    assert "x121" not in out and "start=121" in out
    out = _call(ctx, "read_file", path="big.py", start=990)
    assert "lines 990-1000 of 1000" in out and "1000| x1000" in out and "more lines" not in out


def test_read_file_windows_a_file_over_the_whole_read_limit(ctx, box):
    import code_sandbox
    line = "a" * 99 + "\n"
    box.write_text("log.txt", "x")
    (box.root / "log.txt").write_text(line * (code_sandbox.MAX_READ_BYTES // 100 + 50))
    out = _call(ctx, "read_file", path="log.txt", start=3, lines=20)
    assert "lines 3-22 of" in out, out[:200]


def test_edit_that_breaks_python_is_refused_and_file_kept(ctx, box):
    box.write_text("m.py", "def f():\n    return 1\n")
    out = _call(ctx, "edit_file", path="m.py", old="    return 1", new="    return (1")
    assert "[TOOL ERROR]" in out and "NOT applied" in out, out
    assert box.read_text("m.py") == "def f():\n    return 1\n"
    assert "[TOOL ERROR]" in _call(ctx, "undo_edit", path="m.py")   # nothing to undo


def test_edit_of_an_already_broken_file_is_allowed(ctx, box):
    box.write_text("m.py", "def f(:\n    return 1\n")
    out = _call(ctx, "edit_file", path="m.py", old="def f(:", new="def f():")
    assert out.startswith("Edited"), out


def test_edit_with_wrong_indentation_is_matched_and_reindented(ctx, box):
    box.write_text("m.py", "class A:\n    def f(self):\n        return 1\n")
    out = _call(ctx, "edit_file", path="m.py",
                old="def f(self):\n    return 1", new="def f(self):\n    return 2")
    assert out.startswith("Edited") and "re-indented" in out, out
    assert box.read_text("m.py") == "class A:\n    def f(self):\n        return 2\n"


def test_fuzzy_match_refuses_when_ambiguous(ctx, box):
    box.write_text("m.py", "if a:\n    x = 1\nif b:\n        x = 1\n")
    out = _call(ctx, "edit_file", path="m.py", old="x = 1  ", new="x = 2")
    assert "[TOOL ERROR]" in out


def test_silent_script_says_it_printed_nothing(ctx, box):
    out = _call(ctx, "run_code", code="x = 1")
    assert "printed nothing" in out, out


def test_outline_ranks_the_referenced_core_first(ctx, box):
    box.write_text("aaa_leaf.py", "def leaf():\n    pass\n")
    box.write_text("zzz_core.py", "class Engine:\n    pass\n")
    for i in range(3):
        box.write_text(f"use{i}.py", "from zzz_core import Engine\nEngine()\n")
    out = _call(ctx, "code_outline")
    assert out.index("zzz_core.py") < out.index("aaa_leaf.py"), out


def test_run_code_imports_the_projects_own_modules(ctx, box):
    # Real-model run 2026-09-23: scripts live in .agent/, so `import slug` failed
    # and the hint told the model to pip-install a stranger's 'slug' package.
    box.write_text("slug.py", "def slugify(s):\n    return s.lower()\n")
    out = _call(ctx, "run_code", code="from slug import slugify\nprint(slugify('AB'))")
    assert "ab" in out and "[TOOL ERROR]" not in out, out


def test_host_runner_also_sees_the_project_root(box):
    import sys as _sys
    import code_runner
    box.write_text("mymod.py", "X = 41\n")
    box.write_text(".agent/probe.py", "import mymod\nprint(mymod.X + 1)\n")
    res = code_runner._spawn([_sys.executable, "-u", str(box.root / ".agent/probe.py")],
                             cwd=box.root, timeout=60)
    assert res.ok and "42" in res.output, res.output


def test_a_missing_local_module_is_not_sent_to_pip(ctx, box):
    box.write_text("pkg/helper.py", "X = 1\n")          # not at the top level
    box.write_text("helper.txt", "")
    box.write_text("broken.py", "")
    out = _call(ctx, "run_code", code="import broken.sub")
    assert "do NOT install" in out and "install_packages with" not in out, out


def test_inconsistent_indentation_is_not_repaired(ctx, box):
    # Real-model run: `old` had one line indented and the next not; the repair
    # indented `temperature`, configparser read it as a continuation of num_ctx.
    ini = "[LLM]\nnum_ctx = 8192\ntemperature = 0.85\nmax_tokens = 250\n"
    box.write_text("c.ini", ini)
    out = _call(ctx, "edit_file", path="c.ini",
                old="num_ctx = 8192\n    temperature = 0.85",
                new="num_ctx = 16384\n    temperature = 0.6")
    assert "[TOOL ERROR]" in out, out
    assert box.read_text("c.ini") == ini


def test_list_files_shows_a_tree_so_a_mod_is_one_call(ctx, box):
    # Real-model run: one level per call spent 12 of 16 rounds walking a mod.
    box.write_text("m_unpacked/data/thief/tags/block/medium.json", "{}")
    out = _call(ctx, "list_files", path="m_unpacked")
    assert "m_unpacked/data/thief/" in out and "not expanded" in out, out
    out = _call(ctx, "list_files", path="m_unpacked", depth=5)
    assert "m_unpacked/data/thief/tags/block/medium.json" in out, out


def test_list_files_output_is_capped(ctx, box):
    for i in range(200):
        box.write_text(f"d/f{i:03}.txt", "")
    out = _call(ctx, "list_files")
    assert "more entries not shown" in out and out.count("\n") <= 152, out.count("\n")


def test_files_command_listing_stays_one_level(box):
    box.write_text("a/b/c.txt", "")
    assert box.list_dir(".") == ["a/"]

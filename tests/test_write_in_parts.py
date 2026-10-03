"""Code is written in parts and repaired in place, never rewritten whole.

Sandbox bench 2026-09-24: the regex engine was rewritten whole seven times in
one turn; four rewrites were cut off at the token ceiling. One typo in a big
file must cost one edit_file, not the whole file again.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("F5_TEST_RUN", "1")
import pytest

import tools as T
import sandbox_access as A
from code_sandbox import Sandbox


class _User:
    def __init__(self, level):
        self.prefs = {"sandbox": level}


class _Ctx:
    def __init__(self, sandbox):
        self.sandbox = sandbox
        self.sandbox_user = _User(A.CODE)


@pytest.fixture
def ctx(tmp_path):
    return _Ctx(Sandbox(tmp_path / "sbx"))


def _call(ctx, name, **args):
    return T.execute_tool(ctx, {}, name, args)


def _code(n, tag="x"):
    return "\n".join(f"{tag}{i} = {i}" for i in range(n)) + "\n"


def test_complete_long_new_file_is_accepted(ctx):
    # its tokens are already spent; refusing it only throws them away
    assert _call(ctx, "write_file", path="rx.py", content=_code(200)).startswith("Wrote")


def test_long_file_written_in_parts(ctx):
    assert _call(ctx, "write_file", path="rx.py", content=_code(120, "a")).startswith("Wrote")
    r = _call(ctx, "write_file", path="rx.py", content=_code(120, "b"), append=True)
    assert r.startswith("Appended 120 lines") and "now 2" in r
    text = (ctx.sandbox.root / "rx.py").read_text(encoding="utf-8")
    assert "a0 = 0" in text and "b119 = 119" in text


def test_existing_code_is_not_rewritten_whole(ctx):
    _call(ctx, "write_file", path="rx.py", content=_code(100))
    r = _call(ctx, "write_file", path="rx.py", content=_code(100, "y"))
    assert r.startswith("[TOOL ERROR]") and "edit_file" in r and "read_file" in r
    assert "x5 = 5" in (ctx.sandbox.root / "rx.py").read_text(encoding="utf-8")
    # ...and the one-line fix goes through edit_file
    assert _call(ctx, "edit_file", path="rx.py", old="x5 = 5", new="x5 = 55").startswith("Edited")


def test_small_files_and_non_code_can_be_replaced(ctx):
    _call(ctx, "write_file", path="tiny.py", content=_code(10))
    assert _call(ctx, "write_file", path="tiny.py", content=_code(12, "z")).startswith("Wrote")
    _call(ctx, "write_file", path="notes.txt", content=_code(100))
    assert _call(ctx, "write_file", path="notes.txt", content=_code(90, "q")).startswith("Wrote")


def test_append_needs_an_existing_file(ctx):
    r = _call(ctx, "write_file", path="new.py", content="x = 1", append=True)
    assert r.startswith("[TOOL ERROR]") and "does not exist" in r


def test_start_over_after_delete(ctx):
    _call(ctx, "write_file", path="rx.py", content=_code(100))
    _call(ctx, "delete_path", path="rx.py")
    assert _call(ctx, "write_file", path="rx.py", content=_code(100, "n")).startswith("Wrote")


def test_edit_by_line_range(ctx):
    _call(ctx, "write_file", path="m.py", content="a = 1\nb = 2\nc = 3\n")
    r = _call(ctx, "edit_file", path="m.py", start_line=2, end_line=2, new="    2| b = 20")
    assert r.startswith("Edited") and "lines 2-2" in r
    assert (ctx.sandbox.root / "m.py").read_text(encoding="utf-8").rstrip() == "a = 1\nb = 20\nc = 3"


def test_line_edit_that_breaks_syntax_is_reverted(ctx):
    _call(ctx, "write_file", path="m.py", content="a = 1\nb = 2\n")
    r = _call(ctx, "edit_file", path="m.py", start_line=1, end_line=1, new="a = (")
    assert r.startswith("[TOOL ERROR]") and "NOT applied" in r
    assert (ctx.sandbox.root / "m.py").read_text(encoding="utf-8").rstrip() == "a = 1\nb = 2"


def test_line_range_outside_the_file(ctx):
    _call(ctx, "write_file", path="m.py", content="a = 1\n")
    r = _call(ctx, "edit_file", path="m.py", start_line=5, end_line=6, new="x")
    assert r.startswith("[TOOL ERROR]") and "outside" in r


def test_edit_needs_old_or_lines(ctx):
    _call(ctx, "write_file", path="m.py", content="a = 1\n")
    assert _call(ctx, "edit_file", path="m.py", new="x").startswith("[TOOL ERROR]")


def test_indentation_survives_the_args_model(ctx):
    _call(ctx, "write_file", path="k.py", content="class K:\n    def f(self):\n        return 1\n")
    r = _call(ctx, "edit_file", path="k.py", old="        return 1", new="        return 2")
    assert r.startswith("Edited"), r
    r = _call(ctx, "write_file", path="k.py", content="    def g(self):\n        return 3", append=True)
    assert r.startswith("Appended"), r
    src = (ctx.sandbox.root / "k.py").read_text(encoding="utf-8")
    assert "        return 2" in src and "    def g(self):" in src
    compile(src, "k.py", "exec")


def test_run_code_cannot_overwrite_a_long_code_file(ctx):
    # bench 2026-09-24: "let's use run_code to overwrite rx.py completely"
    _call(ctx, "write_file", path="rx.py", content=_code(100))
    for script in ('full_code = """x = 1"""\nwith open("rx.py", "w") as f:\n    f.write(full_code)\n',
                   "open('/work/rx.py', mode='w').write('x')\n",
                   "from pathlib import Path\nPath('rx.py').write_text('x')\n"):
        r = _call(ctx, "run_code", code=script)
        assert r.startswith("[TOOL ERROR] NOT run") and "is not rewritten whole" in r, r
    assert "x5 = 5" in (ctx.sandbox.root / "rx.py").read_text(encoding="utf-8")


def test_run_code_may_still_read_append_or_write_new_files(ctx):
    _call(ctx, "write_file", path="rx.py", content=_code(100))
    from tool_code_handlers import _rewrites_code_file
    box = ctx.sandbox
    assert _rewrites_code_file(box, "print(open('rx.py').read())") == ""
    assert _rewrites_code_file(box, "open('rx.py', 'a').write('#')") == ""
    assert _rewrites_code_file(box, "open('new.py', 'w').write('x')") == ""
    assert _rewrites_code_file(box, "open('out.txt', 'w').write('x')") == ""

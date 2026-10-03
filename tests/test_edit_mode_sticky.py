"""After a refused whole-file rewrite the loop holds EDIT MODE.

Sandbox bench 2026-09-24 (regex case): one forced read_file|edit_file round was
not enough -- the model read the file, then resent the whole file next round.
Now: one `required` round, then EDIT_MODE_ROUNDS more rounds with no
write_file offered but tool_choice=auto, so the model can still finish.
"""
import os, sys, types, pathlib
os.environ["F5_TEST_RUN"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import test_graph_full as T

REF = ("[TOOL ERROR] NOT written: rx.py already exists (205 lines) and is not "
       "rewritten whole. Fix it in place")


def test_edit_mode_is_sticky_but_not_required():
    seen = []
    script = [T._asst("", [T._tc("write_file", {"path": "rx.py", "content": "x=1\n" * 50})]),
              T._asst("", [T._tc("write_file", {"path": "rx.py", "content": "y=1\n" * 50})]),
              T._asst("", [T._tc("read_file", {"path": "rx.py"})]),
              T._asst("", [T._tc("read_file", {"path": "rx.py", "start": 40})]),
              T._asst("готово", [])]

    def llm(ctx, messages, tools=None, **kw):
        seen.append(({t["function"]["name"] for t in (tools or [])}, kw.get("tool_choice")))
        return script.pop(0) if script else T._asst("done", [])

    tools = T.Tools({"write_file": ["Wrote rx.py.", REF], "read_file": "1| x=1"})
    box = types.SimpleNamespace(root=pathlib.Path("."))
    st = {"user_input": "Напиши rx.py — движок регулярных выражений, прогони pytest",
          "messages": []}
    T._run(T._ctx(sandbox=box, sandbox_user=None), llm, tools, st, fastpath=False)
    assert "write_file" in seen[1][0]                    # before the refusal
    # forced round: only these two schemas; sent "auto" first by design (graph_personality)
    assert seen[2][0] == {"read_file", "edit_file"} and seen[2][1] in ("auto", "required"), seen[2]
    for names, choice in seen[3:5]:
        assert "write_file" not in names and choice == "auto", (names, choice)


if __name__ == "__main__":
    test_edit_mode_is_sticky_but_not_required()
    print("1/1 passed")

"""The history commit must keep THIS turn's user message.

Torture run #5 2026-09-14: the graph sanitizes (drops tool turns) and trims
the persisted history before appending the turn, so `new_msgs[len(old):]`
skipped the user message whenever the old history held tool results. The
archive requests were never in the transcript; «заново» then redid the cat
edit from six turns earlier.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tg_tasks as T


def _old_with_tools():
    return [{"role": "system", "content": "s"},
            {"role": "user", "content": "make the cat black"},
            {"role": "assistant", "content": "", "tool_calls": [{}]},
            {"role": "tool", "content": "edited"},
            {"role": "assistant", "content": "", "tool_calls": [{}]},
            {"role": "tool", "content": "inspected"},
            {"role": "assistant", "content": "done"}]


def test_turn_starts_at_its_own_user_message():
    old = _old_with_tools()
    # what the graph hands back: sanitized old history + this turn
    new = [old[0], old[1], old[-1],
           {"role": "user", "content": "find all bridges"},
           {"role": "assistant", "content": "", "tool_calls": [{}]},
           {"role": "tool", "content": "8 photos"},
           {"role": "assistant", "content": "here"}]
    added = T._turn_messages(new, len(old), ("find all bridges", "найди все мосты"))
    assert [m["role"] for m in added] == ["user", "assistant", "tool", "assistant"]
    assert added[0]["content"] == "find all bridges"
    # the naive slice is what lost the request
    assert "user" not in [m["role"] for m in new[len(old):]]


def test_falls_back_to_the_slice_when_the_input_is_not_found():
    old = [{"role": "system", "content": "s"}]
    new = old + [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    assert T._turn_messages(new, 1, ("nothing-like-it", "")) == new[1:]
    assert T._turn_messages(new, 1, ("x", "")) == new[1:]


def test_wired_into_the_commit():
    import inspect
    src = inspect.getsource(T.TaskRunnerMixin)
    assert "_turn_messages(new_msgs, _hist_snapshot_len" in src


if __name__ == "__main__":
    test_turn_starts_at_its_own_user_message()
    test_falls_back_to_the_slice_when_the_input_is_not_found()
    test_wired_into_the_commit()
    print("3/3 passed")

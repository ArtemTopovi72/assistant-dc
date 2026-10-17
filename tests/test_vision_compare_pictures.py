"""«что общего у двух фото, которые я прислал?»: the vision overview read only the
current picture and the bot answered "you sent just one photo".
When the user relates several pictures the earlier ones are read too."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("F5_TEST_RUN", "1")

import graph


def _yes(answer):
    """The model's yes/no, as the context hands it to the code (core/seams.py)."""
    return lambda q, t, default=False, wait=True: answer(q, t)


class _Ctx:
    def __init__(self, paths, last, yes=lambda q, t: False, look=None):
        self.vision_model = look
        self.image_undo = paths
        self.last_image_path = last
        self.yes_no = _yes(yes)


def _files(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    return str(a), str(b)


def test_earlier_picture_is_read_when_the_user_relates_pictures(tmp_path, monkeypatch):
    a, b = _files(tmp_path)
    calls = []
    look = lambda **k: (calls.append(k), "a till receipt")[1]
    out = graph._add_earlier_pictures(_Ctx([a], b, yes=lambda q, t: True, look=look), {"user_input": "что общего у двух фото?"}, "a man")
    assert "a man" in out and "a till receipt" in out
    # one call seeing both: one at a time, each said "there is no second picture"
    assert len(calls) == 1 and [p for _, p in calls[0]["image_paths"]] == [a, b]


def test_single_picture_question_reads_only_the_current_one(tmp_path, monkeypatch):
    a, b = _files(tmp_path)
    calls = []
    out = graph._add_earlier_pictures(_Ctx([a], b, look=lambda **k: calls.append(1) or "x"), {"user_input": "что на фото?"}, "a man")
    assert out == "a man" and not calls


def test_inspection_description_that_misses_the_request_counts_as_negative():
    import types
    import graph_personality as gp
    def pick(q, t, options, default=""):
        assert "{text}" in q and set(options) == {"met", "partly", "not"}
        return "not" if "woman" in t.lower() else "partly" if "three" in t.lower() else "met"
    ctx = types.SimpleNamespace(choice=pick)
    edit = {"inpaint_image"}
    assert gp._inspection_misses_request("пусть котов будет два", "Three grey cats sit. | Two cats: PRESENT", edit, ctx=ctx) == "partial"
    assert gp._inspection_misses_request("одень его в доспехи", "A woman in armor was added. | Armor: PRESENT", edit, ctx=ctx) is True
    assert gp._inspection_misses_request("пусть котов будет два", "Two cats sit.", edit, ctx=ctx) is False
    # nothing was made this turn -> no verdict about a requested change
    assert not gp._inspection_misses_request("пусть котов будет два", "Three grey cats", set(), ctx=ctx)

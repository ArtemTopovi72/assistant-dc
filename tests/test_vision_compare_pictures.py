"""«что общего у двух фото, которые я прислал?»: the vision overview read only the
current picture and the bot answered "you sent just one photo" (live 2026-10-08).
When the user relates several pictures the earlier ones are read too."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("F5_TEST_RUN", "1")

import graph
import intent


class _Ctx:
    def __init__(self, paths, last):
        self.image_undo = paths
        self.last_image_path = last


def _files(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    return str(a), str(b)


def test_earlier_picture_is_read_when_the_user_relates_pictures(tmp_path, monkeypatch):
    a, b = _files(tmp_path)
    monkeypatch.setattr(graph, "analyze_image_with_llm", lambda **k: "a till receipt")
    monkeypatch.setattr(graph, "downscale_image_bytes", lambda x: x)
    monkeypatch.setattr(intent, "YES_STUB", lambda q, t: True, raising=False)
    out = graph._add_earlier_pictures(_Ctx([a], b), {"user_input": "что общего у двух фото?"}, "a man")
    assert "a man" in out and "a till receipt" in out and "Earlier picture 1 of 1" in out


def test_single_picture_question_reads_only_the_current_one(tmp_path, monkeypatch):
    a, b = _files(tmp_path)
    calls = []
    monkeypatch.setattr(graph, "analyze_image_with_llm", lambda **k: calls.append(1) or "x")
    monkeypatch.setattr(intent, "YES_STUB", lambda q, t: False, raising=False)
    out = graph._add_earlier_pictures(_Ctx([a], b), {"user_input": "что на фото?"}, "a man")
    assert out == "a man" and not calls


def test_inspection_description_that_misses_the_request_counts_as_negative(monkeypatch):
    import graph_personality as gp
    monkeypatch.setattr(intent, "YES_STUB", lambda q, t: "three" in t.lower(), raising=False)
    edit = {"inpaint_image"}
    assert gp._inspection_misses_request("пусть котов будет два", "Three grey cats sit. | Two cats: PRESENT", edit)
    assert not gp._inspection_misses_request("пусть котов будет два", "Two cats sit.", edit)
    # nothing was made this turn -> no verdict about a requested change
    assert not gp._inspection_misses_request("пусть котов будет два", "Three grey cats", set())

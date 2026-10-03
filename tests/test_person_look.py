"""A real person named in a drawing request reaches the engine with their looks.

Live 10-03: «кибернетический Ленин …» was drawn with hair and «какой-то урод»;
the engine does not know faces by name. person_look adds the recognisable
traits to the description (generate) and to the instruction (redraw).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("F5_TEST_RUN", "1")

import person_look as P  # noqa: E402

LOOK = "bald head, reddish goatee and moustache, high domed forehead, narrow eyes"


def _stub(text):
    return ("Vladimir Lenin", LOOK) if "Ленин" in text or "Lenin" in text else ("", "")


def test_a_named_person_gets_their_looks(monkeypatch):
    monkeypatch.setattr(P, "STUB", _stub)
    P._CACHE.clear()
    out = P.with_look("Кибернетический Ленин дерётся с чертями в аду")
    assert out.startswith("Кибернетический Ленин") and "bald head" in out


def test_a_generic_subject_is_left_alone(monkeypatch):
    monkeypatch.setattr(P, "STUB", _stub)
    P._CACHE.clear()
    assert P.with_look("кот в шляпе") == "кот в шляпе"


def test_a_redraw_reads_the_name_from_the_picture_prompt(monkeypatch):
    monkeypatch.setattr(P, "STUB", _stub)
    P._CACHE.clear()
    out = P.with_look("он должен быть лысым", "cyber Lenin fighting demons")
    assert out.startswith("он должен быть лысым") and "goatee" in out


def test_a_failing_model_changes_nothing(monkeypatch):
    def boom(t):
        raise RuntimeError("down")
    monkeypatch.setattr(P, "STUB", boom)
    P._CACHE.clear()
    assert P.with_look("Ленин") == "Ленин"


def test_both_handlers_use_it():
    src = open(os.path.join(ROOT, "agent", "tool_image_handlers.py"), encoding="utf-8").read()
    assert "description = person_look.with_look(description)" in src
    assert "instructions = person_look.with_look(instructions," in src

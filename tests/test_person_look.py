"""A real person named in a drawing request reaches the engine with their looks.

The engine does not know faces by name («кибернетический Ленин …» would be drawn
with hair). person_look adds the recognisable traits to the description
(generate) and to the instruction (redraw).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("F5_TEST_RUN", "1")

import types  # noqa: E402

import person_look as P  # noqa: E402

LOOK = "bald head, reddish goatee and moustache, high domed forehead, narrow eyes"


def _stub(text):
    return ("Vladimir Lenin", LOOK) if "Ленин" in text or "Lenin" in text else ("", "")


def _ctx(look=_stub, search=None):
    """The model's read of the request (and the web search) handed in on the context."""
    deps = types.SimpleNamespace(person_look=look)
    if search is not None:
        deps.person_search = search
    return types.SimpleNamespace(deps=deps)


def test_a_named_person_gets_their_looks():
    P._CACHE.clear()
    out = P.with_look("Кибернетический Ленин дерётся с чертями в аду", ctx=_ctx())
    assert out.startswith("Кибернетический Ленин") and "bald head" in out


def test_a_generic_subject_is_left_alone():
    P._CACHE.clear()
    assert P.with_look("кот в шляпе", ctx=_ctx()) == "кот в шляпе"


def test_a_redraw_reads_the_name_from_the_picture_prompt():
    P._CACHE.clear()
    out = P.with_look("он должен быть лысым", "cyber Lenin fighting demons", ctx=_ctx())
    assert out.startswith("он должен быть лысым") and "goatee" in out


def test_a_failing_model_changes_nothing():
    def boom(t):
        raise RuntimeError("down")
    P._CACHE.clear()
    assert P.with_look("Ленин", ctx=_ctx(boom)) == "Ленин"


def test_both_handlers_use_it():
    src = open(os.path.join(ROOT, "agent", "tool_image_handlers.py"), encoding="utf-8").read()
    assert "description = person_look.with_look(description, ctx=ctx)" in src
    assert "instructions = person_look.with_look(instructions," in src


def test_the_looks_come_from_the_web_when_it_answers():
    """The looks are taken from a web search when it answers."""
    queries = []

    def stub(text):
        if text.startswith("WEB:"):
            assert "red goatee" in text
            return ("", "completely bald, short red goatee and moustache, high forehead")
        return _stub(text)
    P._CACHE.clear()
    out = P.with_look("Ленин в аду", ctx=_ctx(stub, lambda q: queries.append(q) or "Lenin was bald with a red goatee"))
    assert queries and queries[0].startswith("Vladimir Lenin appearance")
    assert "short red goatee" in out


def test_no_web_answer_falls_back_to_the_models_memory():
    P._CACHE.clear()
    assert "bald head" in P.with_look("Ленин", ctx=_ctx(_stub, lambda q: ""))

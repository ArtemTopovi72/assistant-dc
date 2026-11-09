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


def test_web_enabled_request_does_not_reuse_offline_appearance(monkeypatch):
    P._CACHE.clear()
    monkeypatch.setattr(P, '_ask', lambda text, ctx=None: ('Lenin', LOOK))
    monkeypatch.setattr(P, '_look_from_web', lambda ctx, person: 'verified bald head and red beard' if ctx.web_search_enabled else '')
    offline = types.SimpleNamespace(model_name='same-model', no_think=True, web_search_enabled=False)
    online = types.SimpleNamespace(model_name='same-model', no_think=True, web_search_enabled=True)
    assert LOOK in P.note('Lenin', offline)
    assert 'verified bald head and red beard' in P.note('Lenin', online)


def test_failed_web_lookup_can_be_retried(monkeypatch):
    P._CACHE.clear()
    monkeypatch.setattr(P, '_ask', lambda text, ctx=None: ('Lenin', LOOK))
    answers = iter(['', 'verified bald head and red beard'])
    monkeypatch.setattr(P, '_look_from_web', lambda ctx, person: next(answers))
    ctx = types.SimpleNamespace(model_name='same-model', no_think=True, web_search_enabled=True)
    assert LOOK in P.note('Lenin', ctx)
    assert 'verified bald head and red beard' in P.note('Lenin', ctx)


def test_failed_person_read_is_not_cached_and_receives_context(monkeypatch):
    monkeypatch.delenv('F5_TEST_RUN', raising=False)
    P._CACHE.clear()
    seen = []
    ctx = types.SimpleNamespace(model_name='model', no_think=True, web_search_enabled=False)
    def model(passed, *args, **kwargs):
        seen.append(passed)
        return None if len(seen) == 1 else '{"person": "Lenin", "look": "bald head and red beard"}'
    ctx.text_model = model
    assert P.note('Lenin', ctx) == ''
    assert 'bald head and red beard' in P.note('Lenin', ctx)
    assert seen == [ctx, ctx]


def test_web_disabled_does_not_call_search_dependency():
    calls = []
    ctx = _ctx(search=lambda q: calls.append(q))
    ctx.web_search_enabled = False
    assert P._search(ctx, 'Lenin') == ''
    assert calls == []


def test_invalid_person_fields_do_not_poison_cache(monkeypatch):
    monkeypatch.delenv('F5_TEST_RUN', raising=False)
    P._CACHE.clear()
    answers = iter(['{"person": null, "look": null}',
                    '{"person": "Lenin", "look": "bald head and red beard"}'])
    ctx = types.SimpleNamespace(model_name='model', no_think=True, web_search_enabled=False,
                                text_model=lambda *a, **k: next(answers))
    assert P.note('Lenin', ctx) == ''
    assert 'bald head and red beard' in P.note('Lenin', ctx)


def test_short_web_result_does_not_override_or_poison_cache(monkeypatch):
    P._CACHE.clear()
    monkeypatch.setattr(P, '_ask', lambda text, ctx=None: ('Lenin', LOOK))
    replies = iter(['bald head', 'verified bald head with red beard'])
    monkeypatch.setattr(P, '_look_from_web', lambda ctx, person: next(replies))
    ctx = types.SimpleNamespace(model_name='model', no_think=True, web_search_enabled=True)
    assert LOOK in P.note('Lenin', ctx)
    assert 'verified bald head with red beard' in P.note('Lenin', ctx)


def test_short_model_look_does_not_poison_offline_cache(monkeypatch):
    P._CACHE.clear()
    replies = iter([('Lenin', 'bald'), ('Lenin', LOOK)])
    monkeypatch.setattr(P, '_ask', lambda text, ctx=None: next(replies))
    ctx = types.SimpleNamespace(model_name='model', no_think=True, web_search_enabled=False)
    assert P.note('Lenin', ctx) == ''
    assert LOOK in P.note('Lenin', ctx)


def test_person_look_cache_does_not_mix_models(monkeypatch):
    P._CACHE.clear()
    monkeypatch.setattr(P, '_ask', lambda text, ctx=None: ('Lenin', ctx.model_name + ' appearance here'))
    a = types.SimpleNamespace(model_name='model-a', no_think=True, web_search_enabled=False)
    b = types.SimpleNamespace(model_name='model-b', no_think=True, web_search_enabled=False)
    assert 'model-a appearance here' in P.note('Lenin', a)
    assert 'model-b appearance here' in P.note('Lenin', b)

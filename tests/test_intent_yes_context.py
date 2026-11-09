from types import SimpleNamespace

import intent


def test_yes_no_uses_selected_model_and_separate_cache(monkeypatch):
    intent._YES_CACHE.clear()
    monkeypatch.delenv('F5_TEST_RUN', raising=False)
    seen = []
    def model(ctx, *args, **kwargs):
        seen.append(ctx.model_name)
        return 'yes' if ctx.model_name == 'first' else 'no'
    first = SimpleNamespace(model_name='first', no_think=True,
                            deps=SimpleNamespace(call_llm_simple=model))
    second = SimpleNamespace(model_name='second', no_think=True,
                             deps=SimpleNamespace(call_llm_simple=model))
    assert intent.ask_yes('Question: {text}', 'same text', ctx=first) is True
    assert intent.ask_yes('Question: {text}', 'same text', ctx=second) is False
    assert seen == ['first', 'second']
    first.no_think = False
    assert intent.ask_yes('Question: {text}', 'same text', ctx=first) is True
    assert seen == ['first', 'second', 'first']


def test_yes_no_honours_active_turn_context(monkeypatch):
    intent._YES_CACHE.clear()
    monkeypatch.delenv('F5_TEST_RUN', raising=False)
    seen = []
    def model(ctx, *args, **kwargs):
        seen.append(ctx)
        return 'yes'
    ctx = SimpleNamespace(model_name='selected', no_think=False,
                          deps=SimpleNamespace(call_llm_simple=model))
    with intent.turn_scope() as turn:
        turn.cancel_context = ctx
        assert intent.ask_yes('Question: {text}', 'test') is True
    assert seen == [ctx]


def test_choice_uses_turn_model_and_does_not_cache_invalid_reply(monkeypatch):
    import llm
    intent._YES_CACHE.clear()
    monkeypatch.delenv('F5_TEST_RUN', raising=False)
    replies = iter(['unclear', 'first', 'second'])
    seen = []
    def model(ctx, *args, **kwargs):
        seen.append(ctx.model_name)
        return next(replies)
    monkeypatch.setattr(llm, 'call_llm_simple', model)
    first = SimpleNamespace(model_name='first-model', no_think=True)
    second = SimpleNamespace(model_name='second-model', no_think=True)
    with intent.turn_scope() as turn:
        turn.cancel_context = first
        assert intent.ask_choice('Choose: {text}', 'same text', ('first', 'second')) == ''
        assert intent.ask_choice('Choose: {text}', 'same text', ('first', 'second')) == 'first'
    with intent.turn_scope() as turn:
        turn.cancel_context = second
        assert intent.ask_choice('Choose: {text}', 'same text', ('first', 'second')) == 'second'
    assert seen == ['first-model', 'first-model', 'second-model']


def test_cancelled_question_is_not_cached(monkeypatch):
    intent._YES_CACHE.clear()
    monkeypatch.delenv('F5_TEST_RUN', raising=False)
    state = {'cancelled': False}
    def model(ctx, *args, **kwargs):
        state['cancelled'] = True
        return 'yes'
    ctx = SimpleNamespace(model_name='test', no_think=False,
                          is_cancelled=lambda: state['cancelled'],
                          deps=SimpleNamespace(call_llm_simple=model))
    assert intent.ask_yes('Question: {text}', 'test', default=None, ctx=ctx) is None
    assert not intent._YES_CACHE

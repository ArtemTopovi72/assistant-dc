from types import SimpleNamespace

import video
import intent


def test_video_script_reads_use_active_context():
    words = 'один два три четыре пять шесть семь восемь девять десять одиннадцать двенадцать'
    seen = []
    ctx = SimpleNamespace(model_name='selected-model', no_think=True)

    def model(passed, system, user, **kwargs):
        seen.append(passed)
        if 'Copy, character for character' in system:
            return words
        if 'A picture is the FIRST FRAME' in system:
            return 'The cat jumps. It lands. It runs.'
        return 'Short result.'

    ctx.text_model = model
    assert video.dictated_speech('Мужчина говорит: «' + words + '»', 'speaks', ctx=ctx) == words
    assert video.condense_script('Long scene', 20, 10, ctx=ctx) == 'Short result.'
    assert video.action_script('Scene', ctx=ctx) == 'The cat jumps. It lands. It runs.'
    assert seen == [ctx, ctx, ctx]


def test_script_read_retries_empty_reasoning_with_task_model_and_cancellation():
    seen = []
    ctx = SimpleNamespace(model_name='selected-model', no_think=False,
                          cancel_event=object(), is_cancelled=lambda: False)

    def model(passed, system, user, **kwargs):
        seen.append(passed)
        return None if len(seen) == 1 else '{"kinds": ["look"]}'

    ctx.text_model = model
    with intent.turn_scope() as turn:
        turn.cancel_context = ctx
        assert video._ask_json('classify', ['The camera is still.'], ('kinds',), 8)['kinds'] == ['look']
    assert seen[0] is ctx and not ctx.no_think
    assert seen[1] is not ctx and seen[1].no_think
    assert seen[1].model_name == ctx.model_name and seen[1].cancel_event is ctx.cancel_event


def test_script_read_does_not_retry_after_cancel():
    seen = []
    ctx = SimpleNamespace(no_think=False, is_cancelled=lambda: True,
                          text_model=lambda *a, **k: seen.append(1))
    with intent.turn_scope() as turn:
        turn.cancel_context = ctx
        video._ask_json('classify', ['x'], ('kinds',), 8)
    assert seen == [1]


def test_script_read_retries_wrong_length_result():
    seen = []
    ctx = SimpleNamespace(model_name='selected-model', no_think=False, is_cancelled=lambda: False)
    def model(passed, *args, **kwargs):
        seen.append(passed)
        return '{"kinds": []}' if len(seen) == 1 else '{"kinds": ["look"]}'
    ctx.text_model = model
    with intent.turn_scope() as turn:
        turn.cancel_context = ctx
        assert video._ask_json('classify', ['x'], ('kinds',), 8)['kinds'] == ['look']
    assert len(seen) == 2 and seen[1].no_think


def test_sentence_cache_is_scoped_to_model_and_thinking(monkeypatch):
    monkeypatch.setattr(video, '_SCRIPT_READS', {})
    monkeypatch.setattr(video, '_SCRIPT_READ_FAILURES', {})
    calls = []
    def read(sentences):
        ctx = intent.current_turn().cancel_context
        calls.append((ctx.model_name, ctx.no_think))
        return {'sentences': [{'kind': 'look' if ctx.model_name == 'a' else 'action'} for _ in sentences]}
    monkeypatch.setattr(video, '_script_model', read)
    for model, no_think, expected in [('a', True, 'look'), ('b', True, 'action'), ('a', False, 'look')]:
        with intent.turn_scope() as turn:
            turn.cancel_context = SimpleNamespace(model_name=model, no_think=no_think)
            assert video.read_sentences(['same text'])[0]['kind'] == expected
    assert calls == [('a', True), ('b', True), ('a', False)]


def test_sentence_read_failure_does_not_block_another_model(monkeypatch):
    monkeypatch.setattr(video, '_SCRIPT_READS', {})
    monkeypatch.setattr(video, '_SCRIPT_READ_FAILURES', {})
    calls = []
    def read(sentences):
        name = intent.current_turn().cancel_context.model_name
        calls.append(name)
        return None if name == 'a' else {'sentences': [{'kind': 'look'} for _ in sentences]}
    monkeypatch.setattr(video, '_script_model', read)
    for name in ('a', 'a', 'b'):
        with intent.turn_scope() as turn:
            turn.cancel_context = SimpleNamespace(model_name=name, no_think=True)
            result = video.read_sentences(['same text'])[0]
    assert result['kind'] == 'look' and calls == ['a', 'b']


def test_script_read_retries_invalid_kind_and_row():
    for key, invalid, valid in [('kinds', 'nonsense', 'look'), ('sentences', None, {'beats': 1})]:
        seen = []
        ctx = SimpleNamespace(model_name='selected-model', no_think=False, is_cancelled=lambda: False)
        def model(passed, *args, **kwargs):
            seen.append(passed)
            import json
            return json.dumps({key: [invalid if len(seen) == 1 else valid]})
        ctx.text_model = model
        with intent.turn_scope() as turn:
            turn.cancel_context = ctx
            assert video._ask_json('classify', ['x'], (key,), 8)[key] == [valid]
        assert len(seen) == 2 and seen[1].no_think


def test_cancelled_read_does_not_poison_same_model(monkeypatch):
    monkeypatch.setattr(video, '_SCRIPT_READS', {})
    monkeypatch.setattr(video, '_SCRIPT_READ_FAILURES', {})
    calls = []
    def read(sentences):
        calls.append(1)
        return None if len(calls) == 1 else {'sentences': [{'kind': 'look'}]}
    monkeypatch.setattr(video, '_script_model', read)
    ctx = SimpleNamespace(model_name='a', no_think=True, is_cancelled=lambda: True)
    with intent.turn_scope() as turn:
        turn.cancel_context = ctx
        video.read_sentences(['same text'])
    ctx.is_cancelled = lambda: False
    with intent.turn_scope() as turn:
        turn.cancel_context = ctx
        assert video.read_sentences(['same text'])[0]['kind'] == 'look'
    assert len(calls) == 2


def test_reasoning_fallback_is_not_cached_as_full_reasoning(monkeypatch):
    monkeypatch.setattr(video, '_SCRIPT_READS', {})
    monkeypatch.setattr(video, '_SCRIPT_READ_FAILURES', {})
    calls = []
    def read(sentences):
        calls.append(1)
        return {'sentences': [{'kind': 'look'}], '_light_retry': True}
    monkeypatch.setattr(video, '_script_model', read)
    ctx = SimpleNamespace(model_name='a', no_think=False, is_cancelled=lambda: False)
    for _ in range(2):
        with intent.turn_scope() as turn:
            turn.cancel_context = ctx
            assert video.read_sentences(['same text'])[0]['kind'] == 'look'
            assert video.read_sentences(['same text'])[0]['kind'] == 'look'
    assert len(calls) == 2


def test_missing_action_details_keep_valid_scene_classification(monkeypatch):
    monkeypatch.setenv('INTENT_LIVE', '1')
    responses = iter([{'kinds': ['look', 'action']}, None])
    monkeypatch.setattr(video, '_ask_json', lambda *args: next(responses))
    got = video._script_model(['Quiet room ambience.', 'The cat jumps.'])
    assert [r['kind'] for r in got['sentences']] == ['look', 'action']
    assert got['_light_retry']

from types import SimpleNamespace

import image_router


def test_edit_and_region_reads_use_current_context_and_model_cache(monkeypatch):
    monkeypatch.setenv('INTENT_LIVE', '1')
    image_router._EDIT_CACHE.clear()
    image_router._REGION_CACHE.clear()
    seen = []

    def make_context(name, kind, region_answer):
        ctx = SimpleNamespace(model_name=name, no_think=True)

        def model(passed, system, user, **kwargs):
            seen.append((passed, system))
            if system == 'Answer yes or no only.':
                return region_answer
            return '{"kind": "%s", "steps": []}' % kind

        ctx.text_model = model
        return ctx

    first = make_context('model-a', 'style_transfer', 'yes')
    second = make_context('model-b', 'subject_edit', 'no')
    assert image_router.edit_plan('same edit', ctx=first)['kind'] == 'style_transfer'
    assert image_router.edit_plan('same edit', ctx=second)['kind'] == 'subject_edit'
    assert image_router.removes_region('same edit', 'cat', ctx=first)
    assert not image_router.removes_region('same edit', 'cat', ctx=second)
    assert [row[0] for row in seen] == [first, second, first, second]


def test_empty_model_response_is_not_cached(monkeypatch):
    monkeypatch.setenv('INTENT_LIVE', '1')
    image_router._EDIT_CACHE.clear()
    image_router._REGION_CACHE.clear()
    replies = iter([None, '{"kind": "style_transfer", "steps": []}', None, 'yes'])
    ctx = SimpleNamespace(model_name='model-a', text_model=lambda *a, **k: next(replies))
    assert image_router.edit_plan('same edit', ctx=ctx)['kind'] == 'subject_edit'
    assert image_router.edit_plan('same edit', ctx=ctx)['kind'] == 'style_transfer'
    assert not image_router.removes_region('same edit', 'cat', ctx=ctx)
    assert image_router.removes_region('same edit', 'cat', ctx=ctx)


def test_malformed_model_response_is_not_cached(monkeypatch):
    monkeypatch.setenv('INTENT_LIVE', '1')
    image_router._EDIT_CACHE.clear()
    image_router._REGION_CACHE.clear()
    replies = iter(['not json', '{"kind": "style_transfer", "steps": []}', 'maybe', 'yes'])
    ctx = SimpleNamespace(model_name='model-a', text_model=lambda *a, **k: next(replies))
    assert image_router.edit_plan('same edit', ctx=ctx)['kind'] == 'subject_edit'
    assert image_router.edit_plan('same edit', ctx=ctx)['kind'] == 'style_transfer'
    assert not image_router.removes_region('same edit', 'cat', ctx=ctx)
    assert image_router.removes_region('same edit', 'cat', ctx=ctx)


def test_invalid_multi_operation_is_not_cached(monkeypatch):
    monkeypatch.setenv('INTENT_LIVE', '1')
    image_router._EDIT_CACHE.clear()
    replies = iter(['{"kind": "multi_op", "steps": ["", " "]}',
                    '{"kind": "multi_op", "steps": ["remove cat", "add dog"]}'])
    ctx = SimpleNamespace(model_name='model-a', text_model=lambda *a, **k: next(replies))
    assert image_router.edit_plan('same edit', ctx=ctx)['kind'] == 'subject_edit'
    assert image_router.edit_plan('same edit', ctx=ctx)['kind'] == 'multi_op'

from types import SimpleNamespace

import object_refs


def test_search_status_changes_when_reference_lookup_finishes(monkeypatch):
    stages = []
    monkeypatch.setattr(object_refs, 'named_things', lambda *_: [
        {'thing': 'cat', 'query': 'cat', 'kind': 'person'}])
    monkeypatch.setattr(object_refs, 'photo_of', lambda *_: 'cat.jpg')
    ctx = SimpleNamespace(deps=SimpleNamespace(live=True), set_stage=stages.append)
    assert object_refs.find(ctx, 'cat') == [('cat', 'cat.jpg', 'person')]
    assert stages == ['Finding a photo online', 'Preparing the picture']

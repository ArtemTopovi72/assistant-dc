from types import SimpleNamespace
import json

import object_refs


def test_search_status_changes_when_reference_lookup_finishes(monkeypatch):
    stages = []
    monkeypatch.setattr(object_refs, 'named_things', lambda *_: [
        {'thing': 'cat', 'query': 'cat', 'kind': 'person'}])
    monkeypatch.setattr(object_refs, 'photo_of', lambda *_: 'cat.jpg')
    ctx = SimpleNamespace(deps=SimpleNamespace(live=True), set_stage=stages.append)
    assert object_refs.find(ctx, 'cat') == [('cat', 'cat.jpg', 'person')]
    assert stages == ['Finding a photo online', 'Preparing the picture']


def test_generic_mascots_do_not_become_online_character_references():
    candidates = [{'kind': 'character', 'thing': 'a mascot representing G5', 'query': 'G5 mascot'},
                  {'kind': 'character', 'thing': 'Batman', 'query': 'Batman'}]
    ctx = SimpleNamespace(text_model=lambda *a, **kw: json.dumps({'things': candidates}),
                          yes_no=lambda question, text, **kw: text.startswith('Candidate: Batman\n'))
    result = object_refs.named_things(ctx, 'draw a G5 mascot with Batman')
    assert [entry['thing'] for entry in result] == ['Batman']


def test_explicit_logo_is_a_reference_and_not_forbidden_in_the_prompt():
    candidate = {'kind': 'logo', 'thing': 'ChatGPT logo', 'query': 'OpenAI blossom logo'}
    ctx = SimpleNamespace(text_model=lambda *a, **kw: json.dumps({'things': [candidate]}),
                          yes_no=lambda *a, **kw: True)
    assert object_refs.named_things(ctx, 'ChatGPT logo as faces')[0]['kind'] == 'logo'
    note = object_refs.prompt_note([('ChatGPT logo', 'logo.png', 'logo')])
    assert 'exact' in note.lower() and 'no lettering' not in note.lower()


def test_invented_logo_is_not_searched_when_user_did_not_request_it():
    candidate = {'kind': 'logo', 'thing': 'ChatGPT logo', 'query': 'OpenAI blossom logo'}
    ctx = SimpleNamespace(draw_request='Нарисуй кота',
                          text_model=lambda *a, **kw: json.dumps({'things': [candidate]}),
                          yes_no=lambda q, text, **kw: 'Нарисуй кота' not in text)
    assert object_refs.named_things(ctx, 'cat with ChatGPT logo') == []
    ctx.draw_request = ''
    ctx.yes_no = lambda *a, **kw: True
    assert object_refs.named_things(ctx, 'cat with ChatGPT logo') == []


def test_logo_reference_is_not_cropped_or_erased(monkeypatch):
    ctx = SimpleNamespace(deps=SimpleNamespace(image_search_urls=lambda *a, **kw: ['url'],
                                                download_image=lambda *a: True))
    monkeypatch.setattr(object_refs, '_box', lambda *a: ([200, 200, 300, 300], True))
    monkeypatch.setattr(object_refs, '_crop', lambda *a: (_ for _ in ()).throw(AssertionError('cropped')))
    result = object_refs.photo_of(ctx, 'OpenAI logo', 'logo', kind='logo')
    assert result.endswith('c0.jpg')

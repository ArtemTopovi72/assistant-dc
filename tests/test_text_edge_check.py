from types import SimpleNamespace

import draw_text


def test_lettering_cut_by_frame_is_not_accepted(monkeypatch):
    layout = {'elements': [{'desc': 'sign', 'text': 'СОЛ ВЕРНУЛСЯ',
                            'x': .1, 'y': .8, 'w': .8, 'h': .1}]}
    ctx = SimpleNamespace(deps=SimpleNamespace(
        _crop_for_read=lambda *_: None,
        ocr_edge_text=lambda _: ['СОЛ ВЕРНУЛСЯ']))
    monkeypatch.setattr(draw_text, 'read_text', lambda *_: ['СОЛ ВЕРНУЛСЯ'])
    monkeypatch.setattr(draw_text, 'stray_text', lambda *_: [])
    result = draw_text.verify_text(ctx, 'fake.png', layout)
    assert not result['ok']
    assert 'edge' in draw_text.text_problems(result['checks'])[0]


def test_unrelated_single_character_at_edge_does_not_reject_text(monkeypatch):
    layout = {'elements': [{'desc': 'sign', 'text': 'МАМА', 'x': .1, 'y': .2, 'w': .8, 'h': .1}]}
    ctx = SimpleNamespace(deps=SimpleNamespace(_crop_for_read=lambda *_: None,
                                                ocr_edge_text=lambda _: ['А']))
    monkeypatch.setattr(draw_text, 'read_text', lambda *_: ['МАМА'])
    monkeypatch.setattr(draw_text, 'stray_text', lambda *_: [])
    assert draw_text.verify_text(ctx, 'fake.png', layout)['ok']


def test_punctuation_is_not_matched_to_unrelated_edge_text(monkeypatch):
    layout = {'elements': [{'desc': 'sign', 'text': '!!!', 'x': .1, 'y': .2, 'w': .8, 'h': .1}]}
    ctx = SimpleNamespace(deps=SimpleNamespace(_crop_for_read=lambda *_: None,
                                                ocr_edge_text=lambda _: ['OTHER']))
    monkeypatch.setattr(draw_text, 'read_text', lambda *_: ['!!!'])
    monkeypatch.setattr(draw_text, 'stray_text', lambda *_: [])
    assert not any(c.get('clipped') for c in draw_text.verify_text(ctx, 'fake.png', layout)['checks'])

from types import SimpleNamespace

import draw_text
import draw_agent


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


def test_edge_text_matches_lookalike_latin_letters(monkeypatch):
    layout = {'elements': [{'desc': 'sign', 'text': 'СОЛ ВЕРНУЛСЯ', 'x': .1, 'y': .2, 'w': .8, 'h': .1}]}
    ctx = SimpleNamespace(deps=SimpleNamespace(_crop_for_read=lambda *_: None,
                                                ocr_edge_text=lambda _: ['COЛ BEPHУЛCЯ']))
    monkeypatch.setattr(draw_text, 'read_text', lambda *_: ['СОЛ ВЕРНУЛСЯ'])
    monkeypatch.setattr(draw_text, 'stray_text', lambda *_: [])
    assert not draw_text.verify_text(ctx, 'fake.png', layout)['ok']


def test_lettering_planned_at_frame_edge_is_moved_inside():
    layout = {'elements': [{'desc': 'bottom caption', 'text': 'СОЛ ВЕРНУЛСЯ',
                            'x': .03, 'y': .94, 'w': .94, 'h': .06}]}
    assert any(p['kind'] == 'text_at_edge' for p in draw_text.text_report(layout))
    fixed, _ = draw_text.auto_fix_text(layout)
    text = fixed['elements'][0]
    assert text['y'] + text['h'] <= .92 + 1e-8
    assert text['text'] == 'СОЛ ВЕРНУЛСЯ'


def test_edge_margin_does_not_move_label_off_its_host():
    layout = {'elements': [
        {'desc': 'sign', 'x': .96, 'y': .2, 'w': .04, 'h': .4},
        {'desc': 'sign lettering', 'text': 'ABC', 'x': .97, 'y': .3, 'w': .03, 'h': .1},
    ]}
    fixed, _ = draw_text.repair_text(layout, [1])
    host, label = fixed['elements']
    assert label['x'] >= host['x']
    assert label['x'] + label['w'] <= host['x'] + host['w'] + 1e-8


def test_render_moves_near_edge_lettering_even_when_geometry_is_otherwise_valid(monkeypatch):
    layout = {'elements': [{'desc': 'caption', 'text': 'ABC',
                            'x': .1, 'y': .88, 'w': .144, 'h': .08}]}
    rendered = []
    stages = []
    ctx = SimpleNamespace(is_cancelled=lambda: False, deps=SimpleNamespace(
        render_without_collage=lambda ctx, layout, **kw: rendered.append(layout) or 'image.png',
        verify_text=lambda *_: {'checks': [], 'failures': [], 'stray': []},
        text_problems=lambda *_: [],
        critique=lambda *a, **kw: {'ok': True, 'score': 10, 'problems': [], 'ops': []}))
    ctx.set_stage = stages.append
    monkeypatch.setattr(draw_agent, 'relations', lambda *a, **kw: [])
    monkeypatch.setattr(draw_agent, 'geometry_report', lambda *a, **kw: [])
    draw_agent.run(ctx, 'draw ABC', layout=layout, rounds=0)
    text = rendered[0]['elements'][0]
    assert text['y'] + text['h'] <= .92 + 1e-8
    assert stages == ['Drawing a picture', 'Checking the picture']


def test_edge_inset_does_not_resize_lettering_that_is_already_inside():
    layout = {'elements': [{'desc': 'caption', 'text': 'ABC',
                            'x': .1, 'y': .35, 'w': .23, 'h': .08}]}
    result, _ = draw_text.inset_text(layout)
    text = result['elements'][0]
    assert (text['x'], text['w']) == (.1, .23)

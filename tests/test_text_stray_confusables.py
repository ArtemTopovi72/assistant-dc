from types import SimpleNamespace

import draw_text


def test_cyrillic_text_fragments_read_with_latin_lookalikes_are_not_extra(tmp_path):
    crop = tmp_path / 'crop.png'
    crop.write_bytes(b'placeholder')
    layout = {'elements': [
        {'desc': 'sign', 'x': .1, 'y': .1, 'w': .8, 'h': .8},
        {'desc': 'lettering on sign', 'text': 'СОЛ ВЕРНУЛСЯ СНОВА', 'x': .2, 'y': .2, 'w': .5, 'h': .1},
    ]}
    ctx = SimpleNamespace(deps=SimpleNamespace(_crop_for_read=lambda *_: str(crop),
                                                ocr_read=lambda *_: ['BEPHYЛCЯ', 'CHOBA', 'COЛ']))
    assert draw_text.stray_text(ctx, 'image.png', layout, draw_text._text_els(layout)) == []

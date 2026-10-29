from text_restyle import caption_regions, caption_left, protected_boxes, overlaps


def test_selects_caption_words_but_preserves_shirt_labels():
    regions = [
        ([[0, 5], [50, 5], [50, 20], [0, 20]], 'HELLO', .9),
        ([[5, 50], [40, 50], [40, 65], [5, 65]], 'SOL', .9),
        ([[0, 80], [50, 80], [50, 95], [0, 95]], 'GOODBYE', .9),
    ]
    layout = {'elements': [
        {'text': 'HELLO', 'desc': 'text across the top', 'y': .05, 'h': .1},
        {'text': 'SOL', 'desc': 'text on shirt', 'y': .5, 'h': .1},
        {'text': 'GOODBYE', 'desc': 'text across the bottom', 'y': .8, 'h': .1},
    ]}
    chosen, top, bottom = caption_regions(regions, layout, 100, 100)
    assert chosen == [(0, 5, 50, 20), (0, 80, 50, 95)]
    assert top == ['HELLO'] and bottom == ['GOODBYE']


def test_missing_caption_word_never_produces_a_partial_edit():
    layout = {'elements': [{'text': 'HELLO WORLD', 'desc': 'text across the top', 'y': .05, 'h': .1}]}
    regions = [([[0, 5], [50, 5], [50, 20], [0, 20]], 'HELLO', .9)]
    assert caption_regions(regions, layout, 100, 100)[0] == []


def test_clothing_near_edge_is_never_restyled():
    layout = {'elements': [{'text': 'SOL', 'desc': 'text on shirt', 'y': .05, 'h': .1}]}
    regions = [([[0, 5], [50, 5], [50, 20], [0, 20]], 'SOL', .9)]
    assert caption_regions(regions, layout, 100, 100)[0] == []


def test_logo_and_shirt_boxes_are_protected_even_when_they_overlap_a_caption():
    layout = {'elements': [
        {'desc': 'ChatGPT logo badge', 'x': .2, 'y': .1, 'w': .2, 'h': .2},
        {'desc': 'text on shirt', 'text': 'SOL', 'x': .2, 'y': .5, 'w': .2, 'h': .1},
    ]}
    assert protected_boxes(layout, 100, 100) == [(20, 10, 40, 30), (20, 50, 40, 60)]
    assert overlaps((10, 5, 35, 20), protected_boxes(layout, 100, 100)[0])
    assert not overlaps((0, 0, 10, 10), protected_boxes(layout, 100, 100)[0])


def test_visible_old_caption_rejects_the_cleaning():
    regions = [([[0, 5], [50, 5], [50, 20], [0, 20]], 'HELLO', .9)]
    assert caption_left(regions, [(0, 5, 50, 20)], ['HELLO WORLD'])
    assert not caption_left([], [(0, 5, 50, 20)], ['HELLO WORLD'])


def test_rejects_old_text_that_returns_after_compositing(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from PIL import Image
    import image, image_lettering_remove, ocr_reader, text_restyle
    source = tmp_path / 'source.png'
    Image.new('RGB', (100, 100), 'white').save(source)
    region = ([[0, 5], [50, 5], [50, 20], [0, 20]], 'HELLO', .9)
    class Reader:
        def readtext(self, path, **kwargs):
            return [] if str(path).endswith('clean.png') else [region]
    def fill(ctx, src, mask_path, grow):
        clean = tmp_path / 'clean.png'
        Image.new('RGB', (100, 100), 'white').save(clean)
        return str(clean)
    monkeypatch.setattr(image, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr(ocr_reader, '_get', lambda: Reader())
    monkeypatch.setattr(image_lettering_remove, '_faces', lambda p: [])
    ctx = SimpleNamespace(deps=SimpleNamespace(fill_caption=fill))
    rec = {'layout': {'elements': [{'text': 'HELLO', 'desc': 'text across the top', 'y': .05, 'h': .1}]}}
    assert text_restyle.restyle(ctx, str(source), rec) is None


def test_caption_overlapping_protected_logo_is_refused(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from PIL import Image
    import image, image_lettering_remove, ocr_reader, text_restyle
    source = tmp_path / 'source.png'
    Image.new('RGB', (100, 100), 'white').save(source)
    region = ([[0, 0], [99, 0], [99, 40], [0, 40]], 'HI', .9)
    class Reader:
        def readtext(self, path, **kwargs):
            return [region] if str(path) == str(source) else []
    def fill(ctx, src, mask_path, grow):
        raise AssertionError('overlapping protected logo must not reach the editor')
    monkeypatch.setattr(image, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr(ocr_reader, '_get', lambda: Reader())
    monkeypatch.setattr(image_lettering_remove, '_faces', lambda p: [])
    ctx = SimpleNamespace(deps=SimpleNamespace(fill_caption=fill))
    rec = {'layout': {'elements': [
        {'text': 'HI', 'desc': 'text across the top', 'y': .05, 'h': .1},
        {'desc': 'logo', 'x': 0, 'y': 0, 'w': .04, 'h': .04}]}}
    assert text_restyle.restyle(ctx, str(source), rec) is None


def test_explicit_mask_is_bound_to_the_exact_source(tmp_path):
    import hashlib
    from PIL import Image, ImageDraw
    from text_restyle import verified_source_mask
    source = tmp_path / 'original.png'
    Image.new('RGB', (100, 100), 'white').save(source)
    mask = Image.new('L', (100, 100), 0)
    ImageDraw.Draw(mask).rectangle((20, 20, 80, 80), fill=255)
    mask.save(tmp_path / 'protect.png')
    metadata = {'protected_mask': 'protect.png',
                'protected_mask_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'protected_mask_sha256': hashlib.sha256((tmp_path / 'protect.png').read_bytes()).hexdigest()}
    assert verified_source_mask(source, metadata, tmp_path) is not None
    Image.new('RGB', (100, 100), 'black').save(source)
    assert verified_source_mask(source, metadata, tmp_path) is None
    metadata['protected_mask'] = '../outside.png'
    assert verified_source_mask(source, metadata, tmp_path) is None


def test_verified_mask_does_not_authorize_another_protected_box():
    from PIL import Image
    from text_restyle import verified_crossings
    mask = Image.new('L', (100, 100), 255)
    metadata = {'protected_mask_boxes': [[20, 20, 50, 50]]}
    assert verified_crossings([(20, 20, 50, 50)], metadata, mask)
    assert not verified_crossings([(10, 10, 60, 60)], metadata, mask)


def test_full_frame_redraw_is_not_used_for_typography(tmp_path):
    from types import SimpleNamespace
    import tool_image_handlers as H
    image = tmp_path / 'original.png'
    image.write_bytes(b'image')
    ctx = SimpleNamespace(last_image_path=str(image), yes_no=lambda *a, **kw: True,
                          image_engine=SimpleNamespace(load_layout_for=lambda p: None))
    result = H._handle_redraw_image(ctx, {'user_input': 'Сделай красивые надписи'}, {'instructions': 'redraw'})
    assert result.startswith('[TOOL ERROR]') and 'layout' in result


def test_repeat_after_success_does_not_claim_the_edit_failed(tmp_path):
    from types import SimpleNamespace
    import tool_image_handlers as H
    source = tmp_path / 'before.png'
    output = tmp_path / 'after.png'
    source.write_bytes(b'original')
    output.write_bytes(b'result')
    ctx = SimpleNamespace(last_image_path=str(source), set_stage=lambda s: None,
                          image_engine=SimpleNamespace(load_layout_for=lambda p: {'layout': {}}),
                          deps=SimpleNamespace(restyle_captions=lambda *a: str(output)))
    state = {}
    assert 'saved' in H._restyle_captions(ctx, state, str(source))
    second = H._restyle_captions(ctx, state, str(source))
    assert 'already' in second.lower() and not second.startswith('[TOOL ERROR]')


def test_old_layout_candidates_are_loaded_only_from_the_same_chat(tmp_path):
    from types import SimpleNamespace
    from tg_tasks import _layout_candidates
    path = tmp_path / 'image.png'
    path.write_bytes(b'image')
    (tmp_path / 'image.png.layout.json').write_text('{}')
    seen = []
    def recent(chat, kind, limit):
        seen.append((chat, kind, limit))
        return [{'path': str(path)}] if chat == 12 else []
    assert _layout_candidates(SimpleNamespace(image_log=[]), 12, recent) == [str(path)]
    assert _layout_candidates(SimpleNamespace(image_log=[]), 13, recent) == []
    assert seen == [(12, 'photo', 50), (13, 'photo', 50)]


def test_archive_layout_candidates_prefer_newest(tmp_path):
    from types import SimpleNamespace
    from tg_tasks import _layout_candidates
    old, new = tmp_path / 'old.png', tmp_path / 'new.png'
    for path in (old, new):
        path.write_bytes(b'same image')
        (tmp_path / (path.name + '.layout.json')).write_text('{}')
    candidates = _layout_candidates(SimpleNamespace(image_log=[]), 12,
                                    lambda *_: [{'path': str(new)}, {'path': str(old)}])
    assert list(reversed(candidates)) == [str(new), str(old)]


def test_successful_restyle_replaces_fabricated_failure_reply():
    from graph_personality import _caption_result_reply
    state = {'caption_restyle_result': 'out.png', 'image_path': 'out.png'}
    assert _caption_result_reply(state, 'Ошибка', 'Исправь надписи') == 'Готово: обновил надписи.'
    assert _caption_result_reply(state, 'Failure', 'Fix the text').startswith('Done:')
    state['image_path'] = 'another.png'
    assert _caption_result_reply(state, 'Failure', 'Fix the text') == 'Failure'

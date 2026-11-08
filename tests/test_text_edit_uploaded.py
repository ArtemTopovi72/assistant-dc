from types import SimpleNamespace
from PIL import Image
import image_router


def test_text_edit_on_uploaded_image_attempts_verified_edit(monkeypatch):
    calls = []
    monkeypatch.setattr(image_router, 'classify_edit_intent', lambda *a, **k: 'text_edit')
    monkeypatch.setattr(image_router, 'add_text_verified',
                        lambda *a, **k: (calls.append(k) or ('edited.png', 'NEW', 'NEW')))
    ctx = SimpleNamespace(deps=SimpleNamespace(image=SimpleNamespace(load_layout_for=lambda _: None)))
    category, out = image_router._route_edit_request(ctx, 'uploaded.png', 'replace text with "NEW"')
    assert (category, out) == ('text_edit', 'edited.png')
    assert calls[0]['require_ocr'] is True and calls[0]['allow_overlay'] is False


def test_text_edit_does_not_deliver_without_ocr(tmp_path, monkeypatch):
    from PIL import Image
    image = tmp_path / 'source.png'
    Image.new('RGB', (200, 100), 'white').save(image)
    monkeypatch.setattr(image_router, 'TEXT_ADD_TRIES', 1)
    ctx = SimpleNamespace(deps=SimpleNamespace(
        image=SimpleNamespace(edit_image_with_firered=lambda *a, **k: str(image)),
        ocr_read=lambda _: None))
    out, _, _ = image_router.add_text_verified(ctx, str(image), 'replace text with "NEW"',
                                               require_ocr=True, allow_overlay=False)
    assert out is None


def test_combined_ocr_line_matches_each_label_without_accepting_wrong_year(tmp_path, monkeypatch):
    source = tmp_path / 'source.png'
    Image.new('RGB', (200, 100), 'white').save(source)
    monkeypatch.setattr(image_router, 'TEXT_ADD_TRIES', 1)
    ctx = SimpleNamespace(vision_model=lambda *a, **k: 'YES', deps=SimpleNamespace(
        image=SimpleNamespace(edit_image_with_firered=lambda *a, **k: str(source)),
        ocr_read=lambda _: ['Обед ДЛЯ Bcex ШЕФ 2026']))
    instruction = 'write "Обед для всех" on the barrel and "ШЕФ 2026" on the apron'
    out, _, _ = image_router.add_text_verified(ctx, str(source), instruction, allow_overlay=False)
    assert out == str(source)
    ctx.deps.ocr_read = lambda _: ['Обед ДЛЯ Bcex ШЕФ 2025']
    out, _, _ = image_router.add_text_verified(ctx, str(source), instruction, allow_overlay=False)
    assert out is None



def test_cancel_during_text_edit_does_not_start_more_renders(tmp_path, monkeypatch):
    source = tmp_path / 'source.png'
    Image.new('RGB', (200, 100), 'white').save(source)
    monkeypatch.setattr(image_router, 'TEXT_ADD_TRIES', 3)
    calls = []
    ctx = SimpleNamespace(is_cancelled=lambda: bool(calls), deps=SimpleNamespace(
        image=SimpleNamespace(edit_image_with_firered=lambda *a, **k: calls.append(1) or None)))
    out, _, _ = image_router.add_text_verified(ctx, str(source), 'replace text with "NEW"',
                                               require_ocr=True, allow_overlay=False)
    assert out is None
    assert len(calls) == 1


def test_text_edit_verifies_new_label_not_quoted_old_label(monkeypatch):
    seen = []
    monkeypatch.setattr(image_router, 'classify_edit_intent', lambda *a, **k: 'text_edit')
    monkeypatch.setattr(image_router, 'add_text_verified',
                        lambda *a, **k: (seen.append((a, k)) or ('edited.png', 'NEW', 'NEW')))
    ctx = SimpleNamespace(deps=SimpleNamespace(image=SimpleNamespace(load_layout_for=lambda _: None)))
    image_router._route_edit_request(ctx, 'uploaded.png',
        'change the "OLD" inscription to "NEW"', verify_text='the text "NEW"')
    assert seen[0][1]['verify_text'] == 'the text "NEW"'


def test_verify_text_ignores_quoted_original_in_instruction(tmp_path, monkeypatch):
    source = tmp_path / 'source.png'
    Image.new('RGB', (200, 100), 'white').save(source)
    monkeypatch.setattr(image_router, 'TEXT_ADD_TRIES', 1)
    ctx = SimpleNamespace(vision_model=lambda *a, **k: 'YES', deps=SimpleNamespace(
        image=SimpleNamespace(edit_image_with_firered=lambda *a, **k: str(source)),
        ocr_read=lambda _: ['NEW']))
    out, _, wanted = image_router.add_text_verified(ctx, str(source),
        'change "OLD" inscription to "NEW"', verify_text='the text "NEW"', allow_overlay=False)
    assert out == str(source) and wanted == 'NEW'


def test_ocr_split_across_adjacent_lines_matches_label(tmp_path, monkeypatch):
    source = tmp_path / 'source.png'
    Image.new('RGB', (200, 100), 'white').save(source)
    monkeypatch.setattr(image_router, 'TEXT_ADD_TRIES', 1)
    ctx = SimpleNamespace(vision_model=lambda *a, **k: 'YES', deps=SimpleNamespace(
        image=SimpleNamespace(edit_image_with_firered=lambda *a, **k: str(source)),
        ocr_read=lambda _: ['Обед', 'для всех', 'ШЕФ 2026']))
    out, _, _ = image_router.add_text_verified(ctx, str(source),
        'write "Обед для всех" on the barrel and "ШЕФ 2026" on the apron', allow_overlay=False)
    assert out == str(source)

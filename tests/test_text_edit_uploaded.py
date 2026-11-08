from types import SimpleNamespace
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

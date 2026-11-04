from types import SimpleNamespace

from PIL import Image
import image_router


def test_object_text_does_not_fallback_to_center_of_whole_image(tmp_path, monkeypatch):
    source = tmp_path / 'source.png'
    Image.new('RGB', (200, 100), 'white').save(source)
    monkeypatch.setattr(image_router, 'TEXT_ADD_TRIES', 1)
    ctx = SimpleNamespace(
        yes_no=lambda question, text, **kwargs: True,
        deps=SimpleNamespace(
            image=SimpleNamespace(edit_image_with_firered=lambda *a, **k: 'wrong.png'),
            ocr_read=lambda _: ['wrong'], OUTPUT_DIR=tmp_path))
    result, _, _ = image_router.add_text_verified(ctx, str(source), 'write «HELLO» on the barrel')
    assert result is None

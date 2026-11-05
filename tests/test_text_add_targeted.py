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


def test_matching_text_on_wrong_surface_is_retried(tmp_path, monkeypatch):
    source = tmp_path / 'source.png'
    first = tmp_path / 'first.png'
    second = tmp_path / 'second.png'
    for path in (source, first, second):
        Image.new('RGB', (200, 100), 'white').save(path)
    renders = iter([str(first), str(second)])
    answers = iter(['NO', 'YES'])
    monkeypatch.setattr(image_router, 'TEXT_ADD_TRIES', 2)
    ctx = SimpleNamespace(
        vision_model=lambda *a, **k: next(answers),
        deps=SimpleNamespace(image=SimpleNamespace(edit_image_with_firered=lambda *a, **k: next(renders)),
                             ocr_read=lambda _: ['HELLO'], OUTPUT_DIR=tmp_path))
    result, _, _ = image_router.add_text_verified(ctx, str(source), 'write "HELLO" on the barrel')
    assert result == str(second)


def test_placement_still_checked_if_ocr_is_unavailable(tmp_path, monkeypatch):
    source = tmp_path / 'source.png'
    first = tmp_path / 'first.png'
    second = tmp_path / 'second.png'
    for path in (source, first, second):
        Image.new('RGB', (200, 100), 'white').save(path)
    renders = iter([str(first), str(second)])
    answers = iter(['NO', 'YES'])
    monkeypatch.setattr(image_router, 'TEXT_ADD_TRIES', 2)
    ctx = SimpleNamespace(
        vision_model=lambda *a, **k: next(answers),
        deps=SimpleNamespace(image=SimpleNamespace(edit_image_with_firered=lambda *a, **k: next(renders)),
                             ocr_read=lambda _: None, OUTPUT_DIR=tmp_path))
    result, _, _ = image_router.add_text_verified(ctx, str(source), 'write "HELLO" on the barrel')
    assert result == str(second)

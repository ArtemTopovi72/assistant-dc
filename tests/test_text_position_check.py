from types import SimpleNamespace

from text_placement_check import text_position_failure


def test_placement_verifier_flags_text_on_the_wrong_surface(tmp_path):
    image = tmp_path / 'result.png'
    image.write_bytes(b'test')
    calls = []
    def vision(*args, **kwargs):
        calls.append(kwargs)
        return 'NO'
    ctx = SimpleNamespace(vision_model=vision)
    assert text_position_failure(ctx, str(image), 'barrel', 'write "Кал всем" on the barrel')
    assert calls[0]['image_path'] == str(image)
    assert text_position_failure(ctx, str(image), 'бочка', 'Напиши «Кал всем» на бочке')
    assert text_position_failure(ctx, str(image), 'бочка', 'надпись должна быть на самой бочке')


def test_placement_verifier_skips_unrelated_edits_and_unknown_results(tmp_path):
    image = tmp_path / 'result.png'
    image.write_bytes(b'test')
    calls = []
    def vision(*args, **kwargs):
        calls.append(kwargs)
        return 'UNKNOWN'
    ctx = SimpleNamespace(vision_model=vision)
    assert not text_position_failure(ctx, str(image), 'sky', 'make it blue')
    assert not calls
    assert not text_position_failure(ctx, str(image), 'barrel', 'write "Кал всем" on the barrel')
    assert len(calls) == 1


def test_inpaint_result_reports_wrong_text_position():
    from test_tools_full import _ctx, _img, mock_env
    from tools import _handle_inpaint_image

    ctx = _ctx(last_image_path=_img('position_source.png'), last_image_prompt='',
               vision_model=lambda *args, **kwargs: 'NO')
    with mock_env(category='product_edit'):
        result = _handle_inpaint_image(ctx, {}, {'region': 'barrel',
                                                'instructions': 'write "HELLO" on the barrel'})
    assert '[LETTERING CHECK FAILED]' in result

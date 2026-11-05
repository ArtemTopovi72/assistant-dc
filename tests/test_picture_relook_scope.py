from types import SimpleNamespace
import graph


def test_calculation_does_not_reinspect_an_unrelated_recent_photo(tmp_path, monkeypatch):
    photo = tmp_path / 'photo.jpg'
    photo.write_bytes(b'photo')
    ctx = SimpleNamespace(last_image_path=str(photo), image_origin='user', image_pointed_at=False)
    monkeypatch.setattr(graph, '_picture_read', lambda *_: {
        'is_question': True, 'about_picture': False, 'must_call': 'calculate', 'wants': ['calculate']})
    assert not graph.needs_relook(ctx, {'user_input': 'Сколько будет 2+2?'})
    ctx.image_pointed_at = True
    assert graph.needs_relook(ctx, {'user_input': 'Посчитай это'})


def test_implicit_question_can_still_reinspect_the_receipt(tmp_path, monkeypatch):
    photo = tmp_path / 'receipt.jpg'
    photo.write_bytes(b'photo')
    ctx = SimpleNamespace(last_image_path=str(photo), image_origin='user', image_pointed_at=False)
    monkeypatch.setattr(graph, '_picture_read', lambda *_: {
        'is_question': True, 'about_picture': False, 'must_call': '', 'wants': []})
    assert graph.needs_relook(ctx, {'user_input': 'Какой товар дороже?'})


def test_standalone_question_does_not_reinspect_an_old_photo(tmp_path, monkeypatch):
    photo = tmp_path / 'photo.jpg'
    photo.write_bytes(b'photo')
    ctx = SimpleNamespace(last_image_path=str(photo), image_origin='user', image_pointed_at=False)
    monkeypatch.setattr(graph, '_picture_read', lambda *_: {
        'is_question': True, 'about_picture': False, 'must_call': '', 'wants': []})
    monkeypatch.setattr(graph, 'yes_no', lambda _: lambda *args, **kwargs: True)
    assert not graph.needs_relook(ctx, {'user_input': 'Чем резистор отличается от конденсатора?'})

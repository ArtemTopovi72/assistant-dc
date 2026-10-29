from tg_dispatch import _video_edit_request, _video_upload_name


def test_video_edit_is_distinguished_from_video_analysis():
    assert _video_edit_request('Обрежь это видео с 0:05 до 0:20')
    assert _video_edit_request('Add subtitles to the clip')
    assert _video_edit_request('Ускорь ролик')
    assert _video_edit_request('Убери звук')
    assert not _video_edit_request('Расскажи, что происходит в видео')
    assert not _video_edit_request('')


def test_video_uploads_keep_distinct_originals(tmp_path):
    first, second = _video_upload_name('clip.mp4'), _video_upload_name('clip.mp4')
    assert first != second and first.endswith('.mp4') and second.endswith('.mp4')
    (tmp_path / first).write_bytes(b'first')
    (tmp_path / second).write_bytes(b'second')
    assert (tmp_path / first).read_bytes() == b'first'

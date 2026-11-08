"""An attached video's caption is the instruction, its soundtrack is material."""
from types import SimpleNamespace

import tg_bot as T
import pytest
from prompt_guard import user_words, wrap_quoted
from tg_tasks import _typed_part


@pytest.mark.parametrize('description', ['A speaker and a sign saying "wrong words".', ''])
def test_captioned_video_keeps_source_and_soundtrack_separate(tmp_path, monkeypatch, description):
    T.redirect_data_dir(str(tmp_path / 'data'))
    bot = T.TelegramBot('123:TEST', lambda: SimpleNamespace(), lambda: None,
                        lambda: {}, silent_mode=True)
    chat = 999998
    bot._user_store.put(T._User(chat_id=chat, name='test', status='approved'))
    bot._send_text = lambda *a, **k: 1
    bot._api_post = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    bot._dl_bytes = lambda fid: b'video bytes'
    bot._transcribe = lambda *a, **k: 'Чужая фраза из исходного видео.'
    directory = tmp_path / 'seen'
    directory.mkdir()
    sheet = directory / 'sheet.jpg'
    sheet.write_bytes(b'photo')
    clip = directory / 'clip.mp4'
    clip.write_bytes(b'video bytes')
    bot._look_video = lambda *a, **k: {'sheet': str(sheet), 'description': description}
    pushed = []
    bot._backend.push = pushed.append
    caption = 'Продолжи видео, мужчина говорит: «Только новая фраза».'
    bot._resolve_and_push(chat, [{'type': 'voice', 'media': 'video', 'file_id': 'x', 'caption': caption}])
    assert len(pushed) == 1
    task = pushed[0]
    assert task.user_text.startswith(caption)
    assert task.media_transcript == 'Чужая фраза из исходного видео.'
    assert task.source_video_path == str(clip)
    assert user_words(task.user_text) == caption
    assert T._Task.from_dict(task.to_dict()).source_video_path == str(clip)
    if description:
        assert task.image_path == str(sheet)
        assert T._image_by_path(bot._get_session(chat), str(sheet))["src"] == "video"
        assert not any(e.get("src") == "user" for e in bot._get_session(chat).image_log)


def test_legacy_video_sheet_copy_is_not_a_user_photo(tmp_path):
    T.redirect_data_dir(str(tmp_path / "data"))
    sess = T._Session(123)
    sheet = tmp_path / "sheet.jpg"
    copy = tmp_path / "copy.jpg"
    photo = tmp_path / "photo.jpg"
    sheet.write_bytes(b"video sheet")
    copy.write_bytes(b"video sheet")
    photo.write_bytes(b"a real photo")
    T._log_image(sess, str(sheet), src="video")
    legacy = T._image_by_id(sess, T._log_image(sess, str(copy), src="user"))
    actual = T._image_by_id(sess, T._log_image(sess, str(photo), src="user", msg_id=5))
    assert T._is_video_image(sess, legacy)
    assert not T._is_video_image(sess, actual)


def test_language_detection_ignores_attached_video_description():
    caption = 'Кратко перескажи видео.'
    content = caption + '\n' + wrap_quoted('visual description of the attached video',
                                            'The video shows a speaker on stage.')
    assert _typed_part(content) == caption


@pytest.mark.parametrize('photo_first', [True, False])
def test_photo_and_video_remain_distinct_in_either_order(tmp_path, monkeypatch, photo_first):
    import sys
    T.redirect_data_dir(str(tmp_path / 'data'))
    bot = T.TelegramBot('123:TEST', lambda: SimpleNamespace(), lambda: None,
                        lambda: {}, silent_mode=True)
    chat = 999997
    bot._user_store.put(T._User(chat_id=chat, name='test', status='approved'))
    bot._send_text = lambda *a, **k: 1
    bot._api_post = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    bot._dl_bytes = lambda fid: b'photo bytes' if fid == 'photo' else b'video bytes'
    bot._transcribe = lambda *a, **k: ''
    sheet = tmp_path / 'sheet.jpg'; sheet.write_bytes(b'video sheet')
    (tmp_path / 'clip.mp4').write_bytes(b'video bytes')
    bot._look_video = lambda *a, **k: {'sheet': str(sheet), 'description': 'A video.'}
    monkeypatch.setitem(sys.modules, 'ocr_reader', SimpleNamespace(read_pieces=lambda path: []))
    pushed = []
    bot._backend.push = pushed.append
    photo = {'type': 'photo', 'file_id': 'photo'}
    video = {'type': 'voice', 'media': 'video', 'file_id': 'video',
             'caption': 'Compare the photo and video.'}
    bot._resolve_and_push(chat, [photo, video] if photo_first else [video, photo])
    assert len(pushed) == 1
    paths = [pushed[0].image_path, *pushed[0].album_paths]
    assert b'photo bytes' in [open(path, 'rb').read() for path in paths]
    if photo_first:
        assert open(pushed[0].image_path, 'rb').read() == b'photo bytes'
        assert pushed[0].image_is_photo
    else:
        assert pushed[0].image_path == str(sheet)
    assert T._image_by_path(bot._get_session(chat), str(sheet))['src'] == 'video'

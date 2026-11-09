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


@pytest.mark.parametrize('video_first', [True, False])
@pytest.mark.parametrize('caption_on_video', [True, False])
def test_mixed_album_creates_one_task_with_photo_and_video(tmp_path, monkeypatch, video_first, caption_on_video):
    import sys
    T.redirect_data_dir(str(tmp_path / 'data'))
    bot = T.TelegramBot('123:TEST', lambda: SimpleNamespace(), lambda: None, lambda: {}, silent_mode=True)
    chat = 999996
    bot._user_store.put(T._User(chat_id=chat, name='test', status='approved'))
    bot._send_text = lambda *a, **k: 1
    bot._api_post = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    bot._dl_bytes = lambda fid: b'photo bytes' if fid == 'photo' else b'video bytes'
    bot._transcribe = lambda *a, **k: 'speech from the video'
    sheet = tmp_path / 'sheet.jpg'; sheet.write_bytes(b'video sheet')
    clip = tmp_path / 'clip.mp4'; clip.write_bytes(b'video bytes')
    bot._look_video = lambda *a, **k: {'sheet': str(sheet), 'description': 'A green video.'}
    monkeypatch.setitem(sys.modules, 'ocr_reader', SimpleNamespace(read_pieces=lambda path: []))
    pushed = []
    bot._backend.push = pushed.append
    caption = 'Compare the photo and video.'
    photo = {'file_id': 'photo', 'caption': '' if caption_on_video else caption}
    video = {'file_id': 'video', 'media': 'video', 'seconds': 3,
             'caption': caption if caption_on_video else ''}
    item = {'type': 'mixed_album', 'items': [video, photo] if video_first else [photo, video],
            '_inbox_id': 'mixed-durable-id'}
    bot._incoming.put(chat, item, 'mixed-durable-id')
    bot._resolve_and_push(chat, [item])
    assert len(pushed) == 1
    task = pushed[0]
    assert task.image_is_photo and open(task.image_path, 'rb').read() == b'photo bytes'
    assert task.source_video_path == str(clip)
    assert task.incoming_ids == ['mixed-durable-id'] and not bot._incoming.pending()
    assert user_words(task.user_text).strip() == caption
    assert 'speech from the video' in task.user_text


@pytest.mark.parametrize('failure', ['long', 'unreadable'])
def test_mixed_album_video_failure_keeps_photo_and_reports_problem(tmp_path, failure):
    T.redirect_data_dir(str(tmp_path / 'data'))
    bot = T.TelegramBot('123:TEST', lambda: SimpleNamespace(), lambda: None, lambda: {}, silent_mode=True)
    chat = 999995
    bot._user_store.put(T._User(chat_id=chat, name='test', status='approved'))
    sent, pushed = [], []
    bot._send_text = lambda *a, **k: sent.append(a[1]) or 1
    bot._api_post = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    bot._dl_bytes = lambda fid: b'photo bytes' if fid == 'photo' else b''
    bot._transcribe = lambda *a, **k: ''
    bot._look_video = lambda *a, **k: {}
    bot._backend.push = pushed.append
    video = {'file_id': 'video', 'media': 'video', 'seconds': 999999 if failure == 'long' else 3,
             'caption': 'Compare the photo and video.'}
    photo = {'file_id': 'photo', 'caption': ''}
    bot._resolve_and_push(chat, [{'type': 'mixed_album', 'items': [video, photo]}])
    assert not pushed
    assert any('видео' in text.lower() or 'video' in text.lower() for text in sent)


def test_mixed_album_is_not_split_by_a_neighbouring_photo(tmp_path, monkeypatch):
    import sys
    T.redirect_data_dir(str(tmp_path / 'data'))
    bot = T.TelegramBot('123:TEST', lambda: SimpleNamespace(), lambda: None, lambda: {}, silent_mode=True)
    chat = 999994
    bot._user_store.put(T._User(chat_id=chat, name='test', status='approved'))
    bot._send_text = lambda *a, **k: 1
    bot._api_post = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    bot._dl_bytes = lambda fid: {'photo': b'first photo', 'other': b'other photo', 'video': b'video'}[fid]
    bot._transcribe = lambda *a, **k: ''
    sheet = tmp_path / 'sheet.jpg'; sheet.write_bytes(b'sheet')
    clip = tmp_path / 'clip.mp4'; clip.write_bytes(b'video')
    bot._look_video = lambda *a, **k: {'sheet': str(sheet), 'description': 'A video.'}
    monkeypatch.setitem(sys.modules, 'ocr_reader', SimpleNamespace(read_pieces=lambda path: []))
    pushed = []
    bot._backend.push = pushed.append
    mixed = {'type': 'mixed_album', 'items': [{'file_id': 'photo', 'caption': ''},
             {'file_id': 'video', 'media': 'video', 'seconds': 3, 'caption': 'Compare these.'}]}
    bot._resolve_and_push(chat, [mixed, {'type': 'photo', 'file_id': 'other', 'caption': 'Describe this.'}])
    assert len(pushed) == 2
    first = next(t for t in pushed if 'Compare these.' in user_words(t.user_text))
    assert first.source_video_path == str(clip)
    assert open(first.image_path, 'rb').read() == b'first photo'


def test_mixed_album_with_all_files_unreadable_does_not_create_empty_task(tmp_path):
    T.redirect_data_dir(str(tmp_path / 'data'))
    bot = T.TelegramBot('123:TEST', lambda: SimpleNamespace(), lambda: None, lambda: {}, silent_mode=True)
    chat = 999993
    bot._user_store.put(T._User(chat_id=chat, name='test', status='approved'))
    sent, pushed = [], []
    bot._send_text = lambda *a, **k: sent.append(a[1]) or 1
    bot._api_post = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    bot._dl_bytes = lambda fid: b''
    bot._transcribe = lambda *a, **k: ''
    bot._look_video = lambda *a, **k: {}
    bot._backend.push = pushed.append
    bot._resolve_and_push(chat, [{'type': 'mixed_album', 'items': [
        {'file_id': 'photo', 'caption': ''},
        {'file_id': 'video', 'media': 'video', 'seconds': 3, 'caption': 'Compare.'}]}])
    assert not pushed and len(sent) >= 2


def test_mixed_album_does_not_claim_comparison_when_only_audio_is_readable(tmp_path):
    T.redirect_data_dir(str(tmp_path / 'data'))
    bot = T.TelegramBot('123:TEST', lambda: SimpleNamespace(), lambda: None, lambda: {}, silent_mode=True)
    chat = 999990
    bot._user_store.put(T._User(chat_id=chat, name='test', status='approved'))
    bot._get_ctx = lambda: SimpleNamespace()
    bot._get_graph = lambda: object()
    sent, pushed = [], []
    bot._send_text = lambda *a, **k: sent.append(a[1]) or 1
    bot._api_post = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    bot._dl_bytes = lambda fid: b'photo'
    bot._transcribe = lambda *a, **k: 'words from video'
    bot._look_video = lambda *a, **k: {}
    bot._backend.push = pushed.append
    bot._resolve_and_push(chat, [{'type': 'mixed_album', 'items': [
        {'file_id': 'photo'}, {'file_id': 'video', 'media': 'video', 'seconds': 3, 'caption': 'compare'}]}])
    assert not pushed and any('видео' in msg.lower() for msg in sent)


@pytest.mark.parametrize('field,value', [('pending_style_target', 'target-id'), ('awaiting_animate_photo', True)])
def test_armed_photo_action_rejects_mixed_album_without_losing_state(tmp_path, field, value):
    T.redirect_data_dir(str(tmp_path / 'data'))
    bot = T.TelegramBot('123:TEST', lambda: SimpleNamespace(), lambda: None, lambda: {}, silent_mode=True)
    chat = 999992
    bot._user_store.put(T._User(chat_id=chat, name='test', status='approved'))
    sess = bot._get_session(chat); setattr(sess, field, value); bot._store.put(sess)
    sent, pushed = [], []
    bot._send_text = lambda *a, **k: sent.append(a[1]) or 1
    bot._api_post = lambda *a, **k: {}
    bot._activity.log = lambda *a, **k: None
    bot._backend.push = pushed.append
    bot._resolve_and_push(chat, [{'type': 'mixed_album', 'items': [
        {'file_id': 'photo', 'caption': ''},
        {'file_id': 'video', 'media': 'video', 'seconds': 3, 'caption': ''}]}])
    assert not pushed and getattr(bot._get_session(chat), field) == value and sent


@pytest.mark.parametrize('items', [
    [{'file_id': 'v1', 'media': 'video', 'caption': 'compare'}, {'file_id': 'v2', 'media': 'video'}],
    [{'file_id': 'p1'}, {'file_id': 'p2'}, {'file_id': 'v', 'media': 'video', 'caption': 'compare'}],
])
def test_unsupported_multi_media_is_not_silently_analyzed_partially(tmp_path, items):
    T.redirect_data_dir(str(tmp_path / 'data'))
    bot = T.TelegramBot('123:TEST', lambda: SimpleNamespace(), lambda: None, lambda: {}, silent_mode=True)
    chat = 999991
    bot._user_store.put(T._User(chat_id=chat, name='test', status='approved'))
    sent, pushed = [], []
    bot._send_text = lambda *a, **k: sent.append(a[1]) or 1
    bot._backend.push = pushed.append
    bot._resolve_and_push(chat, [{'type': 'mixed_album', 'items': items}])
    assert sent and not pushed

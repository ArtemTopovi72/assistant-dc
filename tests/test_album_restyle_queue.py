"""One album style request must queue a redraw for each picture."""
import os
import pytest
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tg_bot as T
import tg_resolve as R
from fakes import reads
from test_tg_incoming_durability import make_bot


def test_all_photos_style_request_is_distinct_from_single_photo_and_description():
    assert R._wants_all_photos_restyled("все пять — в стиле киберпанк")
    assert R._wants_all_photos_restyled("перерисуй каждую фотографию в аниме стиле")
    assert R._wants_all_photos_restyled("all photos in cyberpunk style")
    assert not R._wants_all_photos_restyled("вторую фотографию в стиле аниме")
    assert not R._wants_all_photos_restyled("сравни все фотографии")
    assert not R._wants_all_photos_restyled("не все, только первую в стиле аниме")


def test_five_album_photos_queue_five_ordered_redraws(monkeypatch):
    saved, queued, sent, charged = [], [], [], []
    monkeypatch.setattr(R, "_save_image_bytes", lambda data: saved.append(data) or f"/{len(saved)}.jpg")
    monkeypatch.setattr(T, "_log_image", lambda sess, path, **kw: f"image-{len(saved)}")
    store = SimpleNamespace(get=lambda cid: SimpleNamespace(is_admin=False),
                            usage_today=lambda cid: {},
                            bump_usage=lambda cid, kind: charged.append(kind))
    backend = SimpleNamespace(chat_depth=lambda cid: 0, depth=lambda: len(queued))
    bot = SimpleNamespace(_user_store=store, _backend=backend, settings={},
                          _persist_task=lambda task: queued.append(task) or True,
                          _queue_note=lambda *a: None,
                          _send_text=lambda *a, **k: sent.append(a[1]),
                          _state_kb=lambda *a: None,
                          _main_menu_kb=lambda *a: None)
    R.ResolveMixin._push_album_restyle_tasks(bot, 42, SimpleNamespace(), "ru",
                                              'все пять в стиле "киберпанк"', b"1",
                                              [b"2", b"3", b"4", b"5"], 1.0, 123, 0)
    assert saved == [b"1", b"2", b"3", b"4", b"5"]
    assert [t.image_path for t in queued] == [f"/{i}.jpg" for i in range(1, 6)]
    assert all(t.reply_to == 123 and t.image_is_photo for t in queued)
    assert all(t.user_text.startswith('[style_preset] call redraw_image') for t in queued)
    assert all('[album_restyle_item]' in t.user_text for t in queued)
    assert all('все пять в стиле \\"киберпанк\\"' in t.user_text for t in queued)
    assert charged.count(T.KIND_IMAGE) == charged.count(T.KIND_TASK) == 5
    assert "5" in sent[-1] and "пропущ" not in sent[-1]


def test_album_restyle_does_not_exceed_daily_image_quota(monkeypatch):
    sent, queued = [], []
    store = SimpleNamespace(get=lambda cid: SimpleNamespace(is_admin=False),
                            usage_today=lambda cid: {T.KIND_IMAGE: 39},
                            bump_usage=lambda *a: None)
    bot = SimpleNamespace(_user_store=store,
                          _backend=SimpleNamespace(chat_depth=lambda cid: 0), settings={},
                          _send_text=lambda *a, **k: sent.append(a[1]),
                          _main_menu_kb=lambda *a: None,
                          _persist_task=lambda task: queued.append(task))
    R.ResolveMixin._push_album_restyle_tasks(bot, 42, SimpleNamespace(), "ru",
                                              "все фото в стиле аниме", b"1", [b"2"],
                                              1.0, 123, 0)
    assert not queued and sent


def test_followup_restyle_queues_each_previous_album_photo(tmp_path):
    bot = make_bot(tmp_path)
    sess = bot._get_session(101)
    paths = [tmp_path / 'first.jpg', tmp_path / 'second.jpg']
    for path, data in zip(paths, (b'first', b'second')):
        path.write_bytes(data)
    sess.turn_album = list(map(str, paths))
    bot._get_ctx = lambda: SimpleNamespace()
    bot.reads = SimpleNamespace(**reads())
    called = []
    bot._push_album_restyle_tasks = lambda *args: called.append(args)
    bot._resolve_and_push(101, [{'type': 'text', 'text': 'Все их сделай в стиле аниме'}])
    assert len(called) == 1
    assert called[0][4:6] == (b'first', [b'second'])


@pytest.mark.parametrize('image_source', ['none', 'target', 'turn', 'video'])
def test_bare_video_link_in_draw_menu_keeps_selected_image(tmp_path, monkeypatch, image_source):
    monkeypatch.delenv('F5_TEST_RUN', raising=False)
    bot = make_bot(tmp_path)
    sess = bot._get_session(101)
    sess.menu, sess.pending_prefix = 'draw', 'generate an image of: '
    old = tmp_path / 'old.jpg'
    old.write_bytes(b'old')
    if image_source != 'none':
        image_id = T._log_image(sess, str(old), src='video' if image_source == 'video' else 'user')
        setattr(sess, 'target_image' if image_source == 'target' else 'turn_image', image_id)
    bot._store.put(sess)
    bot._get_ctx = lambda: SimpleNamespace()
    bot.reads = SimpleNamespace(**reads(read=lambda _: {}, yes=lambda *a: False, pick=lambda *a: None))
    bot._bind_turn = lambda *a: None
    bot._backend = T.InMemoryBackend()
    seen, tasks = [], []
    def read(ctx, sess, chat, user, text, scaffold):
        seen.append(text)
        scaffold.append('[The video was watched.]')
        return False
    bot._read_links = read
    bot._persist_task = lambda task: tasks.append(task) or True
    bot._queue_note = lambda *a, **kw: None
    url = 'https://youtu.be/K7QHpeEsNy4'
    bot._resolve_and_push(101, [{'type': 'text', 'text': url}])
    assert seen == [url]
    has_photo = image_source in ('target', 'turn')
    assert tasks and bool(tasks[0].image_id) == has_photo and not tasks[0].image_path
    assert not tasks[0].user_text.startswith('generate an image of:')
    assert ('вместе с выбранной картинкой' if has_photo else 'Кратко перескажи видео по ссылке:') in tasks[0].user_text


def test_video_link_in_photo_caption_is_read(tmp_path, monkeypatch):
    monkeypatch.delenv('F5_TEST_RUN', raising=False)
    bot = make_bot(tmp_path)
    bot._get_ctx = lambda: SimpleNamespace()
    bot.reads = SimpleNamespace(**reads(read=lambda _: {}, yes=lambda *a: False, pick=lambda *a: None))
    bot._dl_bytes = lambda _: b'photo'
    seen = []
    bot._read_links = lambda ctx, sess, chat, user, text, scaffold: seen.append(text) or True
    url = 'https://youtu.be/K7QHpeEsNy4'
    bot._resolve_and_push(101, [{'type': 'photo', 'file_id': 'p', 'caption': url}])
    assert seen == [url]

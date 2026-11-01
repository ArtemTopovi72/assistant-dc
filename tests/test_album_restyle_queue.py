"""One album style request must queue a redraw for each picture."""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tg_bot as T
import tg_resolve as R


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

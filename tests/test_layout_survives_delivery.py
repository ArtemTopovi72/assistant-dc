"""The layout record has to survive the trip to Telegram and back.

Layout-first editing rests on one assumption: given the path the session logged
when a picture was sent, the layout it was composed from can be found again.
If it cannot, "передвинь кота на диван" silently falls through to the pixel
pipelines -- which repaint the frame instead of moving a box, and that is the
whole failure the layout model exists to avoid.

The record is a sidecar beside the FILE, so anything that copies, moves or
renames a delivered render breaks it. Nothing does today; this is what notices
when something starts to.

Run: venv/Scripts/python.exe -m pytest tests/test_layout_survives_delivery.py -q
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import image as I


def _render(dirpath, name="render_1234.png"):
    p = os.path.join(dirpath, name)
    with open(p, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n" + b"\0" * 64)
    return p


LAYOUT = {"elements": [{"desc": "серый кот", "x": 0.1, "y": 0.1,
                        "w": 0.3, "h": 0.3}]}


def test_a_saved_layout_comes_back_with_its_seed(tmp_path):
    """The seed is what makes a re-render an EDIT rather than a new roll."""
    p = _render(str(tmp_path))
    I.save_layout_for(p, "кухня", LAYOUT, width=1024, height=1024, seed=7)
    rec = I.load_layout_for(p)
    assert rec and rec["layout"]["elements"][0]["desc"] == "серый кот"
    assert rec.get("seed") == 7, rec


def test_the_path_telegram_logs_is_the_path_that_carries_the_layout(tmp_path):
    import tg_bot as T
    T.redirect_data_dir(tempfile.mkdtemp(prefix="layoutdel_"))
    p = _render(str(tmp_path))
    I.save_layout_for(p, "кухня", LAYOUT, width=1024, height=1024, seed=7)

    sess = T._Session(chat_id=1)
    T._log_image(sess, p, label="кухня", src="bot")
    logged = [e.get("path") for e in (sess.image_log or [])]
    assert p in logged, logged
    assert I.load_layout_for(logged[0]), "the layout is unreachable from the log"


def test_a_photo_the_user_uploaded_has_no_layout(tmp_path):
    """And must not pretend to: an uploaded photo has no boxes, so the caller
    correctly falls back to the pixel pipelines."""
    assert I.load_layout_for(_render(str(tmp_path), "upload.jpg")) is None


def test_an_unreadable_record_is_not_a_crash(tmp_path):
    p = _render(str(tmp_path))
    with open(p + ".layout.json", "w", encoding="utf-8") as fh:
        fh.write("{not json")
    assert I.load_layout_for(p) is None


def test_a_record_without_a_layout_is_rejected(tmp_path):
    p = _render(str(tmp_path))
    with open(p + ".layout.json", "w", encoding="utf-8") as fh:
        json.dump({"prompt": "кухня", "seed": 7}, fh)
    assert I.load_layout_for(p) is None


def test_a_copied_render_keeps_its_layout_when_the_sidecar_travels(tmp_path):
    """The constraint, stated as a test rather than as a comment.

    Nothing copies a delivered render today. If something starts to -- a
    thumbnailer, a per-chat archive, a cleanup that rewrites paths -- it has to
    bring image.layout_sidecar(path) along, and this is where that is written
    down.
    """
    import shutil
    src = _render(str(tmp_path))
    I.save_layout_for(src, "кухня", LAYOUT, width=1024, height=1024, seed=7)

    far = tmp_path / "delivered"
    far.mkdir()
    dst = str(far / "copy.png")
    shutil.copy2(src, dst)
    assert I.load_layout_for(dst) is None, "the layout cannot follow a bare copy"

    shutil.copy2(str(I.layout_sidecar(src)), str(I.layout_sidecar(dst)))
    rec = I.load_layout_for(dst)
    assert rec and rec.get("seed") == 7, rec

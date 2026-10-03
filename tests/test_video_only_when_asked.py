"""A clip is made only when the user's own words ask for one.

Live 10-03: «Почему он у тебя не лысый … ты нарисовал какого-то урода, а не
Ленина, ещё и мультяшного» -- a complaint about a picture -- started a video
(«🎬 делаю видео») and burnt the turn's render budget.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("F5_TEST_RUN", "1")

import tool_image_handlers as H  # noqa: E402


class _Ctx:
    last_video_path = None


def _st(text):
    return {"user_input_original": text, "user_input": text}


def test_a_complaint_about_a_picture_is_not_a_video_request():
    said = ("Почему он у тебя не Лысый и где реально, ты нарисовал какого-то урода "
            "а не Ленина ещё и мультяшного")
    assert not H.asks_for_video(_Ctx(), _st(said))
    out = H._handle_generate_video(_Ctx(), _st(said), {"description": "Lenin fights demons"})
    assert out.startswith("[TOOL ERROR] The user did not ask for a video")


def test_real_requests_and_buttons_pass():
    for t in ("сделай видео где он бьёт демона", "оживи фото", "анимируй картинку",
              "make a short clip of it", "[animate] call generate_video with description=\"wave\"",
              "animate this photo: he waves"):
        assert H.asks_for_video(_Ctx(), _st(t)), t


def test_redo_after_a_clip_passes(tmp_path, monkeypatch):
    clip = tmp_path / "v.mp4"
    clip.write_bytes(b"x")
    c = _Ctx(); c.last_video_path = str(clip)
    import intent
    monkeypatch.setattr(intent, "read", lambda ctx, t: {"redo": "same"})
    assert H.asks_for_video(c, _st("давай ещё раз"))
    assert not H.asks_for_video(_Ctx(), _st("давай ещё раз"))     # no clip before: no

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


def test_a_cartoon_style_is_not_an_animation_request():
    """The root of «…ещё и мультяшного» -> 🎬: the video tool said «animation»,
    and «мультяшный» (cartoon) read as animated. Both the tool and the router
    now say a picture's style is not a video (10-03)."""
    import intent
    from tool_descriptions import _GENERATE_VIDEO_DESC
    assert "мультяшный" in _GENERATE_VIDEO_DESC and "not a video" in _GENERATE_VIDEO_DESC
    assert "generate_video only when MOTION is asked for" in intent.SYSTEM
    import tool_image_handlers as H
    assert not H._ASKS_VIDEO.search("ты нарисовал какого-то урода а не Ленина ещё и мультяшного")


def test_a_shot_script_without_the_word_video_is_asked_to_the_model(monkeypatch):
    """Live 10-06: a scene script («Мужчина берёт банку … говорит … бросает …
    опрокидывает стеллаж») has no «видео» and was refused. The model now reads it."""
    import intent
    script = ("Мужчина берёт банку, смотрит на неё внимательно. Ухмыляется, смотрит в камеру. "
              "Говорит \"Полная хуета\" и бросает банку на пол. Опрокидывает весь стеллаж.")
    asked = []
    monkeypatch.setattr(intent, "YES_STUB", lambda q, t: asked.append(t) or "unfolds over time" in q)
    assert H.asks_for_video(_Ctx(), _st(script))
    assert asked and "банку" in asked[0]
    monkeypatch.setattr(intent, "YES_STUB", lambda q, t: False)
    assert not H.asks_for_video(_Ctx(), _st(script))


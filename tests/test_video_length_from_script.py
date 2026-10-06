"""A clip with no asked length is as long as its script needs (5.2 s .. the auto cap),
not always 5.2 s: live 10-06 every multi-action script ended cut short."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("", "core", "agent", "media"):
    sys.path.insert(0, os.path.join(ROOT, d))
os.environ.setdefault("F5_TEST_RUN", "1")

import video as V  # noqa: E402

BROOM = ("The man puts on a formal suit, takes a broom, and begins sweeping the floor; he sweeps "
         "for several seconds, then stops, looks at the camera, and says \"Ладно извините\". "
         "He then resumes sweeping.")


def test_one_beat_stays_short():
    assert V.estimate_seconds("A cat sleeps on a sofa.") == V.frames_to_seconds(V.VIDEO_DEFAULT_FRAMES)


def test_many_beats_get_longer_up_to_the_auto_cap():
    s = V.estimate_seconds(BROOM)
    assert s > 9.0, s
    assert s <= V.frames_to_seconds(V.VIDEO_MAX_FRAMES_AUTO)


def test_spoken_lines_add_time():
    short = V.estimate_seconds('He turns and says «Да».')
    long_ = V.estimate_seconds('He turns and says «Да, я сегодня никуда не пойду, потому что идёт дождь и мне лень».')
    assert long_ > short


def test_continue_template_does_not_count():
    assert V.estimate_seconds(V.CONTINUE_PREFIX + "he waves.") == V.estimate_seconds("he waves.")


def test_generate_video_uses_the_estimate(monkeypatch):
    seen = {}
    monkeypatch.setattr(V, "engine_available", lambda ctx: (True, ""))
    monkeypatch.setattr(V, "_context_ir_on", lambda: False)
    monkeypatch.setattr(V, "VIDEO_SPEECH_STRESS", False)

    def stop(*a, **k):
        seen["frames"] = k.get("frames")
        raise RuntimeError("stop here")
    for name in ("_run_t2va", "_render", "_submit_h3", "run_h3"):
        if hasattr(V, name):
            monkeypatch.setattr(V, name, stop)
    real = V.seconds_to_frames
    monkeypatch.setattr(V, "seconds_to_frames", lambda s: seen.setdefault("secs", s) and real(s))
    try:
        V.generate_video(None, BROOM)
    except Exception:
        pass
    assert seen.get("secs", 0) > 9.0, seen

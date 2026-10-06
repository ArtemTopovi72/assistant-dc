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


LONG = ("The man puts on a formal suit, takes a broom and begins sweeping the floor. He sweeps for "
        "several seconds, then stops, looks at the camera and says «Ладно, извините, я сейчас всё "
        "уберу, честное слово». He then resumes sweeping, knocks over a bucket, slips on the water "
        "and falls. He gets up, laughs and says «Ну и денёк».")


def test_a_long_script_is_split_into_parts_that_each_fit():
    parts = V.split_script(LONG)
    cap = V.frames_to_seconds(V.VIDEO_MAX_FRAMES_AUTO)
    assert len(parts) >= 2, parts
    assert all(V.estimate_seconds(p, capped=False) <= cap for p in parts), parts
    assert " ".join(parts).split() == LONG.split()          # nothing lost, nothing reordered
    assert all(p.count("«") == p.count("»") for p in parts)  # a spoken line is never cut


def test_a_short_script_stays_one_part():
    short = "He picks up a mug, drinks and says «Хорошо»."
    assert V.split_script(short) == [short]


def test_the_parts_are_rendered_as_continuations_and_joined(monkeypatch, tmp_path):
    import tool_image_handlers as H
    import tg_continue
    calls = []

    def fake_gen(ctx, description, **kw):
        out = tmp_path / f"part{len(calls) + 1}.mp4"
        out.write_bytes(b"v")
        calls.append((description, kw.get("images"), kw.get("videos")))
        return {"path": str(out), "status": "success", "seconds": 8.0, "width": 768, "height": 1344}
    monkeypatch.setattr(V, "generate_video", fake_gen)
    monkeypatch.setattr(V, "engine_available", lambda ctx: (True, ""))
    monkeypatch.setattr(tg_continue, "seed_frame", lambda src, out: open(out, "wb").write(b"j") > 0)
    monkeypatch.setattr(tg_continue, "cut_tail", lambda src, out: open(out, "wb").write(b"t") > 0)
    joins = []

    def fake_join(a, b):
        j = tmp_path / f"joined{len(joins)}.mp4"
        j.write_bytes(b"j")
        joins.append((os.path.basename(a), os.path.basename(b)))
        return str(j)
    monkeypatch.setattr(V, "join_continuation", fake_join)
    monkeypatch.setattr(V, "_adopt_output", lambda p: p)
    monkeypatch.setattr(V, "probe", lambda p: {"seconds": 24.0})

    class Ctx:
        reference_images = []
        voice_choice = None
        def set_stage(self, *_): pass
        def is_cancelled(self): return False
        def remember(self, *a): pass
        def memory_text(self): return ""
    st = {}
    out = H._handle_generate_video(Ctx(), st, {"description": LONG, "use_current_images": False})
    n = len(V.split_script(LONG))
    assert len(calls) == n, calls
    assert not calls[0][0].startswith(V.CONTINUE_PREFIX)
    assert all(c[0].startswith(V.CONTINUE_PREFIX) and c[2] and c[1] for c in calls[1:]), calls
    assert len(joins) == n - 1 and st["video_path"].endswith(f"joined{n - 2}.mp4"), (joins, st)
    assert "24.0s" in out


def test_the_part_stage_is_shown_in_russian():
    import stages
    assert stages.translate("Generating a video (2/3)", "ru") == "Генерирую видео (2/3)"

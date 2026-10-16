"""The bot looks at a video, not only listens (a
forwarded round video was reported as «Пересланное голосовое»).

Pure parts run without ffmpeg's vision; the sheet+vision path uses a fake
vision so the real extraction/tiling code runs on a real clip.
"""
import os, sys, shutil, subprocess, inspect
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import video_look as V


def test_sample_times_every_5s_capped():
    assert V.sample_times(18) == [0, 5, 10, 15]
    assert V.sample_times(3) == [0]
    assert V.sample_times(0) == [0.0]
    long = V.sample_times(600)
    assert len(long) == V.MAX_FRAMES and long[0] == 0 and long[-1] < 600


def _clip(tmp_path):
    assert shutil.which("ffmpeg"), "ffmpeg is installed in CI (choco)"
    out = tmp_path / "c.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "color=c=red:s=160x120:d=6", "-f", "lavfi", "-i",
                    "color=c=blue:s=160x120:d=6", "-filter_complex",
                    "[0][1]concat=n=2:v=1:a=0", "-t", "12", str(out)],
                   check=True, timeout=60)
    return out


def test_look_at_video_builds_a_sheet_and_asks_once(tmp_path):
    clip = _clip(tmp_path)
    calls = []
    def vision(p, q, system):
        calls.append((p, q))
        return "Red, then blue."
    r = V.look_at_video(None, str(clip), work_dir=tmp_path / "w", vision=vision)
    assert r["description"] == "Red, then blue."
    # key moments now, not a 5 s grid: one pick in the red part, one in the blue
    assert any(t < 6 for t in r["times"]) and any(t >= 6 for t in r["times"]), r["times"]
    assert Path(r["sheet"]).exists()
    # A projector became «красный цилиндр» at sheet size.
    # ONE call gets every frame as its own full-size picture, time-labelled.
    assert len(calls) == 1 and "12-second video" in calls[0][1], calls
    sent = calls[0][0]
    assert isinstance(sent, list) and len(sent) == len(r["times"]), sent
    assert all(lbl.startswith("Frame ") and "frame_" in p for lbl, p in sent), sent


def test_a_short_clip_is_sampled_densely_and_labels_stay_distinct():
    """A 7 s кружок at 2 fps had two candidates a second, both smeared while the dog
    jumped; two picks inside one second were both labelled «0:02»."""
    assert V.dense_fps(7) > V.dense_fps(45) > V.dense_fps(600) == V.DENSE_FPS
    assert V._stamp(2.04, True) != V._stamp(2.8, True) and V._stamp(2.8) == "0:02"
    assert V._stamp(65.3, True) == "1:05.3"


def test_portion_stamps_are_absolute_even_when_the_model_counts_from_zero():
    times = [365.0, 373.0, 390.0]
    desc = "0:05 — a\n0:13 — b\n6:30 — c\nИтог: x"
    got = V.absolute_stamps(desc, times, 360).splitlines()
    assert got == ["6:05 — a", "6:13 — b", "6:30 — c", "Итог: x"], got
    assert V.absolute_stamps(desc, [5.0, 13.0], 0) == desc


def test_wired_into_the_video_paths():
    import tg_resolve, tg_tasks, tg_dispatch, tg_bot
    assert hasattr(tg_tasks.TaskRunnerMixin, "_look_video")
    src = inspect.getsource(tg_resolve)
    assert src.count("self._look_video(") == 3          # direct, forwarded, video link
    assert "fwd_video_ask" in src
    assert 'msg.get("video_note") or msg.get("video")' in inspect.getsource(tg_dispatch)
    note = tg_bot._video_seen_note({"duration": 33, "times": [0, 5, 10], "description": "a lamp"})
    assert "33 s" in note and "a lamp" in note and "3 frames" in note
    # The storyboard is behind its own button now, not in the ask.
    assert "лампа" not in tg_bot._t("fwd_video_ask", "ru", mins="34 с", seen="0:00 — лампа")
    assert "fwdv:board" in str(tg_bot._fwd_voice_kb("ru", "x", board=True))
    # The 📝 button shows the forwarded transcript verbatim, so the seen line
    # there is the user's language, not the English scaffold note.
    assert 'tg_bot._t("fwd_video_seen"' in src and "_video_seen_note(_seen)" in src
    assert tg_bot._t("fwd_video_seen", "ru", seen="0:00 — лампа").startswith("👁 Раскадровка видео:\n0:00")
    # The 📋 retelling of a forwarded video reads the storyboard too.
    tsrc = inspect.getsource(tg_tasks)
    assert "_retell_transcript" in tsrc and "STORYBOARD" in tsrc and '"👁 " in said' in tsrc
    assert "transcript=" in inspect.getsource(tg_resolve)     # the ears help the eyes
    assert "one line per frame" in V._SHEET_SYSTEM


if __name__ == "__main__":
    import tempfile
    test_sample_times_every_5s_capped()
    test_look_at_video_builds_a_sheet_and_asks_once(Path(tempfile.mkdtemp()))
    test_a_short_clip_is_sampled_densely_and_labels_stay_distinct()
    test_portion_stamps_are_absolute_even_when_the_model_counts_from_zero()
    test_wired_into_the_video_paths()
    print("5/5 passed")

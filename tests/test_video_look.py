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


def test_a_document_in_the_video_is_read_from_its_own_sharp_frames(tmp_path):
    """A receipt panned past a round video's camera came back as «RACUN - NARUDZBA» only, and
    «сколько стоит чай» was answered with a guessed 12 € (live 2026-10-24). The storyboard frame
    is too small to read; several sharp full-size frames of that stretch read every line."""
    clip = _clip(tmp_path)
    calls = []
    def vision(p, q, system):
        calls.append((p, q, system))
        if len(calls) == 1:
            return "0:00 — чек на столе\n0:07 — синий фон\nИтог: чек.\nTEXT AT: 0:00, 0:03"
        return "CAJ MINT 2,000 6,50 13,00"
    r = V.look_at_video(None, str(clip), work_dir=tmp_path / "w", vision=vision)
    assert len(calls) == 2, calls
    frames, _, system = calls[1]
    assert 2 <= len(frames) <= V.READ_FRAMES and all(0 <= float(l.split()[1]) <= 3 + V.READ_PAD for l, _ in frames), frames
    assert "never guess" in system
    assert "TEXT AT" not in r["description"] and "CAJ MINT 2,000 6,50 13,00" in r["description"], r
    # one chat line of it for later reads (chatlog «seen»): the overall line and the document
    assert r["gist"].startswith("Итог: чек.") and "CAJ MINT" in r["gist"] and "0:07" not in r["gist"], r["gist"]
    import chatlog
    line = chatlog._line({"kind": "seen", "data": {"text": r["gist"]}})
    assert line.startswith("bot: [sees in it: Итог: чек.") and "CAJ MINT" in line, line
    assert chatlog._line({"kind": "out", "data": {"method": "sendMessage", "text": "⚙️ Пишу ответ… ⏳ 0:24"}}) == ""
    calls.clear()
    def plain(p, q, system):
        calls.append(1)
        return "0:00 — красное\nИтог: цвета."
    assert "TEXT AT" in V._SHEET_SYSTEM
    V.look_at_video(None, str(clip), work_dir=tmp_path / "w2", vision=plain)
    assert len(calls) == 1


def test_a_short_clip_is_sampled_densely_and_labels_stay_distinct():
    """A 7 s кружок at 2 fps had two candidates a second, both smeared while the dog
    jumped; two picks inside one second were both labelled «0:02»."""
    assert V.dense_fps(7) > V.dense_fps(45) > V.dense_fps(600) == V.DENSE_FPS
    assert V._stamp(2.04, True) != V._stamp(2.8, True) and V._stamp(2.8) == "0:02"
    assert V._stamp(65.3, True) == "1:05.3"


def test_motion_details_compare_two_distinct_positions(tmp_path, monkeypatch):
    import numpy as np
    from PIL import Image
    frames = []
    for i in range(6):
        a = np.zeros((60, 80), dtype=np.float32)
        a[25:35, 10 + (i % 2) * 10:20 + (i % 2) * 10] = 255
        frames.append((i / 6, a))
    def extract(path, times, directory):
        directory.mkdir(parents=True, exist_ok=True)
        result = []
        for i, t in enumerate(times):
            frame = directory / f'{i}.jpg'
            Image.new('RGB', (800, 600)).save(frame)
            result.append((t, frame))
        return result
    monkeypatch.setattr(V, 'extract_frames', extract)
    result = V.motion_details('clip.mp4', frames, tmp_path)
    assert len(result) == 2 and result[0][0] != result[1][0]
    assert all(Image.open(p).width < 800 for _, p in result)
    assert V.motion_details('clip.mp4', [(i, frames[0][1]) for i in range(6)], tmp_path) == []


def test_motion_comparison_requires_two_details(tmp_path, monkeypatch):
    clip = _clip(tmp_path)
    calls = []
    def vision(p, q, system):
        calls.append(q)
        return 'Storyboard survives'
    monkeypatch.setattr(V, 'motion_details', lambda *args: [(0.1, clip)])
    result = V.look_at_video(None, str(clip), work_dir=tmp_path / 'one', vision=vision)
    assert result['description'] == 'Storyboard survives' and len(calls) == 1
    calls.clear()
    monkeypatch.setattr(V, 'motion_details', lambda *args: [(0.1, clip), (0.2, clip)])
    result = V.look_at_video(None, str(clip), work_dir=tmp_path / 'two', vision=vision)
    assert result['description'] == 'Storyboard survives' and len(calls) == 1
    assert 'last two pictures' in calls[0]


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
    test_a_document_in_the_video_is_read_from_its_own_sharp_frames(Path(tempfile.mkdtemp()))
    test_a_short_clip_is_sampled_densely_and_labels_stay_distinct()
    test_portion_stamps_are_absolute_even_when_the_model_counts_from_zero()
    test_wired_into_the_video_paths()
    print("6/6 passed")

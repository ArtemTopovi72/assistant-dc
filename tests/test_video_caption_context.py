"""A forwarded video's caption is the setup of its joke. The retelling ignored it and read a
meme («Когда начальство внедрило ИИ…» over girls hauling sand) as a literal construction site."""
import os, sys, tempfile
from pathlib import Path
os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tg_tasks
import tg_bot
import video_look


def test_the_retelling_is_told_a_mismatched_caption_is_a_joke():
    sys_ru = tg_tasks.retell_system("ru", True)
    assert "meme" in sys_ru and "literal" in sys_ru


def test_the_caption_line_exists_in_both_languages():
    assert "ИИ" in tg_bot._t("fwd_video_caption", "ru", caption="ИИ")
    assert "ИИ" in tg_bot._t("fwd_video_caption", "en", caption="ИИ")


def test_the_vision_pass_sees_the_caption(tmp_path):
    import subprocess
    clip = tmp_path / "c.mp4"
    r = subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=d=3:s=160x120:r=10", str(clip)],
                       capture_output=True)
    if r.returncode:
        import pytest; pytest.skip("no ffmpeg")
    asked = []
    video_look.look_at_video(None, str(clip), work_dir=tmp_path / "w",
                             vision=lambda frames, q, system: asked.append(q) or "x",
                             caption="Когда начальство внедрило ИИ")
    assert asked and "Когда начальство внедрило ИИ" in asked[0]

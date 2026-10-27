"""Rejected media uploads must not silently finish their delivery jobs."""
import json
import os
import sys
import threading
import types
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tg_bot
import tg_video
from tg_cover import CoverMixin


@pytest.mark.parametrize("kind", ["mashup", "rvc"])
def test_cover_reports_a_rejected_optional_upload_and_keeps_the_render(tmp_path, kind):
    output = tmp_path / "render.mp3"
    output.write_bytes(b"completed song")

    class Bot(CoverMixin):
        def __init__(self):
            self.messages = []
            self.uploads = []
            self.modules = {
                "mashup_auto": types.SimpleNamespace(make=lambda *a: str(output)),
                "mashup_stems": types.SimpleNamespace(release_separator=lambda: None),
                "remix": types.SimpleNamespace(resing_available=lambda: True, resing=self.resing,
                                                resing_rvc=lambda *a: str(output)),
            }

        def resing(self, ctx, src, lyrics, info):
            info.update(rvc="artist", trained=False)
            return "first.mp3"

        def _send_audio(self, chat, path, caption):
            self.uploads.append(path)
            return path == "first.mp3"

        def _send_text(self, chat, text, **kw):
            self.messages.append(text)
            return 1

        def _get_ctx(self):
            return None

        def _get_session(self, chat):
            return types.SimpleNamespace(cover_voice="")

    bot = Bot()
    if kind == "mashup":
        bot._mashup_first(1, "en", None, "song1", "song2")
    else:
        bot._cover_render(1, "en", "song1", "new words", "")
    assert tg_bot._t("cover_fail_render", "en") in bot.messages
    assert bot.uploads.count(str(output)) == 1
    assert output.read_bytes() == b"completed song"


@pytest.fixture
def video_directory(tmp_path):
    previous = Path(tg_bot._IMAGE_DIR).parent
    tg_bot.redirect_data_dir(tmp_path)
    Path(tg_bot._IMAGE_DIR).mkdir(parents=True)
    try:
        yield Path(tg_bot._IMAGE_DIR)
    finally:
        tg_bot.redirect_data_dir(previous)


class VideoBot(tg_video.LongVideoMixin):
    def __init__(self):
        self.accept = False
        self.accepted = []
        self.calls = []
        self._activity = types.SimpleNamespace(log=lambda *a: None)
        self.ctx = types.SimpleNamespace(text_model=self.model)

    def model(self, ctx, system, user, **kw):
        self.calls.append(user)
        return ""

    def _get_session(self, chat):
        return types.SimpleNamespace(lang="en")

    def _lang(self, sess):
        return sess.lang

    def _get_ctx(self):
        return self.ctx

    def _api_post(self, *a):
        return {}

    def _send_text(self, chat, text, **kw):
        return 1

    def _send_get_id(self, chat, text, **kw):
        if kw.get("parse_mode") != "HTML":
            return 1
        if not self.accepted or self.accept:
            self.accepted.append(text)
            return len(self.accepted)
        return None

def test_long_video_recovery_retries_only_messages_not_already_accepted(video_directory):
    work = video_directory / "longvideo_aa11"
    work.mkdir()
    state = {"chat": 1, "parts": [{"a": "0:00", "b": "2:00", "note": "A" * 3500 + "\n" + "B" * 3500}]}
    path = work / "material.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    bot = VideoBot()

    bot._recover_long_videos()
    pending = json.loads(path.read_text(encoding="utf-8"))
    assert not pending.get("done")
    assert pending["delivered"] == 1
    assert len(bot.accepted) == 1

    bot.accept = True
    bot._recover_long_videos()
    complete = json.loads(path.read_text(encoding="utf-8"))
    assert complete["done"] is True
    assert len(bot.accepted) == len(complete["delivery"])
    assert len(set(bot.accepted)) == len(bot.accepted)
    bot._recover_long_videos()
    assert len(bot.accepted) == len(complete["delivery"])


def test_long_video_recovery_keeps_notes_when_composing_the_answer_fails(video_directory):
    work = video_directory / "longvideo_bb22"
    work.mkdir()
    state = {"chat": 1, "question": "What happened?", "parts": [{"a": "0:00", "b": "2:00", "board": "A person enters."}]}
    path = work / "material.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    bot = VideoBot()

    bot._recover_long_videos()
    pending = json.loads(path.read_text(encoding="utf-8"))
    assert not pending.get("done")
    assert pending["parts"] == state["parts"]
    assert bot.calls and "What happened?" in bot.calls[0]


def test_long_video_photo_exception_keeps_its_delivery_position(video_directory):
    work = video_directory / "longvideo_dd44"
    work.mkdir()
    photo = work / "sheet.jpg"
    photo.write_bytes(b"retained sheet")
    state = {"chat": 1, "parts": [{"a": "0:00", "b": "2:00", "board": "A person enters."}],
             "delivery": [{"photo": str(photo)}], "delivered": 0}
    bot = VideoBot()
    bot._send_photo = lambda *a, **kw: (_ for _ in ()).throw(OSError("upload connection lost"))

    assert bot._deliver_long_video(1, "en", "dd44", [], state) is False
    pending = json.loads((work / "material.json").read_text(encoding="utf-8"))
    assert pending["delivered"] == 0
    assert photo.exists()


def test_long_storyboard_is_split_before_real_transport_can_truncate_it(video_directory):
    from tg_transport import TransportMixin

    class Bot(tg_video.LongVideoMixin, TransportMixin):
        def __init__(self):
            self.accepted = []

        def _api_post(self, method, payload):
            self.accepted.append(dict(payload))
            return {"ok": True, "result": {"message_id": len(self.accepted)}}

        def _get_session(self, chat):
            return types.SimpleNamespace(lang="en")

        def _lang(self, sess):
            return sess.lang

    work = video_directory / "longvideo_ee55"
    work.mkdir()
    bot = Bot()
    text = "<b>Storyboard</b>\n" + "SCENE " * 1000 + "END_OF_STORYBOARD"
    keyboard = {"inline_keyboard": [[{"text": "more", "callback_data": "lv:parts:ee55"}]]}
    state = {"parts": [{"a": "0:00", "b": "2:00", "board": "notes"}]}

    assert bot._deliver_long_video(1, "en", "ee55", [
        {"text": text, "parse_mode": "HTML", "keyboard": keyboard}], state)

    accepted = [p["text"] for p in bot.accepted]
    assert len(accepted) >= 2
    assert all(len(piece) <= tg_bot._MAX_TEXT for piece in accepted)
    assert "".join(accepted).count("SCENE") == 1000
    assert accepted[-1].endswith("END_OF_STORYBOARD")
    assert all("reply_markup" not in p for p in bot.accepted[:-1])
    assert "reply_markup" in bot.accepted[-1]
    saved = json.loads((work / "material.json").read_text(encoding="utf-8"))
    assert saved["delivered"] == len(accepted) == len(saved["delivery"])


@pytest.mark.parametrize("mode", ["brief", "both"])
def test_initial_long_video_job_keeps_rejected_output_for_recovery(video_directory, tmp_path, mode):
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is required for a local clip")
    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "color=c=black:s=32x32:d=0.4", str(clip)], check=True, capture_output=True)
    bot = VideoBot()
    result = "A" * 3500 + "\n" + "B" * 3500
    bot.ctx.text_model = lambda *a, **kw: result
    bot.ctx.deps = types.SimpleNamespace(WHISPER_LANGUAGE="en")
    bot.modules = {
        "audio": types.SimpleNamespace(transcribe_audio_file=lambda *a, **kw: ""),
        "video_look": types.SimpleNamespace(duration_seconds=lambda p: 0.4,
                                             look_at_video=lambda *a, **kw: {"description": "A person enters."}),
    }
    bot._dl_bytes = lambda fid: clip.read_bytes()
    bot._edit_text = lambda *a, **kw: None
    stop = threading.Event()
    bot._lv_jobs()[1] = stop

    bot._long_video_job(1, {"id": "cc33", "file_id": "file", "out": mode}, stop)

    path = video_directory / "longvideo_cc33" / "material.json"
    pending = json.loads(path.read_text(encoding="utf-8"))
    assert not pending.get("done")
    assert pending["delivered"] == 1
    assert len(bot.accepted) == 1
    assert not (path.parent / "src.mp4").exists()

    bot.accept = True
    bot._recover_long_videos()
    complete = json.loads(path.read_text(encoding="utf-8"))
    assert complete["done"] is True
    assert len(bot.accepted) == len(complete["delivery"])
    assert len(set(bot.accepted)) == len(bot.accepted)

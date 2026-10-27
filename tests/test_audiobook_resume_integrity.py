"""An unsuccessful chapter remains the next chapter when an audiobook resumes."""
import os
import sys
import threading
import types
from pathlib import Path

import pytest
from pydub.generators import Sine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tg_audiobook import AudiobookMixin


class BookBot(AudiobookMixin):
    def __init__(self, directory, failure):
        self.directory = str(directory)
        self.failure = failure
        self.messages = []
        self.delivered = []
        self.voice_clone = types.SimpleNamespace(speak=self.speak)

    def _book_dir(self, chat_id):
        return self.directory

    def _send_text(self, chat_id, text, **kwargs):
        self.messages.append(text)

    def _send_voice_from_wav(self, chat_id, wav, **kwargs):
        if self.failure == "delivery" and Path(wav).name == "ch2.wav":
            return False
        self.delivered.append(Path(wav).name)
        return True

    def speak(self, ctx, ref, ref_text, text, work):
        if self.failure == "render" and text.startswith("Глава 2."):
            return None
        path = Path(work) / "piece.wav"
        Sine(220).to_audio_segment(duration=4000).export(path, format="wav")
        return str(path)


@pytest.mark.parametrize("failure", ["render", "delivery"])
def test_failed_chapter_stays_pending_and_is_delivered_on_resume(tmp_path, failure):
    bot = BookBot(tmp_path, failure)
    ref = tmp_path / "narrator.wav"
    ref.write_bytes(b"reference kept for resume")
    chapters = [[f"Глава {n}", "Путешественник шёл по дороге и вспоминал о родном доме у реки."] for n in (1, 2, 3)]
    bot._job_save(1, {"chapters": chapters, "ref": str(ref), "ref_text": "образец", "next": 0})
    ctx = types.SimpleNamespace(
        is_cancelled=lambda: False, asr_lock=threading.Lock(), transcription_cache={}, save_cache=lambda: None,
        models=types.SimpleNamespace(whisper=types.SimpleNamespace(transcribe=lambda *a, **kw: ([], None))),
        deps=types.SimpleNamespace(_use_gigaam=lambda: False))

    bot._book_run(ctx, 1, "ru", chapters, str(ref), "образец")

    job = bot._job_load(1)
    assert job is not None
    assert job["next"] == 1
    assert bot.delivered == ["ch1.wav"]
    assert not list(tmp_path.glob("run_*"))

    bot.failure = ""
    bot._book_run(ctx, 1, "ru", job["chapters"], job["ref"], job["ref_text"], job["next"])
    assert bot.delivered == ["ch1.wav", "ch2.wav", "ch3.wav"]
    assert bot._job_load(1) is None
    assert not (tmp_path / "job.json").exists()
    assert ref.exists()
    assert not list(tmp_path.glob("run_*"))

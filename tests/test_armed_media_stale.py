"""Slow uploads must not resurrect an abandoned media flow."""
import os
import sys
import threading
import tempfile
from types import SimpleNamespace as NS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT] + [os.path.join(ROOT, d) for d in ("agent", "bot", "core", "imaging")]
import tg_bot
from tg_anim_voices import AnimVoicesMixin
from tg_audiobook import AudiobookMixin
from tg_continue import ContinueMixin
from tg_restyle import RestyleMixin
from tg_cover import CoverMixin
from tg_voice_clone import VoiceCloneMixin


class Media(AnimVoicesMixin, AudiobookMixin, ContinueMixin, RestyleMixin, CoverMixin, VoiceCloneMixin):
    def __init__(self, field, state):
        self.sess = tg_bot._Session(101)
        self.sess.lang = "ru"
        setattr(self.sess, field, state)
        self.entered, self.release = threading.Event(), threading.Event()
        self.effects = []
        self._store = NS(put=lambda sess: self.effects.append("persist"))

    def _get_session(self, cid):
        return self.sess

    def _dl_bytes(self, fid):
        self.entered.set()
        assert self.release.wait(3), "Test did not release fake download"
        return b"media"

    def _send_text(self, *args, **kwargs):
        self.effects.append("send")

    def _run_busy(self, *args):
        self.effects.append("job")

    def _unexpected_directory(self, cid):
        self.effects.append("file")
        raise AssertionError("Abandoned download reached filesystem work")

    _clone_dir = _book_dir = _continue_dir = _restyle_dir = _unexpected_directory

    def _anim_voice_save(self, *args):
        self.effects.append("voice")

    def _cover_have_song(self, *args):
        self.effects.append("song")


def main():
    failures = []
    for field, state, handler, msg in (
        ("anim_voice_state", "collect", "_anim_voice_take_media", {"voice": {"file_id": "sample"}}),
        ("book_state", "want_voice", "_book_take_media", {"voice": {"file_id": "sample"}}),
        ("book_state", "want_book", "_book_take_media", {"document": {"file_id": "book", "file_name": "book.txt"}}),
        ("clone_state", "want_audio", "_clone_take_media", {"voice": {"file_id": "sample"}}),
        ("restyle_state", "want_video", "_restyle_take_media", {"video": {"file_id": "clip"}}),
        ("continue_state", "want_video", "_continue_take_media", {"video": {"file_id": "clip"}}),
        ("continue_state", "want_text", "_continue_take_media", {"photo": [{"file_id": "person"}]}),
        ("continue_state", "want_text", "_continue_take_media", {"video": {"file_id": "person"}}),
        ("cover_state", "want_audio", "_cover_take_media", {"audio": {"file_id": "song"}}),
    ):
        for clear in (False, True, "rearm"):
            bot = Media(field, state)
            errors = []
            def run():
                try:
                    assert getattr(bot, handler)(101, bot.sess, "ru", msg) is True
                except Exception as exc:
                    errors.append(str(exc))
            worker = threading.Thread(target=run)
            worker.start()
            assert bot.entered.wait(3), handler
            if clear is True:
                bot.sess.context_generation += 1
            else:
                setattr(bot.sess, field, "")
                if clear == "rearm":
                    setattr(bot.sess, field, state)
            bot.release.set()
            worker.join(3)
            assert not worker.is_alive()
            if bot.effects or errors:
                failures.append((handler, state, clear, bot.effects, errors))
    assert not failures, failures
    with tempfile.TemporaryDirectory(prefix="armed-media-stale-") as work:
        for kind, field, state in (("clone", "clone_state", "want_audio"),
                                   ("assistant", "clone_state", "want_assistant"),
                                   ("book", "book_state", "want_voice"),
                                   ("read", "book_state", "want_book"),
                                   ("anim", "anim_voice_state", "collect"),
                                   ("frame", "continue_state", "want_video"),
                                   ("person_frame", "continue_state", "want_text")):
            for clear in (False, True, "rearm"):
                bot = Media(field, state)
                bot.sess.anim_voice_slot = 0
                bot.sess.anim_voices = ["saved-voice", "", ""]
                bot.sess.assistant_ref = "saved-assistant"
                bot.sess.book_ref = os.path.join(work, "ref.wav")
                bot.sess.book_ref_text = "saved narrator"
                with open(bot.sess.book_ref, "wb") as stream:
                    stream.write(b"reference")
                bot._clone_dir = bot._book_dir = bot._anim_voice_dir = bot._continue_dir = lambda cid: work
                bot._get_ctx = lambda: None
                bot._dl_bytes = lambda fid: b"media"
                def slow(*args, **kwargs):
                    bot.entered.set()
                    assert bot.release.wait(3)
                    return ("new-reference", "new transcript")
                bot.voice_clone = NS(prepare_reference=slow, CloneError=type("CloneError", (Exception,), {}))
                bot.modules = {"cover": NS(to_wav=slow), "audiobook": NS(read_book=slow, split_chapters=lambda value: [("chapter", "text")])}
                bot.deps = NS(seed_frame=slow, cut_tail=lambda *a: True, face_frame=slow)
                bot._anim_voice_added = lambda *a, **k: bot.effects.append("voice-ready")
                def run_completion():
                    try:
                        expected = (0, state, getattr(bot.sess, "flow_revision", lambda _: 0)(field))
                        if kind in ("clone", "assistant"):
                            bot._clone_prepare(101, "ru", "sample", kind == "assistant", expected)
                        elif kind == "book":
                            bot._book_prepare(101, "ru", "sample", expected)
                        elif kind == "read":
                            path = os.path.join(work, "book.txt")
                            with open(path, "w") as stream:
                                stream.write("book")
                            bot._book_open(101, "ru", path, expected)
                        elif kind == "anim":
                            AnimVoicesMixin._anim_voice_save(bot, 101, bot.sess, "ru", b"audio", ".ogg")
                        else:
                            bot._continue_take_media(101, bot.sess, "ru", {"video": {"file_id": "clip"}})
                    except Exception as exc:
                        errors.append(str(exc))
                errors = []
                worker = threading.Thread(target=run_completion)
                worker.start()
                assert bot.entered.wait(3), kind
                if clear is True:
                    bot.sess.context_generation += 1
                else:
                    setattr(bot.sess, field, "")
                    if clear == "rearm":
                        setattr(bot.sess, field, state)
                bot.release.set()
                worker.join(3)
                assert not worker.is_alive(), kind
                assert not errors and not bot.effects, (kind, clear, errors, bot.effects)
                assert bot.sess.anim_voices == ["saved-voice", "", ""]
                assert bot.sess.assistant_ref == "saved-assistant"
                assert bot.sess.book_ref_text == "saved narrator"
        for field, state, handler, completion, msg in (
            ("clone_state", "want_audio", "_clone_take_media", "_clone_prepare", {"voice": {"file_id": "voice"}}),
            ("book_state", "want_voice", "_book_take_media", "_book_prepare", {"voice": {"file_id": "voice"}}),
            ("book_state", "want_book", "_book_take_media", "_book_open", {"document": {"file_id": "book", "file_name": "book.txt"}}),
        ):
            bot = Media(field, state)
            bot._clone_dir = bot._book_dir = lambda cid: work
            bot._dl_bytes = lambda fid: b"media"
            sources = []
            def prepare(*args):
                sources.append(args[2])
                bot.entered.set()
                assert bot.release.wait(3)
            setattr(bot, completion, prepare)
            errors = []
            def run_worker():
                try:
                    assert getattr(bot, handler)(101, bot.sess, "ru", dict(msg, _worker_media=True)) is True
                except Exception as exc:
                    errors.append(str(exc))
            worker = threading.Thread(target=run_worker)
            worker.start()
            assert bot.entered.wait(1), (handler, "must prepare inline in the existing worker", bot.effects)
            assert worker.is_alive() and "job" not in bot.effects, handler
            bot.release.set()
            worker.join(3)
            assert not worker.is_alive() and not errors, (handler, errors)
            assert getattr(bot, handler)(101, bot.sess, "ru", dict(msg, _worker_media=True)) is True
            assert len(sources) == 2 and sources[0] != sources[1], (handler, sources)
    print("PASS: abandoned media downloads have no side effects")
    return 0


if __name__ == "__main__":
    sys.exit(main())

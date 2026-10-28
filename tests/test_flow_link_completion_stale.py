"""A link/text background job belongs to the flow that accepted it."""
import os
import sys
import tempfile
import threading
from types import SimpleNamespace as NS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests")] + [os.path.join(ROOT, d) for d in ("agent", "bot", "core", "imaging")]
from test_armed_media_stale import Media


def main():
    failures = []
    with tempfile.TemporaryDirectory(prefix="flow-links-") as work:
        sample = os.path.join(work, "sample.wav")
        with open(sample, "wb") as stream:
            stream.write(b"sample")
        for field, state, handler in (("clone_state", "want_audio", "_clone_take_link"),
                                      ("clone_state", "want_assistant", "_clone_take_link"),
                                      ("book_state", "want_voice", "_book_take_link"),
                                      ("anim_voice_state", "collect", "_anim_voice_take_link"),
                                      ("anim_voice_state", "", "_voice_from_request_link"),
                                      ("cover_state", "want_audio", "_cover_take_link")):
            for action in ("clear", "back", "rearm"):
                bot = Media(field, state)
                bot._clone_dir = bot._book_dir = bot._anim_voice_dir = lambda cid: work
                bot._get_ctx = lambda: None
                errors, jobs = [], []
                def fetch(*args, **kwargs):
                    bot.entered.set()
                    assert bot.release.wait(3)
                    return sample
                bot.voice_clone = NS(fetch_sample=fetch, prepare_reference=lambda *a, **k: (sample, "sample voice"),
                                     voice_link_in=lambda *a: "https://youtu.be/abcdefghijk",
                                     CloneError=type("CloneError", (Exception,), {}))
                bot.reads = NS(yes_no=lambda *a, **k: False)
                bot.links = NS(fetch_video=lambda *a: {"data": fetch(), "seconds": 5})
                def background(cid, fn, *args):
                    def run():
                        try:
                            fn(*args)
                        except Exception as exc:
                            errors.append(str(exc))
                    worker = threading.Thread(target=run)
                    jobs.append(worker)
                    worker.start()
                bot._run_busy = background
                assert getattr(bot, handler)(101, bot.sess, "ru", "https://youtu.be/abcdefghijk") is True
                assert bot.entered.wait(3), handler
                bot.effects.clear()
                if action == "clear":
                    bot.sess.clear_context()
                else:
                    setattr(bot.sess, field, "")
                    if action == "rearm":
                        setattr(bot.sess, field, state)
                bot.release.set()
                for worker in jobs:
                    worker.join(3)
                    assert not worker.is_alive()
                if bot.effects or errors:
                    failures.append((handler, state, action, bot.effects, errors))
        for kind, field, state in (("clone_text", "clone_state", "want_text"),
                                   ("cover_text", "cover_state", "want_text"),
                                   ("book_text", "book_state", "want_book"),
                                   ("continue_text", "continue_state", "want_text"),
                                   ("restyle_text", "restyle_state", "want_text"),
                                   ("book_copy", "book_state", "want_book")):
            for action in ("clear", "back", "rearm"):
                bot = Media(field, state)
                bot._clone_dir = bot._book_dir = lambda cid: work
                bot._get_ctx = lambda: None
                bot._api_post = lambda *a: None
                bot._send_voice_from_wav = lambda *a, **k: bot.effects.append("voice")
                bot._send_audio = lambda *a: bot.effects.append("audio")
                bot.sess.clone_ref = bot.sess.book_ref = bot.sess.cover_src = sample
                bot.sess.restyle_src = sample
                bot.sess.clone_ref_text = bot.sess.book_ref_text = "saved voice"
                errors, jobs = [], []
                def slow(*args, **kwargs):
                    bot.entered.set()
                    assert bot.release.wait(3)
                    return "spoken words"
                bot.voice_clone = NS(polish=slow, speak=lambda *a: sample)
                bot.modules = {"remix": NS(resing_available=lambda: True, resing=slow),
                               "mashup_stems": NS(release_separator=lambda: None),
                               "audiobook": NS(read_book=lambda *a, **k: "book", split_chapters=lambda text: [("chapter", "body")]),
                               "shutil": NS(copyfile=slow)}
                def background(cid, fn, *args):
                    def run():
                        try:
                            if kind == "book_text":
                                slow()
                            fn(*args)
                        except Exception as exc:
                            errors.append(str(exc))
                    worker = threading.Thread(target=run)
                    jobs.append(worker)
                    worker.start()
                bot._run_busy = background
                bot._enqueue_item = lambda *a: bot.effects.append("enqueue")
                bot._run_cancellable = lambda *a, **k: bot.effects.append("narration-job")
                if kind == "clone_text":
                    assert bot._clone_take_text(101, bot.sess, "ru", "hello") is True
                elif kind == "cover_text":
                    assert bot._cover_take_text(101, bot.sess, "ru", "lyrics") is True
                elif kind == "book_text":
                    assert bot._book_take_text(101, bot.sess, "ru", "long book " * 200) is True
                elif kind in ("continue_text", "restyle_text"):
                    bot._store = NS(put=slow)
                    background(101, getattr(bot, "_" + kind.replace("_text", "_take_text")), 101, bot.sess, "ru", "new scene")
                else:
                    path = os.path.join(work, "readbook.txt")
                    with open(path, "w") as stream:
                        stream.write("book")
                    background(101, bot._book_open, 101, "ru", path)
                assert bot.entered.wait(3), kind
                bot.effects.clear()
                if action == "clear":
                    bot.sess.clear_context()
                else:
                    setattr(bot.sess, field, "")
                    if action == "rearm":
                        setattr(bot.sess, field, state)
                bot.release.set()
                for worker in jobs:
                    worker.join(3)
                    assert not worker.is_alive()
                if bot.effects or errors:
                    failures.append((kind, action, bot.effects, errors))
    assert not failures, failures
    print("PASS links cannot complete into a newer flow")
    return 0


if __name__ == "__main__":
    sys.exit(main())

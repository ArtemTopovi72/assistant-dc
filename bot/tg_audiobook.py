"""📚 Audiobook (Creativity > Music): a voice sample, then a book file; each chapter comes back as a
voice message that opens with its title («Глава 3. …»).

States on the session (book_state), same shape as 🎙 Clone voice:
  "want_voice"  the button was pressed; the next voice / audio / video is the narrator's sample.
  "want_book"   the voice is ready (sess.book_ref); the next book file (txt, fb2, fb2.zip, epub, pdf, docx)
                -- or a long pasted text -- is read aloud.
While armed, the sample and the book are CONSUMED here: nothing reaches the document library (RAG), the
sandbox, or the agent, and a book that is not sent through this button is never read aloud. The state
is dropped by any other menu press, like every other flow.
"""
import os
import shutil
import uuid

import audiobook
import voice_clone

MIN_PASTED_BOOK = 1500                   # a shorter pasted text is not a book: the user is told to send the file
SPEECH_CHARS_PER_SECOND = 14.0           # measured on the F5 Russian voice, for the «about N min» line


class AudiobookMixin:
    def _book_dir(self, chat_id: int) -> str:
        d = os.path.join(os.path.dirname(str(tg_bot._MASHUP_DIR)), "audiobook", str(chat_id))
        os.makedirs(d, exist_ok=True)
        return d

    def _start_book_flow(self, chat_id: int, sess, lang: str) -> None:
        sess.book_state = "want_voice"
        self._store.put(sess)
        self._send_text(chat_id, tg_bot._t("book_ask_voice", lang), parse_mode="HTML")

    def _book_disarm(self, sess) -> None:
        if getattr(sess, "book_state", ""):
            sess.book_state = ""
            self._store.put(sess)

    @staticmethod
    def _book_doc_of(msg: dict):
        """(file_id, name, size) of a document that is a book by its name, else None."""
        doc = msg.get("document") or {}
        name = (doc.get("file_name") or "").lower()
        if doc.get("file_id") and (name.endswith(".fb2.zip") or os.path.splitext(name)[1] in audiobook.BOOK_EXTS):
            return doc["file_id"], doc.get("file_name") or "book", int(doc.get("file_size") or 0)
        return None

    def _book_take_media(self, chat_id: int, sess, lang: str, msg: dict) -> bool:
        """Consume the sample / the book while armed. True when it was ours."""
        state = getattr(sess, "book_state", "")
        if state == "want_voice":
            media = self._clone_media_of(msg)
            if not media:
                if msg.get("document") or msg.get("photo"):      # armed: never the library or the sandbox
                    self._send_text(chat_id, tg_bot._t("book_ask_voice", lang), parse_mode="HTML")
                    return True
                return False
            data = self._dl_bytes(media[0])
            if not data:
                self._send_text(chat_id, tg_bot._t("clone_fail_no_audio", lang))
                return True
            src = os.path.join(self._book_dir(chat_id), "sample" + media[1])
            with open(src, "wb") as fh:
                fh.write(data)
            self._send_text(chat_id, tg_bot._t("clone_working", lang))
            self._run_busy(chat_id, self._book_prepare, chat_id, lang, src)
            return True
        if state == "want_book":
            book = self._book_doc_of(msg)
            if not book:
                if msg.get("document") or msg.get("audio") or msg.get("voice"):
                    self._send_text(chat_id, tg_bot._t("book_wrong_file", lang))
                    return True
                return False
            fid, name, size = book
            from tg_resolve import _bot_file_limit      # 20 MB on the cloud API, 2 GB on our local server
            if size > _bot_file_limit():
                self._send_text(chat_id, tg_bot._t("book_too_big", lang, mb=f"{size / (1024 * 1024):.0f}",
                                                   limit=f"{_bot_file_limit() // (1024 * 1024)}"))
                return True
            data = self._dl_bytes(fid)
            if not data:
                self._send_text(chat_id, tg_bot._t("book_fail_read", lang))
                return True
            path = os.path.join(self._book_dir(chat_id), "book" + (".fb2.zip" if name.lower().endswith(".fb2.zip")
                                                                  else os.path.splitext(name)[1].lower()))
            with open(path, "wb") as fh:
                fh.write(data)
            self._run_busy(chat_id, self._book_open, chat_id, lang, path)
            return True
        return False

    def _book_prepare(self, chat_id: int, lang: str, src: str) -> None:
        sess = self._get_session(chat_id)
        try:
            ref, text = voice_clone.prepare_reference(self._get_ctx(), src, self._book_dir(chat_id),
                                                      lang=(sess.lang or "ru"))
        except voice_clone.CloneError as exc:
            self._send_text(chat_id, tg_bot._t("clone_fail_" + str(exc), lang))
            return
        except Exception:
            tg_bot.logger.exception("[audiobook] preparing the voice failed for chat %s", chat_id)
            self._send_text(chat_id, tg_bot._t("clone_fail_no_audio", lang))
            return
        sess = self._get_session(chat_id)
        sess.book_ref, sess.book_ref_text, sess.book_state = ref, text, "want_book"
        self._store.put(sess)
        self._send_text(chat_id, tg_bot._t("book_ask_book", lang), parse_mode="HTML")

    def _book_take_text(self, chat_id: int, sess, lang: str, text: str) -> bool:
        """A pasted text while armed for the book: long enough -> the book; short -> a hint, never the agent."""
        if getattr(sess, "book_state", "") != "want_book" or not (text or "").strip():
            return False
        if len(text) < MIN_PASTED_BOOK:
            self._send_text(chat_id, tg_bot._t("book_ask_book", lang), parse_mode="HTML")
            return True
        path = os.path.join(self._book_dir(chat_id), "book.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        self._run_busy(chat_id, self._book_open, chat_id, lang, path)
        return True

    def _book_open(self, chat_id: int, lang: str, path: str) -> None:
        sess = self._get_session(chat_id)
        ref, ref_text = sess.book_ref, sess.book_ref_text
        if not ref or not os.path.exists(ref):
            sess.book_state = "want_voice"
            self._store.put(sess)
            self._send_text(chat_id, tg_bot._t("book_ask_voice", lang), parse_mode="HTML")
            return
        try:
            chapters = audiobook.split_chapters(audiobook.read_book(path))
        except Exception:
            tg_bot.logger.exception("[audiobook] reading the book failed for chat %s", chat_id)
            self._send_text(chat_id, tg_bot._t("book_fail_read", lang))
            return
        finally:
            if os.path.exists(path):
                os.unlink(path)                  # the book is not kept (nor indexed anywhere)
        if not chapters:
            self._send_text(chat_id, tg_bot._t("book_fail_read", lang))
            return
        sess.book_state = ""                     # disarmed before the slow part
        self._store.put(sess)
        minutes = max(1, int(sum(len(b) for _, b in chapters) / SPEECH_CHARS_PER_SECOND / 60))
        self._run_cancellable(chat_id, tg_bot._t("book_working", lang, n=len(chapters), m=minutes),
                              self._book_run, chat_id, lang, chapters, ref, ref_text, lang=lang)

    def _book_run(self, ctx, chat_id: int, lang: str, chapters: list, ref: str, ref_text: str) -> None:
        work = os.path.join(self._book_dir(chat_id), "run_" + uuid.uuid4().hex[:8])   # one per job: two books never share files
        os.makedirs(work, exist_ok=True)

        def synth(text):
            return voice_clone.speak(ctx, ref, ref_text, text, work)
        done = 0
        try:
            for n, (title, body) in enumerate(chapters, start=1):
                if ctx.is_cancelled():
                    return
                wav = os.path.join(work, f"ch{n}.wav")
                if not audiobook.render_chapter(synth, title, body, n, wav, cancelled=ctx.is_cancelled):
                    if ctx.is_cancelled():
                        return
                    self._send_text(chat_id, tg_bot._t("book_chapter_fail", lang, n=n))
                    continue
                parts = audiobook.split_long(wav)
                for k, part in enumerate(parts, start=1):
                    cap = audiobook.caption(title, n, len(chapters), k, len(parts))
                    if not self._send_voice_from_wav(chat_id, part, caption=cap):
                        self._send_text(chat_id, tg_bot._t("book_chapter_fail", lang, n=n))
                        break
                else:
                    done += 1
            self._send_text(chat_id, tg_bot._t("book_done", lang, n=done, total=len(chapters)))
        except Exception:
            tg_bot.logger.exception("[audiobook] failed for chat %s", chat_id)
            self._send_text(chat_id, tg_bot._t("book_fail_run", lang))
        finally:
            shutil.rmtree(work, ignore_errors=True)


import tg_bot  # noqa: E402  (cycle by design; attrs read at call time)

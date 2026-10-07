"""🎚 Remix (Creativity menu; it replaced the separate Cover and Mashup buttons).

A song goes in; then EITHER text (the song sings it in its own voice and melody) OR a second
song (a quick mashup, then the first song sings the second one's words on its own melody).

States on the session (cover_state):
  "want_audio"  the button was pressed; the next voice / audio / video / round
                video / audio-or-video file, or a video link, is the first song.
  "want_text"   the first song is saved; the next plain text is the new lyric, and
                the next clip or link is the second song. Any menu button disarms.
The slow parts (demucs, TTS, SoulX-SVC) run off the poll thread (media/remix.py).
"""
import os

import cover
import remix


class CoverMixin:
    def _cover_dir(self, chat_id: int) -> str:
        # Read at call time: redirect_data_dir() rebinds _MASHUP_DIR's parent in tests.
        d = os.path.join(os.path.dirname(str(tg_bot._MASHUP_DIR)), "cover", str(chat_id))
        os.makedirs(d, exist_ok=True)
        return d

    def _start_cover_flow(self, chat_id: int, sess, lang: str) -> None:
        if not remix.available():
            self._send_text(chat_id, tg_bot._t("cover_fail_off", lang))
            return
        sess.cover_state = "want_audio"
        sess.cover_src = ""
        self._store.put(sess)
        self._send_text(chat_id, tg_bot._t("cover_ask_audio", lang), parse_mode="HTML")

    def _cover_disarm(self, sess) -> None:
        if getattr(sess, "cover_state", ""):
            sess.cover_state = ""
            self._store.put(sess)

    def _cover_take_media(self, chat_id: int, sess, lang: str, msg: dict) -> bool:
        """Consume the song while the flow is armed. True when it was ours."""
        if getattr(sess, "cover_state", "") not in ("want_audio", "want_text"):
            return False
        media = self._clone_media_of(msg)      # same "anything with sound" rule
        if not media:
            return False
        # A whole song as a file (flac, wav) is often past the Bot API's 20 MB:
        # the download fails, and «could not get a sound track» named the wrong cause.
        from tg_resolve import _bot_file_limit
        size = max(int((msg.get(k) or {}).get("file_size") or 0)
                   for k in ("voice", "audio", "video_note", "video", "document"))
        if size > _bot_file_limit():
            self._send_text(chat_id, tg_bot._t("cover_fail_too_big", lang,
                                               mb=f"{size / (1024 * 1024):.0f}",
                                               limit=f"{_bot_file_limit() // (1024 * 1024)}"))
            return True
        data = self._dl_bytes(media[0])
        if not data:
            self._send_text(chat_id, tg_bot._t("cover_fail_no_audio", lang))
            return True
        self._cover_have_song(chat_id, sess, lang, data, media[1])
        return True

    def _cover_take_link(self, chat_id: int, sess, lang: str, text: str) -> bool:
        """A video link (YouTube, VK, TikTok...) while the cover waits for its
        song: the clip's sound is the song (owner 10-03: «если ссылку на ютуб
        дам»). While the words are awaited, only a bare link counts as a new
        song -- a lyric that quotes a link is still the lyric."""
        state = getattr(sess, "cover_state", "")
        if state not in ("want_audio", "want_text"):
            return False
        import tg_links
        url = tg_links.video_url(text)
        if not url or (state == "want_text" and text.strip() != url):
            return False
        self._send_text(chat_id, tg_bot._t("cover_link_fetch", lang))

        def job():
            vid = tg_links.fetch_video(url)
            if vid.get("too_long"):
                self._send_text(chat_id, tg_bot._t("cover_link_long", lang,
                                                   mins=max(1, int(vid.get("seconds", 0)) // 60),
                                                   limit=tg_links.VIDEO_MAX_SECONDS // 60))
                return
            if not vid.get("data"):
                self._send_text(chat_id, tg_bot._t("cover_link_fail", lang))
                return
            self._cover_have_song(chat_id, self._get_session(chat_id), lang, vid["data"], ".mp4")
        self._run_busy(chat_id, job)
        return True

    def _cover_have_song(self, chat_id: int, sess, lang: str, data: bytes, ext: str) -> None:
        """The first song arms the text step; a clip arriving at that step is the SECOND song and starts the render."""
        second = getattr(sess, "cover_state", "") == "want_text" and bool(getattr(sess, "cover_src", ""))
        src = os.path.join(self._cover_dir(chat_id), ("song2" if second else "song") + ext)
        with open(src, "wb") as fh:
            fh.write(data)
        if second:
            return self._cover_go(chat_id, sess, lang, "", src)
        sess.cover_state = "want_text"
        sess.cover_src = src
        self._store.put(sess)
        self._send_text(chat_id, tg_bot._t("cover_ask_text", lang), parse_mode="HTML")

    def _cover_take_text(self, chat_id: int, sess, lang: str, text: str) -> bool:
        if getattr(sess, "cover_state", "") != "want_text" or not (text or "").strip():
            return False
        return self._cover_go(chat_id, sess, lang, text.strip(), "")

    def _cover_go(self, chat_id, sess, lang, lyrics, song2) -> bool:
        src = getattr(sess, "cover_src", "")
        if not src or not os.path.exists(src):
            sess.cover_state = "want_audio"
            self._store.put(sess)
            self._send_text(chat_id, tg_bot._t("cover_ask_audio", lang), parse_mode="HTML")
            return True
        # Disarmed BEFORE the slow work: a message sent while waiting must not
        # start a second cover of the same song.
        sess.cover_state = ""
        self._store.put(sess)
        self._send_text(chat_id, tg_bot._t("cover_working", lang))
        self._run_busy(chat_id, self._cover_render, chat_id, lang, src, lyrics, song2)
        return True

    def _cover_render(self, chat_id: int, lang: str, src: str, lyrics: str, song2: str) -> None:
        ctx = self._get_ctx()
        try:
            if song2:
                self._mashup_first(chat_id, lang, ctx, src, song2)
            info = {}
            if song2 and remix.resing_available():
                # 10-07 «говновоз делать по 2 ссылкам»: the second song gives the words, the
                # first sings them on its own melody in its own singer's voice.
                self._send_text(chat_id, tg_bot._t("cover_two_songs_words", lang))
                out = remix.resing(ctx, src, remix.lyrics_of(ctx, song2), info=info)
            elif song2:
                out = remix.remix_voice(ctx, src, song2)
            elif remix.resing_available():
                out = remix.resing(ctx, src, lyrics, info=info)
            else:
                out = remix.remix_words(ctx, src, lyrics)
            if not self._send_audio(chat_id, out, tg_bot._t("cover_done", lang)):
                self._send_text(chat_id, tg_bot._t("cover_fail_render", lang))
            elif info.get("rvc") and not info.get("trained"):
                # The first cover of an artist went out in a zero-shot timbre; their own RVC
                # voice is trained now on the original's vocal and the cover sent again in it.
                self._send_text(chat_id, tg_bot._t("cover_rvc_training", lang))
                self._send_audio(chat_id, remix.resing_rvc(ctx, info), tg_bot._t("cover_rvc_done", lang))
        except cover.CoverFailed as exc:
            self._send_text(chat_id, tg_bot._t("cover_fail_" + str(exc), lang))
        except Exception:
            tg_bot.logger.exception("[remix] failed for chat %s", chat_id)
            self._send_text(chat_id, tg_bot._t("cover_fail_render", lang))
        finally:
            import mashup_stems
            mashup_stems.release_separator()      # Demucs holds VRAM the next render needs


    def _mashup_first(self, chat_id: int, lang: str, ctx, src: str, song2: str) -> None:
        """The classic mashup (song 1's vocal over song 2's backing, bar-aligned and in key)
        goes out first: it takes ~30 s, the voice swap after it several minutes."""
        try:
            import mashup_auto
            self._send_audio(chat_id, mashup_auto.make(ctx, src, song2), tg_bot._t("mashup_done", lang))
        except Exception:
            tg_bot.logger.exception("[mashup] failed for chat %s", chat_id)


import tg_bot  # noqa: E402  (cycle by design; attrs read at call time)

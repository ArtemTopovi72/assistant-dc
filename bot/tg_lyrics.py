"""✨ Улучшить текст / ✍️ Сочинить текст (🎵 Music menu).

✨ waits for a ready lyric, ✍️ for a theme. Either way the text goes through
lyrics_craft.polish -- rules for rhyme, rhythm and length, the model for sense,
revised until it passes or the rounds run out -- and comes back as TWO
messages: what was changed (or how it was written), then the lyric alone so it
copies in one tap. Under it: 🎵 Спеть (the song engine, words as they are) and
✨ Ещё (another pass).

Session: lyrics_state "" | "improve" | "write"; lyrics_last (the last result).
"""
import html


class LyricsMixin:
    def _start_lyrics_flow(self, chat_id: int, sess, lang: str, mode: str) -> None:
        sess.lyrics_state = mode
        self._store.put(sess)
        self._send_text(chat_id, tg_bot._t("lyr_ask_" + mode, lang), parse_mode="HTML")

    def _lyrics_disarm(self, sess) -> None:
        if getattr(sess, "lyrics_state", ""):
            sess.lyrics_state = ""
            self._store.put(sess)

    def _lyrics_take_text(self, chat_id: int, sess, lang: str, text: str) -> bool:
        mode = getattr(sess, "lyrics_state", "")
        text = (text or "").strip()
        if mode not in ("improve", "write") or not text:
            return False
        sess.lyrics_state = ""                  # one text per press
        self._store.put(sess)
        self._send_text(chat_id, tg_bot._t("lyr_working_" + mode, lang))
        self._run_busy(chat_id, self._lyrics_run, chat_id, lang, mode, text)
        return True

    def _lyrics_run(self, chat_id: int, lang: str, mode: str, text: str) -> None:
        import lyrics_craft as LC
        ctx = self._get_ctx()
        try:
            if mode == "write":
                lyric_lang = "Russian" if lang == "ru" or LC._lang_name(text) == "Russian" else "English"
                first = LC.draft(ctx, text, lyric_lang)
                if not first:
                    raise RuntimeError("empty draft")
                res = LC.polish(ctx, first)
                notes = []
            else:
                res = LC.polish(ctx, text)
                notes = LC.changes_note(ctx, res["original"], res["text"], lang)
        except Exception:
            tg_bot.logger.exception("[lyrics] %s failed for chat %s", mode, chat_id)
            self._send_text(chat_id, tg_bot._t("lyr_fail", lang))
            return
        sess = self._get_session(chat_id)
        sess.lyrics_last = res["text"][:6000]
        self._store.put(sess)
        head = tg_bot._t("lyr_head_" + mode, lang, score=f"{res['score']:.1f}", rounds=res["rounds"])
        body = []
        if mode == "improve" and res["text"] == res["original"]:
            body.append(tg_bot._t("lyr_already_good", lang))
        body += [html.escape(n) for n in notes]
        if res["left"]:
            body.append(tg_bot._t("lyr_left", lang) + "\n" + "\n".join(
                "• " + html.escape(p) for p in res["left"][:5]))
        self._send_text(chat_id, head + ("\n\n" + "\n".join(body) if body else ""), parse_mode="HTML")
        # The lyric alone, no markup: one tap copies it whole.
        self._send_text(chat_id, res["text"], parse_mode=None, keyboard={"inline_keyboard": [[
            {"text": tg_bot._t("lyr_sing_btn", lang), "callback_data": "lyr:sing"},
            {"text": tg_bot._t("lyr_again_btn", lang), "callback_data": "lyr:again"}]]})

    def _cb_lyrics(self, chat_id: int, data: str) -> None:
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        last = getattr(sess, "lyrics_last", "")
        if not last:
            self._send_text(chat_id, tg_bot._t("lyr_gone", lang))
            return
        if data == "lyr:sing":
            from tg_songs import LYRICS_MARK
            self._start_song_generation(chat_id, LYRICS_MARK + "\n" + last, lang)
        elif data == "lyr:again":
            self._send_text(chat_id, tg_bot._t("lyr_working_improve", lang))
            self._run_busy(chat_id, self._lyrics_run, chat_id, lang, "improve", last)


import tg_bot  # noqa: E402  (cycle by design; attrs read at call time)

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
        self._run_cancellable(chat_id, tg_bot._t("lyr_working_" + mode, lang),
                              self._lyrics_run, chat_id, lang, mode, text, lang=lang)
        return True

    def _lyrics_run(self, ctx, chat_id: int, lang: str, mode: str, text: str) -> None:
        import lyrics_craft as LC
        singer = (tg_bot._music_prefs(self._get_session(chat_id)) or {}).get("vocal", "")
        try:
            if mode == "write":
                lyric_lang = "Russian" if lang == "ru" or LC._lang_name(text) == "Russian" else "English"
                res = LC.write(ctx, text, lyric_lang, singer=singer)
                notes = []
            else:
                res = LC.polish(ctx, text, singer=singer)
                notes = LC.changes_note(ctx, res["original"], res["text"], lang)
        except Exception:
            if ctx.is_cancelled():
                return                          # ⛔: «Отменено» is already said
            tg_bot.logger.exception("[lyrics] %s failed for chat %s", mode, chat_id)
            self._send_text(chat_id, tg_bot._t("lyr_fail", lang))
            return
        if ctx.is_cancelled():
            return
        self._lyrics_show(chat_id, lang, mode, res, notes)

    def _lyrics_show(self, chat_id: int, lang: str, mode: str, res: dict, notes: list,
                     buttons: bool = True) -> None:
        """Two messages: what was changed (or how it was written), then the lyric
        alone. `buttons` False when a song is already being made from it."""
        import lyrics_craft as LC
        sess = self._get_session(chat_id)
        sess.lyrics_last = res["text"][:6000]
        self._store.put(sess)
        head = tg_bot._t("lyr_head_" + mode, lang, score=f"{res['score']:.1f}", rounds=res["rounds"])
        body = []
        left = [LC.describe(i, lang) for i in res.get("left_rules", [])] + list(res.get("left_sense", []))
        if mode == "improve" and res["text"] == res["original"]:
            # «Править нечего» above a list of what is left read as a lie (live 10-03)
            body.append(tg_bot._t("lyr_not_better" if left else "lyr_already_good", lang))
        body += [html.escape(n) for n in notes]
        if left:
            body.append(tg_bot._t("lyr_left", lang) + "\n" + "\n".join(
                "• " + html.escape(p) for p in left[:5]))
        self._send_text(chat_id, head + ("\n\n" + "\n".join(body) if body else ""), parse_mode="HTML")
        # The lyric alone, no markup: one tap copies it whole.
        lyric = res["text"]
        kb = {"inline_keyboard": [[
            {"text": tg_bot._t("lyr_sing_btn", lang), "callback_data": "lyr:sing"},
            {"text": tg_bot._t("lyr_again_btn", lang), "callback_data": "lyr:again"}]]} if buttons else None
        self._send_text(chat_id, lyric, parse_mode=None, keyboard=kb)

    def _song_polish_then_sing(self, ctx, chat_id: int, lang: str, draft: str) -> None:
        """🎵 Song with ready words + «✨ Доработать и спеть»: show what was
        polished, the lyric itself, then sing it (owner 10-03)."""
        import lyrics_craft as LC
        from tg_songs import LYRICS_MARK
        try:
            res = LC.polish(ctx, draft, singer=(tg_bot._music_prefs(self._get_session(chat_id)) or {}).get("vocal", ""))
            notes = LC.changes_note(ctx, res["original"], res["text"], lang)
        except Exception:
            tg_bot.logger.exception("[lyrics] polishing a song's words failed for chat %s", chat_id)
            res, notes = None, []
        if ctx.is_cancelled():
            return                              # ⛔: no polish shown, no song started
        if res:
            self._lyrics_show(chat_id, lang, "improve", res, notes, buttons=False)
            words = res["text"]
        else:
            words = draft                      # the polish failed: their own words are sung
        self._send_text(chat_id, tg_bot._t("song_polish_going", lang))
        self._start_song_generation(chat_id, LYRICS_MARK + "\n" + words, lang)

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
            self._run_cancellable(chat_id, tg_bot._t("lyr_working_improve", lang),
                                  self._lyrics_run, chat_id, lang, "improve", last, lang=lang)


import tg_bot  # noqa: E402  (cycle by design; attrs read at call time)

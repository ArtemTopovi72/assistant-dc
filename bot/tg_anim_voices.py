"""🎙 Voices for 🎬 Animate: up to three voice samples the clip's people speak with.

The presets carry a «🎙 Добавить свои голоса» button; it arms
collection: each voice note / audio / round video is one sample, the user may
send them one at a time, and after each the bot says «🎙 Голос N принят» with
«Ещё» / «Готово». Three samples, or «Готово», or «Нет» moves on to the presets.
The samples ride to generate_video as reference audio (<Audio 1..3>) via
ctx.anim_voices (set per task in tg_tasks, cleared after the clip).

Session: anim_voice_state "" | "collect", anim_voices [paths].
"""
import os


MAX = 3


class AnimVoicesMixin:
    def _anim_voice_dir(self, chat_id: int) -> str:
        d = os.path.join(os.path.dirname(str(tg_bot._MASHUP_DIR)), "anim_voices", str(chat_id))
        os.makedirs(d, exist_ok=True)
        return d

    def _send_animate_presets(self, chat_id: int, lang: str) -> None:
        import animate_presets
        rows = []
        for i in range(0, len(animate_presets.ANIMATE_PRESET_ORDER), 2):
            rows.append([{"text": animate_presets.preset_label(k, lang), "callback_data": f"animate_preset:{k}"}
                         for k in animate_presets.ANIMATE_PRESET_ORDER[i:i + 2]])
        rows.append([{"text": tg_bot._t("animate_custom_btn", lang), "callback_data": "animate_custom"}])
        rows.append([{"text": tg_bot._t("anv_btn", lang), "callback_data": "anv:yes"}])
        self._send_text(chat_id, tg_bot._t("animate_ask_preset", lang), keyboard={"inline_keyboard": rows})

    def _animate_ask_voices(self, chat_id: int, sess, lang: str) -> None:
        """Entry of every animate path: straight to the presets. Voice samples
        are an optional button there -- asking «Будут образцы голосов?» first
        annoyed a kid who just wanted it to wave (persona run 2026-09-27)."""
        sess.anim_voices, sess.anim_voice_state = [], ""
        self._store.put(sess)
        self._send_animate_presets(chat_id, lang)

    def _offer_voices(self, chat_id: int, sess, lang: str, request: str, n: int) -> None:
        """generate_video stopped before rendering people who talk: «🎙 Свои
        голоса» collects samples (voice notes or round videos, left to right),
        «▶️ Стандартные» goes on without them. Either way the same request runs again."""
        sess.voice_pending, sess.voice_choice = request, ""
        sess.anim_voices, sess.anim_voice_state = [], ""
        self._store.put(sess)
        self._send_text(chat_id, tg_bot._t("anv_offer", lang, n=n), keyboard={"inline_keyboard": [[
            {"text": tg_bot._t("anv_own", lang), "callback_data": "anv:yes"},
            {"text": tg_bot._t("anv_default", lang), "callback_data": "anv:default"}]]
            + self._vl_rows(sess, lang, "anv:lib") + self._vl_manage_row(lang)})

    def _voices_go(self, chat_id: int, sess, lang: str) -> None:
        """Collection over: rerun the waiting clip request, or show the animate presets."""
        sess.anim_voice_state = ""
        if sess.voice_pending:
            sess.voice_choice = "own" if sess.anim_voices else "default"
            self._store.put(sess)
            self._enqueue_item(chat_id, {"type": "text", "text": sess.voice_pending})
            return
        self._store.put(sess)
        self._send_animate_presets(chat_id, lang)

    def _cb_anim_voices(self, chat_id: int, data: str) -> None:
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        if data.startswith("anv:lib:"):          # a voice from the library is the next speaker
            from tg_voice_library import vl_get
            v = vl_get(sess, data.split(":", 2)[2])
            if v and os.path.exists(v["ref"]) and len(sess.anim_voices) < MAX:
                sess.anim_voices = list(sess.anim_voices) + [v["ref"]]
            return self._anim_voice_added(chat_id, sess, lang)
        if data == "anv:default":
            sess.anim_voices = []
            return self._voices_go(chat_id, sess, lang)
        if data == "anv:yes":
            sess.anim_voice_state = "collect"
            self._store.put(sess)
            rows = self._vl_rows(sess, lang, "anv:lib")
            if rows:
                rows += self._vl_manage_row(lang)
            self._send_text(chat_id, tg_bot._t("anv_send", lang, n=len(sess.anim_voices) + 1, max=MAX),
                            keyboard={"inline_keyboard": rows} if rows else None)
            return
        if data == "anv:more":
            self._send_text(chat_id, tg_bot._t("anv_send", lang, n=len(sess.anim_voices) + 1, max=MAX))
            return
        # anv:no / anv:done
        self._voices_go(chat_id, sess, lang)

    def _anim_voice_take_media(self, chat_id: int, sess, lang: str, msg: dict) -> bool:
        if getattr(sess, "anim_voice_state", "") != "collect":
            return False
        media = self._clone_media_of(msg)        # voice / audio / round video / video
        if not media:
            return False
        data = self._dl_bytes(media[0])
        if not data:
            self._send_text(chat_id, tg_bot._t("anv_fail", lang))
            return True
        self._anim_voice_save(chat_id, sess, lang, data, media[1])
        return True

    def _anim_voice_take_link(self, chat_id: int, sess, lang: str, text: str) -> bool:
        """A video link (YouTube, TikTok, VK...) while collecting voices: its sound is the next sample
        (owner 10-04). True when it was ours."""
        if getattr(sess, "anim_voice_state", "") != "collect":
            return False
        import tg_links
        url = tg_links.video_url(text)
        if not url:
            return False
        self._send_text(chat_id, tg_bot._t("clone_working", lang))

        def job():
            vid = tg_links.fetch_video(url)
            if not vid.get("data"):
                self._send_text(chat_id, tg_bot._t("anv_fail", lang))
                return
            s2 = self._get_session(chat_id)
            if len(s2.anim_voices) >= MAX:
                return
            self._anim_voice_save(chat_id, s2, lang, vid["data"], ".mp4")
        self._run_busy(chat_id, job)
        return True

    def _anim_voice_save(self, chat_id: int, sess, lang: str, data: bytes, ext: str) -> None:
        # A name of its own: voice1.wav was overwritten by the next clip's first
        # voice, so a voice saved in the library came back as someone else's.
        import uuid
        path = os.path.join(self._anim_voice_dir(chat_id), f"voice_{uuid.uuid4().hex[:10]}{ext}")
        with open(path, "wb") as fh:
            fh.write(data)
        try:                                     # a round video / voice note -> the sound alone
            import cover
            path = cover.to_wav(path, os.path.splitext(path)[0] + ".wav")
        except Exception:
            pass                                 # H3's loader still reads most containers
        sess.anim_voices = list(sess.anim_voices) + [path]
        from tg_voice_library import vl_add
        v = vl_add(sess, path)                   # remembered: the last 5 come back as buttons
        self._anim_voice_added(chat_id, sess, lang, new_id=v["id"])

    def _anim_voice_added(self, chat_id: int, sess, lang: str, new_id: str = "") -> None:
        """«Голос N принят» + «💾 Подписать и сохранить» for a voice just sent:
        named, it stays in the library for the next clip (owner 10-03)."""
        n = len(sess.anim_voices)
        save = ([[{"text": tg_bot._t("vl_save_btn", lang), "callback_data": f"vl:name:{new_id}"}]]
                if new_id else [])
        if n >= MAX:
            self._send_text(chat_id, tg_bot._t("anv_full", lang, n=n),
                            keyboard={"inline_keyboard": save} if save else None)
            self._voices_go(chat_id, sess, lang)
            return
        sess.anim_voice_state = "collect"
        self._store.put(sess)
        self._send_text(chat_id, tg_bot._t("anv_got", lang, n=n), keyboard={"inline_keyboard": [[
            {"text": tg_bot._t("anv_more", lang), "callback_data": "anv:more"},
            {"text": tg_bot._t("anv_done", lang), "callback_data": "anv:done"}]]
            + save + self._vl_rows(sess, lang, "anv:lib") + self._vl_manage_row(lang)})


import tg_bot  # noqa: E402  (cycle by design; attrs read at call time)

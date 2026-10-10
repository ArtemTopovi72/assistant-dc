"""🎙 Voices for 🎬 Animate: three voice slots, one per person left to right.

The slot card (owner 10-10: «голоса в видео хочется видеть как
конфигурируемые слоты») lists 1 / 2 / 3 with the voice in each or «стандартный»;
🎤 N arms collection into slot N (a voice note / audio / round video / link, or
a library voice), 🗑 N empties it, «Готово» moves on. The slots ride to
generate_video via ctx.anim_voices (set per task in tg_tasks, cleared after the
clip); an empty slot is "" and that person keeps the default voice.

Session: anim_voice_state "" | "collect", anim_voice_slot (0-based target),
anim_voices [path | ""] by slot.
"""
import html
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

    def _slots(self, sess) -> list:
        v = [p if p and os.path.exists(p) else "" for p in (sess.anim_voices or [])][:MAX]
        return v + [""] * (MAX - len(v))

    def _slot_label(self, sess, ref: str, lang: str) -> str:
        if not ref:
            return tg_bot._t("anv_slot_empty", lang)
        from tg_voice_library import vl_label
        v = next((x for x in (getattr(sess, "voices", None) or []) if x.get("ref") == ref), None)
        return vl_label(v, lang) if v else tg_bot._t("vl_recent", lang)

    def _voice_slots_card(self, chat_id: int, sess, lang: str, new_id: str = "") -> None:
        slots = self._slots(sess)
        lines = [tg_bot._t("anv_slots_title", lang)]
        rows = []
        for i, ref in enumerate(slots):
            lines.append(f"{i + 1}. " + ("🗣 " if ref else "▫️ ") + html.escape(self._slot_label(sess, ref, lang)))
            row = [{"text": f"🎤 {i + 1}", "callback_data": f"anv:slot:{i}"}]
            if ref:
                row.append({"text": f"🗑 {i + 1}", "callback_data": f"anv:clr:{i}"})
            rows.append(row)
        if new_id:
            rows.append([{"text": tg_bot._t("vl_save_btn", lang), "callback_data": f"vl:name:{new_id}"}])
        rows.append([{"text": tg_bot._t("anv_done", lang), "callback_data": "anv:done"}])
        rows += self._vl_manage_row(lang)
        self._send_text(chat_id, "\n".join(lines), parse_mode="HTML", keyboard={"inline_keyboard": rows})

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
            {"text": tg_bot._t("anv_default", lang), "callback_data": "anv:default"}]]})

    def _voices_go(self, chat_id: int, sess, lang: str) -> None:
        """Collection over: rerun the waiting clip request, or show the animate presets."""
        sess.anim_voice_state = ""
        if sess.voice_pending:
            sess.voice_choice = "own" if any(self._slots(sess)) else "default"
            self._store.put(sess)
            self._enqueue_item(chat_id, {"type": "text", "text": sess.voice_pending})
            return
        self._store.put(sess)
        self._send_animate_presets(chat_id, lang)

    def _cb_anim_voices(self, chat_id: int, data: str) -> None:
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        if data.startswith("anv:lib:"):          # a library voice into the armed slot
            from tg_voice_library import vl_get
            v = vl_get(sess, data.split(":", 2)[2])
            if v and os.path.exists(v["ref"]):
                self._put_slot(sess, v["ref"])
            return self._anim_voice_added(chat_id, sess, lang)
        if data == "anv:default":
            sess.anim_voices = []
            return self._voices_go(chat_id, sess, lang)
        if data in ("anv:yes", "anv:more"):
            sess.anim_voice_state = ""
            self._store.put(sess)
            return self._voice_slots_card(chat_id, sess, lang)
        if data.startswith(("anv:slot:", "anv:clr:")):
            try:
                i = int(data.rsplit(":", 1)[1])
            except ValueError:
                return
            if not 0 <= i < MAX:
                return
            if data.startswith("anv:clr:"):
                slots = self._slots(sess)
                slots[i] = ""
                sess.anim_voices, sess.anim_voice_state = slots, ""
                self._store.put(sess)
                return self._voice_slots_card(chat_id, sess, lang)
            sess.anim_voice_slot, sess.anim_voice_state = i, "collect"
            self._store.put(sess)
            rows = self._vl_rows(sess, lang, "anv:lib")
            self._send_text(chat_id, tg_bot._t("anv_send_slot", lang, n=i + 1),
                            keyboard={"inline_keyboard": rows} if rows else None)
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
        self._put_slot(sess, path)
        from tg_voice_library import vl_add
        v = vl_add(sess, path)                   # remembered: the last 5 come back as buttons
        self._anim_voice_added(chat_id, sess, lang, new_id=v["id"])

    def _put_slot(self, sess, ref: str) -> None:
        slots = self._slots(sess)
        i = getattr(sess, "anim_voice_slot", 0)
        if not 0 <= i < MAX:
            i = slots.index("") if "" in slots else MAX - 1
        slots[i] = ref
        sess.anim_voices = slots

    def _anim_voice_added(self, chat_id: int, sess, lang: str, new_id: str = "") -> None:
        """The slot card again, with «💾 Подписать и сохранить» for a voice just
        sent: named, it stays in the library for the next clip (owner 10-03)."""
        sess.anim_voice_state = ""
        self._store.put(sess)
        self._voice_slots_card(chat_id, sess, lang, new_id=new_id)

import tg_bot  # noqa: E402  (cycle by design; attrs read at call time)

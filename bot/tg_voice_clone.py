"""🎙 Clone voice (Creativity menu): clip with speech in, any text out in that voice.

States on the session (clone_state):
  "want_audio"  the button was pressed; the next voice / audio / video /
                round video / audio-or-video file is the voice sample.
  "want_text"   a reference is ready; EVERY text message is spoken in that
                voice until the user presses another button (a second clip
                replaces the voice). Leaving the menu disarms it, the same rule
                that stops a half-collected mashup from eating the next voice.
The slow parts (ffmpeg + GigaAM, then F5) run off the poll thread.
"""
import os
import threading

import voice_clone


class VoiceCloneMixin:
    def _clone_dir(self, chat_id: int) -> str:
        # Read at call time: redirect_data_dir() rebinds _MASHUP_DIR's parent in tests.
        d = os.path.join(os.path.dirname(str(tg_bot._MASHUP_DIR)), "voice_clone", str(chat_id))
        os.makedirs(d, exist_ok=True)
        return d

    def _start_clone_flow(self, chat_id: int, sess, lang: str) -> None:
        sess.clone_state = "want_audio"
        self._store.put(sess)
        rows = self._vl_rows(sess, lang, "vl:clone")        # or a voice already in the library
        self._send_text(chat_id, tg_bot._t("clone_ask_audio", lang), parse_mode="HTML",
                        keyboard={"inline_keyboard": rows} if rows else None)

    def _clone_disarm(self, sess) -> None:
        if getattr(sess, "clone_state", ""):
            sess.clone_state = ""
            self._store.put(sess)

    @staticmethod
    def _clone_media_of(msg: dict):
        """(file_id, suffix) of anything with a sound track, else None."""
        for key, suf in (("voice", ".ogg"), ("audio", ".mp3"), ("video_note", ".mp4"), ("video", ".mp4")):
            m = msg.get(key)
            if m and m.get("file_id"):
                return m["file_id"], suf
        doc = msg.get("document") or {}
        mime = str(doc.get("mime_type") or "")
        if doc.get("file_id") and (mime.startswith("audio/") or mime.startswith("video/")):
            return doc["file_id"], os.path.splitext(doc.get("file_name") or "")[1] or ".bin"
        return None

    def _clone_take_media(self, chat_id: int, sess, lang: str, msg: dict) -> bool:
        """Consume a sample while the flow is armed. True when it was ours."""
        state = getattr(sess, "clone_state", "")
        if state not in ("want_audio", "want_text", "want_assistant"):
            return False
        media = self._clone_media_of(msg)
        if not media:
            return False
        data = self._dl_bytes(media[0])
        if not data:
            self._send_text(chat_id, tg_bot._t("clone_fail_no_audio", lang))
            return True
        for_assistant = state == "want_assistant"
        src = os.path.join(self._clone_dir(chat_id), ("assistant_sample" if for_assistant else "sample") + media[1])
        with open(src, "wb") as fh:
            fh.write(data)
        self._send_text(chat_id, tg_bot._t("clone_working", lang))
        self._run_busy(chat_id, self._clone_prepare, chat_id, lang, src, for_assistant)
        return True

    def _clone_take_link(self, chat_id: int, sess, lang: str, text: str) -> bool:
        """A video link (YouTube, TikTok, VK...) while the flow is armed: its
        sound track is the sample. True when it was ours."""
        state = getattr(sess, "clone_state", "")
        if state not in ("want_audio", "want_text", "want_assistant"):
            return False
        import tg_links
        url = tg_links.video_url(text)
        if not url:
            return False
        for_assistant = state == "want_assistant"
        self._send_text(chat_id, tg_bot._t("clone_working", lang))

        def job():
            vid = tg_links.fetch_video(url)
            if not vid.get("data"):
                self._send_text(chat_id, tg_bot._t("clone_fail_no_audio", lang))
                return
            src = os.path.join(self._clone_dir(chat_id),
                               ("assistant_sample" if for_assistant else "sample") + ".mp4")
            with open(src, "wb") as fh:
                fh.write(vid["data"])
            self._clone_prepare(chat_id, lang, src, for_assistant)
        self._run_busy(chat_id, job)
        return True

    def _clone_prepare(self, chat_id: int, lang: str, src: str, for_assistant: bool = False) -> None:
        """for_assistant: the sample is 🗣 the assistant's own voice (its own folder,
        so a later 🎙 clone never overwrites it), not a one-off 🎙 clone."""
        sess = self._get_session(chat_id)
        ctx = self._get_ctx()
        out_dir = self._clone_dir(chat_id)
        if for_assistant:
            out_dir = os.path.join(out_dir, "assistant")
            os.makedirs(out_dir, exist_ok=True)
        try:
            ref, text = voice_clone.prepare_reference(ctx, src, out_dir,
                                                      lang=(sess.lang or "ru"))
        except voice_clone.CloneError as exc:
            self._send_text(chat_id, tg_bot._t("clone_fail_" + str(exc), lang))
            return
        except Exception:
            tg_bot.logger.exception("[clone] preparing the sample failed for chat %s", chat_id)
            self._send_text(chat_id, tg_bot._t("clone_fail_no_audio", lang))
            return
        sess = self._get_session(chat_id)
        from tg_voice_library import vl_add
        v = vl_add(sess, ref, text)
        save_row = [{"text": tg_bot._t("vl_save_btn", lang), "callback_data": f"vl:name:{v['id']}"}]
        if for_assistant:
            sess.assistant_ref, sess.assistant_ref_text = ref, text
            sess.clone_state = ""
            self._store.put(sess)
            rows = [save_row, [{"text": tg_bot._t("my_voice_reset_btn", lang), "callback_data": "myv:reset"}]]
            msg = tg_bot._t("my_voice_ready", lang)
            if not sess.voice_on:                # changed, but voice replies are off: say so, offer the switch
                msg += "\n\n" + tg_bot._t("my_voice_ready_off", lang)
                rows.insert(0, [{"text": tg_bot._t("my_voice_enable_btn", lang), "callback_data": "myv:on"}])
            self._send_text(chat_id, msg, parse_mode="HTML", keyboard={"inline_keyboard": rows})
            return
        sess.clone_ref = ref
        sess.clone_ref_text = text
        sess.clone_state = "want_text"
        self._store.put(sess)
        self._send_text(chat_id, tg_bot._t("clone_ready", lang), parse_mode="HTML",
                        keyboard={"inline_keyboard": [save_row]})

    def _clone_take_text(self, chat_id: int, sess, lang: str, text: str) -> bool:
        if getattr(sess, "clone_state", "") != "want_text" or not (text or "").strip():
            return False
        if not sess.clone_ref or not os.path.exists(sess.clone_ref):
            sess.clone_state = "want_audio"
            self._store.put(sess)
            self._send_text(chat_id, tg_bot._t("clone_ask_audio", lang), parse_mode="HTML")
            return True
        self._run_busy(chat_id, self._clone_speak,
                       chat_id, lang, sess.clone_ref, sess.clone_ref_text, text)
        return True

    def _clone_speak(self, chat_id: int, lang: str, ref: str, ref_text: str, text: str) -> None:
        ctx = self._get_ctx()
        self._api_post("sendChatAction", {"chat_id": chat_id, "action": "record_voice"})
        wav = None
        try:
            clean = voice_clone.polish(ctx, text)
            wav = voice_clone.speak(ctx, ref, ref_text, clean, self._clone_dir(chat_id))
            if not wav or not self._send_voice_from_wav(chat_id, wav):
                self._send_text(chat_id, tg_bot._t("clone_fail_synth", lang))
        except Exception:
            tg_bot.logger.exception("[clone] synthesis failed for chat %s", chat_id)
            self._send_text(chat_id, tg_bot._t("clone_fail_synth", lang))
        finally:
            if wav and os.path.exists(wav):
                try:
                    os.unlink(wav)              # delivered; ours to drop
                except OSError:
                    pass

    def _cb_my_voice(self, chat_id: int, data: str) -> None:
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        if data == "myv:on":
            sess.voice_on = True
            self._store.put(sess)
            self._send_text(chat_id, tg_bot._t("voice_state", lang, state=tg_bot._t("on", lang)), parse_mode="HTML")
        elif data == "myv:reset":
            sess.assistant_ref = sess.assistant_ref_text = ""
            self._store.put(sess)
            self._send_text(chat_id, tg_bot._t("my_voice_off", lang), parse_mode="HTML")


import tg_bot  # noqa: E402  (cycle by design; attrs read at call time)

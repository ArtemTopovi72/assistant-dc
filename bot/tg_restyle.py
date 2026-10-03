"""🎨 Restyle video (Creativity menu): the user's clip keeps its motion, the look changes.

States on the session (restyle_state), same shape as 🎤 Cover:
  "want_video"  the button was pressed; the next video / round video / video file
                is the source clip.
  "want_text"   the clip is saved; the next plain text is the new look
                («аниме», «зима, снег», «пластилин»). A new clip replaces the
                source; any menu button disarms.
The render (video_control.restyle, ~15 min for 5 s) runs as a queued task, so it
gets the status line, ⛔ Stop and the admin console like every other long job.
"""
import os


PAYLOAD = "[restyle] "


class RestyleMixin:
    def _restyle_dir(self, chat_id: int) -> str:
        d = os.path.join(os.path.dirname(str(tg_bot._MASHUP_DIR)), "restyle", str(chat_id))
        os.makedirs(d, exist_ok=True)
        return d

    def _start_restyle_flow(self, chat_id: int, sess, lang: str) -> None:
        sess.restyle_state, sess.restyle_src = "want_video", ""
        self._store.put(sess)
        self._send_text(chat_id, tg_bot._t("restyle_ask_video", lang), parse_mode="HTML")

    def _restyle_disarm(self, sess) -> None:
        if getattr(sess, "restyle_state", ""):
            sess.restyle_state = ""
            self._store.put(sess)

    @staticmethod
    def _video_of(msg: dict):
        for key in ("video", "video_note", "animation"):
            if msg.get(key):
                return msg[key]["file_id"]
        doc = msg.get("document") or {}
        if (doc.get("mime_type") or "").startswith("video/"):
            return doc["file_id"]
        return None

    def _restyle_take_media(self, chat_id: int, sess, lang: str, msg: dict) -> bool:
        if getattr(sess, "restyle_state", "") not in ("want_video", "want_text"):
            return False
        fid = self._video_of(msg)
        if not fid:
            return False
        data = self._dl_bytes(fid)
        if not data:
            self._send_text(chat_id, tg_bot._t("restyle_fail_dl", lang))
            return True
        src = os.path.join(self._restyle_dir(chat_id), "source.mp4")
        with open(src, "wb") as fh:
            fh.write(data)
        sess.restyle_state, sess.restyle_src = "want_text", src
        self._store.put(sess)
        self._send_text(chat_id, tg_bot._t("restyle_ask_text", lang), parse_mode="HTML")
        return True

    def _restyle_take_text(self, chat_id: int, sess, lang: str, text: str) -> bool:
        if getattr(sess, "restyle_state", "") != "want_text" or not (text or "").strip():
            return False
        src = getattr(sess, "restyle_src", "")
        sess.restyle_state = ""          # disarmed before the slow part
        self._store.put(sess)
        if not src or not os.path.exists(src):
            self._start_restyle_flow(chat_id, sess, lang)
            return True
        self._enqueue_item(chat_id, {"type": "text", "text": PAYLOAD + text.strip()})
        return True

    def _run_restyle(self, chat_id: int, lang: str, prompt: str, ctx) -> bool:
        """The queued task body. True when a clip went out."""
        import video_control
        src = getattr(self._get_session(chat_id), "restyle_src", "")
        if not src or not os.path.exists(src):
            self._send_text(chat_id, tg_bot._t("restyle_fail_dl", lang))
            return False
        try:
            en = prompt
            try:
                import graph_language
                en = graph_language._translate_to_english(ctx, prompt) or prompt
            except Exception:
                pass
            out = video_control.restyle(ctx, src, en)
        except Exception:
            tg_bot.logger.exception("[restyle] failed for chat %s", chat_id)
            out = None
        if getattr(ctx, "is_cancelled", None) and ctx.is_cancelled():
            return False
        if not out or not self._send_video(chat_id, out, tg_bot._t("restyle_done", lang), ctx=ctx):
            self._send_text(chat_id, tg_bot._t("restyle_fail_render", lang))
            return False
        return True


import tg_bot  # noqa: E402  (cycle by design; attrs read at call time)

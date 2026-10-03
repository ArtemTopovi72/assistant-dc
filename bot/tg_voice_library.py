"""📚 The chat's voice library: one list for every voice the user gave the bot.

Every sample lands here -- 🗣 the assistant's voice, 🎙 a clone, 🎙 the voices
for a clip -- so nothing has to be recorded twice. Unnamed voices are the
RECENT ones: the last 2 are kept, older unnamed ones fall off the list (the
file stays on disk). A voice with a name is saved for good, until ✖️.

Where it is used:
  🗣 Голос ассистента   the list under the prompt: tap = the assistant speaks so
  🎙 voices for a clip  tap = the next speaker (left to right) gets that voice
  🎙 Клонировать голос  tap = speak the following texts in that voice

Session: voices [{id, name, ref, text}], voice_naming (id awaiting a name).
"""
import os
import uuid

RECENT = 2          # unnamed voices remembered
MAX = 12            # the whole library


def vl_add(sess, ref: str, text: str = "", name: str = "") -> dict:
    """Put a voice in the library (the same file twice is one entry, moved up)."""
    lib = [v for v in (getattr(sess, "voices", None) or []) if v.get("ref") != ref]
    old = next((v for v in (getattr(sess, "voices", None) or []) if v.get("ref") == ref), None)
    v = dict(old or {}, id=(old or {}).get("id") or uuid.uuid4().hex[:6], ref=ref,
             text=text or (old or {}).get("text", ""), name=name or (old or {}).get("name", ""))
    lib.insert(0, v)
    recent = [x for x in lib if not x.get("name")]
    drop = {x["id"] for x in recent[RECENT:]}
    sess.voices = [x for x in lib if x["id"] not in drop][:MAX]
    return v


def vl_get(sess, vid: str):
    return next((v for v in (getattr(sess, "voices", None) or []) if v.get("id") == vid), None)


def vl_list(sess) -> list:
    """Saved (named) first, then the recent ones; only voices whose file is there."""
    lib = [v for v in (getattr(sess, "voices", None) or []) if v.get("ref") and os.path.exists(v["ref"])]
    return [v for v in lib if v.get("name")] + [v for v in lib if not v.get("name")]


def vl_label(v: dict, lang: str) -> str:
    return v.get("name") or tg_bot._t("vl_recent", lang)


class VoiceLibraryMixin:
    def _vl_rows(self, sess, lang: str, action: str, mark: str = "") -> list:
        """One button per voice, two per row: callback_data = f"{action}:{id}"."""
        btns = [{"text": ("✓ " if mark and v["ref"] == mark else "🗣 ") + vl_label(v, lang)[:24],
                 "callback_data": f"{action}:{v['id']}"} for v in vl_list(sess)[:6]]
        return [btns[i:i + 2] for i in range(0, len(btns), 2)]

    def _vl_screen(self, chat_id: int, sess, lang: str) -> None:
        """🗣 under the «send a sample» prompt: the library to pick from and tidy."""
        rows = []
        for v in vl_list(sess):
            mine = v["ref"] == getattr(sess, "assistant_ref", "")
            rows.append([{"text": ("✓ " if mine else "🗣 ") + vl_label(v, lang)[:24],
                          "callback_data": f"vl:asst:{v['id']}"},
                         {"text": "✏️", "callback_data": f"vl:name:{v['id']}"},
                         {"text": "✖️", "callback_data": f"vl:drop:{v['id']}"}])
        if getattr(sess, "assistant_ref", ""):
            rows.append([{"text": tg_bot._t("my_voice_reset_btn", lang), "callback_data": "myv:reset"}])
        text = tg_bot._t("my_voice_ask", lang)
        if rows and vl_list(sess):
            text += "\n\n" + tg_bot._t("vl_help", lang)
        self._send_text(chat_id, text, parse_mode="HTML",
                        keyboard={"inline_keyboard": rows} if rows else None)

    def _vl_ask_name(self, chat_id: int, sess, lang: str, vid: str) -> None:
        sess.voice_naming = vid
        self._store.put(sess)
        self._send_text(chat_id, tg_bot._t("vl_name_ask", lang))

    def _vl_take_name(self, chat_id: int, sess, lang: str, text: str) -> bool:
        vid = getattr(sess, "voice_naming", "")
        name = " ".join((text or "").split())[:40]
        if not vid or not name:
            return False
        sess.voice_naming = ""
        v = vl_get(sess, vid)
        if v:
            vl_add(sess, v["ref"], v.get("text", ""), name)
            self._send_text(chat_id, tg_bot._t("vl_saved", lang, name=name))
        self._store.put(sess)
        return True

    def _cb_voice_library(self, chat_id: int, data: str) -> None:
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        _, act, vid = (data.split(":") + ["", ""])[:3]
        v = vl_get(sess, vid)
        if not v or not os.path.exists(v.get("ref", "")):
            self._send_text(chat_id, tg_bot._t("vl_gone", lang))
            return
        if act == "name":
            return self._vl_ask_name(chat_id, sess, lang, vid)
        if act == "drop":                    # off the list; the file itself stays
            sess.voices = [x for x in sess.voices if x["id"] != vid]
            if sess.assistant_ref == v["ref"]:
                sess.assistant_ref = sess.assistant_ref_text = ""
            self._store.put(sess)
            self._send_text(chat_id, tg_bot._t("vl_dropped", lang, name=vl_label(v, lang)))
            return
        if act in ("asst", "clone"):
            self._run_busy(chat_id, self._vl_use, chat_id, lang, vid, act)

    def _vl_use(self, chat_id: int, lang: str, vid: str, act: str) -> None:
        """A voice for speech needs its transcript (F5 reads the reference with
        its words); a clip sample has none until it is used like this."""
        sess = self._get_session(chat_id)
        v = vl_get(sess, vid)
        if not v:
            return
        if not v.get("text"):
            import voice_clone
            try:
                ref, text = voice_clone.prepare_reference(self._get_ctx(), v["ref"],
                                                          os.path.dirname(v["ref"]), lang=(sess.lang or "ru"))
            except Exception as exc:
                key = "clone_fail_" + str(exc) if isinstance(exc, voice_clone.CloneError) else "clone_fail_no_audio"
                self._send_text(chat_id, tg_bot._t(key, lang))
                return
            sess = self._get_session(chat_id)
            sess.voices = [dict(x, ref=ref, text=text) if x["id"] == vid else x for x in sess.voices]
            v = vl_get(sess, vid)
        if act == "asst":
            sess.assistant_ref, sess.assistant_ref_text = v["ref"], v["text"]
            sess.clone_state = ""
            self._store.put(sess)
            self._send_text(chat_id, tg_bot._t("vl_asst", lang, name=vl_label(v, lang)))
        else:
            sess.clone_ref, sess.clone_ref_text, sess.clone_state = v["ref"], v["text"], "want_text"
            self._store.put(sess)
            self._send_text(chat_id, tg_bot._t("clone_ready", lang), parse_mode="HTML")


import tg_bot  # noqa: E402  (cycle by design; attrs read at call time)

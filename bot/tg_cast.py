"""👥 Video with several characters: three cast slots, each a picture, what it does, and a voice.

The card lists 1 / 2 / 3; 🖼 N takes the next picture into slot N (an album fills the slots from
N on), ✏️ N takes the next text as what that character does or says, 🎤 N is the voice slot of
tg_anim_voices (the same samples, library and links). «Готово» sends ONE request: the filled
slots in order are Picture 1.. and the voices ride the clip as ctx.anim_voices.

Session: cast_state "" | "card" | "pic:N" | "act:N", cast_pics [path | ""], cast_acts [str]
by slot; the voices are sess.anim_voices.
"""
import html
import os
import time
import uuid

MAX = 3


class CastMixin:
    def _cast_slots(self, sess) -> tuple:
        pics = [p if p and os.path.exists(p) else "" for p in (sess.cast_pics or [])][:MAX]
        acts = [str(a or "") for a in (sess.cast_acts or [])][:MAX]
        return pics + [""] * (MAX - len(pics)), acts + [""] * (MAX - len(acts))

    def _cast_start(self, chat_id: int, sess, lang: str) -> None:
        sess.cast_state, sess.cast_pics, sess.cast_acts = "card", [], []
        sess.anim_voices, sess.anim_voice_state = [], ""
        self._store.put(sess)
        self._cast_card(chat_id, sess, lang)

    def _cast_card(self, chat_id: int, sess, lang: str) -> None:
        pics, acts = self._cast_slots(sess)
        voices = self._slots(sess)
        lines = [tg_bot._t("cast_title", lang)]
        rows = []
        for i in range(MAX):
            lines.append(f"{i + 1}. " + ("🖼 " if pics[i] else "▫️ ")
                         + (html.escape(acts[i][:60]) if acts[i] else tg_bot._t("cast_no_act", lang))
                         + " · " + ("🗣 " + html.escape(self._slot_label(sess, voices[i], lang))
                                    if voices[i] else tg_bot._t("anv_slot_empty", lang)))
            rows.append([{"text": f"🖼 {i + 1}", "callback_data": f"cast:pic:{i}"},
                         {"text": f"✏️ {i + 1}", "callback_data": f"cast:act:{i}"},
                         {"text": f"🎤 {i + 1}", "callback_data": f"cast:voice:{i}"}])
        rows.append([{"text": tg_bot._t("cast_go", lang), "callback_data": "cast:go"}])
        self._send_text(chat_id, "\n".join(lines), parse_mode="HTML", keyboard={"inline_keyboard": rows})

    def _cb_cast(self, chat_id: int, data: str) -> None:
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        _, what, *rest = data.split(":")
        if what == "go":
            return self._cast_go(chat_id, sess, lang)
        try:
            i = int(rest[0])
        except (IndexError, ValueError):
            return
        if not 0 <= i < MAX:
            return
        if what == "voice":
            sess.anim_voice_slot, sess.anim_voice_state = i, "collect"
            self._store.put(sess)
            rows = self._vl_rows(sess, lang, "anv:lib")
            return self._send_text(chat_id, tg_bot._t("anv_send_slot", lang, n=i + 1),
                                   keyboard={"inline_keyboard": rows} if rows else None)
        sess.anim_voice_state = ""
        sess.cast_state = f"{what}:{i}"
        self._store.put(sess)
        self._send_text(chat_id, tg_bot._t("cast_ask_" + what, lang, n=i + 1))

    def _cast_take_photos(self, chat_id: int, sess, lang: str, item: dict) -> None:
        """The picture(s) the 🖼 N press waited for: the album fills slots N, N+1.. in order."""
        ids = ([item["file_id"]] if item.get("file_id") else list(item.get("file_ids") or []))
        pics, _ = self._cast_slots(sess)
        i = int(sess.cast_state.split(":")[1])
        for fid in ids:
            data = self._dl_bytes(fid)
            if not data or i >= MAX:
                continue
            path = _save_image_bytes(data)
            tg_bot._log_image(sess, path, label=tg_bot._t("cast_pic_label", lang, n=i + 1), src="user")
            pics[i] = path
            i += 1
        if not any(pics):
            self._send_text(chat_id, tg_bot._t("img_gone", lang))
            return
        sess.cast_pics, sess.cast_state = pics, "card"
        self._store.put(sess)
        self._cast_card(chat_id, sess, lang)

    def _cast_take_text(self, chat_id: int, sess, lang: str, text: str) -> bool:
        """The text the ✏️ N press waited for. True when it was ours."""
        if not sess.cast_state.startswith("act:"):
            return False
        i = int(sess.cast_state.split(":")[1])
        _, acts = self._cast_slots(sess)
        acts[i] = text.strip()
        sess.cast_acts, sess.cast_state = acts, "card"
        self._store.put(sess)
        self._cast_card(chat_id, sess, lang)
        return True

    def _cast_go(self, chat_id: int, sess, lang: str) -> None:
        import video as video_mod
        pics, acts = self._cast_slots(sess)
        voices = self._slots(sess)
        used = [i for i in range(MAX) if pics[i]]
        if len(used) < 2:
            return self._send_text(chat_id, tg_bot._t("cast_need_two", lang))
        # the filled slots in order are Picture 1.. and each keeps its own voice
        request = video_mod.JOINT_MARK + " " + " ".join(
            f"Picture {n}: {acts[i] or 'appears in the scene'}." for n, i in enumerate(used, start=1))
        sess.anim_voices = [voices[i] for i in used]
        sess.voice_choice = "own" if any(sess.anim_voices) else "default"
        sess.cast_state = ""
        self._store.put(sess)
        first = tg_bot._image_by_path(sess, pics[used[0]]) or {}
        task = tg_bot._Task(
            task_id=str(uuid.uuid4()), chat_id=chat_id, user_text=request,
            image_path=pics[used[0]], image_id=first.get("id", ""), image_is_photo=True,
            album_paths=[pics[i] for i in used[1:]], enqueue_ts=time.time())
        if self._try_steer(chat_id, task):
            return
        with self._task_lock:
            self._pending_journal[task.task_id] = task
        self._backend.push(task)
        self._write_inflight()
        tg_bot.logger.info("Enqueued task %s chat=%s kind=cast depth=%d",
                           task.task_id[:8], chat_id, self._backend.depth())
        self._queue_note(chat_id, task, lang)


from tg_resolve import _save_image_bytes  # noqa: E402
import tg_bot  # noqa: E402  (cycle by design; attrs read at call time)

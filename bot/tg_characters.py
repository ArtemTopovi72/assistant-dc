"""Telegram: draw a trained character.

Creativity -> 🧑 Characters lists whoever the desktop app has both trained AND
ticked for Telegram, as INLINE buttons. Picking one arms `char_prompt`; the
next plain message is the scene, and the character's trigger is prepended for
the render.

Three decisions worth stating:

  * THE LIST IS BUILT AT SEND TIME from characters.telegram_characters(),
    which re-checks the adapter file on disk. A cached list is how the bot
    would keep offering someone whose .safetensors was deleted an hour ago.
  * THE CHOICE IS RE-VALIDATED WHEN THE PROMPT ARRIVES. The keyboard can sit
    on screen for days; between the tap and the prompt a character can be
    untick­ed or its adapter removed, and rendering anyway would silently draw
    a stranger under that name.
  * RENDERING RUNS OFF THE POLL THREAD, like songs and weather: a ComfyUI
    round trip is tens of seconds, and every other chat's updates queue behind
    this one.
"""
import html as _html_mod
import logging
import threading
import turn_trace

import characters as _chars
import image_lora as _lora
logger = logging.getLogger("assistant.tg_bot")   # same channel as tg_bot.py

# How long after the LAST photo of an album the render starts. Telegram
# delivers an album as one message per photo, the caption on the first only;
# rendering on the first would draw before the rest arrived.
_REF_QUIET_S = 2.5

# What the vision model is asked about a reference photo. Identity is
# explicitly OFF the list: the character's face comes from the LoRA, the
# reference lends everything else.
_REF_PROMPT = (
    "This photo is a STYLE AND POSE REFERENCE for a picture of a different "
    "person. Describe, as one compact prompt fragment of 40-80 words, what "
    "should be copied: the pose and gesture, the facial expression, clothing "
    "or its absence, props, camera framing and angle, lighting, colour "
    "palette, film/art style and mood. Do NOT describe who the person is, "
    "their face, age or identity, and do not name any film, actor or brand. "
    "Answer with the fragment only.")


class CharactersMixin:

    def _send_characters_menu(self, chat_id: int, lang: str) -> None:
        available = _chars.telegram_characters()
        rows = [[{"text": rec.get("name") or rec["slug"],
                  "callback_data": "char:" + rec["slug"]}]
                for rec in available]
        sess = self._get_session(chat_id)
        if sess.is_admin:
            # An operator-only escape hatch, not a character to draw with --
            # a distinct callback prefix so it can never collide with a slug.
            rows.append([{"text": tg_bot._t("characters_collect_btn", lang),
                         "callback_data": "lora_collect:start"}])
        if not rows:
            self._send_text(chat_id, tg_bot._t("char_none", lang))
            return
        self._send_text(chat_id, tg_bot._t("menu_characters_title", lang),
                        parse_mode="HTML",
                        keyboard={"inline_keyboard": rows})

    def _pick_character(self, chat_id: int, sess, lang: str, slug: str) -> None:
        rec = _chars.get(slug)
        if not rec or not rec.get("tg_available"):
            # Ticked off, or the adapter file is gone, since this keyboard was
            # drawn. Say so rather than arming a prompt that cannot render.
            self._send_text(chat_id, tg_bot._t("char_gone", lang))
            return
        sess.char_slug = slug
        sess.reg_state = "char_prompt"
        self._store.put(sess)
        self._send_text(
            chat_id,
            tg_bot._t("char_prompt", lang,
                      name=_html_mod.escape(rec.get("name") or slug)),
            parse_mode="HTML")

    def _collect_character_reference(self, chat_id: int, sess, lang: str,
                                     msg: dict) -> None:
        """A photo (or album page) sent while a character is armed.

        Each page is remembered on the session; a short quiet period after
        the last one starts ONE render with every reference and the caption
        as the scene. The pages of an album share media_group_id and only
        the first carries the caption, so the timer, not the caption, decides
        when the set is complete.
        """
        import tg_dispatch as _td
        largest = _td._largest_photo(msg.get("photo") or [])
        if not largest:
            return
        refs = list(getattr(sess, "char_ref_ids", []) or [])
        refs.append(largest["file_id"])
        sess.char_ref_ids = refs[-6:]
        cap = (msg.get("caption") or "").strip()
        if cap:
            sess.char_ref_caption = cap
        self._store.put(sess)
        gen = getattr(self, "_char_ref_gen", {})
        gen[chat_id] = gen.get(chat_id, 0) + 1
        self._char_ref_gen = gen
        my_gen = gen[chat_id]

        def _fire():
            import time as _t
            _t.sleep(_REF_QUIET_S)
            if self._char_ref_gen.get(chat_id) != my_gen:
                return                    # another page arrived; its timer fires
            s2 = self._get_session(chat_id)
            if s2.reg_state != "char_prompt":
                return
            ids = list(getattr(s2, "char_ref_ids", []) or [])
            scene = (getattr(s2, "char_ref_caption", "") or "").strip()
            s2.char_ref_ids = []
            s2.char_ref_caption = ""
            self._store.put(s2)
            if not ids:
                return
            self._activity.log(chat_id, "system",
                               f"[character] {len(ids)} reference photo(s), scene={scene[:60]!r}")
            self._start_character_render(chat_id, s2, lang, scene, ref_file_ids=ids)

        turn_trace.spawn(_fire, name=f"char-ref-{chat_id}")

    def _describe_character_references(self, ctx, file_ids) -> str:
        """Every reference photo, read by the vision model into one fragment."""
        import llm as _llm
        notes = []
        for fid in file_ids:
            try:
                data = self._dl_bytes(fid)
                if not data:
                    continue
                if ctx is not None:
                    ctx.set_stage("Looking at the image")
                out = _llm.analyze_image_with_llm(
                    ctx, image_bytes=data, user_text="Describe the reference.",
                    system_prompt=_REF_PROMPT, temperature=0.1) or ""
                out = " ".join(out.split())
                if out and "cannot" not in out.lower()[:40]:
                    notes.append(out)
            except Exception:
                logger.warning("reference photo read failed", exc_info=True)
        return " ".join(notes)

    def _start_character_render(self, chat_id: int, sess, lang: str,
                                text: str, ref_file_ids=None) -> None:
        rec = _chars.get(getattr(sess, "char_slug", "") or "")
        if not rec or not rec.get("tg_available"):
            sess.reg_state = ""
            sess.char_slug = ""
            self._store.put(sess)
            self._send_text(chat_id, tg_bot._t("char_gone", lang))
            return
        # A character render is not LLM-free: the Ideogram caption is PLANNED by
        # the model, so with no model loaded this would spend minutes and
        # deliver a stranger or nothing. Same refusal as an ordinary turn.
        try:
            _ctx = self._get_ctx()
        except Exception:
            _ctx = None
        # Absent attribute = "does not report a model" (proceed); an explicitly
        # empty one = started without a model on purpose (refuse). Same rule as
        # the queued-turn gate in tg_tasks.
        _mn = getattr(_ctx, "model_name", None) if _ctx is not None else ""
        if _ctx is None or (_mn is not None and not str(_mn).strip()):
            self._send_text(chat_id, tg_bot._t("no_model_loaded", lang))
            return
        # Registered as a real in-flight task BEFORE the thread starts, so the
        # status message can carry the same ⛔ button every other render has.
        # Without this the character flow was the one long operation in the bot
        # with no way out but ⛔ Stop, which also kills everything else queued.
        import uuid
        from tg_queue_backends import _Task
        task_id = "char_" + uuid.uuid4().hex[:12]
        cancel = threading.Event()
        task = _Task(task_id=task_id, chat_id=chat_id, user_text=text or "reference")
        with self._task_lock:
            self._running_task.setdefault(chat_id, []).append(task)
            self._task_cancels[task_id] = cancel
        status_id = self._send_get_id(
            chat_id,
            tg_bot._t("char_drawing", lang, name=rec.get("name") or rec["slug"]),
            keyboard={"inline_keyboard": [[
                {"text": tg_bot._t("cancel_btn", lang),
                 "callback_data": "cancel:" + task_id}]]})
        turn_trace.spawn(self._render_character, chat_id, rec, text, lang, task, cancel, status_id,
                         list(ref_file_ids or []), name=f"char-render-{chat_id}")

    def _render_character(self, chat_id, rec, text, lang,
                          task=None, cancel=None, status_id=None,
                          ref_file_ids=None) -> None:
        import os
        sess = self._get_session(chat_id)
        try:
            base = self._get_ctx()
        except Exception:
            base = None
        # A per-render view with THIS render's cancel event. On the raw shared
        # context the ⛔ button would either do nothing or take down whatever
        # else the bot happens to be doing, which is the collision _scoped_ctx
        # exists to prevent.
        ctx = tg_bot._scoped_ctx(base, cancel_event=cancel) if base is not None else None
        # The status line goes through the SAME pipeline as every queued task:
        # _make_stage_callback owns icon choice, translation, the desktop feed,
        # the activity log and the interruptible gate. This flow bypasses the
        # task queue, so without wiring the callback here it had no stages at
        # all -- a single static "Рисую…" that looked identical to a hang.
        kb = {"inline_keyboard": [[
            {"text": tg_bot._t("cancel_btn", lang),
             "callback_data": "cancel:" + task.task_id}]]} if task else None

        def _update_status(text: str) -> None:
            if not status_id:
                return
            try:
                self._edit_text(chat_id, status_id, text, parse_mode="HTML",
                                keyboard=kb)
            except Exception:
                pass

        if ctx is not None:
            ctx.stage_callback = self._make_stage_callback(
                chat_id, lang, str(chat_id), _update_status)

        try:
            try:
                import image_generate as G
                if ref_file_ids:
                    # The references lend pose, outfit, framing and style;
                    # the words (if any) say what else the scene is.
                    look = self._describe_character_references(ctx, ref_file_ids)
                    if look:
                        text = ((text + ". ") if text else "") + (
                            "Recreate this reference look with the character: " + look)
                    elif not text:
                        text = "a portrait of the character"
                    logger.info("character render with %d reference(s): %s",
                                len(ref_file_ids), text[:200])
                prompt = _lora.prompt_with_trigger(text, rec.get("trigger")
                                                   or rec["slug"])
                # The render itself emits no stage -- it is called directly here
                # rather than through the tool handler that normally does.
                if ctx is not None:
                    ctx.set_stage("Drawing a picture")
                path = G.generate_image_with_comfy(
                    ctx, prompt,
                    lora_name=os.path.basename(rec.get("lora_file") or ""),
                    lora_strength=float(rec.get("strength") or 1.5),
                    trigger=rec.get("trigger") or rec["slug"])
            except Exception:
                logger.exception("character render failed")
                path = None
        finally:
            # Deregister FIRST: a later press must find nothing running and say
            # "already finished" rather than pointing at a task that is gone.
            if task is not None:
                with self._task_lock:
                    lst = self._running_task.get(chat_id)
                    if lst is not None:
                        try: lst.remove(task)
                        except ValueError: pass
                        if not lst:
                            self._running_task.pop(chat_id, None)
                    self._task_cancels.pop(task.task_id, None)
            if status_id:
                # The keyboard is cleared as well as the text: a live ⛔ button
                # left under a finished render points at nothing.
                try:
                    self._edit_text(chat_id, status_id, "✅", parse_mode=None,
                                    keyboard={"inline_keyboard": []})
                except Exception: pass
                try: self._delete(chat_id, status_id)
                except Exception: pass
        # Cleared here rather than at arm time, so the flow is multi-shot the
        # way the draw menu is: one more scene for the same character needs
        # another tap, which is the honest cost of not guessing.
        sess.reg_state = ""
        self._store.put(sess)
        if cancel is not None and cancel.is_set():
            # _cancel_task returned "running" and deliberately did NOT confirm,
            # leaving the confirmation to whoever actually stops -- that is here.
            self._send_text(chat_id, tg_bot._t("cancel_done", lang))
            return
        if not path:
            # Why it failed matters here: while a training run holds the card
            # every render returns None, and "не смог нарисовать" invites the
            # user to try again immediately, for hours.
            busy = None
            try:
                import comfy_client as _cc
                busy = _cc.recent_gpu_refusal()
            except Exception:
                pass
            self._send_text(chat_id, tg_bot._t("gpu_busy" if busy else "char_failed", lang))
            return
        # Registered in the image log like every other delivered picture, so a
        # follow-up ("сделай её светлее", a reply to the photo) can resolve
        # WHICH image it is about. A picture the bot sends without logging is
        # invisible to that machinery. Live: no image-action keyboard and no
        # msg_id meant the render LOOKED like it never landed in the chat --
        # replying to it or tapping a button (neither of which existed) did
        # nothing, same delivery contract every other picture path already
        # honours (tg_tasks._deliver_salvaged_render, tg_callbacks._cb_*).
        img_id = tg_bot._log_image(sess, path, label=text[:80], src="bot")
        self._store.put(sess)
        self._last_photo_msg_id = 0
        ok = self._send_photo(chat_id, path, keyboard=tg_bot._image_kb(lang, img_id))
        if ok:
            tg_bot._log_image(sess, path,
                              msg_id=getattr(self, "_last_photo_msg_id", 0), src="bot")
            # This render runs off the task queue on a throwaway _scoped_ctx
            # copy, so nothing else persists it as "the current picture" the
            # way tg_tasks.py does after every queued task (tg_tasks.py:1222-3).
            # Without this, a plain-text follow-up right after a character
            # render ("what's drawn here?") wrongly asks the user to upload a
            # photo, because the next turn's ctx.last_image_path is seeded
            # from sess.last_image_path (tg_tasks.py:1006).
            sess.last_image_path = path
            sess.turn_image = img_id          # the picture in play for the next turn
            sess.last_image_prompt = prompt
            self._store.put(sess)
        else:
            tg_bot.logger.error("character render sendPhoto failed chat=%s path=%s",
                                 chat_id, path)
            self._send_text(chat_id, tg_bot._t("char_failed", lang))


# Imported at the BOTTOM and read as tg_bot.<name> at call time, never by
# value: tg_bot imports CharactersMixin from here, so a top-of-file import
# would be a back-edge above the class and the cycle would fail to resolve in
# one of the two import orders. Same convention as every other tg_* mixin.
import tg_bot  # noqa: E402

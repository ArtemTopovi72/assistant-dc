"""🎬 The plan of a multi-part clip, shown before minutes of rendering start.

A script longer than one clip is rendered in parts (video.split_script), each
several minutes on the card. generate_video stops before the first part and
hands the parts here: the chat sees every part with its length, the clip's
total and the render time, and can drop a part, condense the script, render or
cancel. ▶ reruns the same request with the approved parts riding
ctx.video_plan_parts (tg_tasks).

Session: video_plan [{text, sec}], video_plan_request, video_plan_ok [texts].
"""
import html
import math
import re


def _cut(text: str, n: int = 90) -> str:
    import video
    t = " ".join(video.strip_timing(text).split())
    return t if len(t) <= n else t[:n - 1].rstrip(",.;:— ") + "…"


def _mins(seconds: float) -> int:
    return max(1, math.ceil(seconds / 60.0))


def _lines_of(raw: str, n: int) -> list:
    """The last {"lines": [...]} in the reply that has `n` lines. The model answered one line,
    then «Wait… let me do all 6» and the whole list: the first JSON was taken and the plan
    showed the English prompts (live 2026-10-20)."""
    import json
    dec, found = json.JSONDecoder(), []
    for m in re.finditer(r'\{\s*"lines"', raw or ""):
        try:
            obj, _ = dec.raw_decode(raw, m.start())
        except ValueError:
            continue
        got = obj.get("lines") if isinstance(obj, dict) else None
        if isinstance(got, list) and len(got) == n:
            found.append(got)
    return found[-1] if found else []


def _summaries(texts: list, lang: str, owner=None) -> list:
    """One short line per part in the chat's language: the parts themselves are the
    English prompts the video model reads, which the chat should not see as is.
    Any failure: the prompts, cut short."""
    if not texts or lang == "en":
        return [""] * len(texts)
    import video
    from seams import text_model
    from utils import safe_json_from_llm
    name = {"ru": "Russian"}.get(lang, lang)
    listing = "\n".join(f"{i}. {video.strip_timing(t)}" for i, t in enumerate(texts, 1))
    lines, raw = [], ""
    # Two tries: a 6-part monologue came back unusable once and the chat saw the English
    # prompts (live 2026-10-20). The budget follows the text: every spoken line is copied
    # whole, and 90 tokens a part did not hold a Russian monologue.
    for _ in range(2):
        try:
            # «говорит по-русски» stood where the user's own words belonged, and a sound
            # line riding every part («пульс искр наполняет сцену») became a part of its own.
            raw = text_model(owner)(
                None,
                f"Retell each numbered video shot description in {name} as ONE short line "
                "(at most 14 words): who does what on screen. A line in double quotes is spoken "
                "aloud: copy it into your line in quotes, character for character, never "
                "paraphrased or replaced by 'says something'. Leave out camera, sound, music, "
                "light and particle effects (sparks, smoke, embers) unless they are the action. "
                f'Reply with JSON only: {{"lines": ["...", "..."]}}, exactly {len(texts)} lines, '
                "one per shot, same order.",
                listing, temperature=0.2, max_tokens=400 + 90 * len(texts) + len(listing) // 2) or ""
            lines = _lines_of(raw, len(texts)) or (safe_json_from_llm(raw, ["lines"]) or {}).get("lines") or []
        except Exception:
            tg_bot.logger.exception("video plan: summaries failed")
            lines = []
        # «…и лает ""»: told to copy spoken lines, the model left empty quotes where a dog barks.
        lines = [re.sub(r'\s*(?:""|«»|“”)', "", re.sub(r"^\d+[.)]\s*", "", str(x).strip())) for x in lines]
        if len(lines) == len(texts) and all(lines):
            return lines
    tg_bot.logger.warning("video plan: no %s summaries for %d parts (got %d); the prompts are shown: %r",
                          name, len(texts), len(lines), raw[-300:])
    return [""] * len(texts)


def _with_summaries(plan: list, lang: str, owner=None) -> list:
    for p, say in zip(plan, _summaries([p["text"] for p in plan], lang, owner)):
        p["say"] = say
    return plan


class VideoPlanMixin:
    def _video_plan_card(self, sess, lang: str) -> tuple:
        import video
        plan = sess.video_plan
        clip = sum(float(p.get("sec") or 0) for p in plan)
        render = sum(video.render_eta_s(float(p.get("sec") or 0)) for p in plan)
        lines = [tg_bot._t("vp_title", lang, n=len(plan), clip=f"{clip:.0f}", mins=_mins(render))]
        lines.append(tg_bot._t("vp_hint", lang))
        rows = [[{"text": tg_bot._t("vp_go", lang, mins=_mins(render)), "callback_data": "vp:go"}]]
        edits = [{"text": f"✏️ {i}", "callback_data": f"vp:edit:{i - 1}"} for i in range(1, len(plan) + 1)]
        rows += [edits[k:k + 5] for k in range(0, len(edits), 5)]
        rows.append([{"text": tg_bot._t("vp_add", lang), "callback_data": "vp:add"}])
        if len(plan) > 1:
            dels = [{"text": f"🗑 {i}", "callback_data": f"vp:del:{i - 1}"} for i in range(1, len(plan) + 1)]
            rows += [dels[k:k + 5] for k in range(0, len(dels), 5)]
            rows.append([{"text": tg_bot._t("vp_squeeze", lang), "callback_data": "vp:squeeze"}])
        rows.append([{"text": tg_bot._t("vp_cancel", lang), "callback_data": "vp:x"}])
        return "\n".join(lines), {"inline_keyboard": rows}

    def _video_plan_script(self, sess, lang: str) -> str:
        return "\n".join(tg_bot._t("vp_part", lang, i=i, sec=f"{float(p.get('sec') or 0):.1f}",
                                   text=html.escape(_cut(p.get("say") or p.get("text", ""))))
                         for i, p in enumerate(sess.video_plan, 1))

    def _send_video_plan(self, chat_id: int, sess, lang: str) -> None:
        """Two messages: the script alone, so it copies clean, then the card with the buttons."""
        self._send_text(chat_id, self._video_plan_script(sess, lang), parse_mode="HTML")
        text, kb = self._video_plan_card(sess, lang)
        self._send_text(chat_id, text, parse_mode="HTML", keyboard=kb)

    def _offer_video_plan(self, chat_id: int, sess, lang: str, request: str, plan: list) -> None:
        sess.video_plan = _with_summaries(
            [{"text": str(p.get("text", "")), "sec": float(p.get("sec") or 0)} for p in plan], lang, self)
        sess.video_plan_request, sess.video_plan_ok = request, []
        sess.video_plan_image = getattr(sess, "turn_image", "") or ""
        self._store.put(sess)
        self._send_video_plan(chat_id, sess, lang)

    def _squeeze_plan(self, chat_id: int, sess, lang: str) -> None:
        """The model rewrites the script tighter; the parts are re-cut from it."""
        import video
        script = " ".join(p["text"] for p in sess.video_plan)
        target = max(1, len(sess.video_plan) - 1)
        parts = []
        # A rewrite the split cuts into as many parts as before condensed nothing: the
        # button once answered 4 parts with 4 parts (live 2026-10-13). One stricter try.
        for fit in (target * video.VIDEO_PART_MAX_S, target * video.VIDEO_PART_MAX_S * 0.7):
            out = video.condense_script(script, sum(p["sec"] for p in sess.video_plan), fit,
                                        sentences=target, ctx=self.reads)
            parts = video.split_script(out) if out else []
            if parts and len(parts) < len(sess.video_plan):
                break
        if not parts or len(parts) >= len(sess.video_plan):
            self._send_text(chat_id, tg_bot._t("vp_squeeze_fail", lang))
            return
        sess.video_plan = _with_summaries([{"text": p, "sec": video.estimate_seconds(p)} for p in parts], lang, self)
        self._store.put(sess)
        self._send_video_plan(chat_id, sess, lang)

    def _edit_plan_part(self, chat_id: int, sess, lang: str, state: str, text: str) -> None:
        """The message an armed «vp_edit:<i>» / «vp_add» was waiting for. The user writes in
        their language; the part is the English script the video model reads, with every
        quoted spoken line left exactly as written."""
        import video
        from seams import text_model
        i = int(state.partition(":")[2]) if state.startswith("vp_edit:") else -1
        if not sess.video_plan or not sess.video_plan_request:
            self._send_text(chat_id, tg_bot._t("vp_gone", lang))
            return
        current = sess.video_plan[i]["text"] if 0 <= i < len(sess.video_plan) else ""
        try:
            out = text_model(self)(
                None,
                "You write one part of a video script for a video model. Input: CURRENT PART "
                "(empty when a part is being added) and the USER MESSAGE, in any language, "
                "usually Russian. If the message describes what should happen, that is the new "
                "part. If it is an instruction to change the current part (add, remove, make "
                "it different), apply it to the current part and keep the rest. Output the part "
                "in English: 1-4 short sentences, each one visible action, who does what. Every "
                "spoken line stays in double quotes, in its own language, character for character "
                "as the user wrote it -- never translated, shortened or fixed; a spoken line of "
                "the current part stays unless the user changes it. If the user says someone "
                "speaks but gives no words, do not invent any. No camera, sound or music "
                "directions unless asked. Reply with the part only.",
                f"CURRENT PART:\n{current}\n\nUSER MESSAGE:\n{text}",
                temperature=0.2, max_tokens=400) or ""
        except Exception:
            tg_bot.logger.exception("video plan: part rewrite failed")
            out = ""
        out = " ".join(out.replace("```", "").split()).strip()
        if not out:
            self._send_text(chat_id, tg_bot._t("vp_edit_fail", lang))
            return
        lost = video._lost_lines(text, out)
        if lost:
            out += " " + " ".join(f'He says: "{x}".' for x in lost)
        parts = video.split_script(out) or [out]
        new = [{"text": p, "sec": video.estimate_seconds(p)} for p in parts]
        new = _with_summaries(new, lang, self)
        plan = sess.video_plan
        sess.video_plan = (plan[:i] + new + plan[i + 1:]) if 0 <= i < len(plan) else plan + new
        self._store.put(sess)
        self._send_video_plan(chat_id, sess, lang)

    def _cb_video_plan(self, chat_id: int, data: str) -> None:
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        if not sess.video_plan or not sess.video_plan_request:
            self._send_text(chat_id, tg_bot._t("vp_gone", lang))
            return
        if data == "vp:x":
            sess.video_plan, sess.video_plan_request, sess.video_plan_ok = [], "", []
            sess.voice_choice, sess.anim_voices = "", []
            self._store.put(sess)
            self._send_text(chat_id, tg_bot._t("vp_cancelled", lang))
            return
        if data == "vp:go":
            sess.video_plan_ok = [p["text"] for p in sess.video_plan]
            self._store.put(sess)
            item = {"type": "text", "text": sess.video_plan_request}
            if getattr(sess, "video_plan_image", ""):
                item["image_id"] = sess.video_plan_image
            self._busy_note(chat_id, lang)
            self._enqueue_item(chat_id, item)
            return
        if data == "vp:squeeze":
            self._send_text(chat_id, tg_bot._t("vp_squeezing", lang))
            self._run_busy(chat_id, self._squeeze_plan, chat_id, sess, lang)
            return
        if data == "vp:add":
            sess.reg_state = "vp_add"
            self._store.put(sess)
            self._send_text(chat_id, tg_bot._t("vp_add_ask", lang))
            return
        if data.startswith("vp:edit:"):
            try:
                i = int(data.rsplit(":", 1)[1])
            except ValueError:
                return
            if 0 <= i < len(sess.video_plan):
                sess.reg_state = f"vp_edit:{i}"
                self._store.put(sess)
                p = sess.video_plan[i]
                self._send_text(chat_id, tg_bot._t("vp_edit_ask", lang, i=i + 1))
                # alone in its own message, as <code>: one tap copies exactly the script, no header to cut off
                self._send_text(chat_id, tg_bot._t("vp_code", lang, text=html.escape(_cut(p.get("say") or p["text"], 200))),
                                parse_mode="HTML")
            return
        if data.startswith("vp:del:"):
            try:
                i = int(data.rsplit(":", 1)[1])
            except ValueError:
                return
            if 0 <= i < len(sess.video_plan) and len(sess.video_plan) > 1:
                sess.video_plan = sess.video_plan[:i] + sess.video_plan[i + 1:]
                self._store.put(sess)
            self._send_video_plan(chat_id, sess, lang)


import tg_bot  # noqa: E402  (cycle by design; attrs read at call time)

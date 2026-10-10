"""🎬 The plan of a multi-part clip, shown before minutes of rendering start.

A script longer than one clip is rendered in parts (video.split_script), each
several minutes on the card. generate_video stops before the first part and
hands the parts here: the chat sees every part with its length, the clip's
total and the render time, and can drop a part, condense the script, render or
cancel (user, 2026-10-10: «показать сколько частей планируется и сколько займёт
и дать что-то уплотнить, что-то выкинуть»). ▶ reruns the same request with the
approved parts riding ctx.video_plan_parts (tg_tasks).

Session: video_plan [{text, sec}], video_plan_request, video_plan_ok [texts].
"""
import html
import math


def _cut(text: str, n: int = 90) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= n else t[:n - 1].rstrip(",.;:— ") + "…"


def _mins(seconds: float) -> int:
    return max(1, math.ceil(seconds / 60.0))


class VideoPlanMixin:
    def _video_plan_card(self, sess, lang: str) -> tuple:
        import video
        plan = sess.video_plan
        clip = sum(float(p.get("sec") or 0) for p in plan)
        render = sum(video.render_eta_s(float(p.get("sec") or 0)) for p in plan)
        lines = [tg_bot._t("vp_title", lang, n=len(plan), clip=f"{clip:.0f}", mins=_mins(render))]
        for i, p in enumerate(plan, 1):
            lines.append(tg_bot._t("vp_part", lang, i=i, sec=f"{float(p.get('sec') or 0):.1f}",
                                   text=html.escape(_cut(p.get("text", "")))))
        lines.append(tg_bot._t("vp_hint", lang))
        rows = [[{"text": tg_bot._t("vp_go", lang, mins=_mins(render)), "callback_data": "vp:go"}]]
        if len(plan) > 1:
            dels = [{"text": f"🗑 {i}", "callback_data": f"vp:del:{i - 1}"} for i in range(1, len(plan) + 1)]
            rows += [dels[k:k + 5] for k in range(0, len(dels), 5)]
            rows.append([{"text": tg_bot._t("vp_squeeze", lang), "callback_data": "vp:squeeze"}])
        rows.append([{"text": tg_bot._t("vp_cancel", lang), "callback_data": "vp:x"}])
        return "\n".join(lines), {"inline_keyboard": rows}

    def _offer_video_plan(self, chat_id: int, sess, lang: str, request: str, plan: list) -> None:
        sess.video_plan = [{"text": str(p.get("text", "")), "sec": float(p.get("sec") or 0)} for p in plan]
        sess.video_plan_request, sess.video_plan_ok = request, []
        self._store.put(sess)
        text, kb = self._video_plan_card(sess, lang)
        self._send_text(chat_id, text, parse_mode="HTML", keyboard=kb)

    def _squeeze_plan(self, chat_id: int, sess, lang: str) -> None:
        """The model rewrites the script tighter; the parts are re-cut from it."""
        import llm
        import video
        script = " ".join(p["text"] for p in sess.video_plan)
        target = max(1, len(sess.video_plan) - 1)
        top = video.frames_to_seconds(video.VIDEO_MAX_FRAMES_AUTO)
        try:
            out = llm.call_llm_simple(
                None,
                "You condense a video shot script. Keep the same people, place, order of "
                "events and every quoted spoken line word for word; cut repeated, minor and "
                "lingering actions and merge small beats. Reply with the condensed script "
                "only, in the script's language, no comments.",
                f"The script runs about {sum(p['sec'] for p in sess.video_plan):.0f} s. "
                f"Make it fit about {target * top:.0f} s.\n\n{script}",
                temperature=0.3, max_tokens=900) or ""
        except Exception:
            tg_bot.logger.exception("video plan: condensing failed chat=%s", chat_id)
            out = ""
        out = " ".join(out.split()).strip()
        if not out:
            self._send_text(chat_id, tg_bot._t("vp_squeeze_fail", lang))
            return
        parts = video.split_script(out)
        sess.video_plan = [{"text": p, "sec": video.estimate_seconds(p)} for p in parts]
        self._store.put(sess)
        text, kb = self._video_plan_card(sess, lang)
        self._send_text(chat_id, text, parse_mode="HTML", keyboard=kb)

    def _cb_video_plan(self, chat_id: int, data: str) -> None:
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        if not sess.video_plan or not sess.video_plan_request:
            self._send_text(chat_id, tg_bot._t("vp_gone", lang))
            return
        if data == "vp:x":
            sess.video_plan, sess.video_plan_request, sess.video_plan_ok = [], "", []
            self._store.put(sess)
            self._send_text(chat_id, tg_bot._t("vp_cancelled", lang))
            return
        if data == "vp:go":
            sess.video_plan_ok = [p["text"] for p in sess.video_plan]
            self._store.put(sess)
            self._enqueue_item(chat_id, {"type": "text", "text": sess.video_plan_request})
            return
        if data == "vp:squeeze":
            self._send_text(chat_id, tg_bot._t("vp_squeezing", lang))
            self._run_busy(chat_id, self._squeeze_plan, chat_id, sess, lang)
            return
        if data.startswith("vp:del:"):
            try:
                i = int(data.rsplit(":", 1)[1])
            except ValueError:
                return
            if 0 <= i < len(sess.video_plan) and len(sess.video_plan) > 1:
                sess.video_plan = sess.video_plan[:i] + sess.video_plan[i + 1:]
                self._store.put(sess)
            text, kb = self._video_plan_card(sess, lang)
            self._send_text(chat_id, text, parse_mode="HTML", keyboard=kb)


import tg_bot  # noqa: E402  (cycle by design; attrs read at call time)

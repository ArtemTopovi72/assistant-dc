"""Long videos in the Telegram bot: portion by portion, with timecodes.

A round note or a short clip is looked at and listened to in one go
(tg_resolve, video_look). A LONG video -- a half-hour walk, a lecture --
cannot fit on one contact sheet, so it is cut into portions (default 5
minutes): every portion is transcribed, its key moments are picked by frame
difference (video_look.key_times), the sheet is described with the speech
as context, and a retelling ties what is said to what is shown -- all with
ABSOLUTE timecodes («12:04 — …»). Each portion is delivered as soon as it is
done, so the user reads part 1 while part 2 is still cooking, and ⛔ Stop
ends the job between portions.

Settings live under Creativity ▸ 🎬 Video (the user's call: no new tab, the
existing menus), shaped like tg_music: an inline menu that STATES the
current value, redrawn in place on every tap.
"""
from __future__ import annotations

import html as _html_mod
import re
import logging
import os
import subprocess
import threading
import turn_trace
import time
import uuid
from pathlib import Path

from tg_strings import _DEFAULT_LANG, _nav_back_row, _t

logger = logging.getLogger("assistant.tg_video")

# ── settings ─────────────────────────────────────────────────────────────────
FIELDS: tuple = ("chunk", "frames", "out", "long")
CHUNKS = ("2", "5", "10")            # minutes per portion
FRAMES = ("6", "12", "20")           # key frames per portion
OUTS = ("both", "board", "retell")   # what each portion delivers
LONGS = ("1", "2", "5")              # a video longer than this (minutes) is "long"
DEFAULTS = {"chunk": "5", "frames": "12", "out": "both", "long": "2"}
_TABLE = {"chunk": CHUNKS, "frames": FRAMES, "out": OUTS, "long": LONGS}


def _resolve(sess, field: str) -> str:
    raw = getattr(sess, "video_" + field, "")
    val = raw if isinstance(raw, str) else ""
    return val if val in _TABLE[field] else DEFAULTS[field]


def chunk_seconds(sess) -> int: return int(_resolve(sess, "chunk")) * 60
def frames_per_chunk(sess) -> int: return int(_resolve(sess, "frames"))
def output_mode(sess) -> str: return _resolve(sess, "out")
def long_threshold(sess) -> int: return int(_resolve(sess, "long")) * 60


def is_long(sess, seconds: int) -> bool:
    return int(seconds or 0) >= long_threshold(sess)


def _value_label(field: str, value: str, lang: str) -> str:
    if field in ("chunk", "long"):
        return _t("vs_min", lang, n=value)
    if field == "frames":
        return _t("vs_frames_n", lang, n=value)
    return _t("vs_out_" + value, lang)


def _video_menu_kb(sess, lang: str = _DEFAULT_LANG) -> dict:
    rows = [[{"text": f"{_t('vs_' + f, lang)}: {_value_label(f, _resolve(sess, f), lang)}",
              "callback_data": "video:open:" + f}] for f in FIELDS]
    rows.append([{"text": _t("vs_reset", lang), "callback_data": "video:reset"}])
    rows.append(_nav_back_row(lang))
    return {"inline_keyboard": rows}


def _field_kb(sess, field: str, lang: str = _DEFAULT_LANG) -> dict:
    cur = _resolve(sess, field)
    rows, row = [], []
    for v in _TABLE[field]:
        row.append({"text": ("✅ " if v == cur else "") + _value_label(field, v, lang),
                    "callback_data": f"video:set:{field}:{v}"})
        if len(row) == 2:
            rows.append(row); row = []
    if row:
        rows.append(row)
    rows.append([{"text": _t("vs_back", lang), "callback_data": "video:menu"}])
    return {"inline_keyboard": rows}


def _video_menu_text(sess, lang: str = _DEFAULT_LANG) -> str:
    return _t("vs_title", lang) + "\n\n" + _html_mod.escape(_t("vs_hint", lang))


def _field_text(sess, field: str, lang: str = _DEFAULT_LANG) -> str:
    return (_t("vs_field_title", lang, field=_t("vs_" + field, lang)) + "\n\n"
            + _html_mod.escape(_t("vs_pick_" + field, lang)))


def summary(sess, lang: str = _DEFAULT_LANG) -> str:
    return " · ".join(f"{_t('vs_' + f, lang)}: {_value_label(f, _resolve(sess, f), lang)}"
                      for f in ("chunk", "frames", "out"))


def _ask_kb(lang: str, job_id: str) -> dict:
    return {"inline_keyboard": [
        [{"text": _t("lv_go", lang), "callback_data": "lv:go:" + job_id}],
        [{"text": _t("lv_settings", lang), "callback_data": "video:menu"},
         {"text": _t("lv_skip", lang), "callback_data": "lv:skip:" + job_id}],
    ]}


def _stamp(t: float) -> str:
    t = int(t); h, r = divmod(t, 3600); m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def cut_portion(src: str, start: float, length: float, dst: str) -> bool:
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}",
                        "-i", src, "-t", f"{length:.2f}", "-c", "copy", dst],
                       capture_output=True, check=True, timeout=300)
        return os.path.exists(dst) and os.path.getsize(dst) > 0
    except Exception:
        logger.warning("cut %s +%s failed", start, length, exc_info=True)
        return False


# ── the job ──────────────────────────────────────────────────────────────────
# One structure for every retelling (a voice note, a round video, a portion of
# a long video): the shape the user liked before the one-shot call replaced
# the agent -- a one-line gist, the key points, and a block of what is asked
# of the listener. The model writes the section labels verbatim; render_retelling
# turns them into bold lines for Telegram.
_SECTIONS = {
    "ru": ("Основная тема:", "Ключевые моменты:", "Что требуется:", "ничего — вопросов к слушателю нет"),
    "en": ("Main topic:", "Key points:", "What is asked:", "nothing — no questions for the listener"),
}


def _structure_rules(lang: str, timecoded: bool, ask: bool = True) -> str:
    """The retelling shape. A voice note or a round video sent to the chat is a
    message TO the user, so it ends with what is asked of them; a long video
    (a tutorial, a lecture) is not addressed to anyone -- no such block."""
    topic, key, asked, none = _SECTIONS.get(lang, _SECTIONS["ru"])
    rules = (
        "Answer in EXACTLY this structure, section labels verbatim, nothing before or after:\n"
        f"{topic} <2-3 sentences -- what this is about and the speaker's attitude>\n\n"
        f"{key}\n"
        + ("• <m:ss — Short title: what happens on screen and what is said about it>  "
           "(one bullet per moment, 4-10 bullets, timecodes absolute)\n" if timecoded else
           "• <Short title: the point in one or two sentences, with the speaker's own facts, "
           "names and numbers>  (3-8 bullets)\n"))
    if ask:
        rules += (f"\n{asked}\n• <what the speaker asks of the listener / viewer -- a question, "
                  f"a request, a deadline; or a single bullet «{none}»>")
    return rules


_RETELL_SYSTEM = (
    "You retell one PORTION of a long video. You get the TRANSCRIPT of what is "
    "said in it and a STORYBOARD of what is shown (one line per timecode; the "
    "timecodes are absolute positions in the whole video). Tie the words to what "
    "is visible; keep only what is informative. ")


_HEAD_RE = re.compile(r"^\s*(?:\*\*|__)?\s*(Основная тема|Ключевые моменты|Что требуется|Main topic|Key points|What is asked)\s*:\s*(?:\*\*|__)?\s*(.*)$",
                      re.IGNORECASE)


def render_retelling(text: str, title: str) -> str:
    """HTML for Telegram: a bold title, bold section labels, tidy bullets."""
    out = ["📋 <b>" + _html_mod.escape(title) + "</b>"]
    for raw in (text or "").splitlines():
        line = raw.strip().replace("**", "")
        if not line:
            continue
        m = _HEAD_RE.match(line)
        if m:
            rest = m.group(2).strip()
            out.append("\n<b>" + _html_mod.escape(m.group(1)) + ":</b>" + (" " + _html_mod.escape(rest) if rest else ""))
            continue
        line = re.sub(r"^[-*·]\s+", "• ", line)
        out.append(_html_mod.escape(line))
    return "\n".join(out).strip()


class LongVideoMixin:
    """Mixed into TelegramBot: the ask, the job, Stop."""

    def _lv_jobs(self) -> dict:
        if not hasattr(self, "_long_video_jobs"):
            self._long_video_jobs = {}
        return self._long_video_jobs

    def _long_video_offer(self, chat_id: int, sess, lang: str, item: dict) -> None:
        """A long video arrived: say how it will be handled and ask to start.
        The file is NOT downloaded yet -- the user may decline."""
        job_id = uuid.uuid4().hex[:8]
        sess.long_video = {"id": job_id, "file_id": item.get("file_id", ""), "url": item.get("url", ""),
                           "seconds": int(item.get("seconds") or 0),
                           "caption": item.get("caption") or ""}
        self._store.put(sess)
        mins = max(1, round(sess.long_video["seconds"] / 60))
        self._send_text(chat_id, _t("lv_offer", lang, mins=mins, setup=summary(sess, lang)),
                        parse_mode="HTML", keyboard=_ask_kb(lang, job_id))
        self._activity.log(chat_id, "system", f"[long video] offered {mins} min, job {job_id}")

    def _cb_long_video(self, chat_id: int, data: str) -> None:
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        parts = data.split(":")
        action, job_id = (parts[1] if len(parts) > 1 else ""), (parts[2] if len(parts) > 2 else "")
        pending = getattr(sess, "long_video", None) or {}
        if not pending or pending.get("id") != job_id:
            self._send_text(chat_id, _t("lv_gone", lang)); return
        if action == "skip":
            sess.long_video = None; self._store.put(sess)
            self._send_text(chat_id, _t("lv_skipped", lang),
                            keyboard=self._main_menu_kb(sess, lang)); return
        if action != "go":
            return
        if chat_id in self._lv_jobs():
            self._send_text(chat_id, _t("lv_busy", lang)); return
        sess.long_video = None; self._store.put(sess)
        stop = threading.Event()
        self._lv_jobs()[chat_id] = stop
        turn_trace.spawn(self._long_video_job, chat_id, pending, stop, name=f"long-video-{chat_id}")

    def _long_video_stop(self, chat_id: int) -> bool:
        """Called from ⛔ Stop / /cancel. True if a long-video job was running."""
        ev = self._lv_jobs().get(chat_id)
        if ev is None:
            return False
        ev.set()
        return True

    def _long_video_job(self, chat_id: int, spec: dict, stop: threading.Event) -> None:
        import tg_bot
        import video_look
        import llm as _llm
        from audio import transcribe_audio_file
        sess = self._get_session(chat_id)
        lang = self._lang(sess)
        chunk = chunk_seconds(sess); nframes = frames_per_chunk(sess); mode = output_mode(sess)
        work = Path(tg_bot._IMAGE_DIR) / f"longvideo_{spec['id']}"
        work.mkdir(parents=True, exist_ok=True)
        src = str(work / "src.mp4")
        try:
            self._send_text(chat_id, _t("lv_downloading", lang))
            if spec.get("url"):
                # a link: the whole film, no 10-minute cap (ponytail: held in RAM
                # once, ~1 GB for 90 min at 720p; stream to disk if films grow)
                import tg_links
                data = tg_links.fetch_video(spec["url"], max_seconds=6 * 3600).get("data")
            else:
                data = self._dl_bytes(spec["file_id"])
            if not data:
                self._send_text(chat_id, _t("lv_no_file", lang),
                                keyboard=self._main_menu_kb(sess, lang)); return
            Path(src).write_bytes(data); del data
            total = video_look.duration_seconds(src) or float(spec.get("seconds") or 0)
            n_parts = max(1, int((total + chunk - 1) // chunk))
            self._send_text(chat_id, _t("lv_start", lang, parts=n_parts, mins=chunk // 60,
                                        total=_stamp(total)))
            ctx = self._get_ctx()
            # The film's language, once for all portions -- not the chat's.
            from audio import detect_media_language
            asr_lang = detect_media_language(ctx, src) or lang or "ru"
            done = 0
            for k in range(n_parts):
                if stop.is_set():
                    break
                t0 = k * chunk
                length = min(chunk, max(0.0, total - t0)) or chunk
                part = str(work / f"part_{k + 1:03d}.mp4")
                self._api_post("sendChatAction", {"chat_id": chat_id, "action": "typing"})
                if not cut_portion(src, t0, length, part):
                    self._send_text(chat_id, _t("lv_part_failed", lang, n=k + 1)); continue
                said = ""
                try:
                    said = (transcribe_audio_file(ctx, part, lang_hint=asr_lang) or "").strip()
                except Exception:
                    logger.warning("portion %d: ASR failed", k + 1, exc_info=True)
                if stop.is_set():
                    break
                seen = video_look.look_at_video(ctx, part, max_frames=nframes,
                                                work_dir=work / f"look_{k + 1:03d}",
                                                lang=lang, transcript=said, offset=t0)
                board = (seen.get("description") or "").strip()
                head = _t("lv_part_head", lang, n=k + 1, of=n_parts,
                          a=_stamp(t0), b=_stamp(min(total, t0 + length)))
                if mode in ("both", "board") and board:
                    self._send_text(chat_id, head + "\n👁 <b>" + _t("lv_board", lang) + "</b>\n"
                                    + _html_mod.escape(board), parse_mode="HTML")
                if mode in ("both", "retell") and (said or board):
                    if stop.is_set():
                        break
                    payload = ""
                    if said:
                        payload += "TRANSCRIPT:\n" + said + "\n\n"
                    if board:
                        payload += "STORYBOARD:\n" + board
                    lang_line = {"ru": " Write in Russian.", "en": " Write in English."}.get(lang, " Write in Russian.")
                    try:
                        tale = (_llm.call_llm_simple(ctx, _RETELL_SYSTEM + lang_line + "\n"
                                                     + _structure_rules(lang, timecoded=True, ask=False), payload,
                                                     temperature=0.3, max_tokens=700,
                                                     prefill="<think></think>") or "").strip()
                    except Exception:
                        logger.exception("portion %d: retell failed", k + 1); tale = ""
                    if tale:
                        prefix = head + "\n" if mode == "retell" else ""
                        for chunk_txt in tg_bot._split_html(prefix + render_retelling(tale, _t("lv_retell", lang))):
                            self._send_text(chat_id, chunk_txt, parse_mode="HTML")
                if seen.get("sheet") and mode != "retell":
                    try:
                        self._send_photo(chat_id, seen["sheet"])
                    except Exception:
                        logger.debug("sheet send failed", exc_info=True)
                done += 1
                self._activity.log(chat_id, "system",
                                   f"[long video] part {k + 1}/{n_parts} {_stamp(t0)}: "
                                   f"{len(seen.get('times') or [])} frames, {len(said)} chars")
            if stop.is_set():
                self._send_text(chat_id, _t("lv_stopped", lang, done=done, of=n_parts),
                                keyboard=self._main_menu_kb(sess, lang))
            else:
                self._send_text(chat_id, _t("lv_done", lang, parts=done, total=_stamp(total)),
                                keyboard=self._main_menu_kb(sess, lang))
        except Exception:
            logger.exception("long video job failed chat=%s", chat_id)
            self._send_text(chat_id, _t("lv_failed", lang), keyboard=self._main_menu_kb(sess, lang))
        finally:
            self._lv_jobs().pop(chat_id, None)
            try:
                if os.path.exists(src):
                    os.unlink(src)
                for p in work.glob("part_*.mp4"):
                    p.unlink()
            except Exception:
                pass

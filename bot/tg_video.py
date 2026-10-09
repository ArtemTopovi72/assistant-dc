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
# «brief»: one retelling of the whole video at the end and questions on demand -- per-portion
# storyboards (11 parts x 15 lines + a sheet each) buried the chat (10-09); the others stay for
# whoever wants every part
OUTS = ("brief", "both", "board", "retell")   # what each portion delivers
LONGS = ("1", "2", "5")              # a video longer than this (minutes) is "long"
DEFAULTS = {"chunk": "5", "frames": "12", "out": "brief", "long": "2"}
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
        [{"text": _t("lv_ask", lang), "callback_data": "lv:ask:" + job_id}],
        [{"text": _t("lv_settings", lang), "callback_data": "video:menu"},
         {"text": _t("lv_skip", lang), "callback_data": "lv:skip:" + job_id}],
    ]}


def _after_kb(lang: str, job_id: str) -> dict:
    """Under a finished video: ask about it, or open every portion."""
    return {"inline_keyboard": [
        [{"text": _t("lv_more", lang), "callback_data": "lv:more:" + job_id}],
        [{"text": _t("lv_parts", lang), "callback_data": "lv:parts:" + job_id}],
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


_OVERVIEW_SYSTEM = (
    "You retell a WHOLE long video from the retellings of its portions, in order (the "
    "timecodes are absolute positions in the video). Merge them into one retelling: the "
    "thread through the whole video and its turning points, not a list of the portions. ")

_ANSWER_SYSTEM = (
    "You answer the user's question about a long video from its notes: for every portion, its "
    "retelling, its STORYBOARD (what is shown, one line per absolute timecode) and its TRANSCRIPT "
    "(what is said). Answer only from these notes. Start with the direct answer in one or two "
    "sentences, then the moments that show it as bullets «m:ss — what happens there». When the "
    "notes do not show it, say so plainly. Plain text, no markdown. ")


def material_text(parts: list, budget: int = 24000) -> str:
    """The notes an answer reads: every portion's retelling and storyboard, and as much of its
    transcript as the budget leaves (a 90-minute film's full transcript would not fit the model)."""
    def block(p: dict, said: str) -> str:
        out = f"=== {p['a']}-{p['b']} ===\n"
        if p.get("note"):
            out += "RETELLING:\n" + p["note"] + "\n"
        if p.get("board"):
            out += "STORYBOARD:\n" + p["board"] + "\n"
        if said:
            out += "TRANSCRIPT:\n" + said + "\n"
        return out
    base = "\n".join(block(p, "") for p in parts)
    room = (budget - len(base)) // max(1, len(parts))
    if room < 200:
        return base[:budget]
    return "\n".join(block(p, (p.get("said") or "")[:room]) for p in parts)[:budget]


def _material_path(image_dir: str, job_id: str) -> Path:
    return Path(image_dir) / f"longvideo_{re.sub(r'[^0-9a-f]', '', job_id)}" / "material.json"


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


def _save_material(work: Path, parts: list) -> None:
    """After every portion, so a Stop or a crash keeps what was watched answerable."""
    import json
    try:
        (work / "material.json").write_text(json.dumps({"parts": parts}, ensure_ascii=False), encoding="utf-8")
    except OSError:
        logger.warning("long video: notes not saved", exc_info=True)


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
        if action in ("more", "parts"):
            return self._long_video_after(chat_id, sess, lang, action, job_id)
        pending = getattr(sess, "long_video", None) or {}
        if not pending or pending.get("id") != job_id:
            self._send_text(chat_id, _t("lv_gone", lang)); return
        if action == "skip":
            sess.long_video = None; self._store.put(sess)
            self._send_text(chat_id, _t("lv_skipped", lang),
                            keyboard=self._main_menu_kb(sess, lang)); return
        if action == "ask":
            # the next message is the question; the video stays pending until it comes
            sess.reg_state = "lv_question:" + job_id; self._store.put(sess)
            self._send_text(chat_id, _t("lv_ask_prompt", lang)); return
        if action != "go":
            return
        self._long_video_start(chat_id, sess, lang, pending)

    def _long_video_start(self, chat_id: int, sess, lang: str, pending: dict) -> None:
        if chat_id in self._lv_jobs():
            self._send_text(chat_id, _t("lv_busy", lang)); return
        sess.long_video = None; self._store.put(sess)
        stop = threading.Event()
        self._lv_jobs()[chat_id] = stop
        turn_trace.spawn(self._long_video_job, chat_id, pending, stop, name=f"long-video-{chat_id}")

    def _long_video_question(self, chat_id: int, sess, lang: str, state: str, text: str) -> None:
        """The message an armed «lv_question:<id>» / «lv_more:<id>» was waiting for. Before the
        job: the job runs and answers it. After it: answered from the saved notes, no re-watch."""
        kind, _, job_id = state.partition(":")
        if kind == "lv_question":
            pending = dict(getattr(sess, "long_video", None) or {})
            if pending.get("id") != job_id:
                self._send_text(chat_id, _t("lv_gone", lang)); return
            pending["question"] = text
            return self._long_video_start(chat_id, sess, lang, pending)
        import json
        import tg_bot
        try:
            mat = json.loads(_material_path(tg_bot._IMAGE_DIR, job_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._send_text(chat_id, _t("lv_gone", lang)); return
        turn_trace.spawn(self._long_video_answer, chat_id, lang, job_id, mat.get("parts") or [], text,
                         name=f"long-video-q-{chat_id}")

    def _long_video_answer(self, chat_id: int, lang: str, job_id: str, parts: list, question: str) -> None:
        import llm as _llm
        import tg_bot
        self._api_post("sendChatAction", {"chat_id": chat_id, "action": "typing"})
        lang_line = {"ru": " Write in Russian.", "en": " Write in English."}.get(lang, " Write in Russian.")
        try:
            answer = (_llm.call_llm_simple(self._get_ctx(), _ANSWER_SYSTEM + lang_line,
                                           "QUESTION: " + question + "\n\nNOTES:\n" + material_text(parts),
                                           temperature=0.2, max_tokens=900, prefill="<think></think>") or "").strip()
        except Exception:
            logger.exception("long video: answer failed"); answer = ""
        if not answer:
            self._send_text(chat_id, _t("lv_failed", lang), keyboard=_after_kb(lang, job_id)); return
        pieces = tg_bot._split_html("❓ <b>" + _html_mod.escape(question[:200]) + "</b>\n\n"
                                    + _html_mod.escape(answer.replace("**", "")))
        for i, piece in enumerate(pieces):
            self._send_text(chat_id, piece, parse_mode="HTML",
                            keyboard=_after_kb(lang, job_id) if i == len(pieces) - 1 else None)

    def _long_video_overview(self, chat_id: int, lang: str, job_id: str, parts: list) -> None:
        """«brief»: one retelling of the whole video from the portions' own retellings."""
        import llm as _llm
        import tg_bot
        notes = "\n\n".join(f"=== {p['a']}-{p['b']} ===\n" + (p.get("note") or p.get("board") or "")
                             for p in parts)
        lang_line = {"ru": " Write in Russian.", "en": " Write in English."}.get(lang, " Write in Russian.")
        try:
            tale = (_llm.call_llm_simple(self._get_ctx(), _OVERVIEW_SYSTEM + lang_line + "\n"
                                         + _structure_rules(lang, timecoded=True, ask=False), notes[:24000],
                                         temperature=0.3, max_tokens=1100, prefill="<think></think>") or "").strip()
        except Exception:
            logger.exception("long video: overview failed"); tale = ""
        if not tale:
            self._send_text(chat_id, _t("lv_failed", lang), keyboard=_after_kb(lang, job_id)); return
        pieces = tg_bot._split_html(render_retelling(tale, _t("lv_overview", lang)))
        for i, piece in enumerate(pieces):
            self._send_text(chat_id, piece, parse_mode="HTML",
                            keyboard=_after_kb(lang, job_id) if i == len(pieces) - 1 else None)

    def _long_video_after(self, chat_id: int, sess, lang: str, action: str, job_id: str) -> None:
        """❓ another question, or 📜 every portion's storyboard and retelling, on demand."""
        import json
        import tg_bot
        try:
            mat = json.loads(_material_path(tg_bot._IMAGE_DIR, job_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._send_text(chat_id, _t("lv_gone", lang)); return
        if action == "more":
            sess.reg_state = "lv_more:" + job_id; self._store.put(sess)
            self._send_text(chat_id, _t("lv_ask_prompt", lang)); return
        parts = mat.get("parts") or []
        for k, p in enumerate(parts):
            msg = _t("lv_part_head", lang, n=k + 1, of=len(parts), a=p["a"], b=p["b"])
            if p.get("board"):
                msg += "\n👁 <b>" + _t("lv_board", lang) + "</b>\n" + _html_mod.escape(p["board"])
            if p.get("note"):
                msg += "\n\n" + render_retelling(p["note"], _t("lv_retell", lang))
            for piece in tg_bot._split_html(msg):
                self._send_text(chat_id, piece, parse_mode="HTML")

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
        question = (spec.get("question") or "").strip()
        quiet = bool(question) or mode == "brief"        # nothing per portion, one answer at the end
        material: list = []
        status_id = None
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
            if quiet:
                status_id = self._send_get_id(chat_id, _t("lv_progress", lang, n=1, of=n_parts, total=_stamp(total)))
            else:
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
                if status_id and k:
                    self._edit_text(chat_id, status_id, _t("lv_progress", lang, n=k + 1, of=n_parts,
                                                           total=_stamp(total)))
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
                if not quiet and mode in ("both", "board") and board:
                    self._send_text(chat_id, head + "\n👁 <b>" + _t("lv_board", lang) + "</b>\n"
                                    + _html_mod.escape(board), parse_mode="HTML")
                tale = ""
                if (quiet or mode in ("both", "retell")) and (said or board):
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
                    if tale and not quiet:
                        prefix = head + "\n" if mode == "retell" else ""
                        for chunk_txt in tg_bot._split_html(prefix + render_retelling(tale, _t("lv_retell", lang))):
                            self._send_text(chat_id, chunk_txt, parse_mode="HTML")
                material.append({"a": _stamp(t0), "b": _stamp(min(total, t0 + length)),
                                 "board": board, "said": said, "note": tale})
                _save_material(work, material)
                if seen.get("sheet") and not quiet and mode != "retell":
                    try:
                        self._send_photo(chat_id, seen["sheet"])
                    except Exception:
                        logger.debug("sheet send failed", exc_info=True)
                done += 1
                self._activity.log(chat_id, "system",
                                   f"[long video] part {k + 1}/{n_parts} {_stamp(t0)}: "
                                   f"{len(seen.get('times') or [])} frames, {len(said)} chars")
            if status_id:
                self._edit_text(chat_id, status_id, _t("lv_watched", lang, parts=done, total=_stamp(total)))
            if quiet and material and question:
                self._long_video_answer(chat_id, lang, spec["id"], material, question)
            elif quiet and material:
                self._long_video_overview(chat_id, lang, spec["id"], material)
            if stop.is_set():
                self._send_text(chat_id, _t("lv_stopped", lang, done=done, of=n_parts),
                                keyboard=_after_kb(lang, spec["id"]) if material else self._main_menu_kb(sess, lang))
            elif not quiet:
                self._send_text(chat_id, _t("lv_done", lang, parts=done, total=_stamp(total)),
                                keyboard=_after_kb(lang, spec["id"]))
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

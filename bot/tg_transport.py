"""Telegram HTTP transport and media delivery.

Split out of tg_bot.py: everything that speaks to api.telegram.org — the
retrying POST/GET wrappers, message send/edit/delete, and the photo, video,
document and voice delivery paths with their parse-mode fallbacks.

Module-level configuration and formatting helpers are read through ``tg_bot``
rather than imported by value, so a test that patches ``tg_bot._MAX_TEXT`` (or
any of the other names below) still changes the behaviour of this module.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import requests
import chatlog as _chatlog


# A context variable (like chatlog and turn_trace), so work spawned for a task
# still replies to the message that asked for it.
_REPLY: contextvars.ContextVar = contextvars.ContextVar("tg_reply_to", default=None)

_EXIT_WORDS = ("cancel", "back", "close", "menu", "stop", "abort", "no")
_EXIT_MARKS = ("↩", "✖", "❌", "⛔", "🔙", "⬅")


def _is_exit(btn: dict) -> bool:
    """Does this inline button already lead out of the menu?"""
    data = str(btn.get("callback_data") or "").lower()
    if any(w in data.replace("_", ":").split(":") for w in _EXIT_WORDS):
        return True
    return str(btn.get("text") or "").lstrip().startswith(_EXIT_MARKS)


@contextlib.contextmanager
def replying_to(chat_id, msg_id):
    """Everything a task sends is threaded to the message that asked for it."""
    token = _REPLY.set((chat_id, msg_id) if msg_id else None)
    try:
        yield
    finally:
        _REPLY.reset(token)


# Display width of a row of buttons, in characters (an emoji counts two).
# Telegram cuts a label that is wider than its share of the row («Без на…»).
# Calibrated on the live 16:57 keyboard: at three a row «🔄 Ещё раз» fits and
# «🔤 Без надписей» is cut. A knob, not a constant: clients differ.
TG_ROW_WIDTH = int(os.getenv("TG_ROW_WIDTH", "34"))


def _label_width(text: str) -> int:
    w = 0
    for ch in str(text or ""):
        o = ord(ch)
        if o in (0xFE0F, 0x200D):           # emoji variation selector / joiner
            continue
        w += 2 if (o >= 0x1F000 or 0x2190 <= o <= 0x2BFF) else 1
    return w


def fit_rows(rows: list) -> list:
    """Rows whose labels do not fit their share of the width are split, order
    kept; rows that fit are left as they are. Every keyboard the bot sends goes
    through here (_logged_post), so no menu can be laid out too tight."""
    out = []
    for row in rows or []:
        if not isinstance(row, list):
            out.append(row); continue
        cur = []
        for b in row:
            cand = cur + [b]
            widest = max(_label_width(x.get("text") if isinstance(x, dict) else x) for x in cand)
            if cur and widest * len(cand) > TG_ROW_WIDTH:
                out.append(cur); cur = [b]
            else:
                cur = cand
        if cur:
            out.append(cur)
    return out


def _fit_markup(markup):
    """reply_markup as a dict or as its JSON text -> the same, rows fitted."""
    try:
        m = json.loads(markup) if isinstance(markup, str) else markup
        if not isinstance(m, dict):
            return markup
        for key in ("inline_keyboard", "keyboard"):
            if isinstance(m.get(key), list):
                m = dict(m, **{key: fit_rows(m[key])})
        if isinstance(m.get("keyboard"), list):
            # A menu the client may collapse leaves a picker message with no way
            # back (live 10-02: «Размер» -> quality tapped -> no menu at all).
            m.setdefault("is_persistent", True)
        return json.dumps(m) if isinstance(markup, str) else m
    except Exception:
        return markup


def _logged_post(url, **kw):
    """requests.post, with the message copied into the chat's full transcript."""
    payload = kw.get("json") if kw.get("json") is not None else kw.get("data")
    if isinstance(payload, dict) and payload.get("reply_markup"):
        payload["reply_markup"] = _fit_markup(payload["reply_markup"])
    tgt = _REPLY.get()
    if (tgt and isinstance(payload, dict) and "/send" in url and "sendChatAction" not in url
            and str(payload.get("chat_id")) == str(tgt[0]) and "reply_parameters" not in payload):
        rp = {"message_id": tgt[1], "allow_sending_without_reply": True}
        payload["reply_parameters"] = rp if kw.get("json") is not None else json.dumps(rp)
    if isinstance(payload, dict):
        files = kw.get("files") or {}
        _chatlog.outgoing(url.rsplit("/", 1)[-1],
                          dict(payload, **({"_files": list(files)} if files else {})))
    return requests.post(url, **kw)


import config as _config
import tg_artifacts_store as _artifacts


def _uri_not_readable(body: str) -> bool:
    """The local server could not use a file:// path (it cannot see it)."""
    low = (body or "").lower()
    return any(k in low for k in ("wrong file identifier", "file not found",
                                  "failed to get http url content",
                                  "can't open file", "file must be non-empty"))


class TransportMixin:
    def _log_artifact(self, chat_id, kind: str, path: str, label: str = "") -> None:
        """Best-effort durable record of a file just handed to a real chat.

        Called from the one place in each media sender that KNOWS delivery
        succeeded, so this table only ever holds things genuinely sent —
        never a guess about what might exist on disk. See
        tg_artifacts_store's module docstring.
        """
        try:
            _artifacts.log_artifact(chat_id, kind, path, label=label or "")
        except Exception:
            tg_bot.logger.exception("artifact logging failed for %s %s", kind, path)

    def _dl_bytes(self, file_id: str) -> Optional[bytes]:
        data, fp = self._dl_bytes_raw(file_id)
        if data:
            # every file a user sends goes through here: the turn keeps a copy
            import turn_trace
            turn_trace.media_bytes(data, os.path.splitext(fp)[1])
        return data

    def _dl_bytes_raw(self, file_id: str):
        fp = ""
        try:
            r  = self._api_get("getFile", {"file_id": file_id})
            fp = (r.get("result") or {}).get("file_path")
            if not fp: return None, ""
            # A local Bot API server (--local) answers getFile with an
            # absolute path on this machine; nothing to download.
            if os.path.isabs(fp) and os.path.exists(fp):
                with open(fp, "rb") as fh:
                    return fh.read(), fp
            r2 = requests.get(
                f"{_config.TG_API_BASE}/file/bot{self.token}/{fp}",
                timeout=tg_bot._DOWNLOAD_TIMEOUT)
            return (r2.content if r2.status_code == 200 else None), fp
        except Exception:
            tg_bot.logger.exception("dl_bytes failed %s", file_id)
            return None, fp

    @staticmethod
    def _is_parse_error(resp) -> bool:
        """True when Telegram rejected the message for its FORMATTING alone.

        Anything else (blocked bot, bad chat_id, network) must not trigger a
        resend — only a markup problem is recoverable by dropping the markup.
        """
        if not isinstance(resp, dict) or resp.get("ok"):
            return False
        # One shared rule with the media senders. The old test here was
        # "parse"/"entit"/"tag" as bare substrings, which also matched
        # "Request Entity Too Large" and "chat not found ... tag" — resending
        # those without markup fails identically.
        return tg_bot._is_markup_error(resp.get("description", ""))

    def _post_message(self, method: str, p: dict, parse_mode, keyboard) -> dict:
        """sendMessage/editMessageText with a plain-text retry.

        A single malformed tag used to cost the user the ENTIRE answer: the
        response was never inspected, so a "can't parse entities" rejection looked
        exactly like success and the reply was silently dropped.
        """
        # The one language checkpoint: every message and inline button of a
        # Russian chat (ui_lang_guard: translates known stages, logs leftovers).
        try:
            import ui_lang_guard as _lg
            _chat = p.get("chat_id")
            _lang = self._lang(self._get_session(_chat)) if _chat else ""
            if _lang == "ru":
                p["text"] = _lg.guard_text(p.get("text", ""), _lang, _chat)
                keyboard = _lg.guard_keyboard(keyboard, _lang, _chat)
        except Exception:
            pass
        if parse_mode: p["parse_mode"] = parse_mode
        if keyboard and "keyboard" in keyboard:
            # Menus stay open: without this a client folds the reply keyboard
            # away after a press and the user reopens it for every setting.
            keyboard = {**keyboard, "is_persistent": True}
        if keyboard:   p["reply_markup"] = json.dumps(keyboard)
        r = self._api_post(method, p)
        if parse_mode and self._is_parse_error(r):
            tg_bot.logger.warning("%s rejected markup (%s) — resending as plain text",
                           method, str(r.get("description"))[:120])
            p2 = dict(p)
            p2.pop("parse_mode", None)
            p2["text"] = tg_bot._html_to_plain(p["text"])[:tg_bot._MAX_TEXT]
            r = self._api_post(method, p2)
        # A short menu reply to a button press joins the tail the next press wipes (tg_dispatch).
        tail = getattr(self, "_menu_tail", {}).get(p.get("chat_id"))
        if (tail and tail["open"] and method == "sendMessage" and len(p.get("text", "")) <= 200
                and not getattr(_REPLY, "target", None) and (r.get("result") or {}).get("message_id")):
            tail["ids"].append(r["result"]["message_id"])
        return r

    def _send_text(self, chat_id, text, parse_mode=None, keyboard=None):
        keyboard = self._with_wait_cancel(chat_id, keyboard)
        self._post_message("sendMessage",
                           {"chat_id": chat_id, "text": text[:tg_bot._MAX_TEXT]},
                           parse_mode, keyboard)

    def _with_wait_cancel(self, chat_id, keyboard):
        """While the chat waits for the user's next message, whatever goes out
        carries ✖️ Отмена (a reply keyboard cannot mix with it: left alone)."""
        try:
            if keyboard is not None and "inline_keyboard" not in keyboard:
                return keyboard
            sess = self._get_session(chat_id)
            rows = list((keyboard or {}).get("inline_keyboard") or [])
            if not sess.waiting_for_input():
                # A menu of buttons with no way out (👤 Аккаунт) is a trap too:
                # every inline menu gets ↩ Назад unless it already has an exit.
                if rows and not any(_is_exit(b) for r in rows for b in r):
                    rows.append([{"text": tg_bot._b("back", self._lang(sess)),
                                  "callback_data": "nav:close"}])
                    return {**keyboard, "inline_keyboard": rows}
                return keyboard
            if any(b.get("callback_data") == "wait:cancel" for r in rows for b in r):
                return keyboard
            rows.append([{"text": tg_bot._t("wait_cancel_btn", self._lang(sess)),
                          "callback_data": "wait:cancel"}])
            return {**(keyboard or {}), "inline_keyboard": rows}
        except Exception:
            return keyboard

    def _send_get_id(self, chat_id, text, parse_mode=None,
                     keyboard=None) -> Optional[int]:
        r = self._post_message("sendMessage",
                               {"chat_id": chat_id, "text": text[:tg_bot._MAX_TEXT]},
                               parse_mode, keyboard)
        return (r.get("result") or {}).get("message_id")

    def _edit_text(self, chat_id, msg_id, text, parse_mode=None, keyboard=None):
        self._post_message("editMessageText",
                           {"chat_id": chat_id, "message_id": msg_id,
                            "text": text[:tg_bot._MAX_TEXT]},
                           parse_mode, keyboard)

    def _archive(self, chat_id, path: str) -> str:
        """Link a delivered file into runtime/generated/<user>/<category>/."""
        import nice_names
        try:
            user = self._user_store.get(chat_id)
            who = (getattr(user, "name", "") or "") if user else ""
        except Exception:
            who = ""
        return nice_names.named_path(chat_id, path, who)

    def _send_document(self, chat_id, path: str, caption: str = "") -> bool:
        """Send a file. Used for reports too long to read as chat messages."""
        _orig, path = path, self._archive(chat_id, path)   # the log keeps the original
        data: dict = {"chat_id": chat_id}
        if caption:
            data["caption"] = tg_bot._safe_caption(caption)
            data["parse_mode"] = "HTML"
        # The self-hosted server runs with --local, so it reads the file from
        # disk by URI: no multipart upload (a 117 MB zip took minutes and hit
        # the 180 s timeout) and the 2 GB limit instead of the cloud's 50 MB.
        by_uri = bool(getattr(_config, "TG_API_LOCAL", False))
        tries, degraded = 0, False
        while tries < tg_bot._API_RETRIES:
            tries += 1
            try:
                if by_uri:
                    r = _logged_post(f"{self._api}/sendDocument",
                                      data={**data, "document": Path(path).resolve().as_uri()},
                                      timeout=600)
                    if r.status_code == 400 and _uri_not_readable(r.text):
                        # A path the server cannot see (another machine, a
                        # sandbox mount) still has the upload to fall back on.
                        # Anything else -- a blocked bot, a caption it cannot
                        # parse -- would fail the upload the same way.
                        tg_bot.logger.warning("sendDocument by URI refused (%s): %s -- uploading",
                                              r.status_code, r.text[:200])
                        by_uri = False
                        tries -= 1
                        continue
                else:
                    with open(path, "rb") as fh:
                        r = _logged_post(f"{self._api}/sendDocument",
                                          data=data, files={"document": fh}, timeout=600)
                if r.status_code == 200:
                    self._log_artifact(chat_id, "document", _orig, caption)
                    return True
                tg_bot.logger.warning("sendDocument HTTP %s attempt %d: %s",
                               r.status_code, tries, r.text[:200])
                # Markup the caption cannot carry must not cost the user the FILE.
                # Retrying the identical body just fails three times; strip the
                # formatting and send the document anyway.
                if not degraded and "parse_mode" in data and tg_bot._is_caption_parse_error(r):
                    tg_bot.logger.warning("sendDocument rejected caption markup — "
                                   "resending the file with a plain caption")
                    degraded = True
                    data.pop("parse_mode", None)
                    data["caption"] = tg_bot._html_to_plain(caption)[:tg_bot._MAX_CAPTION]
                    tries -= 1        # our own bad caption, not a transport failure:
                    continue          # do not spend a retry (or a backoff) on it
            except Exception as exc:
                tg_bot.logger.warning("sendDocument attempt %d error: %s", tries, exc)
            if tries < tg_bot._API_RETRIES:
                time.sleep(1.5 ** (tries - 1))
        return False

    def _send_audio(self, chat_id, path: str, caption: str = "") -> bool:
        """Send a generated song as a playable Telegram audio file.

        Mirrors _send_document's plain retry loop rather than _send_video's
        (no re-encode/thumbnail machinery — a wav from music.generate_music
        needs neither). Returns True only on an accepted upload, same
        "false means tell the user it failed" contract as the other media
        senders.
        """
        if not (path and os.path.exists(path)):
            tg_bot.logger.error("[audio delivery] no file to send: %r", path)
            return False
        _orig, path = path, self._archive(chat_id, path)   # the log keeps the original
        data: dict = {"chat_id": chat_id}
        if caption:
            data["caption"] = tg_bot._safe_caption(caption)
            data["parse_mode"] = "HTML"
        tries, degraded = 0, False
        while tries < tg_bot._API_RETRIES:
            tries += 1
            try:
                with open(path, "rb") as fh:
                    r = _logged_post(f"{self._api}/sendAudio",
                                      data=data, files={"audio": fh}, timeout=180)
                if r.status_code == 200:
                    self._log_artifact(chat_id, "music", _orig, caption)
                    return True
                tg_bot.logger.warning("sendAudio HTTP %s attempt %d: %s",
                               r.status_code, tries, r.text[:200])
                if not degraded and "parse_mode" in data and tg_bot._is_caption_parse_error(r):
                    tg_bot.logger.warning("sendAudio rejected caption markup — "
                                   "resending the file with a plain caption")
                    degraded = True
                    data.pop("parse_mode", None)
                    data["caption"] = tg_bot._html_to_plain(caption)[:tg_bot._MAX_CAPTION]
                    tries -= 1
                    continue
            except Exception as exc:
                tg_bot.logger.warning("sendAudio attempt %d error: %s", tries, exc)
            if tries < tg_bot._API_RETRIES:
                time.sleep(1.5 ** (tries - 1))
        return False

    def _send_video(self, chat_id, path: str, caption: str = "",
                    keyboard=None, ctx=None) -> bool:
        """Send a generated clip as a playable Telegram video.

        Returns True only if Telegram actually accepted it. That matters more here
        than anywhere else: a clip costs many minutes of GPU, and the failure mode
        this guards against is the bot cheerfully announcing a video it never sent
        (the same class of bug as the image-delivery skips). The caller must treat
        False as "tell the user it could not be delivered".

        sendVideo caps bot uploads at 50MB, so an over-size clip is re-encoded
        first; if it still will not fit, it goes as a document rather than being
        dropped — a file the user can download beats nothing.
        """
        import nice_names
        self._archive(chat_id, path)   # archive copy in runtime/generated/
        if not (path and os.path.exists(path)):
            tg_bot.logger.error("[video delivery] no file to send: %r", path)
            return False

        import video as _vid
        send_path = path
        try:
            send_path = _vid.fit_for_telegram(path, ctx=ctx)
        except Exception as exc:
            tg_bot.logger.warning("[video delivery] could not fit %s for Telegram: %s", path, exc)

        info = {}
        try:
            info = _vid.probe(send_path)
        except Exception:
            pass

        data: dict = {"chat_id": chat_id, "supports_streaming": True}
        if caption:
            data["caption"] = tg_bot._safe_caption(caption)
            data["parse_mode"] = "HTML"
        # Telegram renders the clip much better when told its real shape/length,
        # otherwise it can show a black 0:00 placeholder until the user taps it.
        if info.get("width"):
            data["width"] = int(info["width"])
            data["height"] = int(info["height"])
        if info.get("seconds"):
            data["duration"] = int(round(float(info["seconds"])))
        if keyboard is not None:
            data["reply_markup"] = json.dumps(keyboard)

        thumb_path = None
        try:
            thumb_path = _vid.make_thumbnail(send_path)
        except Exception:
            thumb_path = None

        try:
            tries, degraded = 0, False
            while tries < tg_bot._API_RETRIES:
                tries += 1
                files = {}
                fhs = []
                try:
                    fh = open(send_path, "rb")
                    fhs.append(fh)
                    files["video"] = (nice_names.display_name(path, nice_names.TITLES.get(chat_id, "")), fh)
                    if thumb_path and os.path.exists(thumb_path):
                        tfh = open(thumb_path, "rb")
                        fhs.append(tfh)
                        files["thumbnail"] = tfh
                    r = _logged_post(f"{self._api}/sendVideo", data=data,
                                      files=files, timeout=600)
                    if r.status_code == 200:
                        # Log the ORIGINAL render, not send_path: a re-encoded
                        # Telegram-fit copy is a scratch file, deleted in the
                        # `finally` below.
                        self._log_artifact(chat_id, "video", path, caption)
                        return True
                    tg_bot.logger.warning("sendVideo HTTP %s attempt %d: %s",
                                   r.status_code, tries, r.text[:200])
                    if r.status_code == 413 or "too big" in (r.text or "").lower():
                        break          # retrying the same oversize upload is pointless
                    # A caption is decoration; the CLIP cost minutes of GPU. Never
                    # let bad markup on the former throw away the latter.
                    if not degraded and "parse_mode" in data and tg_bot._is_caption_parse_error(r):
                        tg_bot.logger.warning("sendVideo rejected caption markup — "
                                       "resending the clip with a plain caption")
                        degraded = True
                        data.pop("parse_mode", None)
                        data["caption"] = tg_bot._html_to_plain(caption)[:tg_bot._MAX_CAPTION]
                        tries -= 1
                        continue
                except Exception as exc:
                    tg_bot.logger.warning("sendVideo attempt %d error: %s", tries, exc)
                finally:
                    for f in fhs:
                        try:
                            f.close()
                        except Exception:
                            pass
                if tries < tg_bot._API_RETRIES:
                    time.sleep(1.5 ** (tries - 1))

            tg_bot.logger.warning("[video delivery] sendVideo failed; falling back to "
                           "sendDocument for %s", send_path)
            if self._send_document(chat_id, send_path, caption):
                return True
            tg_bot.logger.error("[video delivery] sendVideo+sendDocument both failed: %s",
                         send_path)
            return False
        finally:
            # The re-encoded copy and the poster frame are scratch files; the
            # ORIGINAL render is the user's and is never deleted here.
            for tmp in (thumb_path, send_path if send_path != path else None):
                if tmp and os.path.exists(tmp):
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass

    def _delete(self, chat_id, msg_id):
        self._api_post("deleteMessage", {"chat_id": chat_id, "message_id": msg_id})

    def _scrub_secret(self, chat_id: int, msg: dict) -> None:
        """Delete a message the user sent that contained a plaintext password.

        Registration, login and 🔑 Change password all ask the user to TYPE their
        password into the chat. Telegram keeps that message forever, on their device
        and on their other sessions, in the clear — anyone who later picks up the
        unlocked phone can scroll up and read it. Best-effort: deletion needs the
        message to be under 48h old and can simply fail, which must never break the
        flow that was in progress.
        """
        if isinstance(msg, dict):
            msg["_secret"] = True       # chatlog writes a placeholder, not the password
        mid = (msg or {}).get("message_id")
        if not mid:
            return
        try:
            self._delete(chat_id, mid)
        except Exception as exc:
            tg_bot.logger.debug("could not scrub secret message %s: %s", mid, exc)

    def _send_photo(self, chat_id, path, keyboard=None) -> bool:
        """Send an image as a Telegram photo.

        Telegram sendPhoto rejects files > 10 MB or larger than 10000px on a
        side. For large upscaled images we shrink to a Telegram-friendly copy
        (max 4096px, max 8 MB JPEG) before sending; if the shrunk copy still
        fails we fall back to sendDocument (50 MB limit).  Returns True on
        success, False if all attempts fail.
        """
        self._archive(chat_id, path)   # archive copy in runtime/generated/
        _TG_PHOTO_LIMIT = 8 * 1024 * 1024   # 8 MB hard limit for sendPhoto
        _TG_DIM_LIMIT   = 4096

        # Build a Telegram-safe copy of the file when needed.
        send_path = path
        tmp_path  = None
        try:
            file_size = os.path.getsize(path)
            from PIL import Image as _PILImage
            with _PILImage.open(path) as _probe:
                _pw, _ph = _probe.size
            # Telegram also rejects width + height > 10000 px, whatever the bytes.
            if file_size > _TG_PHOTO_LIMIT or _pw + _ph > 10000:
                try:
                    img = _PILImage.open(path)
                    if img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")        # RGBA/P cannot be saved as JPEG
                    w, h = img.size
                    # Scale so the longest side is at most 4096px
                    if max(w, h) > _TG_DIM_LIMIT:
                        ratio = _TG_DIM_LIMIT / max(w, h)
                        img = img.resize((int(w * ratio), int(h * ratio)),
                                         _PILImage.LANCZOS)
                    # Save as progressive JPEG; try quality 90 first, drop if still big
                    import tempfile
                    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
                    os.close(tmp_fd)
                    for quality in (90, 80, 70, 60):
                        img.save(tmp_path, "JPEG", quality=quality, optimize=True)
                        if os.path.getsize(tmp_path) <= _TG_PHOTO_LIMIT:
                            break
                    send_path = tmp_path
                    tg_bot.logger.info("sendPhoto: shrunk %.1f MB → %.1f MB for Telegram",
                                file_size / 1e6, os.path.getsize(tmp_path) / 1e6)
                except Exception as exc:
                    tg_bot.logger.warning("sendPhoto: could not shrink image: %s", exc)
                    # fall through — try sending the original; may fail on size
        except Exception:
            pass

        data: dict = {"chat_id": chat_id}
        if keyboard:
            data["reply_markup"] = json.dumps(keyboard)

        try:
            return self._send_photo_file(chat_id, path, send_path, data)
        finally:
            if tmp_path:                # the shrunk copy, sent or not
                try: os.unlink(tmp_path)
                except Exception: pass

    def _send_photo_file(self, chat_id, path, send_path, data) -> bool:
        # --- primary: sendPhoto ---
        for attempt in range(tg_bot._API_RETRIES):
            try:
                with open(send_path, "rb") as fh:
                    r = _logged_post(f"{self._api}/sendPhoto",
                                      data=data, files={"photo": fh}, timeout=120)
                if r.status_code == 200:
                    # Remember WHICH message this picture became: a Telegram reply
                    # carries the message id, and that is what turns "this one" into
                    # a file path instead of a guess.
                    self._last_photo_msg_id = tg_bot._msg_id_of(r)
                    return True
                tg_bot.logger.warning("sendPhoto HTTP %s attempt %d: %s",
                               r.status_code, attempt, r.text[:200])
                # 413 / "file is too big" → don't retry sendPhoto, go to document
                if r.status_code in (400, 413) and any(
                        k in r.text.lower() for k in ("too big", "dimensions", "photo_invalid")):
                    break
            except Exception as exc:
                tg_bot.logger.warning("sendPhoto attempt %d error: %s", attempt, exc)
            if attempt < tg_bot._API_RETRIES - 1:
                time.sleep(1.5 ** attempt)

        # --- fallback: sendDocument (50 MB limit, no compression) ---
        tg_bot.logger.warning("sendPhoto failed; falling back to sendDocument for %s", path)
        try:
            with open(path, "rb") as fh:
                r = _logged_post(f"{self._api}/sendDocument",
                                  data=data, files={"document": fh}, timeout=180)
            if r.status_code == 200:
                tg_bot.logger.info("sendDocument fallback succeeded for %s", path)
                self._last_photo_msg_id = tg_bot._msg_id_of(r)
                return True
            tg_bot.logger.error("sendDocument fallback HTTP %s: %s", r.status_code, r.text[:200])
        except Exception as exc:
            tg_bot.logger.error("sendDocument fallback error: %s", exc)
        return False

    def _send_voice_from_wav(self, chat_id: int, wav_path: str,
                             keyboard=None) -> bool:
        """Convert an existing WAV to OGG/Opus and send as a Telegram voice note.

        Attaches *keyboard* (reply_markup) to the voice message when provided,
        so callers can skip sending a separate text message just to show buttons.
        Returns True on success, False on failure.
        """
        ogg_path = None
        try:
            ogg_path = wav_path.rsplit(".", 1)[0] + "_tg.ogg"
            proc = subprocess.run(
                ["ffmpeg", "-y", "-i", wav_path,
                 "-c:a", "libopus", "-b:a", "64k", ogg_path],
                capture_output=True, timeout=60,
            )
            if proc.returncode != 0:
                stderr = proc.stderr.decode(errors="replace")
                tg_bot.logger.error("voice_from_wav: ffmpeg rc=%d: %s",
                             proc.returncode, stderr[-400:])
                self._activity.log(chat_id, "error",
                    f"[voice] ffmpeg failed rc={proc.returncode}: {stderr[-200:]}")
                return False
            if not os.path.exists(ogg_path) or os.path.getsize(ogg_path) < 64:
                tg_bot.logger.error("voice_from_wav: OGG missing/empty at %s", ogg_path)
                return False
            self._api_post("sendChatAction", {"chat_id": chat_id, "action": "record_voice"})
            data: dict = {"chat_id": chat_id}
            if keyboard:
                data["reply_markup"] = json.dumps(keyboard)
            for attempt in range(tg_bot._API_RETRIES):
                try:
                    with open(ogg_path, "rb") as fh:
                        r = _logged_post(f"{self._api}/sendVoice",
                                          data=data,
                                          files={"voice": ("voice.ogg", fh, "audio/ogg")},
                                          timeout=120)
                    if r.status_code == 200:
                        return True
                    tg_bot.logger.warning("sendVoice HTTP %s attempt %d: %s",
                                   r.status_code, attempt, r.text[:300])
                    self._activity.log(chat_id, "error",
                        f"[voice] sendVoice HTTP {r.status_code}: {r.text[:200]}")
                except Exception as exc:
                    tg_bot.logger.warning("sendVoice attempt %d error: %s", attempt, exc)
                if attempt < tg_bot._API_RETRIES - 1:
                    time.sleep(1.5 ** attempt)
            tg_bot.logger.error("voice_from_wav: sendVoice failed after %d attempts", tg_bot._API_RETRIES)
            return False
        except Exception:
            tg_bot.logger.exception("voice_from_wav unexpected error chat=%s", chat_id)
            return False
        finally:
            if ogg_path:
                try:
                    if os.path.exists(ogg_path): os.unlink(ogg_path)
                except Exception: pass

    def _send_voice(self, ctx, chat_id, text) -> bool:
        """Synthesise text to WAV, convert to OGG/Opus, send as a voice note.

        Returns True only when Telegram accepted the voice message — the caller
        falls back to plain text on False, so a TTS failure never swallows the
        reply."""
        ogg_path = None
        try:
            from audio import synth_single_segment
            from graph import spoken_text
            text = spoken_text(ctx, text)
            sess = self._get_session(chat_id)
            ref = getattr(sess, "assistant_ref", "")
            if ref and os.path.exists(ref):
                from voice_clone import _CloneCtx
                wav_path = synth_single_segment(_CloneCtx(ctx, ref, sess.assistant_ref_text),
                                                0, "clone", text)
            else:
                wav_path = synth_single_segment(ctx, 0, "default", text)
            if not wav_path or not os.path.exists(wav_path):
                tg_bot.logger.warning("sendVoice: synth returned no WAV (path=%s)", wav_path)
                return False

            wav_size = os.path.getsize(wav_path)
            if wav_size == 0:
                tg_bot.logger.warning("sendVoice: WAV file is empty at %s", wav_path)
                return False

            # Convert WAV → OGG/Opus (Telegram voice note requirement)
            ogg_path = wav_path.replace(".wav", "_tg.ogg")
            proc = subprocess.run(
                ["ffmpeg", "-y", "-i", wav_path,
                 "-c:a", "libopus", "-b:a", "64k", ogg_path],
                capture_output=True, timeout=60,
            )
            if proc.returncode != 0:
                stderr = proc.stderr.decode(errors="replace")
                tg_bot.logger.error("sendVoice: ffmpeg failed (rc=%d):\n%s",
                             proc.returncode, stderr[-800:])
                return False

            if not os.path.exists(ogg_path) or os.path.getsize(ogg_path) < 64:
                tg_bot.logger.error("sendVoice: OGG missing or empty after ffmpeg at %s", ogg_path)
                return False

            tg_bot.logger.debug("sendVoice: WAV %d bytes → OGG %d bytes",
                         wav_size, os.path.getsize(ogg_path))

            self._api_post("sendChatAction",
                           {"chat_id": chat_id, "action": "record_voice"})

            for attempt in range(tg_bot._API_RETRIES):
                try:
                    with open(ogg_path, "rb") as fh:
                        r = _logged_post(
                            f"{self._api}/sendVoice",
                            data={"chat_id": chat_id},
                            files={"voice": ("voice.ogg", fh, "audio/ogg")},
                            timeout=120)
                    if r.status_code == 200:
                        tg_bot.logger.debug("sendVoice: delivered to chat=%s", chat_id)
                        return True
                    tg_bot.logger.warning("sendVoice HTTP %s attempt %d: %s",
                                   r.status_code, attempt, r.text[:300])
                except Exception as exc:
                    tg_bot.logger.warning("sendVoice attempt %d error: %s", attempt, exc)
                if attempt < tg_bot._API_RETRIES - 1:
                    time.sleep(1.5 ** attempt)
            tg_bot.logger.error("sendVoice: failed after %d attempts for chat=%s",
                         tg_bot._API_RETRIES, chat_id)
            return False

        except Exception:
            tg_bot.logger.exception("sendVoice unexpected error chat=%s", chat_id)
            return False
        finally:
            if ogg_path:
                try:
                    if os.path.exists(ogg_path): os.unlink(ogg_path)
                except Exception: pass

    def _api_post(self, method: str, payload: dict = None, **kw) -> dict:
        url = f"{self._api}/{method}"
        for attempt in range(tg_bot._API_RETRIES):
            try:
                r = _logged_post(url, json=payload or {},
                                  timeout=tg_bot._API_TIMEOUT, **kw)
                body = r.json()
                if r.status_code == 429:
                    retry_after = (body.get("parameters") or {}).get("retry_after", 30)
                    tg_bot.logger.warning("API %s 429 rate-limit; sleeping %ss", method, retry_after)
                    time.sleep(retry_after); continue
                return body
            except Exception as exc:
                if attempt < tg_bot._API_RETRIES - 1: time.sleep(1.5 ** attempt)
                else: tg_bot.logger.warning("API %s failed: %s", method, exc)
        return {}

    def _api_get(self, method: str, params: dict = None) -> dict:
        url = f"{self._api}/{method}"
        for attempt in range(tg_bot._API_RETRIES):
            try:
                r = requests.get(url, params=params or {}, timeout=tg_bot._API_TIMEOUT)
                if r.status_code == 429:
                    body = r.json()
                    retry_after = (body.get("parameters") or {}).get("retry_after", 30)
                    tg_bot.logger.warning("API GET %s 429; sleeping %ss", method, retry_after)
                    time.sleep(retry_after); continue
                return r.json()
            except Exception as exc:
                if attempt < tg_bot._API_RETRIES - 1: time.sleep(1.5 ** attempt)
                else: tg_bot.logger.warning("API GET %s failed: %s", method, exc)
        return {}

    def _get_me(self) -> Optional[dict]:
        r = self._api_get("getMe")
        return r.get("result") if r.get("ok") else None

    def _identify(self) -> tuple:
        """(info, error) — who we are, or why we cannot start.

        "Bad token or no network" told an operator nothing: those two need
        completely different fixes, and one of them is not their fault. _api_get
        already distinguishes them — it returns {} when the request never got an
        answer, and Telegram's own {ok: false, error_code, description} when it
        did — so the caller can say which.
        """
        r = self._api_get("getMe")
        if r.get("ok"):
            return (r.get("result") or {}), ""
        if not r:
            return None, ("❌  Telegram is unreachable — check the network, "
                          "VPN or proxy, then press Start again")
        code = r.get("error_code")
        desc = (r.get("description") or "")[:120]
        if code == 401:
            return None, ("❌  Telegram rejected this token (401) — paste a fresh "
                          "one from @BotFather")
        return None, f"❌  Telegram refused to start the bot ({code}): {desc}"


import tg_bot  # noqa: E402  (cycle by design; attrs read at call time)

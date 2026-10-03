"""Drive the REAL Telegram bot with fake users, and keep everything it sends.

The whole stack is real -- `TelegramBot` polling, debounce, consumers, the
graph, LM Studio, ComfyUI, Whisper and F5 -- except `api.telegram.org`, which
is answered from this process. Every outbound call is recorded per chat and
every uploaded file (photo, voice, audio, document) is kept on disk, so a
scenario can look at the picture, transcribe the voice note back with the
bot's own Whisper, and time each turn.

This is not a test; it makes real GPU calls and must not run under run_all.
Stop the desktop app first (it holds Whisper/F5 and the bot's token):

    venv/Scripts/python.exe bench/live_tg_drive.py --scenario music,picture,chat

Results land in runtime/live_drive/<stamp>/ : transcript.jsonl, one folder of
media per chat, and a summary printed at the end.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# bench/ holds scripts named like app modules (knowledge.py): keep it off the path
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]
sys.path.insert(0, ROOT)
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

if os.getenv("F5_TEST_RUN"):
    raise SystemExit("live_tg_drive makes real GPU calls; unset F5_TEST_RUN")

import logging
logging.basicConfig(level=logging.INFO, filename=str(Path(ROOT) / "runtime" / "live_drive_app.log"),
                    filemode="a", encoding="utf-8",
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
for _noisy in ("urllib3", "httpx", "httpcore", "PIL", "numba", "faster_whisper", "matplotlib"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

import requests  # noqa: E402

STAMP = time.strftime("%Y%m%d_%H%M%S")
OUT = Path(ROOT) / "runtime" / "live_drive" / STAMP
OUT.mkdir(parents=True, exist_ok=True)
DATA = OUT / "tg_data"

_real_post, _real_get = requests.post, requests.get
# Whatever base the bot is configured for -- the cloud API or a self-hosted
# telegram-bot-api (TG_API_BASE) -- is answered from here; the fake token
# must never reach a real server of either kind.
import config as _config  # noqa: E402
_TG_BASES = ("https://api.telegram.org", _config.TG_API_BASE)


class Wire:
    """The fake Telegram: an inbox of updates, and everything the bot sent."""

    def __init__(self):
        self.lock = threading.Lock()
        self.inbox: list[dict] = []
        self.next_update = 1
        self.next_msg = 1000
        self.sent: list[dict] = []          # every outbound call, in order
        self.files: dict[str, Path] = {}    # file_id -> local path (inbound)
        self.msgs: dict[int, dict] = {}     # message_id -> the message as Telegram holds it

        self.log = io.open(OUT / "transcript.jsonl", "a", encoding="utf-8")

    # -- what the bot sees --------------------------------------------------
    def push(self, upd: dict) -> None:
        with self.lock:
            upd = dict(upd, update_id=self.next_update)
            self.next_update += 1
            self.inbox.append(upd)

    def _drain(self, offset: int) -> list:
        with self.lock:
            out = [u for u in self.inbox if u["update_id"] >= offset]
            self.inbox = [u for u in self.inbox if u["update_id"] >= offset]
            return out

    def new_id(self) -> int:
        """One sequence for the whole chat, users and bot alike, as in Telegram."""
        with self.lock:
            self.next_msg += 1
            return self.next_msg

    def _record(self, method: str, chat_id, payload: dict, path: str = "") -> dict:
        if method.startswith("edit"):         # an edit keeps its message and its id
            mid = int(payload.get("message_id") or 0)
            ev = {"t": round(time.time(), 3), "method": method, "chat_id": chat_id,
                  "message_id": mid, "payload": payload, "file": path}
            with self.lock:
                self.sent.append(ev)
                if mid in self.msgs and payload.get("text"):
                    self.msgs[mid]["text"] = payload["text"]
            return {"ok": True, "result": {"message_id": mid, "chat": {"id": chat_id}}}
        mid = self.new_id()
        m = {"message_id": mid, "chat": {"id": chat_id, "type": "private"}, "date": int(time.time()),
             "from": {"id": 1, "is_bot": True, "first_name": "drive"}}
        if payload.get("text") or payload.get("caption"):
            m["text" if method == "sendMessage" else "caption"] = payload.get("text") or payload.get("caption")
        if method == "sendPhoto":
            fid = f"botphoto_{mid}"; m["photo"] = [{"file_id": fid, "width": 1024, "height": 1024}]
            if path: self.files[fid] = Path(path)
        rm = payload.get("reply_markup")
        if rm: m["reply_markup"] = json.loads(rm) if isinstance(rm, str) else rm
        self.msgs[mid] = m
        ev = {"t": round(time.time(), 3), "method": method, "chat_id": chat_id,
              "message_id": mid, "payload": payload, "file": path}
        with self.lock:
            self.sent.append(ev)
            self.log.write(json.dumps(ev, ensure_ascii=False) + "\n"); self.log.flush()
        txt = payload.get("text") or payload.get("caption") or ""
        print(f"  <- [{chat_id}] {method} {txt[:160]!r} {('FILE ' + path) if path else ''}")
        return {"ok": True, "result": {"message_id": mid, "chat": {"id": chat_id}}}

    # -- requests.* by URL ---------------------------------------------------
    def post(self, url, *a, **kw):
        if not url.startswith(_TG_BASES):
            return _real_post(url, *a, **kw)
        method = url.rsplit("/", 1)[-1]
        payload = dict(kw.get("json") or kw.get("data") or {})
        chat_id = payload.get("chat_id")
        path = ""
        files = kw.get("files") or {}
        for kind, fh in sorted(files.items(), key=lambda kv: kv[0] == "thumbnail", reverse=True):
            # the thumbnail goes FIRST so the real media (video) is the path kept
            name, blob = kind, fh
            if isinstance(fh, tuple):
                name, blob = fh[0], fh[1]
            data = blob.read() if hasattr(blob, "read") else blob
            d = OUT / f"chat_{chat_id}"; d.mkdir(exist_ok=True)
            ext = os.path.splitext(str(getattr(blob, "name", name)))[1] or {
                "photo": ".png", "voice": ".ogg", "audio": ".mp3",
                "video": ".mp4", "document": ".bin"}.get(kind, ".bin")
            p = d / f"{int(time.time()*1000)}_{kind}{ext}"
            p.write_bytes(data); path = str(p)
        if method == "getMe":
            return _Resp({"ok": True, "result": {"id": 1, "username": "live_drive_bot",
                                                 "first_name": "drive"}})
        if method == "answerCallbackQuery":
            return _Resp({"ok": True, "result": True})
        if method == "deleteMessage":        # recorded quietly: a viewer needs to see it vanish
            self.msgs.pop(int(payload.get("message_id") or 0), None)
            with self.lock:
                self.sent.append({"t": round(time.time(), 3), "method": method, "chat_id": chat_id,
                                  "message_id": 0, "payload": payload, "file": ""})
            return _Resp({"ok": True, "result": True})
        if method in ("editMessageReplyMarkup", "sendChatAction", "setMyCommands"):
            return _Resp({"ok": True, "result": True})
        return _Resp(self._record(method, chat_id, payload, path))

    def get(self, url, *a, **kw):
        if not url.startswith(_TG_BASES):
            return _real_get(url, *a, **kw)
        if "/file/bot" in url:
            fid = url.rsplit("/", 1)[-1]
            p = self.files.get(fid)
            return _Resp(None, content=p.read_bytes() if p else b"", status=200 if p else 404)
        method = url.rsplit("/", 1)[-1]
        params = kw.get("params") or {}
        if method == "getMe":
            return _Resp({"ok": True, "result": {"id": 1, "username": "live_drive_bot",
                                                 "first_name": "drive"}})
        if method == "getUpdates":
            deadline = time.time() + 0.4
            while time.time() < deadline:
                out = self._drain(int(params.get("offset") or 0))
                if out:
                    return _Resp({"ok": True, "result": out})
                time.sleep(0.05)
            return _Resp({"ok": True, "result": []})
        if method == "getFile":
            fid = params.get("file_id")
            return _Resp({"ok": True, "result": {"file_id": fid, "file_path": fid}})
        return _Resp({"ok": True, "result": {}})


class _Resp:
    def __init__(self, body, content=b"", status=200):
        self._body, self.content, self.status_code = body, content, status
        self.text = json.dumps(body) if body is not None else ""
    def json(self):
        return self._body


WIRE = Wire()
requests.post = WIRE.post
requests.get = WIRE.get


class Chat:
    """One fake user."""

    def __init__(self, bot, chat_id: int, name: str, lang: str = "ru"):
        self.bot, self.id, self.name, self.lang = bot, chat_id, name, lang
        self._seen = 0
        # Approved straight away: registration is covered by the suites.
        import tg_bot as T
        u = T._User(chat_id=chat_id, name=name, tg_username=name.lower(),
                    password_hash="", status="approved")
        bot._user_store.put(u)
        s = bot._get_session(chat_id); s.lang = lang; bot._store.put(s)

    def _msg(self, reply_to: int = 0, forward_from: str = "", ago: int = 0, **fields) -> dict:
        """A message as Telegram delivers it. reply_to: a message id in this chat;
        forward_from: someone else's name (their words, forwarded); ago: seconds
        in the past (a backlog while the bot was down)."""
        m = {"message_id": WIRE.new_id(),
             "chat": {"id": self.id, "type": "private"},
             "from": {"id": self.id, "username": self.name.lower(),
                      "first_name": self.name, "language_code": self.lang},
             "date": int(time.time()) - int(ago)}
        m.update(fields)
        if reply_to and reply_to in WIRE.msgs:
            m["reply_to_message"] = dict(WIRE.msgs[reply_to])
        if forward_from:
            fid = 5550000 + sum(map(ord, forward_from)) % 9999
            who = {"id": fid, "is_bot": False, "first_name": forward_from}
            m["forward_origin"] = {"type": "user", "date": m["date"] - 600, "sender_user": who}
            m["forward_from"], m["forward_date"] = who, m["date"] - 600
        WIRE.msgs[m["message_id"]] = m
        with WIRE.lock:        # the viewer shows what the user sent, under its real id
            WIRE.sent.append({"t": round(time.time(), 3), "method": "user", "chat_id": self.id,
                              "message_id": m["message_id"], "payload": {k: v for k, v in m.items()
                                                                          if k not in ("chat", "from")},
                              "file": ""})
        return m

    def say(self, text: str, **kw) -> int:
        print(f"  -> [{self.id} {self.name}] {text!r} {kw or ''}")
        m = self._msg(text=text, **kw)
        WIRE.push({"message": m})
        return m["message_id"]

    def voice(self, path: str, forwarded: bool = False) -> None:
        fid = f"voice_{self.id}_{int(time.time()*1000)}.ogg"
        WIRE.files[fid] = Path(path)
        print(f"  -> [{self.id} {self.name}] VOICE {path} {'(forwarded)' if forwarded else ''}")
        m = self._msg(voice={"file_id": fid, "duration": 3, "mime_type": "audio/ogg"})
        if forwarded:
            m["forward_from"] = {"id": 5550001, "first_name": "Serezha"}
            m["forward_date"] = int(time.time()) - 600
        WIRE.push({"message": m})

    def video_note(self, path: str, forwarded: bool = False, seconds: int = 0) -> None:
        fid = f"vnote_{self.id}_{int(time.time()*1000)}.mp4"
        WIRE.files[fid] = Path(path)
        print(f"  -> [{self.id} {self.name}] VIDEO NOTE {path} {'(forwarded)' if forwarded else ''}")
        m = self._msg(video_note={"file_id": fid, "duration": seconds or 10, "length": 240})
        if forwarded:
            m["forward_from"] = {"id": 5550002, "first_name": "Darya"}
            m["forward_date"] = int(time.time()) - 600
        WIRE.push({"message": m})

    def document(self, path: str, caption: str = "", self_forward: bool = False) -> None:
        fid = f"doc_{self.id}_{int(time.time()*1000)}"
        WIRE.files[fid] = Path(path)
        print(f"  -> [{self.id} {self.name}] DOCUMENT {path} {caption!r}"
              f"{' (self-forwarded)' if self_forward else ''}")
        import mimetypes
        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
        m = self._msg(document={"file_id": fid, "file_name": os.path.basename(path),
                                "mime_type": mime, "file_size": os.path.getsize(path)})
        if caption: m["caption"] = caption
        if self_forward:
            # Telegram marks a message forwarded from Saved Messages with the
            # user's own id (live 2026-09-14 19:29).
            m["forward_origin"] = {"type": "user", "date": int(time.time()) - 60,
                                   "sender_user": dict(m["from"])}
            m["forward_from"] = dict(m["from"]); m["forward_date"] = int(time.time()) - 60
        WIRE.push({"message": m})

    def photo(self, path: str, caption: str = "", group: str = "", **kw) -> dict:
        fid = f"photo_{self.id}_{int(time.time()*1000)}_{os.path.basename(path)}"
        WIRE.files[fid] = Path(path)
        print(f"  -> [{self.id} {self.name}] PHOTO {path} {caption!r} {kw or ''}")
        # Telegram sends a small and a large size; the bot must take the last.
        extra = {"media_group_id": group} if group else {}
        if caption: extra["caption"] = caption
        m = self._msg(photo=[{"file_id": fid + "_s", "width": 90, "height": 90, "file_size": 1},
                             {"file_id": fid, "width": 1024, "height": 1024, "file_size": 1}], **extra, **kw)
        WIRE.files[fid + "_s"] = Path(path)
        WIRE.push({"message": m})
        return m

    def album(self, paths: list, caption: str = "", **kw) -> list:
        """Several photos in one send: one update each, one media_group_id, caption on the first."""
        g = str(WIRE.new_id())
        return [self.photo(p, caption if i == 0 else "", group=g, **kw) for i, p in enumerate(paths)]

    def press(self, data: str, message_id: int = 0) -> None:
        print(f"  -> [{self.id} {self.name}] PRESS {data!r}")
        WIRE.push({"callback_query": {
            "id": str(int(time.time() * 1000)), "from": {"id": self.id, "username": self.name.lower()},
            "data": data,
            "message": WIRE.msgs.get(message_id) or {"message_id": message_id or 1,
                                                    "chat": {"id": self.id, "type": "private"}}}})

    # -- what came back ------------------------------------------------------
    def new(self) -> list[dict]:
        evs = [e for e in WIRE.sent if e["chat_id"] == self.id and e["method"] != "user"]
        out, self._seen = evs[self._seen:], len(evs)
        return out

    def wait(self, *, until=None, timeout: float = 600, quiet: float = 8.0) -> list[dict]:
        """Wait for the turn to finish: `until(ev)` true, or `quiet` s of silence
        after at least one message. Returns the new events."""
        t0 = time.time(); got = []
        last = None
        while time.time() - t0 < timeout:
            new = self.new()
            if new:
                got.extend(new); last = time.time()
                if until and any(until(e) for e in new):
                    break
            if last and time.time() - last > quiet and not self.busy():
                break
            time.sleep(0.5)
        return got

    def busy(self) -> bool:
        b = self.bot
        with b._task_lock:
            return bool(b._chat_busy.get(self.id, 0)) or bool(b._running_task.get(self.id))


def final_text(evs) -> str:
    return "\n".join(e["payload"].get("text") or e["payload"].get("caption") or ""
                     for e in evs if e["method"] in ("sendMessage", "sendPhoto", "sendVoice",
                                                     "sendAudio", "sendDocument"))


def files_of(evs, kind: str) -> list[str]:
    return [e["file"] for e in evs if e["method"] == kind and e["file"]]


def build(model: str | None = None):
    import tg_bot as T
    T.redirect_data_dir(DATA)
    from app_runtime import build_runtime
    from config import MODEL_NAME
    ctx, base, graph = build_runtime(model or MODEL_NAME, False, status_cb=print)
    import copy
    bot = T.TelegramBot("1:LIVE", lambda: ctx, lambda: graph,
                        lambda: copy.deepcopy(base),
                        on_status=lambda s: print("   status:", s),
                        on_stage=lambda cid, st, *_: print(f"   stage[{cid}]: {st}") if st else None,
                        silent_mode=True)
    assert bot.start(), "bot did not start"
    return bot, ctx


def transcribe(ctx, path: str) -> str:
    from audio import transcribe_audio_file
    return transcribe_audio_file(ctx, path)

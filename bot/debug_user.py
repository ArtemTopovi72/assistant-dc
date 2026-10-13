"""🧪 A debug user that talks to the RUNNING bot: its messages join the same queue as
everyone's, what the bot sends it lands in a file instead of Telegram. Lets an outside
driver (bench/debug_user.py) use the app as a user would, without stopping it, and
see how the queue copes when real people write at the same time.

Off unless runtime/debug_user/enabled exists.

  inbox/*.json   one Telegram-shaped update each, picked up and deleted
  files/         what the user "sends"; a file_id "dbg:<name>" points here
  outbox.jsonl   every send/edit to the debug chat, a line each
  out/           the files the bot sent, copied
"""
import itertools
import json
import os
import shutil
import threading
import time
import uuid

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "runtime", "debug_user")
CHAT_ID = 777000001             # no real Telegram chat has this id
# Debug users 0..4 are chats CHAT_ID..CHAT_ID+4: several at once show how the queue
# shares itself out.
USERS = 5
_LOCK = threading.Lock()
_IDS = itertools.count(int(time.time()) % 1_000_000 * 100)


def enabled() -> bool:
    return os.path.exists(os.path.join(ROOT, "enabled"))


def is_debug(chat_id) -> bool:
    try:
        return CHAT_ID <= int(chat_id) < CHAT_ID + USERS
    except (TypeError, ValueError):
        return False


def pull() -> list:
    """The waiting updates, oldest first; each file is taken once."""
    box = os.path.join(ROOT, "inbox")
    if not os.path.isdir(box):
        return []
    out = []
    for name in sorted(os.listdir(box)):
        if not name.endswith(".json"):
            continue
        p = os.path.join(box, name)
        try:
            with open(p, encoding="utf-8") as fh:
                out.append(json.load(fh))
        except (OSError, ValueError):
            continue            # still being written; next round
        try:
            os.remove(p)
        except OSError:
            pass
    return out


def file_path(file_id: str) -> str:
    """The local file behind a "dbg:<name>" file_id, only inside files/ or out/."""
    name = os.path.basename(file_id[4:])
    for sub in ("files", "out"):
        p = os.path.join(ROOT, sub, name)
        if os.path.exists(p):
            return p
    return ""


def record(url: str, kw: dict):
    """A send to the debug chat: written to the outbox, answered like Telegram would."""
    method = url.rsplit("/", 1)[-1]
    payload = kw.get("json") if kw.get("json") is not None else kw.get("data") or {}
    payload = {k: (v if isinstance(v, (str, int, float, bool, list, dict, type(None))) else str(v))
               for k, v in dict(payload).items()}
    mid = next(_IDS)
    chat = int(payload.get("chat_id") or CHAT_ID)
    saved = []
    os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
    for field, f in (kw.get("files") or {}).items():
        fh = f[1] if isinstance(f, tuple) else f
        name = f"{mid}_{field}_{os.path.basename(getattr(fh, 'name', '') or field)}"
        dst = os.path.join(ROOT, "out", name)
        try:
            data = fh.read() if hasattr(fh, "read") else fh
            with open(dst, "wb") as out:
                out.write(data if isinstance(data, bytes) else str(data).encode())
            saved.append(dst)
        except Exception:
            pass
    for v in payload.values():         # file:// or a path the bot passes as a string
        if isinstance(v, str) and os.path.isabs(v.replace("file:///", "")) and os.path.exists(v.replace("file:///", "")):
            dst = os.path.join(ROOT, "out", f"{mid}_{os.path.basename(v)}")
            shutil.copy(v.replace("file:///", ""), dst)
            saved.append(dst)
    with _LOCK, open(os.path.join(ROOT, "outbox.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"t": time.time(), "chat_id": chat, "method": method, "message_id": mid,
                             "payload": payload, "files": saved}, ensure_ascii=False) + "\n")
    result = {"message_id": mid, "date": int(time.time()), "chat": {"id": chat, "type": "private"}}
    if saved:
        fid = "dbg:" + os.path.basename(saved[0])
        result.update(photo=[{"file_id": fid, "file_unique_id": fid}],
                      document={"file_id": fid}, video={"file_id": fid},
                      audio={"file_id": fid}, voice={"file_id": fid})
    return _Resp({"ok": True, "result": result})


class _Resp:
    status_code = 200

    def __init__(self, body):
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


def ensure_user(store, user_cls) -> None:
    """The debug users are approved, ordinary (not admin) users."""
    for i in range(USERS):
        if store.get(CHAT_ID + i) is None:
            store.put(user_cls(chat_id=CHAT_ID + i, name=f"Debug {i}",
                               tg_username=f"debug_user_{i}", status="approved"))


def loop(bot) -> None:
    """Feeds the inbox into the bot the way the poll loop feeds Telegram's updates."""
    import tg_bot
    import tg_userstore
    while bot._running:
        if enabled():
            try:
                ensure_user(bot._user_store, tg_userstore._User)
                serve()
                for upd in pull():
                    tg_bot.logger.info("debug user: update %s", str(upd)[:200])
                    bot._dispatch_logged(upd)
            except Exception:
                tg_bot.logger.exception("debug user loop")
        time.sleep(0.5)


# ---- the user side: what bench/debug_user.py and the GUI Debug chat send ----

OUTBOX = os.path.join(ROOT, "outbox.jsonl")
SHOT = os.path.join(ROOT, "gui_shot.png")


def _put(update: dict) -> None:
    box = os.path.join(ROOT, "inbox")
    os.makedirs(box, exist_ok=True)
    name = f"{time.time():.6f}_{uuid.uuid4().hex[:6]}.json"
    tmp = os.path.join(box, name + ".part")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(update, fh, ensure_ascii=False)
    os.replace(tmp, os.path.join(box, name))


def _chat(user: int) -> int:
    if not 0 <= int(user) < USERS:
        raise ValueError(f"debug user {user}: 0..{USERS - 1}")
    return CHAT_ID + int(user)


def _msg(user: int = 0, **kw) -> dict:
    cid = _chat(user)
    m = {"message_id": int(time.time() * 1000) % 2_000_000_000, "date": int(time.time()),
         "chat": {"id": cid, "type": "private"},
         "from": {"id": cid, "is_bot": False, "first_name": f"Debug {user}",
                  "username": f"debug_user_{user}", "language_code": "ru"}}
    m.update(kw)
    return m


def _file(path: str) -> str:
    os.makedirs(os.path.join(ROOT, "files"), exist_ok=True)
    name = uuid.uuid4().hex[:8] + "_" + os.path.basename(path)
    shutil.copy(path, os.path.join(ROOT, "files", name))
    return "dbg:" + name


def _photo(path: str) -> list:
    from PIL import Image
    w, h = Image.open(path).size
    fid = _file(path)
    return [{"file_id": fid, "file_unique_id": fid, "width": w, "height": h}]


def say(text: str, user: int = 0) -> None:
    _put({"update_id": 0, "message": _msg(user, text=text)})


def photo(path: str, caption: str = "", group: str = "", user: int = 0) -> None:
    m = _msg(user, photo=_photo(path))
    if caption:
        m["caption"] = caption
    if group:
        m["media_group_id"] = group
    _put({"update_id": 0, "message": m})


def album(paths: list, caption: str = "", user: int = 0) -> None:
    g = uuid.uuid4().hex[:10]
    for i, p in enumerate(paths):
        photo(p, caption if i == 0 else "", g, user)


def press(data: str, user: int = 0) -> None:
    cid = _chat(user)
    _put({"update_id": 0, "callback_query": {
        "id": uuid.uuid4().hex, "data": data,
        "from": {"id": cid, "username": f"debug_user_{user}"},
        "message": _msg(user, text="")}})


def outbox(since: float = 0.0, user=None) -> list:
    """What the bot sent since `since`; one debug user's only when `user` is given."""
    if not os.path.exists(OUTBOX):
        return []
    want = None if user is None else _chat(user)
    with open(OUTBOX, encoding="utf-8") as fh:
        return [e for e in map(json.loads, fh) if e["t"] >= since
                and (want is None or e.get("chat_id", CHAT_ID) == want)]


# ---- the HTTP API: the same user, driven over 127.0.0.1 (the GUI tab, scripts) ----

API_PORT = int(os.getenv("DEBUG_USER_PORT", "8765"))


def _api_handler():
    from http.server import BaseHTTPRequestHandler
    from urllib.parse import parse_qs, urlparse

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body):
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/messages":
                q = parse_qs(u.query)
                since = float((q.get("since") or ["0"])[0])
                user = int((q.get("user") or ["0"])[0])
                return self._send(200, {"chat_id": _chat(user), "messages": outbox(since, user)})
            if u.path == "/health":
                return self._send(200, {"ok": True, "enabled": enabled()})
            self._send(404, {"error": "unknown path"})

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                path = urlparse(self.path).path
                user = int(body.get("user") or 0)
                if path == "/say":
                    say(str(body["text"]), user)
                elif path == "/photo":
                    photo(str(body["path"]), str(body.get("caption") or ""), "", user)
                elif path == "/album":
                    album([str(p) for p in body["paths"]], str(body.get("caption") or ""), user)
                elif path == "/press":
                    press(str(body["data"]), user)
                elif path == "/shot":
                    # the Debug chat tab takes it on the GUI thread: shows itself, saves the window
                    with open(SHOT + ".req", "w", encoding="utf-8") as fh:
                        fh.write("1")
                    return self._send(200, {"ok": True, "path": SHOT})
                else:
                    return self._send(404, {"error": "unknown path"})
                self._send(200, {"ok": True, "t": time.time()})
            except (KeyError, ValueError, OSError) as exc:
                self._send(400, {"error": str(exc)})
    return H


_SERVER = None


def serve() -> None:
    """Started once by the bot's debug thread; localhost only."""
    global _SERVER
    if _SERVER is not None:
        return
    from http.server import ThreadingHTTPServer
    try:
        _SERVER = ThreadingHTTPServer(("127.0.0.1", API_PORT), _api_handler())
    except OSError:
        _SERVER = False             # port taken: tried once, the inbox still works
        raise
    threading.Thread(target=_SERVER.serve_forever, daemon=True, name="debug-user-api").start()

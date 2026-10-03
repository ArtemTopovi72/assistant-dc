"""A Telegram-looking chat in the browser, wired to the REAL bot stack
(bench/live_tg_drive.py fakes only api.telegram.org).

    venv/Scripts/python.exe -u bench/tg_sim.py        -> http://127.0.0.1:8765/

The page and scripts share one JSON API, so a person and an agent can drive
the same chat and both see what users see (edits, deletions, buttons, media):
  GET  /events?chat=ID&since=N     outbound calls for that chat
  POST /say    {chat, text, reply_to?, forward_from?, ago?}
  POST /press  {chat, data, message_id}
  POST /upload {chat, kind: photo|voice|document, name, b64, caption, reply_to?, forward_from?}
  POST /album  {chat, files: [{name, b64}], caption, forward_from?}
reply_to is a message id in the chat; forward_from a name (someone else's words);
ago seconds in the past (a backlog while the bot was down). User messages come
back in /events as method "user" under their real ids.
"""
import base64, json, os, sys, tempfile, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import bench.live_tg_drive as D  # noqa: E402

PORT = int(os.getenv("TG_SIM_PORT", "8765"))
PAGE = Path(__file__).with_name("tg_sim.html")
CHATS: dict[int, "D.Chat"] = {}
BOT = None
_lock = threading.Lock()


def chat(cid) -> "D.Chat":
    cid = int(cid)
    with _lock:
        if cid not in CHATS:
            CHATS[cid] = D.Chat(BOT, cid, f"user{cid % 1000}")
        return CHATS[cid]


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/":
            b = PAGE.read_bytes()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers(); self.wfile.write(b); return
        if u.path == "/events":
            cid = int(q.get("chat", 0)); since = int(q.get("since", 0))
            chat(cid)
            with D.WIRE.lock:
                evs = [e for e in D.WIRE.sent if e["chat_id"] == cid]
            c = CHATS[cid]
            with BOT._task_lock:
                busy = bool(BOT._chat_busy.get(cid, 0)) or bool(BOT._running_task.get(cid))
            return self._json({"events": evs[since:], "n": len(evs), "busy": busy})
        if u.path == "/file":
            p = Path(q.get("p", "")).resolve()
            if D.OUT.resolve() not in p.parents or not p.is_file():
                return self._json({"error": "no"}, 404)
            import mimetypes
            b = p.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(str(p))[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); return
        self._json({"error": "404"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        c = chat(body.get("chat", 1))
        kw = {k: body[k] for k in ("reply_to", "forward_from", "ago") if body.get(k)}
        if "reply_to" in kw: kw["reply_to"] = int(kw["reply_to"])
        if self.path == "/say":
            c.say(body["text"], **kw)
        elif self.path == "/album":
            d = D.OUT / "inbound"; d.mkdir(exist_ok=True); paths = []
            for f in body.get("files") or []:
                p = d / os.path.basename(f.get("name") or "photo.jpg"); p.write_bytes(base64.b64decode(f["b64"]))
                paths.append(str(p))
            kw.pop("ago", None); c.album(paths, body.get("caption", ""), **kw)
        elif self.path == "/press":
            c.press(body["data"], int(body.get("message_id") or 0))
        elif self.path == "/upload":
            d = D.OUT / "inbound"; d.mkdir(exist_ok=True)
            p = d / os.path.basename(body.get("name") or "file.bin")
            p.write_bytes(base64.b64decode(body["b64"]))
            kind = body.get("kind", "document")
            kw.pop("ago", None)
            if kind == "photo": c.photo(str(p), body.get("caption", ""), **kw)
            elif kind == "voice": c.voice(str(p))
            else: c.document(str(p), body.get("caption", ""))
        else:
            return self._json({"error": "404"}, 404)
        self._json({"ok": True})


if __name__ == "__main__":
    BOT, _ctx = D.build()
    print("bot up; output ->", D.OUT)
    print(f"open http://127.0.0.1:{PORT}/")
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()

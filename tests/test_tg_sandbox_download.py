"""/files as buttons: a tap on a file downloads it, a tap on a folder opens it.

The paths never ride in callback_data (Telegram caps it at 64 bytes); a press
carries only an index into the listing drawn for that chat. So the checks are
about that indirection: the index resolves to the right file, a folder opens in
place, ⬆ goes up, a list from before a restart is redrawn instead of guessed,
and a revoked grant stops a tap exactly like it stops /files.

Delivery: the self-hosted Bot API server (--local) is sent a file:// URI, not an
upload, so a 117 MB DCIM.zip goes out without a multipart POST; a refused URI
falls back to the upload.

Run: venv/Scripts/python.exe tests/test_tg_sandbox_download.py
"""
import json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import tg_bot as T

_DATA_DIR = tempfile.mkdtemp(prefix="tgtest_sbxdl_")
T.redirect_data_dir(_DATA_DIR)

import sandbox_access as A
import code_sandbox as CS
from pathlib import Path

CS.SANDBOX_BASE = Path(tempfile.mkdtemp(prefix="tgtest_sbxdl_root_"))

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    if cond: OK += 1; print(f"PASS  {name}")
    else:    BAD += 1; print(f"FAIL  {name}   {extra}")


bot = T.TelegramBot("123:TEST", lambda: None, lambda: None, lambda: {}, silent_mode=True)
sent, docs = [], []
def rec(method, payload=None, **kw):
    sent.append((method, payload or {}))
    return {"ok": True, "result": {"message_id": 500 + len(sent)}}
bot._api_post = rec
bot._send_document = lambda chat_id, path, caption="": docs.append(path) or True

CID = 4242
u = T._User(chat_id=CID, name="t", status="approved")
u.prefs = {"sandbox": A.FILES}
bot._user_store.put(u)
box = CS.sandbox_for(CID)
box.write_text("build_mod.py", "print(1)")
box.write_text("mods_unpacked/META-INF/MANIFEST.MF", "Manifest-Version: 1.0")
box.write_text("mods_unpacked/readme.txt", "hi")


def kb():
    for method, p in reversed(sent):
        if p.get("reply_markup"):
            return method, json.loads(p["reply_markup"])
    return None, {}


def buttons():
    _, k = kb()
    return [(b["text"], b["callback_data"]) for row in k.get("inline_keyboard", []) for b in row]


def tap(label_part):
    # No sent.clear(): the listing stays on screen after a file goes out, and
    # the next tap is on that same keyboard.
    data = next(d for t, d in buttons() if label_part in t)
    bot._dispatch_callback({"id": "x", "data": data,
                            "message": {"message_id": 777, "chat": {"id": CID, "type": "private"}}})


sess = bot._get_session(CID)
bot._handle_sandbox_command(CID, sess, bot._user_store.get(CID), "ru", "/files")
labels = [t for t, _ in buttons()]
check("every entry is a button", any("build_mod.py" in t for t in labels)
      and any("mods_unpacked" in t for t in labels), labels)
check("callback_data carries an index, never the path",
      all(len(d.encode()) <= 64 and "/" not in d for _, d in buttons()), buttons())
check("a file button shows its size", any("build_mod.py ·" in t for t in labels), labels)
check("the root has no ⬆ button", not any(d == "sbx:up" for _, d in buttons()))

tap("mods_unpacked")
method, _ = kb()
check("a folder opens IN PLACE (the message is edited)", method == "editMessageText", method)
labels = [t for t, _ in buttons()]
check("its entries are listed", any("readme.txt" in t for t in labels)
      and any("META-INF" in t for t in labels), labels)
check("and it has a ⬆ button", any(d == "sbx:up" for _, d in buttons()))

docs.clear()
tap("readme.txt")
check("a file tap sends THAT file", len(docs) == 1 and docs[0].endswith(
    os.path.join("mods_unpacked", "readme.txt")), docs)

tap("⬆")
check("⬆ goes back to the parent", any("build_mod.py" in t for t, _ in buttons()), buttons())

# The list from before a restart: the numbers mean nothing any more.
bot._sbx_lists.clear()
sent.clear(); docs.clear()
bot._dispatch_callback({"id": "y", "data": "sbx:0",
                        "message": {"message_id": 777, "chat": {"id": CID, "type": "private"}}})
check("a stale list sends nothing", docs == [], docs)
check("and is redrawn instead", any("build_mod.py" in t for t, _ in buttons()), buttons())

# A file deleted since the list was drawn.
docs.clear()
os.remove(box.resolve("build_mod.py"))
tap("build_mod.py")
check("a file that is gone is not sent", docs == [], docs)

# Revocation stops a tap exactly like it stops /files.
A.set_level(bot._user_store, CID, A.OFF)
sent.clear(); docs.clear()
bot._dispatch_callback({"id": "z", "data": "sbx:0",
                        "message": {"message_id": 777, "chat": {"id": CID, "type": "private"}}})
txt = " ".join(p.get("text", "") for _, p in sent)
check("a revoked user's old buttons send nothing", docs == [], docs)
check("and say the folder is not enabled", "не включена" in txt, txt[:100])


# ── delivery through the local Bot API server ───────────────────────────────
import requests
import config as C
import tg_transport

posted = []
class _R:
    def __init__(self, code, text=""): self.status_code, self.text = code, text
    def json(self): return {"ok": self.status_code == 200, "description": self.text}

real_post, real_local = requests.post, C.TG_API_LOCAL
bot2 = T.TelegramBot("123:TEST", lambda: None, lambda: None, lambda: {}, silent_mode=True)
f = Path(tempfile.mkdtemp()) / "DCIM.zip"
f.write_bytes(b"PK\x03\x04" + b"0" * 1024)
try:
    C.TG_API_LOCAL = True
    requests.post = lambda url, data=None, files=None, **kw: (
        posted.append({"data": dict(data or {}), "files": bool(files)}) or _R(200))
    ok = bot2._send_document(CID, str(f))
    check("local server: the file goes out", ok)
    check("by file:// URI, with no upload",
          posted and posted[0]["data"].get("document", "").startswith("file:") and not posted[0]["files"],
          posted)

    posted.clear()
    seq = iter([_R(400, "Bad Request: wrong file identifier/HTTP URL specified"), _R(200)])
    requests.post = lambda url, data=None, files=None, **kw: (
        posted.append({"data": dict(data or {}), "files": bool(files)}) or next(seq))
    ok = bot2._send_document(CID, str(f))
    check("a refused URI falls back to the upload", ok and len(posted) == 2
          and posted[1]["files"] and "document" not in posted[1]["data"], posted)

    posted.clear()
    C.TG_API_LOCAL = False
    requests.post = lambda url, data=None, files=None, **kw: (
        posted.append({"data": dict(data or {}), "files": bool(files)}) or _R(200))
    bot2._send_document(CID, str(f))
    check("the cloud API still gets an upload", posted and posted[0]["files"], posted)
finally:
    requests.post, C.TG_API_LOCAL = real_post, real_local

print(f"\n{OK}/{OK + BAD} checks passed")
sys.exit(0 if BAD == 0 else 1)

"""A document over Telegram's 20 MB bot download limit is named as such.

Live 2026-09-13: a forwarded 112 MB DCIM.zip came back as
«Не удалось прочитать DCIM.zip», the user asked «Почему?», and the bot had
no idea — getFile had simply refused. The dispatcher now carries file_size
and the resolver explains the limit before trying to download.
"""
import os, sys, tempfile
os.environ.setdefault("F5_TEST_RUN", "1")
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tg_bot as T
T.redirect_data_dir(tempfile.mkdtemp(prefix="docbig_"))
import tg_resolve

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

# The machine now carries real api_id/api_hash in .env, so config reports the
# local Bot API (2 GB) at import time; this half of the suite is about the
# CLOUD limit and pins it.
import config as C
C.TG_API_LOCAL = False

bot = T.TelegramBot("123:TEST", lambda: None, lambda: None, lambda: {}, silent_mode=True)
bot.posts = []
bot._api_post = lambda m, p=None, **k: (bot.posts.append((m, p or {})), {"ok": True, "result": {"message_id": 1}})[1]
bot._api_get = lambda *a, **k: {}
bot._activity.log = lambda *a, **k: None
bot._backend = T.InMemoryBackend()
downloads = []
bot._dl_bytes = lambda fid: (downloads.append(fid), None)[1]
CID = 881077
s = bot._get_session(CID); s.clear_context(); s.reg_state = ""; s.lang = "ru"; bot._store.put(s)

def sent():
    return [p["text"] for m, p in bot.posts if m == "sendMessage"]

bot._resolve_and_push(CID, [{"type": "document", "file_id": "f", "filename": "DCIM.zip",
                             "file_size": 112 * 1024 * 1024 + 100_000, "caption": ""}])
msg = " ".join(sent())
check("the reply names the 20 MB limit", "20 МБ" in msg, msg)
check("...as 'this bot', because the cap depends on the API server", "этот бот" in msg, msg)
check("...and the file's size", "112 МБ" in msg, msg)
check("...and the file name", "DCIM.zip" in msg, msg)
check("it is not the generic 'could not read'", "Не удалось прочитать" not in msg, msg)
check("getFile is not even attempted", downloads == [], downloads)
check("nothing is queued to the agent", bot._backend.depth() == 0)

bot.posts.clear()
bot._resolve_and_push(CID, [{"type": "document", "file_id": "g", "filename": "small.pdf",
                             "file_size": 3 * 1024 * 1024, "caption": ""}])
check("a small file still goes down the normal path (download attempted)", downloads == ["g"], downloads)
check("...and a failed download is still 'could not read'", any("Не удалось прочитать" in t for t in sent()), sent())

# the dispatcher passes file_size through
import inspect, tg_dispatch
check("dispatcher carries file_size on document items", '"file_size": int(doc.get("file_size")' in inspect.getsource(tg_dispatch))
check("the limit is Telegram's 20 MB on the cloud API", tg_resolve._bot_file_limit() == 20 * 1024 * 1024)
import config as C, tg_transport
C.TG_API_LOCAL = True
check("...and 2 GB behind a local Bot API server", tg_resolve._bot_file_limit() == 2000 * 1024 * 1024)
C.TG_API_LOCAL = False
src = inspect.getsource(tg_transport.TransportMixin._dl_bytes_raw)
check("the transport uses TG_API_BASE, not a hardcoded host", "api.telegram.org" not in src and "TG_API_BASE" in src)
check("a local-mode absolute file_path is read from disk", "os.path.isabs(fp)" in src)

print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)

"""The self-hosted Bot API server is wired in: config, launcher, bot start.

Live 2026-09-14: a 112 MB forwarded archive cannot pass the cloud API's 20 MB
getFile cap, so the official server (--local) is started by the app itself
once TG_API_ID/TG_API_HASH are in .env, and the token is moved off the cloud
with one logOut.
"""
import os, sys, inspect
os.environ.setdefault("F5_TEST_RUN", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import config as C
import tg_local_api as L

OK = BAD = 0
def check(name, cond, extra=""):
    global OK, BAD
    OK += bool(cond); BAD += (not cond)
    print(("PASS  " if cond else "FAIL  ") + name, ("  " + str(extra)) if (extra and not cond) else "")

# credentials switch the base to the local server; without them nothing changes
saved = (C.TG_API_ID, C.TG_API_HASH, C.TG_API_BASE, C.TG_API_LOCAL)
C.TG_API_ID, C.TG_API_HASH, C.TG_API_LOCAL = "", "", False
check("without credentials the local server is disabled", not L.enabled())
check("...and ensure_server is a no-op", L.ensure_server() is False)
C.TG_API_ID, C.TG_API_HASH = "123", "a" * 32
C.TG_API_BASE, C.TG_API_LOCAL = "http://127.0.0.1:8081", True
check("with credentials it is enabled", L.enabled())

class _R:
    def __init__(s, code, body): s.status_code, s.text, s._b = code, str(body), body
    def json(s): return s._b
class _Http:
    def __init__(s, local_codes): s.local, s.calls = list(local_codes), []
    def get(s, url, **k):
        s.calls.append(("get", url)); return _R(s.local.pop(0), {"ok": True, "result": {}} if s.local == [] or True else {})
    def post(s, url, **k):
        s.calls.append(("post", url)); return _R(200, {"ok": True, "result": True})

# already accepted locally: no logOut
h = _Http([200])
check("a token the local server accepts needs no logOut",
      L.ensure_bot_logged_in("1:T", http=h) and not any(c[0] == "post" for c in h.calls), h.calls)
# 401 locally -> one cloud logOut -> accepted
L.time.sleep = lambda *_: None
h = _Http([401, 200])
ok = L.ensure_bot_logged_in("1:T", http=h)
posts = [u for m, u in h.calls if m == "post"]
check("401 locally triggers exactly one cloud logOut", ok and posts == ["https://api.telegram.org/bot1:T/logOut"], h.calls)
check("...and local getMe goes to TG_API_BASE", h.calls[0][1].startswith("http://127.0.0.1:8081/bot1:T/getMe"), h.calls)

# the pieces are wired in
import tg_bot
check("TelegramBot.start ensures the local server and login", "tg_local_api" in inspect.getsource(tg_bot.TelegramBot.start))
src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "launch_all.py"), encoding="utf-8").read()
check("launch_all starts the server before the GUI", "tg_local_api.ensure_server" in src and src.index("tg_local_api") < src.index("assistant.main()"))
csrc = inspect.getsource(C)
check("config defaults TG_API_BASE to the local port when credentials are set", "TG_API_ID and TG_API_HASH" in csrc)
check(".env.example documents the two fields", "TG_API_HASH" in open(".env.example", encoding="utf-8").read())
check("the server command runs --local with the credentials", "--local" in inspect.getsource(L.ensure_server) and "--api-hash" in inspect.getsource(L.ensure_server))

# credentials typed into the GUI override .env and flip the base
os.environ.pop("TG_API_BASE", None)
check("GUI credentials enable the local server", L.apply_credentials("77", "b" * 32) and C.TG_API_LOCAL and "8081" in C.TG_API_BASE)
check("clearing them returns to the cloud", not L.apply_credentials("", "") and not C.TG_API_LOCAL)
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import gui_telegram_tab as G
gsrc = inspect.getsource(G)
check("the Telegram tab has masked api_id/api_hash fields",
      "api_hash_in" in gsrc and gsrc.count("QLineEdit.Password") >= 3)
check("...remembered on Start like the token", "_TG_API_HASH_KEY, self.api_hash_in" in gsrc and "_TG_TOKEN_KEY,  token" in gsrc)
check("...and applied before the bot starts", gsrc.index("apply_credentials") < gsrc.index("self._bot = TelegramBot("))

# the server is really started (it never was on Linux: STARTUPINFO and
# creationflags made Popen raise there, and the error was only logged)
import tempfile, time as _time
_tmp = tempfile.mkdtemp()
_flag = os.path.join(_tmp, "started")
if os.name == "nt":
    _exe = os.path.join(_tmp, "fake.cmd")
    open(_exe, "w").write(f'@echo off\r\necho %*> "{_flag}"\r\n')
else:
    _exe = os.path.join(_tmp, "fake.sh")
    open(_exe, "w").write(f'#!/bin/sh\necho "$@" > "{_flag}"\n')
    os.chmod(_exe, 0o755)
_saved_exe, _saved_up = C.TG_API_EXE, L.server_up
C.TG_API_ID, C.TG_API_HASH, C.TG_API_LOCAL = "123", "a" * 32, True
C.TG_API_BASE = "http://127.0.0.1:8081"
C.TG_API_EXE = _exe
L.server_up = lambda *a, **k: os.path.exists(_flag)
L.time.sleep = _time.sleep
check("ensure_server starts the executable", L.ensure_server(root=_tmp, wait_s=10), _flag)
check("...with --local and the credentials", os.path.exists(_flag) and "--local" in open(_flag).read())
check("...listening on loopback only, not the LAN",
      os.path.exists(_flag) and "--http-ip-address 127.0.0.1" in open(_flag).read())
C.TG_API_EXE, L.server_up = _saved_exe, _saved_up

C.TG_API_ID, C.TG_API_HASH, C.TG_API_BASE, C.TG_API_LOCAL = saved
print("\n%d/%d checks passed" % (OK, OK + BAD))
sys.exit(1 if BAD else 0)

"""Self-hosted Telegram Bot API server (telegram-bot-api --local).

Telegram's cloud Bot API refuses getFile over 20 MB; a forwarded 112 MB
DCIM.zip could only be explained, not read. The official server, built by
scripts/build_telegram_bot_api.cmd and run here with --local, lifts that to
2 GB and hands files back as absolute paths on this disk (tg_transport reads
them directly). It needs the app credentials from https://my.telegram.org/apps
(TG_API_ID / TG_API_HASH in .env, never committed).

Two things the server does NOT do by itself:
  * a bot token logged in on the cloud is refused locally (401) until the
    cloud is told `logOut` once -- ensure_bot_logged_in does that on demand;
  * nothing starts it -- ensure_server does, idempotently, from launch_all.
"""
from __future__ import annotations

import logging
import os
import subprocess
import time

import config as _config

logger = logging.getLogger("assistant.tg_local_api")

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
_NEW_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)


def apply_credentials(api_id: str, api_hash: str) -> bool:
    """Take credentials from the GUI (saved in QSettings like the token).

    They override .env for this process; with both present TG_API_BASE moves
    to the local server, with either missing it goes back to the cloud unless
    TG_API_BASE was set explicitly in the environment.
    """
    api_id, api_hash = (api_id or "").strip(), (api_hash or "").strip()
    _config.TG_API_ID, _config.TG_API_HASH = api_id, api_hash
    if not os.getenv("TG_API_BASE"):
        _config.TG_API_BASE = ("http://127.0.0.1:%d" % _config.TG_API_PORT
                               if (api_id and api_hash) else "https://api.telegram.org")
        _config.TG_API_LOCAL = _config.TG_API_BASE != "https://api.telegram.org"
    return enabled()


def enabled() -> bool:
    return bool(_config.TG_API_LOCAL and _config.TG_API_ID and _config.TG_API_HASH)


def server_up(timeout: float = 1.5) -> bool:
    """The local server answers on its port (any HTTP status counts)."""
    import urllib.request
    import urllib.error
    try:
        urllib.request.urlopen(_config.TG_API_BASE + "/", timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


def ensure_server(root: str | None = None, wait_s: float = 20.0) -> bool:
    """Start telegram-bot-api --local unless it is already answering."""
    if not enabled():
        return False
    if server_up():
        return True
    exe = _config.TG_API_EXE
    if not os.path.exists(exe):
        logger.error("TG_API_ID/HASH are set but %s is missing — run "
                     "scripts\\build_telegram_bot_api.cmd; staying on the cloud API", exe)
        return False
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    work = os.path.join(root, "runtime", "tbapi")
    os.makedirs(os.path.join(work, "tmp"), exist_ok=True)
    log_path = os.path.join(root, "runtime", "tbapi.log")
    try:
        log = open(log_path, "a", encoding="utf-8", errors="replace")
    except Exception:
        log = subprocess.DEVNULL
    args = [exe, "--local", "--api-id", str(_config.TG_API_ID),
            "--api-hash", str(_config.TG_API_HASH),
            "--http-port", str(_config.TG_API_PORT),
            "--dir", work, "--temp-dir", os.path.join(work, "tmp"),
            "--max-connections", "10", "--verbosity", "1"]
    try:
        if os.name == "nt":
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = 0
            kw = {"creationflags": _NO_WINDOW | _NEW_GROUP, "startupinfo": si}
        else:   # STARTUPINFO and creationflags are Windows-only (ValueError here)
            kw = {"start_new_session": True}
        subprocess.Popen(args, cwd=work, stdout=log, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, **kw)
    except Exception as exc:
        logger.error("could not start telegram-bot-api: %s", exc)
        return False
    deadline = time.time() + wait_s
    while time.time() < deadline:
        if server_up():
            logger.info("telegram-bot-api --local is up on %s (files up to 2 GB)",
                        _config.TG_API_BASE)
            return True
        time.sleep(0.5)
    logger.error("telegram-bot-api did not answer within %.0fs — see %s", wait_s, log_path)
    return False


def ensure_bot_logged_in(token: str, http=None) -> bool:
    """Make the local server accept `token`.

    A token still logged in on api.telegram.org gets 401 Unauthorized from a
    local server. Telegram's documented cure is one `logOut` call on the cloud
    (the bot cannot log back in there for ~10 minutes, which is fine: this
    machine is its home now). Returns True when local getMe answers ok.
    """
    import requests as _rq
    http = http or _rq
    if not (enabled() and token):
        return False
    local = f"{_config.TG_API_BASE}/bot{token}/getMe"
    try:
        r = http.get(local, timeout=10)
        if r.status_code == 200 and (r.json() or {}).get("ok"):
            return True
        if r.status_code != 401:
            logger.warning("local getMe: HTTP %s %s", r.status_code, r.text[:200])
            return False
    except Exception as exc:
        logger.warning("local getMe failed: %s", exc)
        return False
    logger.warning("bot is still logged in on the cloud API — sending logOut once")
    try:
        r = http.post(f"https://api.telegram.org/bot{token}/logOut", timeout=20)
        logger.info("cloud logOut: HTTP %s %s", r.status_code, r.text[:200])
    except Exception as exc:
        logger.error("cloud logOut failed (Telegram unreachable?): %s", exc)
        return False
    for _ in range(6):
        time.sleep(2)
        try:
            r = http.get(local, timeout=10)
            if r.status_code == 200 and (r.json() or {}).get("ok"):
                logger.info("bot accepted by the local Bot API server")
                return True
        except Exception:
            pass
    logger.error("local server still refuses the token after logOut")
    return False

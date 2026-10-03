"""One-click launcher: start ComfyUI Desktop + LM Studio (+ server), then the
Assistant GUI. Targeted by the desktop shortcut (run via pythonw, no console).

Each external app is only started if its server isn't already up, so re-running
is safe. Errors are written to launch_error.log (pythonw has no console).
"""
import os
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import install_paths  # noqa: E402  -- the source folders, for this process and the venv
install_paths.ensure(write_venv=__name__ == "__main__")
import win_runtime  # noqa: E402
win_runtime.preload_newest_msvcp()   # before PyQt5 brings its old 14.26 (core/win_runtime.py)

# ── single-instance guard ────────────────────────────────────────────────────
# The shortcut/watchdog that runs this script has, in practice, fired twice
# within milliseconds of each other (two `launch_all.py` -> two `assistant.
# main()` -> two TelegramBot instances polling the SAME bot token). Telegram
# then handed each of them an overlapping batch of updates, so every message
# got answered twice (or more) with no way to tell from inside the bot that
# it wasn't alone -- a user saw four "checking weather" replies to one typed
# city with no way to explain it short of `tasklist`.
#
# msvcrt.locking() is an OS-level advisory lock on the open file descriptor:
# it is held only as long as this process is alive and is released
# automatically (by Windows itself) on a crash or kill, so there is no stale-
# lock state to clean up -- unlike a plain "lock file exists" check, which a
# process that died without deleting it would leave permanently jammed.
_SINGLETON_LOCK_PATH = os.path.join(ROOT, ".assistant_singleton.lock")
_singleton_handle = None  # kept open for the process lifetime; GC would drop the lock


def _acquire_singleton_lock() -> bool:
    global _singleton_handle
    try:
        fd = os.open(_SINGLETON_LOCK_PATH, os.O_RDWR | os.O_CREAT)
        _singleton_handle = os.fdopen(fd, "r+b", buffering=0)
        # msvcrt.locking locks at the CURRENT position: always byte 0, so a
        # file that grows can never move the lock to a byte nobody else tries.
        _singleton_handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(_singleton_handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            # Same semantics on POSIX: an advisory lock the kernel drops when
            # the process dies, so a crash leaves nothing stale behind. flock,
            # not lockf: it is held per open file like msvcrt, so a second open
            # in the SAME process is refused too.
            import fcntl
            fcntl.flock(_singleton_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Who holds it, for the refusal message: 2026-09-24 a restart was refused
        # with no app process alive, and the holder could only be found by
        # killing candidates. Fixed-width PID in bytes 1..10, never byte 0.
        try:
            _singleton_handle.seek(1)
            _singleton_handle.write(f"{os.getpid():>10}".encode())
        except Exception:
            pass
        return True
    except OSError:
        if _singleton_handle is not None:
            try: _singleton_handle.close()
            except Exception: pass
            _singleton_handle = None
        return False

COMFY_EXE = os.getenv("COMFY_EXE", os.path.join(os.path.expanduser("~"), "AppData", "Local", "Programs",
                                                "ComfyUI", "ComfyUI.exe"))
# ComfyUI from SOURCE, preferred over the Desktop app.
#
# Ideogram 4 needs ComfyUI >= 0.24 (CLIPLoader type "ideogram4", DualModelGuider,
# CFGOverride). ComfyUI Desktop bundles 0.22.3 and updates only through its own
# updater, so 0.28.0 lives beside it and is launched against the SAME venv, models
# and custom_nodes via --base-directory. If that tree is missing we fall back to the
# Desktop app, which still serves everything except Ideogram.
#
# 0.30.0 adds the MiniMaxH3* nodes that video generation needs, and still carries
# the Ideogram 4 support 0.28.0 was kept for — so it supersedes rather than sits
# beside it. 0.33.0 is a from-source main-HEAD checkout (see
# docs/music_generation.md) needed because MiniMaxMusic3TextEncode /
# EmptyMiniMaxMusic3LatentAudio for song generation land after the v0.32.0
# tag — no stable release has them yet. Every ComfyUI-<ver> tree shares
# models and custom_nodes via --base-directory, so rolling back is just
# deleting the newer directory (or setting COMFY_SRC): pick the HIGHEST
# version present rather than hardcoding one, and fall back to the Desktop
# app if none is there. Trees through 0.30.0/0.32.0 share one venv; 0.33.0+
# carries its own (_comfy_python below) since its deps outran the shared one.
def _newest_comfy_src():
    pinned = os.getenv("COMFY_SRC")
    if pinned:
        return pinned
    base = os.getenv("COMFY_BASE_DIR", os.path.join(os.path.expanduser("~"), "Documents", "ComfyUI"))
    best, best_key = None, ()
    try:
        for name in os.listdir(base):
            if not name.startswith("ComfyUI-"):
                continue
            path = os.path.join(base, name)
            if not os.path.exists(os.path.join(path, "main.py")):
                continue
            try:
                key = tuple(int(x) for x in name.split("-", 1)[1].split("."))
            except ValueError:
                continue
            if key > best_key:
                best, best_key = path, key
    except OSError:
        pass
    return best or os.path.join(base, "ComfyUI-0.28.0")


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR_COMFY, INPUT_DIR_COMFY   # one place for every generation
import config as _config  # noqa: E402

COMFY_SRC = _newest_comfy_src()
COMFY_BASE_DIR = os.getenv("COMFY_BASE_DIR", os.path.join(os.path.expanduser("~"), "Documents", "ComfyUI"))


# Where a venv keeps its interpreter: Scripts\\python.exe on Windows, bin/python
# elsewhere (setup.sh installs ComfyUI on Linux too).
_VENV_PY = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")


def _comfy_python(src):
    """Which python.exe runs COMFY_SRC.

    Every ComfyUI-<ver> tree up through 0.32.0 shares one venv at
    COMFY_BASE_DIR\\.venv. Newer trees (0.33.0+, cloned from source after
    MiniMax Music3 landed post-v0.32.0 — see docs/music_generation.md) were
    set up with their OWN venv living inside the tree itself (named after
    whatever the clone was called at setup time, e.g. .venvmain, .venv032),
    because the shared venv's pinned deps predate what these need. Prefer
    a venv found inside COMFY_SRC; fall back to the historical shared one
    so 0.28.0/0.30.0/0.32.0 keep working unchanged.
    """
    if os.path.isdir(src):
        for venv_name in os.listdir(src):
            cand = os.path.join(src, venv_name, *_VENV_PY)
            if venv_name.startswith(".venv") and os.path.exists(cand):
                return cand
    return os.path.join(COMFY_BASE_DIR, ".venv", *_VENV_PY)


# python.exe, NOT pythonw.exe: ComfyUI logs heavily and dies early when its stdout
# has nowhere to go, so it is started detached with the output redirected to a file.
COMFY_PY = os.getenv("COMFY_PY", _comfy_python(COMFY_SRC))
COMFY_PORT = os.getenv("COMFY_PORT", "8000")
COMFY_LOG = os.getenv("COMFY_LOG", os.path.join(COMFY_BASE_DIR, "comfyui_launch.log"))
LMSTUDIO_EXE = os.getenv("LMSTUDIO_EXE", os.path.join(os.path.expanduser("~"), "AppData", "Local",
                                                      "Programs", "LM Studio", "LM Studio.exe"))
LMS_CLI = os.getenv("LMS_CLI", os.path.join(os.path.expanduser("~"), ".lmstudio", "bin",
                                            "lms.exe" if os.name == "nt" else "lms"))

# From config, so COMFY_URL / LM_STUDIO_BASE in .env move the health checks
# with the servers. These were hardcoded -- and "localhost" for LM Studio,
# which config.py documents as a ~2 s IPv6 detour on this machine.
COMFY_URL = _config.COMFY_URL.rstrip("/") + "/"
LMSTUDIO_URL = _config.LM_STUDIO_BASE.rstrip("/") + "/v1/models"
_DETACHED = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
# Without its own process group, a console CLOSE event reaches the child: ComfyUI died
# mid-render (11/28 steps) with Intel's "program aborting due to window-CLOSE event"
# when the launching console went away. Its own group makes it survive that.
_NEW_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)


def _detach_kw(flags):
    """Popen kwargs that detach a child. creationflags is Windows-only: POSIX
    Popen raises ValueError on any non-zero value, which the callers swallowed,
    so on Linux nothing was ever started. There a new session does the job."""
    if os.name == "nt":
        return {"creationflags": flags}
    return {"start_new_session": True}


def _up(url, timeout=1.5):
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except Exception:
        return False


def _spawn(path, *args):
    if not os.path.exists(path):
        return False
    try:
        subprocess.Popen([path, *args], cwd=os.path.dirname(path) or None, **_detach_kw(_DETACHED))
        return True
    except Exception:
        return False


def _spawn_comfy_source():
    """Start ComfyUI from source on COMFY_PORT. Returns True if it was launched."""
    main_py = os.path.join(COMFY_SRC, "main.py")
    if not (os.path.exists(main_py) and os.path.exists(COMFY_PY)):
        return False
    try:
        log = open(COMFY_LOG, "w", encoding="utf-8", errors="replace")
    except Exception:
        log = subprocess.DEVNULL
    try:
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        # DETACHED_PROCESS conflicts with CREATE_NO_WINDOW on Windows and can still
        # pop a visible console. Use STARTUPINFO(SW_HIDE) + CREATE_NO_WINDOW instead
        # — SW_HIDE is the most reliable way to suppress the window.
        kw = _detach_kw(_NO_WINDOW | _NEW_GROUP)
        if os.name == "nt":
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = 0  # SW_HIDE
            kw["startupinfo"] = si
        # --reserve-vram 8: live, 2026-09-20 -- with nothing reserved, ComfyUI's
        # dynamic-VRAM manager (comfy-aimdo) sometimes decides an H3 render's model
        # fits fully in VRAM ("loaded completely") when just enough is free, and
        # that path measured 6-29x SLOWER per step than a partial CPU offload
        # ("loaded partially") on the exact same model/workflow -- reproduced
        # cold on a freshly restarted server, so it is not warm-up or fragmentation.
        # Was 8GB (41s/step -> 10.5s/step on a 4-step H3 render). 2026-09-27: user
        # chose 1GB; if H3 regresses, disable NVIDIA Sysmem Fallback for python
        # (user-side setting) or reserve per model.
        subprocess.Popen(
            [COMFY_PY, main_py, "--base-directory", COMFY_BASE_DIR, "--port", COMFY_PORT,
             "--use-ck-attention", "--reserve-vram", "1",
             "--output-directory", str(OUTPUT_DIR_COMFY), "--input-directory", str(INPUT_DIR_COMFY)],
            cwd=COMFY_SRC, env=env,
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **kw)
        return True
    except Exception:
        return False


def _lock_holder() -> str:
    """' (PID n: image.exe)' for whoever wrote the lock file, '' if unknown."""
    try:
        with open(_SINGLETON_LOCK_PATH, "rb") as f:
            f.seek(1)
            pid = int(f.read(10).decode().strip())
        try:
            import psutil       # portable; tasklist exists only on Windows
            name = psutil.Process(pid).name()
        except ImportError:
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                                 capture_output=True, text=True, timeout=10,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
            name = out.split(",")[0].strip('"') if out.startswith('"') else "not running"
        except Exception:
            name = "not running"
        return f" (PID {pid}: {name})"
    except Exception:
        return ""


def main():
    # 0) refuse to run alongside an already-running instance (see the
    #    single-instance guard above for why this matters).
    if not _acquire_singleton_lock():
        with open(os.path.join(ROOT, "launch_error.log"), "a", encoding="utf-8") as f:
            f.write("launch_all.py: another instance already holds the "
                    f"singleton lock{_lock_holder()} -- exiting without starting a duplicate.\n")
        return

    # 1) ComfyUI (serves on :8000) — only needed at image-generation time.
    #    Source build first (has Ideogram 4), Desktop app as the fallback.
    if not _up(COMFY_URL):
        if not _spawn_comfy_source():
            _spawn(COMFY_EXE)

    # 2) LM Studio app + its local server (:1234) — the assistant's picker needs it
    if not _up(LMSTUDIO_URL):
        _spawn(LMSTUDIO_EXE)
        if os.path.exists(LMS_CLI):
            try:
                if os.name != "nt":
                    # headless Linux install: no app above, the daemon hosts the server
                    subprocess.run([LMS_CLI, "daemon", "up"], timeout=60,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.Popen([LMS_CLI, "server", "start"], **_detach_kw(_DETACHED))
            except Exception:
                pass

    # 3) wait briefly for the LM Studio API so the startup model list is populated
    for _ in range(25):
        if _up(LMSTUDIO_URL):
            break
        time.sleep(1)

    # 3a) Docker Desktop -- the isolated backend for run_code. Started here,
    #     not awaited: the engine needs ~5-20 s and the GUI must not wait.
    try:
        import code_runner
        if not code_runner.docker_available(timeout=4) and code_runner.desktop_binary():
            code_runner.ensure_docker(wait_s=0)
    except Exception:
        pass

    # 3a') the sandbox image, built in the background once the engine is up
    #     (first run only; a few minutes; never blocks the GUI).
    def _build_image():
        try:
            import code_runner
            if code_runner.ensure_docker(wait_s=180):
                code_runner.ensure_image()
        except Exception:
            pass
    threading.Thread(target=_build_image, name="sandbox-image", daemon=True).start()

    # 3b) self-hosted Telegram Bot API server (only with TG_API_ID/HASH in .env)
    try:
        import tg_local_api
        tg_local_api.ensure_server(ROOT)
    except Exception:
        pass

    # 4) launch the assistant GUI (in this process)
    os.environ.setdefault("USE_GUI", "1")
    import assistant
    assistant.main()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        with open(os.path.join(ROOT, "launch_error.log"), "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        raise

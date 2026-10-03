"""Is this install actually able to run? One command, one verdict.

    venv\\Scripts\\python scripts\\healthcheck.py          (Windows)
    venv/bin/python scripts/healthcheck.py               (Linux)

Run by the setup script at the end, and on its own whenever something looks
wrong. Every check says what it found and, when it is not OK, what to do. The
exit code is 0 only when nothing REQUIRED failed; optional parts (ComfyUI,
Docker, voice) are reported as WARN and never fail the run, because the app
works without them -- with those features off.

Checks, cheapest first:
  python      the interpreter is the one the project is built for
  packages    the heavy dependencies import (torch, PyQt5, f5_tts, ...)
  config      config.py loads with this .env, runtime/ is writable
  ffmpeg      on PATH and runs (voice notes, video, music all need it)
  voice       F5 weights, vocoder, reference voice (optional -> voice off)
  lmstudio    the server answers on LM_STUDIO_BASE and serves the chat model
  chat        one real completion from that model (end-to-end LLM smoke)
  comfyui     answers on COMFY_URL (optional -> no images/video/music)
  docker      engine up for run_code (optional -> code execution off)
  gui         the real main window is built and closed headlessly
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

OK, WARN, FAIL, SKIP = "OK", "WARN", "FAIL", "SKIP"
REQUIRED_PY = (3, 13)


class Result:
    def __init__(self, name, status, detail="", fix=""):
        self.name, self.status, self.detail, self.fix = name, status, detail, fix


def _http_json(url, timeout=5.0, payload=None):
    """GET (or POST payload) -> parsed JSON. Raises on anything else."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    # Never through a proxy: these are local servers, and a corporate
    # HTTP(S)_PROXY would otherwise swallow 127.0.0.1 requests.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body.decode("utf-8", errors="replace"))


# -- checks ------------------------------------------------------------------

def check_python():
    v = sys.version_info[:2]
    if v == REQUIRED_PY:
        return Result("python", OK, f"{sys.version.split()[0]} at {sys.executable}")
    return Result("python", WARN, f"{sys.version.split()[0]} (the project is built and "
                  f"tested on {REQUIRED_PY[0]}.{REQUIRED_PY[1]})",
                  "re-run the setup script; it creates venv/ with Python 3.13")


_PACKAGES = (
    ("torch", True), ("PyQt5.QtWidgets", True), ("langgraph", True),
    ("faster_whisper", True), ("f5_tts", True), ("requests", True),
    ("playwright", True), ("sounddevice", False), ("PyQt5.QtWebEngineWidgets", False),
)


def check_packages():
    out = []
    for mod, required in _PACKAGES:
        p = subprocess.run([sys.executable, "-c", f"import {mod}"], cwd=ROOT,
                           capture_output=True, text=True, timeout=300)
        if p.returncode == 0:
            continue
        err = (p.stderr or "").strip().splitlines()[-1:] or ["?"]
        out.append((mod, required, err[0][:200]))
    if not out:
        try:
            import torch
            cuda = torch.cuda.is_available()
            dev = (torch.cuda.get_device_name(0) if cuda else "CPU only")
        except Exception as exc:          # pragma: no cover - import checked above
            dev = f"torch error: {exc}"
        return Result("packages", OK, f"all import; torch device: {dev}")
    hard = [m for m in out if m[1]]
    lines = "; ".join(f"{m}: {e}" for m, _r, e in out)
    return Result("packages", FAIL if hard else WARN, lines,
                  "re-run the setup script (it reinstalls what is missing)")


def check_config():
    try:
        import config
    except Exception as exc:
        return Result("config", FAIL, f"config.py does not load: {exc}",
                      "fix the value named above in .env")
    try:
        probe = tempfile.NamedTemporaryFile(dir=str(config.OUTPUT_DIR), delete=True)
        probe.close()
    except Exception as exc:
        return Result("config", FAIL, f"runtime folder not writable: {config.OUTPUT_DIR} ({exc})",
                      "check the folder's permissions or set ASSISTANT_OUTPUT_DIR")
    env = os.path.join(ROOT, ".env")
    return Result("config", OK, (".env loaded" if os.path.exists(env) else "no .env (defaults)")
                  + f"; runtime at {config.OUTPUT_DIR}")


def check_ffmpeg():
    exe = shutil.which("ffmpeg")
    if not exe:
        return Result("ffmpeg", FAIL, "not on PATH",
                      "Windows: winget install Gyan.FFmpeg  |  Linux: sudo apt install ffmpeg"
                      "  (then open a new terminal)")
    try:
        p = subprocess.run([exe, "-hide_banner", "-version"], capture_output=True,
                           text=True, timeout=30)
        first = (p.stdout or "").splitlines()[:1]
        if p.returncode != 0:
            return Result("ffmpeg", FAIL, f"{exe} exits {p.returncode}", "reinstall ffmpeg")
        return Result("ffmpeg", OK, first[0] if first else exe)
    except Exception as exc:
        return Result("ffmpeg", FAIL, f"{exe}: {exc}", "reinstall ffmpeg")


def check_voice():
    import config
    missing = []
    if not config.WEIGHTS_PATH.exists():
        missing.append(f"F5-TTS weights ({config.WEIGHTS_PATH.name})")
    if not (config.VOCOS_DIR / "config.yaml").exists():
        missing.append("vocoder (vocos/)")
    if not config.DC_REF_WAV.exists():
        missing.append(f"reference voice ({config.DC_REF_WAV})")
    if not missing:
        return Result("voice", OK, "weights, vocoder and reference voice present")
    hints = []
    if any(not m.startswith("reference voice") for m in missing):
        hints.append("run the setup script again: it downloads the vocoder and the "
                     "F5-TTS weights (models/f5/)")
    if not config.DC_REF_WAV.exists():
        hints.append("put a 5-15 s clean recording at that path, or set ASSISTANT_REF_WAV in .env")
    return Result("voice", WARN, "voice output OFF -- missing " + ", ".join(missing),
                  "; ".join(hints))


def _served_ids(base):
    data = _http_json(base.rstrip("/") + "/v1/models", timeout=4)
    if not isinstance(data, dict) or "data" not in data:
        raise ValueError("answered, but not like LM Studio (no 'data' list) -- "
                         "is another program using that port?")
    return [m.get("id", "") for m in data.get("data") or []]


def check_lmstudio():
    import config
    base = config.LM_STUDIO_BASE
    try:
        ids = _served_ids(base)
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
        return Result("lmstudio", FAIL, f"no server at {base} ({getattr(exc, 'reason', exc)})",
                      "start LM Studio and its server (the setup script and the launcher "
                      "do this: `lms server start`)")
    except ValueError as exc:
        return Result("lmstudio", FAIL, f"{base}: {exc}",
                      "free the port, or point LM_STUDIO_BASE at the real server")
    want = config.MODEL_NAME
    if want in ids:
        return Result("lmstudio", OK, f"{base} serves {want}")
    return Result("lmstudio", FAIL, f"{base} is up but does not serve {want!r} "
                  f"(has: {', '.join(i for i in ids if i)[:200] or 'nothing'})",
                  f"lms get {want}   (or set MODEL_NAME in .env to one you have)")


def check_chat():
    """One real completion: the whole LLM path, not just a port."""
    import config
    payload = {"model": config.MODEL_NAME, "max_tokens": 16, "temperature": 0,
               "messages": [{"role": "user", "content": "Reply with the single word: ready"}]}
    t0 = time.monotonic()
    try:
        data = _http_json(config.LM_STUDIO_URL, timeout=300, payload=payload)
        text = (data["choices"][0]["message"].get("content") or
                data["choices"][0]["message"].get("reasoning_content") or "")
    except Exception as exc:
        return Result("chat", FAIL, f"completion failed: {exc}",
                      "open LM Studio and check that the model loads (VRAM, context size)")
    return Result("chat", OK, f"model answered in {time.monotonic() - t0:.1f}s: "
                  f"{text.strip()[:40]!r}")


def check_comfyui():
    import config
    try:
        _http_json(config.COMFY_URL.rstrip("/") + "/system_stats", timeout=4)
        return Result("comfyui", OK, f"answers at {config.COMFY_URL}")
    except Exception:
        return Result("comfyui", WARN, f"not reachable at {config.COMFY_URL} -- images, "
                      "video and music are off",
                      "the launcher starts it; if it is not installed, run the setup "
                      "script again (it installs ComfyUI, its nodes and models)")


def check_docker():
    try:
        import code_runner
        if code_runner.docker_available(timeout=8):
            return Result("docker", OK, "engine up (run_code is isolated)")
        exe = code_runner.docker_binary()
        if exe:
            if os.name == "nt":
                return Result("docker", WARN, "installed, engine not running -- run_code off",
                              "start Docker Desktop (the launcher does it)")
            # Linux: the launcher cannot start a root daemon, and a user outside
            # the docker group sees a running engine as "not running".
            err = _docker_info_error(exe)
            if "permission denied" in err.lower():
                return Result("docker", WARN, "engine running but not usable by this user -- run_code off",
                              "sudo usermod -aG docker $USER, then log out and back in")
            return Result("docker", WARN, "installed, engine not running -- run_code off",
                          "sudo systemctl enable --now docker")
    except Exception as exc:
        return Result("docker", WARN, f"check failed: {exc}")
    return Result("docker", WARN, "not installed -- run_code (code execution) off",
                  "optional: install Docker Desktop" if os.name == "nt"
                  else "optional: install Docker Engine (https://docs.docker.com/engine/install/)")


def _docker_info_error(exe: str) -> str:
    try:
        p = subprocess.run([exe, "info"], capture_output=True, text=True, timeout=8)
        return (p.stderr or "") + (p.stdout or "")
    except Exception as exc:
        return str(exc)


_GUI_CHILD = textwrap.dedent(r'''
    import os, sys
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["USE_GUI"] = "0"
    sys.path.insert(0, %(root)r)
    import faulthandler; faulthandler.enable()   # a native crash prints its Python stack
    import logging; logging.basicConfig(level=logging.CRITICAL)
    import win_runtime; win_runtime.preload_newest_msvcp()   # as the app does, before PyQt5
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QTimer
    import gui
    app = QApplication.instance() or QApplication(sys.argv)
    real = gui.ModelLoader
    gui.ModelLoader = type("NoopLoader", (real,), {"start": lambda s: None, "run": lambda s: None})
    w = gui.AssistantWindow("healthcheck", True, "high")
    gui.ModelLoader = real
    w.show()
    QTimer.singleShot(300, w.close)
    QTimer.singleShot(1500, app.quit)
    app.exec_()
    print("GUI-OK", flush=True)
    os._exit(0)
''')


def check_gui(timeout=180):
    with tempfile.TemporaryDirectory(prefix="hc_gui_") as tmp:
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                   ASSISTANT_OUTPUT_DIR=os.path.join(tmp, "runtime"))
        try:
            p = subprocess.run([sys.executable, "-c", _GUI_CHILD % {"root": ROOT}],
                               cwd=ROOT, env=env, capture_output=True, text=True,
                               errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired:
            return Result("gui", FAIL, f"main window did not open within {timeout}s")
    if p.returncode == 0 and "GUI-OK" in p.stdout:
        return Result("gui", OK, "main window builds and closes cleanly (headless)")
    err = (p.stderr or "") + (p.stdout or "")
    lines = err.strip().splitlines()
    # faulthandler's dump ("Windows fatal exception" / "Fatal Python error" +
    # the stack) says where a native crash happened; keep it whole.
    start = next((i for i, ln in enumerate(lines) if "fatal" in ln.lower()), None)
    tail = lines[start:start + 25] if start is not None else lines[-3:]
    sep = "\n               " if start is not None else " | "
    return Result("gui", FAIL, f"exit {p.returncode}: " + sep.join(tail)[:3000],
                  "see crash.log in the project folder")


ALL = ("python", "packages", "config", "ffmpeg", "voice", "lmstudio", "chat",
       "comfyui", "docker", "gui")


def run(only=None, skip=(), quiet=False):
    """Run the checks; returns the Result list."""
    funcs = {
        "python": check_python, "packages": check_packages, "config": check_config,
        "ffmpeg": check_ffmpeg, "voice": check_voice, "lmstudio": check_lmstudio,
        "chat": check_chat, "comfyui": check_comfyui, "docker": check_docker,
        "gui": check_gui,
    }
    results = []
    names = [n for n in ALL if (not only or n in only) and n not in skip]
    for name in names:
        if name == "chat" and any(r.name == "lmstudio" and r.status != OK for r in results):
            r = Result("chat", SKIP, "LM Studio is not ready")
        elif name in ("voice", "lmstudio", "chat", "comfyui", "gui") and any(
                r.name in ("packages", "config") and r.status == FAIL for r in results):
            r = Result(name, SKIP, "fix the failures above first")
        else:
            try:
                r = funcs[name]()
            except Exception as exc:   # a check that crashes is a failed check, not a crash
                r = Result(name, FAIL, f"check crashed: {type(exc).__name__}: {exc}")
        results.append(r)
        if not quiet:
            print(format_result(r), flush=True)
    return results


def format_result(r):
    line = f"  [{r.status:<4}] {r.name:<9} {r.detail}"
    if r.fix and r.status in (WARN, FAIL):
        line += f"\n               -> {r.fix}"
    return line


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="*", choices=ALL, help="run just these checks")
    ap.add_argument("--skip", nargs="*", choices=ALL, default=[], help="skip these checks")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("Assistant DC health check")
    results = run(args.only, args.skip)
    failed = [r for r in results if r.status == FAIL]
    warned = [r for r in results if r.status == WARN]
    print()
    if failed:
        print(f"NOT READY: {len(failed)} required check(s) failed: "
              + ", ".join(r.name for r in failed))
        return 1
    print("READY" + (f" (with {len(warned)} optional part(s) off: "
                     + ", ".join(r.name for r in warned) + ")" if warned else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Install, configure and verify Assistant DC. Safe to run again at any time.

You normally do not run this file yourself: setup.ps1 (Windows) / setup.sh
(Linux) create venv/ with Python 3.13 and then hand over to it. Every step
below first looks at what is already there and only does the missing part,
so a second run is quick and an interrupted run simply continues.

    python scripts/setup.py [--no-models] [--no-install] [--no-start] [--cpu]

Steps:
   1. environment   OS, CPU, GPU
   2. vcruntime     current Visual C++ runtime (Windows; stops a 0xc0000005 crash)
   3. torch         CUDA build when an NVIDIA GPU is present, CPU build otherwise
   4. packages      requirements.txt (+ overrides.txt)
   5. import path   source folders on the venv's path (install_paths)
   6. browser       Playwright's Chromium (web search / research)
   7. config        .env from .env.example, never overwritten
   8. ffmpeg        on PATH (installed with winget on Windows)
   9. voice         vocoder + Russian F5-TTS weights (Misha24-10, v4 winter)
  10. lmstudio      LM Studio (winget / headless on Linux), server, chat + embedding models
  10b. comfyui      ComfyUI v0.38.2, pinned custom nodes, the graphs' model files
  11. shortcut      desktop shortcut (Windows)
  12. health        scripts/healthcheck.py -- the verdict
  13. start         the app (unless --no-start)
"""
from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
IS_WIN = os.name == "nt"

TORCH_PINS = ("torch==2.8.0", "torchaudio==2.8.0", "torchvision==0.23.0")
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
VOCOS_REPO = "charactr/vocos-mel-24khz"
VOCOS_FILES = ("config.yaml", "pytorch_model.bin")
EMBED_MODEL_DEFAULT = "text-embedding-bge-m3"
# What `lms get` fetches for a served model id, in order of preference. The id
# LM Studio serves is derived from the repo name, so the house model's id comes
# from the first repo; the QAT/MTP release is the same model if that one is gone.
MODEL_SOURCES = {
    "gemma4-26b-a4b-uncensored-hauhaucs-balanced": [
        "https://huggingface.co/HauhauCS/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced",
        "https://huggingface.co/HauhauCS/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP",
    ],
}
# The oldest msvcp140 torch & co. are happy with. PyQt5 bundles 14.26.
MIN_VCRUNTIME = (14, 40)

STEPS: list = []          # (name, status, detail)


# -- output --------------------------------------------------------------------

def _p(msg=""):
    print(msg, flush=True)


def step(name):
    _p(f"\n== {name} " + "=" * max(3, 60 - len(name)))


def record(name, status, detail=""):
    STEPS.append((name, status, detail))
    mark = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL", "skip": "SKIP"}[status]
    _p(f"  [{mark}] {detail}" if detail else f"  [{mark}]")


class SetupError(Exception):
    """A step that cannot continue. Carries what to do about it."""


def run(cmd, timeout=3600, check=True, env=None, quiet=False, retries=0):
    """Run a command, streaming its output. Raises SetupError on failure.

    `retries` re-runs a command that failed (network installs): uv and
    pip resume where they stopped, so a retry is cheap.
    """
    shown = " ".join(str(c) for c in cmd)
    for attempt in range(retries + 1):
        if not quiet:
            _p(f"  $ {shown}")
        try:
            p = subprocess.run([str(c) for c in cmd], cwd=str(ROOT), env=env, timeout=timeout,
                               stdout=subprocess.PIPE if quiet else None,
                               stderr=subprocess.STDOUT if quiet else None,
                               text=True, errors="replace")
        except FileNotFoundError:
            raise SetupError(f"{cmd[0]} not found")
        except subprocess.TimeoutExpired:
            if attempt < retries:
                continue
            raise SetupError(f"timed out after {timeout}s: {shown}")
        if p.returncode == 0 or not check:
            return p
        if attempt < retries:
            _p(f"  ... failed (exit {p.returncode}); retrying ({attempt + 1}/{retries})")
            time.sleep(5 * (attempt + 1))
            continue
        tail = ("\n" + "\n".join((p.stdout or "").strip().splitlines()[-15:])) if quiet else ""
        raise SetupError(f"exit {p.returncode}: {shown}{tail}")


def uv_exe():
    exe = shutil.which("uv")
    if exe:
        return exe
    for cand in (Path.home() / ".local" / "bin" / ("uv.exe" if IS_WIN else "uv"),
                 Path.home() / ".cargo" / "bin" / ("uv.exe" if IS_WIN else "uv")):
        if cand.exists():
            return str(cand)
    return None


def pip_install(args, retries=2, timeout=3600):
    """uv pip install into THIS interpreter; pip only when uv is missing."""
    uv = uv_exe()
    if uv:
        return run([uv, "pip", "install", "--python", sys.executable, *args],
                   retries=retries, timeout=timeout)
    if "--override" in args:
        raise SetupError("uv is required (overrides.txt cannot be applied by pip). "
                         "Run setup.ps1 / setup.sh, which installs uv.")
    return run([sys.executable, "-m", "pip", "install", *args], retries=retries, timeout=timeout)


# -- 1. environment --------------------------------------------------------------

def detect_gpu():
    """(name, vram_mb) of the first NVIDIA GPU, or None."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        p = subprocess.run([exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        first = (p.stdout or "").strip().splitlines()[0]
        name, mem = [x.strip() for x in first.split(",")[:2]]
        return name, int(float(mem))
    except Exception:
        return None


def s_environment(args):
    step("1. environment")
    gpu = None if args.cpu else detect_gpu()
    _p(f"  {platform.system()} {platform.release()} {platform.machine()}, "
       f"Python {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info[:2] != (3, 13):
        record("environment", "warn", "Python is not 3.13; the project is pinned and tested on "
               "3.13 (setup.ps1 / setup.sh create a 3.13 venv)")
    if sys.prefix == sys.base_prefix:
        raise SetupError("not running inside a virtual environment -- run setup.ps1 / "
                         "setup.sh, which create venv/ and call this script from it")
    if gpu:
        record("environment", "ok", f"NVIDIA {gpu[0]}, {gpu[1] // 1024} GB VRAM")
        if gpu[1] < 20000:
            _p("  note: tested on 24 GB; with less, video generation will not fit and the "
               "chat model may need a smaller variant (MODEL_NAME in .env)")
    else:
        record("environment", "warn", "no NVIDIA GPU found -- CPU mode: chat works through "
               "LM Studio, voice is slow, image/video generation is unavailable")
    return gpu


# -- 2. Visual C++ runtime (Windows) ---------------------------------------------

def s_vcruntime(args):
    step("2. Visual C++ runtime")
    if not IS_WIN:
        record("vcruntime", "skip", "not Windows")
        return
    sys.path.insert(0, str(ROOT / "core"))
    import win_runtime
    sys_dll = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "msvcp140.dll")
    ver = win_runtime._version(sys_dll) if os.path.exists(sys_dll) else ()
    if ver and ver[:2] >= MIN_VCRUNTIME:
        record("vcruntime", "ok", "msvcp140 %s" % ".".join(map(str, ver)))
        return
    have = ".".join(map(str, ver)) if ver else "missing"
    if args.no_install or not shutil.which("winget"):
        record("vcruntime", "warn", f"system msvcp140 is {have}; PyQt5's bundled 14.26 can then "
               "crash the app (0xc0000005). Install the latest 'Microsoft Visual C++ "
               "Redistributable x64' from https://aka.ms/vs/17/release/vc_redist.x64.exe")
        return
    _p(f"  system msvcp140 is {have} -- installing the current runtime with winget")
    p = run(["winget", "install", "--id", "Microsoft.VCRedist.2015+.x64", "-e", "--silent",
             "--accept-package-agreements", "--accept-source-agreements"], check=False)
    ver = win_runtime._version(sys_dll) if os.path.exists(sys_dll) else ()
    if ver and ver[:2] >= MIN_VCRUNTIME:
        record("vcruntime", "ok", "installed msvcp140 %s" % ".".join(map(str, ver)))
    else:
        record("vcruntime", "warn", f"winget exit {p.returncode}; runtime still {have}. "
               "Install https://aka.ms/vs/17/release/vc_redist.x64.exe by hand")


# -- 3. torch ----------------------------------------------------------------------

def _torch_state():
    """(version, cuda_build) of the installed torch, or (None, None)."""
    p = subprocess.run([sys.executable, "-c",
                        "import torch,sys; sys.stdout.write(torch.__version__+'|'+str(torch.version.cuda))"],
                       capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        return None, None
    ver, _, cuda = p.stdout.strip().rpartition("\n")[-1].partition("|")
    return ver, (None if cuda in ("None", "") else cuda)


def s_torch(args, gpu):
    step("3. torch")
    want_cuda = bool(gpu) and platform.machine().lower() in ("amd64", "x86_64")
    ver, cuda = _torch_state()
    want_ver = TORCH_PINS[0].split("==")[1]
    if ver and ver.split("+")[0] == want_ver and (bool(cuda) == want_cuda or not want_cuda):
        record("torch", "ok", f"torch {ver} already installed ({'CUDA ' + cuda if cuda else 'CPU'})")
        return
    index = TORCH_CUDA_INDEX if want_cuda else TORCH_CPU_INDEX
    _p(f"  installing torch {want_ver} ({'CUDA 12.8' if want_cuda else 'CPU'}) -- a large download")
    try:
        # --reinstall only when switching CPU<->CUDA: same version string, other build.
        extra = [a for n in ("torch", "torchaudio", "torchvision")
                 for a in ("--reinstall-package", n)] if ver and ver.split("+")[0] == want_ver else []
        pip_install([*TORCH_PINS, "--index-url", index, *extra], retries=2)
    except SetupError as exc:
        # download.pytorch.org unreachable (proxy, firewall): PyPI carries the
        # same version -- CPU on Windows, CUDA on Linux.
        _p(f"  pytorch.org index failed ({str(exc).splitlines()[0]}); trying PyPI")
        pip_install(list(TORCH_PINS), retries=2)
    ver, cuda = _torch_state()
    if not ver:
        raise SetupError("torch did not install")
    if want_cuda and not cuda:
        record("torch", "warn", f"torch {ver} is a CPU build although an NVIDIA GPU is present "
               "(pytorch.org was unreachable); re-run setup when it is")
    else:
        record("torch", "ok", f"torch {ver} ({'CUDA ' + cuda if cuda else 'CPU'})")


# -- 4. packages -------------------------------------------------------------------

def s_packages(args):
    step("4. packages")
    reqs = ROOT / ("requirements-dev.txt" if args.dev else "requirements.txt")
    pip_install(["-r", str(reqs), "--override", str(ROOT / "overrides.txt")], retries=2)
    record("packages", "ok", f"{reqs.name} satisfied")


# -- 5. import path ----------------------------------------------------------------

def s_import_path(args):
    step("5. import path")
    import install_paths
    target = install_paths.ensure()
    if not target:
        raise SetupError("could not write the import-path file into the venv")
    p = subprocess.run([sys.executable, "-c", "import config, tg_bot"], cwd=str(Path.home()),
                       capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        raise SetupError("the source folders are still not importable:\n"
                         + "\n".join(p.stderr.strip().splitlines()[-5:]))
    record("import path", "ok", str(target))


# -- 6. browser --------------------------------------------------------------------

def s_browser(args):
    step("6. browser (Playwright Chromium)")
    _p("  installing Chromium for web pages (a few minutes the first time)")
    try:
        run([sys.executable, "-m", "playwright", "install", "chromium"], retries=2, timeout=1800,
            quiet=True)
        record("browser", "ok", "Chromium ready")
    except SetupError as exc:
        record("browser", "warn", f"Chromium not installed ({str(exc).splitlines()[0]}); "
               "web pages that need a browser will be skipped")


# -- 7. config ---------------------------------------------------------------------

def s_config(args):
    step("7. config")
    env, example = ROOT / ".env", ROOT / ".env.example"
    if env.exists():
        have = set(re.findall(r"^\s*([A-Z0-9_]+)\s*=", env.read_text(encoding="utf-8", errors="replace"), re.M))
        new = [k for k in re.findall(r"^\s*([A-Z0-9_]+)\s*=", example.read_text(encoding="utf-8"), re.M)
               if k not in have]
        record("config", "ok", ".env kept as it is" + (
            f" (newer settings available in .env.example: {', '.join(new[:8])}"
            f"{'...' if len(new) > 8 else ''})" if new else ""))
    else:
        shutil.copyfile(example, env)
        record("config", "ok", "created .env from .env.example (edit it to change settings)")
    for d in ("runtime", "models"):
        (ROOT / d).mkdir(exist_ok=True)


# -- 7b. Linux system libraries ----------------------------------------------------

def _apt_install(pkgs, args) -> bool:
    """apt-get install pkgs when that can run unattended (root, or sudo without
    a password). -> whether it ran and succeeded."""
    if IS_WIN or args.no_install or not pkgs or not shutil.which("apt-get"):
        return False
    if os.geteuid() == 0:
        pre = []
    elif shutil.which("sudo") and subprocess.run(["sudo", "-n", "true"], capture_output=True,
                                                 timeout=15).returncode == 0:
        pre = ["sudo", "-n"]
    else:
        return False
    env = dict(os.environ, DEBIAN_FRONTEND="noninteractive")
    _p(f"  apt-get install {' '.join(pkgs)}")
    subprocess.run(pre + ["apt-get", "update", "-q"], capture_output=True, timeout=600, env=env)
    p = subprocess.run(pre + ["apt-get", "install", "-y", "-q", *pkgs], capture_output=True,
                       text=True, timeout=1800, env=env)
    return p.returncode == 0


def _deb_package(soname: str) -> str:
    """libxcb-icccm.so.4 -> libxcb-icccm4; libxkbcommon-x11.so.0 -> libxkbcommon-x11-0."""
    stem, _, ver = soname.partition(".so.")
    ver = ver.split(".")[0]
    name = f"{stem}{'-' if stem[-1:].isdigit() else ''}{ver}" if ver else stem
    return name.lower()


def _qt_missing_libs() -> list:
    """Shared libraries Qt's X11 plugin needs that this machine lacks."""
    try:
        import PyQt5
    except ImportError:
        return []
    plug = Path(PyQt5.__file__).parent / "Qt5" / "plugins" / "platforms" / "libqxcb.so"
    if not plug.exists() or not shutil.which("ldd"):
        return []
    out = subprocess.run(["ldd", str(plug)], capture_output=True, text=True, timeout=60).stdout
    return sorted({ln.split()[0] for ln in out.splitlines() if "not found" in ln})


def s_qt_libs(args):
    if IS_WIN or sys.platform == "darwin":
        return
    step("7b. window libraries (Linux)")
    missing = _qt_missing_libs()
    if missing:
        pkgs = sorted({_deb_package(m) for m in missing})
        if _apt_install(pkgs, args):
            missing = _qt_missing_libs()
        if missing:
            record("qt-libs", "warn", "the app window cannot open (Qt's xcb plugin lacks "
                   f"{', '.join(missing)}): sudo apt install {' '.join(pkgs)}  "
                   "(Telegram and the headless parts work without it)")
            return
    record("qt-libs", "ok", "Qt can open a window")


# -- 8. ffmpeg ---------------------------------------------------------------------

def _winget_links():
    return Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links"


def s_ffmpeg(args):
    step("8. ffmpeg")
    if IS_WIN and _winget_links().is_dir():
        os.environ["PATH"] = str(_winget_links()) + os.pathsep + os.environ.get("PATH", "")
    exe = shutil.which("ffmpeg")
    if not exe and IS_WIN and not args.no_install and shutil.which("winget"):
        run(["winget", "install", "--id", "Gyan.FFmpeg", "-e", "--silent",
             "--accept-package-agreements", "--accept-source-agreements"], check=False)
        if _winget_links().is_dir():
            os.environ["PATH"] = str(_winget_links()) + os.pathsep + os.environ["PATH"]
        exe = shutil.which("ffmpeg")
    if not exe and not IS_WIN and _apt_install(["ffmpeg"], args):
        exe = shutil.which("ffmpeg")
    if exe:
        record("ffmpeg", "ok", exe)
        return
    how = ("winget install Gyan.FFmpeg" if IS_WIN else
           "sudo apt install ffmpeg   (or your distribution's package manager)")
    raise SetupError(f"ffmpeg is not installed and could not be installed automatically: {how}")


# -- 9. voice ----------------------------------------------------------------------

def _hf_download(repo, filename, dest_dir):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo, filename=filename, local_dir=str(dest_dir))


F5_REPO = "Misha24-10/F5-TTS_RUSSIAN"
F5_FOLDER = "F5TTS_v1_Base_v4_winter"


def _download_f5(dest: Path) -> str:
    """The newest checkpoint (+ vocab.txt) of the Russian F5-TTS folder into dest.

    The file name is not hard-coded: the folder is listed and the .safetensors
    with the highest step number wins, so a re-upload under a new step still
    installs. F5_WEIGHTS_REPO / F5_WEIGHTS_DIR / F5_WEIGHTS_FILE override."""
    from huggingface_hub import list_repo_files
    repo = os.getenv("F5_WEIGHTS_REPO", "").strip() or F5_REPO
    folder = os.getenv("F5_WEIGHTS_DIR", "").strip().strip("/") or F5_FOLDER
    files = [f for f in list_repo_files(repo) if f.startswith(folder + "/")]
    pick = os.getenv("F5_WEIGHTS_FILE", "").strip()
    if pick:
        pick = pick if "/" in pick else f"{folder}/{pick}"
    else:
        cands = [f for f in files if f.endswith(".safetensors")]
        if not cands:
            raise FileNotFoundError(f"no .safetensors in {repo}/{folder}")
        pick = max(cands, key=lambda f: [int(n) for n in re.findall(r"\d+", Path(f).stem)][-1:] or [-1])
    stage = ROOT / "models" / ".f5_download"
    dest.mkdir(parents=True, exist_ok=True)
    wanted = [pick] + [f for f in files if Path(f).name == "vocab.txt"]
    for f in wanted:
        _p(f"  downloading {repo}/{f}")
        got = Path(_hf_download(repo, f, stage))
        shutil.move(str(got), str(dest / Path(f).name))
    shutil.rmtree(stage, ignore_errors=True)
    return Path(pick).name


def _weights_now() -> Path:
    import importlib
    config = importlib.import_module("config")
    return Path(config._f5_weights())


def s_voice(args):
    step("9. voice")
    import importlib
    os.environ.setdefault("PYTHONUTF8", "1")
    config = importlib.import_module("config")
    notes = []
    vocos = Path(config.VOCOS_DIR)
    if all((vocos / f).exists() for f in VOCOS_FILES):
        notes.append("vocoder present")
    elif args.no_models:
        notes.append("vocoder missing (--no-models)")
    else:
        try:
            vocos.mkdir(parents=True, exist_ok=True)
            for f in VOCOS_FILES:
                if not (vocos / f).exists():
                    _p(f"  downloading {VOCOS_REPO}/{f}")
                    _hf_download(VOCOS_REPO, f, vocos)
            notes.append("vocoder downloaded")
        except Exception as exc:
            notes.append(f"vocoder download failed ({type(exc).__name__}: {str(exc)[:120]})")
    f5_dir = Path(config.F5_DIR)
    have = sorted(f5_dir.glob("*.safetensors")) if f5_dir.is_dir() else []
    if have:
        notes.append(f"F5 weights present ({have[-1].name})")
    elif args.no_models:
        notes.append("F5 weights not downloaded (--no-models)")
    else:
        try:
            got = _download_f5(f5_dir)
            notes.append(f"F5 weights downloaded ({got})")
        except Exception as exc:
            notes.append(f"F5 weights download failed ({type(exc).__name__}: {str(exc)[:160]}) "
                         "-- run the setup again to retry")
    weights = _weights_now()
    ref = Path(config.DC_REF_WAV)
    if not ref.exists():
        notes.append("no reference voice: set ASSISTANT_REF_WAV in .env to a 5-15 s recording")
    ok = weights.exists() and ref.exists() and all((vocos / f).exists() for f in VOCOS_FILES)
    record("voice", "ok" if ok else "warn",
           ("; ".join(notes)) + ("" if ok else " -- the app runs with voice output off"))


# -- 10. LM Studio -----------------------------------------------------------------

def lms_exe():
    exe = shutil.which("lms")
    if exe:
        return exe
    cand = Path.home() / ".lmstudio" / "bin" / ("lms.exe" if IS_WIN else "lms")
    return str(cand) if cand.exists() else None


def lmstudio_app():
    if IS_WIN:
        cand = Path(os.getenv("LMSTUDIO_EXE", Path(os.environ.get("LOCALAPPDATA", "")) /
                              "Programs" / "LM Studio" / "LM Studio.exe"))
        return str(cand) if cand.exists() else None
    for cand in (os.getenv("LMSTUDIO_EXE", ""), str(Path.home() / "Applications" / "LM-Studio.AppImage")):
        if cand and os.path.exists(cand):
            return cand
    return None


def _served(base):
    import json
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(base.rstrip("/") + "/v1/models", timeout=4) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
        return [m.get("id", "") for m in data.get("data") or []]
    except Exception:
        return None


def _wait(pred, timeout, every=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = pred()
        if v:
            return v
        time.sleep(every)
    return pred()


def _downloaded_models(lms):
    p = subprocess.run([lms, "ls"], capture_output=True, text=True, timeout=120,
                       errors="replace")
    return p.stdout or ""


def s_lmstudio(args):
    step("10. LM Studio (the chat model)")
    import importlib
    config = importlib.import_module("config")
    base = config.LM_STUDIO_BASE
    lms = lms_exe()
    if not lms and not lmstudio_app() and IS_WIN and not args.no_install and shutil.which("winget"):
        _p("  LM Studio is not installed -- installing with winget")
        run(["winget", "install", "--id", "ElementLabs.LMStudio", "-e", "--silent",
             "--accept-package-agreements", "--accept-source-agreements"], check=False)
    if not lms and not IS_WIN and not args.no_install and shutil.which("curl"):
        # Linux/macOS: LM Studio's headless build (llmster daemon + lms CLI),
        # no desktop app needed; it lands in ~/.lmstudio/bin.
        _p("  LM Studio is not installed -- installing the headless build")
        run(["bash", "-c", "curl -fsSL https://lmstudio.ai/install.sh | bash"], check=False,
            retries=1, timeout=1800)
        lms = lms_exe()
        if lms:
            subprocess.run([lms, "daemon", "up"], capture_output=True, text=True, timeout=180)
    app = lmstudio_app()
    if not lms and app:
        # The app installs its `lms` CLI on first launch.
        _p("  starting LM Studio once so it installs its command-line tool")
        subprocess.Popen([app], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, cwd=os.path.dirname(app),
                         creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
        lms = _wait(lms_exe, 90)
    if not lms:
        record("lmstudio", "warn" if args.no_models else "fail",
               "LM Studio is not installed. Get it from https://lmstudio.ai, "
               "start it once, then run setup again")
        return
    if _served(base) is None:
        _p("  starting the LM Studio server")
        if not IS_WIN:
            subprocess.run([lms, "daemon", "up"], capture_output=True, text=True, timeout=180)
        subprocess.run([lms, "server", "start"], capture_output=True, text=True, timeout=120)
        if not _wait(lambda: _served(base) is not None, 60):
            record("lmstudio", "warn" if args.no_models else "fail", f"the LM Studio server does not answer at {base} -- "
                   "open LM Studio > Developer > Start Server (port 1234, or set LM_STUDIO_BASE)")
            return
    if _served(base) is None:
        record("lmstudio", "fail", f"nothing answers at {base}")
        return
    listing = _downloaded_models(lms)
    wanted = [config.MODEL_NAME, os.getenv("EMBED_MODEL", EMBED_MODEL_DEFAULT)]
    missing = [m for m in wanted if m.split("/")[-1].lower() not in listing.lower()]
    if missing and args.no_models:
        record("lmstudio", "warn", f"server up; not downloaded (--no-models): {', '.join(missing)}")
        return
    for m in missing:
        _p(f"  downloading {m} (the chat model is ~17 GB; this takes a while)")
        err = None
        for src in MODEL_SOURCES.get(m, [m]):
            try:
                run([lms, "get", src, "--yes"], timeout=6 * 3600, retries=1)
                err = None
                break
            except SetupError as exc:
                err = exc
        if err is not None:
            record("lmstudio", "fail", f"could not download {m}: {str(err).splitlines()[0]}. "
                   f"Download it in LM Studio's search tab, or set MODEL_NAME in .env")
            return
    record("lmstudio", "ok", f"server at {base}; models: {', '.join(wanted)}")


# -- 10b. ComfyUI -------------------------------------------------------------------

def s_comfyui(args, gpu):
    step("10b. ComfyUI (pictures, video, songs)")
    if args.no_comfy:
        record("comfyui", "skip", "--no-comfy")
        return
    import comfy_setup as C
    uv = uv_exe()
    if not uv:
        record("comfyui", "fail", "uv not found")
        return
    want_cuda = bool(gpu) and not args.cpu
    index = TORCH_CUDA_INDEX if want_cuda else TORCH_CPU_INDEX
    try:
        st, detail = C.install_server(uv, index, _p)
        if st != "ok":
            record("comfyui", st, detail)
            return
        st2, detail2 = C.install_nodes(uv, _p)
        notes = [detail, detail2]
        worst = st2
        if args.no_models:
            notes.append("model files not downloaded (--no-models)")
        else:
            media = tuple(m for m in args.media.split(",") if m in C.MEDIA)
            st3, detail3 = C.download_models(media, _p)
            notes.append(detail3)
            worst = "warn" if "warn" in (st2, st3) else "ok"
        record("comfyui", worst, "; ".join(notes))
    except Exception as exc:
        record("comfyui", "warn", f"ComfyUI setup stopped ({type(exc).__name__}: "
               f"{str(exc)[:200]}) -- images/video/songs stay off; run the setup again")


# -- 11. shortcut ------------------------------------------------------------------

def s_shortcut(args):
    step("11. desktop shortcut")
    if not IS_WIN:
        record("shortcut", "skip", "start with ./start.sh")
        return
    pyw = Path(sys.executable).with_name("pythonw.exe")
    target = pyw if pyw.exists() else Path(sys.executable)
    ps = (
        "$s=(New-Object -ComObject WScript.Shell).CreateShortcut("
        "[IO.Path]::Combine([Environment]::GetFolderPath('Desktop'),'Assistant DC.lnk'));"
        f"$s.TargetPath='{target}';"
        f"$s.Arguments='\"{ROOT / 'scripts' / 'launch_all.py'}\"';"
        f"$s.WorkingDirectory='{ROOT}';"
        f"$s.IconLocation='{ROOT / 'assets' / 'assistant.ico'}';"
        "$s.Save()")
    p = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
                       capture_output=True, text=True, timeout=60)
    if p.returncode == 0:
        record("shortcut", "ok", "'Assistant DC' on the desktop")
    else:
        record("shortcut", "warn", f"could not create it ({(p.stderr or '').strip()[:120]}); "
               "start with start.cmd")


# -- 12. health --------------------------------------------------------------------

def s_health(args):
    step("12. health check")
    skip = ["python"]
    if args.no_models:
        # CI / offline install: no chat model is expected yet.
        skip += ["lmstudio", "chat"]
    cmd = [sys.executable, str(ROOT / "scripts" / "healthcheck.py"), "--skip", *skip]
    p = subprocess.run(cmd, cwd=str(ROOT), timeout=1800)
    record("health", "ok" if p.returncode == 0 else "fail",
           "all required checks passed" if p.returncode == 0 else "see the checks above")
    return p.returncode == 0


# -- 13. start ---------------------------------------------------------------------

def _has_display():
    return IS_WIN or bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def start_app():
    pyw = Path(sys.executable).with_name("pythonw.exe")
    exe = str(pyw if (IS_WIN and pyw.exists()) else sys.executable)
    kw = {"creationflags": getattr(subprocess, "DETACHED_PROCESS", 0)} if IS_WIN else {"start_new_session": True}
    subprocess.Popen([exe, str(ROOT / "scripts" / "launch_all.py")], cwd=str(ROOT),
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, **kw)


def how_to_use():
    start = ("double-click 'Assistant DC' on the desktop, or run start.cmd" if IS_WIN
             else "./start.sh")
    check = "venv\\Scripts\\python" if IS_WIN else "venv/bin/python"
    _p(f"""
How to use it
  start      {start}
  stop       close the window (LM Studio / ComfyUI keep running; quit them from the tray)
  check      {check} scripts/healthcheck.py
  settings   .env in this folder (every key is explained in .env.example)
  telegram   open the Telegram tab in the app and paste a @BotFather token
  update     git pull, then run the setup script again""")


# -- main --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Install, configure and verify Assistant DC.")
    ap.add_argument("--no-models", action="store_true",
                    help="do not download the chat/embedding models or the vocoder")
    ap.add_argument("--no-install", action="store_true",
                    help="never install system software (winget); only report what is missing")
    ap.add_argument("--no-start", action="store_true", help="do not start the app at the end")
    ap.add_argument("--cpu", action="store_true", help="install the CPU build of torch even with a GPU")
    ap.add_argument("--dev", action="store_true", help="also install the test dependencies")
    ap.add_argument("--no-comfy", action="store_true",
                    help="do not install ComfyUI (no pictures, video or songs)")
    ap.add_argument("--media", default="image,music,video",
                    help="which ComfyUI model sets to download (image,music,video; ~155 GB all)")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.environ.setdefault("PYTHONUTF8", "1")
    _p(f"Assistant DC setup in {ROOT}")
    t0 = time.monotonic()
    healthy = False
    try:
        gpu = s_environment(args)
        s_vcruntime(args)
        s_torch(args, gpu)
        s_packages(args)
        s_import_path(args)
        s_browser(args)
        s_config(args)
        s_qt_libs(args)
        s_ffmpeg(args)
        s_voice(args)
        s_lmstudio(args)
        s_comfyui(args, gpu)
        s_shortcut(args)
        healthy = s_health(args)
    except SetupError as exc:
        record("setup", "fail", str(exc))
    except KeyboardInterrupt:
        _p("\nInterrupted. Nothing is half-installed in a way that matters: run the setup "
           "again and it continues where it stopped.")
        return 130

    _p("\n" + "=" * 64)
    for name, status, detail in STEPS:
        _p(f"  {status.upper():<4}  {name:<12} {detail.splitlines()[0][:110] if detail else ''}")
    failed = [s for s in STEPS if s[1] == "fail"]
    _p(f"\n  took {time.monotonic() - t0:.0f}s")
    if failed or not healthy:
        _p("\nSETUP INCOMPLETE -- fix the FAIL lines above and run the setup again "
           "(it skips everything that is already done).")
        return 1
    _p("\nSETUP COMPLETE -- Assistant DC is ready.")
    how_to_use()
    if not args.no_start and _has_display():
        _p("\nStarting Assistant DC ...")
        start_app()
    return 0


if __name__ == "__main__":
    sys.exit(main())

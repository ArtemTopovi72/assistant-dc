"""Install YuE2, the song engine, in one command. `setup` leaves it out (owner
10-03: «делай сам»): its torch/transformers pins clash with the app, so it
lives in its own venv, and it needs a CUDA card.

    venv\Scripts\python scripts\install_yue2.py   # Python 3.12 comes from uv (setup installs it)
    python scripts/install_yue2.py --cuda cu130    # another torch CUDA build
    python scripts/install_yue2.py --no-models     # the venv only

What it makes (media/music.py: yue2_available() checks exactly these):
    venv_yue2/                 Python 3.12, torch 2.10.0 (CUDA), yue2-infer 0.1.6
    models_ext/YuE2-3B/        m-a-p/YuE2-3B
    models_ext/YuE2-Vae/       m-a-p/YuE2-Vae (a plain folder: the hub cache needs
                               symlinks, which Windows refuses without developer mode)

Run again at any time: a step already done is skipped.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VENV = os.path.join(ROOT, "venv_yue2")
MODELS = os.path.join(ROOT, "models_ext")
# scripts/yue2_render.py patches GraphAR / CachedNAR as they are at this commit
YUE2_PKG = "yue2-infer @ git+https://github.com/multimodal-art-projection/YuE@1dc1c50"
TORCH = "torch==2.10.0"
REPOS = {"YuE2-3B": "m-a-p/YuE2-3B", "YuE2-Vae": "m-a-p/YuE2-Vae"}


def venv_python() -> str:
    return os.path.join(VENV, *(("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")))


def has_weights(folder: str) -> bool:
    return (os.path.isfile(os.path.join(folder, "config.json"))
            and bool(glob.glob(os.path.join(folder, "*.safetensors"))))


def _uv() -> str:
    """uv, which setup installs: it fetches Python 3.12 itself, so no system
    Python 3.12 is needed (a «py -3.12» that is not there failed the venv)."""
    found = shutil.which("uv")
    if found:
        return found
    for p in (os.path.expanduser("~/.local/bin/uv"), os.path.expanduser("~/.cargo/bin/uv"),
              os.path.expanduser("~/.local/bin/uv.exe"), os.path.expanduser("~/.cargo/bin/uv.exe")):
        if os.path.isfile(p):
            return p
    return ""


def base_python(given: str) -> list:
    """How the venv is made: --python, else uv with Python 3.12 (YuE2's own),
    else a system 3.12."""
    if given:
        return given.split() + ["-m", "venv"]
    if _uv():
        return [_uv(), "venv", "--seed", "-p", "3.12"]
    if os.name == "nt" and shutil.which("py"):
        return ["py", "-3.12", "-m", "venv"]
    return [shutil.which("python3.12") or sys.executable, "-m", "venv"]


def run(cmd: list) -> None:
    print("  $ " + " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def make_venv(python: list) -> None:
    if os.path.isfile(venv_python()):
        print("venv_yue2: already there")
        return
    print("venv_yue2: creating")
    run(python + [VENV])
    run([venv_python(), "-m", "pip", "install", "--upgrade", "pip"])


def installed() -> bool:
    return subprocess.call([venv_python(), "-c", "import yue2, torch; assert torch.cuda.is_available()"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def install_packages(cuda: str) -> None:
    if installed():
        print("yue2-infer: already installed, CUDA torch")
        return
    # torch first from the CUDA index: otherwise the resolver takes PyPI's CPU wheel on Windows
    run([venv_python(), "-m", "pip", "install", TORCH,
         "--index-url", f"https://download.pytorch.org/whl/{cuda}"])
    run([venv_python(), "-m", "pip", "install", YUE2_PKG])


def download_models() -> None:
    for name, repo in REPOS.items():
        dest = os.path.join(MODELS, name)
        if has_weights(dest):
            print(f"{name}: already there")
            continue
        print(f"{name}: downloading {repo}")
        os.makedirs(dest, exist_ok=True)
        run([venv_python(), "-c",
             "import sys; from huggingface_hub import snapshot_download; "
             "snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])", repo, dest])
        if not has_weights(dest):
            raise SystemExit(f"{name}: the download finished but {dest} has no config.json + *.safetensors")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--python", default="", help='interpreter for the venv, e.g. "py -3.12"')
    ap.add_argument("--cuda", default="cu128", help="torch CUDA build: cu126, cu128 (default), cu130")
    ap.add_argument("--no-models", action="store_true", help="do not download the weights")
    args = ap.parse_args(argv)
    make_venv(base_python(args.python))
    install_packages(args.cuda)
    if not args.no_models:
        download_models()
    if not installed():
        print("\nyue2 is installed but torch sees no CUDA card: try another --cuda build.")
        return 1
    print("\nYuE2 is ready. Restart the assistant: 🎵 songs are on.")
    print("Optional, 2x faster: flash-attn in venv_yue2 (scripts/yue2_fa2.py picks it up).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

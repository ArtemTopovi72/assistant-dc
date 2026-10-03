"""ComfyUI for Assistant DC: the server, its custom nodes and the model files
the graphs in workflows/ name. Used by scripts/setup.py; safe to run again.

    venv/Scripts/python scripts/comfy_setup.py [--media image,music,video] [--check]

Layout (what scripts/launch_all.py starts):
    COMFY_BASE_DIR                  default ~/Documents/ComfyUI
      ComfyUI-<version>/            source tree, with its own .venv inside
      custom_nodes/                 ComfyUI-GGUF, Comfyui-QwenEditUtils
      models/<kind>/<file>          shared by every source tree (--base-directory)

An existing install (any ComfyUI-<ver> tree, or the Desktop app) is left as it
is: only missing nodes and model files are added.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

IS_WIN = os.name == "nt"
COMFY_REPO = "https://github.com/comfyanonymous/ComfyUI.git"
# Pinned: every node class the graphs in workflows/ use was checked against
# this release (MiniMax H3 + Music3, Flux2 latents, DualModelGuider, CFGNorm,
# CFGOverride, CreateVideo/SaveVideo). Its requirements.txt pins its own deps.
COMFY_TAG = "v0.38.2"
COMFY_VERSION = COMFY_TAG.lstrip("v")
COMFY_TORCH = ("torch==2.8.0", "torchvision==0.23.0", "torchaudio==2.8.0")

# Custom nodes the graphs use, pinned to the commits that were checked.
NODES = [
    ("ComfyUI-GGUF", "https://github.com/city96/ComfyUI-GGUF.git",
     "6ea2651e7df66d7585f6ffee804b20e92fb38b8a"),                          # CLIPLoaderGGUF
    ("Comfyui-QwenEditUtils", "https://github.com/lrzjason/Comfyui-QwenEditUtils.git",
     "cdd4d028c6491d27a40092d7795158668cec9189"),
    # ^ TextEncodeQwenImageEditPlusAdvance_lrzjason, CropWithPadInfo
]

# (media, models/ subfolder, file name the graph uses, source repos in order,
#  other names the file has in those repos, approx GB)
MODELS = [
    # drawing (Ideogram 4); the graph's int8 files are local quantizations,
    # comfy_client.MODEL_FALLBACKS sends these official fp8 ones instead
    ("image", "diffusion_models", "ideogram4_fp8_scaled.safetensors",
     ["Comfy-Org/Ideogram-4"], [], 9.3),
    ("image", "diffusion_models", "ideogram4_unconditional_fp8_scaled.safetensors",
     ["Comfy-Org/Ideogram-4"], ["ideogram4_uncond_fp8_scaled.safetensors"], 9.3),
    ("image", "text_encoders", "qwen3vl_8b_fp8_scaled.safetensors",
     ["Comfy-Org/Ideogram-4", "Comfy-Org/Qwen3-VL"], [], 10.6),
    ("image", "vae", "flux2-vae.safetensors",
     ["Comfy-Org/flux2-dev", "Comfy-Org/Lens", "Comfy-Org/flux2-klein-9B"], [], 0.3),
    # photo edits (FireRed)
    ("image", "diffusion_models", "FireRed-Image-Edit-1.1_fp8mixed_comfy.safetensors",
     ["cocorang/FireRed-Image-Edit-1.1-FP8_And_BF16"], [], 21.4),
    ("image", "loras", "FireRed-Image-Edit-1.1-Lightning-8steps-v1.2.safetensors",
     ["FireRedTeam/FireRed-Image-Edit-1.1-ComfyUI"], [], 0.9),
    ("image", "text_encoders", "qwen_2.5_vl_7b_fp8_scaled.safetensors",
     ["Comfy-Org/Qwen-Image_ComfyUI"], [], 9.4),
    ("image", "vae", "qwen_image_vae.safetensors",
     ["Comfy-Org/Qwen-Image_ComfyUI"], [], 0.3),
    # songs (MiniMax Music 3)
    ("music", "diffusion_models", "minimax_music3_dit_fp32.safetensors",
     ["Comfy-Org/MiniMax-Music-3"], [], 9.8),
    ("music", "text_encoders", "minimax_music3_text_encoder_bf16.safetensors",
     ["Comfy-Org/MiniMax-Music-3"], [], 18.5),
    ("music", "vae", "minimax_music3_dav.safetensors",
     ["Comfy-Org/MiniMax-Music-3"], [], 0.2),
    # video (MiniMax H3)
    ("video", "diffusion_models", "MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-comfy-int8-convrot.safetensors",
     ["xmarre/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-ComfyUI"], [], 21.0),
    ("video", "diffusion_models", "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors",
     ["WarmBloodAban/Minimax-h3_Singularity"], [], 21.0),
    ("video", "text_encoders", "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
     ["Abiray/MiniMax-H3-GGUF"], [], 14.6),
    ("video", "vae", "minimax_h3_video_vae_fp16.safetensors",
     ["Comfy-Org/MiniMax-H3", "Abiray/MiniMax-H3-GGUF"], [], 5.2),
    ("video", "vae", "minimax_h3_audio_vae_fp32.safetensors",
     ["Comfy-Org/MiniMax-H3", "Abiray/MiniMax-H3-GGUF"], [], 0.6),
    ("video", "loras", "minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors",
     ["Kijai/MiniMax-H3_comfy"], [], 0.3),
    ("video", "loras", "minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors",
     ["lightx2v/Minimax-h3-Turbo", "Kijai/MiniMax-H3_comfy"], [], 1.0),
    ("video", "loras", "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
     ["lightx2v/Minimax-h3-Turbo", "Kijai/MiniMax-H3_comfy"], [], 1.0),
]
MEDIA = ("image", "music", "video")


def comfy_base() -> Path:
    return Path(os.getenv("COMFY_BASE_DIR") or Path.home() / "Documents" / "ComfyUI")


def comfy_exe() -> Path:
    return Path(os.getenv("COMFY_EXE") or Path.home() / "AppData" / "Local" / "Programs"
                / "ComfyUI" / "ComfyUI.exe")


def _version_key(name: str):
    try:
        return tuple(int(x) for x in name.split("-", 1)[1].split("."))
    except (IndexError, ValueError):
        return None


def source_trees(base: Path) -> list:
    """ComfyUI-<ver> trees with a main.py, newest first (launch_all picks the newest)."""
    out = []
    if base.is_dir():
        for d in base.iterdir():
            key = _version_key(d.name) if d.name.startswith("ComfyUI-") else None
            if key is not None and (d / "main.py").exists():
                out.append((key, d))
    return [d for _k, d in sorted(out, reverse=True)]


def tree_python(tree: Path) -> Path | None:
    """The python of a tree's own venv (.venv*), like launch_all._comfy_python."""
    if tree.is_dir():
        for d in sorted(tree.iterdir()):
            if d.name.startswith(".venv"):
                cand = d / ("Scripts/python.exe" if IS_WIN else "bin/python")
                if cand.exists():
                    return cand
    shared = comfy_base() / ".venv" / ("Scripts/python.exe" if IS_WIN else "bin/python")
    return shared if shared.exists() else None


def _run(cmd, log, timeout=7200, cwd=None):
    log("  $ " + " ".join(str(c) for c in cmd))
    p = subprocess.run([str(c) for c in cmd], cwd=cwd, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"exit {p.returncode}: {' '.join(str(c) for c in cmd[:4])} ...")


def install_server(uv: str, torch_index: str | None, log) -> tuple[str, str]:
    """ComfyUI COMFY_TAG in COMFY_BASE_DIR/ComfyUI-<ver> with its own venv.

    Older trees stay where they are; launch_all starts the newest, which is
    this one. -> (status, detail)."""
    base = comfy_base()
    tree = base / f"ComfyUI-{COMFY_VERSION}"
    py = tree / ".venv" / ("Scripts/python.exe" if IS_WIN else "bin/python")
    marker = tree / ".venv" / ".assistant_dc_ready"
    if (tree / "main.py").exists() and marker.exists():
        return "ok", f"ComfyUI {COMFY_VERSION} present: {tree}"
    if not shutil.which("git"):
        return "fail", "git is not installed (needed to fetch ComfyUI): https://git-scm.com"
    base.mkdir(parents=True, exist_ok=True)
    if not (tree / "main.py").exists():
        shutil.rmtree(tree, ignore_errors=True)
        _run(["git", "-c", "advice.detachedHead=false", "clone", "--depth", "1",
              "--branch", COMFY_TAG, COMFY_REPO, tree], log)
    if not py.exists():
        _run([uv, "venv", "-p", "3.13", tree / ".venv"], log)
    _run([uv, "pip", "install", "--python", py, *COMFY_TORCH,
          *(["--index-url", torch_index] if torch_index else [])], log)
    _run([uv, "pip", "install", "--python", py, "-r", tree / "requirements.txt"], log)
    marker.write_text(COMFY_TAG, encoding="utf-8")
    return "ok", f"ComfyUI {COMFY_VERSION} installed: {tree}"


def _git(args, cwd) -> str:
    p = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=600)
    if p.returncode:
        raise RuntimeError((p.stderr or p.stdout).strip()[:300])
    return p.stdout.strip()


def _pin(repo_dir: Path, sha: str, log) -> str:
    """Check repo_dir out at sha. -> "" when done, else why it was left alone."""
    if not (repo_dir / ".git").exists():
        return "not a git checkout"
    try:
        head = _git(["rev-parse", "HEAD"], repo_dir)
    except RuntimeError:
        head = ""                       # freshly initialised, nothing checked out
    if head == sha:
        return ""
    if head and _git(["status", "--porcelain", "--untracked-files=no"], repo_dir):
        return "has local changes"
    log(f"  pinning {repo_dir.name} to {sha[:10]}")
    _git(["fetch", "--depth", "1", "origin", sha], repo_dir)
    _git(["-c", "advice.detachedHead=false", "checkout", sha], repo_dir)
    return ""


def install_nodes(uv: str, log) -> tuple[str, str]:
    base = comfy_base()
    nodes_dir = base / "custom_nodes"
    nodes_dir.mkdir(parents=True, exist_ok=True)
    existing = {d.name.lower(): d for d in nodes_dir.iterdir() if d.is_dir()}
    notes, left = [], []
    for name, url, sha in NODES:
        d = existing.get(name.lower())
        if d is None:
            d = nodes_dir / name
            _run(["git", "init", "-q", d], log)
            _git(["remote", "add", "origin", url], d)
            notes.append(name)
        why = _pin(d, sha, log)
        if why:
            left.append(f"{d.name} ({why})")
    py = tree_python(base / f"ComfyUI-{COMFY_VERSION}")
    if py:
        for name, _url, _sha in NODES:
            req = (existing.get(name.lower()) or nodes_dir / name) / "requirements.txt"
            if req.exists():
                _run([uv, "pip", "install", "--python", py, "-r", req], log)
    detail = ("added " + ", ".join(notes) + "; ") if notes else ""
    detail += "custom nodes at their pinned commits"
    if left:
        return "warn", detail + "; left as they are: " + ", ".join(left)
    return "ok", detail


def _present(models_dir: Path, name: str) -> Path | None:
    if not models_dir.is_dir():
        return None
    for p in models_dir.rglob(name):
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


_LISTINGS: dict = {}


def _find(repo: str, names: list) -> str | None:
    from huggingface_hub import list_repo_files
    if repo not in _LISTINGS:
        _LISTINGS[repo] = list_repo_files(repo)
    for n in names:
        for f in _LISTINGS[repo]:
            if f == n or f.endswith("/" + n):
                return f
    return None


def missing_models(media=MEDIA) -> list:
    models = comfy_base() / "models"
    return [m for m in MODELS if m[0] in media and not _present(models, m[2])]


def download_models(media, log) -> tuple[str, str]:
    """Fetch the graphs' model files that are not there yet. -> (status, detail)."""
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # xet hung on "reconstructing" here
    from huggingface_hub import hf_hub_download
    base = comfy_base()
    todo = missing_models(media)
    if not todo:
        return "ok", f"all model files present ({', '.join(media)})"
    need = sum(m[5] for m in todo)
    base.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(base).free / 1024 ** 3
    if free < need + 10:
        return "warn", (f"need ~{need:.0f} GB for {len(todo)} model files but only {free:.0f} GB "
                        f"free at {base}; free space or set COMFY_BASE_DIR to a bigger disk, "
                        "then run the setup again (or pick fewer with --media)")
    log(f"  {len(todo)} model files, ~{need:.0f} GB (resumable; {free:.0f} GB free)")
    failed = []
    for i, (_media, sub, name, repos, alts, gb) in enumerate(todo, 1):
        dest = base / "models" / sub / name
        stage = base / "models" / ".download"
        got = None
        for repo in repos:
            try:
                path = _find(repo, [name, *alts])
                if not path:
                    continue
                log(f"  [{i}/{len(todo)}] {repo}/{path} (~{gb:g} GB)")
                got = Path(hf_hub_download(repo_id=repo, filename=path, local_dir=str(stage)))
                break
            except Exception as exc:
                log(f"    {repo}: {type(exc).__name__}: {str(exc)[:160]}")
        if got is None:
            failed.append(name)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(got), str(dest))
    if failed:
        return "warn", (f"{len(failed)} of {len(todo)} model files could not be downloaded: "
                        + ", ".join(failed[:4]) + (" ..." if len(failed) > 4 else "")
                        + " -- run the setup again to retry")
    return "ok", f"{len(todo)} model files downloaded"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--media", default=",".join(MEDIA))
    ap.add_argument("--check", action="store_true", help="only list what is missing")
    a = ap.parse_args(argv)
    media = tuple(m.strip() for m in a.media.split(",") if m.strip())
    todo = missing_models(media)
    print(f"ComfyUI base: {comfy_base()}  trees: {[t.name for t in source_trees(comfy_base())]}")
    for m in todo:
        print(f"  missing  {m[1]}/{m[2]}  (~{m[5]:g} GB, {m[3][0]})")
    if a.check:
        return 0 if not todo else 1
    status, detail = download_models(media, print)
    print(status.upper(), detail)
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())

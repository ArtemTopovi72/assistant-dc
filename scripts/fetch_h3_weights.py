"""Download the MiniMax-H3 weights this project needs, into the ComfyUI model tree.

Chosen for a 24GB Ampere card (RTX 3090), which is why these are GGUF and not the
officially packaged fp8/nvfp4 safetensors:

  * fp8_scaled needs sm_89+ (Ada) for native fp8; on Ampere ComfyUI upcasts it,
    so you pay the full bf16 compute cost for a file that is only smaller on disk.
  * nvfp4_awq is a Blackwell (RTX 50-series) format outright.
  * GGUF Q4_K_M runs natively on Ampere through ComfyUI-GGUF's UnetLoaderGGUF,
    which is already installed here.

Two checkpoints, because H3 ships as two task-specific models and between them they
cover every mode this project exposes:

  FL2VA  — text-to-video (no image), image-to-video (one image),
           first-and-last-frame (two images)
  Ref2VA — omni-reference: up to 9 images, up to 3 video clips, audio

Total ~60GB. Resumable: re-running skips whatever is already complete, so an
interrupted download costs only the file it died on.

Run: venv/Scripts/python.exe scripts/fetch_h3_weights.py [--check]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# The ComfyUI --base-directory: models/ lives here, shared by every versioned
# ComfyUI-<ver> source tree next to it.
COMFY_BASE = Path(os.getenv("COMFY_BASE_DIR", os.path.expanduser(r"~\Documents\ComfyUI")))

REPO = "Abiray/MiniMax-H3-GGUF"

# (path inside the repo, ComfyUI models/ subfolder, approx GB, why we need it)
FILES = [
    ("unet/MiniMax-H3-FL2VA-Q4_K_M.gguf", "unet", 19.86,
     "text-to-video, image-to-video, first+last frame"),
    ("unet/MiniMax-H3-Ref2VA-Q4_K_M.gguf", "unet", 19.85,
     "multi-image reference, video+image, audio reference"),
    ("text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf", "text_encoders", 14.58,
     "H3-Encoder (Qwen3-VL-32B); H3 will not run without it"),
    ("vae/minimax_h3_video_vae_fp16.safetensors", "vae", 5.21,
     "H3-VisualVAE (f16t4d24) — decodes the video latents"),
    ("vae/minimax_h3_audio_vae_fp32.safetensors", "vae", 0.61,
     "H3-AudioVAE — decodes the native stereo track"),
]


def target_of(repo_path: str, subdir: str) -> Path:
    return COMFY_BASE / "models" / subdir / Path(repo_path).name


def human(gb: float) -> str:
    return f"{gb:,.2f} GB"


def report() -> tuple[list, float, float]:
    """What is already here, what is still missing."""
    missing, have_gb, need_gb = [], 0.0, 0.0
    print(f"ComfyUI base : {COMFY_BASE}")
    print(f"Source repo  : {REPO}\n")
    for repo_path, subdir, gb, why in FILES:
        dest = target_of(repo_path, subdir)
        # A partial file from an interrupted run is smaller than advertised; treat
        # anything under 98% as missing so it gets resumed rather than trusted.
        actual = dest.stat().st_size / 1e9 if dest.exists() else 0.0
        ok = actual >= gb * 0.98
        if ok:
            have_gb += actual
        else:
            need_gb += gb - actual
            missing.append((repo_path, subdir, gb, why))
        mark = "OK  " if ok else ("PART" if actual else "  --")
        print(f"  [{mark}] {human(gb):>10}  models/{subdir}/{Path(repo_path).name}")
        print(f"          {why}")
    print()
    print(f"  present: {human(have_gb)}    still to fetch: {human(need_gb)}")
    return missing, have_gb, need_gb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report what is present and exit without downloading")
    args = ap.parse_args()

    missing, _have, need_gb = report()
    if args.check:
        return 0
    if not missing:
        print("Everything is already downloaded.")
        return 0

    free_gb = 0.0
    try:
        import shutil
        free_gb = shutil.disk_usage(COMFY_BASE).free / 1e9
    except Exception:
        pass
    if free_gb and free_gb < need_gb * 1.05:
        print(f"REFUSING: needs ~{human(need_gb)} but only {human(free_gb)} free.")
        return 2
    print(f"disk free: {human(free_gb)} — proceeding\n")

    from huggingface_hub import hf_hub_download

    started = time.time()
    for i, (repo_path, subdir, gb, _why) in enumerate(missing, 1):
        dest_dir = COMFY_BASE / "models" / subdir
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{i}/{len(missing)}] {repo_path}  ({human(gb)}) -> models/{subdir}/",
              flush=True)
        t0 = time.time()
        try:
            got = hf_hub_download(
                repo_id=REPO, filename=repo_path,
                local_dir=str(dest_dir.parent.parent / "_h3_staging"),
                resume_download=True,
            )
        except Exception as exc:
            print(f"    FAILED: {exc}", flush=True)
            print("    (a 401/403 means the MiniMax H3 community licence still needs "
                  "accepting on the HuggingFace repo page while signed in)", flush=True)
            return 1
        final = target_of(repo_path, subdir)
        try:
            if final.exists():
                final.unlink()
            os.replace(got, final)
        except OSError:
            # Different volume, or the staging file is a symlink into the HF cache:
            # fall back to a copy so the model still lands where ComfyUI looks.
            import shutil as _sh
            _sh.copyfile(got, final)
        mins = (time.time() - t0) / 60
        print(f"    done in {mins:.1f} min -> {final}", flush=True)

    print(f"\nAll weights present. Total elapsed {(time.time()-started)/60:.1f} min.")
    print("Next: restart ComfyUI so the new models are picked up.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

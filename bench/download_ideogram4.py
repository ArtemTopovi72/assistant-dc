"""Fetch what Ideogram 4 LoRA training needs, into E:\hf-cache.

Two repos, and NOT all of either:

  * ideogram-ai/ideogram-4-fp8 -- transformer, VAE, tokenizer, configs. The
    repo also ships `unconditional_transformer` (8.65 GB), which is only used
    for the negative-CFG unconditional LoRA at inference, and its own fp8
    `text_encoder` (8.18 GB), which the toolkit deliberately ignores: see
    ideogram4.py:_load_text_encoder -- "the ideogram repo only ships an fp8
    copy, so load the public bf16 model directly". Downloading both would cost
    16.8 GB for nothing.
  * Qwen/Qwen3-VL-8B-Instruct -- the frozen text encoder the toolkit actually
    loads, bf16, 16.3 GB.

Same two hard-won settings as the old model fetch: XET off (it downloads the
bytes and then hangs forever on "reconstructing file" here) and the cache on
E:, because C: does not have room.

    venv/Scripts/python.exe bench/download_ideogram4.py
"""
import os
from pathlib import Path

CACHE = Path(r"E:\hf-cache")
os.environ["HF_HUB_DISABLE_XET"] = "1"

WANTED = [
    ("ideogram-ai/ideogram-4-fp8",
     ["unconditional_transformer/*", "text_encoder/*", "assets/*", "*.png",
      "*.jpg"]),
    ("Qwen/Qwen3-VL-8B-Instruct", []),
]


def size_gb(p: Path) -> float:
    return (sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024 ** 3
            if p.is_dir() else 0.0)


def main() -> int:
    from huggingface_hub import snapshot_download
    for repo, skip in WANTED:
        folder = CACHE / ("models--" + repo.replace("/", "--"))
        print("\n=== %s (have %.2f GB)" % (repo, size_gb(folder)), flush=True)
        path = snapshot_download(repo_id=repo, cache_dir=str(CACHE),
                                 max_workers=4, ignore_patterns=skip or None)
        print("  -> %.2f GB at %s" % (size_gb(folder), path), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

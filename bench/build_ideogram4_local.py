"""Assemble a local Ideogram 4 model dir for the trainer from ComfyUI's files.

`ideogram-ai/ideogram-4-fp8` is a GATED repo: fetching it returns 401 without
granted access. But the weights are already on this machine, shipped for
ComfyUI, and the trainer turns out to want almost exactly that format --
Ideogram4Config is built in CODE rather than read from a config.json, and
_load_component_state_dict just wants `<base>/<component>/
diffusion_pytorch_model.safetensors`.

One real incompatibility, and it is arithmetic, not naming. ComfyUI stores one
SCALAR fp8 scale per tensor; the toolkit's dequantizer does
`w * scale.unsqueeze(1)`, which needs a per-output-channel vector and raises on
a 0-dim tensor. Expanding the scalar to a vector of length out_features is
numerically identical -- every channel had that same scale already -- and keeps
the weights in fp8 so the file stays small.

    venv/Scripts/python.exe bench/build_ideogram4_local.py
"""
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

COMFY = Path.home() / "Documents" / "ComfyUI" / "models"
SRC_TRANSFORMER = COMFY / "diffusion_models" / "ideogram4_fp8_scaled.safetensors"
SRC_VAE = COMFY / "vae" / "flux2-vae.safetensors"
OUT = Path(r"E:\ideogram4-local")


def convert_transformer(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tensors, meta = {}, {}
    with safe_open(str(src), framework="pt") as f:
        meta = f.metadata() or {}
        dropped = 0
        for k in f.keys():
            # ComfyUI ships 211 `*.comfy_quant` u8 blobs describing its own
            # quantization scheme. They are not model parameters and the
            # toolkit's loader has never heard of them, so they must not reach
            # load_from_state_dict.
            if k.endswith(".comfy_quant"):
                dropped += 1
                continue
            tensors[k] = f.get_tensor(k)
    print("  comfy_quant blobs dropped: %d" % dropped)
    fixed = 0
    for k in list(tensors):
        if not k.endswith(".weight_scale"):
            continue
        scale = tensors[k]
        if scale.dim() != 0:
            continue                       # already per-channel; leave it alone
        w = tensors.get(k[: -len("_scale")])
        if w is None:
            continue
        tensors[k] = scale.reshape(1).repeat(w.shape[0]).contiguous()
        fixed += 1
    print("  scalar scales expanded to per-channel: %d" % fixed)
    save_file(tensors, str(dst), metadata={k: str(v) for k, v in meta.items()})
    print("  wrote %s (%.2f GB)" % (dst, dst.stat().st_size / 1024 ** 3))


def main() -> int:
    for p in (SRC_TRANSFORMER, SRC_VAE):
        if not p.is_file():
            print("missing:", p)
            return 1
    print("transformer:", SRC_TRANSFORMER.name)
    convert_transformer(SRC_TRANSFORMER,
                        OUT / "transformer" / "diffusion_pytorch_model.safetensors")

    vae_dst = OUT / "vae" / "diffusion_pytorch_model.safetensors"
    vae_dst.parent.mkdir(parents=True, exist_ok=True)
    if not vae_dst.is_file():
        print("vae: copying", SRC_VAE.name)
        shutil.copy2(str(SRC_VAE), str(vae_dst))
    print("  %s (%.2f GB)" % (vae_dst, vae_dst.stat().st_size / 1024 ** 3))
    print("\nmodel dir:", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

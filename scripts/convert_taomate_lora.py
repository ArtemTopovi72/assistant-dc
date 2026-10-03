"""TaoMate-H3 3-step LoRA (PEFT-style keys) -> ComfyUI LoRA keys.

    venv/Scripts/python scripts/convert_taomate_lora.py SRC DST
blocks.N.x.lora_a / lora_b  ->  diffusion_model.blocks.N.x.lora_A.weight / lora_B.weight
plus an alpha per module (adapter_config: rank 128, alpha 128 -> scale 1.0),
exactly the layout of the lightx2v turbo LoRA ComfyUI already loads.
"""
import json, os, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def convert(src: str, dst: str) -> int:
    cfg_path = os.path.join(os.path.dirname(src), "adapter_config.json")
    alpha = float(json.load(open(cfg_path)).get("alpha", 128.0)) if os.path.exists(cfg_path) else 128.0
    out = {}
    with safe_open(src, "pt") as f:
        for k in f.keys():
            base, part = k.rsplit(".", 1)
            if part not in ("lora_a", "lora_b"):
                raise ValueError(f"unexpected key {k}")
            t = f.get_tensor(k).to(torch.bfloat16).contiguous()
            out[f"diffusion_model.{base}.lora_{part[-1].upper()}.weight"] = t
            out.setdefault(f"diffusion_model.{base}.alpha", torch.tensor(alpha))
    save_file(out, dst)
    return len(out)


if __name__ == "__main__":
    print(convert(sys.argv[1], sys.argv[2]), "tensors written")

import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import torch
from safetensors.torch import save_file, load_file
import convert_taomate_lora as C


def test_keys_and_alpha(tmp_path):
    src = tmp_path / "adapter_model.safetensors"
    save_file({"blocks.0.attn.qkv_proj.lora_a": torch.ones(4, 8),
               "blocks.0.attn.qkv_proj.lora_b": torch.ones(8, 4),
               "token_refiner.blocks.1.mlp.fc2.lora_a": torch.ones(4, 8),
               "token_refiner.blocks.1.mlp.fc2.lora_b": torch.ones(8, 4)}, str(src))
    (tmp_path / "adapter_config.json").write_text(json.dumps({"rank": 4, "alpha": 4.0}))
    dst = tmp_path / "out.safetensors"
    C.convert(str(src), str(dst))
    d = load_file(str(dst))
    assert set(d) == {"diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight",
                      "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight",
                      "diffusion_model.blocks.0.attn.qkv_proj.alpha",
                      "diffusion_model.token_refiner.blocks.1.mlp.fc2.lora_A.weight",
                      "diffusion_model.token_refiner.blocks.1.mlp.fc2.lora_B.weight",
                      "diffusion_model.token_refiner.blocks.1.mlp.fc2.alpha"}
    assert d["diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight"].shape == (4, 8)
    assert float(d["diffusion_model.blocks.0.attn.qkv_proj.alpha"]) == 4.0
    assert d["diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight"].dtype == torch.bfloat16

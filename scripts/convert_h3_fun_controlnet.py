"""alibaba-pai MiniMax-H3-Fun-Controlnet-Union (diffusers keys) -> ComfyUI keys.

to_q/to_k/to_v -> qkv_proj (concatenated), norm_q/k -> q_norm/k_norm, to_out.0 -> out_proj,
ff.net.0.proj -> mlp.fc1 with its halves SWAPPED (diffusers SwiGLU is [up, gate], Comfy's is
[gate, up]; verified 2026-09-25 by row-norm correlation against the base DiT block: 0.99999 swapped
vs 0.963 as-is), ff.net.2 -> mlp.fc2.
"""
import sys
import torch
from safetensors.torch import load_file, save_file

src, dst = sys.argv[1], sys.argv[2]
sd = load_file(src)
out = {}
for k, v in sd.items():
    if ".attn.to_k." in k or ".attn.to_v." in k:
        continue
    if ".attn.to_q." in k:
        out[k.replace(".to_q.", ".qkv_proj.")] = torch.cat(
            [v, sd[k.replace(".to_q.", ".to_k.")], sd[k.replace(".to_q.", ".to_v.")]], 0).contiguous()
    elif ".ff.net.0.proj." in k:
        h = v.shape[0] // 2
        out[k.replace(".ff.net.0.proj.", ".mlp.fc1.")] = torch.cat([v[h:], v[:h]], 0).contiguous()
    else:
        out[k.replace(".attn.norm_q.", ".attn.q_norm.").replace(".attn.norm_k.", ".attn.k_norm.")
             .replace(".attn.to_out.0.", ".attn.out_proj.").replace(".ff.net.2.", ".mlp.fc2.")] = v
save_file(out, dst, metadata={"format": "pt"})
print(len(sd), "->", len(out), "tensors")

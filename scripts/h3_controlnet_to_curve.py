"""Re-express a full-adaln H3 Fun ControlNet (Comfy keys) in the curve-basis adaln form our
int8 base checkpoints use (adaln_t_table [grid, k], no time embedder, no silu before adaln).

Full:  y = W silu(TE(t)) + b          (TE from a full-form H3, here the FL2VA GGUF)
Curve: y = Wc c(t) + bc               (c(t) = interpolated row of the base's adaln_t_table)
Fit silu(TE(t_g)) ~= A c_g + e0 over the table grid by least squares, then Wc = W A, bc = b + W e0.
Usage: python h3_controlnet_to_curve.py <controlnet-comfy> <base-curve.safetensors> <full.gguf> <out>
"""
import math
import sys

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ctrl_p, base_p, gguf_p, out_p = sys.argv[1:5]
import gguf
from gguf.quants import dequantize

r = gguf.GGUFReader(gguf_p)
te = {}
for t in r.tensors:
    if t.name.startswith("time_embedder."):
        a = dequantize(t.data, t.tensor_type).astype(np.float32)
        te[t.name] = torch.from_numpy(a.reshape([int(x) for x in reversed(t.shape)]) if a.ndim == 1 and len(t.shape) > 1 else a)
W_in, b_in = te["time_embedder.proj_in.weight"], te["time_embedder.proj_in.bias"]
W_out, b_out = te["time_embedder.proj_out.weight"], te["time_embedder.proj_out.bias"]
if W_in.shape[0] != b_in.shape[0]:
    W_in = W_in.T
if W_out.shape[0] != b_out.shape[0]:
    W_out = W_out.T
table = safe_open(base_p, "pt").get_tensor("adaln_t_table").float()          # [grid, k]
grid = table.shape[0]
t = torch.linspace(0.0, 1.0, grid)
half = W_in.shape[1] // 2
freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
args = t[:, None] * freqs[None]
emb = torch.cat([torch.cos(args), torch.sin(args)], -1)
E = torch.nn.functional.silu((torch.nn.functional.silu(emb @ W_in.T + b_in)) @ W_out.T + b_out)  # [grid, D]
X = torch.cat([table, torch.ones(grid, 1)], 1)                               # [grid, k+1]
sol = torch.linalg.lstsq(X.double(), E.double()).solution.float()            # [k+1, D]
A, e0 = sol[:-1].T, sol[-1]                                                  # [D, k], [D]
res = (X @ sol - E).norm() / E.norm()
print(f"grid {grid}, k {table.shape[1]}, D {E.shape[1]}, relative fit residual {res:.2e}")
sd = load_file(ctrl_p)
for k in [k for k in sd if k.endswith("adaln_proj.linear.weight")]:
    W = sd[k].float()
    bk = k[:-len("weight")] + "bias"
    b = sd[bk].float()
    sd[k] = (W @ A).to(sd[k].dtype).contiguous()
    sd[bk] = (b + W @ e0).to(torch.float32).contiguous()
save_file(sd, out_p, metadata={"format": "pt", "minimax_h3_fun_controlnet": "adaln_basis"})
print("wrote", out_p)

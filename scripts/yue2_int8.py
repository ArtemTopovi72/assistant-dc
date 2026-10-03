"""Int8 weight-only AR linears for YuE2 on Ampere (the package's FP8 needs sm_89).

One-token decode is memory-bound: 3B bf16 weights = ~6 GB read per token, the
3090's 936 GB/s caps it near 150 tok/s and we measured ~105. Int8 weights
(per-output-channel scales, torch._weight_int8pack_mm) halve the bytes.
Plugged into the package's own FP8 hooks: pipe.quantization is set to "fp8"
so _load_model() prepares the AR and synthesize() restores exact BF16 before
the NAR. Originals stay ON THE GPU (swap, no 6 GB PCIe copy per song).
"""
import torch
from torch import nn


class Int8Linear(nn.Linear):
    """An nn.Linear by type (GraphAR accepts only those, and checks weight.dtype):
    `weight` is an empty placeholder of the model dtype, the data is `qweight`."""
    def __init__(self, linear):
        nn.Module.__init__(self)
        w = linear.weight.detach().float()
        scale = w.abs().amax(dim=1).clamp_min(1e-8) / 127.0
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.weight = nn.Parameter(torch.empty(0, dtype=linear.weight.dtype, device=w.device), requires_grad=False)
        self.register_buffer("qweight", (w / scale[:, None]).round().clamp(-127, 127).to(torch.int8))
        self.register_buffer("scale", scale.to(linear.weight.dtype))
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach().clone(), requires_grad=False)

    def forward(self, x):
        shape = x.shape
        out = torch._weight_int8pack_mm(x.reshape(-1, self.in_features).contiguous(), self.qweight, self.scale)
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*shape[:-1], self.out_features)


def install(pipe) -> None:
    import yue2.quantization as q

    def prepare(model, device):
        done = getattr(model, "_yue2_int8", None)
        if done and done.get("active"):
            return
        store = done or {"orig": {}, "q": {}}
        for name, module in list(model.named_modules()):
            if not q.AR_LINEAR.fullmatch(name):
                continue
            if name not in store["q"]:
                module.to(device)
                store["orig"][name] = module
                store["q"][name] = Int8Linear(module).to(device)
            q._replace(model, name, store["q"][name])
        store["active"] = True
        object.__setattr__(model, "_yue2_int8", store)

    def restore(model):
        store = getattr(model, "_yue2_int8", None) if model is not None else None
        if not store or not store.get("active"):
            return
        for name, original in store["orig"].items():
            q._replace(model, name, original)
        store["active"] = False

    q.prepare_fp8_ar = prepare
    q.restore_ar = restore
    pipe.quantization = "fp8"          # routes through the hooks above

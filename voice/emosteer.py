"""EmoSteer-style emotion control for F5-TTS: training-free activation steering (arXiv 2508.03543, simplified).

A steering vector per DiT block = mean activation of emotional speech minus mean activation of neutral speech
(first residual stream of the block, last flow step, mean over time). At synthesis it is added to every token of
the block's residual stream, scaled by alpha relative to the token's own norm, and the token is renormalised
to its original norm. Simplifications against the paper: one vector per block (no per-token selection / top-k),
vectors built from Russian Dusha clips instead of English ESD.

    st = Steerer(tts_model)
    st.record(); infer_process(...clip...); vec = st.take()          # activations of one clip, {block: [D]}
    st.apply({block: vector}, alpha=1.0); infer_process(...); st.off()
"""
import torch


class Steerer:
    def __init__(self, model):
        self.blocks = list(model.transformer.transformer_blocks)
        self.mode = "off"
        self.alpha = 0.0
        self.vec = {}
        self.rec = {}
        for i, b in enumerate(self.blocks):
            b._orig_forward = b.forward
            b.forward = self._make(i, b)

    def _make(self, i, b):
        def forward(x, t, mask=None, rope=None):
            norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = b.attn_norm(x, emb=t)
            x = x + gate_msa.unsqueeze(1) * b.attn(x=norm, mask=mask, rope=rope)
            if self.mode == "record":
                self.rec[i] = x[0].detach().float().mean(dim=0).cpu()          # conditional branch, mean over time -> [D]
            elif self.mode == "apply" and i in self.vec and self.alpha:
                orig = x.norm(dim=-1, keepdim=True)
                v = self.vec[i].to(x.device, x.dtype)
                steered = x + self.alpha * orig * v                                  # v is unit-ish; scaled by the token's own norm
                x = steered * (orig / (steered.norm(dim=-1, keepdim=True) + 1e-8))
            norm = b.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            return x + gate_mlp.unsqueeze(1) * b.ff(norm)
        return forward

    def record(self):
        self.mode, self.rec = "record", {}

    def take(self) -> dict:
        self.mode = "off"
        return dict(self.rec)

    def apply(self, vec: dict, alpha: float):
        self.mode, self.vec, self.alpha = "apply", vec, alpha

    def off(self):
        self.mode, self.alpha = "off", 0.0


def build_vectors(emotional: list, neutral: list) -> dict:
    """[{block: [D]}] x2 -> {block: unit direction [D]} = normalised (mean emotional - mean neutral)."""
    out = {}
    for i in neutral[0]:
        e = torch.stack([r[i] for r in emotional]).mean(0)
        n = torch.stack([r[i] for r in neutral]).mean(0)
        d = e - n
        out[i] = d / (d.norm() + 1e-8)
    return out

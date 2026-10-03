"""FlashAttention-2 for YuE2's AR decode on Windows.

yue2.cuda_graph's "flash" backend calls torch's own varlen kernel
(aten._flash_attention_forward), which the Windows torch build ships without
("USE_FLASH_ATTENTION was not enabled"), so we ran the masked SDPA fallback.
This routes that one call to the flash-attn package's flash_attn_with_kvcache,
which is the decode kernel proper (per-branch cache lengths, graph-capturable).
install() returns False when flash-attn is missing; the caller keeps sdpa.
"""


class _Aten:
    def __init__(self, real, fn):
        self._real, self._flash_attention_forward = real, fn

    def __getattr__(self, name):
        return getattr(self._real, name)


class _Ops:
    def __init__(self, real, aten):
        self._real, self.aten = real, aten

    def __getattr__(self, name):
        return getattr(self._real, name)


class _Torch:
    def __init__(self, real, ops):
        self._real, self.ops = real, ops

    def __getattr__(self, name):
        return getattr(self._real, name)


def install() -> bool:
    try:
        import torch
        from flash_attn import flash_attn_with_kvcache
    except Exception:
        return False
    import yue2.cuda_graph as cg

    def varlen(q, k, v, cu_q, cu_k, max_q, max_k, dropout, causal, debug, *, seqused_k=None, **_):
        # q: (branches, heads, dim) -- one new token per branch; k/v: the whole
        # packed cache (branches * capacity, kv_heads, dim), sequence-major.
        b = q.shape[0]
        kc = k.view(b, max_k, k.shape[-2], k.shape[-1])
        vc = v.view(b, max_k, v.shape[-2], v.shape[-1])
        out = flash_attn_with_kvcache(q[:, None], kc, vc, cache_seqlens=seqused_k)
        return (out[:, 0],)

    varlen.default = torch.ops.aten._flash_attention_forward.default   # the schema probe reads it
    shim = _Torch(torch, _Ops(torch.ops, _Aten(torch.ops.aten, varlen)))
    cg.torch = shim

    # NAR: [tokens, heads, dim] self-attention over the whole song. Flash needs
    # no mask and no query chunking (the 1024-row chunks were an OOM guard).
    import yue2.nar as nar
    from flash_attn import flash_attn_func
    _orig = nar.attention

    def attention(q, k, v, *, causal=False, backend="sdpa", query_chunk_size=None):
        if q.device.type != "cuda" or q.dtype not in (torch.bfloat16, torch.float16):
            return _orig(q, k, v, causal=causal, backend=backend, query_chunk_size=query_chunk_size)
        return flash_attn_func(q[None], k[None], v[None], causal=causal)[0]

    nar.attention = attention
    return True

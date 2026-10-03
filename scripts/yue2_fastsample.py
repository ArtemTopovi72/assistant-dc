"""Same sampling distribution as yue2.sampling.distribution, fewer kernels.

The reference builds a -inf mask over the whole 184k vocab, applies the
repetition penalty over all of it, and SORTS the whole vocab for top-p every
token (0.8 ms/token semantic, 1.7 ms abc on the 3090). Here: penalty only on
the few ids in the window, top-k first, top-p over those k values only; the
result is scattered back into a full-size -inf tensor, so multinomial sees the
same probabilities (ties may order differently).
"""
import torch

_MASKS = {}


def _allowed(scores, phase, end):
    from yue2.protocol import CODEC_OFFSET, CODEC_SIZE, EOD
    key = (phase, end, scores.shape[-1], scores.device, scores.dtype)
    m = _MASKS.get(key)
    if m is None:
        m = torch.full((scores.shape[-1],), float("-inf"), device=scores.device, dtype=scores.dtype)
        if phase == "abc":
            m[:EOD] = 0
        else:
            m[CODEC_OFFSET:CODEC_OFFSET + CODEC_SIZE] = 0
        m[end] = 0
        _MASKS[key] = m
    return m


def distribution(logits, sampling, history, step, phase, legacy_off=False):
    from yue2.protocol import ABC_END, MUSIC_END
    if legacy_off:                       # historical BF16 path: leave it exact
        return _REF(logits, sampling, history, step, phase, legacy_off)
    scores = logits.float() + _allowed(logits.float(), phase, ABC_END if phase == "abc" else MUSIC_END)
    end = ABC_END if phase == "abc" else MUSIC_END
    if step < sampling.min_tokens:
        scores[..., end] = -torch.inf
    recent = history[-sampling.penalty_window:]
    if sampling.repetition_penalty != 1.0 and recent:
        ids = torch.as_tensor(recent, dtype=torch.long, device=scores.device)
        counts = torch.zeros(scores.shape[-1], device=scores.device, dtype=scores.dtype)
        counts.index_add_(0, ids, torch.ones_like(ids, dtype=scores.dtype))
        cur = scores[..., ids]
        alpha = sampling.repetition_penalty ** counts[ids]
        scores[..., ids] = torch.where(cur < 0, cur * alpha, cur / alpha)   # repeats write the same value
    if sampling.temperature == 0:
        return scores
    if sampling.temperature != 1:
        scores = scores / sampling.temperature
    k = min(sampling.top_k, scores.shape[-1])
    # k + 16: values tied with the k-th survive in the reference (it masks
    # scores < threshold); bf16 logits tie often. No host sync on this path.
    values, indices = scores.topk(min(k + 16, scores.shape[-1]))
    values = values.masked_fill(values < values[..., k - 1, None], -torch.inf)
    if sampling.top_p < 1:
        probabilities = values.softmax(-1)
        removed = probabilities.cumsum(-1) - probabilities > sampling.top_p
        removed[..., :1] = False
        values = values.masked_fill(removed, -torch.inf)
    out = torch.full_like(scores, -torch.inf)
    out.scatter_(-1, indices, values)
    return out


_REF = None


def install() -> None:
    global _REF
    import yue2.sampling as s
    if _REF is None:
        _REF = s.distribution
    s.distribution = distribution

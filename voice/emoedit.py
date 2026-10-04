"""SEmoEdit-style emotion editing for F5 (arXiv 2609.34648): FlowEdit velocity transport between two conditions.

Source condition  = neutral reference of the voice (Rs).   Target condition = the SAME voice with the wanted emotion (Rt).
1. X0 = ordinary F5 sample with Rs.
2. Along the flow, for noise e: Xs_t = (1-t)e + t*X0 ;  Xt_t = Z + Xs_t - X0 ;
   Z += dt * strength * ( v(Xt_t | Rt) - v(Xs_t | Rs) ),  Z starts at X0.
3. Z is the edited mel -> vocoder.  Words come from the text, the voice from Rs, the emotion from Rt.

    wav24k = edit(ctx, rs_wav, rs_text, rt_wav, rt_text, "text", strength=1.0)
"""
import numpy as np
import torch

from f5_tts.infer.utils_infer import preprocess_ref_audio_text, hop_length, target_rms, target_sample_rate
from f5_tts.model.utils import convert_char_to_pinyin, get_epss_timesteps
import torchaudio


def _cond(model, wav_path, device):
    a, sr = torchaudio.load(wav_path)
    a = a.mean(0, keepdim=True) if a.shape[0] > 1 else a
    rms = torch.sqrt(torch.mean(a ** 2))
    if rms < target_rms:
        a = a * target_rms / rms
    if sr != target_sample_rate:
        a = torchaudio.transforms.Resample(sr, target_sample_rate)(a)
    a = a.to(device)
    mel = model.mel_spec(a).permute(0, 2, 1)                       # 1, n, 100
    return mel.to(next(model.parameters()).dtype), float(rms)


@torch.inference_mode()
def edit(ctx, rs_wav, rs_text, rt_wav, rt_text, gen_text, strength=1.0, steps=10, cfg=2.0, sway=-1.0, seed=0,
         speed=1.0, tau_skip=0):
    model, vocoder = ctx.models.tts_model, ctx.models.vocoder
    dev = next(model.parameters()).device
    dt_ = next(model.parameters()).dtype
    rs_file, rs_text = preprocess_ref_audio_text(rs_wav, rs_text)
    rt_file, rt_text = preprocess_ref_audio_text(rt_wav, rt_text)
    if not rs_text.endswith(" "):
        rs_text += " "
    if not rt_text.endswith(" "):
        rt_text += " "
    cs, rms = _cond(model, rs_file, dev)
    ct, _ = _cond(model, rt_file, dev)
    ls, lt = cs.shape[1], ct.shape[1]
    gl = int(ls / len(rs_text.encode()) * len(gen_text.encode()) / speed)         # generated frames, shared by both branches
    texts = {}
    for k, rt in (("s", rs_text), ("t", rt_text)):
        texts[k] = list_idx(model, convert_char_to_pinyin([rt + gen_text]), dev)

    def pad(c, n):
        return torch.nn.functional.pad(c, (0, 0, 0, n - c.shape[1]))
    cond = {"s": pad(cs, ls + gl), "t": pad(ct, lt + gl)}
    lens = {"s": ls, "t": lt}

    def velocity(k, x_gen, e_ref):
        n = lens[k]
        x = torch.cat([e_ref[k], x_gen], 1)
        t_ = cond[k]
        step_cond = t_.clone(); step_cond[:, n:] = 0
        return None, x, step_cond

    g = torch.Generator(device=dev).manual_seed(seed)
    e_gen = torch.randn(1, gl, 100, generator=g, device=dev, dtype=dt_)
    e_ref = {k: torch.randn(1, lens[k], 100, generator=g, device=dev, dtype=dt_) for k in "st"}

    ts = get_epss_timesteps(steps, device=dev, dtype=dt_)
    ts = ts + sway * (torch.cos(torch.pi / 2 * ts) - 1 + ts)

    def v(k, x_gen, t):
        n = lens[k]
        x = torch.cat([e_ref[k], x_gen], 1)
        sc = cond[k].clone(); sc[:, n:] = 0
        pred = model.transformer(x=x, cond=sc, text=texts[k], time=t, mask=None, cfg_infer=True, cache=False)
        p, nul = torch.chunk(pred, 2, dim=0)
        out = p + (p - nul) * cfg
        return out[:, n:]

    # 1. source sample X0 with Rs
    x = e_gen.clone() if False else torch.cat([e_ref["s"], e_gen], 1)
    sc = cond["s"].clone(); sc[:, ls:] = 0
    for i in range(len(ts) - 1):
        t, tn = ts[i], ts[i + 1]
        pred = model.transformer(x=x, cond=sc, text=texts["s"], time=t, mask=None, cfg_infer=True, cache=False)
        p, nul = torch.chunk(pred, 2, dim=0)
        x = x + (tn - t) * (p + (p - nul) * cfg)
    x0 = x[:, ls:]
    # 2. transport
    z = x0.clone()
    for i in range(tau_skip, len(ts) - 1):
        t, tn = ts[i], ts[i + 1]
        xs = (1 - t) * e_gen + t * x0
        xt = z + (xs - x0)
        z = z + (tn - t) * strength * (v("t", xt, t) - v("s", xs, t))
    mel = z.float().permute(0, 2, 1)
    wav = vocoder.decode(mel)
    if rms < target_rms:
        wav = wav * rms / target_rms
    return wav.squeeze().cpu().numpy()


def list_idx(model, final_text_list, dev):
    from f5_tts.model.utils import list_str_to_idx
    return list_str_to_idx(final_text_list, model.vocab_char_map).to(dev)

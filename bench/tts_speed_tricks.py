"""F5 speed tricks A/B on our checkpoint: CFG skip on late steps, torch.compile.

    venv/Scripts/python.exe bench/tts_speed_tricks.py --variants base,cfgskip0.5,cfgskip0.7,compile

cfgskipX: past flow time X the unconditional pass is dropped (DiTReducio-style);
the guided flow becomes the plain conditional one, half the transformer work on
those steps. Same phrases and WER (GigaAM) as tts_nfe_ab; WAVs in
outputs/speed_tricks/<variant>/.
"""
import argparse, json, shutil, sys, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench"))
import torch
from f5_tts.model.backbones.dit import DiT
from tts_nfe_ab import PHRASES, words, wer
from tts_whisper_roundtrip import _ctx

_orig_forward = DiT.forward
SKIP_AFTER = [None]


def _forward(self, x, cond, text, time, mask=None, drop_audio_cond=False, drop_text=False,
             cfg_infer=False, cache=False):
    t = float(time.flatten()[0]) if torch.is_tensor(time) else float(time)
    if cfg_infer and SKIP_AFTER[0] is not None and t >= SKIP_AFTER[0]:
        pred = _orig_forward(self, x, cond, text, time, mask=mask, drop_audio_cond=False,
                             drop_text=False, cache=cache)
        return torch.cat([pred, pred], dim=0)
    return _orig_forward(self, x, cond, text, time, mask=mask, drop_audio_cond=drop_audio_cond,
                         drop_text=drop_text, cfg_infer=cfg_infer, cache=cache)


DiT.forward = _forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="base,cfgskip0.5,cfgskip0.7,compile")
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    import audio as A
    ctx = _ctx()
    A.synth_single_segment(ctx, 0, "DC", "Прогрев.", apply_stress=True)
    out = {}
    for v in a.variants.split(","):
        SKIP_AFTER[0] = float(v[7:]) if v.startswith("cfgskip") else None
        if v == "compile":
            m = ctx.models.tts_model.transformer
            m.transformer_blocks = torch.nn.ModuleList(torch.compile(b, dynamic=True) for b in m.transformer_blocks)
            for _ in range(3):
                A.synth_single_segment(ctx, 0, "DC", PHRASES[0], apply_stress=True)
        d = ROOT / "outputs" / "speed_tricks" / v
        d.mkdir(parents=True, exist_ok=True)
        errs = n = 0
        secs = []
        for rep in range(a.reps):
            for i, ph in enumerate(PHRASES):
                t0 = time.time()
                wav = A.synth_single_segment(ctx, i, "DC", ph, apply_stress=True)
                secs.append(time.time() - t0)
                if not wav:
                    errs += len(words(ph)); n += len(words(ph)); continue
                heard = A.transcribe_audio_file(ctx, wav, engine="gigaam") or ""
                e, m_ = wer(ph, heard); errs += e; n += m_
                if rep == 0:
                    shutil.copy(wav, d / f"{i:02d}.wav")
                if e:
                    print(f"  {v} [{i}] {e} err: {heard}", flush=True)
        secs.sort()
        out[v] = {"wer_pct": round(100 * errs / n, 2), "median_s": round(secs[len(secs) // 2], 3)}
        print(f"{v}: {out[v]}", flush=True)
    json.dump(out, open(ROOT / "outputs" / "speed_tricks" / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main()

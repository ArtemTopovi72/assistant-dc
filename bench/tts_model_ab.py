"""Current Russian F5 checkpoint vs ESpeech-TTS-1_RL-V2 (RL-tuned Russian F5) at the same 10 steps, same voice, same
text pipeline (stress marks, F5 preprocessing). WER by GigaAM, speaker similarity to the reference (Resemblyzer),
liveliness (F0 spread). WAVs in outputs/model_ab/<model>/ for a listen.

    unset F5_TEST_RUN; venv/Scripts/python.exe -u bench/tts_model_ab.py [--voice DC_short_ref12.wav]
"""
import argparse
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
for sub in ("", "bench", "voice", "agent", "core", "media", "bot"):
    sys.path.insert(0, str(ROOT / sub))
from tts_cfg_ab import PHRASES, wer, liveliness      # noqa: E402

MORE = [
    "Привет! Сегодня в Москве облачно, около двенадцати градусов, к вечеру возможен дождь.",
    "Я нашёл три подходящих варианта, самый дешёвый стоит полторы тысячи рублей.",
    "Напомню завтра в девять утра про встречу с врачом.",
    "Если коротко, то ошибка была в том, что файл открывался дважды.",
    "Поезд отправляется с Ленинградского вокзала в двадцать три тридцать.",
    "Спасибо, что подождал! Вот что удалось выяснить по твоему вопросу.",
]
CKPT = ROOT / "models_ext" / "espeech-rl-v2" / "espeech_tts_rlv2.pt"
VOCAB = ROOT / "models_ext" / "espeech-rl-v2" / "vocab.txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="DC_short_ref12.wav")
    a = ap.parse_args()
    import numpy as np
    import config
    import live_tg_drive as D
    import voice_clone
    from audio import transcribe_audio_file
    from f5_tts.infer.utils_infer import load_model
    from f5_tts.model import DiT
    from resemblyzer import VoiceEncoder, preprocess_wav
    _bot, ctx = D.build()
    cur = ctx.models.tts_model
    cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    rl = load_model(DiT, cfg, str(CKPT), vocab_file=str(VOCAB), device=config.DEVICE)
    out = ROOT / "outputs" / "model_ab"
    ref, ref_text = voice_clone.prepare_reference(ctx, str(ROOT / a.voice), str(out / "_ref"), lang="ru")
    enc = VoiceEncoder()
    ref_emb = enc.embed_utterance(preprocess_wav(ref))
    texts = list(PHRASES.values()) + MORE
    res = {}
    for name, model in (("current", cur), ("espeech_rl_v2", rl)):
        ctx.models.tts_model = model
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        voice_clone.speak(ctx, ref, ref_text, "Прогрев.", str(out / "_tmp"))
        rows = []
        for i, text in enumerate(texts):
            for rep in range(2):
                wav = voice_clone.speak(ctx, ref, ref_text, text, str(d))
                heard = transcribe_audio_file(ctx, wav, lang_hint="ru") or ""
                st, sw = liveliness(wav)
                sim = float(enc.embed_utterance(preprocess_wav(wav)) @ ref_emb)
                os.replace(wav, d / f"t{i}_{rep}.wav")
                rows.append((wer(text, heard), sim, st))
                print(f"{name:14s} t{i} r{rep}  WER {rows[-1][0]:.2f}  SIM {sim:.3f}  F0sd {st:.2f}", flush=True)
        res[name] = np.mean(rows, axis=0)
    ctx.models.tts_model = cur
    print("== mean")
    for name, (w, s, f) in res.items():
        print(f"{name:14s} WER {w:.3f}  SIM {s:.3f}  F0 sd {f:.2f} st")


if __name__ == "__main__":
    main()

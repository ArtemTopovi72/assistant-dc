"""EmoSteer on the Russian F5 at 10 steps: steering vectors (emotion - neutral) from Dusha clips, alpha sweep,
judged by a Russian SER model (target-emotion probability), WER (GigaAM) and speaker similarity.

    unset F5_TEST_RUN; venv/Scripts/python.exe -u bench/emosteer_ab.py [--n 30] [--alphas 0,0.1,0.2,0.4]
"""
import argparse
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
for sub in ("", "bench", "voice", "agent", "core", "media", "bot"):
    sys.path.insert(0, str(ROOT / sub))
from tts_cfg_ab import PHRASES, wer                      # noqa: E402

CLIPS = ROOT / "models_ext" / "dusha" / "clips"
SER = ROOT / "models_ext" / "ser-dusha"
EMOS = ("angry", "sad", "positive")
TEXTS = [PHRASES["narration"], PHRASES["question"],
         "Напомню завтра в девять утра про встречу с врачом.",
         "Поезд отправляется с Ленинградского вокзала в двадцать три тридцать."]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--alphas", default="0,0.1,0.2,0.4")
    ap.add_argument("--blocks", default="all", help="all | lo (0-10) | hi (11-21) | mid (6-15)")
    ap.add_argument("--steps", type=int, default=0, help="steer only the first N flow steps (0 = all)")
    ap.add_argument("--genonly", action="store_true", help="leave reference-audio tokens untouched")
    ap.add_argument("--tag", default="")
    ap.add_argument("--voice", default="DC_short_ref12.wav")
    a = ap.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    import librosa
    import config
    import live_tg_drive as D
    import voice_clone
    import emosteer
    from audio import transcribe_audio_file
    from f5_tts.infer.utils_infer import infer_process
    from resemblyzer import VoiceEncoder, preprocess_wav
    from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
    _bot, ctx = D.build()
    model = ctx.models.tts_model
    st = emosteer.Steerer(model)
    out = ROOT / "outputs" / "emosteer"
    out.mkdir(parents=True, exist_ok=True)

    ser = HubertForSequenceClassification.from_pretrained(str(SER)).to("cuda").eval()
    fe = Wav2Vec2FeatureExtractor.from_pretrained(str(SER))
    labels = {v: int(k) for k, v in ser.config.id2label.items()}

    def emo_probs(path):
        y, _ = librosa.load(path, sr=16000)
        inp = fe(y, sampling_rate=16000, return_tensors="pt").to("cuda")
        with torch.no_grad():
            return torch.softmax(ser(**inp).logits, -1)[0].cpu().numpy()

    vf = out / "vecs.pt"
    # 1. activations of Dusha clips (reconstruct each clip from itself so the DiT sees that emotion)
    acts = {}
    for emo in (() if vf.exists() else ("neutral",) + EMOS):
        rows = []
        for clip in sorted(CLIPS.glob(f"{emo}_*.wav"))[: a.n]:
            text = (transcribe_audio_file(ctx, str(clip), lang_hint="ru") or "").strip()
            if len(text) < 6:
                continue
            st.record()
            try:
                infer_process(str(clip), text, text, model, ctx.models.vocoder, nfe_step=config.TTS_NFE_STEP,
                              cfg_strength=2.0, device=config.DEVICE)
                rows.append(st.take())
            except Exception as e:
                st.off(); print("skip", clip.name, e)
        acts[emo] = rows
        print(f"{emo}: {len(rows)} clips", flush=True)
    if vf.exists():
        vecs = torch.load(vf)
    else:
        vecs = {e: emosteer.build_vectors(acts[e], acts["neutral"]) for e in EMOS}
        torch.save(vecs, vf)
    nb = len(model.transformer.transformer_blocks)
    blocks = {"all": None, "lo": set(range(nb // 2)), "hi": set(range(nb // 2, nb)), "mid": set(range(nb // 4, nb * 3 // 4))}[a.blocks]
    ref, ref_text = voice_clone.prepare_reference(ctx, str(ROOT / a.voice), str(out / "_ref"), lang="ru")
    skip = int(sf.info(ref).duration * 24000 / 256) if a.genonly else 0

    # 2. sweep
    enc = VoiceEncoder()
    ref_emb = enc.embed_utterance(preprocess_wav(ref))
    voice_clone.speak(ctx, ref, ref_text, "Прогрев.", str(out / "_tmp"))
    alphas = [float(x) for x in a.alphas.split(",")]
    print(f"{'emo':9} {'alpha':>6} {'P(target)':>10} {'P(neutral)':>11} {'WER':>6} {'SIM':>6}")
    for emo in EMOS:
        for al in alphas:
            st.apply(vecs[emo], al, skip=skip, blocks=blocks, steps=a.steps or None)
            rows = []
            d = out / f"{emo}_{a.tag}{al:g}"
            d.mkdir(exist_ok=True)
            for i, text in enumerate(TEXTS):
                wav = voice_clone.speak(ctx, ref, ref_text, text, str(d))
                heard = transcribe_audio_file(ctx, wav, lang_hint="ru") or ""
                p = emo_probs(wav)
                sim = float(enc.embed_utterance(preprocess_wav(wav)) @ ref_emb)
                os.replace(wav, d / f"t{i}.wav")
                rows.append((p[labels[emo]], p[labels["neutral"]], wer(text, heard), sim))
            m = np.mean(rows, axis=0)
            print(f"{emo:9} {al:6.3f} {m[0]:10.3f} {m[1]:11.3f} {m[2]:6.3f} {m[3]:6.3f}", flush=True)
        st.off()


if __name__ == "__main__":
    main()

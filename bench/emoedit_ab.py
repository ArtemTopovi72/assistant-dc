"""SEmoEdit on our F5: emotional reference = the user's own voice shaped by PSOLA (voice/prosody.py), then velocity transport.
    unset F5_TEST_RUN; venv/Scripts/python.exe -u bench/emoedit_ab.py [--strengths 0.5,1,1.5]
Output: outputs/emoedit/<emotion>_<strength>/t<i>.wav + SER / F0 / WER table."""
import argparse, os, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
for sub in ("", "bench", "voice", "agent", "core", "media", "bot"):
    sys.path.insert(0, str(ROOT / sub))
from tts_cfg_ab import PHRASES, wer, liveliness   # noqa

TEXTS = [PHRASES["narration"], PHRASES["question"], "Напомню завтра в девять утра про встречу с врачом.",
         "Поезд отправляется с Ленинградского вокзала в двадцать три тридцать."]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strengths", default="1.0")
    ap.add_argument("--emotions", default="joy,anger,sad,neutral")
    ap.add_argument("--voice", default="DC_short_ref12.wav")
    a = ap.parse_args()
    import numpy as np, soundfile as sf, torch, librosa
    import live_tg_drive as D, voice_clone, prosody, emoedit
    from audio import transcribe_audio_file
    from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
    _b, ctx = D.build()
    out = ROOT / "outputs" / "emoedit"; out.mkdir(parents=True, exist_ok=True)
    rs, rs_text = voice_clone.prepare_reference(ctx, str(ROOT / a.voice), str(out / "_ref"), lang="ru")
    ser = HubertForSequenceClassification.from_pretrained(str(ROOT / "models_ext/ser-dusha")).to("cuda").eval()
    fe = Wav2Vec2FeatureExtractor.from_pretrained(str(ROOT / "models_ext/ser-dusha"))
    lab = {v: int(k) for k, v in ser.config.id2label.items()}
    tgt_key = {"joy": "positive", "anger": "angry", "sad": "sad", "neutral": "neutral"}
    def probs(p):
        y, _ = librosa.load(p, sr=16000)
        with torch.no_grad():
            return torch.softmax(ser(**fe(y, sampling_rate=16000, return_tensors="pt").to("cuda")).logits, -1)[0].cpu().numpy()
    print(f"{'emo':8} {'str':>4} {'P(tgt)':>7} {'WER':>5} {'F0sd':>5}")
    for emo in a.emotions.split(","):
        rt = str(out / f"rt_{emo}.wav"); prosody.shape(rs, emo, rt)
        for s in [float(x) for x in a.strengths.split(",")]:
            d = out / f"{emo}_{s:g}"; d.mkdir(exist_ok=True); rows = []
            for i, text in enumerate(TEXTS):
                gt = ctx_text(ctx, text)
                w = emoedit.edit(ctx, rs, rs_text, rt, rs_text, gt, strength=s)
                p = str(d / f"t{i}.wav"); sf.write(p, w, 24000)
                heard = transcribe_audio_file(ctx, p, lang_hint="ru") or ""
                pr = probs(p); st, _ = liveliness(p)
                rows.append((pr[lab[tgt_key[emo]]], wer(text, heard), st))
            m = np.mean(rows, 0); print(f"{emo:8} {s:4.1f} {m[0]:7.3f} {m[1]:5.3f} {m[2]:5.2f}", flush=True)

def ctx_text(ctx, t):
    from audio import preprocess_text_for_synthesis
    return preprocess_text_for_synthesis(ctx, t, use_censoring=True, apply_stress=True)

if __name__ == "__main__":
    main()

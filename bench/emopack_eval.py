"""Stage C: F5 with each converted emotional reference (Dusha prosody in the user's timbre): emotion / WER / similarity.
    unset F5_TEST_RUN; venv/Scripts/python.exe -u bench/emopack_eval.py"""
import json, os, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
for sub in ("", "bench", "voice", "agent", "core", "media", "bot"):
    sys.path.insert(0, str(ROOT / sub))
from tts_cfg_ab import PHRASES, wer, liveliness   # noqa
TEXTS = [PHRASES["narration"], PHRASES["question"], "Напомню завтра в девять утра про встречу с врачом.",
         "Поезд отправляется с Ленинградского вокзала в двадцать три тридцать."]
import numpy as np, librosa, torch
import live_tg_drive as D, voice_clone
from audio import transcribe_audio_file
from resemblyzer import VoiceEncoder, preprocess_wav
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
_b, ctx = D.build()
pick = json.loads((ROOT / "outputs/emopack/pick.json").read_text(encoding="utf-8"))
ser = HubertForSequenceClassification.from_pretrained(str(ROOT / "models_ext/ser-dusha")).to("cuda").eval()
fe = Wav2Vec2FeatureExtractor.from_pretrained(str(ROOT / "models_ext/ser-dusha")); lab = {v: int(k) for k, v in ser.config.id2label.items()}
enc = VoiceEncoder()
out = ROOT / "outputs/emopack/synth"; out.mkdir(parents=True, exist_ok=True)
dc, dc_text = voice_clone.prepare_reference(ctx, str(ROOT / "DC_short_ref12.wav"), str(out / "_r"), lang="ru")
dc_emb = enc.embed_utterance(preprocess_wav(dc))
def probs(p):
    y, _ = librosa.load(p, sr=16000)
    with torch.no_grad():
        return torch.softmax(ser(**fe(y, sampling_rate=16000, return_tensors="pt").to("cuda")).logits, -1)[0].cpu().numpy()
def run(name, ref, ref_text, target):
    d = out / name; d.mkdir(exist_ok=True); rows = []
    for i, t in enumerate(TEXTS):
        w = voice_clone.speak(ctx, ref, ref_text, t, str(d))
        if not w: continue
        p = str(d / f"t{i}.wav"); os.replace(w, p)
        heard = transcribe_audio_file(ctx, p, lang_hint="ru") or ""
        pr = probs(p); sim = float(enc.embed_utterance(preprocess_wav(p)) @ dc_emb)
        rows.append((pr[lab[target]], wer(t, heard), sim, liveliness(p)[0]))
    m = np.mean(rows, 0); print(f"{name:12} P({target[:5]}) {m[0]:.3f} WER {m[1]:.3f} SIM {m[2]:.3f} F0sd {m[3]:.2f}", flush=True)
run("base_DC", dc, dc_text, "neutral")
for emo in ("angry", "sad", "positive", "neutral"):
    for i, r in enumerate(pick[emo]):
        c = ROOT / f"outputs/emopack/conv/{emo}_{i}.wav"
        if c.exists():
            ref, rt = str(c), (transcribe_audio_file(ctx, str(c), lang_hint="ru") or "").strip()
            run(f"{emo}_{i}", ref, rt, emo)

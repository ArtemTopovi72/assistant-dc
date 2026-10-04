"""Stage A: pick the most clearly emotional real Russian Dusha clips (SER confidence) + ASR text -> outputs/emopack/pick.json
    unset F5_TEST_RUN; venv/Scripts/python.exe bench/emopack_pick.py [--per 6]"""
import argparse, json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
for sub in ("", "bench", "voice", "agent", "core", "media", "bot"):
    sys.path.insert(0, str(ROOT / sub))
ap = argparse.ArgumentParser(); ap.add_argument("--per", type=int, default=6); a = ap.parse_args()
import librosa, torch
import live_tg_drive as D
from audio import transcribe_audio_file
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
_b, ctx = D.build()
SER = ROOT / "models_ext/ser-dusha"
ser = HubertForSequenceClassification.from_pretrained(str(SER)).to("cuda").eval()
fe = Wav2Vec2FeatureExtractor.from_pretrained(str(SER)); lab = {v: int(k) for k, v in ser.config.id2label.items()}
out = ROOT / "outputs" / "emopack"; out.mkdir(parents=True, exist_ok=True)
pick = {}
for emo in ("angry", "sad", "positive", "neutral"):
    rows = []
    for c in sorted((ROOT / "models_ext/dusha/clips").glob(f"{emo}_*.wav")):
        y, _ = librosa.load(c, sr=16000)
        if not 3.0 <= len(y) / 16000 <= 9.0:
            continue
        with torch.no_grad():
            p = torch.softmax(ser(**fe(y, sampling_rate=16000, return_tensors="pt").to("cuda")).logits, -1)[0, lab[emo]].item()
        rows.append((p, str(c)))
    rows.sort(reverse=True)
    pick[emo] = []
    for p, c in rows:
        t = (transcribe_audio_file(ctx, c, lang_hint="ru") or "").strip()
        if len(t) >= 15:
            pick[emo].append({"clip": c, "p": round(p, 3), "text": t})
        if len(pick[emo]) >= a.per:
            break
    print(emo, [(r["p"], r["text"][:40]) for r in pick[emo]], flush=True)
(out / "pick.json").write_text(json.dumps(pick, ensure_ascii=False, indent=1), encoding="utf-8")

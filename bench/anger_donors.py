"""Better anger donors: scan Dusha test split for angry clips, rank by SER confidence, convert the best to the DC timbre
(Seed-VC), synthesise phrases with F5 on each converted reference, keep the ones that still read as anger.
    unset F5_TEST_RUN; venv/Scripts/python.exe -u bench/anger_donors.py [--scan 700] [--top 12] [--keep 3]
Writes outputs/anger_donors/result.json; with --apply replaces the anger entries of voice/emodonors.json."""
import argparse, csv, json, os, subprocess, sys, tarfile
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
for sub in ("", "bench", "voice", "agent", "core", "media", "bot"):
    sys.path.insert(0, str(ROOT / sub))
os.chdir(ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--scan", type=int, default=700); ap.add_argument("--top", type=int, default=12)
ap.add_argument("--keep", type=int, default=3); ap.add_argument("--apply", action="store_true")
a = ap.parse_args()
OUT = ROOT / "outputs/anger_donors"; (OUT / "raw").mkdir(parents=True, exist_ok=True); (OUT / "conv").mkdir(exist_ok=True)
RAW = ROOT / "models_ext/dusha/raw/data"
angry = {r["file_name"] for r in csv.DictReader(open(RAW / "test.csv", encoding="utf-8")) if r["label"] == "angry"}

# 1. pull angry clips out of the tarball (stream, stop after --scan)
got = []
with tarfile.open(RAW / "test.tar.gz", "r|gz") as tf:
    for m in tf:
        if m.isfile() and m.name.lstrip("./") in angry:
            dst = OUT / "raw" / Path(m.name).name
            dst.write_bytes(tf.extractfile(m).read()); got.append(dst)
            if len(got) >= a.scan:
                break
print("extracted", len(got), flush=True)

import librosa, numpy as np, torch
from tts_cfg_ab import wer
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
SER = ROOT / "models_ext/ser-dusha"
ser = HubertForSequenceClassification.from_pretrained(str(SER)).to("cuda").eval()
fe = Wav2Vec2FeatureExtractor.from_pretrained(str(SER)); lab = {v: int(k) for k, v in ser.config.id2label.items()}


def p_angry(y):
    with torch.no_grad():
        return torch.softmax(ser(**fe(y, sampling_rate=16000, return_tensors="pt").to("cuda")).logits, -1)[0, lab["angry"]].item()


# 2. rank: duration 3-9 s, loud enough, SER confidence
rows = []
for c in got:
    y, _ = librosa.load(c, sr=16000)
    if not 3.0 <= len(y) / 16000 <= 9.0:
        continue
    rows.append((p_angry(y), str(c)))
rows.sort(reverse=True)
print("candidates", len(rows), "best", [round(r[0], 3) for r in rows[:5]], flush=True)

import live_tg_drive as D, voice_clone
from audio import transcribe_audio_file
_b, ctx = D.build()
cand = []
for p, c in rows:
    t = (transcribe_audio_file(ctx, c, lang_hint="ru") or "").strip()
    if len(t) >= 15:
        cand.append({"clip": c, "p": round(p, 3), "text": t})
    if len(cand) >= a.top:
        break
print("candidates with text", len(cand), flush=True)

# 3. Seed-VC onto the DC timbre (one worker, models loaded once)
ref, rt = voice_clone.prepare_reference(ctx, str(ROOT / "DC_short_ref12.wav"), str(OUT / "_r"), lang="ru")
jobs = [{"source": r["clip"], "target": str(Path(ref).resolve()), "out": str(OUT / "conv" / f"a{i}.wav")} for i, r in enumerate(cand)]
(OUT / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
PY = Path(__import__("config").venv_python(ROOT / "venv_qwen"))
subprocess.run([str(PY), str(ROOT / "scripts/seedvc_batch.py"), str(OUT / "jobs.json")], cwd=str(ROOT), check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# 4. F5 on each converted reference: does it still sound angry, and is the text intact?
PH = ["Как ты посмел так со мной поступить, убирайся отсюда немедленно!",
      "Я же сто раз говорил тебе не трогать мои вещи, сколько можно повторять?",
      "Завтра в девять утра у вас приём у врача в третьем кабинете."]
res = []
for i, r in enumerate(cand):
    cw = OUT / "conv" / f"a{i}.wav"
    if not cw.exists():
        continue
    ct = (transcribe_audio_file(ctx, str(cw), lang_hint="ru") or "").strip()
    ps, ws = [], []
    for j, t in enumerate(PH):
        w = voice_clone.speak(ctx, str(cw), ct, t, str(OUT), emotion="", ik3=False)
        if not w:
            continue
        y, _ = librosa.load(w, sr=16000)
        ps.append(p_angry(y)); ws.append(wer(t, transcribe_audio_file(ctx, w, lang_hint="ru") or ""))
    if ps:
        res.append({**r, "conv": str(cw), "conv_text": ct, "p_synth": round(float(np.mean(ps)), 3), "wer": round(float(np.mean(ws)), 3)})
        print(i, r["p"], "->", res[-1]["p_synth"], "WER", res[-1]["wer"], "|", r["text"][:40], flush=True)
res.sort(key=lambda r: r["p_synth"] - r["wer"], reverse=True)
(OUT / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("BEST", [(r["p_synth"], r["wer"]) for r in res[: a.keep]], flush=True)

if a.apply and res:
    import shutil
    man = ROOT / "voice/emodonors.json"; d = json.loads(man.read_text(encoding="utf-8"))
    new = []
    for k, r in enumerate(res[: a.keep]):
        name = f"anger_{k}.wav"
        shutil.copy(r["clip"], ROOT / "models_ext/emodonors" / name)
        new.append({"file": name, "text": r["text"], "src": Path(r["clip"]).name})
    d["anger"] = new
    man.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    print("applied", len(new), flush=True)

"""Russian IK-3 question: rise on the stressed vowel of the centre word, fall after. Measures F0 around each candidate word.
    venv/Scripts/python.exe bench/prosody_q.py"""
import sys, re
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
for sub in ("", "voice", "agent", "core", "media", "bot", "bench"):
    sys.path.insert(0, str(ROOT / sub))
import numpy as np, librosa
import live_tg_drive as D, prosody
from audio import preprocess_text_for_synthesis
_b, ctx = D.build()
Q = ["Ты правда думаешь, что всё так просто?", "Вы завтра приедете на вокзал?", "Это он сломал мою машину?"]
out = ROOT / "outputs" / "prosody_q"; out.mkdir(parents=True, exist_ok=True)
import voice_clone
ref, rt = voice_clone.prepare_reference(ctx, str(ROOT / "DC_short_ref12.wav"), str(out / "_r"), lang="ru")
for qi, q in enumerate(Q):
    wav = voice_clone.speak(ctx, ref, rt, q, str(out))
    base = str(out / f"q{qi}_base.wav"); Path(wav).replace(base)
    words = prosody.word_times(ctx, base); st = preprocess_text_for_synthesis(ctx, q, use_censoring=False, apply_stress=True)
    print(qi, q, "|", st, [(w, round(a, 2), round(b, 2)) for w, a, b in words])
    for fi in range(len(words)):
        if len(words[fi][0]) < 4: continue
        c = prosody.centre_window(words, st, fi)
        d = str(out / f"q{qi}_focus{fi}_{re.sub(r'[^А-Яа-яЁё]', '', words[fi][0])}.wav"); prosody.shape(base, "question", d, center=c)
        y, sr = librosa.load(d, sr=16000); f0, v, _ = librosa.pyin(y, fmin=60, fmax=500, sr=sr, hop_length=160)
        tt = np.arange(len(f0)) * 160 / sr; ok = v & ~np.isnan(f0)
        med = np.median(f0[ok]); win = ok & (tt >= c[0] - 0.05) & (tt <= c[1] + 0.05)
        rel = 12 * np.log2(np.nanmedian(f0[win]) / med) if win.any() else float("nan")
        after = ok & (tt > c[1] + 0.3); fall = 12 * np.log2(np.nanmedian(f0[after]) / med) if after.any() else float("nan")
        print("   focus", words[fi][0], "centre", [round(float(x), 2) for x in c], f"on-centre {rel:+.1f}st after {fall:+.1f}st")

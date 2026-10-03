"""Long-audio split for GigaAM: blind 20 s cuts vs cut-at-pause.

Real read speech (ruLibriSpeech test) concatenated into ~60 s takes; each
clip's own edge silence is trimmed so the only pauses are inside the speech.
    venv/Scripts/python bench/asr_long_split.py [n_clips]
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")
import numpy as np
from bench.asr_ru_shootout import load_samples, normalise
import audio, onnx_asr


def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        p, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (rw != hw))
    return d[len(h)], len(r)


def trim(x, thr=0.01):
    idx = np.where(np.abs(x) > thr)[0]
    return x[idx[0]:idx[-1] + 1] if len(idx) else x


n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
rows = load_samples(n, "rulibrispeech")
m = onnx_asr.load_model("gigaam-v3-e2e-rnnt")
takes, cur, txt = [], [], []
for r in rows:
    x, t = (r["audio"], r["text"]) if isinstance(r, dict) else r[:2]
    cur.append(trim(x)); txt.append(t)
    if sum(len(c) for c in cur) > 60 * 16000:
        takes.append((np.concatenate(cur), " ".join(txt))); cur, txt = [], []
tot = {"blind": [0, 0], "pause": [0, 0]}
for x, ref in takes:
    for mode in tot:
        if mode == "blind":
            parts = [x[i:i + 320000] for i in range(0, len(x), 320000)]
        else:
            parts = audio._split_at_pauses(x)
        hyp = " ".join((m.recognize(p, sample_rate=16000) or "") for p in parts if len(p) >= 1600)
        e, nw = wer(normalise(ref), normalise(hyp))
        tot[mode][0] += e; tot[mode][1] += nw
print(len(takes), "takes of ~60s")
for k, (e, nw) in tot.items():
    print(f"{k:6s} WER {100 * e / nw:.2f}%  ({e}/{nw})")

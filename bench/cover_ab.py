"""A/B of the melody the MuLaCover cover follows: shipped YourMT3 on the full mix vs MuScriptor on the vocal stem.
Same seed, same lyrics, same tags; the score is frame-wise chroma cosine between the vocal stems of reference and cover.
    unset F5_TEST_RUN; venv/Scripts/python.exe -u bench/cover_ab.py <song.mp3> [--seed 7]
Stop the desktop app first (the worker wants the card)."""
import json, os, subprocess, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
for sub in ("", "bench", "voice", "agent", "core", "media", "bot"):
    sys.path.insert(0, str(ROOT / sub))
os.chdir(ROOT)
import numpy as np, librosa
import cover, config
src = sys.argv[1]; seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 7
OUT = ROOT / "outputs/cover_ab"; OUT.mkdir(parents=True, exist_ok=True)
ref = cover.to_wav(src, str(OUT / "ref.wav"))
vox = cover.vocal_stem(ref, str(OUT / "ref_vox.wav"))
LYR = cover.cover_lyrics("\n".join(["Мы идём по тёплому асфальту", "Ветер гонит листья вдоль дорог", "Я не знаю, что нас ждёт, но завтра",
                                   "Будет солнце, будет новый срок", "", "Пой со мной, пока горит рассвет",
                                   "Пой со мной, пока нас слышит свет", "Нет ни страха, нет вопросов, нет", "Только этот тихий добрый свет"]))
TAGS = "topic:[Hope]; genre:[pop]; instrument:[Piano,drums,electric guitar]; mood:[hopeful]"
PY = config.venv_python(ROOT / "venv_mula")


def render(name, env):
    out = OUT / f"{name}.wav"
    if not out.exists():
        job = {"ref": ref, "vocals": vox, "lyrics": LYR, "tags": TAGS, "seed": seed, "out": str(out)}
        jf = OUT / f"{name}.json"; jf.write_text(json.dumps(job), encoding="utf-8")
        r = subprocess.run([str(PY), str(ROOT / "scripts/mulacover_render.py"), str(jf)], cwd=str(ROOT),
                           env={**os.environ, **env}, capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(name, "rc", r.returncode, [l for l in r.stdout.splitlines() if "muscriptor" in l.lower()][:3], flush=True)
        if r.returncode:
            print(r.stderr[-800:])
    return str(out)


def chroma(wav):
    y, sr = librosa.load(wav, sr=22050)
    return librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=1024)


def score(a, b):
    n = min(a.shape[1], b.shape[1])
    a, b = a[:, :n], b[:, :n]
    cos = (a * b).sum(0) / (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) + 1e-9)
    return float(cos.mean())


ref_c = chroma(vox or ref)
for name, env in (("yourmt3", {"MULACOVER_YOURMT3": "1"}), ("muscriptor", {})):
    w = render(name, env)
    if not os.path.exists(w):
        continue
    v = cover.vocal_stem(w, str(OUT / f"{name}_vox.wav"))
    print(f"{name:11} melody-chroma similarity to the original vocal: {score(ref_c, chroma(v or w)):.3f}", flush=True)

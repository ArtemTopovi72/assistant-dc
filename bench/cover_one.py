"""One cover of a real song, with the numbers that say how close it is: length, tempo, key, vocal-melody chroma.
    unset F5_TEST_RUN; venv/Scripts/python.exe -u bench/cover_one.py song.mp4 lyrics.txt "<tags>" [--bpm 136] [--name run1]
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

src, lyr_file, tags = sys.argv[1], sys.argv[2], sys.argv[3]
arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
bpm, name = arg("--bpm"), arg("--name", "one")
OUT = ROOT / "outputs/cover_one"; OUT.mkdir(parents=True, exist_ok=True)
ref = cover.to_wav(src, str(OUT / "ref.wav"))
vox = cover.vocal_stem(ref, str(OUT / "ref_vox.wav"))
out = OUT / f"{name}.wav"
job = {"ref": ref, "vocals": vox, "lyrics": cover.cover_lyrics(Path(lyr_file).read_text(encoding="utf-8")), "tags": tags,
       "seed": 7, "out": str(out), "bpm": float(bpm) if bpm else None}
jf = OUT / f"{name}.json"; jf.write_text(json.dumps(job), encoding="utf-8")
r = subprocess.run([str(config.venv_python(ROOT / "venv_mula")), str(ROOT / "scripts/mulacover_render.py"), str(jf)],
                   cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
print([l for l in r.stdout.splitlines() if "muscriptor" in l.lower()][:4], "rc", r.returncode)
if r.returncode:
    print(r.stderr[-800:])


def stats(p):
    y, sr = librosa.load(p, sr=22050)
    t, _ = librosa.beat.beat_track(y=y, sr=sr)
    return y, sr, len(y) / sr, float(np.atleast_1d(t)[0])


ya, sr, da, ta = stats(ref)
yb, _, db, tb = stats(str(out))
print(f"duration {da:.0f}s -> {db:.0f}s   tempo {ta:.0f} -> {tb:.0f} bpm")
va = librosa.feature.chroma_cqt(y=librosa.load(vox, sr=22050)[0], sr=sr, hop_length=1024)
vb = librosa.feature.chroma_cqt(y=librosa.load(cover.vocal_stem(str(out), str(OUT / f"{name}_vox.wav")), sr=22050)[0], sr=sr, hop_length=1024)
n = min(va.shape[1], vb.shape[1])
cos = (va[:, :n] * vb[:, :n]).sum(0) / (np.linalg.norm(va[:, :n], axis=0) * np.linalg.norm(vb[:, :n], axis=0) + 1e-9)
print(f"vocal melody chroma similarity {cos.mean():.3f}")

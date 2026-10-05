"""The same song in another voice: Demucs splits the recording, the ORIGINAL backing track stays, the vocal stem goes
through Seed-VC singing conversion (F0 kept) onto a target voice, and the two are mixed back. Nothing is regenerated, so it
sounds like the original by construction (the words are the original's).
    unset F5_TEST_RUN; venv/Scripts/python.exe -u bench/svc_cover.py song.mp4 [target.wav] [--name x]"""
import json, os, subprocess, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
for sub in ("", "bench", "voice", "agent", "core", "media", "bot"):
    sys.path.insert(0, str(ROOT / sub))
os.chdir(ROOT)
import numpy as np, soundfile as sf
import config, cover, mashup_stems
from mashup_stems import SAMPLE_RATE

src = sys.argv[1]
target = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else str(config.DC_REF_WAV)
name = sys.argv[sys.argv.index("--name") + 1] if "--name" in sys.argv else "svc"
OUT = ROOT / "outputs/svc_cover"; OUT.mkdir(parents=True, exist_ok=True)
ref = cover.to_wav(src, str(OUT / "ref.wav"))
stems = mashup_stems.separate(ref)
sf.write(OUT / "vocals.wav", stems["vocals"], SAMPLE_RATE)
back = mashup_stems.backing_of(stems)
sf.write(OUT / "backing.wav", back, SAMPLE_RATE)
conv = OUT / f"{name}_vocals.wav"
if not conv.exists():
    (OUT / "jobs.json").write_text(json.dumps([{"source": str(OUT / "vocals.wav"), "target": target, "out": str(conv)}]), encoding="utf-8")
    subprocess.run([str(config.venv_python(ROOT / "venv_qwen")), str(ROOT / "scripts/seedvc_batch.py"), str(OUT / "jobs.json")],
                   cwd=str(ROOT), check=True)
v, sr = sf.read(conv, dtype="float32")
if v.ndim > 1:
    v = v.mean(1)
if sr != SAMPLE_RATE:
    import librosa
    v = librosa.resample(v, orig_sr=sr, target_sr=SAMPLE_RATE)
n = min(len(v), len(back))
mix = back[:n] + v[:n]
mix /= max(1.0, float(np.abs(mix).max()) / 0.95)
wav = OUT / f"{name}.wav"
sf.write(wav, mix, SAMPLE_RATE)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-b:a", "192k", str(OUT / f"{name}.mp3")], check=True)
print("done", OUT / f"{name}.mp3")

"""Our words in the ORIGINAL singer's voice over the ORIGINAL backing track (prototype).
Demucs splits the recording; Whisper times the words of the original vocal; our lines are spread over those words by syllable
share, spoken by TTS, squeezed into their spans, converted by Seed-VC onto the original singer (target = the original vocal
itself) and mixed over the original backing. Pitch is not imposed yet (--pitch uses the original contour via pyworld).
    unset F5_TEST_RUN; venv/Scripts/python.exe -u bench/svc_lyrics.py song.mp4 lyrics.txt [--name x] [--pitch]
Stop the desktop app first."""
import json, os, re, subprocess, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
for sub in ("", "bench", "voice", "agent", "core", "media", "bot"):
    sys.path.insert(0, str(ROOT / sub))
os.chdir(ROOT)
import librosa, numpy as np, soundfile as sf
import config, cover, mashup_stems
from mashup_stems import SAMPLE_RATE as SR

src, lyr_file = sys.argv[1], sys.argv[2]
name = sys.argv[sys.argv.index("--name") + 1] if "--name" in sys.argv else "words"
OUT = ROOT / "outputs/svc_lyrics"; (OUT / "lines").mkdir(parents=True, exist_ok=True)
VOW = re.compile(r"[аеёиоуыэюяaeiouy]", re.I)
syl = lambda s: max(1, len(VOW.findall(s)))

# 1. stems (cached)
if not (OUT / "vocals.wav").exists():
    ref = cover.to_wav(src, str(OUT / "ref.wav"))
    st = mashup_stems.separate(ref)
    sf.write(OUT / "vocals.wav", st["vocals"], SR)
    sf.write(OUT / "backing.wav", mashup_stems.backing_of(st), SR)
vox, _ = sf.read(OUT / "vocals.wav", dtype="float32"); back, _ = sf.read(OUT / "backing.wav", dtype="float32")

import live_tg_drive as D
_b, ctx = D.build()
# 2. word times of the original vocal (cached)
wj = OUT / "words.json"
if not wj.exists():
    segs, _ = ctx.models.whisper.transcribe(str(OUT / "vocals.wav"), language="ru", word_timestamps=True)
    words = [{"w": w.word.strip(), "s": w.start, "e": w.end} for sg in segs for w in (sg.words or [])]
    wj.write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
words = json.loads(wj.read_text(encoding="utf-8"))
print("original words timed:", len(words), flush=True)

# 3. our lines -> spans of the original words, by cumulative syllable share
lines = [l.strip() for l in Path(lyr_file).read_text(encoding="utf-8").splitlines()
         if l.strip() and not re.fullmatch(r"\[.*\]", l.strip())]
ow = np.cumsum([syl(w["w"]) for w in words]); ow = ow / ow[-1]
nl = np.cumsum([syl(l) for l in lines]); nl = nl / nl[-1]
spans = []
for i, l in enumerate(lines):
    lo, hi = (nl[i - 1] if i else 0.0), nl[i]
    idx = [k for k in range(len(words)) if lo <= (ow[k - 1] if k else 0.0) + (ow[k] - (ow[k - 1] if k else 0.0)) / 2 < hi]
    if idx:
        spans.append((words[idx[0]]["s"], words[idx[-1]]["e"]))
    else:
        spans.append(None)
print("lines", len(lines), "spans", sum(s is not None for s in spans), flush=True)

# 4. speak every line, fit it into its span
import audio
canvas = np.zeros(len(vox), dtype=np.float32)
for i, (l, sp) in enumerate(zip(lines, spans)):
    if sp is None:
        continue
    w = audio.synth_single_segment(ctx, 5000 + i, "DC", l, out_stem=str(OUT / "lines" / f"l{i}"))
    if not w:
        continue
    y, sr = librosa.load(w, sr=SR)
    y, _ = librosa.effects.trim(y, top_db=35)
    want = max(0.3, sp[1] - sp[0])
    rate = float(np.clip(len(y) / SR / want, 0.5, 2.5))
    y = librosa.effects.time_stretch(y, rate=rate)
    s0 = int(sp[0] * SR)
    canvas[s0:s0 + len(y)] += y[:max(0, len(canvas) - s0)]
if "--pitch" in sys.argv:
    # the spoken line takes the ORIGINAL singer's pitch contour (same timeline), so Seed-VC gets a melody to follow
    import pyworld as pw
    R = 22050
    a = librosa.resample(canvas.astype(np.float64), orig_sr=SR, target_sr=R)
    o = librosa.resample(vox.astype(np.float64), orig_sr=SR, target_sr=R)
    fa, ta = pw.harvest(a, R, f0_floor=70, f0_ceil=500, frame_period=10.0)
    fo, _ = pw.harvest(o, R, f0_floor=70, f0_ceil=700, frame_period=10.0)
    n = min(len(fa), len(fo)); fa, fo = fa[:n], fo[:n]
    ok = np.flatnonzero(fo > 0)
    f_new = np.zeros(n)
    if len(ok):
        held = np.interp(np.arange(n), ok, fo[ok])      # unvoiced gaps in the singer hold the last pitch
        f_new = np.where(fa > 0, held, 0.0)
    sp = pw.cheaptrick(a, fa, ta[:len(fa)], R)[:n]; ap = pw.d4c(a, fa, ta[:len(fa)], R)[:n]
    y = pw.synthesize(f_new, sp, ap, R, 10.0)
    canvas = librosa.resample(y.astype(np.float32), orig_sr=R, target_sr=SR)
    canvas = np.pad(canvas, (0, max(0, len(vox) - len(canvas))))[:len(vox)]
    name += "_pitch"
sf.write(OUT / f"{name}_speech.wav", canvas, SR)

# 5. onto the original singer: the target is the loudest 20 s of the original vocal
hop = SR * 20
best = max(range(0, max(1, len(vox) - hop), SR * 5), key=lambda a: float(np.abs(vox[a:a + hop]).mean()))
sf.write(OUT / "singer_ref.wav", vox[best:best + hop], SR)
conv = OUT / f"{name}_conv.wav"
if conv.exists():
    conv.unlink()
(OUT / "jobs.json").write_text(json.dumps([{"source": str(OUT / f"{name}_speech.wav"), "target": str(OUT / "singer_ref.wav"), "out": str(conv)}]), encoding="utf-8")
subprocess.run([str(config.venv_python(ROOT / "venv_qwen")), str(ROOT / "scripts/seedvc_batch.py"), str(OUT / "jobs.json")], cwd=str(ROOT), check=True)
v, sr = sf.read(conv, dtype="float32")
v = v.mean(1) if v.ndim > 1 else v
if sr != SR:
    v = librosa.resample(v, orig_sr=sr, target_sr=SR)
n = min(len(v), len(back))
mix = back[:n] + v[:n]
mix /= max(1.0, float(np.abs(mix).max()) / 0.95)
sf.write(OUT / f"{name}.wav", mix, SR)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(OUT / f"{name}.wav"), "-b:a", "192k", str(OUT / f"{name}.mp3")], check=True)
print("done", OUT / f"{name}.mp3")

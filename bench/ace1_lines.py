"""ACE-Step v1 FlowEdit, window by window, best of several seeds per window.
The original phrases (Whisper word times) are grouped into ~14 s windows; our lyrics are spread over the windows by SYLLABLE share,
so every window gets as many new syllables as the original sings there. Each window is edited as one piece (source lyrics = what
Whisper heard there), n_avg averages the edit direction, several seeds are tried and the one whose Whisper text is closest to the
target wins; loudness matched to the original, spliced back with a crossfade.
    venv_ace1/Scripts/python.exe bench/ace1_lines.py song.wav words.json lyrics.txt out.wav [n_min=0.3] [seeds=3] [max_windows=0]
Note: in task=edit only guidance_scale / n_min / n_max / n_avg / infer_step / seeds matter (guidance_scale_text/lyric, ERG unused)."""
import difflib, json, os, re, sys
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
song, wj, lyr, out = [os.path.abspath(a) for a in sys.argv[1:5]]
n_min = float(sys.argv[5]) if len(sys.argv) > 5 else 0.3
n_seeds = int(sys.argv[6]) if len(sys.argv) > 6 else 3
limit = int(sys.argv[7]) if len(sys.argv) > 7 else 0
TAGS = os.environ.get("ACE_TAGS", "russian chanson, pop, 90s, male baritone vocal, synthesizer, saxophone, drum machine, melancholic")
A = os.path.join(R, "models_ext", "acestep1"); os.chdir(A); sys.path.insert(0, A)
import numpy as np, soundfile as sf, librosa, tempfile
VOW = re.compile(r"[аеёиоуыэюяaeiouy]", re.I)
syl = lambda s: len(VOW.findall(s))
norm = lambda s: re.sub(r"[^а-яёa-z ]", "", s.lower().replace("ё", "е"))
words = json.load(open(wj, encoding="utf-8"))
lines = [l.strip() for l in open(lyr, encoding="utf-8").read().splitlines() if l.strip() and not re.fullmatch(r"\[.*\]", l.strip())]

phr, cur = [], []                                   # original phrases: (start, end, text)
for w in words:
    if cur and w["s"] - cur[-1]["e"] > 0.4:
        phr.append((cur[0]["s"], cur[-1]["e"], " ".join(x["w"] for x in cur))); cur = []
    cur.append(w)
phr.append((cur[0]["s"], cur[-1]["e"], " ".join(x["w"] for x in cur)))
wins, g = [], []                                    # ~14 s windows of whole phrases
for ph in phr:
    if g and ph[1] - g[0][0] > 14:
        wins.append(g); g = []
    g.append(ph)
wins.append(g)

# our lyrics as a stream of words (line breaks kept), cut into windows by syllable share
stream = [(wd, li) for li, l in enumerate(lines) for wd in l.split()]
need = np.array([sum(syl(p[2]) for p in g) for g in wins], float)
have = np.array([max(1, syl(wd)) for wd, _ in stream], float)
cut = np.searchsorted(np.cumsum(have) / have.sum(), np.cumsum(need) / need.sum(), side="right")
jobs, lo = [], 0
for g, hi in zip(wins, cut):
    chunk = stream[lo:hi]; lo = hi
    if not chunk:
        continue
    tgt, prev = [], None
    for wd, li in chunk:
        if li != prev:
            tgt.append(wd); prev = li
        else:
            tgt[-1] += " " + wd
    jobs.append((g[0][0], g[-1][1], [p[2] for p in g], tgt))
if limit:
    jobs = jobs[:limit]
SKIP = int(os.environ.get("ACE_SKIP", "0"))   # resume: song = the partly edited output, windows before SKIP are done

y, sr = sf.read(song, dtype="float32")
y = y if y.ndim > 1 else np.stack([y, y], 1)
from faster_whisper import WhisperModel
from acestep.pipeline_ace_step import ACEStepPipeline
asr = WhisperModel("large-v3", device="cuda", compute_type="float16")
p = ACEStepPipeline(checkpoint_dir=os.path.join(A, "checkpoints"), dtype="bfloat16", torch_compile=False, cpu_offload=False)
tmp = tempfile.mkdtemp(prefix="acel_")
PAD, XF = 1.0, int(0.4 * sr)
for n, (s, e, src, tgt) in enumerate(jobs):
    if n < SKIP:
        continue
    a, b = max(0, int((s - PAD) * sr)), min(len(y), int((e + PAD) * sr))
    win = os.path.join(tmp, f"w{n}.wav"); sf.write(win, y[a:b], sr)
    best = (-1.0, None, "")
    for k in range(n_seeds):
        o = os.path.join(tmp, f"e{n}_{k}.wav")
        p(audio_duration=(b - a) / sr, prompt=TAGS, lyrics="[verse]\n" + "\n".join(src), infer_step=60, guidance_scale=15.0,
          task="edit", src_audio_path=win, edit_target_prompt=TAGS, edit_target_lyrics="[verse]\n" + "\n".join(tgt),
          edit_n_min=n_min, edit_n_max=1.0, edit_n_avg=3, save_path=o, manual_seeds=str(1000 * n + k))
        heard = " ".join(sg.text for sg in asr.transcribe(o, language="ru")[0])
        score = difflib.SequenceMatcher(None, norm(heard), norm(" ".join(tgt))).ratio()
        if score > best[0]:
            best = (score, o, heard)
    z, zsr = sf.read(best[1], dtype="float32")
    z = z if z.ndim > 1 else np.stack([z, z], 1)
    if zsr != sr:
        z = librosa.resample(z.T, orig_sr=zsr, target_sr=sr).T
    z = z[: b - a]; z = np.pad(z, ((0, (b - a) - len(z)), (0, 0)))
    z *= float(np.sqrt((y[a:b] ** 2).mean()) / max(1e-6, np.sqrt((z ** 2).mean())))   # the edit comes out louder
    ramp = np.ones(len(z), dtype=np.float32); k = min(XF, len(z) // 2)
    ramp[:k] = np.linspace(0, 1, k); ramp[-k:] = np.linspace(1, 0, k)
    y[a:b] = y[a:b] * (1 - ramp[:, None]) + z * ramp[:, None]
    print(f"win {n + 1}/{len(jobs)} [{s:.1f}-{e:.1f}] score {best[0]:.2f} | {' / '.join(tgt)} | heard: {best[2].strip()}", flush=True)
    sf.write(out, y, sr)
print("done", out, flush=True)

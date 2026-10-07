"""🎛 Mashup: the vocal of song A over the backing of song B (AutoMashup, ax-le/automashup).

The vocal is never cut or stretched, so every word survives; the BED adapts to it:
  1. Demucs stems of both songs;
  2. beat_this downbeats of both mixes;
  3. B's backing is transposed into A's key (shortest way, relative minor = same key);
  4. B's bars are laid one by one onto A's bar grid with one rubberband time map, so B's
     drums land on A's downbeats for the whole song (B bars loop if A is longer).
Mixed at the vocal/backing balance song A itself had. Measured against our old
cut-the-vocal engine (2026-08-19): WER 47% vs 74% -- see mashup-upstream-benchmark.
"""
import logging
import os
import subprocess
import tempfile

import numpy as np

from config import OUTPUT_DIR

logger = logging.getLogger("assistant.mashup_auto")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUBBERBAND = os.path.join(ROOT, "models_ext", "rubberband", "rubberband-4.0.0-gpl-executable-windows",
                          "rubberband-r3.exe")
SR = 44100
_NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def key_of(y: np.ndarray, sr: int = SR) -> tuple:
    """(tonic 0..11, is_minor) by Krumhansl-Schmuckler on the mean chroma."""
    import librosa
    c = librosa.feature.chroma_cqt(y=librosa.resample(y, orig_sr=sr, target_sr=22050), sr=22050).mean(1)
    best = max(((np.corrcoef(c, np.roll(p, t))[0, 1], t, m) for m, p in ((False, _MAJOR), (True, _MINOR))
                for t in range(12)))
    return best[1], best[2]


def semitones(src: tuple, dst: tuple) -> int:
    """Shift that takes key `src` to key `dst`; relative minor counts as its major (A min = C maj)."""
    s = (src[0] + 3) % 12 if src[1] else src[0]
    d = (dst[0] + 3) % 12 if dst[1] else dst[0]
    k = (d - s) % 12
    return k - 12 if k > 6 else k


def downbeats(path: str) -> np.ndarray:
    """Downbeat times (s) of a song, beat_this on the GPU when free."""
    import torch
    from beat_this.inference import File2Beats
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _, db = File2Beats(device=dev, dbn=False)(path)
    return np.asarray(db, dtype=float)


def match_bars(bars_a: np.ndarray, bars_b: np.ndarray) -> np.ndarray:
    """B's bar grid brought to A's bar length: a B "bar" twice as long (half-time
    detection) is split, one half as long is paired -- else every B bar would be
    squeezed or stretched 2x."""
    la, lb = np.median(np.diff(bars_a)), np.median(np.diff(bars_b))
    r = lb / la
    if r > 1.45:
        mids = (bars_b[:-1] + bars_b[1:]) / 2
        bars_b = np.sort(np.concatenate([bars_b, mids]))
    elif r < 0.7:
        bars_b = bars_b[::2]
    return bars_b


def build_bed(back_b: np.ndarray, bars_b: np.ndarray, bars_a: np.ndarray, shift: int, total: int,
              work: str) -> np.ndarray:
    """B's backing, bar by bar onto A's grid, transposed by `shift`; `total` samples long."""
    import soundfile as sf
    nb = len(bars_b) - 1
    sb = (bars_b * SR).astype(int)
    pieces, src_marks, fade = [], [0], int(0.005 * SR)
    for i in range(len(bars_a) - 1):
        j = i % nb
        p = back_b[sb[j]:sb[j + 1]].copy()
        if i and j == 0 and len(p) > 2 * fade:          # looped back to bar 0: no click at the seam
            p[:fade] *= np.linspace(0, 1, fade)[:, None] if p.ndim > 1 else np.linspace(0, 1, fade)
        pieces.append(p)
        src_marks.append(src_marks[-1] + len(p))
    src = np.concatenate(pieces)
    dst_marks = (bars_a * SR).astype(int) - int(bars_a[0] * SR)
    out_len = int(dst_marks[-1])
    sp, op, tm = os.path.join(work, "bed_in.wav"), os.path.join(work, "bed_out.wav"), os.path.join(work, "map.txt")
    sf.write(sp, src, SR)
    with open(tm, "w") as fh:  # no "0 0" line: with it rubberband lands every later mark ~110-160 ms early
        for s, d in zip(src_marks[1:], dst_marks[1:]):
            fh.write(f"{s} {d}\n")
    subprocess.run([RUBBERBAND, "-q", "--fine", "-M", tm, "-D", f"{out_len / SR:.6f}", "-p", str(shift),
                    sp, op], check=True, creationflags=_NOWIN, capture_output=True)
    bed, _ = sf.read(op, dtype="float32")
    full = np.zeros((total,) + bed.shape[1:], dtype=np.float32)
    a0 = int(bars_a[0] * SR)
    n = min(len(bed), total - a0)
    full[a0:a0 + n] = bed[:n]
    return full


def _rms(x: np.ndarray, gate: float = 0.0) -> float:
    m = np.abs(x.mean(1) if x.ndim > 1 else x)
    frames = m[: len(m) // 2048 * 2048].reshape(-1, 2048).mean(1)
    loud = frames[frames > gate] if gate else frames
    return float(np.sqrt((loud ** 2).mean())) if len(loud) else 1e-6


def mix(vox: np.ndarray, bed: np.ndarray, ratio: float, out_mp3: str) -> str:
    """Vocal over bed at `ratio` (vocal rms / backing rms of the source song); -1 dB peak."""
    import soundfile as sf
    n = min(len(vox), len(bed))
    v, b = vox[:n], bed[:n]
    v = v * (ratio * _rms(b) / max(_rms(v, gate=0.01), 1e-6))
    m = v + b
    m /= max(1.0, float(np.abs(m).max()) / 0.89)
    wav = out_mp3[:-4] + ".wav"
    sf.write(wav, m, SR)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-b:a", "192k", out_mp3], check=True,
                   creationflags=_NOWIN)
    os.unlink(wav)
    return out_mp3


def make(ctx, vocal_song: str, bed_song: str, out_mp3: str = "") -> str:
    """The vocal of `vocal_song` over the backing of `bed_song`. Returns an mp3 path."""
    import cover
    import mashup_stems
    work = tempfile.mkdtemp(prefix="mashup_")
    a = cover.to_wav(vocal_song, os.path.join(work, "a.wav"))
    b = cover.to_wav(bed_song, os.path.join(work, "b.wav"))
    sa, sb = mashup_stems.separate(a), mashup_stems.separate(b)
    vox_a, back_a, back_b = sa["vocals"], mashup_stems.backing_of(sa), mashup_stems.backing_of(sb)
    mashup_stems.release_separator()
    bars_a, bars_b = downbeats(a), downbeats(b)
    bars_b = match_bars(bars_a, bars_b)
    mono = lambda x: x.mean(1) if x.ndim > 1 else x
    shift = semitones(key_of(mono(back_b)), key_of(mono(back_a)))
    logger.info("mashup: %d/%d bars, bed shifted %+d st", len(bars_a), len(bars_b), shift)
    bed = build_bed(back_b, bars_b, bars_a, shift, len(vox_a), work)
    ratio = _rms(vox_a, gate=0.01) / max(_rms(back_a), 1e-6)
    out = out_mp3 or os.path.join(OUTPUT_DIR, f"mashup_{os.getpid()}_{np.random.randint(1e9)}.mp3")
    return mix(vox_a, bed, ratio, out)

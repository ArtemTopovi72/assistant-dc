"""🎚 Remix: one song in, plus either new words or a second song.

  words:  the song's own melody, voice and backing, but it sings YOUR text
          (the way "Rammstein sings <anything>" memes are made).
  voice:  the voice of song 1 sings the melody and words of song 2 over song 2's backing.

Nothing is generated from scratch, so it sounds like the originals by construction.
Pipeline: Demucs stems -> (words: Whisper times the original words, TTS speaks our lines into those
spans, pyworld lays the original singer's pitch on them) -> SoulX-Singer-SVC puts the singer's
timbre on it (Seed-VC if SoulX is not installed) -> mixed over the original backing.
"""
import json
import logging
import os
import re
import subprocess
import tempfile
import time

import numpy as np

import cover
import music
from config import OUTPUT_DIR, scratch_path, venv_python

logger = logging.getLogger("assistant.remix")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SX = os.path.join(ROOT, "models_ext", "soulx_singer")
SX_PY = venv_python("Z:/venvs/venv_soulx")   # on the SSD: C: is nearly full, E: is an HDD
SX_MODEL = os.path.join(SX, "pretrained_models", "SoulX-Singer", "model-svc.pt")
SEEDVC_PY = venv_python(os.path.join(ROOT, "venv_qwen"))
SR = 44100
_VOW = re.compile(r"[аеёиоуыэюяaeiouy]", re.I)


def available() -> bool:
    return os.path.isfile(SX_MODEL) and os.path.isfile(SX_PY) or os.path.isfile(SEEDVC_PY)


def _syl(s: str) -> int:
    return max(1, len(_VOW.findall(s)))


def _stems(song: str, work: str, name: str) -> tuple:
    """song -> (vocals, backing) as float arrays at SR; also kept as wavs in `work`."""
    import soundfile as sf
    import mashup_stems
    ref = cover.to_wav(song, os.path.join(work, name + "_ref.wav"))
    st = mashup_stems.separate(ref)
    vox, back = st["vocals"], mashup_stems.backing_of(st)
    sf.write(os.path.join(work, name + "_vox.wav"), vox, SR)
    return vox, back


def _singer_ref(vox: np.ndarray, work: str, name: str, sec: int = 15) -> str:
    """The loudest `sec` seconds of a vocal stem: the voice the SVC copies."""
    import soundfile as sf
    hop = SR * sec
    best = max(range(0, max(1, len(vox) - hop), SR * 5), key=lambda a: float(np.abs(vox[a:a + hop]).mean()))
    p = os.path.join(work, name + "_singer.wav")
    sf.write(p, vox[best:best + hop], SR)
    return p


def _f0_npy(wav: str, npy: str) -> None:
    import librosa
    import pyworld as pw
    import soundfile as sf
    y, _ = librosa.load(wav, sr=24000, mono=True)
    sf.write(wav, y, 24000)
    f, _ = pw.harvest(y.astype(np.float64), 24000, f0_floor=70, f0_ceil=800, frame_period=20.0)
    np.save(npy, f.astype(np.float32)[: len(y) // 480 + 1])


def sing_as(ctx, source: str, prompt: str, out: str) -> str:
    """`source` (singing) in the timbre of `prompt`. SoulX-Singer-SVC, else Seed-VC."""
    work = tempfile.mkdtemp(prefix="remix_svc_")
    if os.path.isfile(SX_MODEL) and os.path.isfile(SX_PY):
        import shutil
        shutil.copy(source, os.path.join(work, "src.wav"))
        shutil.copy(prompt, os.path.join(work, "prm.wav"))
        for k in ("src", "prm"):
            _f0_npy(os.path.join(work, k + ".wav"), os.path.join(work, k + "_f0.npy"))
        cmd = [SX_PY, os.path.join(SX, "cli", "inference_svc.py"), "--device", "cuda", "--model_path", SX_MODEL,
               "--config", os.path.join(SX, "soulxsinger", "config", "soulxsinger.yaml"),
               "--prompt_wav_path", os.path.join(work, "prm.wav"), "--target_wav_path", os.path.join(work, "src.wav"),
               "--prompt_f0_path", os.path.join(work, "prm_f0.npy"), "--target_f0_path", os.path.join(work, "src_f0.npy"),
               "--save_dir", os.path.join(work, "o"), "--auto_shift", "--pitch_shift", "0", "--n_steps", "32", "--fp16"]
        env = {**os.environ, "PYTHONPATH": SX}
        music.run_gpu_worker(ctx, SX_PY, "", {}, "SoulX-SVC", 900, cmd=cmd, env=env)
        gen = os.path.join(work, "o", "generated.wav")
        if os.path.isfile(gen):
            shutil.copy(gen, out)
            return out
        logger.warning("remix: SoulX gave no file, falling back to Seed-VC")
    job = [{"source": source, "target": prompt, "out": out}]
    jf = os.path.join(work, "jobs.json")
    with open(jf, "w", encoding="utf-8") as fh:
        json.dump(job, fh)
    music.run_gpu_worker(ctx, SEEDVC_PY, "seedvc_batch.py", {}, "Seed-VC", 900,
                         cmd=[SEEDVC_PY, os.path.join(ROOT, "scripts", "seedvc_batch.py"), jf])
    if not os.path.isfile(out):
        raise cover.CoverFailed("render")
    return out


def _mix(voice_wav: str, back: np.ndarray, out_mp3: str) -> str:
    import librosa
    import soundfile as sf
    v, sr = sf.read(voice_wav, dtype="float32")
    v = v.mean(1) if v.ndim > 1 else v
    if sr != SR:
        v = librosa.resample(v, orig_sr=sr, target_sr=SR)
    n = min(len(v), len(back))
    mix = back[:n] + v[:n]
    mix /= max(1.0, float(np.abs(mix).max()) / 0.95)
    wav = out_mp3[:-4] + ".wav"
    sf.write(wav, mix, SR)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-b:a", "192k", out_mp3], check=True,
                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    os.unlink(wav)
    return out_mp3


def _line_spans(words: list, lines: list) -> list:
    """Spread our lines over the original word timings by cumulative syllable share."""
    ow = np.cumsum([_syl(w["w"]) for w in words]).astype(float)
    ow /= ow[-1]
    nl = np.cumsum([_syl(l) for l in lines]).astype(float)
    nl /= nl[-1]
    spans = []
    for i in range(len(lines)):
        lo, hi = (nl[i - 1] if i else 0.0), nl[i]
        idx = [k for k in range(len(words)) if lo <= ((ow[k - 1] if k else 0.0) + ow[k]) / 2 < hi]
        spans.append((words[idx[0]]["s"], words[idx[-1]]["e"]) if idx else None)
    return spans


def _speak_on_melody(ctx, vox: np.ndarray, lines: list, work: str) -> str:
    """Our lines, spoken into the spans of the original words, carrying the original singer's pitch."""
    import librosa
    import pyworld as pw
    import soundfile as sf
    import audio
    vp = os.path.join(work, "orig_vox.wav")
    sf.write(vp, vox, SR)
    segs, _ = ctx.models.whisper.transcribe(vp, word_timestamps=True)
    words = [{"w": w.word.strip(), "s": w.start, "e": w.end} for sg in segs for w in (sg.words or []) if w.word.strip()]
    if len(words) < 3:
        raise cover.CoverFailed("no_words")
    canvas = np.zeros(len(vox), dtype=np.float32)
    for i, (line, sp) in enumerate(zip(lines, _line_spans(words, lines))):
        if sp is None:
            continue
        w = audio.synth_single_segment(ctx, 5000 + i, "DC", line, out_stem=os.path.join(work, f"l{i}"))
        if not w:
            continue
        y, _ = librosa.load(w, sr=SR)
        y, _ = librosa.effects.trim(y, top_db=35)
        rate = float(np.clip(len(y) / SR / max(0.3, sp[1] - sp[0]), 0.5, 2.5))
        y = librosa.effects.time_stretch(y, rate=rate)
        s0 = int(sp[0] * SR)
        canvas[s0:s0 + len(y)] += y[:max(0, len(canvas) - s0)]
    R = 22050                                       # the speech takes the original pitch contour on the same timeline
    a = librosa.resample(canvas.astype(np.float64), orig_sr=SR, target_sr=R)
    o = librosa.resample(vox.astype(np.float64), orig_sr=SR, target_sr=R)
    fa, ta = pw.harvest(a, R, f0_floor=70, f0_ceil=500, frame_period=10.0)
    fo, _ = pw.harvest(o, R, f0_floor=70, f0_ceil=700, frame_period=10.0)
    n = min(len(fa), len(fo))
    fa, fo = fa[:n], fo[:n]
    ok = np.flatnonzero(fo > 0)
    if not len(ok):
        raise cover.CoverFailed("no_words")
    f_new = np.where(fa > 0, np.interp(np.arange(n), ok, fo[ok]), 0.0)
    sp_ = pw.cheaptrick(a, fa, ta[:n], R)[:n]
    ap = pw.d4c(a, fa, ta[:n], R)[:n]
    y = librosa.resample(pw.synthesize(f_new, sp_, ap, R, 10.0).astype(np.float32), orig_sr=R, target_sr=SR)
    p = os.path.join(work, "speech_on_melody.wav")
    sf.write(p, np.pad(y, (0, max(0, len(vox) - len(y))))[:len(vox)], SR)
    return p


def _out() -> str:
    return str(OUTPUT_DIR / f"remix_{int(time.time() * 1000)}.mp3")


def remix_words(ctx, song: str, lyrics: str) -> str:
    """`song` sings `lyrics` (its own melody, voice, backing). Returns an mp3 path; raises cover.CoverFailed."""
    lines = [l.strip() for l in (lyrics or "").splitlines() if l.strip() and not re.fullmatch(r"\[.*\]", l.strip())]
    if not lines:
        raise cover.CoverFailed("no_words")
    work = tempfile.mkdtemp(prefix="remix_")
    vox, back = _stems(song, work, "a")
    speech = _speak_on_melody(ctx, vox, lines, work)
    conv = sing_as(ctx, speech, _singer_ref(vox, work, "a"), os.path.join(work, "conv.wav"))
    return _mix(conv, back, _out())


def remix_voice(ctx, voice_song: str, melody_song: str) -> str:
    """The voice of `voice_song` sings `melody_song` (its words and melody) over its backing."""
    work = tempfile.mkdtemp(prefix="remix_")
    vox1, _ = _stems(voice_song, work, "a")
    vox2, back2 = _stems(melody_song, work, "b")
    conv = sing_as(ctx, os.path.join(work, "b_vox.wav"), _singer_ref(vox1, work, "a"), os.path.join(work, "conv.wav"))
    return _mix(conv, back2, _out())

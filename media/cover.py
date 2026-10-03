"""🎤 Cover: re-sing an existing track with new (or its own) words — MuLaCover.

MuLaCover keeps the reference's melody and arrangement feel and sings the lyric
it is given; the length follows the reference. It runs in venv_mula with the
card to itself (music.run_gpu_worker). Weights are CC BY-NC: the user allowed
the button for every bot user (personal, free bot — non-commercial).
"""
from config import scratch_path, venv_python
import logging
import os
import random
import re
import subprocess
import time

import music
from config import OUTPUT_DIR

logger = logging.getLogger("assistant.cover")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MULA_PYTHON = venv_python(os.path.join(ROOT, "venv_mula"))
CKPT = os.path.join(ROOT, "models_ext", "mulacover_ckpt")
TIMEOUT = 900
DEFAULT_TAGS = "pop, clear lead vocal, full band"
_STYLE_LINE = re.compile(r"^\s*(стиль|style)\s*[:\-—]\s*(.+)$", re.I)


class CoverFailed(RuntimeError):
    pass


def available() -> bool:
    return (os.path.isfile(MULA_PYTHON)
            and os.path.isdir(os.path.join(CKPT, "MuLaCover"))
            and os.path.isdir(os.path.join(CKPT, "HeartCodec-oss")))


def split_style(text: str) -> tuple:
    """A message -> (lyrics, tags). A line «Стиль: рок, мужской вокал» sets the tags."""
    tags, lines = "", []
    for line in (text or "").splitlines():
        m = _STYLE_LINE.match(line)
        if m and not tags:
            tags = m.group(2).strip()
        else:
            lines.append(line)
    return "\n".join(lines).strip(), tags


def cover_lyrics(text: str) -> str:
    """Section tags in the model's form ([Verse], blank line between); no tags = one verse."""
    body = music.normalize_lyrics(text)
    if not body.strip():
        return ""
    if not body.lstrip().startswith("["):
        body = "[verse]\n" + body
    out = []
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            if out:
                out.append("")
            out.append("[" + s[1:-1].strip().split()[0].capitalize() + "]")
        elif s:
            out.append(s)
    return "\n".join(out) + "\n"


def to_wav(src: str, dst: str) -> str:
    """Any voice / audio / video file -> 44.1 kHz stereo wav (the codec's input)."""
    p = subprocess.run(["ffmpeg", "-y", "-i", src, "-vn", "-ac", "2", "-ar", "44100", dst],
                       capture_output=True, timeout=120,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if p.returncode != 0 or not os.path.isfile(dst) or os.path.getsize(dst) < 10_000:
        raise CoverFailed("no_audio")
    return dst


def original_words(ctx, wav: str) -> str:
    """The words the reference sings: demucs vocal stem -> GigaAM, split into lines."""
    import tempfile
    import soundfile as sf
    import audio
    import mashup_stems
    from mashup_dsp import SAMPLE_RATE
    vox = mashup_stems.separate(wav)["vocals"]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as t:
        tmp = t.name
    try:
        sf.write(tmp, vox, SAMPLE_RATE)
        text = audio._gigaam_transcribe(ctx, tmp) or ""
    finally:
        os.unlink(tmp)                  # our own temp file
    lines = [s.strip() for s in re.split(r"(?<=[.!?,])\s+", text) if s.strip()]
    return "\n".join(lines)


def make_cover(ctx, src: str, lyrics: str, tags: str = "", seed: int = 0) -> str:
    """Re-sing `src` with `lyrics`. Returns the cover's path; raises CoverFailed(reason)."""
    if not available():
        raise CoverFailed("off")
    lyr = cover_lyrics(lyrics)
    if not lyr:
        raise CoverFailed("no_words")
    stamp = int(time.time() * 1000)
    ref = to_wav(src, str(scratch_path(OUTPUT_DIR, f"_INTERMEDIATE_cover_ref_{stamp}.wav")))
    out = str(OUTPUT_DIR / f"cover_{stamp}.wav")
    job = {"ref": ref, "lyrics": lyr, "tags": tags or DEFAULT_TAGS,
           "seed": seed or random.randint(1, 2**31 - 1), "out": out}
    logger.info("cover: %s, tags %r, %d lyric lines", os.path.basename(src), job["tags"],
                lyr.count("\n"))
    t0 = time.time()
    try:
        _, tail = music.run_gpu_worker(ctx, MULA_PYTHON, "mulacover_render.py", job,
                                       "MuLaCover", TIMEOUT)
    finally:
        try:
            os.unlink(ref)              # our own intermediate
        except OSError:
            pass
    if not music._valid_audio_file(out):
        logger.error("cover: no file: %s", tail)
        raise CoverFailed("render")
    logger.info("cover: %s in %.0fs", os.path.basename(out), time.time() - t0)
    return out

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
DEFAULT_TAGS = "topic:[Love]; genre:[pop]; instrument:[Piano,acoustic guitar,drums]; mood:[warm]"
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


_FIELDS = ("topic", "genre", "instrument", "mood")


def named_style(ctx, tags: str, lyrics: str = "") -> str:
    """The model was trained on ONE line of named fields, `topic:[Longing]; genre:[country]; instrument:[Strings,acoustic
    guitar]; mood:[hopeful]`; a free list like «pop, full band» is off-distribution and is barely heard. Free text from the
    user (Russian or English) is turned into that line by the model; any failure keeps the words as the genre."""
    t = (tags or "").strip()
    if not t:
        return DEFAULT_TAGS
    if "genre:[" in t.lower():
        return t
    try:
        import lyrics_craft
        from utils import safe_json_from_llm
        raw = lyrics_craft._call(
            ctx, "style", "Turn the user's style wish for a song into English fields for a cover generator. Reply JSON "
            '{"topic": one or two words about what the song is about, "genre": one genre, "instrument": up to three instruments '
            'separated by commas, "mood": one mood word}. Keep the wish: the gender/voice of the singer goes into instrument '
            "as a word like male vocal.", f"Wish: {t}\nLyrics start: {lyrics[:200]}", temperature=0.0, max_tokens=120)
        d = safe_json_from_llm(raw, list(_FIELDS)) or {}
        if all(str(d.get(k) or "").strip() for k in _FIELDS):
            return "; ".join(f"{k}:[{str(d[k]).strip()}]" for k in _FIELDS)
    except Exception:                                            # noqa: BLE001 -- the plain wrap below still works
        logger.warning("cover style fields failed", exc_info=True)
    return f"topic:[Love]; genre:[{t}]; instrument:[Piano,drums]; mood:[warm]"


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


def vocal_stem(wav: str, dst: str) -> str:
    """Demucs vocal stem of `wav` -> `dst` (the melody is read from it, not from the full mix). "" when it fails."""
    try:
        import soundfile as sf
        import mashup_stems
        from mashup_stems import SAMPLE_RATE
        sf.write(dst, mashup_stems.separate(wav)["vocals"], SAMPLE_RATE)
        return dst
    except Exception:                                            # noqa: BLE001 -- the worker then reads the full mix
        logger.warning("cover: vocal stem failed", exc_info=True)
        return ""

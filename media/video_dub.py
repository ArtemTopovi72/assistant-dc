"""Re-voice the Russian speech of a rendered H3 clip with F5 and the stress accentor.

H3 voices its lines itself and reads Russian stress its own way. Neither the
U+0301 marks nor a CAPS stressed vowel steer it (A/B 10-06, "электроудочники,
гнилое, анализы, каменный замок": wrong in all three). Handing it a finished
speech track as <Audio 1> "fully copied" does not work either: it spoke English
gibberish over it.

So the speech is replaced after the render:
- Demucs splits the clip's sound into the voice and the rest;
- F5 clones that very voice (the clip's own speech is the reference) and says
  the same lines through the house text path, accentor included;
- every new word is time-stretched (rubberband) onto the word H3 spoke at that
  moment, so the lips still land on the words;
- the new voice goes back over the untouched background.

Any failure returns "" and the caller keeps H3's own sound.
"""
from __future__ import annotations

import difflib
import logging
import os
import re
import subprocess
import tempfile

import numpy as np

logger = logging.getLogger("assistant.video_dub")

SR = 44100
_NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_WORD = re.compile(r"[а-яёa-z0-9]+", re.I)


def _norm(w: str) -> str:
    return w.lower().replace("ё", "е")


def _run(args: list) -> bool:
    p = subprocess.run(args, capture_output=True, timeout=300, creationflags=_NOWIN)
    if p.returncode != 0:
        logger.warning("video dub: %s failed: %s", args[0], p.stderr.decode("utf-8", "replace")[-400:])
    return p.returncode == 0


def _words(ctx, wav: str) -> list:
    """[(normalized word, start, end)] -- one entry per spoken word."""
    segs, _ = ctx.models.whisper.transcribe(wav, language="ru", word_timestamps=True)
    out = []
    for sg in segs:
        for w in sg.words or ():
            for tok in _WORD.findall(w.word):
                out.append((_norm(tok), float(w.start), float(w.end)))
    return out


def anchors(old: list, new: list) -> list:
    """[(new_t, old_t)] pairs where the same word starts in both takes, in order.
    Words Whisper heard differently are simply skipped: the stretch between two
    agreeing words covers them."""
    sm = difflib.SequenceMatcher(a=[w for w, *_ in old], b=[w for w, *_ in new], autojunk=False)
    pairs = []
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            pairs.append((new[b + k][1], old[a + k][1]))
    return pairs


def _stretch(y: np.ndarray, target: int, work: str, k: int) -> np.ndarray:
    """`y` played in exactly `target` samples, pitch kept (rubberband)."""
    import soundfile as sf
    if len(y) < 256 or target < 256:
        return np.zeros(max(0, target), dtype=np.float32)
    ratio = len(y) / target                      # >1 = speed up
    if abs(ratio - 1) > 0.02:
        src, dst = os.path.join(work, f"s{k}.wav"), os.path.join(work, f"s{k}o.wav")
        sf.write(src, y, SR)
        tempo = min(4.0, max(0.25, ratio))
        if _run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-af",
                 f"rubberband=tempo={tempo:.4f}:transients=smooth:formant=preserved", dst]):
            y = sf.read(dst, dtype="float32")[0]
    out = np.zeros(target, dtype=np.float32)
    out[:min(target, len(y))] = y[:target]
    return out


def warp(new_y: np.ndarray, pairs: list, total: int, work: str, stretch=None) -> np.ndarray:
    """The new take laid onto the old timeline: the piece between two anchors is
    stretched to the old piece's length; before the first and after the last
    anchor it keeps its own speed."""
    out = np.zeros(total, dtype=np.float32)
    if not pairs:
        return out
    pts = [(int(nt * SR), int(ot * SR)) for nt, ot in pairs]
    # Lead-in before the first matched word, ending where the old one starts.
    n0, o0 = pts[0]
    lead = new_y[:n0][-o0:] if o0 > 0 else new_y[:0]
    out[o0 - len(lead):o0] = lead
    for k, ((na, oa), (nb, ob)) in enumerate(zip(pts, pts[1:])):
        if nb <= na or ob <= oa:
            continue
        seg = (stretch or _stretch)(new_y[na:nb], ob - oa, work, k)
        out[oa:oa + len(seg)] = seg[:max(0, total - oa)]
    nl, ol = pts[-1]
    tail = new_y[nl:][:max(0, total - ol)]
    out[ol:ol + len(tail)] = tail
    return out


_BRIDGES = (
    "Сегодня обычный день, на улице тихо и спокойно.",
    "Утром мы долго гуляли по парку у реки.",
    "Погода хорошая, можно немного отдохнуть после работы.",
)


def _bridge_sentence(line: str) -> str:
    """The neutral sentence sharing the fewest words with `line`."""
    said = {_norm(w) for w in _WORD.findall(line)}
    return min(_BRIDGES, key=lambda s: len(said & {_norm(w) for w in _WORD.findall(s)}))


def redub(ctx, video: str, lines: list, out_path: str = "") -> str:
    """`video` with its speech re-said by F5 (same voice, correct stress). "" on failure."""
    import soundfile as sf
    import mashup_stems
    import voice_clone
    text = " ".join(l.strip() for l in lines if l and l.strip())
    if not text or not os.path.exists(video) or getattr(getattr(ctx, "models", None), "whisper", None) is None:
        return ""
    work = tempfile.mkdtemp(prefix="vdub_")
    mix = os.path.join(work, "mix.wav")
    if not _run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-vn", "-ac", "1", "-ar", str(SR), mix]):
        return ""
    try:
        stems = mashup_stems.separate(mix)
        voice = stems["vocals"]
        rest = mashup_stems.backing_of(stems)
    except Exception:
        logger.warning("video dub: separation failed", exc_info=True)
        return ""
    vwav = os.path.join(work, "voice.wav")
    sf.write(vwav, voice, SR)
    old = _words(ctx, vwav)
    if len(old) < 2:
        logger.info("video dub: no speech heard in the clip, kept as is")
        return ""
    # The clip's own speech is the voice to clone: from the first word to the last.
    a, b = max(0.0, old[0][1] - 0.15), old[-1][2] + 0.25
    ref = os.path.join(work, "ref.wav")
    sf.write(ref, voice[int(a * SR):int(b * SR)], SR)
    ref_text = " ".join(w for w, *_ in old)
    # That sample says the very words being re-said, with H3's stress: F5 copied it
    # (live 10-06: «ва́ли» survived the redub though the text said вал+и). So the
    # clone first says a neutral sentence sharing no words with the line, and that
    # take -- same timbre, no borrowed stress -- is the voice for the line.
    bridge_text = _bridge_sentence(text)
    bridge = voice_clone.speak(ctx, ref, ref_text, bridge_text, work)
    if bridge:
        ref, ref_text = bridge, bridge_text
    new_wav = voice_clone.speak(ctx, ref, ref_text, text, work)
    if not new_wav:
        logger.warning("video dub: F5 produced nothing")
        return ""
    new_y, nsr = sf.read(new_wav, dtype="float32")
    if new_y.ndim > 1:
        new_y = new_y.mean(axis=1)
    if nsr != SR:
        import librosa
        new_y = librosa.resample(new_y, orig_sr=nsr, target_sr=SR)
    new_words = _words(ctx, new_wav)
    pairs = anchors(old, new_words)
    if len(pairs) < max(2, len(old) // 2):
        logger.warning("video dub: only %d of %d words line up, kept H3's voice", len(pairs), len(old))
        return ""
    laid = warp(new_y, pairs, len(voice), work)
    # Same loudness as the voice it replaces.
    rms = lambda x: float(np.sqrt(np.mean(np.square(x[np.abs(x) > 1e-4])))) if np.any(np.abs(x) > 1e-4) else 0.0
    if rms(laid) > 0:
        laid *= rms(voice) / rms(laid)
    track = rest[:len(laid)] + laid[:len(rest)]
    peak = float(np.max(np.abs(track))) or 1.0
    if peak > 0.98:
        track *= 0.98 / peak
    twav = os.path.join(work, "track.wav")
    sf.write(twav, track, SR)
    out_path = out_path or os.path.splitext(video)[0] + "_dub.mp4"
    if not _run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-i", twav, "-map", "0:v", "-map", "1:a",
                 "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", out_path]):
        return ""
    logger.info("video dub: %d/%d words anchored -> %s", len(pairs), len(old), os.path.basename(out_path))
    return out_path

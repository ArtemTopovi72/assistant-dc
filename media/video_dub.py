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


REF_MAX_S = 10.0       # F5 trims a longer reference itself, and then its text is wrong
CHUNK_WORDS = 24       # one F5 call: a whole 104-word monologue at once came back as noise


HUM_UNDER_WORDS = 0.1   # what is left of the old voice's residue under its own words


def _duck(n: int, words: list, floor: float = HUM_UNDER_WORDS, fade: float = 0.06) -> np.ndarray:
    """A gain of `floor` over every word span [(w, start, end)], 1 between them, `fade` s ramps."""
    g = np.ones(n, dtype=np.float32)
    for _, a, b in words:
        g[max(0, int(a * SR)):min(n, int(b * SR))] = floor
    k = max(1, int(fade * SR))
    return np.convolve(g, np.ones(k, dtype=np.float32) / k, mode="same")


def _ref_span(words: list, top: float = REF_MAX_S, avoid=frozenset()) -> tuple:
    """(first, last) index of the longest run of `words` [(w, start, end)] lasting <= `top` s
    and saying none of `avoid` (normalized words)."""
    best, i = (0, 0), 0
    for j in range(len(words)):
        # «четыре-пять» heard as «4 5»: a sample whose text F5 cannot read as said throws it off
        if _norm(words[j][0]) in avoid or any(c.isdigit() for c in words[j][0]):
            i = j + 1
            continue
        while i < j and words[j][2] - words[i][1] > top:
            i += 1
        if j - i > best[1] - best[0]:
            best = (i, j)
    return best


def _chunks(text: str, top: int = CHUNK_WORDS) -> list:
    """Whole sentences, packed up to `top` words a piece."""
    out, cur = [], ""
    for s in re.split(r"(?<=[.!?…])\s+", text.strip()):
        if cur and len((cur + " " + s).split()) > top:
            out.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip()
    return out + [cur] if cur else out


REF_MIN_WORDS = 5      # a shorter clean stretch gives F5 too little of the voice


def _ref_for(ctx, voice_clone, voice: np.ndarray, old: list, part: str, work: str) -> tuple:
    """(reference wav, its text) for saying `part`: the clip's own speech where it says none of
    `part`'s words. A sample saying the very words carries H3's stress and F5 copies it
    (live 10-06: «ва́ли»; 10-20: «рюкза́чком» though the text said рюкзачк+ом, while the same
    line from a sample of other words came out right). Failing that, a neutral bridge
    sentence said in that voice first."""
    import soundfile as sf
    said = frozenset(_norm(w) for w in _WORD.findall(part))
    i0, i1 = _ref_span(old, avoid=said)
    clean = i1 - i0 + 1 >= REF_MIN_WORDS and not said & {_norm(w) for w, *_ in old[i0:i1 + 1]}
    if not clean:
        i0, i1 = _ref_span(old)
    os.makedirs(work, exist_ok=True)
    a, b = max(0.0, old[i0][1] - 0.15), old[i1][2] + 0.25
    ref = os.path.join(work, "ref.wav")
    sf.write(ref, voice[int(a * SR):int(b * SR)], SR)
    ref_text = " ".join(w for w, *_ in old[i0:i1 + 1])
    if not clean:
        bridge_text = _bridge_sentence(part)
        bridge = voice_clone.speak(ctx, ref, ref_text, bridge_text, work)
        if bridge:
            return bridge, bridge_text
    return ref, ref_text


def _speak_chunks(ctx, voice_clone, voice: np.ndarray, old: list, text: str, work: str) -> str:
    """`text` said in the cloned voice a few sentences at a time, joined with a short gap."""
    import soundfile as sf
    pieces, sr = [], SR
    for k, part in enumerate(_chunks(text)):
        here = os.path.join(work, f"c{k}")
        ref, ref_text = _ref_for(ctx, voice_clone, voice, old, part, here)
        w = voice_clone.speak(ctx, ref, ref_text, part, here)
        if not w:
            return ""
        y, sr = sf.read(w, dtype="float32")
        pieces += [y.mean(axis=1) if y.ndim > 1 else y, np.zeros(int(0.12 * sr), dtype="float32")]
    if not pieces:
        return ""
    out = os.path.join(work, "said.wav")
    sf.write(out, np.concatenate(pieces), sr)
    return out


def redub(ctx, video: str, lines: list, out_path: str = "") -> str:
    """`video` with its speech re-said by F5 (same voice, correct stress). "" on failure."""
    text = " ".join(l.strip() for l in lines if l and l.strip())
    if not text or not os.path.exists(video) or getattr(getattr(ctx, "models", None), "whisper", None) is None:
        return ""
    with tempfile.TemporaryDirectory(prefix="vdub_") as work:
        return _redub(ctx, video, text, out_path, work)


def _redub(ctx, video: str, text: str, out_path: str, work: str) -> str:
    import soundfile as sf
    import mashup_stems
    import voice_clone
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
    # Demucs files H3's hum and score under "vocals" (tail of run 16: vocals 0.142 rms of a 0.145
    # mix, the other stems 0.004): replaced whole, the clip lost its music. Only the speech in it
    # is replaced; what the speech gate leaves goes back with the backing -- ducked under the old
    # words, where it still carries H3's own voice and was heard as an echo of the new one.
    hum = None
    try:
        import noisereduce
        speech = noisereduce.reduce_noise(y=voice, sr=SR, stationary=False).astype(np.float32)
        hum, voice = voice - speech, speech
    except Exception:
        logger.warning("video dub: no speech gate, the vocal stem is replaced whole", exc_info=True)
    vwav = os.path.join(work, "voice.wav")
    sf.write(vwav, voice, SR)
    old = _words(ctx, vwav)
    if len(old) < 2:
        logger.info("video dub: no speech heard in the clip, kept as is")
        return ""
    if hum is not None:
        rest = rest[:len(hum)] + (hum * _duck(len(hum), old))[:len(rest)]
    # The clip's own speech is the voice to clone, at most REF_MAX_S of it: from the first word to
    # the last, a 55 s clip went in, F5 cut it to 12 s and kept the whole transcript, and the voice
    # and its text no longer matched (2026-10-20). Each piece gets its own sample (_ref_for).
    new_wav = _speak_chunks(ctx, voice_clone, voice, old, text, work)
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

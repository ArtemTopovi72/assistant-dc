"""Voice cloning for the bot: any clip with speech in -> that voice reads any text.

Three steps, none of them Telegram-specific:

  prepare_reference(src)   ffmpeg to mono 24 kHz, find the densest run of
                           speech and cut 6-11 s of it on silence (F5 listens
                           to <=12 s and a longer reference bleeds its tail
                           into the reply), then transcribe exactly that cut.
                           The transcript must describe the audio F5 hears or
                           the output turns to gibberish.
  polish(ctx, text)        punctuation and capitals only. The model may not
                           change a single word: if the words differ the
                           original is kept.
  speak(ctx, ref_wav, ref_text, text, out_dir)
                           the ordinary F5 path (audio.synth_single_segment),
                           so the text gets the same treatment as every other
                           voice reply: acronyms spelled out (USB -> ю-эс-би),
                           numbers as words, stress marks, the trailing dots
                           that stop F5 eating the end of the phrase.
"""
from seams import chat_model, dep
import logging
import os
import re
import subprocess
import uuid
from typing import Optional, Tuple

logger = logging.getLogger("assistant." + __name__)

MIN_SPEECH_MS = 3000       # less than this is not enough voice to clone
TARGET_MIN_MS = 6000
TARGET_MAX_MS = 11000      # under F5's 12 s cap, with room for the edge silence
MAX_TEXT = 1500            # one voice note, not an audiobook


class CloneError(Exception):
    """Carries a reason key the caller turns into a sentence for the user."""


def _to_wav(src: str, dst: str) -> None:
    p = subprocess.run(["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "24000", dst],
                       capture_output=True, timeout=120)
    if p.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) < 1000:
        raise CloneError("no_audio")


def _voice_spans(seg):
    """[(start_ms, end_ms)] where Silero VAD hears a VOICE, or None when VAD is
    unavailable. Loudness alone picked the music of a YouTube clip and the
    sample had no words; a voice under music still counts."""
    try:
        import numpy as np, torch
        from utils import torch_threads_kept
        with torch_threads_kept():
            from silero_vad import get_speech_timestamps
        from audio import _load_silero_vad_model
        mono = seg.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        w = torch.from_numpy(np.array(mono.get_array_of_samples(), dtype=np.float32) / 32768.0)
        ts = get_speech_timestamps(w, _load_silero_vad_model(), sampling_rate=16000,
                                   min_silence_duration_ms=400)
        return [(t["start"] // 16, t["end"] // 16) for t in ts]
    except Exception:
        return None


def _word_gap(seg, lo: int, hi: int) -> Optional[int]:
    """The middle of the last quiet dip between words in [lo, hi] ms, or None. A word list
    read without a long pause (the DC sample) was cut at 11 s blind, inside «возможно»:
    the transcript had the whole word, the audio «воз», and F5 said «можно» at the start
    of every audiobook piece (2026-10-17)."""
    from pydub import silence
    win = seg[lo:hi]
    thresh = max(win.dBFS - 14, -55) if win.dBFS != float("-inf") else -55
    gaps = silence.detect_silence(win, min_silence_len=90, silence_thresh=thresh, seek_step=5)
    if not gaps:
        return None
    # a dip inside «по-видимому» is shorter than the pause after a comma
    longest = max(b - a for a, b in gaps)
    a, b = [g for g in gaps if g[1] - g[0] >= 0.6 * longest][-1]
    return lo + (a + b) // 2


def pick_speech(seg, *, min_ms: int = TARGET_MIN_MS, max_ms: int = TARGET_MAX_MS, voice_spans=None):
    """The best 6-11 s of speech: consecutive non-silent chunks, densest first.

    A voice note usually opens with a breath and trails off; a video has music
    between phrases. Chunks are the stretches between >=400 ms pauses; the
    window that packs the most speech into <= max_ms wins, and it is cut on
    those pauses so it never ends mid-word.
    """
    spans = (voice_spans or _voice_spans)(seg)
    if spans is None:
        from pydub import silence
        thresh = max(seg.dBFS - 16, -50) if seg.dBFS != float("-inf") else -50
        spans = silence.detect_nonsilent(seg, min_silence_len=400, silence_thresh=thresh, seek_step=10)
    spans = [(a, b) for a, b in spans if b - a >= 250]
    if not spans or sum(b - a for a, b in spans) < MIN_SPEECH_MS:
        raise CloneError("too_little_speech")
    best, best_speech, gapped = None, -1, set()
    for i in range(len(spans)):
        speech, j = 0, i
        while j < len(spans) and spans[j][1] - spans[i][0] <= max_ms:
            speech += spans[j][1] - spans[j][0]
            j += 1
        if j == i:                                   # one chunk longer than the cap
            a = spans[i][0]
            b = _word_gap(seg, a + min_ms, a + max_ms)
            cand, speech = (a, b or a + max_ms), (b or a + max_ms) - a
            if b:
                gapped.add(cand)
        else:
            cand = (spans[i][0], spans[j - 1][1])
        if speech > best_speech:
            best, best_speech = cand, speech
        if cand[1] - cand[0] >= min_ms and speech >= 0.85 * (cand[1] - cand[0]):
            break                                    # dense enough: stop at the first good one
    a, b = best
    # cut inside a pause between words: the usual 250 ms tail took the next word's start
    tail = 0 if best in gapped else 250
    return seg[max(0, a - 150):min(len(seg), b + tail)]


def _transcribe(ctx, wav: str, lang: str) -> str:
    import audio
    return (audio.transcribe_audio_file(ctx, wav, lang_hint=lang or "ru") or "").strip()


def prepare_reference(ctx, src: str, out_dir: str, lang: str = "ru") -> Tuple[str, str]:
    """-> (reference wav, its transcript). Raises CloneError with a reason key."""
    from pydub import AudioSegment
    os.makedirs(out_dir, exist_ok=True)
    tag = uuid.uuid4().hex[:8]
    full = os.path.join(out_dir, f"clone_src_{tag}.wav")
    ref = os.path.join(out_dir, f"clone_ref_{tag}.wav")
    _to_wav(src, full)
    try:
        seg = AudioSegment.from_file(full)
        cut = pick_speech(seg)
        cut = cut.apply_gain(-20.0 - cut.dBFS) if cut.dBFS != float("-inf") else cut
        cut.export(ref, format="wav")
    finally:
        try:
            os.unlink(full)                  # our own intermediate file
        except OSError:
            pass
    text = _transcribe(ctx, ref, lang)
    if len(re.findall(r"\w+", text)) < 3:
        raise CloneError("no_words")
    try:
        import emopack
        emopack.build_async(ref, ctx)           # emotional variants of this voice, ready a few minutes later
    except Exception:                           # noqa: BLE001 -- an extra, never a blocker
        pass
    return ref, text


def fetch_sample(url: str, out_dir: str, fetch=None, name: str = "sample") -> str:
    """THE way a video link becomes a voice sample, for every front end (clone, audiobook,
    animation voices, a link inside a request): the downloaded clip on disk, "" when it cannot
    be had. What is cut out of it is prepare_reference's job. `fetch` is the front end's own
    downloader (the bot injects its cached one); the default is the same one."""
    if fetch is None:
        from tg_links import fetch_video as fetch
    vid = fetch(url)
    if not vid.get("data"):
        return ""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}_{uuid.uuid4().hex[:8]}.mp4")
    with open(path, "wb") as fh:
        fh.write(vid["data"])
    return path


_ENCODER = []


def speaker_similarity(clip: str, ref: str, window: float = 3.0) -> Optional[float]:
    """Mean cosine similarity of the clip's speech, in `window`-second pieces, to the voice of
    `ref` (Resemblyzer on the CPU); None when it cannot be told. A render that drifted off the
    reference voice shows in the mean (live 2026-10-20: 0.46..0.95 per part at the same reference).
    A broken check returns None: it must never block a render."""
    import tempfile
    try:
        import numpy as np
        import soundfile as sf
        from resemblyzer import VoiceEncoder, preprocess_wav
        if not _ENCODER:
            _ENCODER.append(VoiceEncoder("cpu"))
        enc = _ENCODER[0]
        wav = os.path.join(tempfile.mkdtemp(prefix="spksim_"), "a.wav")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", clip, "-vn", "-ac", "1", "-ar", "16000", wav],
                       capture_output=True, check=True, timeout=60)
        y, sr = sf.read(wav)
        want = enc.embed_utterance(preprocess_wav(ref))
        sims = []
        for s in np.arange(0, max(len(y) / sr - window, 0.1), window / 2):
            seg = y[int(s * sr):int((s + window) * sr)]
            if len(seg) < sr * 2 or np.abs(seg).mean() < 0.003:
                continue
            e = enc.embed_utterance(preprocess_wav(seg, source_sr=sr))
            sims.append(float(np.dot(e, want) / (np.linalg.norm(e) * np.linalg.norm(want))))
        return float(np.mean(sims)) if sims else None
    except Exception:
        logger.warning("speaker similarity check failed", exc_info=True)
        return None


def voice_link_in(ask, text: str) -> str:
    """The video link in a request that the user names as the source of a VOICE («голос возьми
    отсюда: <ссылка>»), not as something to watch or retell; "" otherwise. `ask` is a yes_no."""
    from tg_links import video_url
    url = video_url(text)
    if not url or not ask(
            "A user asked to animate a picture so that a person in it speaks. Message: {text}. "
            "Does the user give the link as the source of that person's VOICE (to copy the voice "
            "from), not as something to watch, read or retell?", text):
        return ""
    return url


_WORDS = re.compile(r"[\wё]+", re.I)


def _words(s: str) -> list:
    return [w.lower().replace("ё", "е") for w in _WORDS.findall(s or "")]


def polish(ctx, text: str) -> str:
    """Punctuation and capitalisation only; any change of words is rejected."""
    text = re.sub(r"\s+", " ", (text or "")).strip()[:MAX_TEXT]
    if not text or ctx is None:
        return text
    try:
        import llm
        res = chat_model(ctx)(
            ctx, [{"role": "system", "content":
                   "Fix punctuation and capital letters in the user's text so it reads "
                   "naturally aloud: commas, full stops, question marks, dashes. Do NOT "
                   "add, remove, reorder, translate or respell any word. Output only the "
                   "corrected text."},
                  {"role": "user", "content": text}],
            tools=[], tool_choice="none", temperature=0.0,
            max_tokens=max(200, len(text)), prefill="<think></think>")
        out = re.sub(r"\s+", " ", ((res or {}).get("content") or "")).strip().strip('"«»')
    except Exception:
        logger.warning("voice clone: punctuation pass failed", exc_info=True)
        return text
    return out if out and _words(out) == _words(text) else text


class _CloneCtx:
    """The shared ctx, with this chat's reference on top.

    Setting custom_ref_wav on the shared ctx would hand one chat's voice to
    another chat's reply in flight; this wrapper keeps the override local.
    """

    def __init__(self, ctx, ref_wav: str, ref_text: str, cfg: float = 0.0, speed_mul: float = 0.0,
                 emotion=None, ik3=None):
        object.__setattr__(self, "_ctx", ctx)
        object.__setattr__(self, "tts_cfg", cfg)
        object.__setattr__(self, "tts_speed_mul", speed_mul)
        object.__setattr__(self, "custom_ref_wav", ref_wav)
        object.__setattr__(self, "custom_ref_text", ref_text)
        object.__setattr__(self, "delivery_emotion", emotion)
        object.__setattr__(self, "delivery_ik3", ik3)

    def __getattr__(self, name):
        return getattr(self._ctx, name)

    def __setattr__(self, name, value):
        setattr(self._ctx, name, value)


def speak(ctx, ref_wav: str, ref_text: str, text: str, out_dir: str, cfg: float = 0.0, speed_mul: float = 0.0,
          emotion: str = "", ik3: Optional[bool] = None) -> Optional[str]:
    """Synthesize `text` in the cloned voice. -> wav path or None. cfg / speed_mul (0 = default) shape the delivery;
    emotion ("" / None = decided per line, joy | anger | sad | neutral = forced) and ik3 (False = no Russian
    question intonation) override voice/delivery.py."""
    import audio
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"clone_out_{uuid.uuid4().hex[:8]}.wav")
    cc = _CloneCtx(ctx, ref_wav, ref_text, cfg, speed_mul, emotion or None, ik3)   # None = decided per line
    return dep(ctx, "synth_single_segment", audio.synth_single_segment)(cc, 0, "clone", text, out_stem=stem)

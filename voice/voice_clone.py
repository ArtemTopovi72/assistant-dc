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
  speak(ctx, ref, text)    the ordinary F5 path (audio.synth_single_segment),
                           so the text gets the same treatment as every other
                           voice reply: acronyms spelled out (USB -> ю-эс-би),
                           numbers as words, stress marks, the trailing dots
                           that stop F5 eating the end of the phrase.
"""
import logging
import os
import re
import subprocess
import tempfile
import uuid
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

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
    sample had no words (live 10-03); a voice under music still counts."""
    try:
        import numpy as np, torch
        from silero_vad import get_speech_timestamps
        from audio import _load_silero_vad_model
        mono = seg.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        w = torch.from_numpy(np.array(mono.get_array_of_samples(), dtype=np.float32) / 32768.0)
        ts = get_speech_timestamps(w, _load_silero_vad_model(), sampling_rate=16000,
                                   min_silence_duration_ms=400)
        return [(t["start"] // 16, t["end"] // 16) for t in ts]
    except Exception:
        return None


def pick_speech(seg, *, min_ms: int = TARGET_MIN_MS, max_ms: int = TARGET_MAX_MS):
    """The best 6-11 s of speech: consecutive non-silent chunks, densest first.

    A voice note usually opens with a breath and trails off; a video has music
    between phrases. Chunks are the stretches between >=400 ms pauses; the
    window that packs the most speech into <= max_ms wins, and it is cut on
    those pauses so it never ends mid-word.
    """
    spans = _voice_spans(seg)
    if spans is None:
        from pydub import silence
        thresh = max(seg.dBFS - 16, -50) if seg.dBFS != float("-inf") else -50
        spans = silence.detect_nonsilent(seg, min_silence_len=400, silence_thresh=thresh, seek_step=10)
    spans = [(a, b) for a, b in spans if b - a >= 250]
    if not spans or sum(b - a for a, b in spans) < MIN_SPEECH_MS:
        raise CloneError("too_little_speech")
    best, best_speech = None, -1
    for i in range(len(spans)):
        speech, j = 0, i
        while j < len(spans) and spans[j][1] - spans[i][0] <= max_ms:
            speech += spans[j][1] - spans[j][0]
            j += 1
        if j == i:                                   # one chunk longer than the cap
            a = spans[i][0]
            cand, speech = (a, a + max_ms), max_ms
        else:
            cand = (spans[i][0], spans[j - 1][1])
        if speech > best_speech:
            best, best_speech = cand, speech
        if cand[1] - cand[0] >= min_ms and speech >= 0.85 * (cand[1] - cand[0]):
            break                                    # dense enough: stop at the first good one
    a, b = best
    return seg[max(0, a - 150):min(len(seg), b + 250)]


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
        res = llm.send_to_lm_studio(
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
    return audio.synth_single_segment(cc, 0, "clone", text, out_stem=stem)

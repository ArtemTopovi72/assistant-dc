"""Emotion by prosody: PSOLA (Praat) reshapes pitch contour, tempo and loudness of an already synthesised wav.
Words and voice colour stay intact (formants are not touched), so unlike activation steering it cannot break the text.

    shape("in.wav", "joy", "out.wav")        # joy | anger | sad | neutral | question

Each preset (semitones): shift of the median, range multiplier around the median, declination within a phrase
(start -> end), rise over the last 30% of the phrase, tempo multiplier (>1 faster), gain in dB.
"""
import numpy as np
import parselmouth
from parselmouth.praat import call

PRESETS = {
    #            shift range start  end  rise tempo gain
    "joy":      (+3.0, 1.6, +1.0, +0.5, 0.0, 1.05, +2.0),   # high pitch, wide range, lively
    "anger":    (+1.0, 1.35, +1.5, -0.5, 0.0, 1.05, +9.0),  # very loud, sometimes high
    "sad":      (-3.5, 0.75, +2.5, -5.0, 0.0, 0.88, -3.0),  # low, falls from high to low, slower, quiet
    "neutral":  (-0.5, 0.35, 0.0, -0.5, 0.0, 0.95, -1.0),   # nearly flat, distant, smooth
    "question": (0.0, 1.25, 0.0, -0.5, +6.0, 1.0, 0.0),     # rises at the end
}


def _phrases(times, gap=0.30):
    """Index ranges of voiced frames split by pauses longer than `gap` s."""
    out, s = [], 0
    for i in range(1, len(times)):
        if times[i] - times[i - 1] > gap:
            out.append((s, i)); s = i
    out.append((s, len(times)))
    return out


def shape(src: str, emotion: str, dst: str, strength: float = 1.0) -> str:
    shift, rng, p0, p1, rise, tempo, gain = PRESETS[emotion]
    shift, p0, p1, rise, gain = (v * strength for v in (shift, p0, p1, rise, gain))
    rng = 1 + (rng - 1) * strength
    tempo = 1 + (tempo - 1) * strength
    snd = parselmouth.Sound(src)
    man = call(snd, "To Manipulation", 0.01, 70, 400)
    pt = call(man, "Extract pitch tier")
    n = call(pt, "Get number of points")
    if n < 5:
        snd.save(dst, "WAV"); return dst
    t = np.array([call(pt, "Get time from index", i + 1) for i in range(n)])
    f = np.array([call(pt, "Get value at index", i + 1) for i in range(n)])
    med = np.median(f)
    st = 12 * np.log2(f / med)
    new = np.empty_like(st)
    for a, b in _phrases(t):
        k = np.linspace(0, 1, b - a)
        decl = p0 + (p1 - p0) * k                                # hi -> lo (or the reverse) across the phrase
        tail = rise * np.clip((k - 0.7) / 0.3, 0, 1) ** 1.5      # final rise
        new[a:b] = st[a:b] * rng + shift + decl + tail
    f2 = med * 2 ** (new / 12)
    call(pt, "Remove points between", 0, t[-1] + 1)
    for ti, fi in zip(t, f2):
        call(pt, "Add point", float(ti), float(fi))
    call([man, pt], "Replace pitch tier")
    if tempo != 1:
        dur = call(man, "Extract duration tier")
        call(dur, "Add point", 0.0, 1 / tempo)
        call([man, dur], "Replace duration tier")
    out = call(man, "Get resynthesis (overlap-add)")
    y = out.values[0]
    y = y * 10 ** (gain / 20)
    peak = np.abs(y).max()
    if peak > 0.95:                                              # soft limiter instead of clipping
        y = np.tanh(y / peak * 1.5) / np.tanh(1.5) * 0.95
    parselmouth.Sound(y, out.sampling_frequency).save(dst, "WAV")
    return dst

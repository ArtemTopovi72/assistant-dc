"""Signal-level primitives for the mashup engine: time-stretch/pitch-shift,
equal-power crossfades, level matching, and the tempo/key arithmetic that
decides how far a vocal is moved.

The leaf of the mashup layering (dsp <- analysis <- arrangement <- mashup):
nothing here knows what a stem, a section or a song is. Split out of
mashup.py unchanged.
"""
import logging
import os

import numpy as np

logger = logging.getLogger("assistant.mashup")


SAMPLE_RATE = 44100          # htdemucs' native rate; everything resamples to it


# Octave folding cannot do better than sqrt(2): two tempos exactly a tritone
# of tempo apart (say 128 over 90) are equally far from 1.0 whichever octave
# you pick. That is the real bound on how far a vocal ever gets stretched, and
# it is worth stating as a constant because it is the worst case a listener
# can hear -- typical pairs land far closer to 1.0.
_MAX_STRETCH = 2.0 ** 0.5


def _rubberband(y: np.ndarray, sr: int, tempo: float = 1.0,
                semitones: float = 0.0) -> np.ndarray:
    """Time-stretch / pitch-shift through ffmpeg's rubberband filter.

    librosa's phase vocoder smears transients, which on a vocal is exactly the
    audible damage a mashup cannot afford. ffmpeg here is built with
    librubberband, which is what auto-mashup-mix (amamenko) uses for the same
    job, so the better algorithm costs nothing but a subprocess hop. Falls back
    to librosa if the filter or the binary is missing, because the mashup
    working badly beats it not working at all.
    """
    import librosa
    if abs(tempo - 1.0) < 1e-3 and abs(semitones) < 1e-3:
        return y
    import shutil
    import subprocess
    import tempfile as _tf
    import soundfile as sf
    if shutil.which("ffmpeg"):
        src = dst = ""
        try:
            fd, src = _tf.mkstemp(suffix=".wav"); os.close(fd)
            fd, dst = _tf.mkstemp(suffix=".wav"); os.close(fd)
            sf.write(src, y, sr)
            filt = "rubberband=tempo=%.6f:pitch=%.6f" % (
                max(tempo, 1e-3), 2.0 ** (semitones / 12.0))
            r = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                 "-i", src, "-filter:a", filt, dst],
                capture_output=True, timeout=300)
            if r.returncode == 0:
                out, _sr = sf.read(dst, dtype="float32")
                if out.size:
                    return (out.mean(axis=1) if out.ndim > 1 else out).astype(np.float32)
            logger.warning("rubberband failed (rc=%s), falling back to the phase "
                           "vocoder: %s", r.returncode, r.stderr[-200:])
        except Exception as exc:
            logger.warning("rubberband unavailable (%s), using the phase vocoder", exc)
        finally:
            for p in (src, dst):
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
    if abs(tempo - 1.0) >= 1e-3:
        y = librosa.effects.time_stretch(y, rate=tempo)
    if abs(semitones) >= 1e-3:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)
    return np.asarray(y, dtype=np.float32)


def split_tempo(ratio: float, split: float = 0.5) -> tuple:
    """(vocal_rate, bed_rate) -- share one tempo adjustment between BOTH tracks.

    Straight from the OTAC idea in Ishizaki et al., ISMIR 2009 (by way of
    poke19962008/Mash-Up): stretching one song the whole way to meet the other
    concentrates every artefact in that one song, and the vocal is the worst
    place to put them. Meeting in the middle halves the damage on each side --
    a 7.2% gap becomes 3.5% each way, which is under the threshold where a
    stretch starts to warble.

    `split` is how much of the (logarithmic) adjustment the VOCAL carries:
    1.0 reproduces the old all-on-the-vocal behaviour, 0.0 puts it all on the
    backing, 0.5 shares it evenly.
    """
    if ratio <= 0 or not np.isfinite(ratio):
        return 1.0, 1.0
    s = min(max(float(split), 0.0), 1.0)
    return float(ratio ** s), float(ratio ** (s - 1.0))


def _equal_power_fade(n: int) -> tuple:
    """(fade_in, fade_out) curves that sum to constant POWER, not amplitude.

    A linear crossfade dips in the middle -- two uncorrelated signals at half
    amplitude each carry a quarter of the power -- and that dip is audible on
    every section join. sqrt curves hold the level through the join.
    """
    t = np.linspace(0.0, 1.0, max(n, 1), dtype=np.float32)
    return np.sqrt(t), np.sqrt(1.0 - t)


def _place(dst: np.ndarray, src: np.ndarray, at: int, fade: int) -> None:
    """Write `src` into `dst` at sample `at`, crossfading into what is there."""
    if src.size == 0 or at >= len(dst):
        return
    n = min(len(src), len(dst) - at)
    if n <= 0:
        return
    seg = src[:n].copy()
    f = int(min(fade, n // 2))
    if f > 0:
        fin, _fout = _equal_power_fade(f)
        seg[:f] *= fin              # rise in
        seg[-f:] *= fin[::-1]       # the same curve reversed = fall out
    dst[at:at + n] += seg


def _fit_span(src: np.ndarray, want: int, max_stretch: float = 2.0) -> np.ndarray:
    """Halve or double `src` until stretching it onto `want` stays sane.

    Halving and doubling are the only edits that keep a bar a bar: take the
    first half of an eight-bar section and you have four bars, still starting
    on a downbeat. Anything else would need the stretch to cover the whole
    difference, and a 3x stretch of a drum kit sounds like a tape fault.
    """
    if src.size == 0 or want <= 0:
        return src
    for _ in range(6):
        if len(src) <= want * max_stretch:
            break
        half = len(src) // 2
        if half < want // 2:
            break
        src = src[:half]
    for _ in range(6):
        if len(src) * max_stretch >= want:
            break
        src = np.tile(src, 2)
    return src


def stretch_ratio(vocal_bpm: float, instr_bpm: float) -> float:
    """How much to speed the vocal up (>1) or slow it down (<1).

    Folded by octaves first: half- and double-time are the same groove, so a
    100 BPM vocal over a 190 BPM bed is a 0.95x nudge, not a 1.9x smear.
    """
    if vocal_bpm <= 0 or instr_bpm <= 0:
        return 1.0
    ratio = instr_bpm / vocal_bpm
    if not np.isfinite(ratio) or ratio <= 0:
        return 1.0
    # Pick the octave NEAREST 1.0 in one step. This used to be two sequential
    # while-loops, halving then doubling, and the second could push the ratio
    # straight back out past the bound the first had just enforced: a 128 BPM
    # vocal over a 90 BPM bed folded to 0.70, below the floor, then doubled to
    # 1.41 -- a bigger stretch than the unfolded value would have been.
    ratio /= 2.0 ** round(np.log2(ratio))
    return float(ratio)


def semitone_shift(vocal_key: int, instr_key: int) -> int:
    """Semitones to move the vocal onto the bed's key, in [-6, +5].

    Wrapped to the nearer direction so a C vocal over a B bed goes DOWN one
    semitone rather than up eleven, which would leave it an octave adrift.
    A tritone is equidistant either way and is sent DOWN, because shifting a
    voice up six semitones is the more audible damage of the two.
    """
    if vocal_key < 0 or instr_key < 0:
        return 0
    diff = (instr_key - vocal_key) % 12
    return diff - 12 if diff >= 6 else diff


# -- the mashup ------------------------------------------------------------
def _normalize(y: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = float(np.max(np.abs(y))) if y.size else 0.0
    return (y * (peak / m)).astype(np.float32) if m > 1e-6 else y.astype(np.float32)


def _rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0


def _balance(vocal: np.ndarray, bed: np.ndarray, bed_gain_db: float) -> tuple:
    """Scale the bed to sit `bed_gain_db` under the vocal, by RMS not by peak.

    Peak-matching the two was measurably wrong: a vocal stem is spiky (high
    peaks, low average) while a full band mix is dense, so normalising both to
    the same PEAK left the bed louder to the ear than the voice -- the
    correlation of the mix against the bed came out higher than against the
    vocal, which is exactly backwards for a mashup. Matching average level
    instead makes bed_gain_db mean what it says: how far under the voice the
    backing sits.
    """
    rv, rb = _rms(vocal), _rms(bed)
    if rv <= 1e-6 or rb <= 1e-6:
        return vocal, bed
    target = rv * (10.0 ** (bed_gain_db / 20.0))
    return vocal, (bed * (target / rb)).astype(np.float32)


def _tile_to(y: np.ndarray, n: int) -> np.ndarray:
    """Loop `y` until it covers n samples. A 30s bed under a 90s vocal is
    better repeated than cut short, and these are short clips."""
    if y.size == 0:
        return np.zeros(n, dtype=np.float32)
    reps = int(np.ceil(n / len(y)))
    return np.tile(y, reps)[:n].astype(np.float32)

"""What the mashup engine hears in a track: tempo, key, section boundaries,
the silences a vocal leaves, and how well two songs go together.

Read-only analysis — every function takes audio (or the results of earlier
analysis) and returns numbers. Split out of mashup.py unchanged.
"""
import logging

import numpy as np

from mashup_dsp import SAMPLE_RATE, semitone_shift, stretch_ratio

logger = logging.getLogger("assistant.mashup")


# Krumhansl-Schmuckler key profiles, used to pick a key from a chroma vector.
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])


_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


# -- analysis --------------------------------------------------------------
def estimate_tempo(y: np.ndarray, sr: int = SAMPLE_RATE) -> tuple:
    """(bpm, beat_times). bpm is 0.0 when nothing beat-like was found."""
    import librosa
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
        bpm = float(np.atleast_1d(tempo)[0])
        return (bpm if np.isfinite(bpm) and bpm > 0 else 0.0), np.asarray(beats)
    except Exception as exc:
        logger.warning("tempo estimation failed: %s", exc)
        return 0.0, np.array([])


def estimate_key(y: np.ndarray, sr: int = SAMPLE_RATE) -> tuple:
    """(pitch_class 0-11, 'major'|'minor', name) by chroma correlation.

    Returns (-1, "", "") when the audio has no usable pitch content -- the
    normal answer for speech and for a drums-only track. Callers must read
    that as "do not pitch-shift", never as "shift to C".
    """
    import librosa
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        vec = chroma.mean(axis=1)
        if not np.isfinite(vec).all() or vec.sum() <= 0:
            return -1, "", ""
        vec = vec / vec.sum()
        best, best_score = (-1, "", ""), -2.0
        for shift in range(12):
            rolled = np.roll(vec, -shift)
            for profile, mode in ((_MAJOR_PROFILE, "major"),
                                  (_MINOR_PROFILE, "minor")):
                score = float(np.corrcoef(rolled, profile)[0, 1])
                if np.isfinite(score) and score > best_score:
                    best_score, best = score, (shift, mode, _NOTE_NAMES[shift])
        return best
    except Exception as exc:
        logger.warning("key estimation failed: %s", exc)
        return -1, "", ""


def _has_energy(y: np.ndarray, floor: float = 1e-4) -> bool:
    return bool(y.size) and bool(np.any(np.abs(y) > floor))


def best_offset(vocal: np.ndarray, bed: np.ndarray, sr: int = SAMPLE_RATE,
                max_shift_s: float = 12.0) -> int:
    """Samples to delay `vocal` by so its groove sits on the bed's.

    Aligning only the FIRST detected beat -- which is what this did -- puts one
    instant in the right place and lets everything after it drift: over a four
    minute track a 1% tempo error is 2.4 seconds adrift by the end, and the
    first beat of a vocal stem is whichever breath the tracker happened to
    like. Cross-correlating the two onset envelopes instead picks the offset
    that lines up the most onsets across the WHOLE track, which is the thing a
    listener actually hears.
    """
    import librosa
    try:
        hop = 512
        ov = librosa.onset.onset_strength(y=vocal, sr=sr, hop_length=hop)
        ob = librosa.onset.onset_strength(y=bed, sr=sr, hop_length=hop)
        if ov.size < 8 or ob.size < 8:
            return 0
        ov = (ov - ov.mean()) / (ov.std() or 1.0)
        ob = (ob - ob.mean()) / (ob.std() or 1.0)
        n = min(len(ov), len(ob))
        ov, ob = ov[:n], ob[:n]
        max_lag = int(max_shift_s * sr / hop)
        corr = np.correlate(ob, ov, mode="full")
        centre = len(ov) - 1
        lo = max(centre, centre - max_lag)          # only DELAY the vocal:
        hi = min(len(corr), centre + max_lag + 1)   # negative lag would cut it
        if hi <= lo:
            return 0
        lag = int(np.argmax(corr[lo:hi]) + (lo - centre))
        return max(0, lag * hop)
    except Exception as exc:
        logger.warning("onset alignment failed, falling back to no offset: %s", exc)
        return 0


# -- song structure --------------------------------------------------------
def detect_sections(y: np.ndarray, sr: int = SAMPLE_RATE,
                    target_len_s: float = 16.0) -> list:
    """[{"start", "end", "label"}, ...] -- where the song changes.

    auto-mashup-mix gets this from Spotify's audio-analysis endpoint and uses
    it to place each vocal section against the matching instrumental one. We
    have no such metadata, so the structure is found locally: beat-synchronous
    chroma, agglomerative boundaries, then the sections clustered so repeats of
    the same part share a `label` (verse / chorus, without naming them).

    Boundaries are on BEATS, always. A section join that lands between beats
    clicks, and the whole point of arranging by section is that the joins are
    inaudible.
    """
    import librosa
    try:
        if y.size < sr * 4:
            return []
        _tempo, beat_f = librosa.beat.beat_track(y=y, sr=sr)
        if np.size(beat_f) < 8:
            return []
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        sync = librosa.util.sync(chroma, beat_f, aggregate=np.median)
        beat_t = librosa.frames_to_time(beat_f, sr=sr)
        dur = len(y) / float(sr)
        k = int(max(2, min(round(dur / max(target_len_s, 4.0)), sync.shape[1] // 4)))
        if k < 2:
            return []
        bounds = librosa.segment.agglomerative(sync, k)
        bounds = sorted(set([0] + list(bounds) + [sync.shape[1]]))
        secs = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a < 2:
                continue
            start = float(beat_t[min(a, len(beat_t) - 1)])
            end = float(beat_t[min(b, len(beat_t) - 1)]) if b < len(beat_t) else dur
            if end - start < 2.0:
                continue
            secs.append({"start": start, "end": end,
                         "feat": sync[:, a:b].mean(axis=1)})
        if len(secs) < 2:
            return []
        # Label repeats of the same part alike, so a chorus can be recognised
        # as the same material wherever it recurs.
        try:
            from sklearn.cluster import AgglomerativeClustering
            feats = np.vstack([s["feat"] for s in secs])
            n = int(min(4, max(2, len(secs) // 2)))
            labels = AgglomerativeClustering(n_clusters=n).fit_predict(feats)
        except Exception:
            labels = list(range(len(secs)))
        for s, lab in zip(secs, labels):
            s["label"] = int(lab)
            s.pop("feat", None)
        # Clamped AFTER labelling, so a merged section keeps the label of the
        # part it started as rather than losing it to the merge.
        return _even_out_sections(secs, beat_t, target_len_s)
    except Exception as exc:
        logger.warning("section detection failed, falling back to one block: %s", exc)
        return []


def _even_out_sections(secs: list, beat_t: np.ndarray,
                       target_len_s: float) -> list:
    """Merge tiny sections and split enormous ones, on beat boundaries.

    Raw agglomerative boundaries are wildly uneven: on a real 259s track they
    came back as an 89-second block followed by four 5-second slivers. Both
    ends are unusable here -- a 5s section is not a musical part, and an 89s
    one gets filled by looping a short vocal chunk a dozen times, which sounds
    worse than the drift the arrangement exists to fix. Sections are clamped to
    roughly half-to-double the target length, with every boundary still landing
    on a beat.
    """
    if not secs:
        return secs
    lo, hi = max(target_len_s * 0.5, 4.0), target_len_s * 2.0

    merged = []
    for s in secs:
        if merged and (s["end"] - s["start"]) < lo:
            merged[-1]["end"] = s["end"]        # absorb the sliver
        else:
            merged.append(dict(s))
    # A short trailing section has nothing after it to merge into; fold it back.
    while len(merged) > 1 and (merged[-1]["end"] - merged[-1]["start"]) < lo:
        merged[-2]["end"] = merged[-1]["end"]
        merged.pop()

    out = []
    for s in merged:
        span = s["end"] - s["start"]
        if span <= hi:
            out.append(s)
            continue
        parts = int(np.ceil(span / target_len_s))
        edges = np.linspace(s["start"], s["end"], parts + 1)
        for a, b in zip(edges[:-1], edges[1:]):
            # Snap to the nearest beat so the join still lands on the grid.
            if beat_t.size:
                a = float(beat_t[int(np.argmin(np.abs(beat_t - a)))])
                b = float(beat_t[int(np.argmin(np.abs(beat_t - b)))])
            if b - a >= lo * 0.5:
                out.append({"start": float(a), "end": float(b),
                            "label": s.get("label", 0)})
    return out


def vocal_gaps(vocal: np.ndarray, sr: int = SAMPLE_RATE,
               floor_db: float = -42.0, min_gap_s: float = 0.18) -> list:
    """[(start, end), ...] where nobody is singing.

    THE thing that shredded the lyrics. Section boundaries were snapped to
    BEATS, and a beat lands wherever it lands -- straight through the middle of
    a word as often as not. Transcribing the arranged vocal with the project's
    own Whisper measured a 100.7% word error rate against the untouched stem,
    with words guillotined mid-syllable ("Ой, хорош.") and fragments Whisper
    could only hallucinate at.

    A cut inside a silence is inaudible; a cut inside a vowel is a splice. So
    the arrangement asks this where the singer is not singing, and moves its
    boundaries there.
    """
    import librosa
    if vocal.size < sr // 2:
        return []
    hop = 512
    rms = librosa.feature.rms(y=vocal, hop_length=hop)[0]
    db = 20 * np.log10(np.maximum(rms, 1e-9))
    quiet = db < floor_db
    gaps, run = [], None
    for i, q in enumerate(quiet):
        if q and run is None:
            run = i
        elif not q and run is not None:
            a, b = run * hop / sr, i * hop / sr
            if b - a >= min_gap_s:
                gaps.append((a, b))
            run = None
    if run is not None:
        a, b = run * hop / sr, len(quiet) * hop / sr
        if b - a >= min_gap_s:
            gaps.append((a, b))
    return gaps


def snap_to_silence(t: float, gaps: list, max_move_s: float = 2.5) -> float:
    """Move a cut point into the nearest silence, if one is close enough.

    Refuses to move further than `max_move_s`: dragging a boundary halfway
    across a section to find silence would wreck the timing the section
    alignment exists to provide. When nothing is near, the caller keeps the
    musical boundary and accepts the splice.
    """
    if not gaps:
        return t
    best, best_d = t, max_move_s
    for a, b in gaps:
        # Aim at the MIDDLE of the gap: furthest from either neighbouring word.
        mid = 0.5 * (a + b)
        d = abs(mid - t)
        if d < best_d:
            best, best_d = mid, d
    return float(best)


def compatibility(vocal_bpm: float, instr_bpm: float,
                  vocal_key: int, instr_key: int,
                  max_semitones: int = 2,
                  vocal_mode: str = "", instr_mode: str = "",
                  max_tempo_pct: float = 5.0) -> tuple:
    """(ok, reasons) -- whether these two tracks can mash up at all.

    This is the thing the pipeline was missing, and no amount of better
    separation or better alignment substitutes for it. A real pair measured
    139.7 BPM over 74.9 (not an octave apart: 74.9x2 is 149.8, so locking them
    costs a 7.2% stretch) in A major over C major (three semitones, fixable
    only by a shift that artefacts the whole vocal). The output was correctly
    assembled and still unlistenable, because those two songs do not go
    together. Saying so beforehand is worth more than twenty seconds of GPU
    spent proving it.
    """
    reasons = []
    if vocal_bpm > 0 and instr_bpm > 0:
        stretch = stretch_ratio(vocal_bpm, instr_bpm)
        pct = abs(stretch - 1.0) * 100.0
        # 5% is auto-mashup-mix's published filter, and it picks pairs that
        # demonstrably work at scale -- a better number than the 6% I guessed.
        if pct > max_tempo_pct:
            reasons.append(
                "tempos are %.0f and %.0f BPM, which need a %.0f%% stretch to "
                "lock (over %.0f%% it audibly warbles)"
                % (vocal_bpm, instr_bpm, pct, max_tempo_pct))
    # Mode was missing entirely, and it is the cheapest filter there is: a
    # major vocal over a minor backing clashes even when both are nominally in
    # the same key, because the thirds fight. auto-mashup-mix screens on mode
    # FIRST, before key or tempo.
    if vocal_mode and instr_mode and vocal_mode != instr_mode:
        reasons.append("one is %s and the other %s, so the thirds clash "
                       "whatever the key" % (vocal_mode, instr_mode))
    if vocal_key >= 0 and instr_key >= 0:
        semis = semitone_shift(vocal_key, instr_key)
        if abs(semis) > max_semitones:
            reasons.append(
                "keys are %s and %s, %d semitones apart (more than %d cannot be "
                "shifted without artefacting)"
                % (_NOTE_NAMES[vocal_key], _NOTE_NAMES[instr_key],
                   abs(semis), max_semitones))
    return (not reasons), reasons

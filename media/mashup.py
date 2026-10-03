"""Mashup engine: the vocal of one track over the instrumental of another.

Kept free of any Telegram/GUI import, like weather.py and music.py, so both
surfaces drive the same code and it can be unit tested headlessly.

The pipeline is the classic mashup recipe, in the order the steps have to
happen:

  1. SEPARATE   both inputs into stems (Demucs htdemucs). The vocal donor
                gives up its `vocals`; the instrumental donor gives up
                everything BUT its vocals.
  2. TEMPO      time-stretch the vocal onto the instrumental's BPM.
  3. KEY        pitch-shift the vocal onto the instrumental's key.
  4. ALIGN      slide the vocal so its first strong beat lands on one of the
                instrumental's.
  5. MIX        level-match, duck the bed under the vocal, sum, normalize.

A voice note takes a shorter path: speech has no musical key, and stretching
it to a song's tempo makes it sound seasick, so steps 2-3 are skipped for it
(see `vocal_is_speech`). It still gets aligned and ducked.

Nothing here raises a bare exception for a bad input: callers get
MashupUnavailable carrying a sentence a user can act on, because both
surfaces show it as a chat/status message rather than a traceback.
"""
import logging
import os
import tempfile
import time

import numpy as np

logger = logging.getLogger("assistant.mashup")


# The engine was one 1109-line module; it is now four layers plus this one.
# Everything below is re-exported so `mashup.X` keeps working for every caller
# and every test — the split moved code, not the public surface.
#
# These are also the seams the suites patch (`mashup.separate = fake`): because
# make_mashup lives here and resolves them through THIS module's globals, a
# rebind still takes effect. Moving make_mashup out would silently break that.
from mashup_dsp import (                       # noqa: F401  (re-export)
    SAMPLE_RATE, _MAX_STRETCH, _rubberband, split_tempo, _equal_power_fade,
    _place, _fit_span, stretch_ratio, semitone_shift, _normalize, _rms,
    _balance, _tile_to,
)
from mashup_analysis import (                  # noqa: F401  (re-export)
    _MAJOR_PROFILE, _MINOR_PROFILE, _NOTE_NAMES, estimate_tempo, estimate_key,
    _has_energy, best_offset, detect_sections, _even_out_sections, vocal_gaps,
    snap_to_silence, compatibility,
)
from mashup_stems import (                     # noqa: F401  (re-export)
    STEM_MODEL, _BACKING_STEMS, MashupUnavailable, _get_separator,
    release_separator, separate, backing_of, percussive_of,
)
from mashup_arrange import (                   # noqa: F401  (re-export)
    arrange_to_sections, _pick_bed_section, arrange_bed_to_vocal,
)


# How far the backing may be pitch-shifted in bed_to_vocal. A voice shows
# formant damage past two semitones; a mixed instrumental stays clean to
# a tritone, and past that the nearer direction flips anyway.
BED_MAX_SEMITONES = 6


def make_mashup(vocal_path: str, instr_path: str, out_path: str, *,
                vocal_is_speech: bool = False, device: str = "",
                match_tempo: bool = True, match_key: bool = True,
                max_semitones: int = 2, tempo_split: float = 0.5,
                section_align: bool = True, arrange: str = "bed_to_vocal",
                bed_gain_db: float = -3.5,
                refuse_incompatible: bool = False, progress=None) -> dict:
    """Write a mashup to out_path and return a report of what it did.

    `vocal_path` supplies the voice, `instr_path` the backing track. The report
    ({"bpm_vocal", "bpm_instr", "stretch", "semitones", "key_vocal",
    "key_instr", "seconds", "took"}) is what the surfaces show: a mashup that
    declined to match tempo looks identical to one that could not, and these
    numbers are the only way for a listener to tell which happened.

    `arrange` picks which side bends. "vocal_to_bed" cuts vocal sections onto
    the instrumental's structure; "bed_to_vocal" leaves the vocal completely
    alone -- no cut, no stretch, no pitch shift -- and rebuilds the backing to
    follow it, which is what a benchmark against ax-le/automashup said keeps
    the lyric intact. Both are kept so the two can go on being compared.
    "bed_to_vocal" is the default: the August benchmark put it at 80% of the
    lyric kept against 46% for "vocal_to_bed", and a year of "vocal_to_bed"
    by default is what made the feature sound broken (2026-09-14).

    `progress(stage)` is called with a short key before each slow step so a
    surface can say what it is doing. It is optional and never required.
    """
    import librosa
    import soundfile as sf

    def _say(stage):
        if progress:
            try:
                progress(stage)
            except Exception:
                pass

    t_start = time.time()
    _say("separating")
    # Speech is already a bare voice: running it through the separator would
    # only find (and throw away) a "backing track" that is really room noise.
    vocal_drums = np.array([], dtype=np.float32)
    if vocal_is_speech:
        y, _sr = librosa.load(vocal_path, sr=SAMPLE_RATE, mono=True)
        vocal = np.asarray(y, dtype=np.float32)
    else:
        vocal_stems = separate(vocal_path, device)
        vocal = vocal_stems.get("vocals")
        # KEPT, not discarded: the donor's own drums are what its tempo gets
        # measured on. Measuring it on the vocal stem was the root error --
        # see percussive_of().
        vocal_drums = percussive_of(vocal_stems)
        if vocal is None or not np.any(np.abs(vocal) > 1e-4):
            raise MashupUnavailable(
                "I could not find any singing in the first track -- send a song "
                "with vocals, or a voice message.")
    instr_stems = separate(instr_path, device)
    bed = backing_of(instr_stems)
    bed_drums = percussive_of(instr_stems)
    if not np.any(np.abs(bed) > 1e-4):
        raise MashupUnavailable(
            "The second track has no instrumental left once the vocals are "
            "removed -- send something with music behind it.")

    _say("analysing")
    # Tempo off the DRUMS of each side, falling back to the full signal only
    # when a side has none (a drumless ballad, or speech).
    # ...but only where there ARE drums. An a cappella or a drumless ballad has
    # a silent percussive stem, and measuring tempo on silence returns 0 BPM --
    # which would then disable both the stretch and the alignment.
    bpm_v, beats_v = estimate_tempo(vocal_drums if _has_energy(vocal_drums) else vocal)
    bpm_i, beats_i = estimate_tempo(bed_drums if _has_energy(bed_drums) else bed)
    # Key still comes from the pitched material, not the drums -- chroma of a
    # drum track is noise, and would send the pitch shift somewhere arbitrary.
    key_v = estimate_key(vocal) if not vocal_is_speech else (-1, "", "")
    key_i = estimate_key(bed)

    # Speech keeps its own tempo and pitch: stretching a spoken line onto a
    # beat makes it sound seasick, and it has no key to move.
    # Judged BEFORE the expensive stretch/shift/mix, and reported either way.
    # Speech is exempt: a voice note over a beat has no tempo or key to clash.
    compat_ok, compat_why = (True, [])
    if not vocal_is_speech:
        compat_ok, compat_why = compatibility(
            bpm_v, bpm_i, key_v[0], key_i[0],
            # the backing moves in bed_to_vocal, and it takes a bigger shift
            BED_MAX_SEMITONES if (arrange == "bed_to_vocal" and section_align)
            else max_semitones,
            vocal_mode=key_v[1], instr_mode=key_i[1])
        if compat_why:
            logger.info("mashup compatibility: %s", "; ".join(compat_why))
        if not compat_ok and refuse_incompatible:
            raise MashupUnavailable(
                "These two tracks do not go together: " + "; ".join(compat_why)
                + ". Pick a pair closer in tempo and key.")

    # In bed_to_vocal the backing is rebuilt section by section onto the
    # vocal's own timing, so every global adjustment to the vocal is not just
    # unnecessary but the exact damage this mode exists to avoid.
    bed_first = (arrange == "bed_to_vocal"
                 and section_align and not vocal_is_speech)

    ratio = 1.0
    v_rate = b_rate = 1.0
    if match_tempo and not vocal_is_speech and not bed_first:
        ratio = stretch_ratio(bpm_v, bpm_i)
        if abs(ratio - 1.0) > 0.01:
            _say("stretching")
            # THE DIRECTION. This was rate=1.0/ratio, which is backwards:
            # stretch_ratio returns how much the vocal must SPEED UP, and
            # librosa/rubberband both take rate>1 as faster. Measured on a
            # click track, a 100 BPM vocal that needed to reach 110 came out at
            # 90.7 -- moved away from the bed, roughly doubling the mismatch it
            # was supposed to remove. It is the single largest reason the
            # output sounded wrong.
            #
            # And the adjustment is SHARED between the two tracks rather than
            # loaded entirely onto the vocal -- see split_tempo.
            v_rate, b_rate = split_tempo(ratio, tempo_split)
            try:
                vocal = _rubberband(vocal, SAMPLE_RATE, tempo=v_rate)
                if abs(b_rate - 1.0) > 0.005:
                    bed = _rubberband(bed, SAMPLE_RATE, tempo=b_rate)
            except Exception as exc:
                logger.warning("time stretch failed, leaving tempo alone: %s", exc)
                ratio = v_rate = b_rate = 1.0
    elif bed_first and match_tempo:
        # Reported, not applied: the per-section stretch below is what actually
        # matches the tempo here, and it does it without touching the vocal.
        ratio = stretch_ratio(bpm_v, bpm_i)

    semis = 0
    key_gap = 0            # a shift that was refused as too damaging
    if match_key and not vocal_is_speech:
        semis = semitone_shift(key_v[0], key_i[0])
        # Beyond a couple of semitones the shift does more damage than the key
        # clash it fixes: a real pair measured A major over C major, and the
        # resulting +3 artefacted the whole vocal. Declining is reported in the
        # result (`key_gap`) rather than done silently, so a clash the listener
        # hears has a visible reason.
        # In bed_to_vocal the BACKING moves, and an instrumental takes a
        # shift of up to a tritone without the formant damage a voice
        # shows at three -- the same A-over-C pair shipped with no shift at
        # all (semitones=0, key_gap=3) and played out of key (2026-09-14).
        limit = BED_MAX_SEMITONES if bed_first else max_semitones
        if abs(semis) > limit:
            logger.info("key gap of %d semitones exceeds the %d-semitone limit "
                        "— leaving the %s at its own pitch", semis, limit,
                        "backing" if bed_first else "vocal")
            key_gap, semis = semis, 0
        if semis:
            _say("pitching")
            try:
                if bed_first:
                    # Same interval, opposite side. semitone_shift says how far
                    # the VOCAL would move to reach the backing's key; here the
                    # backing moves to the vocal's instead, because the vocal
                    # is the one thing this mode will not touch.
                    bed = _rubberband(bed, SAMPLE_RATE, semitones=-semis)
                else:
                    vocal = _rubberband(vocal, SAMPLE_RATE, semitones=semis)
            except Exception as exc:
                logger.warning("pitch shift failed, leaving key alone: %s", exc)
                semis = 0

    _say("aligning")
    # Line the whole groove up, not one instant of it. This used to align the
    # first detected beat of each side, which puts a single point in the right
    # place and lets everything after it drift -- and the "first beat" of a
    # vocal stem is whichever breath the tracker liked. Cross-correlating the
    # onset envelopes picks the offset that lines up the most onsets across the
    # entire track.
    # bed_to_vocal has nothing to align: the backing is about to be built onto
    # the vocal's own section boundaries, so it starts where the vocal starts.
    lead_in = 0 if bed_first else best_offset(vocal, bed)
    if not bed_first and not lead_in and beats_i.size and beats_v.size:
        # Fallback for material with too few onsets to correlate (a sparse
        # ballad, a pad): the old first-beat rule is still better than nothing.
        offset_s = float(beats_i[0]) - float(beats_v[0]) / max(ratio, 1e-6)
        lead_in = max(0, int(offset_s * SAMPLE_RATE))
    if lead_in:
        vocal = np.concatenate([np.zeros(lead_in, dtype=np.float32), vocal])

    # Section-by-section placement, when the structure of both sides can be
    # found. One global offset means a verse can sit over a breakdown, and any
    # residual drift lasts the rest of the track; placing each vocal section at
    # an instrumental section's start re-syncs at every boundary, so drift
    # cannot accumulate past one section. Falls back to the single-offset mix
    # whenever either structure is unreadable -- short clips, speech, ambient
    # material with no clear boundaries.
    placements = []
    if section_align and not vocal_is_speech:
        _say("arranging")
        v_secs = detect_sections(vocal)
        b_secs = detect_sections(bed)
        if len(v_secs) >= 2 and len(b_secs) >= 2:
            if bed_first:
                arranged, placements = arrange_bed_to_vocal(
                    len(vocal), v_secs, bed, b_secs)
            else:
                arranged, placements = arrange_to_sections(
                    vocal, v_secs, len(bed), b_secs)
            if placements and _has_energy(arranged):
                if bed_first:
                    bed = arranged
                else:
                    vocal = arranged
            else:
                logger.info("section arrangement placed nothing — keeping the "
                            "single-offset mix")
                placements = []
        else:
            logger.info("structure unreadable (%d vocal / %d bed sections) — "
                        "keeping the single-offset mix", len(v_secs), len(b_secs))

        # bed_to_vocal deferred the global tempo match, because arranging the
        # backing section by section IS the tempo match. When the arrangement
        # does not happen -- unreadable structure, nothing placed -- that
        # deferral would otherwise ship a mix with no tempo matching at all,
        # which is worse than either arrangement. So it is applied here, late,
        # and the mix falls back to exactly the single-offset behaviour.
        if bed_first and not placements:
            bed_first = False
            if match_tempo and abs(ratio - 1.0) > 0.01:
                _say("stretching")
                v_rate, b_rate = split_tempo(ratio, tempo_split)
                try:
                    vocal = _rubberband(vocal, SAMPLE_RATE, tempo=v_rate)
                    if abs(b_rate - 1.0) > 0.005:
                        bed = _rubberband(bed, SAMPLE_RATE, tempo=b_rate)
                except Exception as exc:
                    logger.warning("late time stretch failed, leaving tempo "
                                   "alone: %s", exc)
                    v_rate = b_rate = 1.0
            lead_in = best_offset(vocal, bed)
            if lead_in:
                vocal = np.concatenate(
                    [np.zeros(lead_in, dtype=np.float32), vocal])

    _say("mixing")
    # An arranged vocal is already exactly bed-length, so the bed is used as
    # is; only the single-offset path needs the bed looped or trimmed to fit.
    # bed_to_vocal runs the other way round: the arranged BED is already exactly
    # vocal-length, so the output follows the vocal and the vocal is never
    # trimmed or tiled to reach it.
    n = len(vocal) if (bed_first and placements) else (
        len(bed) if placements else max(len(vocal), 1))
    bed_fit = _tile_to(bed, n)
    if placements and not bed_first and len(vocal) != n:
        vocal = _tile_to(vocal, n) if len(vocal) < n else vocal[:n]
    vocal, bed_fit = _balance(vocal, bed_fit, bed_gain_db)
    mix = _normalize(vocal + bed_fit, 0.97)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    try:
        sf.write(out_path, mix, SAMPLE_RATE)
    except Exception as exc:
        raise MashupUnavailable("Could not save the mashup: %s" % exc) from exc

    return {"path": out_path,
            "bpm_vocal": round(bpm_v, 1), "bpm_instr": round(bpm_i, 1),
            "stretch": round(ratio, 3), "semitones": semis,
            "key_vocal": key_v[2] + ((" " + key_v[1]) if key_v[1] else ""),
            "key_instr": key_i[2] + ((" " + key_i[1]) if key_i[1] else ""),
            "key_gap": key_gap,
            "arrange": "bed_to_vocal" if bed_first else "vocal_to_bed",
            "sections": len(placements),
            "placements": placements,
            "vocal_rate": round(v_rate, 3),
            "bed_rate": round(b_rate, 3),
            "compatible": bool(compat_ok),
            "warnings": list(compat_why),
            "offset_s": round(lead_in / SAMPLE_RATE, 2),
            "speech": bool(vocal_is_speech),
            "seconds": round(len(mix) / SAMPLE_RATE, 1),
            "took": round(time.time() - t_start, 1)}


def engine_available() -> bool:
    """True when a mashup could actually run -- the surfaces explain rather
    than offering a button that always fails."""
    try:
        import demucs.api   # noqa: F401
        import librosa      # noqa: F401
        import soundfile    # noqa: F401
        return True
    except Exception:
        return False


def default_out_path(tag: str = "") -> str:
    base = "mashup_%d%s.wav" % (int(time.time()), ("_" + tag) if tag else "")
    try:
        from config import OUTPUT_DIR
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        return os.path.join(OUTPUT_DIR, base)
    except Exception:
        return os.path.join(tempfile.gettempdir(), base)

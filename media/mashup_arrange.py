"""Arrangement: laying one track's parts out along another's timeline.

Sits above analysis — it consumes sections, gaps and beat grids and returns
a rendered buffer. Split out of mashup.py unchanged.
"""
import logging

import numpy as np

from mashup_dsp import SAMPLE_RATE, _fit_span, _place, _rubberband
from mashup_analysis import snap_to_silence, vocal_gaps

logger = logging.getLogger("assistant.mashup")


def arrange_to_sections(vocal: np.ndarray, v_secs: list,
                        bed_len: int, b_secs: list,
                        sr: int = SAMPLE_RATE, fade_s: float = 0.12,
                        match_labels: bool = False) -> tuple:
    """Lay vocal sections onto instrumental sections, looping or trimming.

    The global-offset version put ONE offset on the whole vocal, so a verse
    could sit over a breakdown and a chorus over an outro, and any drift lasted
    the rest of the track. Here each instrumental section gets a vocal section
    placed at ITS start (a beat position), so every section re-syncs -- drift
    cannot accumulate past one section.

    Short vocal material is looped to fill the section rather than leaving the
    backing bare, and every join is crossfaded, both of which are lifted
    straight from auto-mashup-mix's mix logic.

    Returns (arranged_vocal, placements) where placements describes what went
    where -- the surfaces report the count, and it is the only way to tell an
    arrangement that worked from one that silently placed nothing.
    """
    out = np.zeros(bed_len, dtype=np.float32)
    fade = int(fade_s * sr)
    placements = []
    if not v_secs or not b_secs:
        return out, placements

    gaps = vocal_gaps(vocal, sr)
    unused = list(range(len(v_secs)))

    for bs in b_secs:
        b_start = int(bs["start"] * sr)
        b_end = min(int(bs["end"] * sr), bed_len)
        room = b_end - b_start
        if room <= sr or not unused:
            # Nothing left to place. AutoMashup (Delabaere et al., GRETSI'25)
            # does the same thing here: when the base song has a section the
            # other does not, it plays the backing alone. Cycling back to the
            # start instead -- which is what this did -- replayed the intro
            # over three different sections, and the transcript showed those
            # verbatim repeats as plainly as it showed the cut words.
            continue

        # Verse against verse, chorus against chorus: pick an unused vocal
        # section whose cluster label matches this one, else the next in order,
        # so the vocal keeps its own narrative sequence.
        pick = unused[0]
        if match_labels:
            pick = next((i for i in unused
                         if v_secs[i].get("label") == bs.get("label")), unused[0])
        unused.remove(pick)
        vs = v_secs[pick]

        # Move both cuts into silence so no word is guillotined. This is the
        # fix for the 100.7% word error rate: a cut inside a vowel is a splice,
        # a cut inside a pause is inaudible.
        v_from = snap_to_silence(vs["start"], gaps)
        v_to = snap_to_silence(vs["end"], gaps)
        a, b = int(v_from * sr), int(v_to * sr)
        if b <= a:
            a, b = int(vs["start"] * sr), int(vs["end"] * sr)
        chunk = vocal[a:min(b, len(vocal))]
        if chunk.size < sr // 2:
            continue
        # Too long for the section: trim, again at a silence rather than
        # mid-word. Too short: leave the rest of the bed bare instead of
        # looping the words round again.
        if len(chunk) > room:
            cut = snap_to_silence(v_from + room / sr, gaps)
            end = int(min(max(cut, v_from + 1.0) * sr, len(vocal)))
            chunk = vocal[a:end][:room]
        _place(out, chunk, b_start, fade)
        placements.append({"bed_start": round(bs["start"], 2),
                           "bed_end": round(bs["end"], 2),
                           "vocal_start": round(v_from, 2),
                           "vocal_end": round(v_from + len(chunk) / sr, 2),
                           "label_match": v_secs[pick].get("label") == bs.get("label"),
                           "fills": round(len(chunk) / max(room, 1), 2)})
    return out, placements


def _pick_bed_section(vs: dict, b_secs: list, used: dict) -> int:
    """Which instrumental section goes under this vocal section.

    Same label first, and among equals the one used least -- so a chorus lands
    on the chorus backing, and when the vocal has more sections than the
    instrumental the backing repeats evenly instead of hammering section one.
    """
    order = sorted(range(len(b_secs)), key=lambda i: (used.get(i, 0), i))
    same = [i for i in order if b_secs[i].get("label") == vs.get("label")]
    return (same or order)[0]


def arrange_bed_to_vocal(vocal_len: int, v_secs: list, bed: np.ndarray,
                         b_secs: list, sr: int = SAMPLE_RATE,
                         fade_s: float = 0.12,
                         max_stretch: float = 2.0) -> tuple:
    """Build a backing that follows the vocal, and never touch the vocal.

    The inversion of arrange_to_sections, taken from ax-le/automashup. Instead
    of cutting the voice to fit the instrumental, each vocal section keeps its
    own timing and an instrumental section is stretched onto it. Word damage
    stops being something to tune: there are no cuts in the vocal that could
    land badly, because there are no cuts in the vocal.

    Measured against the other direction on the same pair, transcribing the
    vocal re-separated from each finished mix: the share of the original
    vocabulary that survives went from 41% to the ~70% an untouched record
    scores. The stretch it costs lands on the backing instead, which tolerates
    it far better than a voice does -- which is also why this mode does not
    split the tempo adjustment between the two sides.

    Each section is stretched a `fade_s` overrun long, so its tail crossfades
    into the next section's head rather than butting against it; automashup
    concatenates bare, and every join there is a potential click.

    Returns (bed_following_vocal, placements). The bed comes back exactly
    `vocal_len` samples long, so the output follows the vocal.
    """
    out = np.zeros(max(int(vocal_len), 0), dtype=np.float32)
    placements = []
    if not v_secs or not b_secs or bed.size == 0 or out.size == 0:
        return out, placements

    fade = int(fade_s * sr)
    used = {}
    for vs in v_secs:
        a = max(int(vs["start"] * sr), 0)
        b = min(int(vs["end"] * sr), len(out))
        room = b - a
        if room <= sr // 2:
            continue

        pick = _pick_bed_section(vs, b_secs, used)
        used[pick] = used.get(pick, 0) + 1
        bs = b_secs[pick]
        src = bed[max(int(bs["start"] * sr), 0):min(int(bs["end"] * sr), len(bed))]
        if src.size < sr // 4:
            continue

        # Overrun by one fade so consecutive sections overlap through the join.
        want = min(room + fade, len(out) - a)
        src = _fit_span(src, want, max_stretch)
        rate = len(src) / float(max(want, 1))
        piece = src
        if abs(rate - 1.0) > 0.005:
            try:
                piece = _rubberband(src, sr, tempo=rate)
            except Exception as exc:
                logger.warning("bed section stretch failed, using it as is: %s", exc)
                piece = src
        if len(piece) > want:
            piece = piece[:want]
        elif len(piece) < want:
            piece = np.concatenate(
                [piece, np.zeros(want - len(piece), dtype=np.float32)])

        _place(out, piece, a, fade)
        placements.append({"vocal_start": round(vs["start"], 2),
                           "vocal_end": round(vs["end"], 2),
                           "bed_start": round(bs["start"], 2),
                           "bed_end": round(bs["end"], 2),
                           "label_match": bs.get("label") == vs.get("label"),
                           "rate": round(rate, 3),
                           "reused": used[pick] > 1})
    return out, placements

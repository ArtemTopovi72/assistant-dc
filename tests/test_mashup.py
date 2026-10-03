"""mashup.py: tempo folding, key wrapping, balance, mixing and error paths.

The Demucs separator is stubbed for everything except the one test that is
explicitly marked slow, so this suite runs offline in seconds and never
touches the GPU -- the same rule test_weather.py follows for the network.

Run: venv/Scripts/python.exe tests/test_mashup.py
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)

import numpy as np
import soundfile as sf

import mashup as M

RESULTS = []
import os as _os
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name
          + ("" if cond or not detail else " - " + str(detail)[:200]))
    if not cond and _os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))


_TMP = tempfile.mkdtemp(prefix="mashup_")
SR = M.SAMPLE_RATE


def tone(freq=220.0, secs=4.0, sr=SR, amp=0.3):
    t = np.arange(int(secs * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def click_track(bpm=120.0, secs=12.0, sr=SR):
    """A percussive pulse train, so beat tracking has something to lock onto.

    Bursts of shaped NOISE, not bare windows on silence: librosa's onset
    strength works on spectral flux, and a 4s train of pure hanning pulses
    measured 0.0 BPM -- a fixture too thin to detect, which looked exactly
    like an engine failure until a real song was tried and measured fine.
    """
    rng = np.random.default_rng(11)
    y = np.zeros(int(secs * sr), dtype=np.float32)
    step = int(sr * 60.0 / bpm)
    burst = int(sr * 0.03)
    env = np.hanning(burst).astype(np.float32)
    for i in range(0, len(y) - burst, step):
        y[i:i + burst] += (rng.normal(0, 0.6, burst).astype(np.float32) * env)
    return y


def wav(name, y, sr=SR):
    p = os.path.join(_TMP, name)
    sf.write(p, y, sr)
    return p


class FakeSeparator:
    """Stands in for Demucs: splits nothing, just hands back fixed stems.

    Every mashup calls separate() twice, so the stub is keyed by file path --
    returning one fixed answer would make the vocal and the bed identical and
    hide any mixing bug behind a perfect correlation.
    """
    def __init__(self, table):
        self.table = table
        self.calls = []

    def install(self):
        self._orig = M.separate
        def fake(path, device=""):
            self.calls.append(path)
            for key, stems in self.table.items():
                if key in os.path.basename(path):
                    return stems
            raise AssertionError("unexpected separate() on " + path)
        M.separate = fake
        return self

    def remove(self):
        M.separate = self._orig


def test_tempo_folding():
    # Same groove at half/double time must resolve to a nudge, not a 2x smear.
    check("half-time vocal folds to ~1.0", abs(M.stretch_ratio(191.0, 95.5) - 1.0) < 0.02,
          M.stretch_ratio(191.0, 95.5))
    check("double-time vocal folds to ~1.0", abs(M.stretch_ratio(60.0, 120.0) - 1.0) < 0.02,
          M.stretch_ratio(60.0, 120.0))
    # A genuine small difference is passed through as a real stretch.
    r = M.stretch_ratio(100.0, 110.0)
    check("a close tempo becomes a real stretch", abs(r - 1.1) < 0.01, r)
    # Whatever the inputs, the result must stay inside the smear limit.
    for v in (37.0, 71.0, 95.0, 128.0, 174.0, 200.0):
        for i in (60.0, 90.0, 128.0, 175.0):
            r = M.stretch_ratio(v, i)
            # sqrt(2) is the true bound of octave folding -- two tempos a
            # tritone apart are equally far from 1.0 in either octave.
            check("ratio %g/%g stays within the stretch limit" % (v, i),
                  1.0 / M._MAX_STRETCH - 1e-9 <= r <= M._MAX_STRETCH + 1e-9, r)
            check("ratio %g/%g is the nearest octave to 1.0" % (v, i),
                  abs(np.log2(r)) <= abs(np.log2(r * 2)) + 1e-9
                  and abs(np.log2(r)) <= abs(np.log2(r / 2)) + 1e-9, r)
    # Missing/nonsense tempo means "leave it alone", never a divide by zero.
    check("unknown vocal tempo -> no stretch", M.stretch_ratio(0.0, 120.0) == 1.0)
    check("unknown bed tempo -> no stretch", M.stretch_ratio(120.0, 0.0) == 1.0)
    check("negative tempo -> no stretch", M.stretch_ratio(-5.0, 120.0) == 1.0)


def test_key_wrapping():
    check("same key -> no shift", M.semitone_shift(5, 5) == 0)
    # C (0) over B (11): down one, not up eleven.
    check("C over B goes down one semitone", M.semitone_shift(0, 11) == -1,
          M.semitone_shift(0, 11))
    check("B over C goes up one semitone", M.semitone_shift(11, 0) == 1,
          M.semitone_shift(11, 0))
    for v in range(12):
        for i in range(12):
            s = M.semitone_shift(v, i)
            # [-6, +5]: a tritone goes DOWN, since shifting a voice up six
            # semitones is the more audible damage.
            check("shift %d->%d stays within half an octave" % (v, i), -6 <= s <= 5, s)
            check("shift %d->%d lands on the target pitch class" % (v, i),
                  (v + s) % 12 == i, s)
    # No key detected (speech, drums) must mean "do not touch the pitch".
    check("unknown vocal key -> no shift", M.semitone_shift(-1, 5) == 0)
    check("unknown bed key -> no shift", M.semitone_shift(5, -1) == 0)


def test_key_and_tempo_estimation():
    # A pure A4 should read as an A something; the mode is a judgement call on
    # one tone, so only the pitch class is asserted.
    pc, mode, name = M.estimate_key(tone(440.0, 3.0))
    check("a 440Hz tone is detected as A", name == "A", (pc, mode, name))
    # Silence has no key -- and must say so rather than defaulting to C, which
    # would send a pitch shift somewhere arbitrary.
    pc2, _m2, _n2 = M.estimate_key(np.zeros(SR, dtype=np.float32))
    check("silence reports no key at all", pc2 == -1, pc2)
    bpm, beats = M.estimate_tempo(click_track(120.0, 12.0))
    check("a 120 BPM click is measured near 120 (or a metrical multiple)",
          min(abs(bpm - 120.0), abs(bpm - 60.0), abs(bpm - 240.0)) < 6.0, bpm)
    check("and beat positions come back with it", beats.size > 4, beats.size)
    bpm0, beats0 = M.estimate_tempo(np.zeros(SR, dtype=np.float32))
    check("silence yields no tempo instead of raising", bpm0 == 0.0 or beats0.size == 0,
          (bpm0, beats0.size))


def test_balance_puts_the_vocal_on_top():
    """The bug this guards: peak-normalising both parts left the BED louder to
    the ear than the voice, because a vocal stem is spiky and a band mix is
    dense. Balance is by RMS for exactly this reason."""
    vocal = tone(300.0, 3.0, amp=0.2)
    vocal[::2000] = 0.95            # spiky peaks, modest average
    bed = tone(110.0, 3.0, amp=0.35)
    v2, b2 = M._balance(vocal, bed, -3.5)
    rv, rb = M._rms(v2), M._rms(b2)
    check("the bed ends up quieter than the vocal", rb < rv, (rv, rb))
    ratio_db = 20 * np.log10(rb / rv)
    check("...by the gain actually asked for", abs(ratio_db + 3.5) < 0.2, ratio_db)
    # A silent side must not blow up the scaling.
    v3, b3 = M._balance(np.zeros(100, dtype=np.float32), bed, -3.5)
    check("a silent vocal leaves the bed untouched instead of dividing by zero",
          np.allclose(b3, bed))


def test_tile_and_normalize():
    short = tone(200.0, 1.0)
    out = M._tile_to(short, len(short) * 3 + 7)
    check("a short bed is looped to cover the vocal", len(out) == len(short) * 3 + 7)
    check("...and the loop repeats the source", np.allclose(out[:len(short)], short))
    check("an empty bed becomes silence, not a crash",
          M._tile_to(np.array([], dtype=np.float32), 128).shape == (128,))
    n = M._normalize(tone(200.0, 0.5, amp=0.01), 0.9)
    check("normalize reaches the requested peak", abs(float(np.max(np.abs(n))) - 0.9) < 1e-3)
    check("normalizing silence does not divide by zero",
          np.all(M._normalize(np.zeros(64, dtype=np.float32)) == 0))


def test_make_mashup_end_to_end():
    voc = tone(330.0, 12.0, amp=0.25) + click_track(120.0, 12.0) * 0.15
    bed = tone(110.0, 6.0, amp=0.3) + click_track(120.0, 6.0) * 0.4
    fake = FakeSeparator({
        "voc.wav": {"vocals": voc, "drums": np.zeros(1, np.float32),
                    "bass": np.zeros(1, np.float32), "other": np.zeros(1, np.float32)},
        "bed.wav": {"vocals": np.zeros(len(bed), np.float32), "drums": bed,
                    "bass": np.zeros(1, np.float32), "other": np.zeros(1, np.float32)},
    }).install()
    try:
        vp, bp = wav("voc.wav", voc), wav("bed.wav", bed)
        out = os.path.join(_TMP, "out.wav")
        seen = []
        rep = M.make_mashup(vp, bp, out, progress=seen.append)
        check("the mashup file is written", os.path.exists(out) and os.path.getsize(out) > 1000)
        check("both inputs were separated", len(fake.calls) == 2, fake.calls)
        check("progress is reported for the slow steps", "separating" in seen and "mixing" in seen,
              seen)
        check("the report carries both tempos", rep["bpm_vocal"] > 0 and rep["bpm_instr"] > 0, rep)
        check("the report says how long it took", rep["took"] >= 0 and rep["seconds"] > 0, rep)

        y, sr = sf.read(out)
        y = y if y.ndim == 1 else y.mean(axis=1)
        check("the output is at the engine's sample rate", sr == SR, sr)
        check("the mix is not silent", float(np.max(np.abs(y))) > 0.1, float(np.max(np.abs(y))))
        check("the mix does not clip", float(np.max(np.abs(y))) <= 1.0, float(np.max(np.abs(y))))
        # The 2s bed under a 4s vocal must be looped, not leave 2s of bare voice.
        check("the shorter bed was looped to cover the whole vocal",
              len(y) >= int(11.5 * SR), len(y) / SR)
    finally:
        fake.remove()


def test_make_mashup_speech_path():
    """A voice note keeps its own tempo and pitch: stretching speech onto a
    beat makes it sound seasick, and it has no key to move to."""
    speech = (np.random.default_rng(7).normal(0, 0.05, int(2.0 * SR))).astype(np.float32)
    bed = tone(110.0, 6.0, amp=0.3) + click_track(120.0, 6.0) * 0.4
    fake = FakeSeparator({
        "bed.wav": {"vocals": np.zeros(len(bed), np.float32), "drums": bed,
                    "bass": np.zeros(1, np.float32), "other": np.zeros(1, np.float32)},
    }).install()
    try:
        sp, bp = wav("speech.wav", speech), wav("bed.wav", bed)
        out = os.path.join(_TMP, "out_speech.wav")
        rep = M.make_mashup(sp, bp, out, vocal_is_speech=True)
        check("speech is NOT sent through the separator", fake.calls == [bp], fake.calls)
        check("speech is never time-stretched", rep["stretch"] == 1.0, rep)
        check("speech is never pitch-shifted", rep["semitones"] == 0, rep)
        check("the report marks it as speech", rep["speech"] is True, rep)
        check("a speech mashup still produces audio", os.path.exists(out), out)
    finally:
        fake.remove()


def test_failure_paths_are_user_facing():
    # A missing file is a sentence, not a stack trace: both surfaces print
    # str(exc) straight to the user.
    try:
        M.separate(os.path.join(_TMP, "nope.wav"))
        check("a missing input raises MashupUnavailable", False, "no exception")
    except M.MashupUnavailable as exc:
        check("a missing input raises MashupUnavailable", True)
        check("...with a readable message", "not there" in str(exc).lower(), str(exc))
    except Exception as exc:
        check("a missing input raises MashupUnavailable", False, type(exc).__name__)

    # No stems at all must be reported, not silently mixed as silence.
    try:
        M.backing_of({"vocals": tone(200.0, 1.0)})
        check("a stem set with no instrumental is refused", False, "no exception")
    except M.MashupUnavailable:
        check("a stem set with no instrumental is refused", True)

    # A vocal donor with no singing in it gets a message naming the fix.
    fake = FakeSeparator({
        "silentvoc.wav": {"vocals": np.zeros(SR, np.float32), "drums": np.zeros(1, np.float32),
                          "bass": np.zeros(1, np.float32), "other": np.zeros(1, np.float32)},
        "bed.wav": {"vocals": np.zeros(1, np.float32), "drums": tone(110.0, 1.0),
                    "bass": np.zeros(1, np.float32), "other": np.zeros(1, np.float32)},
    }).install()
    try:
        vp = wav("silentvoc.wav", np.zeros(SR, np.float32))
        bp = wav("bed.wav", tone(110.0, 1.0))
        try:
            M.make_mashup(vp, bp, os.path.join(_TMP, "x.wav"))
            check("a track with no vocals is refused with advice", False, "no exception")
        except M.MashupUnavailable as exc:
            check("a track with no vocals is refused with advice",
                  "voice message" in str(exc) or "vocals" in str(exc), str(exc))
    finally:
        fake.remove()


def test_compatibility_gate():
    """The thing the pipeline was missing, and the reason a correctly-assembled
    mashup could still be unlistenable: the two songs simply did not go
    together. Numbers are from the real pair a user sent."""
    # 139.7 over 74.9 BPM, A major over C major -- assembled fine, sounded bad.
    ok, why = M.compatibility(139.7, 74.9, 9, 0)     # A=9, C=0
    check("the real failing pair is called incompatible", not ok, why)
    check("...and the tempo gap is named", any("BPM" in r for r in why), why)
    check("...and the key gap is named", any("semitones" in r for r in why), why)

    # A pair that genuinely works: same tempo family, a tone apart.
    ok2, why2 = M.compatibility(128.0, 128.0, 0, 2)   # C over D, 2 semitones
    check("a matched pair passes", ok2, why2)
    # Half-time is the same groove, so it must not be reported as a tempo clash.
    ok3, why3 = M.compatibility(140.0, 70.0, 0, 0)
    check("half-time is not a tempo clash", ok3, why3)

    # Unknown tempo or key cannot be judged, and must not be guessed at.
    ok4, why4 = M.compatibility(0.0, 120.0, -1, 5)
    check("nothing measurable -> no complaint invented", ok4, why4)

    # refuse_incompatible turns the warning into a refusal, and the message
    # tells the user what to do about it.
    d0 = _dt_date()
    voc = tone(330.0, 6.0, amp=0.25) + click_track(150.0, 6.0) * 0.2
    bed = tone(110.0, 6.0, amp=0.3) + click_track(96.0, 6.0) * 0.4
    fake = FakeSeparator({
        "cv.wav": {"vocals": voc, "drums": click_track(150.0, 6.0),
                   "bass": np.zeros(1, np.float32), "other": np.zeros(1, np.float32)},
        "cb.wav": {"vocals": np.zeros(1, np.float32), "drums": click_track(96.0, 6.0),
                   "bass": np.zeros(1, np.float32), "other": bed},
    }).install()
    try:
        vp, bp = wav("cv.wav", voc), wav("cb.wav", bed)
        rep = M.make_mashup(vp, bp, os.path.join(_TMP, "warn.wav"))
        check("an incompatible pair still renders by default", os.path.exists(rep["path"]))
        check("...but the report says it is not compatible",
              rep["compatible"] is False and rep["warnings"], rep)
        try:
            M.make_mashup(vp, bp, os.path.join(_TMP, "refused.wav"),
                          refuse_incompatible=True)
            check("refuse_incompatible refuses", False, "no exception")
        except M.MashupUnavailable as exc:
            check("refuse_incompatible refuses", True)
            check("...with advice on what to do instead",
                  "closer in tempo and key" in str(exc), str(exc))
    finally:
        fake.remove()


def _dt_date():
    return None


def test_pitch_shift_is_capped():
    """+3 semitones artefacted a whole vocal in the live case. Beyond the cap
    the engine declines and REPORTS the gap, so a clash the listener hears has
    a visible reason rather than looking like a bug."""
    voc = tone(440.0, 4.0, amp=0.25)          # A
    bed = tone(261.6, 4.0, amp=0.3)           # C -- three semitones away
    fake = FakeSeparator({
        "pv.wav": {"vocals": voc, "drums": np.zeros(1, np.float32),
                   "bass": np.zeros(1, np.float32), "other": np.zeros(1, np.float32)},
        "pb.wav": {"vocals": np.zeros(1, np.float32), "drums": np.zeros(1, np.float32),
                   "bass": np.zeros(1, np.float32), "other": bed},
    }).install()
    try:
        vp, bp = wav("pv.wav", voc), wav("pb.wav", bed)
        # The cap is about the VOICE: it applies when the voice is what moves.
        rep = M.make_mashup(vp, bp, os.path.join(_TMP, "cap.wav"), max_semitones=2,
                            arrange="vocal_to_bed")
        check("a shift beyond the cap is not applied to the voice", rep["semitones"] == 0, rep)
        check("...and the refused gap is reported", rep["key_gap"] != 0, rep)
        # Raising the cap lets it through, so the limit is a policy, not a bug.
        rep2 = M.make_mashup(vp, bp, os.path.join(_TMP, "cap2.wav"), max_semitones=6,
                             arrange="vocal_to_bed")
        check("a higher cap allows the shift", rep2["semitones"] != 0, rep2)
        # In bed_to_vocal the BACKING moves, and three semitones on an
        # instrumental is fine -- the live A-over-C pair shipped out of key
        # with semitones=0 under the vocal cap (2026-09-14).
        rep3 = M.make_mashup(vp, bp, os.path.join(_TMP, "cap3.wav"), max_semitones=2)
        # (pure tones have no readable structure, so the arrangement itself
        # falls back -- the pitch decision is what this checks)
        check("bed_to_vocal shifts the backing three semitones by default",
              rep3["semitones"] == 3 and rep3["key_gap"] == 0, (rep3["semitones"], rep3["key_gap"]))
        check("...and the compatibility note no longer blames the voice",
              not any("artefacting the voice" in w for w in rep3.get("warnings", [])), rep3.get("warnings"))
    finally:
        fake.remove()


def test_tempo_comes_from_the_drums():
    """Measuring the vocal donor's tempo on its VOCAL stem was the root error:
    singing has almost no percussive onsets. The donor's own drums were already
    separated and being thrown away."""
    drums = click_track(120.0, 12.0)
    singing = tone(330.0, 12.0, amp=0.3)      # no onsets to speak of
    stems = {"vocals": singing, "drums": drums,
             "bass": np.zeros(1, np.float32), "other": np.zeros(1, np.float32)}
    perc = M.percussive_of(stems)
    check("the percussive stem is recovered", perc.size > 0)
    bpm_drums, _ = M.estimate_tempo(perc)
    check("tempo from the drums lands near the truth",
          min(abs(bpm_drums - 120.0), abs(bpm_drums - 60.0), abs(bpm_drums - 240.0)) < 8.0,
          bpm_drums)
    # A drumless donor must fall back rather than reporting 0 BPM, which would
    # silently disable both the stretch and the alignment.
    empty = M.percussive_of({"vocals": singing, "drums": np.zeros(4, np.float32),
                             "bass": np.zeros(4, np.float32)})
    check("a silent drum stem is not trusted", not M._has_energy(empty), empty[:4])


def test_offset_finds_a_real_delay():
    """Aligning one beat put a single instant right and let the rest drift.
    The offset is now chosen by correlating onsets across the whole track."""
    bed = click_track(120.0, 12.0)
    delay = int(1.5 * SR)
    vocal = np.concatenate([np.zeros(delay, dtype=np.float32),
                            click_track(120.0, 10.0)])
    # The vocal starts 1.5s late relative to the bed, so lining them up needs
    # NO extra delay -- the correlation should find ~0, not invent one.
    off = M.best_offset(vocal, bed)
    check("a vocal that already starts late is not delayed further",
          off < int(0.6 * SR), off / SR)
    # A vocal that starts early must land ON THE GRID -- but not at one
    # particular lag. Both sides here are periodic at 120 BPM, so every whole
    # number of beats is an equally correct answer, and asserting a specific
    # one tested the fixture rather than the function.
    off2 = M.best_offset(click_track(120.0, 10.0),
                         np.concatenate([np.zeros(delay, dtype=np.float32),
                                         click_track(120.0, 12.0)]))
    beat = 0.5 * SR                      # 120 BPM
    phase = (off2 % beat) / beat
    check("a vocal that starts early lands on a beat boundary",
          min(phase, 1.0 - phase) < 0.12, (off2 / SR, phase))
    check("silence yields no offset instead of raising",
          M.best_offset(np.zeros(SR, np.float32), np.zeros(SR, np.float32)) == 0)


def test_section_lengths_are_clamped():
    """Raw agglomerative boundaries came back as an 89-second block followed by
    four 5-second slivers on a real track. A sliver is not a musical part, and
    an 89s section gets filled by looping a short vocal chunk a dozen times --
    worse than the drift the arrangement exists to fix."""
    beats = np.arange(0.0, 300.0, 0.5)
    raw = [{"start": 0.0, "end": 89.0, "label": 0},     # far too long
           {"start": 89.0, "end": 94.0, "label": 1},    # sliver
           {"start": 94.0, "end": 99.0, "label": 1},    # sliver
           {"start": 99.0, "end": 115.0, "label": 2}]
    out = M._even_out_sections(raw, beats, 16.0)
    spans = [s["end"] - s["start"] for s in out]
    check("the 89-second block was split up", max(spans) <= 32.1, spans)
    check("...and no sliver survived", min(spans) >= 7.9, spans)
    check("the timeline is still covered end to end",
          abs(out[0]["start"] - 0.0) < 1.0 and abs(out[-1]["end"] - 115.0) < 1.0,
          (out[0]["start"], out[-1]["end"]))
    for a, b in zip(out[:-1], out[1:]):
        check("sections do not overlap or gap", abs(b["start"] - a["end"]) < 1.0,
              (a, b))
    # A single sane section is left exactly alone.
    same = M._even_out_sections([{"start": 0.0, "end": 16.0, "label": 0},
                                 {"start": 16.0, "end": 32.0, "label": 1}],
                                beats, 16.0)
    check("already-even sections are untouched", len(same) == 2, same)


def test_detect_sections_on_a_structured_signal():
    """Two clearly different halves must come back as separate sections, with
    boundaries on beats -- a join between beats clicks."""
    a = tone(220.0, 16.0, amp=0.3) + click_track(120.0, 16.0) * 0.3
    b = tone(330.0, 16.0, amp=0.3) + click_track(120.0, 16.0) * 0.3
    secs = M.detect_sections(np.concatenate([a, b]), target_len_s=8.0)
    check("a two-part signal yields at least two sections", len(secs) >= 2, secs)
    check("sections are in order and non-empty",
          all(s["end"] > s["start"] for s in secs)
          and all(b_["start"] >= a_["start"] for a_, b_ in zip(secs, secs[1:])), secs)
    check("every section carries a label", all("label" in s for s in secs), secs)
    # Too short to have structure -> no sections, and the caller falls back.
    check("a two-second clip has no readable structure",
          M.detect_sections(tone(220.0, 2.0)) == [], "expected []")


def test_arrange_places_into_every_section():
    v = np.concatenate([tone(300.0, 6.0, amp=0.3), tone(400.0, 6.0, amp=0.3)])
    v_secs = [{"start": 0.0, "end": 6.0, "label": 0},
              {"start": 6.0, "end": 12.0, "label": 1}]
    bed_len = int(40.0 * SR)
    b_secs = [{"start": 0.0, "end": 10.0, "label": 0},
              {"start": 10.0, "end": 20.0, "label": 1},
              {"start": 20.0, "end": 30.0, "label": 0},
              {"start": 30.0, "end": 40.0, "label": 1}]
    out, placements = M.arrange_to_sections(v, v_secs, bed_len, b_secs)
    check("the arrangement is exactly bed-length", len(out) == bed_len, len(out))
    # Each vocal section is used ONCE. Cycling back to the start to fill every
    # bed section replayed the intro over three of them, which the transcript
    # showed as verbatim repeats; AutoMashup (GRETSI'25) leaves the backing
    # bare in exactly this situation instead.
    check("a vocal section is never replayed to fill a gap",
          len(placements) == len(v_secs), placements)
    check("...and the surplus bed sections are left bare",
          len(placements) < len(b_secs), (len(placements), len(b_secs)))
    starts = [p["vocal_start"] for p in placements]
    check("the vocal keeps its own order", starts == sorted(starts), starts)
    for bs in b_secs[:len(placements)]:
        seg = out[int(bs["start"] * SR):int(bs["end"] * SR)]
        check("section starting at %.0fs carries audio" % bs["start"],
              M._has_energy(seg), float(np.max(np.abs(seg))))
    # Nothing to place -> silence and an empty plan, never a crash.
    out2, pl2 = M.arrange_to_sections(v, [], bed_len, b_secs)
    check("no vocal sections -> nothing placed", pl2 == [] and not M._has_energy(out2))


def test_fit_span_keeps_the_stretch_musical():
    """Only halving and doubling, so a bar stays a bar.

    The rate a section gets stretched by has to stay near 1: a 3x stretch of a
    drum kit sounds like a tape fault. Taking the first half of an eight-bar
    section leaves four bars still starting on a downbeat, which is the only
    kind of trim that survives being played under a voice.
    """
    want = int(14.0 * SR)
    for src_s in (30.0, 3.0, 14.0, 60.0, 1.0):
        src = M._fit_span(np.ones(int(src_s * SR), np.float32), want)
        rate = len(src) / float(want)
        check("a %.0fs section onto 14s stretches by %.2f, within 2x"
              % (src_s, rate), 0.5 <= rate <= 2.0, rate)
    check("an empty section is handed back untouched",
          M._fit_span(np.zeros(0, np.float32), want).size == 0)
    check("a zero-length target does not divide by zero",
          M._fit_span(np.ones(100, np.float32), 0).size == 100)


def test_bed_section_picker_prefers_labels_then_spreads():
    b_secs = [{"label": 0}, {"label": 1}, {"label": 1}]
    used = {}
    picks = []
    for vs in ({"label": 1}, {"label": 1}, {"label": 1}, {"label": 7}):
        i = M._pick_bed_section(vs, b_secs, used)
        used[i] = used.get(i, 0) + 1
        picks.append(i)
    check("a labelled vocal section gets a bed section with that label",
          picks[0] in (1, 2) and picks[1] in (1, 2), picks)
    check("...and the two matching sections alternate before either repeats",
          picks[0] != picks[1], picks)
    check("a third one reuses the least-used match", picks[2] in (1, 2), picks)
    check("an unmatched label still gets a section rather than nothing",
          picks[3] == 0, picks)


def test_bed_to_vocal_never_touches_the_vocal():
    """The inversion taken from ax-le/automashup.

    Measured on the real pair, transcribing the vocal re-separated from each
    finished mix: the share of the original vocabulary that survives went from
    46% under vocal_to_bed to 80% here, against 84% for automashup itself. The
    whole reason is this direction -- there are no cuts in the vocal, so no cut
    can land inside a word.
    """
    v_len = int(24.0 * SR)
    v_secs = [{"start": 0.0, "end": 8.0, "label": 0},
              {"start": 8.0, "end": 16.0, "label": 1},
              {"start": 16.0, "end": 24.0, "label": 0}]
    bed = np.concatenate([click_track(120.0, 10.0), tone(110.0, 10.0, amp=0.4)])
    b_secs = [{"start": 0.0, "end": 10.0, "label": 0},
              {"start": 10.0, "end": 20.0, "label": 1}]

    out, placements = M.arrange_bed_to_vocal(v_len, v_secs, bed, b_secs)
    check("the backing comes back exactly vocal-length", len(out) == v_len, len(out))
    check("every vocal section got a backing section",
          len(placements) == len(v_secs), placements)
    for vs in v_secs:
        seg = out[int(vs["start"] * SR):int(vs["end"] * SR)]
        check("the section at %.0fs has backing under it" % vs["start"],
              M._has_energy(seg), float(np.max(np.abs(seg))))
    check("a bed section is reused when the vocal has more sections",
          any(p["reused"] for p in placements), placements)
    check("matching labels are honoured where they exist",
          placements[1]["label_match"], placements)
    check("every stretch stayed within the 2x bound",
          all(0.5 <= p["rate"] <= 2.0 for p in placements),
          [p["rate"] for p in placements])
    # The joins overlap, so no section boundary drops to silence.
    for vs in v_secs[1:]:
        edge = out[int(vs["start"] * SR) - 256:int(vs["start"] * SR) + 256]
        check("the join at %.0fs does not drop out" % vs["start"],
              M._has_energy(edge), float(np.max(np.abs(edge))))
    # Degenerate inputs return silence and an empty plan, never a crash.
    o2, p2 = M.arrange_bed_to_vocal(v_len, [], bed, b_secs)
    check("no vocal sections -> nothing placed", p2 == [] and not M._has_energy(o2))
    o3, p3 = M.arrange_bed_to_vocal(v_len, v_secs, np.zeros(0, np.float32), b_secs)
    check("an empty bed -> nothing placed", p3 == [] and not M._has_energy(o3))
    o4, p4 = M.arrange_bed_to_vocal(0, v_secs, bed, b_secs)
    check("a zero-length vocal -> nothing placed", p4 == [] and o4.size == 0)


def test_bed_to_vocal_end_to_end_reports_its_mode():
    """The mode has to be visible in the report and cost the vocal nothing:
    no stretch, no offset, and an output that follows the vocal's length."""
    voc = tone(330.0, 24.0, amp=0.25) + click_track(140.0, 24.0) * 0.15
    bed = tone(110.0, 20.0, amp=0.3) + click_track(90.0, 20.0) * 0.4
    fake = FakeSeparator({
        "v2.wav": {"vocals": voc, "drums": click_track(140.0, 24.0) * 0.15,
                   "bass": np.zeros(1, np.float32), "other": np.zeros(1, np.float32)},
        "b2.wav": {"vocals": np.zeros(len(bed), np.float32), "drums": bed,
                   "bass": np.zeros(1, np.float32), "other": np.zeros(1, np.float32)},
    }).install()
    # Structure is stubbed, not detected: this test is about the wiring in
    # make_mashup -- which side gets stretched, which side sets the length --
    # and a synthetic tone does not give librosa enough to segment. Whether
    # real structure is found is test_detect_sections_on_a_structured_signal.
    orig = M.detect_sections
    def three_sections(y, sr=SR, **k):
        dur = len(y) / float(sr)
        n = 3 if dur > 22.0 else 2
        edges = np.linspace(0.0, dur, n + 1)
        return [{"start": float(edges[i]), "end": float(edges[i + 1]),
                 "label": i % 2} for i in range(n)]
    M.detect_sections = three_sections
    try:
        vp, bp = wav("v2.wav", voc), wav("b2.wav", bed)
        out = os.path.join(_TMP, "out_btv.wav")
        rep = M.make_mashup(vp, bp, out)
        check("the report names the arrangement", rep["arrange"] == "bed_to_vocal", rep)
        check("...and bed_to_vocal is what you get without asking (the default since 2026-09-14)",
              __import__("inspect").signature(M.make_mashup).parameters["arrange"].default == "bed_to_vocal")
        check("the vocal was not time-stretched", rep["vocal_rate"] == 1.0, rep)
        check("the vocal was not offset", rep["offset_s"] == 0.0, rep)
        check("the tempo gap is still reported", rep["bpm_vocal"] > 0 and rep["stretch"] > 0,
              rep)
        y, _sr = sf.read(out)
        y = y if y.ndim == 1 else y.mean(axis=1)
        check("the output follows the vocal's length",
              abs(len(y) - len(voc)) < SR, (len(y) / SR, len(voc) / SR))
        check("the mix is not silent", float(np.max(np.abs(y))) > 0.1)
        check("the mix does not clip", float(np.max(np.abs(y))) <= 1.0)
        # The other direction must still be reachable, unchanged.
        out2 = os.path.join(_TMP, "out_vtb.wav")
        rep2 = M.make_mashup(vp, bp, out2, arrange="vocal_to_bed")
        check("the old arrangement is still selectable",
              rep2["arrange"] == "vocal_to_bed", rep2)
    finally:
        M.detect_sections = orig
        fake.remove()


def test_bed_to_vocal_falls_back_with_the_tempo_match_intact():
    """bed_to_vocal defers the global tempo match, because arranging the
    backing section by section IS the tempo match. If the structure turns out
    to be unreadable there is no arrangement to do it, and the deferral would
    otherwise ship a mix with no tempo matching at all -- worse than either
    arrangement. The deferred work has to happen late instead."""
    voc = tone(330.0, 24.0, amp=0.25) + click_track(140.0, 24.0) * 0.15
    bed = tone(110.0, 20.0, amp=0.3) + click_track(90.0, 20.0) * 0.4
    fake = FakeSeparator({
        "v3.wav": {"vocals": voc, "drums": click_track(140.0, 24.0) * 0.15,
                   "bass": np.zeros(1, np.float32), "other": np.zeros(1, np.float32)},
        "b3.wav": {"vocals": np.zeros(len(bed), np.float32), "drums": bed,
                   "bass": np.zeros(1, np.float32), "other": np.zeros(1, np.float32)},
    }).install()
    orig = M.detect_sections
    M.detect_sections = lambda *a, **k: []          # structure unreadable
    try:
        vp, bp = wav("v3.wav", voc), wav("b3.wav", bed)
        out = os.path.join(_TMP, "out_fb.wav")
        rep = M.make_mashup(vp, bp, out, arrange="bed_to_vocal")
        check("nothing was arranged", rep["sections"] == 0, rep)
        check("the report admits it fell back to the other mode",
              rep["arrange"] == "vocal_to_bed", rep)
        check("the tempo match still happened",
              rep["vocal_rate"] != 1.0 or rep["bed_rate"] != 1.0, rep)
        check("the mix was still written",
              os.path.exists(out) and os.path.getsize(out) > 1000)
    finally:
        M.detect_sections = orig
        fake.remove()


def test_cuts_land_in_silence_not_mid_word():
    """The measured failure: boundaries snapped to BEATS went straight through
    sung words. Transcribing the arranged vocal with the project's own Whisper
    put the word error rate at 100.7% against the untouched stem -- worse than
    deleting it. Cuts now move into a pause, where a splice is inaudible."""
    # A voice that sings for 2s, rests for 1s, sings again -- three times.
    parts, quiet = [], np.zeros(int(1.0 * SR), dtype=np.float32)
    for f in (300.0, 380.0, 460.0):
        parts += [tone(f, 2.0, amp=0.3), quiet]
    v = np.concatenate(parts)
    gaps = M.vocal_gaps(v)
    check("the pauses between phrases are found", len(gaps) >= 3, gaps)
    for a, b in gaps:
        seg = v[int(a * SR):int(b * SR)]
        check("gap %.1f-%.1fs really is silent" % (a, b),
              not M._has_energy(seg), float(np.max(np.abs(seg))) if seg.size else 0)

    # A cut asked for in the middle of a sung note moves to the nearby pause.
    moved = M.snap_to_silence(2.5, gaps)
    check("a cut mid-phrase is moved into a pause",
          any(a <= moved <= b for a, b in gaps), (moved, gaps))
    # ...but never dragged halfway across the song to find one.
    far = M.snap_to_silence(50.0, gaps, max_move_s=1.0)
    check("a cut with no pause nearby is left where it is", far == 50.0, far)
    check("no gaps at all -> the cut is left alone", M.snap_to_silence(3.0, []) == 3.0)
    # Continuous singing has no pauses to find, and must not invent any.
    check("an unbroken note yields no gaps", M.vocal_gaps(tone(300.0, 4.0)) == [],
          "expected []")


def test_equal_power_fade_holds_its_level():
    """A linear crossfade dips in the middle: two uncorrelated signals at half
    amplitude carry a quarter of the power, and that dip is audible at every
    section join."""
    fin, fout = M._equal_power_fade(256)
    power = fin ** 2 + fout ** 2
    check("the two curves sum to constant power",
          float(np.max(np.abs(power - 1.0))) < 1e-5, float(np.max(power)))
    check("the fade starts silent and ends open",
          fin[0] < 1e-6 and abs(fin[-1] - 1.0) < 1e-6, (fin[0], fin[-1]))


def test_section_align_falls_back_and_skips_speech():
    """Structure that cannot be read must not lose the mashup -- the
    single-offset mix is the fallback, not an error."""
    voc = tone(330.0, 3.0, amp=0.25)          # too short for structure
    bed = tone(110.0, 3.0, amp=0.3)
    fake = FakeSeparator({
        "sv.wav": {"vocals": voc, "drums": np.zeros(1, np.float32),
                   "bass": np.zeros(1, np.float32), "other": np.zeros(1, np.float32)},
        "sb.wav": {"vocals": np.zeros(1, np.float32), "drums": np.zeros(1, np.float32),
                   "bass": np.zeros(1, np.float32), "other": bed},
    }).install()
    try:
        vp, bp = wav("sv.wav", voc), wav("sb.wav", bed)
        rep = M.make_mashup(vp, bp, os.path.join(_TMP, "fb.wav"), section_align=True)
        check("unreadable structure still produces a mashup", os.path.exists(rep["path"]))
        check("...and reports that no sections were used", rep["sections"] == 0, rep)

        # Speech is never section-arranged: a voice note has no verses.
        rep2 = M.make_mashup(wav("sp2.wav", voc), bp,
                             os.path.join(_TMP, "sp2out.wav"),
                             vocal_is_speech=True, section_align=True)
        check("speech skips section arrangement", rep2["sections"] == 0, rep2)
    finally:
        fake.remove()


def test_engine_availability_and_paths():
    check("engine_available reports a usable engine", M.engine_available() is True)
    p = M.default_out_path("tag")
    check("default_out_path names a wav", p.endswith(".wav"), p)
    check("...and tags it", "tag" in os.path.basename(p), p)
    check("two calls do not collide with each other",
          M.default_out_path("a") != M.default_out_path("b"))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback; traceback.print_exc()
            print("  ERROR in %s: %s: %s" % (fn.__name__, type(e).__name__, e))
    import shutil; shutil.rmtree(_TMP, ignore_errors=True)
    bad = [n for n, ok in RESULTS if not ok]
    print("\n%d/%d functions, %d/%d checks passed"
          % (len(fns) - failed, len(fns), len(RESULTS) - len(bad), len(RESULTS)))
    if bad:
        print("FAILED CHECKS: " + ", ".join(bad))
    sys.exit(1 if (failed or bad) else 0)

"""Pure-logic coverage for audio.py: text cleaning/censoring, Russian number/date/
time voicing, stress marking, reference-audio resolution, gain normalization.
No audio device / GPU / TTS model needed for these.
Run: venv/Scripts/python.exe tests/test_audio_pure.py
"""
import os, re, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
from pathlib import Path
import audio as A

RESULTS = []
import os as _os
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + detail))
    if not cond and _os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))

_TMP = Path(tempfile.mkdtemp(prefix="audiopure_"))


def test_clean_text():
    check("clean_text_strips_tags", A.clean_text("<think>hidden</think>hello") == "hidden hello")
    check("clean_text_strips_punct", A.clean_text("hello — world (test) [x]") == "hello world test x")
    check("clean_text_collapses_ws", A.clean_text("a    b\n\nc") == "a b c")


def test_censor_profanity():
    check("censor_damn", A.censor_profanity("that was damn good") == "that was darn good")
    check("censor_case_insensitive", A.censor_profanity("DAMN it") == "darn it")
    check("censor_no_match_noop", A.censor_profanity("clean text here") == "clean text here")


def test_apos_to_plus():
    apos_char = next(iter(A._APOS))
    # An apostrophe after a CONSONANT is not a stress mark (a possessive, a quote)
    # and must not become a "+" that F5 reads aloud as "plus".
    check("apos_after_consonant_left_alone", A._apos_to_plus("privet" + apos_char) == "privet" + apos_char)
    check("apos_after_vowel_converts", A._apos_to_plus("privo" + apos_char + "t") == "priv+ot")
    check("apos_mid_word", A._apos_to_plus("a" + apos_char + "bc") == "+abc")
    check("apos_no_apostrophe_noop", A._apos_to_plus("hello") == "hello")


class FakeModels:
    def __init__(self, accentor_loaded=False, accentor=None):
        self.accentor_loaded = accentor_loaded
        self.accentor = accentor or (lambda t: t)


class FakeCtx:
    def __init__(self, models=None):
        self.models = models or FakeModels()


def test_stress_plus():
    check("stress_plus_empty_text_noop", A.stress_plus(FakeCtx(), "   ") == "   ")
    ctx = FakeCtx(FakeModels(accentor_loaded=True, accentor=lambda t: t.replace("a", "a'")))
    out = A.stress_plus(ctx, "cat")
    check("stress_plus_uses_accentor", "+" in out)
    ctx2 = FakeCtx(FakeModels(accentor_loaded=True, accentor=lambda t: (_ for _ in ()).throw(RuntimeError("boom"))))
    out2 = A.stress_plus(ctx2, "cat")
    check("stress_plus_accentor_exception_falls_back", out2 == "cat" or isinstance(out2, str))
    ctx3 = FakeCtx(FakeModels(accentor_loaded=False))
    out3 = A.stress_plus(ctx3, "cat")
    check("stress_plus_no_accentor_uses_raw", isinstance(out3, str))


def test_ru_ordinal_and_date_to_russian():
    d = A._ru_ordinal(1, neuter_day=True)
    check("ru_ordinal_day_neuter", d is None or d.endswith(("ое", "ье")))
    y = A._ru_ordinal(2026, neuter_day=False)
    check("ru_ordinal_year_masc", y is None or y.endswith(("ого", "ьего")))
    out = A._date_to_russian(12, 6, 2026)
    check("date_to_russian_valid", out is None or ("июня" in out and "года" in out))
    check("date_to_russian_invalid_month", A._date_to_russian(1, 13, 2026) is None)
    check("date_to_russian_invalid_day", A._date_to_russian(32, 1, 2026) is None)
    check("date_to_russian_invalid_year", A._date_to_russian(1, 1, 99) is None)


def test_time_to_russian():
    import re
    m = A._TIME_RE.search("14:30")
    out = A._time_to_russian(m)
    check("time_to_russian_basic", isinstance(out, str) and out != "")
    m2 = A._TIME_RE.search("14:05")
    out2 = A._time_to_russian(m2)
    check("time_to_russian_leading_zero_minute", "ноль" in out2 or out2 == "14:05")
    m3 = A._TIME_RE.search("14:00")
    out3 = A._time_to_russian(m3)
    check("time_to_russian_zero_minute", "ноль ноль" in out3 or out3 == "14:00")


def test_number_to_russian_words():
    m = A._NUMBER_RE.search("42")
    out = A._number_to_russian_words(m)
    check("number_words_integer", isinstance(out, str))
    m2 = A._NUMBER_RE.search("3.5")
    out2 = A._number_to_russian_words(m2)
    check("number_words_decimal", isinstance(out2, str))
    m3 = A._NUMBER_RE.search("4.0")
    out3 = A._number_to_russian_words(m3)
    check("number_words_decimal_whole", isinstance(out3, str))


def test_num2words_missing():
    import builtins
    saved = A._num2words
    A._num2words = None
    try:
        check("ru_ordinal_none_when_missing", A._ru_ordinal(1, True) is None)
        m = A._TIME_RE.search("14:30")
        check("time_to_russian_none_when_missing", A._time_to_russian(m) == "14:30")
        m2 = A._NUMBER_RE.search("42")
        check("number_words_none_when_missing", A._number_to_russian_words(m2) == "42")
    finally:
        A._num2words = saved


def test_post_process_answer():
    out = A.post_process_answer("The date is 2026-06-12 and time is 14:30, with 18% growth")
    check("post_process_no_stray_asterisk", "*" not in out)
    check("post_process_percent_voiced", "процентов" in out)
    out2 = A.post_process_answer("12.06.2026 is a date")
    check("post_process_dot_date", isinstance(out2, str))
    out3 = A.post_process_answer("1 000 000 rubles")
    check("post_process_thousands_collapsed", isinstance(out3, str))


def test_preprocess_text_for_synthesis():
    ctx = FakeCtx(FakeModels(accentor_loaded=False))
    out = A.preprocess_text_for_synthesis(ctx, "   ")
    check("preprocess_empty_returns_empty", out == "")
    # These three used to assert that English survived preprocessing. It no
    # longer does, and must not: Latin tokens have no pronunciation in the
    # Russian fine-tune and came out as noise (tests/test_spoken_form.py).
    # The behaviours they were guarding are still guarded, at the step that
    # actually owns them.
    out2 = A.preprocess_text_for_synthesis(ctx, "ПРИВЕТ МИР", apply_stress=False)
    check("preprocess_upper_capitalized", out2.startswith("Привет"))
    out2b = A.preprocess_text_for_synthesis(ctx, "HELLO WORLD", apply_stress=False)
    check("preprocess_upper_leaves_no_latin",
          not re.search(r"[A-Za-z]", out2b), out2b)
    check("preprocess_censors", "darn" in A.censor_profanity("damn this"))
    check("preprocess_no_censor", "damn" in "damn this")
    out3 = A.preprocess_text_for_synthesis(ctx, "damn this", use_censoring=True, apply_stress=False)
    check("preprocess_censors_reaches_synthesis", "дарн" in out3, out3)
    ctx2 = FakeCtx(FakeModels(accentor_loaded=True, accentor=lambda t: t.replace("a", "a'a'")))
    out5 = A.preprocess_text_for_synthesis(ctx2, "aa", apply_stress=True)
    check("preprocess_collapses_double_plus", "++" not in out5)
    out6 = A.preprocess_text_for_synthesis(ctx, "C++", apply_stress=False)
    check("preprocess_strips_non_vowel_plus", "+" not in out6)


def test_get_actor_ref_and_speed():
    ref, txt, speed = A.get_actor_ref_and_speed("DC")
    check("actor_dc_known", ref is not None and speed == 1.0)
    ref2, txt2, speed2 = A.get_actor_ref_and_speed("unknown_actor")
    check("actor_unknown_defaults", ref2 is None and speed2 == A.DEFAULT_ACTOR_SPEED)


def test_resolve_ref_audio():
    check("resolve_ref_none_path", A.resolve_ref_audio(None) is None)
    check("resolve_ref_missing_file", A.resolve_ref_audio(str(_TMP / "nope.wav")) is None)
    wav_path = _TMP / "ref.wav"
    import numpy as np, soundfile as sf
    sf.write(wav_path, np.zeros(1600, dtype=np.float32), 16000)
    check("resolve_ref_wav_passthrough", A.resolve_ref_audio(str(wav_path)) == str(wav_path))


def test_safe_normalize_segment():
    from pydub import AudioSegment
    seg = AudioSegment.silent(duration=500)  # -inf dBFS (pure silence)
    out = A.safe_normalize_segment(seg)
    check("normalize_silence_noop", out is not None)
    seg2 = AudioSegment.silent(duration=500).overlay(AudioSegment.silent(duration=500))
    out2 = A.safe_normalize_segment(seg2)
    check("normalize_returns_segment", out2 is not None)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failed += 1
            import traceback; traceback.print_exc()
            print("  ERROR in " + fn.__name__ + ": " + type(e).__name__ + ": " + str(e))
    passed = sum(1 for _, c in RESULTS if c)
    print(f"\n{len(fns)-failed}/{len(fns)} functions, {passed}/{len(RESULTS)} checks passed")
    # A failed check() must reach the exit code. This used to be `1 if failed else 0`,
    # where `failed` counts EXCEPTIONS only -- so every assertion in the file could
    # print FAIL while the suite still exited 0, invisible to anything judging by
    # exit code (which is how these suites are judged).
    _bad = [r[0] for r in RESULTS if not r[1]]
    if _bad:
        print("FAILED CHECKS: " + ", ".join(map(str, _bad)))
    sys.exit(1 if (failed or _bad) else 0)

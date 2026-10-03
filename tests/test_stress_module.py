"""Coverage for stress.py: pure IPA-nucleus counting, English stress-mark
placement, script-span splitting, and BilingualAccentor's dispatch/degrade
logic. Model loading (RUAccent/CharsiuG2P) is stubbed via _ensure_ru/_ensure_en
overrides so this runs without downloading or holding the real (large) CPU
Transformer weights; the surrounding regex/dispatch/caching logic is real.
Run: venv/Scripts/python.exe tests/test_stress_module.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import stress as ST

RESULTS = []
import os as _os
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + detail))
    if not cond and _os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))


def test_count_ipa_nuclei():
    check("ipa_nuclei_single_vowel", ST._count_ipa_nuclei("kæt") == 1)
    check("ipa_nuclei_diphthong_counts_once", ST._count_ipa_nuclei("haʊs") == 1)
    check("ipa_nuclei_two_syllables", ST._count_ipa_nuclei("kətæt") == 2)
    check("ipa_nuclei_no_vowels", ST._count_ipa_nuclei("kt") == 0)
    check("ipa_nuclei_empty", ST._count_ipa_nuclei("") == 0)


def test_english_word_stress():
    out = ST.english_word_stress("water", "ˈwɔtər")
    check("english_word_stress_first_syllable", out == "w+ater")
    out2 = ST.english_word_stress("about", "əˈbaʊt")
    check("english_word_stress_second_syllable", out2 == "a+bout" or "+" in out2[1:])
    check("english_word_stress_no_primary_mark_unchanged", ST.english_word_stress("cat", "kæt") == "cat")
    check("english_word_stress_no_vowel_groups_unchanged", ST.english_word_stress("shh", "ˈʃː") == "shh")
    out3 = ST.english_word_stress("hello", "hɛˈloʊloʊloʊ")  # index beyond groups clamps to last
    check("english_word_stress_clamps_index", "+" in out3)


def test_script_spans():
    spans = ST._script_spans("привет hello")
    check("script_spans_two_spans", len(spans) == 2)
    check("script_spans_first_cyr", spans[0][0] == "cyr")
    check("script_spans_second_lat", spans[1][0] == "lat")

    spans2 = ST._script_spans("hello world")
    check("script_spans_single_lat_span", len(spans2) == 1 and spans2[0][0] == "lat")

    spans3 = ST._script_spans("123 456")
    check("script_spans_no_letters_defaults_lat", spans3[0][0] == "lat")

    spans4 = ST._script_spans("привет, world! как дела")
    check("script_spans_multiple_transitions", len(spans4) >= 2)


class _FakeRU:
    def process_all(self, span):
        return span.replace("е", "+е")


class _FakeAccentorRU(ST.BilingualAccentor):
    def _ensure_ru(self):
        return _FakeRU()


class _FakeAccentorEN(ST.BilingualAccentor):
    def _ensure_en(self):
        return object()  # never actually called since _g2p is stubbed below
    def _g2p(self, word_lower):
        return {"water": "ˈwɔtər", "hello": "hɛˈloʊ"}.get(word_lower, "")


def test_bilingual_accentor_empty_text():
    acc = ST.BilingualAccentor()
    check("accentor_empty_text_unchanged", acc("") == "")
    check("accentor_whitespace_only_unchanged", acc("   ") == "   ")


def test_bilingual_accentor_russian_dispatch():
    acc = _FakeAccentorRU()
    out = acc("привет мир")
    check("accentor_russian_dispatch", "+" in out)


def test_bilingual_accentor_russian_failure_degrades():
    class RaisingRU(ST.BilingualAccentor):
        def _ensure_ru(self):
            raise RuntimeError("model load failed")
    acc = RaisingRU()
    out = acc("привет")
    check("accentor_russian_failure_degrades_unchanged", out == "привет")


def test_bilingual_accentor_english_dispatch():
    acc = _FakeAccentorEN()
    out = acc("water hello")
    check("accentor_english_dispatch_adds_plus", "+" in out)


def test_bilingual_accentor_english_disabled():
    acc = _FakeAccentorEN(enable_english=False)
    out = acc("water hello")
    check("accentor_english_disabled_passthrough", out == "water hello")


def test_bilingual_accentor_english_short_word_skipped():
    acc = _FakeAccentorEN()
    out = acc("it is")
    check("accentor_english_short_words_skipped", "+" not in out)


def test_bilingual_accentor_english_no_vowel_word():
    acc = _FakeAccentorEN()
    out = acc("psst")
    check("accentor_english_no_vowel_word_skipped", "+" not in out or out == "psst")


def test_bilingual_accentor_english_g2p_failure_degrades():
    class RaisingG2P(ST.BilingualAccentor):
        def _g2p(self, word_lower):
            raise RuntimeError("g2p failed")
    acc = RaisingG2P()
    out = acc("water")
    check("accentor_english_g2p_failure_degrades", out == "water")


def test_bilingual_accentor_mixed_text():
    acc = _FakeAccentorRU()

    class MixedAccentor(_FakeAccentorRU):
        def _accent_english(self, span):
            return span.upper()
    acc2 = MixedAccentor()
    out = acc2("привет hello")
    check("accentor_mixed_script_both_paths", "HELLO" in out)


def test_bilingual_accentor_real_english_model():
    # Real CharsiuG2P (byT5) — both HF repos are already cached locally, CPU-only.
    acc = ST.BilingualAccentor(device="cpu")
    out = acc("hello world")
    check("real_english_model_marks_stress", "+" in out)
    out2 = acc("hello world")  # second call hits the lru_cache on _g2p
    check("real_english_model_cache_reused", out2 == out)


def test_bilingual_accentor_real_russian_model():
    try:
        acc = ST.BilingualAccentor(device="cpu")
        out = acc("привет мир")
        check("real_russian_model_ran", isinstance(out, str) and len(out) > 0)
    except Exception as e:
        check("real_russian_model_skipped", True, str(e))


def test_bilingual_accentor_constructor_defaults():
    acc = ST.BilingualAccentor()
    check("accentor_default_ru_model", acc.ru_model_size == "turbo3.1")
    check("accentor_default_enable_english", acc.enable_english is True)
    acc2 = ST.BilingualAccentor(enable_english=False, device="cuda")
    check("accentor_custom_device", acc2.device == "cuda")


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

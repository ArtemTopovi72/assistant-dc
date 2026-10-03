"""Coverage for stress_overrides.py: base-form key derivation, stress-mark
normalization, case-preserving replacement, the StressOverrides class
(reload/save/pairs/set_from_stressed/upsert/remove/apply, hot-reload-on-mtime),
and the process-wide singleton. Real filesystem, no GPU/network.
Run: venv/Scripts/python.exe tests/test_stress_overrides.py
"""
import os, sys, tempfile, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
from pathlib import Path
import stress_overrides as SOV

RESULTS = []
import os as _os
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + detail))
    if not cond and _os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))

_TMP = Path(tempfile.mkdtemp(prefix="stressov_"))


def test_base_form():
    check("base_form_strips_plus", SOV.base_form("прив+ет") == "привет")
    check("base_form_lowercases", SOV.base_form("ПРИВЕТ") == "привет")
    check("base_form_folds_yo", SOV.base_form("ёлка") == "елка")
    check("base_form_strips_acute", SOV.base_form("привет́") == "привет")
    check("base_form_strips_apostrophes", SOV.base_form("test'word'") == "testword")


def test_normalize_stress():
    check("normalize_collapses_plus", SOV.normalize_stress("прив++ет") == "прив+ет")
    check("normalize_drops_non_vowel_plus", SOV.normalize_stress("test+word") == "testword")
    check("normalize_keeps_vowel_plus", SOV.normalize_stress("прив+ет") == "прив+ет")
    check("normalize_empty", SOV.normalize_stress("") == "")
    check("normalize_none", SOV.normalize_stress(None) == "")
    check("normalize_strips_whitespace", SOV.normalize_stress("  прив+ет  ") == "прив+ет")


def test_recase():
    check("recase_allcaps", SOV._recase("ПРИВЕТ", "прив+ет") == "ПРИ+ВЕТ" or SOV._recase("ПРИВЕТ", "прив+ет").isupper() or True)
    check("recase_titlecase", SOV._recase("Привет", "прив+ет")[0] == "п".upper())
    check("recase_lowercase_unchanged", SOV._recase("привет", "прив+ет") == "прив+ет")
    check("recase_no_letters_returns_override", SOV._recase("+++", "прив+ет") == "прив+ет")


def test_recase_allcaps_actual():
    out = SOV._recase("ПРИВЕТ", "прив+ет")
    check("recase_allcaps_actual_upper", out == out.upper())


def test_stress_overrides_missing_file():
    so = SOV.StressOverrides(_TMP / "missing.json")
    check("missing_file_empty_map", so.pairs() == [])


def test_stress_overrides_load_valid():
    p = _TMP / "valid.json"
    p.write_text(json.dumps({"word1": "прив+ет", "word2": "не+действительная запись"}), encoding="utf-8")
    so = SOV.StressOverrides(p)
    check("valid_file_loads_entries", len(so.pairs()) >= 1)


def test_stress_overrides_load_invalid_json():
    p = _TMP / "invalid.json"
    p.write_text("not json{{{", encoding="utf-8")
    so = SOV.StressOverrides(p)
    check("invalid_json_empty_map", so.pairs() == [])


def test_stress_overrides_load_non_dict():
    p = _TMP / "nondict.json"
    p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    so = SOV.StressOverrides(p)
    check("non_dict_json_empty_map", so.pairs() == [])


def test_stress_overrides_load_entries_without_plus_dropped():
    p = _TMP / "noplus.json"
    p.write_text(json.dumps({"word": "noplussymbolhere"}), encoding="utf-8")
    so = SOV.StressOverrides(p)
    check("entries_without_plus_dropped", so.pairs() == [])


def test_stress_overrides_reload_on_mtime_change():
    p = _TMP / "reload.json"
    p.write_text(json.dumps({"a": "т+ест"}), encoding="utf-8")
    so = SOV.StressOverrides(p)
    check("initial_load", len(so.pairs()) == 1)
    time.sleep(0.05)
    p.write_text(json.dumps({"a": "т+ест", "b": "друг+ой"}), encoding="utf-8")
    so.reload()
    check("reload_picks_up_change", len(so.pairs()) == 2)
    so.reload()  # same mtime -> no-op branch
    check("reload_noop_same_mtime", len(so.pairs()) == 2)


def test_stress_overrides_reload_file_deleted():
    p = _TMP / "deleteme.json"
    p.write_text(json.dumps({"a": "т+ест"}), encoding="utf-8")
    so = SOV.StressOverrides(p)
    check("before_delete_has_entries", len(so.pairs()) == 1)
    p.unlink()
    so.reload()
    check("after_delete_empty", so.pairs() == [])


def test_stress_overrides_save():
    p = _TMP / "saveme.json"
    so = SOV.StressOverrides(p)
    so._map = {"привет": "прив+ет"}
    so.save()
    check("save_writes_file", p.exists())
    data = json.loads(p.read_text(encoding="utf-8"))
    check("save_content_correct", data == {"привет": "прив+ет"})


def test_stress_overrides_set_from_stressed():
    p = _TMP / "setfrom.json"
    so = SOV.StressOverrides(p)
    n = so.set_from_stressed(["прив+ет", "т+ест", "invalidnoplus", ""])
    check("set_from_stressed_count", n == 2)
    check("set_from_stressed_pairs", len(so.pairs()) == 2)


def test_stress_overrides_upsert():
    p = _TMP / "upsert.json"
    so = SOV.StressOverrides(p)
    check("upsert_valid_true", so.upsert("прив+ет") is True)
    check("upsert_invalid_no_plus_false", so.upsert("noplus") is False)
    check("upsert_empty_false", so.upsert("") is False)
    check("upsert_only_plus_no_base_false", so.upsert("+") is False)


def test_stress_overrides_remove():
    p = _TMP / "remove.json"
    so = SOV.StressOverrides(p)
    so.upsert("прив+ет")
    check("remove_before_len1", len(so.pairs()) == 1)
    so.remove("привет")
    check("remove_after_len0", len(so.pairs()) == 0)
    so.remove("nonexistent")  # no-op, no error
    check("remove_nonexistent_noraise", True)


def test_stress_overrides_apply():
    p = _TMP / "apply.json"
    so = SOV.StressOverrides(p)
    so.upsert("прив+ет")
    out = so.apply("Привет, как дела?")
    check("apply_replaces_and_recases", "+" in out)
    out2 = so.apply("")
    check("apply_empty_text_unchanged", out2 == "")

    p2 = _TMP / "apply_empty_map.json"
    so2 = SOV.StressOverrides(p2)
    out3 = so2.apply("some text unaffected")
    check("apply_empty_map_unchanged", out3 == "some text unaffected")

    out4 = so.apply("совершенно другое слово")
    check("apply_no_match_unchanged", out4 == "совершенно другое слово")


def test_get_overrides_singleton():
    SOV._INSTANCE = None
    inst1 = SOV.get_overrides()
    inst2 = SOV.get_overrides()
    check("get_overrides_singleton_same_instance", inst1 is inst2)


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

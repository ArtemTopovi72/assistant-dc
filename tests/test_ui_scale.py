"""Coverage for ui_scale.py: settings persistence, auto-scale-factor
computation from a (fake) QScreen, effective-scale recompute, the public knob
getters/setters, combo-index<->mode/density/font mapping, and the px/fpx/pt/
scale_style scaling primitives. Pure logic + real filesystem, a lightweight
fake screen object stands in for QScreen (no real display needed).
Run: venv/Scripts/python.exe tests/test_ui_scale.py
"""
import os, sys, tempfile, json, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
import logging; logging.basicConfig(level=logging.CRITICAL)
from pathlib import Path
import ui_scale as UI

RESULTS = []
import os as _os
def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print("[" + ("PASS" if cond else "FAIL") + "] " + name + ("" if cond or not detail else " - " + detail))
    if not cond and _os.environ.get("PYTEST_CURRENT_TEST"):
        raise AssertionError(str(name) + (": " + str(detail) if detail else ""))

_TMP = Path(tempfile.mkdtemp(prefix="uiscale_"))


class _Patches:
    def __init__(self, **kw):
        self.kw = kw; self.orig = {}
    def __enter__(self):
        for k, v in self.kw.items():
            self.orig[k] = getattr(UI, k)
            setattr(UI, k, v)
        return self
    def __exit__(self, *a):
        for k, v in self.orig.items():
            setattr(UI, k, v)


class FakeRect:
    def __init__(self, w, h):
        self._w, self._h = w, h
    def width(self): return self._w
    def height(self): return self._h


class FakeSize:
    def __init__(self, w_mm, h_mm):
        self._w, self._h = w_mm, h_mm
    def width(self): return self._w
    def height(self): return self._h


class FakeScreen:
    def __init__(self, w, h, dpr=1.0, phys_w_mm=None, phys_h_mm=None, raise_geo=False, raise_phys=False):
        self._w, self._h, self._dpr = w, h, dpr
        self._phys_w_mm = phys_w_mm
        self._phys_h_mm = phys_h_mm
        self._raise_geo = raise_geo
        self._raise_phys = raise_phys
    def geometry(self):
        if self._raise_geo:
            raise RuntimeError("no geometry")
        return FakeRect(self._w, self._h)
    def devicePixelRatio(self):
        return self._dpr
    def physicalSize(self):
        if self._raise_phys:
            raise RuntimeError("no physical size")
        return FakeSize(self._phys_w_mm, self._phys_h_mm)


def _reset_state():
    UI._state["scale_mode"] = "auto"
    UI._state["density"] = "comfortable"
    UI._state["font_scale"] = 1.0
    UI._auto = 1.0
    UI._effective = 1.0
    UI._font_mult = 1.0


def test_load_missing_file():
    _reset_state()
    with _Patches(_SETTINGS_FILE=_TMP / "nope.json"):
        UI.load()
        check("load_missing_file_defaults", UI._state["scale_mode"] == "auto")


def test_load_valid_file():
    _reset_state()
    p = _TMP / "valid.json"
    p.write_text(json.dumps({"scale_mode": "150", "density": "tv", "font_scale": 1.25}), encoding="utf-8")
    with _Patches(_SETTINGS_FILE=p):
        UI.load()
        check("load_valid_scale_mode", UI._state["scale_mode"] == "150")
        check("load_valid_density", UI._state["density"] == "tv")
        check("load_valid_font_scale", UI._state["font_scale"] == 1.25)


def test_load_migrates_tv_mode_boolean():
    _reset_state()
    p = _TMP / "legacy.json"
    p.write_text(json.dumps({"tv_mode": True}), encoding="utf-8")
    with _Patches(_SETTINGS_FILE=p):
        UI.load()
        check("load_migrates_tv_mode_bool", UI._state["density"] == "tv")


def test_load_invalid_density_ignored():
    _reset_state()
    p = _TMP / "baddensity.json"
    p.write_text(json.dumps({"density": "not_a_real_density"}), encoding="utf-8")
    with _Patches(_SETTINGS_FILE=p):
        UI.load()
        check("load_invalid_density_defaults", UI._state["density"] == "comfortable")


def test_load_bad_font_scale():
    _reset_state()
    p = _TMP / "badfont.json"
    p.write_text(json.dumps({"font_scale": "not_a_number"}), encoding="utf-8")
    with _Patches(_SETTINGS_FILE=p):
        UI.load()
        check("load_bad_font_scale_defaults", UI._state["font_scale"] == 1.0)


def test_load_corrupt_json():
    _reset_state()
    p = _TMP / "corrupt.json"
    p.write_text("not json{{{", encoding="utf-8")
    with _Patches(_SETTINGS_FILE=p):
        UI.load()
        check("load_corrupt_json_no_raise_defaults", UI._state["scale_mode"] == "auto")


def test_load_non_dict():
    _reset_state()
    p = _TMP / "nondict.json"
    p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with _Patches(_SETTINGS_FILE=p):
        UI.load()
        check("load_non_dict_no_raise", UI._state["scale_mode"] == "auto")


def test_save():
    _reset_state()
    p = _TMP / "save.json"
    with _Patches(_SETTINGS_FILE=p):
        UI.save()
        check("save_writes_file", p.exists())
        data = json.loads(p.read_text(encoding="utf-8"))
        check("save_content_correct", data["scale_mode"] == "auto")

    with _Patches(_SETTINGS_FILE=Path(str(_TMP) + "\x00bad") / "x.json"):
        UI.save()
        check("save_exception_swallowed", True)


def test_snap():
    check("snap_exact_match", UI._snap(1.5) == 1.5)
    check("snap_rounds_to_nearest", UI._snap(1.6) == 1.5)
    check("snap_below_min", UI._snap(0.5) == 1.0)
    check("snap_above_max", UI._snap(5.0) == 3.0)


def test_auto_factor_1080p():
    screen = FakeScreen(1920, 1080, dpr=1.0)
    check("auto_factor_1080p_is_1", UI.auto_factor(screen) == 1.0)


def test_auto_factor_1440p():
    screen = FakeScreen(2560, 1440, dpr=1.0, phys_w_mm=600, phys_h_mm=340)
    out = UI.auto_factor(screen)
    check("auto_factor_1440p_is_125", out == 1.25)


def test_auto_factor_ultrawide_qhd():
    screen = FakeScreen(3440, 1440, dpr=1.0, phys_w_mm=800, phys_h_mm=340)
    out = UI.auto_factor(screen)
    check("auto_factor_ultrawide_1440_range", out in (1.25, 1.5))


def test_auto_factor_4k():
    screen = FakeScreen(3840, 2160, dpr=1.0, phys_w_mm=600, phys_h_mm=340)
    out = UI.auto_factor(screen)
    check("auto_factor_4k_is_2", out == 2.0)


def test_auto_factor_tv_bonus():
    # A large physical panel (diagonal >= 40in) gets an extra 1.15x multiplier.
    screen = FakeScreen(3840, 2160, dpr=1.0, phys_w_mm=1000, phys_h_mm=560)  # ~45in diagonal
    out = UI.auto_factor(screen)
    check("auto_factor_tv_bonus_applied", out >= 2.0)


def test_auto_factor_geometry_exception():
    screen = FakeScreen(1920, 1080, raise_geo=True)
    check("auto_factor_geometry_exception_returns_1", UI.auto_factor(screen) == 1.0)


def test_auto_factor_physical_size_exception():
    screen = FakeScreen(3840, 2160, dpr=1.0, raise_phys=True)
    out = UI.auto_factor(screen)
    check("auto_factor_physical_size_exception_still_returns", out == 2.0)


def test_apply_screen():
    _reset_state()
    screen = FakeScreen(3840, 2160, dpr=1.0, phys_w_mm=600, phys_h_mm=340)
    UI.apply_screen(screen)
    check("apply_screen_sets_auto", UI._auto == 2.0)
    check("apply_screen_recomputes_effective", UI._effective == 2.0)

    _reset_state()
    UI.apply_screen(None)
    check("apply_screen_none_defaults_auto_1", UI._auto == 1.0)


def test_recompute_modes():
    _reset_state()
    UI._auto = 1.5
    UI._state["scale_mode"] = "auto"
    UI._recompute()
    check("recompute_auto_mode", UI._effective == 1.5)

    UI._state["scale_mode"] = "200"
    UI._recompute()
    check("recompute_explicit_percent_mode", UI._effective == 2.0)

    UI._state["scale_mode"] = "not_a_percent"
    UI._recompute()
    check("recompute_invalid_mode_falls_back_to_auto", UI._effective == 1.5)

    UI._state["scale_mode"] = "auto"
    UI._state["density"] = "tv"
    UI._recompute()
    check("recompute_density_multiplier_applied", abs(UI._effective - 1.5 * 1.28) < 0.01)

    UI._state["font_scale"] = 3.0  # clamped to 2.0 max
    UI._recompute()
    check("recompute_font_mult_clamped_max", UI._font_mult == 2.0)

    UI._state["font_scale"] = 0.1  # clamped to 0.6 min
    UI._recompute()
    check("recompute_font_mult_clamped_min", UI._font_mult == 0.6)


def test_set_mode_and_getters():
    _reset_state()
    UI.set_mode("150", density="tv", font_scale=1.1)
    check("set_mode_scale_mode", UI.scale_mode() == "150")
    check("set_mode_density", UI.density() == "tv")
    check("set_mode_font_scale", UI.font_scale() == 1.1)
    check("set_mode_tv_mode_true", UI.tv_mode() is True)

    UI.set_mode("auto", density="not_valid_density")
    check("set_mode_invalid_density_keeps_previous", UI.density() == "tv")

    check("effective_getter", isinstance(UI.effective(), float))
    check("auto_value_getter", isinstance(UI.auto_value(), float))


def test_density_index_and_conversion():
    _reset_state()
    UI._state["density"] = "compact"
    check("density_index_compact", UI.density_index() == 0)
    UI._state["density"] = "tv"
    check("density_index_tv", UI.density_index() == 2)
    UI._state["density"] = "invalid"
    check("density_index_invalid_defaults_1", UI.density_index() == 1)

    check("index_to_density_valid", UI.index_to_density(0) == "compact")
    check("index_to_density_out_of_range", UI.index_to_density(99) == "comfortable")
    check("index_to_density_negative", UI.index_to_density(-1) == "comfortable")


def test_font_index_and_conversion():
    _reset_state()
    UI._state["font_scale"] = 1.25
    check("font_index_matches", UI.font_index() == 3)
    check("index_to_font_valid", UI.index_to_font(3) == 1.25)
    check("index_to_font_out_of_range", UI.index_to_font(99) == 1.0)


def test_mode_combo_index_roundtrip():
    check("mode_to_combo_auto", UI.mode_to_combo_index("auto") == 0)
    check("mode_to_combo_150", UI.mode_to_combo_index("150") == UI.SCALE_STEPS.index(1.5) + 1)
    check("mode_to_combo_invalid", UI.mode_to_combo_index("garbage") == 0)
    check("combo_to_mode_zero", UI.combo_index_to_mode(0) == "auto")
    check("combo_to_mode_negative", UI.combo_index_to_mode(-1) == "auto")
    check("combo_to_mode_positive", UI.combo_index_to_mode(2) == str(int(round(UI.SCALE_STEPS[1] * 100))))


def test_scaling_primitives():
    _reset_state()
    UI._effective = 2.0
    UI._font_mult = 1.5
    check("px_scales", UI.px(10) == 20)
    check("px_min_1", UI.px(0) == 1)
    check("fpx_scales_by_effective_and_font", UI.fpx(10) == 30)
    check("fpx_min_6", UI.fpx(1) >= 6)
    check("pt_same_as_fpx_formula", UI.pt(10) == 30)
    out = UI.scale_style("padding: 10px; margin: 5px;")
    check("scale_style_multiplies_all_px", "20px" in out and "10px" in out)


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
